// ============================================================================
// assembly/ns_assembler_scalar.cc - Navier-Stokes assembly for scalar DoFHandlers
//
// REFACTORED VERSION: Works with separate DoFHandlers instead of FESystem
//
// Equation (Nochetto 14e):
//   u_t + (u·∇)u - div(ν_θ T(u)) + ∇p = μ₀(m·∇)h + (λ/ε)θ∇ψ + f_g
//
// Uses semi-implicit time stepping:
//   - Viscosity: θ-method (θ implicit, 1-θ explicit)
//   - Convection: explicit (from previous iteration)
//   - Pressure: implicit (saddle point)
//   - Forces: explicit (from current c, μ, φ)
// ============================================================================
#include "assembly/ns_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include "utilities/nsch_mms.h"

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Sigmoid function H(x) = 1/(1 + e^(-x))  [Eq 18]
// ============================================================================
namespace
{
    inline double sigmoid(double x)
    {
        if (x > 20.0) return 1.0;
        if (x < -20.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Phase-dependent viscosity [Eq 17]
    //   ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
    inline double compute_viscosity_nochetto(double theta, double epsilon,
                                             double nu_water, double nu_ferro)
    {
        return nu_water + (nu_ferro - nu_water) * sigmoid(theta / epsilon);
    }

    // Phase-dependent susceptibility [Eq 17]
    //   κ_θ = κ₀ H(θ/ε)
    inline double compute_susceptibility_nochetto(double theta, double epsilon, double kappa_0)
    {
        return kappa_0 * sigmoid(theta / epsilon);
    }
}

// ============================================================================
// Main assembly function
// ============================================================================
template <int dim>
void assemble_ns_system_scalar(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::DoFHandler<dim>& mu_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>&  ux_old_solution,
    const dealii::Vector<double>&  uy_old_solution,
    const dealii::Vector<double>&  c_solution,
    const dealii::Vector<double>&  mu_solution,
    const dealii::Vector<double>*  phi_solution,
    const NSCHParameters&          params,
    double                         dt,
    double                         current_time,
    dealii::SparseMatrix<double>&  ns_matrix,
    dealii::Vector<double>&        ns_rhs,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map)
{
    ns_matrix = 0;
    ns_rhs = 0;

    const auto& fe_Q2 = ux_dof_handler.get_fe();
    const auto& fe_Q1 = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell_Q2 = fe_Q2.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = fe_Q1.n_dofs_per_cell();

    // Quadrature
    dealii::QGauss<dim> quadrature(params.fe_degree_velocity + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for each scalar field
    dealii::FEValues<dim> ux_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> p_fe_values(fe_Q1, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> c_fe_values(fe_Q2, quadrature,
        dealii::update_values);

    dealii::FEValues<dim> mu_fe_values(fe_Q2, quadrature,
        dealii::update_gradients);

    // FEValues for magnetic potential (conditional)
    std::unique_ptr<dealii::FEValues<dim>> phi_fe_values;
    if (params.enable_magnetic && phi_dof_handler != nullptr)
    {
        phi_fe_values = std::make_unique<dealii::FEValues<dim>>(
            fe_Q2, quadrature,
            dealii::update_gradients | dealii::update_hessians);
    }

    // Local matrices for the 3x3 block structure (ux, uy, p)
    // We'll assemble contributions from each combination
    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_uy_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_p_ux(dofs_per_cell_Q1, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_p_uy(dofs_per_cell_Q1, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_p_p(dofs_per_cell_Q1, dofs_per_cell_Q1);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_p(dofs_per_cell_Q1);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_local_dofs(dofs_per_cell_Q1);

    // Storage for solution values at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> c_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> mu_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);

    // Physical parameters
    const double theta_time = params.theta;  // Time stepping parameter
    const double lambda = params.lambda;
    const double epsilon = params.epsilon;
    const double nu_water = params.nu_water;
    const double nu_ferro = params.nu_ferro;
    const double mu_0 = params.mu_0;
    const double kappa_0 = params.chi_m;

    // Gravity parameters
    const bool use_gravity = params.enable_gravity && !params.mms_mode;
    const double g_mag = params.gravity;
    const double r_density = params.density_ratio;
    const double g_angle_rad = params.gravity_angle * M_PI / 180.0;
    dealii::Tensor<1, dim> g_direction;
    g_direction[0] = std::cos(g_angle_rad);
    g_direction[1] = std::sin(g_angle_rad);

    // MMS source terms (if enabled)
    MMSSourceU<dim> mms_source_u;
    if (params.mms_mode)
        mms_source_u.set_time(current_time);

    // Diagnostic tracking
    double max_F_cap = 0.0, max_F_mag = 0.0, max_F_grav = 0.0, max_h_norm = 0.0;

    // Iterate over cells - all DoFHandlers share the same mesh
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();
    auto c_cell = c_dof_handler.begin_active();
    auto mu_cell = mu_dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator phi_cell;
    if (params.enable_magnetic && phi_dof_handler != nullptr)
        phi_cell = phi_dof_handler->begin_active();

    for (; ux_cell != ux_dof_handler.end();
         ++ux_cell, ++uy_cell, ++p_cell, ++c_cell, ++mu_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        c_fe_values.reinit(c_cell);
        mu_fe_values.reinit(mu_cell);

        if (params.enable_magnetic && phi_dof_handler != nullptr)
        {
            phi_fe_values->reinit(phi_cell);
            ++phi_cell;
        }

        // Reset local matrices
        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux = 0;  local_p_uy = 0;  local_p_p = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        // Get DoF indices
        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get solution values at quadrature points
        ux_fe_values.get_function_values(ux_old_solution, ux_old_values);
        uy_fe_values.get_function_values(uy_old_solution, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old_solution, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old_solution, uy_old_gradients);
        c_fe_values.get_function_values(c_solution, c_values);
        mu_fe_values.get_function_gradients(mu_solution, mu_gradients);

        if (params.enable_magnetic && phi_dof_handler != nullptr && phi_solution != nullptr)
        {
            phi_fe_values->get_function_gradients(*phi_solution, phi_gradients);
            phi_fe_values->get_function_hessians(*phi_solution, phi_hessians);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            const double ux_old = ux_old_values[q];
            const double uy_old = uy_old_values[q];
            const dealii::Tensor<1, dim>& grad_ux_old = ux_old_gradients[q];
            const dealii::Tensor<1, dim>& grad_uy_old = uy_old_gradients[q];

            const double c_val = c_values[q];
            const dealii::Tensor<1, dim>& grad_mu = mu_gradients[q];

            // Phase-dependent viscosity
            const double nu = compute_viscosity_nochetto(c_val, epsilon, nu_water, nu_ferro);

            // ================================================================
            // Capillary force [Eq 10]: F_cap = (λ/ε)θ∇ψ
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            if (!params.mms_mode)
            {
                const double coeff = lambda / epsilon;
                F_cap[0] = coeff * c_val * grad_mu[0];
                F_cap[1] = coeff * c_val * grad_mu[1];
            }

            // ================================================================
            // Magnetic force: F_mag = μ₀ κ_θ (h·∇)h
            // where h = -∇φ and ∇h = -Hess(φ)
            // ================================================================
            dealii::Tensor<1, dim> F_mag;
            if (params.enable_magnetic && !params.mms_mode &&
                phi_dof_handler != nullptr && phi_solution != nullptr)
            {
                const dealii::Tensor<1, dim>& grad_phi = phi_gradients[q];
                const dealii::Tensor<2, dim>& hess_phi = phi_hessians[q];

                // h = -∇φ
                dealii::Tensor<1, dim> h;
                h[0] = -grad_phi[0];
                h[1] = -grad_phi[1];

                // ∇h = -Hess(φ)
                dealii::Tensor<2, dim> grad_h;
                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                        grad_h[i][j] = -hess_phi[i][j];

                max_h_norm = std::max(max_h_norm, h.norm());

                // κ_θ = κ₀ H(θ/ε)
                const double kappa_theta = compute_susceptibility_nochetto(c_val, epsilon, kappa_0);

                // Kelvin force: F_mag = μ₀ κ_θ (h·∇)h
                const double coeff = mu_0 * kappa_theta;
                for (unsigned int i = 0; i < dim; ++i)
                {
                    F_mag[i] = 0.0;
                    for (unsigned int j = 0; j < dim; ++j)
                        F_mag[i] += h[j] * grad_h[i][j];
                    F_mag[i] *= coeff;
                }
            }

            // ================================================================
            // Gravity [Eq 19]: f_g = (1 + r H(θ/ε)) g
            // ================================================================
            dealii::Tensor<1, dim> F_gravity;
            if (use_gravity)
            {
                const double H_theta = sigmoid(c_val / epsilon);
                const double gravity_factor = 1.0 + r_density * H_theta;
                F_gravity[0] = gravity_factor * g_mag * g_direction[0];
                F_gravity[1] = gravity_factor * g_mag * g_direction[1];
            }

            // Track max forces
            max_F_cap = std::max(max_F_cap, F_cap.norm());
            max_F_mag = std::max(max_F_mag, F_mag.norm());
            max_F_grav = std::max(max_F_grav, F_gravity.norm());

            // ================================================================
            // Convection term: (u_old · ∇)u_old
            // ================================================================
            double conv_ux = ux_old * grad_ux_old[0] + uy_old * grad_ux_old[1];
            double conv_uy = ux_old * grad_uy_old[0] + uy_old * grad_uy_old[1];

            // MMS source
            dealii::Tensor<1, dim> S_u;
            if (params.mms_mode)
            {
                S_u[0] = mms_source_u.value(x_q, 0);
                S_u[1] = mms_source_u.value(x_q, 1);
            }

            // ================================================================
            // Assemble local contributions
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                // ============================================================
                // RHS contributions for momentum equations
                // ============================================================

                // Time derivative: (1/dt)(u_old, v)
                double rhs_ux = (1.0 / dt) * ux_old * phi_ux_i;
                double rhs_uy = (1.0 / dt) * uy_old * phi_uy_i;

                // Explicit viscosity: -(1-θ)ν(∇u_old : ∇v)
                rhs_ux -= (1.0 - theta_time) * nu * (grad_ux_old * grad_phi_ux_i);
                rhs_uy -= (1.0 - theta_time) * nu * (grad_uy_old * grad_phi_uy_i);

                // Convection (explicit): -((u_old·∇)u_old, v)
                rhs_ux -= conv_ux * phi_ux_i;
                rhs_uy -= conv_uy * phi_uy_i;

                // External forces
                if (!params.mms_mode)
                {
                    rhs_ux += F_cap[0] * phi_ux_i + F_mag[0] * phi_ux_i + F_gravity[0] * phi_ux_i;
                    rhs_uy += F_cap[1] * phi_uy_i + F_mag[1] * phi_uy_i + F_gravity[1] * phi_uy_i;
                }

                // MMS source
                if (params.mms_mode)
                {
                    rhs_ux += S_u[0] * phi_ux_i;
                    rhs_uy += S_u[1] * phi_uy_i;
                }

                local_rhs_ux(i) += rhs_ux * JxW;
                local_rhs_uy(i) += rhs_uy * JxW;

                // ============================================================
                // Matrix contributions: velocity-velocity blocks
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // Mass term: (1/dt)(u, v)
                    local_ux_ux(i, j) += (1.0 / dt) * phi_ux_i * phi_ux_j * JxW;
                    local_uy_uy(i, j) += (1.0 / dt) * phi_uy_i * phi_uy_j * JxW;

                    // Implicit viscosity: θν(∇u : ∇v)
                    local_ux_ux(i, j) += theta_time * nu * (grad_phi_ux_i * grad_phi_ux_j) * JxW;
                    local_uy_uy(i, j) += theta_time * nu * (grad_phi_uy_i * grad_phi_uy_j) * JxW;

                    // Grad-div stabilization (optional)
                    if (params.grad_div_gamma > 0.0)
                    {
                        // div(u) = ∂ux/∂x + ∂uy/∂y
                        // (γ div(u), div(v))
                        local_ux_ux(i, j) += params.grad_div_gamma * grad_phi_ux_i[0] * grad_phi_ux_j[0] * JxW;
                        local_ux_uy(i, j) += params.grad_div_gamma * grad_phi_ux_i[0] * grad_phi_uy_j[1] * JxW;
                        local_uy_ux(i, j) += params.grad_div_gamma * grad_phi_uy_i[1] * grad_phi_ux_j[0] * JxW;
                        local_uy_uy(i, j) += params.grad_div_gamma * grad_phi_uy_i[1] * grad_phi_uy_j[1] * JxW;
                    }
                }

                // ============================================================
                // Matrix contributions: velocity-pressure coupling
                // -(p, div(v)) term from momentum equation
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);

                    // -(p, ∂v_x/∂x) and -(p, ∂v_y/∂y)
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // ================================================================
            // Continuity equation: -(div(u), q) = 0
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // -(∂u_x/∂x, q) and -(∂u_y/∂y, q)
                    local_p_ux(i, j) -= grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) -= grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }

        // ====================================================================
        // Assemble into global coupled matrix using index maps
        // ====================================================================

        // ux-ux block
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto gi = ux_to_ns_map[ux_local_dofs[i]];
            ns_rhs(gi) += local_rhs_ux(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_ux_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_ux_uy(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                ns_matrix.add(gi, p_to_ns_map[p_local_dofs[j]], local_ux_p(i, j));
        }

        // uy-uy block
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto gi = uy_to_ns_map[uy_local_dofs[i]];
            ns_rhs(gi) += local_rhs_uy(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_uy_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_uy_uy(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                ns_matrix.add(gi, p_to_ns_map[p_local_dofs[j]], local_uy_p(i, j));
        }

        // p row (continuity)
        for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
        {
            const auto gi = p_to_ns_map[p_local_dofs[i]];
            ns_rhs(gi) += local_rhs_p(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_p_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_p_uy(i, j));
            // p-p block is zero (no pressure stabilization)
        }
    }

    // Diagnostic output
    static double last_forces_time = -1.0;
    if (std::abs(current_time - last_forces_time) > 0.01)
    {
        if (params.verbose)
        {
            std::cout << "[FORCES] t=" << current_time
                      << " |h|_max=" << max_h_norm
                      << " |F_cap|_max=" << max_F_cap
                      << " |F_mag|_max=" << max_F_mag
                      << " |F_grav|_max=" << max_F_grav
                      << std::endl;
        }
        last_forces_time = current_time;
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ns_system_scalar<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>*,
    const NSCHParameters&,
    double, double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&);