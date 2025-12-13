// ============================================================================
// assembly/ns_assembler.cc - Navier-Stokes Assembly Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/ns_assembler.h"
#include "physics/material_properties.h"
#include "physics/kelvin_force.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <int dim>
void assemble_ns_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>& ux_old,
    const dealii::Vector<double>& uy_old,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::SparseMatrix<double>& ns_matrix,
    dealii::Vector<double>& ns_rhs)
{
    (void)current_time;  // For future MMS use

    ns_matrix = 0;
    ns_rhs = 0;

    const auto& fe_Q2 = ux_dof_handler.get_fe();
    const auto& fe_Q1 = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell_Q2 = fe_Q2.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = fe_Q1.n_dofs_per_cell();

    // Quadrature
    dealii::QGauss<dim> quadrature(params.fe.degree_velocity + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for each field
    dealii::FEValues<dim> ux_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> p_fe_values(fe_Q1, quadrature,
        dealii::update_values);

    dealii::FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        dealii::update_values);

    dealii::FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        dealii::update_gradients);

    // FEValues for magnetic potential (conditional)
    std::unique_ptr<dealii::FEValues<dim>> phi_fe_values;
    if (params.magnetic.enabled && phi_dof_handler != nullptr)
    {
        phi_fe_values = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler->get_fe(), quadrature,
            dealii::update_gradients | dealii::update_hessians);
    }

    // Local matrices (9 blocks for 3×3 system)
    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_uy_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_p_ux(dofs_per_cell_Q1, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_p_uy(dofs_per_cell_Q1, dofs_per_cell_Q2);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_p(dofs_per_cell_Q1);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_local_dofs(dofs_per_cell_Q1);

    // Solution values at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);

    // Physical parameters
    const double theta_time = params.time.theta;  // Time stepping parameter
    const double lambda = params.ch.lambda;
    const double epsilon = params.ch.epsilon;
    const double nu_water = params.ns.nu_water;
    const double nu_ferro = params.ns.nu_ferro;
    const double mu_0 = params.ns.mu_0;
    const double chi_0 = params.magnetization.chi_0;
    const double grad_div_gamma = params.ns.grad_div;

    // Material properties helper
    MaterialProperties mat_props(nu_water, nu_ferro, chi_0, epsilon);

    // Gravity
    const bool use_gravity = params.gravity.enabled;
    const double g_mag = params.gravity.magnitude;
    const double r_density = params.ns.r;
    dealii::Tensor<1, dim> g_direction = params.gravity.direction;

    // Diagnostic tracking
    double max_F_cap = 0.0, max_F_mag = 0.0, max_F_grav = 0.0;

    // Cell iterators (all share same mesh)
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator phi_cell;
    if (params.magnetic.enabled && phi_dof_handler != nullptr)
        phi_cell = phi_dof_handler->begin_active();

    for (; ux_cell != ux_dof_handler.end();
         ++ux_cell, ++uy_cell, ++p_cell, ++theta_cell, ++psi_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        if (params.magnetic.enabled && phi_dof_handler != nullptr)
        {
            phi_fe_values->reinit(phi_cell);
            ++phi_cell;
        }

        // Reset local matrices
        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux = 0;  local_p_uy = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        // Get DoF indices
        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get solution values at quadrature points
        ux_fe_values.get_function_values(ux_old, ux_old_values);
        uy_fe_values.get_function_values(uy_old, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);
        theta_fe_values.get_function_values(theta_solution, theta_values);
        psi_fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (params.magnetic.enabled && phi_dof_handler != nullptr && phi_solution != nullptr)
        {
            phi_fe_values->get_function_gradients(*phi_solution, phi_gradients);
            phi_fe_values->get_function_hessians(*phi_solution, phi_hessians);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);

            const double ux_q = ux_old_values[q];
            const double uy_q = uy_old_values[q];
            const dealii::Tensor<1, dim>& grad_ux_old = ux_old_gradients[q];
            const dealii::Tensor<1, dim>& grad_uy_old = uy_old_gradients[q];

            const double theta_q = theta_values[q];
            const dealii::Tensor<1, dim>& grad_psi = psi_gradients[q];

            // Phase-dependent viscosity [Eq. 17]
            const double nu = mat_props.viscosity(theta_q);

            // ================================================================
            // Capillary force [Eq. 10]: F_cap = (λ/ε)θ∇ψ
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            {
                const double coeff = lambda / epsilon;
                F_cap[0] = coeff * theta_q * grad_psi[0];
                F_cap[1] = coeff * theta_q * grad_psi[1];
            }

            // ================================================================
            // Kelvin force [Eq. 14f]: F_mag = μ₀χ(θ)(H·∇)H
            // ================================================================
            dealii::Tensor<1, dim> F_mag;
            if (params.magnetic.enabled && phi_dof_handler != nullptr && phi_solution != nullptr)
            {
                const dealii::Tensor<1, dim>& grad_phi = phi_gradients[q];
                const dealii::Tensor<2, dim>& hess_phi = phi_hessians[q];

                // H = -∇φ, ∇H = -Hess(φ)
                auto H = compute_magnetic_field<dim>(grad_phi);
                auto grad_H = compute_grad_H<dim>(hess_phi);

                // χ(θ) = χ₀ H(θ/ε)
                const double chi_theta = mat_props.susceptibility(theta_q);

                // F_mag = μ₀χ(θ)(H·∇)H
                F_mag = compute_kelvin_force<dim>(H, grad_H, chi_theta, mu_0);
            }

            // ================================================================
            // Gravity [Eq. 19]: F_grav = (1 + r·H(θ/ε))g
            // ================================================================
            dealii::Tensor<1, dim> F_grav;
            if (use_gravity)
            {
                const double H_theta = MaterialProperties::heaviside(theta_q / epsilon);
                const double gravity_factor = 1.0 + r_density * H_theta;
                F_grav = gravity_factor * g_mag * g_direction;
            }

            // Track max forces for diagnostics
            max_F_cap = std::max(max_F_cap, F_cap.norm());
            max_F_mag = std::max(max_F_mag, F_mag.norm());
            max_F_grav = std::max(max_F_grav, F_grav.norm());

            // ================================================================
            // Convection term: (u_old · ∇)u_old
            // ================================================================
            const double conv_ux = ux_q * grad_ux_old[0] + uy_q * grad_ux_old[1];
            const double conv_uy = ux_q * grad_uy_old[0] + uy_q * grad_uy_old[1];

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
                // RHS: momentum equations
                // ============================================================

                // Time derivative: (1/dt)(u_old, v)
                double rhs_ux = (1.0 / dt) * ux_q * phi_ux_i;
                double rhs_uy = (1.0 / dt) * uy_q * phi_uy_i;

                // Explicit viscosity: -(1-θ)ν(∇u_old, ∇v)
                rhs_ux -= (1.0 - theta_time) * nu * (grad_ux_old * grad_phi_ux_i);
                rhs_uy -= (1.0 - theta_time) * nu * (grad_uy_old * grad_phi_uy_i);

                // Convection (explicit): -((u_old·∇)u_old, v)
                rhs_ux -= conv_ux * phi_ux_i;
                rhs_uy -= conv_uy * phi_uy_i;

                // External forces
                rhs_ux += (F_cap[0] + F_mag[0] + F_grav[0]) * phi_ux_i;
                rhs_uy += (F_cap[1] + F_mag[1] + F_grav[1]) * phi_uy_i;

                local_rhs_ux(i) += rhs_ux * JxW;
                local_rhs_uy(i) += rhs_uy * JxW;

                // ============================================================
                // Matrix: velocity-velocity blocks
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // Mass: (1/dt)(u, v)
                    local_ux_ux(i, j) += (1.0 / dt) * phi_ux_i * phi_ux_j * JxW;
                    local_uy_uy(i, j) += (1.0 / dt) * phi_uy_i * phi_uy_j * JxW;

                    // Implicit viscosity: θν(∇u, ∇v)
                    local_ux_ux(i, j) += theta_time * nu * (grad_phi_ux_i * grad_phi_ux_j) * JxW;
                    local_uy_uy(i, j) += theta_time * nu * (grad_phi_uy_i * grad_phi_uy_j) * JxW;

                    // Grad-div stabilization (optional)
                    if (grad_div_gamma > 0.0)
                    {
                        // (γ div(u), div(v))
                        local_ux_ux(i, j) += grad_div_gamma * grad_phi_ux_i[0] * grad_phi_ux_j[0] * JxW;
                        local_ux_uy(i, j) += grad_div_gamma * grad_phi_ux_i[0] * grad_phi_uy_j[1] * JxW;
                        local_uy_ux(i, j) += grad_div_gamma * grad_phi_uy_i[1] * grad_phi_ux_j[0] * JxW;
                        local_uy_uy(i, j) += grad_div_gamma * grad_phi_uy_i[1] * grad_phi_uy_j[1] * JxW;
                    }
                }

                // ============================================================
                // Matrix: velocity-pressure coupling -(p, div(v))
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

                    // -(∂u_x/∂x + ∂u_y/∂y, q)
                    local_p_ux(i, j) -= grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) -= grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }

        // ====================================================================
        // Assemble into global matrix using index maps
        // ====================================================================

        // ux block
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

        // uy block
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

        // p block (continuity)
        for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
        {
            const auto gi = p_to_ns_map[p_local_dofs[i]];
            ns_rhs(gi) += local_rhs_p(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_p_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_p_uy(i, j));
        }
    }

    // Diagnostic output
    if (params.output.verbose)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)
        {
            std::cout << "[NS] Forces: |F_cap|=" << max_F_cap
                      << ", |F_mag|=" << max_F_mag
                      << ", |F_grav|=" << max_F_grav << "\n";
        }
        ++call_count;
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ns_system<2>(
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
    const Parameters&,
    double,
    double,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&);