// ============================================================================
// assembly/ch_assembler_scalar.cc - Cahn-Hilliard assembly for scalar DoFHandlers
//
// REFACTORED VERSION: Works with separate DoFHandlers instead of FESystem
//
// Equations (Nochetto 14a-14b):
//   θ_t + div(uθ) + γ Δψ = 0        (phase field evolution)
//   ψ - ε Δθ + (1/ε) f(θ) = 0       (chemical potential)
//
// Weak form:
//   Eq (14a): (1/dt)(θ,φ) - (uθ,∇φ) - γ(∇ψ,∇φ) = (1/dt)(θ_old,φ)
//   Eq (14b): (ψ,χ) + ε(∇θ,∇χ) + (1/ε)(f'(θ_old)θ,χ) = -(1/ε)(f(θ_old) - f'(θ_old)θ_old, χ)
// ============================================================================
#include "assembly/ch_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include "utilities/nsch_mms.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Double-well potential derivatives [Eq 2]
// F(θ) = (1/4)(θ² - 1)² for θ ∈ [-1, 1]
// f(θ) = F'(θ) = θ³ - θ
// f'(θ) = F''(θ) = 3θ² - 1
// ============================================================================
namespace
{
    inline double cahn_hilliard_f(double c)
    {
        return c * c * c - c;
    }

    inline double cahn_hilliard_f_prime(double c)
    {
        return 3.0 * c * c - 1.0;
    }
}

// ============================================================================
// Main assembly function
// ============================================================================
template <int dim>
void assemble_ch_system_scalar(
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::DoFHandler<dim>& mu_dof_handler,
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::Vector<double>&  c_old_solution,
    const dealii::Vector<double>&  ux_solution,
    const dealii::Vector<double>&  uy_solution,
    const NSCHParameters&          params,
    double                         dt,
    double                         current_time,
    dealii::SparseMatrix<double>&  ch_matrix,
    dealii::Vector<double>&        ch_rhs,
    const std::vector<dealii::types::global_dof_index>& c_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& mu_to_ch_map)
{
    ch_matrix = 0;
    ch_rhs = 0;

    // All scalar fields use Q2, so they have the same FE
    const auto& fe_Q2 = c_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe_Q2.n_dofs_per_cell();

    // Quadrature
    dealii::QGauss<dim> quadrature(params.fe_degree_phase + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for each scalar field
    dealii::FEValues<dim> c_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> mu_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> ux_fe_values(fe_Q2, quadrature,
        dealii::update_values);

    dealii::FEValues<dim> uy_fe_values(fe_Q2, quadrature,
        dealii::update_values);

    // Local matrices for the 2x2 block structure
    dealii::FullMatrix<double> local_matrix_cc(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_cm(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_mc(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_mm(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_c(dofs_per_cell);
    dealii::Vector<double> local_rhs_m(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> c_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> mu_local_dofs(dofs_per_cell);

    // Storage for solution values at quadrature points
    std::vector<double> c_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> c_old_gradients(n_q_points);
    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);

    // Physical parameters [Nochetto notation]
    const double epsilon = params.epsilon;   // ε: interface thickness
    const double gamma   = params.mobility;  // γ: mobility

    // MMS source terms (if enabled)
    MMSSourceC<dim> mms_source_c;
    MMSSourceMu<dim> mms_source_mu;
    if (params.mms_mode)
    {
        mms_source_c.set_time(current_time);
        mms_source_mu.set_time(current_time);
    }

    // Iterate over cells - all DoFHandlers share the same mesh
    auto c_cell = c_dof_handler.begin_active();
    auto mu_cell = mu_dof_handler.begin_active();
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();

    for (; c_cell != c_dof_handler.end(); ++c_cell, ++mu_cell, ++ux_cell, ++uy_cell)
    {
        c_fe_values.reinit(c_cell);
        mu_fe_values.reinit(mu_cell);
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        local_matrix_cc = 0;
        local_matrix_cm = 0;
        local_matrix_mc = 0;
        local_matrix_mm = 0;
        local_rhs_c = 0;
        local_rhs_m = 0;

        // Get solution values at quadrature points
        c_fe_values.get_function_values(c_old_solution, c_old_values);
        c_fe_values.get_function_gradients(c_old_solution, c_old_gradients);
        ux_fe_values.get_function_values(ux_solution, ux_values);
        uy_fe_values.get_function_values(uy_solution, uy_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = c_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = c_fe_values.quadrature_point(q);

            const double theta_old = c_old_values[q];
            const dealii::Tensor<1, dim>& grad_theta_old = c_old_gradients[q];

            // Velocity at quadrature point
            dealii::Tensor<1, dim> u;
            u[0] = ux_values[q];
            u[1] = uy_values[q];

            // Double-well potential derivatives
            const double f_old  = cahn_hilliard_f(theta_old);
            const double fp_old = cahn_hilliard_f_prime(theta_old);

            // MMS source terms
            double S_theta = 0.0, S_psi = 0.0;
            if (params.mms_mode)
            {
                S_theta = mms_source_c.value(x_q);
                S_psi   = mms_source_mu.value(x_q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Shape functions for c (θ)
                const double phi_i = c_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_i = c_fe_values.shape_grad(i, q);

                // Shape functions for μ (ψ) - same FE, same values
                const double chi_i = mu_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_chi_i = mu_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = c_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = c_fe_values.shape_grad(j, q);
                    const double chi_j = mu_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_chi_j = mu_fe_values.shape_grad(j, q);

                    // ========================================================
                    // Equation (14a): θ_t + div(uθ) + γ Δψ = 0
                    // Weak: (1/dt)(θ,φ) - (uθ,∇φ) - γ(∇ψ,∇φ) = (1/dt)(θ_old,φ)
                    // ========================================================

                    // (1/dt)(θ, φ)  [c-c block]
                    local_matrix_cc(i, j) += (1.0 / dt) * phi_i * phi_j * JxW;

                    // -(uθ, ∇φ) = -θ(u·∇φ)  [c-c block, advection]
                    local_matrix_cc(i, j) -= phi_j * (u * grad_phi_i) * JxW;

                    // -γ(∇ψ, ∇φ)  [c-μ block]
                    local_matrix_cm(i, j) -= gamma * (grad_chi_j * grad_phi_i) * JxW;

                    // ========================================================
                    // Equation (14b): ψ - ε Δθ + (1/ε) f(θ) = 0
                    // Weak: (ψ,χ) + ε(∇θ,∇χ) + (1/ε)(f'(θ_old)θ,χ) = RHS
                    // ========================================================

                    // (ψ, χ)  [μ-μ block]
                    local_matrix_mm(i, j) += chi_i * chi_j * JxW;

                    // +ε(∇θ, ∇χ)  [μ-c block]
                    local_matrix_mc(i, j) += epsilon * (grad_phi_j * grad_chi_i) * JxW;

                    // +(1/ε)(f'(θ_old)θ, χ)  [μ-c block, linearized potential]
                    local_matrix_mc(i, j) += (1.0 / epsilon) * fp_old * phi_j * chi_i * JxW;
                }

                // ============================================================
                // RHS contributions
                // ============================================================

                // Eq (14a) RHS: (1/dt)(θ_old, φ) + MMS source
                local_rhs_c(i) += (1.0 / dt) * theta_old * phi_i * JxW;
                if (params.mms_mode)
                    local_rhs_c(i) += S_theta * phi_i * JxW;

                // Eq (14b) RHS: -(1/ε)(f(θ_old) - f'(θ_old)θ_old, χ) + MMS source
                local_rhs_m(i) -= (1.0 / epsilon) * (f_old - fp_old * theta_old) * chi_i * JxW;
                if (params.mms_mode)
                    local_rhs_m(i) += S_psi * chi_i * JxW;
            }
        }

        // Get DoF indices for this cell
        c_cell->get_dof_indices(c_local_dofs);
        mu_cell->get_dof_indices(mu_local_dofs);

        // ====================================================================
        // Assemble into global coupled matrix using index maps
        // ====================================================================
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const auto gi_c = c_to_ch_map[c_local_dofs[i]];
            const auto gi_m = mu_to_ch_map[mu_local_dofs[i]];

            // RHS
            ch_rhs(gi_c) += local_rhs_c(i);
            ch_rhs(gi_m) += local_rhs_m(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const auto gj_c = c_to_ch_map[c_local_dofs[j]];
                const auto gj_m = mu_to_ch_map[mu_local_dofs[j]];

                // c-c block
                ch_matrix.add(gi_c, gj_c, local_matrix_cc(i, j));

                // c-μ block
                ch_matrix.add(gi_c, gj_m, local_matrix_cm(i, j));

                // μ-c block
                ch_matrix.add(gi_m, gj_c, local_matrix_mc(i, j));

                // μ-μ block
                ch_matrix.add(gi_m, gj_m, local_matrix_mm(i, j));
            }
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ch_system_scalar<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const NSCHParameters&,
    double, double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&);