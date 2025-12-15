// ============================================================================
// assembly/ch_assembler.cc - Cahn-Hilliard System Assembler (CORRECTED)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
//
// CORRECTED to match paper:
//   1. Advection uses LAGGED θ^{k-1} (on RHS), not implicit θ^k
//   2. f(θ) is LAGGED f(θ^{k-1}) (on RHS), not Newton linearized
//   3. Stabilization term (1/η)(δθ^k, Υ) is included
//
// Paper's discrete scheme:
//
//   Eq 42a: (δθ^k/τ, Λ) - (U^k θ^{k-1}, ∇Λ) - γ(∇ψ^k, ∇Λ) = 0
//
//   Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/ε)(f(θ^{k-1}), Υ) + (1/η)(δθ^k, Υ) = 0
//
// Rearranged for matrix form (θ^k, ψ^k on LHS):
//
//   Eq 42a: (1/τ)(θ^k, Λ) - γ(∇ψ^k, ∇Λ) = (1/τ)(θ^{k-1}, Λ) + (U^k θ^{k-1}, ∇Λ)
//
//   Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/η)(θ^k, Υ) = -(1/ε)(f(θ^{k-1}), Υ) + (1/η)(θ^{k-1}, Υ)
//
// ============================================================================

#include "ch_assembler.h"
#include "utilities/parameters.h"
#include "diagnostics/ch_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>

// ============================================================================
// Double-well potential (Eq. 2, p.498)
//   F(θ) = (1/4)(θ² - 1)²
//   f(θ) = F'(θ) = θ³ - θ
// ============================================================================
inline double f_double_well(double theta)
{
    return theta * theta * theta - theta;
}

// ============================================================================
// Main assembly function (CORRECTED to match Paper Eq. 42a-42b)
// ============================================================================
template <int dim>
void assemble_ch_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>&  theta_old,
    const dealii::Vector<double>&  ux_solution,
    const dealii::Vector<double>&  uy_solution,
    const Parameters&              params,
    double                         dt,
    double                         current_time,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::SparseMatrix<double>&  matrix,
    dealii::Vector<double>&        rhs)
{


    // Zero output
    matrix = 0;
    rhs = 0;

    // Get FE from DoFHandler (both θ and ψ use same FE)
    const auto& fe = theta_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // Quadrature
    const unsigned int quad_degree = params.fe.degree_phase + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for each field
    dealii::FEValues<dim> theta_fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> psi_fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients);

    // For velocity (same mesh assumed)
    dealii::FEValues<dim> ux_fe_values(fe, quadrature, dealii::update_values);
    dealii::FEValues<dim> uy_fe_values(fe, quadrature, dealii::update_values);

    // Local matrices for 2x2 block structure
    dealii::FullMatrix<double> local_matrix_tt(dofs_per_cell, dofs_per_cell);  // θ-θ
    dealii::FullMatrix<double> local_matrix_tp(dofs_per_cell, dofs_per_cell);  // θ-ψ
    dealii::FullMatrix<double> local_matrix_pt(dofs_per_cell, dofs_per_cell);  // ψ-θ
    dealii::FullMatrix<double> local_matrix_pp(dofs_per_cell, dofs_per_cell);  // ψ-ψ
    dealii::Vector<double> local_rhs_t(dofs_per_cell);
    dealii::Vector<double> local_rhs_p(dofs_per_cell);

    // DoF indices
    std::vector<dealii::types::global_dof_index> theta_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_local_dofs(dofs_per_cell);

    // Solution values at quadrature points
    std::vector<double> theta_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_old_gradients(n_q_points);
    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);

    // Physical parameters
    const double epsilon = params.ch.epsilon;
    const double gamma = params.ch.gamma;
    const double eta = params.ch.eta;  // Stabilization parameter (η ~ ε)

    // Validate stabilization parameter
    if (eta <= 0.0)
    {
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "[CH] WARNING: eta <= 0. Paper requires eta > 0 for stability.\n"
                      << "  Recommended: eta ~ epsilon = " << epsilon << "\n";
            warned = true;
        }
    }

    // Check if velocity is provided
    const bool have_velocity = (ux_solution.size() > 0) &&
                               (uy_solution.size() > 0) &&
                               (ux_solution.l2_norm() + uy_solution.l2_norm() > 1e-14);

    // MMS source terms (only if MMS mode enabled)
    const bool mms_mode = params.mms.enabled;
    CHSourceTheta<dim> source_theta(gamma);
    CHSourcePsi<dim> source_psi(epsilon);
    if (mms_mode)
    {
        source_theta.set_time(current_time);
        source_psi.set_time(current_time);
    }

    // Iterate over cells
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    for (; theta_cell != theta_dof_handler.end(); ++theta_cell, ++psi_cell)
    {
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        // Get velocity values
        if (have_velocity)
        {
            typename dealii::DoFHandler<dim>::active_cell_iterator
                ux_cell(&theta_dof_handler.get_triangulation(),
                        theta_cell->level(), theta_cell->index(),
                        &theta_dof_handler);
            ux_fe_values.reinit(ux_cell);
            uy_fe_values.reinit(ux_cell);

            ux_fe_values.get_function_values(ux_solution, ux_values);
            uy_fe_values.get_function_values(uy_solution, uy_values);
        }
        else
        {
            std::fill(ux_values.begin(), ux_values.end(), 0.0);
            std::fill(uy_values.begin(), uy_values.end(), 0.0);
        }

        // Reset local contributions
        local_matrix_tt = 0;
        local_matrix_tp = 0;
        local_matrix_pt = 0;
        local_matrix_pp = 0;
        local_rhs_t = 0;
        local_rhs_p = 0;

        // Get old solution values
        theta_fe_values.get_function_values(theta_old, theta_old_values);

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const double theta_old_q = theta_old_values[q];

            // Velocity at quadrature point
            dealii::Tensor<1, dim> U;
            U[0] = ux_values[q];
            if constexpr (dim >= 2)
                U[1] = uy_values[q];

            // Double-well derivative at θ^{k-1} (LAGGED, not linearized!)
            const double f_old = f_double_well(theta_old_q);

            // Shape function loop
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // θ test function Λ
                const double Lambda_i = theta_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_Lambda_i = theta_fe_values.shape_grad(i, q);

                // ψ test function Υ
                const double Upsilon_i = psi_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_Upsilon_i = psi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // θ trial function
                    const double theta_j = theta_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_theta_j = theta_fe_values.shape_grad(j, q);

                    // ψ trial function
                    const double psi_j = psi_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_psi_j = psi_fe_values.shape_grad(j, q);

                    // ============================================================
                    // Eq 42a: (1/τ)(θ^k, Λ) - γ(∇ψ^k, ∇Λ) = RHS
                    // ============================================================

                    // θ-θ block: (1/τ)(θ, Λ)
                    local_matrix_tt(i, j) += (1.0 / dt) * theta_j * Lambda_i * JxW;

                    // θ-ψ block: -γ(∇ψ, ∇Λ)
                    local_matrix_tp(i, j) -= gamma * (grad_psi_j * grad_Lambda_i) * JxW;

                    // ============================================================
                    // Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/η)(θ^k, Υ) = RHS
                    // ============================================================

                    // ψ-ψ block: (ψ, Υ)
                    local_matrix_pp(i, j) += psi_j * Upsilon_i * JxW;

                    // ψ-θ block: ε(∇θ, ∇Υ) + (1/η)(θ, Υ)
                    local_matrix_pt(i, j) += epsilon * (grad_theta_j * grad_Upsilon_i) * JxW;
                    if (eta > 0.0)
                    {
                        local_matrix_pt(i, j) += (1.0 / eta) * theta_j * Upsilon_i * JxW;
                    }
                }

                // ============================================================
                // RHS contributions
                // ============================================================

                // Eq 42a RHS: (1/τ)(θ^{k-1}, Λ) + (U^k θ^{k-1}, ∇Λ)
                //
                // Note: Paper has -(U^k θ^{k-1}, ∇Λ) on LHS, so +(U^k θ^{k-1}, ∇Λ) on RHS
                local_rhs_t(i) += (1.0 / dt) * theta_old_q * Lambda_i * JxW;
                local_rhs_t(i) += theta_old_q * (U * grad_Lambda_i) * JxW;

                // Eq 42b RHS: -(1/ε)(f(θ^{k-1}), Υ) + (1/η)(θ^{k-1}, Υ)
                //
                // Note: Paper has +(1/ε)(f(θ^{k-1}), Υ) on LHS, so -(1/ε)(f(θ^{k-1}), Υ) on RHS
                local_rhs_p(i) -= (1.0 / epsilon) * f_old * Upsilon_i * JxW;
                if (eta > 0.0)
                {
                    local_rhs_p(i) += (1.0 / eta) * theta_old_q * Upsilon_i * JxW;
                }

                // MMS source terms (if enabled)
                if (mms_mode)
                {
                    const auto& q_point = theta_fe_values.quadrature_point(q);
                    const double S_theta = source_theta.value(q_point);
                    const double S_psi = source_psi.value(q_point);

                    local_rhs_t(i) += S_theta * Lambda_i * JxW;
                    local_rhs_p(i) += S_psi * Upsilon_i * JxW;
                }
            }
        }

        // Get global DoF indices
        theta_cell->get_dof_indices(theta_local_dofs);
        psi_cell->get_dof_indices(psi_local_dofs);

        // ============================================================
        // Assemble into global coupled matrix using index maps
        // ============================================================
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const auto gi_t = theta_to_ch_map[theta_local_dofs[i]];
            const auto gi_p = psi_to_ch_map[psi_local_dofs[i]];

            // RHS
            rhs(gi_t) += local_rhs_t(i);
            rhs(gi_p) += local_rhs_p(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const auto gj_t = theta_to_ch_map[theta_local_dofs[j]];
                const auto gj_p = psi_to_ch_map[psi_local_dofs[j]];

                // θ-θ block
                matrix.add(gi_t, gj_t, local_matrix_tt(i, j));

                // θ-ψ block
                matrix.add(gi_t, gj_p, local_matrix_tp(i, j));

                // ψ-θ block
                matrix.add(gi_p, gj_t, local_matrix_pt(i, j));

                // ψ-ψ block
                matrix.add(gi_p, gj_p, local_matrix_pp(i, j));
            }
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ch_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    double,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&);