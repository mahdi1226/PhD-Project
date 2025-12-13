// ============================================================================
// assembly/ch_assembler.cc - Cahn-Hilliard System Assembler Implementation
//
// Extracted from OLD nsch_problem_solver.cc lines 336-450
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 14a-14b, p.499
//
// Weak form derivation:
//   Eq 14a: θ_t + div(uθ) - γΔψ = 0
//     → (θ_t, φ) + (div(uθ), φ) + γ(∇ψ, ∇φ) = 0        [IBP on Laplacian]
//     → (1/τ)(θ - θ_old, φ) - (uθ, ∇φ) + γ(∇ψ, ∇φ) = 0  [IBP on div, BDF1]
//
//   Eq 14b: ψ + (1/ε)f(θ) - εΔθ = 0
//     → (ψ, χ) + (1/ε)(f(θ), χ) + ε(∇θ, ∇χ) = 0        [IBP on Laplacian]
//     Linearization: f(θ) ≈ f(θ_old) + f'(θ_old)(θ - θ_old)
//
// MMS MODE: When params.mms.enabled is true, adds source terms S_θ and S_ψ
// to RHS for manufactured solution verification.
// ============================================================================

#include "ch_assembler.h"
#include "utilities/parameters.h"
#include "diagnostics/ch_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// Double-well potential (Eq. 2, p.498)
//   F(θ) = (1/4)(θ² - 1)²
//   f(θ) = F'(θ) = θ³ - θ
//   f'(θ) = F''(θ) = 3θ² - 1
// ============================================================================
inline double f_double_well(double theta)
{
    return theta * theta * theta - theta;
}

inline double f_prime_double_well(double theta)
{
    return 3.0 * theta * theta - 1.0;
}

// ============================================================================
// Main assembly function
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

    // For velocity (if provided, same FE degree as phase)
    // We need separate FEValues only if velocity uses different mesh/FE
    // Here we assume same mesh, so we can evaluate velocity at same quadrature points
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
    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);

    // Physical parameters
    const double epsilon = params.ch.epsilon;
    const double gamma = params.ch.gamma;

    // Check if velocity is provided (non-empty AND non-zero norm)
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

    // Iterate over cells - all DoFHandlers share the same triangulation
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    for (; theta_cell != theta_dof_handler.end(); ++theta_cell, ++psi_cell)
    {
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        // Get corresponding velocity cells (same mesh)
        if (have_velocity)
        {
            // Create cell accessors for velocity DoFHandlers
            // They share the same triangulation, so same level/index
            typename dealii::DoFHandler<dim>::active_cell_iterator
                ux_cell(&theta_dof_handler.get_triangulation(),
                        theta_cell->level(), theta_cell->index(),
                        &theta_dof_handler);  // Using theta_dof for now (same FE)
            ux_fe_values.reinit(ux_cell);
            uy_fe_values.reinit(ux_cell);

            ux_fe_values.get_function_values(ux_solution, ux_values);
            uy_fe_values.get_function_values(uy_solution, uy_values);
        }
        else
        {
            // Zero velocity
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
            dealii::Tensor<1, dim> u;
            u[0] = ux_values[q];
            if constexpr (dim >= 2)
                u[1] = uy_values[q];

            // Double-well derivatives at θ_old
            const double f_old = f_double_well(theta_old_q);
            const double fp_old = f_prime_double_well(theta_old_q);

            // Shape function loop
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // θ shape functions
                const double phi_i = theta_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_i = theta_fe_values.shape_grad(i, q);

                // ψ shape functions (same FE, but conceptually separate)
                const double chi_i = psi_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_chi_i = psi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = theta_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = theta_fe_values.shape_grad(j, q);
                    const double chi_j = psi_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_chi_j = psi_fe_values.shape_grad(j, q);

                    // ================================================================
                    // θ equation (Eq. 14a): (1/τ)(θ,φ) - (uθ,∇φ) + γ(∇ψ,∇φ) = (1/τ)(θ_old,φ)
                    // ================================================================

                    // θ-θ block: (1/τ)(θ,φ) - (uθ,∇φ)
                    // Mass: (1/τ)(θ,φ)
                    local_matrix_tt(i, j) += (1.0 / dt) * phi_i * phi_j * JxW;
                    // Advection (conservative): -(uθ,∇φ) = -θ(u·∇φ)
                    local_matrix_tt(i, j) -= phi_j * (u * grad_phi_i) * JxW;

                    // θ-ψ block: +γ(∇ψ,∇φ)
                    local_matrix_tp(i, j) -= gamma * (grad_chi_j * grad_phi_i) * JxW;

                    // ================================================================
                    // ψ equation (Eq. 14b): (ψ,χ) + ε(∇θ,∇χ) + (1/ε)(f'θ,χ) = (1/ε)(f'θ_old - f, χ)
                    // ================================================================

                    // ψ-ψ block: (ψ,χ)
                    local_matrix_pp(i, j) += chi_i * chi_j * JxW;

                    // ψ-θ block: ε(∇θ,∇χ) + (1/ε)(f'(θ_old)θ,χ)
                    local_matrix_pt(i, j) += epsilon * (grad_phi_j * grad_chi_i) * JxW;
                    local_matrix_pt(i, j) += (1.0 / epsilon) * fp_old * phi_j * chi_i * JxW;
                }

                // ================================================================
                // RHS contributions
                // ================================================================

                // θ equation RHS: (1/τ)(θ_old,φ)
                local_rhs_t(i) += (1.0 / dt) * theta_old_q * phi_i * JxW;

                // ψ equation RHS: (1/ε)(f'(θ_old)θ_old - f(θ_old), χ)
                // = (1/ε)((3θ²-1)θ - (θ³-θ), χ)
                // = (1/ε)(3θ³ - θ - θ³ + θ, χ)
                // = (1/ε)(2θ³, χ)
                // But let's keep the general form for clarity:
                local_rhs_p(i) += (1.0 / epsilon) * (fp_old * theta_old_q - f_old) * chi_i * JxW;

                // MMS source terms (if enabled)
                if (mms_mode)
                {
                    const auto& q_point = theta_fe_values.quadrature_point(q);
                    const double S_theta = source_theta.value(q_point);
                    const double S_psi = source_psi.value(q_point);

                    // Add source to θ equation: (S_θ, φ)
                    local_rhs_t(i) += S_theta * phi_i * JxW;

                    // Add source to ψ equation: (S_ψ, χ)
                    local_rhs_p(i) += S_psi * chi_i * JxW;
                }
            }
        }

        // Get global DoF indices
        theta_cell->get_dof_indices(theta_local_dofs);
        psi_cell->get_dof_indices(psi_local_dofs);

        // ================================================================
        // Assemble into global coupled matrix using index maps
        // ================================================================
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