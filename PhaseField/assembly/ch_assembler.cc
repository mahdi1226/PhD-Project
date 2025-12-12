// ============================================================================
// assembly/ch_assembler.cc - Cahn-Hilliard System Assembler
//
// FIXED VERSION: Uses index maps for coupled system assembly
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-b, p.505
//
// Discrete scheme:
//   (42a): (δΘ^k/τ, Φ) - (U^{k-1}Θ^{k-1}, ∇Φ) + γ(∇Ψ^k, ∇Φ) = 0
//   (42b): (Ψ^k, Λ) - ε(∇Θ^k, ∇Λ) - (1/ε)(f(Θ^{k-1}), Λ) = 0
//
// Newton linearization of f(θ):
//   f(θ^k) ≈ f(θ^{k-1}) + f'(θ^{k-1})(θ^k - θ^{k-1})
// ============================================================================

#include "ch_assembler.h"
#include "output/logger.h"
#include "physics/material_properties.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

template <int dim>
CHAssembler<dim>::CHAssembler(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      CHAssembler constructed");
}

template <int dim>
void CHAssembler<dim>::assemble(double dt, double current_time)
{
    (void)current_time;

    const double inv_tau = 1.0 / dt;
    const double gamma   = problem_.params_.ch.gamma;    // Mobility
    const double epsilon = problem_.params_.ch.epsilon;  // Interface thickness

    // Clear matrix and RHS
    problem_.ch_matrix_ = 0;
    problem_.ch_rhs_    = 0;

    // Setup FE values
    const dealii::QGauss<dim> quadrature(problem_.fe_Q2_.degree + 1);

    dealii::FEValues<dim> fe_values(problem_.fe_Q2_, quadrature,
                                     dealii::update_values |
                                     dealii::update_gradients |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);

    const unsigned int dofs_per_cell = problem_.fe_Q2_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();

    // Local matrices for 2x2 block structure [θ | ψ]
    dealii::FullMatrix<double> local_theta_theta(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_theta_psi(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_psi_theta(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_psi_psi(dofs_per_cell, dofs_per_cell);

    dealii::Vector<double> local_rhs_theta(dofs_per_cell);
    dealii::Vector<double> local_rhs_psi(dofs_per_cell);

    // DoF indices
    std::vector<dealii::types::global_dof_index> theta_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_dofs(dofs_per_cell);

    // Field values at quadrature points
    std::vector<double> theta_old_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_theta_old(n_q_points);
    std::vector<double> ux_vals(n_q_points), uy_vals(n_q_points);

    // Iterate over cells
    auto theta_cell = problem_.theta_dof_handler_.begin_active();
    auto psi_cell = problem_.psi_dof_handler_.begin_active();
    auto ux_cell = problem_.ux_dof_handler_.begin_active();
    auto uy_cell = problem_.uy_dof_handler_.begin_active();

    for (; theta_cell != problem_.theta_dof_handler_.end();
         ++theta_cell, ++psi_cell, ++ux_cell, ++uy_cell)
    {
        fe_values.reinit(theta_cell);

        // Reset local matrices
        local_theta_theta = 0; local_theta_psi = 0;
        local_psi_theta = 0; local_psi_psi = 0;
        local_rhs_theta = 0; local_rhs_psi = 0;

        // Get DoF indices
        theta_cell->get_dof_indices(theta_dofs);
        psi_cell->get_dof_indices(psi_dofs);

        // Get field values
        fe_values.get_function_values(problem_.theta_old_solution_, theta_old_vals);
        fe_values.get_function_gradients(problem_.theta_old_solution_, grad_theta_old);
        fe_values.get_function_values(problem_.ux_solution_, ux_vals);
        fe_values.get_function_values(problem_.uy_solution_, uy_vals);

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);

            const double theta_old = theta_old_vals[q];
            const auto& grad_theta = grad_theta_old[q];

            // Velocity at quadrature point
            dealii::Tensor<1, dim> u;
            u[0] = ux_vals[q];
            u[1] = uy_vals[q];

            // Double-well potential and its derivative [Eq. 2-3]
            // f(θ) = θ³ - θ,  f'(θ) = 3θ² - 1
            const double f_old = MaterialProperties::double_well_derivative(theta_old);
            const double fp_old = 3.0 * theta_old * theta_old - 1.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const auto grad_phi_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values.shape_value(j, q);
                    const auto grad_phi_j = fe_values.shape_grad(j, q);

                    // ==========================================================
                    // Equation (42a): θ equation
                    // (δΘ/τ, Φ) - (UΘ, ∇Φ) + γ(∇Ψ, ∇Φ) = 0
                    // ==========================================================

                    // (1/τ)(θ, φ)  [θ-θ block]
                    local_theta_theta(i, j) += inv_tau * phi_i * phi_j * JxW;

                    // -(Uθ, ∇φ) → explicit, linearized: -θ(U·∇φ)  [θ-θ block]
                    local_theta_theta(i, j) -= phi_j * (u * grad_phi_i) * JxW;

                    // +γ(∇ψ, ∇φ)  [θ-ψ block]
                    local_theta_psi(i, j) += gamma * (grad_phi_j * grad_phi_i) * JxW;

                    // ==========================================================
                    // Equation (42b): ψ equation (chemical potential)
                    // (Ψ, Λ) - ε(∇Θ, ∇Λ) - (1/ε)(f(Θ), Λ) = 0
                    //
                    // With Newton linearization:
                    // (ψ, χ) - ε(∇θ, ∇χ) - (1/ε)(f'(θ_old)θ, χ) = -(1/ε)(f(θ_old) - f'(θ_old)θ_old, χ)
                    // ==========================================================

                    // (ψ, χ)  [ψ-ψ block]
                    local_psi_psi(i, j) += phi_i * phi_j * JxW;

                    // -ε(∇θ, ∇χ)  [ψ-θ block]
                    local_psi_theta(i, j) -= epsilon * (grad_phi_j * grad_phi_i) * JxW;

                    // -(1/ε)(f'(θ_old)θ, χ)  [ψ-θ block, linearized potential]
                    local_psi_theta(i, j) -= (1.0 / epsilon) * fp_old * phi_j * phi_i * JxW;
                }

                // ==============================================================
                // RHS contributions
                // ==============================================================

                // Eq (42a) RHS: (1/τ)(θ_old, φ)
                local_rhs_theta(i) += inv_tau * theta_old * phi_i * JxW;

                // Eq (42b) RHS: +(1/ε)(f(θ_old) - f'(θ_old)θ_old, χ)
                // Note: f - f'θ = (θ³-θ) - (3θ²-1)θ = θ³ - θ - 3θ³ + θ = -2θ³
                // So RHS = -(1/ε)(-2θ³) = (2/ε)θ³
                local_rhs_psi(i) += (1.0 / epsilon) * (f_old - fp_old * theta_old) * phi_i * JxW;
            }
        }

        // =====================================================================
        // Distribute to global system using INDEX MAPS
        // =====================================================================
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const auto gi_theta = problem_.theta_to_ch_map_[theta_dofs[i]];
            const auto gi_psi = problem_.psi_to_ch_map_[psi_dofs[i]];

            problem_.ch_rhs_(gi_theta) += local_rhs_theta(i);
            problem_.ch_rhs_(gi_psi) += local_rhs_psi(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const auto gj_theta = problem_.theta_to_ch_map_[theta_dofs[j]];
                const auto gj_psi = problem_.psi_to_ch_map_[psi_dofs[j]];

                // θ-θ block
                problem_.ch_matrix_.add(gi_theta, gj_theta, local_theta_theta(i, j));

                // θ-ψ block
                problem_.ch_matrix_.add(gi_theta, gj_psi, local_theta_psi(i, j));

                // ψ-θ block
                problem_.ch_matrix_.add(gi_psi, gj_theta, local_psi_theta(i, j));

                // ψ-ψ block
                problem_.ch_matrix_.add(gi_psi, gj_psi, local_psi_psi(i, j));
            }
        }
    }
}

template class CHAssembler<2>;
// template class CHAssembler<3>;