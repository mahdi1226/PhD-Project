// ============================================================================
// assembly/ch_assembler.cc - Cahn-Hilliard System Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-b, p.505
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

// ============================================================================
// assemble()
//
// Block system (Eq. 42a-b):
//   [θ-θ: (1/τ)M       θ-ψ: -γK    ] [Θ^k]   [rhs_θ]
//   [ψ-θ: (1/η)M + εK  ψ-ψ: M      ] [Ψ^k] = [rhs_ψ]
//
// rhs_θ = (1/τ)(θ_old, Λ) + (U·θ_old, ∇Λ)
// rhs_ψ = (1/η)(θ_old, Υ) - (1/ε)(f(θ_old), Υ)
//
// ASSUMPTION: Convection term sign is +(U·θ_old, ∇Λ) on RHS
// BASIS: Eq. 42a has -(U^k Θ^{k-1}, ∇Λ) on LHS, moved to RHS flips sign
// QUESTION: Verify convection term sign convention matches paper
// ============================================================================
template <int dim>
void CHAssembler<dim>::assemble(double dt, double /*current_time*/)
{
    const double epsilon = problem_.params_.ch.epsilon;
    const double gamma   = problem_.params_.ch.gamma;
    const double eta     = problem_.params_.ch.eta;
    const double inv_tau = 1.0 / dt;
    const double inv_eta = 1.0 / eta;
    const double inv_eps = 1.0 / epsilon;

    const unsigned int n_theta = problem_.theta_dof_handler_.n_dofs();

    problem_.ch_matrix_ = 0;
    problem_.ch_rhs_    = 0;

    const dealii::QGauss<dim> quadrature(problem_.fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values(problem_.fe_Q2_, quadrature,
                                     dealii::update_values |
                                     dealii::update_gradients |
                                     dealii::update_JxW_values);

    const unsigned int dofs_per_cell = problem_.fe_Q2_.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    dealii::FullMatrix<double> local_matrix(2 * dofs_per_cell, 2 * dofs_per_cell);
    dealii::Vector<double>     local_rhs(2 * dofs_per_cell);

    std::vector<dealii::types::global_dof_index> theta_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> ux_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> uy_dofs(dofs_per_cell);

    std::vector<double> theta_old_vals(n_q_points);
    std::vector<double> ux_vals(n_q_points);
    std::vector<double> uy_vals(n_q_points);

    for (const auto& cell : problem_.theta_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        cell->get_dof_indices(theta_dofs);

        // Get corresponding cells for other handlers
        typename dealii::DoFHandler<dim>::active_cell_iterator
            psi_cell(&problem_.triangulation_, cell->level(), cell->index(),
                     &problem_.psi_dof_handler_);
        psi_cell->get_dof_indices(psi_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            ux_cell(&problem_.triangulation_, cell->level(), cell->index(),
                    &problem_.ux_dof_handler_);
        ux_cell->get_dof_indices(ux_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            uy_cell(&problem_.triangulation_, cell->level(), cell->index(),
                    &problem_.uy_dof_handler_);
        uy_cell->get_dof_indices(uy_dofs);

        // Evaluate old solution at quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            theta_old_vals[q] = 0;
            ux_vals[q] = 0;
            uy_vals[q] = 0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi = fe_values.shape_value(i, q);
                theta_old_vals[q] += problem_.theta_old_solution_[theta_dofs[i]] * phi;
                ux_vals[q] += problem_.ux_solution_[ux_dofs[i]] * phi;
                uy_vals[q] += problem_.uy_solution_[uy_dofs[i]] * phi;
            }
        }

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const double theta_old = theta_old_vals[q];
            const double f_old = MaterialProperties::double_well_derivative(theta_old);

            dealii::Tensor<1, dim> U;
            U[0] = ux_vals[q];
            if constexpr (dim >= 2) U[1] = uy_vals[q];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const auto grad_phi_i = fe_values.shape_grad(i, q);

                // RHS
                local_rhs(i) += (inv_tau * theta_old * phi_i
                               + theta_old * (U * grad_phi_i)) * JxW;
                local_rhs(i + dofs_per_cell) += (inv_eta * theta_old * phi_i
                                                - inv_eps * f_old * phi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values.shape_value(j, q);
                    const auto grad_phi_j = fe_values.shape_grad(j, q);

                    // θ-θ: (1/τ)M
                    local_matrix(i, j) += inv_tau * phi_i * phi_j * JxW;

                    // θ-ψ: -γK
                    local_matrix(i, j + dofs_per_cell) +=
                        -gamma * (grad_phi_i * grad_phi_j) * JxW;

                    // ψ-θ: (1/η)M + εK
                    local_matrix(i + dofs_per_cell, j) +=
                        (inv_eta * phi_i * phi_j + epsilon * (grad_phi_i * grad_phi_j)) * JxW;

                    // ψ-ψ: M
                    local_matrix(i + dofs_per_cell, j + dofs_per_cell) +=
                        phi_i * phi_j * JxW;
                }
            }
        }

        // Distribute to global
        // ASSUMPTION: Direct matrix addition without distribute_local_to_global
        // BASIS: Works for uniform mesh; AMR needs proper block constraint handling
        // QUESTION: Need combined constraint matrix for coupled θ-ψ system with AMR?
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const auto gi_theta = theta_dofs[i];
            const auto gi_psi   = n_theta + psi_dofs[i];

            problem_.ch_rhs_(gi_theta) += local_rhs(i);
            problem_.ch_rhs_(gi_psi)   += local_rhs(i + dofs_per_cell);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const auto gj_theta = theta_dofs[j];
                const auto gj_psi   = n_theta + psi_dofs[j];

                problem_.ch_matrix_.add(gi_theta, gj_theta, local_matrix(i, j));
                problem_.ch_matrix_.add(gi_theta, gj_psi, local_matrix(i, j + dofs_per_cell));
                problem_.ch_matrix_.add(gi_psi, gj_theta, local_matrix(i + dofs_per_cell, j));
                problem_.ch_matrix_.add(gi_psi, gj_psi, local_matrix(i + dofs_per_cell, j + dofs_per_cell));
            }
        }
    }
}

template class CHAssembler<2>;
//template class CHAssembler<3>;