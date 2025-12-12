// ============================================================================
// assembly/ns_assembler.cc - Navier-Stokes System Assembler
//
// FIXED VERSION: Uses index maps for coupled system assembly
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42e-f, p.505
// ============================================================================

#include "ns_assembler.h"
#include "output/logger.h"
#include "physics/material_properties.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>

template <int dim>
NSAssembler<dim>::NSAssembler(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      NSAssembler constructed");
}

template <int dim>
void NSAssembler<dim>::assemble(double dt, double current_time)
{
    (void)current_time;

    const double inv_tau = 1.0 / dt;
    const double nu_w    = problem_.params_.ns.nu_water;
    const double nu_f    = problem_.params_.ns.nu_ferro;
    const double lambda  = problem_.params_.ch.lambda;
    const double epsilon = problem_.params_.ch.epsilon;

    // Clear matrix and RHS
    problem_.ns_matrix_ = 0;
    problem_.ns_rhs_    = 0;

    // Setup FE values
    const dealii::QGauss<dim> quadrature(problem_.fe_Q2_.degree + 1);

    dealii::FEValues<dim> fe_Q2(problem_.fe_Q2_, quadrature,
                                 dealii::update_values |
                                 dealii::update_gradients |
                                 dealii::update_quadrature_points |
                                 dealii::update_JxW_values);

    dealii::FEValues<dim> fe_Q1(problem_.fe_Q1_, quadrature,
                                 dealii::update_values);

    const unsigned int dofs_per_cell_Q2 = problem_.fe_Q2_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = problem_.fe_Q1_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();

    // Local matrices for 3x3 block structure
    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_uy_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_p_ux(dofs_per_cell_Q1, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_p_uy(dofs_per_cell_Q1, dofs_per_cell_Q2);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_p(dofs_per_cell_Q1);

    // DoF indices
    std::vector<dealii::types::global_dof_index> ux_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_dofs(dofs_per_cell_Q1);

    // Field values at quadrature points
    std::vector<double> theta_vals(n_q_points);
    std::vector<double> ux_old_vals(n_q_points), uy_old_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_ux_old(n_q_points), grad_uy_old(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_vals(n_q_points);

    // Iterate over cells
    auto ux_cell = problem_.ux_dof_handler_.begin_active();
    auto uy_cell = problem_.uy_dof_handler_.begin_active();
    auto p_cell = problem_.p_dof_handler_.begin_active();
    auto theta_cell = problem_.theta_dof_handler_.begin_active();
    auto psi_cell = problem_.psi_dof_handler_.begin_active();

    for (; ux_cell != problem_.ux_dof_handler_.end();
         ++ux_cell, ++uy_cell, ++p_cell, ++theta_cell, ++psi_cell)
    {
        fe_Q2.reinit(ux_cell);
        fe_Q1.reinit(p_cell);

        // Reset local matrices
        local_ux_ux = 0; local_uy_uy = 0;
        local_ux_p = 0; local_uy_p = 0;
        local_p_ux = 0; local_p_uy = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        // Get DoF indices
        ux_cell->get_dof_indices(ux_dofs);
        uy_cell->get_dof_indices(uy_dofs);
        p_cell->get_dof_indices(p_dofs);

        // Get field values
        fe_Q2.get_function_values(problem_.theta_old_solution_, theta_vals);
        fe_Q2.get_function_values(problem_.ux_old_solution_, ux_old_vals);
        fe_Q2.get_function_values(problem_.uy_old_solution_, uy_old_vals);
        fe_Q2.get_function_gradients(problem_.ux_old_solution_, grad_ux_old);
        fe_Q2.get_function_gradients(problem_.uy_old_solution_, grad_uy_old);
        fe_Q2.get_function_gradients(problem_.psi_solution_, grad_psi_vals);

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_Q2.JxW(q);

            // Phase-dependent viscosity: ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
            const double H_theta = MaterialProperties::heaviside(theta_vals[q] / epsilon);
            const double nu = nu_w + (nu_f - nu_w) * H_theta;

            const double ux_old = ux_old_vals[q];
            const double uy_old = uy_old_vals[q];

            // Convection: (u_old · ∇)u_old
            const double conv_ux = ux_old * grad_ux_old[q][0] + uy_old * grad_ux_old[q][1];
            const double conv_uy = ux_old * grad_uy_old[q][0] + uy_old * grad_uy_old[q][1];

            // Capillary force: (λ/ε) θ ∇ψ  [Eq. 42e RHS]
            const double cap_coeff = lambda / epsilon;
            const double F_cap_x = cap_coeff * theta_vals[q] * grad_psi_vals[q][0];
            const double F_cap_y = cap_coeff * theta_vals[q] * grad_psi_vals[q][1];

            // Gravity force (optional)
            double F_grav_x = 0, F_grav_y = 0;
            if (problem_.params_.gravity.enabled)
            {
                const double r = problem_.params_.ns.r;
                const double g = problem_.params_.gravity.magnitude;
                F_grav_y = -(1.0 + r * H_theta) * g;
            }

            // Assemble local matrices
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_i = fe_Q2.shape_value(i, q);
                const auto grad_phi_i = fe_Q2.shape_grad(i, q);

                // RHS: (1/τ) u_old - convection + forces
                local_rhs_ux(i) += (inv_tau * ux_old * phi_i
                                   - conv_ux * phi_i
                                   + F_cap_x * phi_i
                                   + F_grav_x * phi_i) * JxW;

                local_rhs_uy(i) += (inv_tau * uy_old * phi_i
                                   - conv_uy * phi_i
                                   + F_cap_y * phi_i
                                   + F_grav_y * phi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_j = fe_Q2.shape_value(j, q);
                    const auto grad_phi_j = fe_Q2.shape_grad(j, q);

                    // Mass: (1/τ)(u, v)
                    const double mass = inv_tau * phi_i * phi_j;

                    // Viscous: ν(∇u, ∇v)
                    const double viscous = nu * (grad_phi_i * grad_phi_j);

                    local_ux_ux(i, j) += (mass + viscous) * JxW;
                    local_uy_uy(i, j) += (mass + viscous) * JxW;
                }

                // Pressure-velocity coupling
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double psi_j = fe_Q1.shape_value(j, q);

                    // B^T: -(p, div v)
                    local_ux_p(i, j) -= grad_phi_i[0] * psi_j * JxW;
                    local_uy_p(i, j) -= grad_phi_i[1] * psi_j * JxW;
                }
            }

            // Continuity equation: (div u, q) = 0
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const double psi_i = fe_Q1.shape_value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const auto grad_phi_j = fe_Q2.shape_grad(j, q);

                    // B: (q, div u)
                    local_p_ux(i, j) -= grad_phi_j[0] * psi_i * JxW;
                    local_p_uy(i, j) -= grad_phi_j[1] * psi_i * JxW;
                }
            }
        }

        // =====================================================================
        // Distribute to global system using INDEX MAPS
        // =====================================================================
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto gi_ux = problem_.ux_to_ns_map_[ux_dofs[i]];
            const auto gi_uy = problem_.uy_to_ns_map_[uy_dofs[i]];

            problem_.ns_rhs_(gi_ux) += local_rhs_ux(i);
            problem_.ns_rhs_(gi_uy) += local_rhs_uy(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
            {
                const auto gj_ux = problem_.ux_to_ns_map_[ux_dofs[j]];
                const auto gj_uy = problem_.uy_to_ns_map_[uy_dofs[j]];

                problem_.ns_matrix_.add(gi_ux, gj_ux, local_ux_ux(i, j));
                problem_.ns_matrix_.add(gi_uy, gj_uy, local_uy_uy(i, j));
            }

            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
            {
                const auto gj_p = problem_.p_to_ns_map_[p_dofs[j]];

                problem_.ns_matrix_.add(gi_ux, gj_p, local_ux_p(i, j));
                problem_.ns_matrix_.add(gi_uy, gj_p, local_uy_p(i, j));
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
        {
            const auto gi_p = problem_.p_to_ns_map_[p_dofs[i]];

            problem_.ns_rhs_(gi_p) += local_rhs_p(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
            {
                const auto gj_ux = problem_.ux_to_ns_map_[ux_dofs[j]];
                const auto gj_uy = problem_.uy_to_ns_map_[uy_dofs[j]];

                problem_.ns_matrix_.add(gi_p, gj_ux, local_p_ux(i, j));
                problem_.ns_matrix_.add(gi_p, gj_uy, local_p_uy(i, j));
            }
        }
    }
}

template class NSAssembler<2>;
// template class NSAssembler<3>;