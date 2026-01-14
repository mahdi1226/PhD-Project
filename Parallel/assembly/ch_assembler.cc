// ============================================================================
// assembly/ch_assembler.cc - Parallel Cahn-Hilliard Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
//
// Paper's discrete scheme:
//   Eq 42a: (δθ^k/τ, Λ) - (U^k θ^{k-1}, ∇Λ) - γ(∇ψ^k, ∇Λ) = 0
//   Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/ε)(f(θ^{k-1}), Υ) + (1/η)(δθ^k, Υ) = 0
//
// where η = ε (stabilization parameter)
//
// MPI-SAFE VERSION: Uses triangulation-based cell iteration to ensure
// all DoFHandlers reference the same physical cell, regardless of MPI
// partitioning or ghost layer configuration.
// ============================================================================

#include "assembly/ch_assembler.h"
#include "utilities/parameters.h"
#include "physics/material_properties.h"
#include "mms/ch/ch_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Main assembly function
// ============================================================================
template <int dim>
void assemble_ch_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old,
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    const dealii::AffineConstraints<double>& ch_constraints,
    dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    matrix = 0;
    rhs = 0;

    // Get reference to triangulation (all DoFHandlers share the same one)
    const auto& triangulation = theta_dof_handler.get_triangulation();

    const auto& fe_theta = theta_dof_handler.get_fe();
    const auto& fe_vel = ux_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe_theta.n_dofs_per_cell();

    const unsigned int quad_degree = params.fe.degree_phase + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> theta_fe_values(fe_theta, quadrature,
                                          dealii::update_values | dealii::update_gradients |
                                          dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> psi_fe_values(fe_theta, quadrature,
                                        dealii::update_values | dealii::update_gradients);

    // Use velocity FE for velocity evaluation
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature, dealii::update_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature, dealii::update_values);

    dealii::FullMatrix<double> local_matrix(2 * dofs_per_cell, 2 * dofs_per_cell);
    dealii::Vector<double> local_rhs(2 * dofs_per_cell);

    std::vector<dealii::types::global_dof_index> theta_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> coupled_dof_indices(2 * dofs_per_cell);

    std::vector<double> theta_old_values(n_q_points);
    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);

    const double eps = params.physics.epsilon;
    const double gamma = params.physics.mobility;
    const double eta = params.physics.epsilon;
    const bool have_velocity = (ux_solution.size() > 0) && (uy_solution.size() > 0);

    // For MMS, also check if velocity has actual content
    const bool use_velocity_convection = have_velocity &&
        (ux_solution.linfty_norm() > 1e-14 || uy_solution.linfty_norm() > 1e-14);


    // ========================================================================
    // MPI-SAFE ITERATION: Iterate over theta_dof_handler and construct
    // corresponding cells for other DoFHandlers using level/index.
    // This guarantees all cells refer to the same physical triangulation cell.
    // ========================================================================
    for (const auto& theta_cell : theta_dof_handler.active_cell_iterators())
    {
        // Only process locally owned cells
        if (!theta_cell->is_locally_owned())
            continue;

        // Construct corresponding cells for other DoFHandlers from same tria cell
        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &triangulation, theta_cell->level(), theta_cell->index(), &psi_dof_handler);
        const typename dealii::DoFHandler<dim>::active_cell_iterator ux_cell(
            &triangulation, theta_cell->level(), theta_cell->index(), &ux_dof_handler);
        const typename dealii::DoFHandler<dim>::active_cell_iterator uy_cell(
            &triangulation, theta_cell->level(), theta_cell->index(), &uy_dof_handler);

        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        if (have_velocity)
        {
            ux_fe_values.reinit(ux_cell);
            uy_fe_values.reinit(uy_cell);

            ux_fe_values.get_function_values(ux_solution, ux_values);
            uy_fe_values.get_function_values(uy_solution, uy_values);
        }
        else
        {
            std::fill(ux_values.begin(), ux_values.end(), 0.0);
            std::fill(uy_values.begin(), uy_values.end(), 0.0);
        }

        local_matrix = 0;
        local_rhs = 0;

        theta_fe_values.get_function_values(theta_old, theta_old_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const double theta_old_q = theta_old_values[q];

            dealii::Tensor<1, dim> U;
            U[0] = ux_values[q];
            if constexpr (dim >= 2)
                U[1] = uy_values[q];

            const double f_old = double_well_derivative(theta_old_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double Lambda_i = theta_fe_values.shape_value(i, q);
                const auto& grad_Lambda_i = theta_fe_values.shape_grad(i, q);
                const double Upsilon_i = psi_fe_values.shape_value(i, q);
                const auto& grad_Upsilon_i = psi_fe_values.shape_grad(i, q);

                // Indices in local coupled system
                const unsigned int i_theta = i;
                const unsigned int i_psi = dofs_per_cell + i;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double theta_j = theta_fe_values.shape_value(j, q);
                    const auto& grad_theta_j = theta_fe_values.shape_grad(j, q);
                    const double psi_j = psi_fe_values.shape_value(j, q);
                    const auto& grad_psi_j = psi_fe_values.shape_grad(j, q);

                    const unsigned int j_theta = j;
                    const unsigned int j_psi = dofs_per_cell + j;

                    // Eq 42a LHS: θ-θ and θ-ψ blocks
                    local_matrix(i_theta, j_theta) += (1.0 / dt) * theta_j * Lambda_i * JxW;
                    local_matrix(i_theta, j_psi) -= gamma * (grad_psi_j * grad_Lambda_i) * JxW;

                    // Eq 42b LHS: ψ-ψ and ψ-θ blocks
                    local_matrix(i_psi, j_psi) += psi_j * Upsilon_i * JxW;
                    local_matrix(i_psi, j_theta) += eps * (grad_theta_j * grad_Upsilon_i) * JxW;
                    local_matrix(i_psi, j_theta) += (1.0 / eta) * theta_j * Upsilon_i * JxW;
                }

                // Eq 42a RHS
                local_rhs(i_theta) += (1.0 / dt) * theta_old_q * Lambda_i * JxW;
                local_rhs(i_theta) += theta_old_q * (U * grad_Lambda_i) * JxW;

                // Eq 42b RHS
                local_rhs(i_psi) -= (1.0 / eps) * f_old * Upsilon_i * JxW;
                local_rhs(i_psi) += (1.0 / eta) * theta_old_q * Upsilon_i * JxW;
            }
        }

        // ====================================================================
        // MMS source terms (if enabled)
        // ====================================================================
        if (params.enable_mms)
        {
            static bool printed = false;
            if (!printed && dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
                std::cout << "[CH_ASSEMBLER] MMS ENABLED - adding source terms\n";
                printed = true;
            }
            const double L_y = params.domain.y_max - params.domain.y_min;

            // Use source with convection if velocity is provided, otherwise standalone
            if (use_velocity_convection)
            {
                CHSourceThetaWithConvection<dim> source_theta(gamma, dt, L_y);
                CHSourcePsi<dim> source_psi(eps, dt, L_y);
                source_theta.set_time(current_time);
                source_psi.set_time(current_time);

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    const double JxW = theta_fe_values.JxW(q);
                    const auto& x_q = theta_fe_values.quadrature_point(q);

                    const double f_theta = source_theta.value(x_q);
                    const double f_psi = source_psi.value(x_q);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double Lambda_i = theta_fe_values.shape_value(i, q);
                        const double Upsilon_i = psi_fe_values.shape_value(i, q);

                        local_rhs(i) += f_theta * Lambda_i * JxW;
                        local_rhs(dofs_per_cell + i) += f_psi * Upsilon_i * JxW;
                    }
                }
            }
            else
            {
                CHSourceTheta<dim> source_theta(gamma, dt, L_y);
                CHSourcePsi<dim> source_psi(eps, dt, L_y);
                source_theta.set_time(current_time);
                source_psi.set_time(current_time);

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    const double JxW = theta_fe_values.JxW(q);
                    const auto& x_q = theta_fe_values.quadrature_point(q);

                    const double f_theta = source_theta.value(x_q);
                    const double f_psi = source_psi.value(x_q);

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double Lambda_i = theta_fe_values.shape_value(i, q);
                        const double Upsilon_i = psi_fe_values.shape_value(i, q);

                        local_rhs(i) += f_theta * Lambda_i * JxW;
                        local_rhs(dofs_per_cell + i) += f_psi * Upsilon_i * JxW;
                    }
                }
            }
        }

        // Get global DoF indices from corresponding cells
        theta_cell->get_dof_indices(theta_local_dofs);
        psi_cell->get_dof_indices(psi_local_dofs);

        // Build coupled DoF indices
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            coupled_dof_indices[i] = theta_to_ch_map[theta_local_dofs[i]];
            coupled_dof_indices[dofs_per_cell + i] = psi_to_ch_map[psi_local_dofs[i]];
        }

        // Distribute to global system with constraints
        ch_constraints.distribute_local_to_global(
            local_matrix, local_rhs,
            coupled_dof_indices,
            matrix, rhs);
    }

    // Compress for MPI
    matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ch_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const Parameters&,
    double,
    double,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    dealii::TrilinosWrappers::MPI::Vector&);