// ============================================================================
// navier_stokes/navier_stokes_assemble.cc — Core NS Assembly
//
// Implements:
//   assemble_stokes() — Core NS for standalone testing
//
// Internal:
//   assemble_ns_core() — Time derivative, viscous, convection, pressure
//
// LHS: (1/τ)(U^n, V) + ν(T(U^n), T(V))/2 + B_h(U^{n-1}; U^n, V) − (P, ∇·V)
//      (∇·U^n, Q) = 0
// RHS: (1/τ)(U^{n-1}, V) + (f, V)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-f
// ============================================================================

#include "navier_stokes.h"

#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>

// ============================================================================
// Helper: T(V) for test function V = (φ_ux, 0)
// T(U) = ∇U + (∇U)^T  (symmetric gradient, factor 2 on diagonal)
// ============================================================================
template <int dim>
static dealii::SymmetricTensor<2, dim> compute_T_test_ux(
    const dealii::Tensor<1, dim>& grad_phi_ux)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_phi_ux[0];
    T[0][1] = grad_phi_ux[1];
    return T;
}

// Helper: T(V) for test function V = (0, φ_uy)
template <int dim>
static dealii::SymmetricTensor<2, dim> compute_T_test_uy(
    const dealii::Tensor<1, dim>& grad_phi_uy)
{
    dealii::SymmetricTensor<2, dim> T;
    T[1][1] = 2.0 * grad_phi_uy[1];
    T[0][1] = grad_phi_uy[0];
    return T;
}


// ============================================================================
// assemble_ns_core() — Core Navier-Stokes assembly
//
// LHS: (1/τ)(U^n, V) + ν(T(U^n), T(V))/2 + B_h(U^{n-1}; U^n, V) − (P, ∇·V)
//      (∇·U^n, Q) = 0
// RHS: (1/τ)(U^{n-1}, V) + (f, V)
//
// Skew-symmetric convection: B_h from skew_forms.h (Eq. 37)
// ============================================================================
template <int dim>
static void assemble_ns_core(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>* body_force,
    double body_force_time)
{
    ns_matrix = 0;
    ns_rhs = 0;

    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_p = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_p.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> p_fe_values(fe_p, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_ux_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_ux_p(dofs_per_cell_vel, dofs_per_cell_p);
    dealii::FullMatrix<double> local_uy_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_uy_p(dofs_per_cell_vel, dofs_per_cell_p);
    dealii::FullMatrix<double> local_p_ux(dofs_per_cell_p, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_p_uy(dofs_per_cell_p, dofs_per_cell_vel);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_p(dofs_per_cell_p);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> p_local_dofs(dofs_per_cell_p);

    std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> coupled_p_dofs(dofs_per_cell_p);

    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux = 0;  local_p_uy = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        ux_fe_values.get_function_values(ux_old, ux_old_values);
        uy_fe_values.get_function_values(uy_old, uy_old_values);

        if (include_convection)
        {
            ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
            uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            const double ux_old_q = ux_old_values[q];
            const double uy_old_q = uy_old_values[q];
            dealii::Tensor<1, dim> U_old;
            U_old[0] = ux_old_q;
            U_old[1] = uy_old_q;

            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            dealii::Tensor<1, dim> F_source;
            F_source = 0;
            if (body_force != nullptr)
                F_source += (*body_force)(x_q, body_force_time);

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                local_rhs_ux(i) += F_source[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_source[1] * phi_uy_i * JxW;

                if (include_time_derivative)
                {
                    local_rhs_ux(i) += (ux_old_q / dt) * phi_ux_i * JxW;
                    local_rhs_uy(i) += (uy_old_q / dt) * phi_uy_i * JxW;
                }

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    if (include_time_derivative)
                    {
                        local_ux_ux(i, j) += (1.0 / dt) * phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += (1.0 / dt) * phi_uy_j * phi_uy_i * JxW;
                    }

                    // Viscous: ν(T(U), T(V))/2
                    local_ux_ux(i, j) += (nu / 2.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu / 2.0) * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += (nu / 2.0) * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += (nu / 2.0) * (T_U_x * T_V_y) * JxW;


                    if (include_convection)
                    {
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }
                }

                // Pressure gradient: −(p, ∇·V)
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // Continuity: (∇·U, q) = 0
            for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    local_p_ux(i, j) += grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) += grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }  // end quadrature loop

        // Distribute to global (reuse pre-allocated coupled_*_dofs)
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }
        for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            coupled_p_dofs[i] = p_to_ns_map[p_local_dofs[i]];

        ns_constraints.distribute_local_to_global(local_ux_ux, local_rhs_ux, coupled_ux_dofs, ns_matrix, ns_rhs);
        ns_constraints.distribute_local_to_global(local_ux_uy, coupled_ux_dofs, coupled_uy_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_ux_p, coupled_ux_dofs, coupled_p_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_uy_ux, coupled_uy_dofs, coupled_ux_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_uy_uy, local_rhs_uy, coupled_uy_dofs, ns_matrix, ns_rhs);
        ns_constraints.distribute_local_to_global(local_uy_p, coupled_uy_dofs, coupled_p_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_p_ux, coupled_p_dofs, coupled_ux_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_p_uy, coupled_p_dofs, coupled_uy_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_rhs_p, coupled_p_dofs, ns_rhs);
    }  // end cell loop

    ns_matrix.compress(dealii::VectorOperation::add);
    ns_rhs.compress(dealii::VectorOperation::add);
}


// ============================================================================
// PUBLIC: assemble_stokes()
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_stokes(
    double dt, double nu,
    bool include_time_derivative,
    bool include_convection,
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>* body_force,
    double body_force_time)
{
    last_assembled_viscosity_ = nu;
    last_assembled_dt_ = include_time_derivative ? dt : -1.0;

    assemble_ns_core<dim>(
        ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
        ux_old_relevant_, uy_old_relevant_,
        nu, dt,
        include_time_derivative, include_convection,
        ux_to_ns_map_, uy_to_ns_map_, p_to_ns_map_,
        ns_constraints_, ns_matrix_, ns_rhs_,
        body_force, body_force_time);
}


// ============================================================================
// Explicit instantiations
// ============================================================================
template void NSSubsystem<2>::assemble_stokes(double, double, bool, bool,
    const std::function<dealii::Tensor<1, 2>(const dealii::Point<2>&, double)>*, double);
template void NSSubsystem<3>::assemble_stokes(double, double, bool, bool,
    const std::function<dealii::Tensor<1, 3>(const dealii::Point<3>&, double)>*, double);