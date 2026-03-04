// ============================================================================
// navier_stokes/navier_stokes_assemble.cc — Core NS Assembly
//
// Implements:
//   assemble_stokes() — Core NS for standalone testing
//
// Internal:
//   assemble_ns_core() — Time derivative, viscous, convection, pressure
//
// LHS: (1/τ)(U^n, V) + ν(D(U^n), D(V)) + B_h(U^{n-1}; U^n, V) − (P, ∇·V)
//      (∇·U^n, Q) = 0
// RHS: (1/τ)(U^{n-1}, V) + (f, V)
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021), Eq 2.6
//            Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-f
// ============================================================================

#include "navier_stokes.h"

#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>

// ============================================================================
// Helper: T(V) for test function V = (φ_ux, 0)
// T(U) = ∇U + (∇U)^T  (= 2D(U), where D = ½(∇U + ∇U^T) is the symmetric gradient)
//
// NOTE: The viscous bilinear form is (ν D(U), D(V)) = (ν/4)(T(U), T(V))
// since T = 2D and (2D):(2D) = 4(D:D).  See Zhang Eq 2.6, Nochetto Eq 14e.
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
// LHS: (1/τ)(U^n, V) + ν(D(U^n), D(V)) + B_h(U^{n-1}; U^n, V) − (P, ∇·V)
//      (∇·U^n, Q) = 0
// RHS: (1/τ)(U^{n-1}, V) + (f, V)
//
// D(U) = ½(∇U + ∇U^T).  Zhang Eq 2.6, Nochetto Eq 14e/42e.
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

                    // Viscous: (ν D(U), D(V)) = (ν/4)(T(U), T(V))
                    // since T = 2D, so T:T = 4(D:D)
                    // Zhang Eq 2.6, Nochetto Eq 14e (T = D = ½(∇u+∇u^T))
                    local_ux_ux(i, j) += (nu / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu / 4.0) * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += (nu / 4.0) * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += (nu / 4.0) * (T_U_x * T_V_y) * JxW;


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
// PUBLIC: assemble_coupled()
//
// Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021), Eq 3.11:
//
// Variable viscosity ν(Φ^n), Kelvin force (three terms), gravity ρ(Φ^n)g,
// capillary Φ^{n-1}∇W^n, b_stab stabilization.
//
// M and φ are passed from the previous time step (m^{n-1}, h̃^{n-1}).
//
// LHS: (1/δt)(ũ^n, v) + ν(Φ^n)(D(ũ^n), D(v)) + b(u^{n-1}; ũ^n, v)
//       + b_stab(m^{n-1}, ũ^n, v) − (p, ∇·v)
// RHS: (1/δt)(u^{n-1}, v) + μ₀((m^{n-1}·∇)h̃^{n-1}, v)
//       + μ₀/2(m^{n-1}×h̃^{n-1}, ∇×v) + μ₀(m^{n-1}×∇×h̃^{n-1}, v)
//       + Φ^{n-1}∇W^n·v + ρ(Φ^n)g·v
// ============================================================================
#include "physics/material_properties.h"
#include "physics/kelvin_force.h"
#include "physics/applied_field.h"

template <int dim>
void NSSubsystem<dim>::assemble_coupled(
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const dealii::DoFHandler<dim>&               theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>&               psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>&               phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    const dealii::DoFHandler<dim>&               M_dof_handler,
    double current_time,
    bool include_convection)
{
    using namespace dealii;

    last_assembled_dt_ = dt;
    last_assembled_viscosity_ = 0.5 * (params_.physics.nu_water + params_.physics.nu_ferro);

    ns_matrix_ = 0;
    ns_rhs_    = 0;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const auto& fe_p   = p_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p   = fe_p.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_values | update_gradients);
    FEValues<dim> p_fe_values(fe_p, quadrature, update_values);

    // Cross-subsystem FE values
    FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> phi_fe_values(phi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients | update_hessians);
    FEValues<dim> M_fe_values(M_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);

    // Local matrices
    FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_ux_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_ux_p(dofs_per_cell_vel, dofs_per_cell_p);
    FullMatrix<double> local_uy_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_p(dofs_per_cell_vel, dofs_per_cell_p);
    FullMatrix<double> local_p_ux(dofs_per_cell_p, dofs_per_cell_vel);
    FullMatrix<double> local_p_uy(dofs_per_cell_p, dofs_per_cell_vel);

    Vector<double> local_rhs_ux(dofs_per_cell_vel);
    Vector<double> local_rhs_uy(dofs_per_cell_vel);
    Vector<double> local_rhs_p(dofs_per_cell_p);

    std::vector<types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> p_local_dofs(dofs_per_cell_p);
    std::vector<types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> coupled_p_dofs(dofs_per_cell_p);

    // Old velocity values at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<Tensor<1, dim>> uy_old_gradients(n_q_points);

    // Cross-subsystem values at quadrature points
    // Zhang Eq 3.11: nu(Phi^n), rho(Phi^n) use CURRENT theta
    //                capillary theta factor uses OLD Phi^{n-1}
    std::vector<double>         theta_values(n_q_points);      // theta^n (current)
    std::vector<double>         theta_old_values(n_q_points);  // theta^{n-1} (old, for capillary)
    std::vector<Tensor<1,dim>>  theta_gradients(n_q_points);
    std::vector<double>         psi_values(n_q_points);
    std::vector<Tensor<1,dim>>  psi_gradients(n_q_points);
    std::vector<double>         phi_values(n_q_points);
    std::vector<Tensor<1,dim>>  phi_gradients(n_q_points);
    std::vector<Tensor<2,dim>>  phi_hessians(n_q_points);
    std::vector<double>         Mx_values(n_q_points);
    std::vector<double>         My_values(n_q_points);
    std::vector<Tensor<1,dim>>  Mx_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  My_gradients(n_q_points);

    // Gravity vector
    Tensor<1, dim> gravity;
    if (params_.enable_gravity)
    {
        for (unsigned int d = 0; d < dim; ++d)
            gravity[d] = params_.physics.gravity_magnitude
                       * params_.physics.gravity_direction[d];
    }

    // Zhang Eq 3.11: NO density on time derivative
    // Kelvin force uses m^{n-1} and h̃^{n-1} (from previous step, not yet updated)

    // Iterate over all cells (synchronized across all DoFHandlers)
    auto ux_cell    = ux_dof_handler_.begin_active();
    auto uy_cell    = uy_dof_handler_.begin_active();
    auto p_cell     = p_dof_handler_.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell   = psi_dof_handler.begin_active();
    auto phi_cell   = phi_dof_handler.begin_active();
    auto M_cell     = M_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler_.end();
         ++ux_cell, ++uy_cell, ++p_cell,
         ++theta_cell, ++psi_cell, ++phi_cell, ++M_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);
        phi_fe_values.reinit(phi_cell);
        M_fe_values.reinit(M_cell);

        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux  = 0; local_p_uy  = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Extract values from all fields at quadrature points
        ux_fe_values.get_function_values(ux_old_relevant_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_relevant_, uy_old_values);

        if (include_convection)
        {
            ux_fe_values.get_function_gradients(ux_old_relevant_, ux_old_gradients);
            uy_fe_values.get_function_gradients(uy_old_relevant_, uy_old_gradients);
        }

        theta_fe_values.get_function_values(theta_relevant, theta_values);
        theta_fe_values.get_function_values(theta_old_relevant, theta_old_values);
        theta_fe_values.get_function_gradients(theta_relevant, theta_gradients);
        psi_fe_values.get_function_values(psi_relevant, psi_values);
        psi_fe_values.get_function_gradients(psi_relevant, psi_gradients);
        phi_fe_values.get_function_values(phi_relevant, phi_values);
        phi_fe_values.get_function_gradients(phi_relevant, phi_gradients);
        phi_fe_values.get_function_hessians(phi_relevant, phi_hessians);
        M_fe_values.get_function_values(Mx_relevant, Mx_values);
        M_fe_values.get_function_values(My_relevant, My_values);
        M_fe_values.get_function_gradients(Mx_relevant, Mx_gradients);
        M_fe_values.get_function_gradients(My_relevant, My_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Old velocity
            Tensor<1, dim> U_old;
            U_old[0] = ux_old_values[q];
            U_old[1] = uy_old_values[q];

            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            // Variable viscosity ν(θ^n) — Zhang Eq 3.11: use CURRENT theta
            const double theta_q = theta_values[q];
            const double theta_old_q = theta_old_values[q];
            const double nu_q = viscosity(theta_q, params_.physics.epsilon,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // Magnetization m^{n-1} (provided externally from DG field)
            Tensor<1, dim> M;
            M[0] = Mx_values[q];
            M[1] = My_values[q];

            // M gradients for b_stab (Zhang Eq 3.11)
            const Tensor<1, dim>& grad_Mx = Mx_gradients[q];
            const Tensor<1, dim>& grad_My = My_gradients[q];

            // h̃ = ∇φ — Zhang Eq 3.11 Kelvin force uses h̃^{n-1} = ∇φ^{n-1}
            // CRITICAL: Do NOT add h_a here! The Poisson equation
            //   (∇φ, ∇χ) = (h_a - m, ∇χ)
            // means φ already encodes the applied field effect.
            // Adding h_a would double-count the applied field.
            const Tensor<2, dim>& hess_phi = phi_hessians[q];
            const Tensor<2, dim>& grad_H = hess_phi;  // ∇h̃ = Hess(φ)

            // h̃ vector = ∇φ at this quadrature point
            Tensor<1, dim> H_vec;
            H_vec[0] = phi_gradients[q][0];
            H_vec[1] = phi_gradients[q][1];

            // grad(h_a) needed only for ∇×h̃ term (curl)
            // Since ∇×(∇φ) = 0 within CG elements, ∇×h̃ = 0
            Tensor<2, dim> grad_h_a;  // zero — not needed

            // Kelvin RHS term 1: μ₀(m·∇)H — Zhang Eq 3.11
            const Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Kelvin RHS term 3: μ₀(m × ∇×h̃, v) — Zhang Eq 3.11
            // h̃ = ∇φ, and ∇×(∇φ) = 0 within CG elements.
            // So this term is identically zero.
            Tensor<1, dim> M_cross_curlH;  // zero
            M_cross_curlH[0] = 0.0;
            M_cross_curlH[1] = 0.0;

            // Kelvin RHS term 2: μ₀/2(m × H̃, ∇×v) — assembled per test function
            // In 2D: m × H̃ is scalar: m_x*H_y - m_y*H_x
            const double M_cross_H = M[0] * H_vec[1] - M[1] * H_vec[0];

            const double mu0 = params_.physics.mu_0;

            // Gravity body force: ρ(θ^n)g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, params_.physics.epsilon,
                                                   params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: Φ^{n-1} ∇W^n — Zhang Eq 3.11
            // Zhang has +(Φ^{n-1}∇W^n, v) on the LHS.
            // Moved to RHS: -(Φ∇W, v) = -(θ·(-∇ψ), v) = +(θ∇ψ, v).
            // So F_capillary = +θ_old·∇ψ on the RHS.
            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_old_q * grad_psi_q;
            }

            // Mass coefficient: 1/dt  (Zhang Eq 3.11: NO density on time derivative)
            const double mass_coeff = 1.0 / dt;

            // b_stab precompute: quantities depending on m^{n-1} at this quad point
            // (v·∇)m: for test V=(φ_i, 0): V·∇m_x = φ_i * ∂m_x/∂x, V·∇m_y = φ_i * ∂m_y/∂x
            //         for test V=(0, φ_i): V·∇m_x = φ_i * ∂m_x/∂y, V·∇m_y = φ_i * ∂m_y/∂y
            // div(m) = ∂m_x/∂x + ∂m_y/∂y
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                // RHS: capillary + gravity
                local_rhs_ux(i) += (F_capillary[0] + F_gravity[0]) * phi_ux_i * JxW;
                local_rhs_uy(i) += (F_capillary[1] + F_gravity[1]) * phi_uy_i * JxW;

                // RHS: MMS source (only active when set via set_mms_source)
                if (mms_source_)
                {
                    const auto F_mms = mms_source_(x_q, current_time);
                    local_rhs_ux(i) += F_mms[0] * phi_ux_i * JxW;
                    local_rhs_uy(i) += F_mms[1] * phi_uy_i * JxW;
                }

                // RHS: Kelvin term 1 — μ₀((m·∇)H̃, v)
                local_rhs_ux(i) += mu0 * kelvin[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu0 * kelvin[1] * phi_uy_i * JxW;

                // RHS: Kelvin term 2 — μ₀/2(m × H̃, ∇×v)
                // In 2D: ∇×v for V=(φ_ux, 0) is -∂φ_ux/∂y, for V=(0,φ_uy) is ∂φ_uy/∂x
                // Combined: ∇×V = ∂φ_uy/∂x - ∂φ_ux/∂y
                // But we assemble ux and uy separately:
                //   V=(φ_i, 0): curl_v = -grad_phi_ux_i[1]
                //   V=(0, φ_i): curl_v = grad_phi_uy_i[0]
                if (!params_.disable_kelvin_term2)
                {
                local_rhs_ux(i) += 0.5 * mu0 * M_cross_H * (-grad_phi_ux_i[1]) * JxW;
                local_rhs_uy(i) += 0.5 * mu0 * M_cross_H * ( grad_phi_uy_i[0]) * JxW;
                }

                // RHS: Kelvin term 3 — μ₀(m × ∇×H̃, v)
                local_rhs_ux(i) += mu0 * M_cross_curlH[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu0 * M_cross_curlH[1] * phi_uy_i * JxW;

                // RHS: (1/dt) * U^{n-1}
                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                // b_stab precompute for test function V_i
                // For V_i = (φ_ux_i, 0):
                //   (V_i·∇)m = (φ_ux_i * ∂m_x/∂x, φ_ux_i * ∂m_y/∂x)  (vector)
                //   (∇·V_i)m = (∂φ_ux_i/∂x * m_x, ∂φ_ux_i/∂x * m_y)  (vector)
                //   m×(∇×V_i): ∇×V_i = -∂φ_ux_i/∂y (scalar), m×ω = (m_y*ω, -m_x*ω)
                const double bstab_Vgrad_m_ux_x_i = phi_ux_i * grad_Mx[0];  // (V_i·∇)m_x
                const double bstab_Vgrad_m_ux_y_i = phi_ux_i * grad_My[0];  // (V_i·∇)m_y
                const double bstab_divV_m_ux_x_i  = grad_phi_ux_i[0] * M[0]; // (∇·V_i)*m_x
                const double bstab_divV_m_ux_y_i  = grad_phi_ux_i[0] * M[1]; // (∇·V_i)*m_y
                const double curl_V_ux_i = -grad_phi_ux_i[1];
                const double bstab_mcurl_ux_x_i =  M[1] * curl_V_ux_i; // (m×∇×V_i)_x
                const double bstab_mcurl_ux_y_i = -M[0] * curl_V_ux_i; // (m×∇×V_i)_y

                // For V_i = (0, φ_uy_i):
                const double bstab_Vgrad_m_uy_x_i = phi_uy_i * grad_Mx[1];
                const double bstab_Vgrad_m_uy_y_i = phi_uy_i * grad_My[1];
                const double bstab_divV_m_uy_x_i  = grad_phi_uy_i[1] * M[0];
                const double bstab_divV_m_uy_y_i  = grad_phi_uy_i[1] * M[1];
                const double curl_V_uy_i = grad_phi_uy_i[0];
                const double bstab_mcurl_uy_x_i =  M[1] * curl_V_uy_i;
                const double bstab_mcurl_uy_y_i = -M[0] * curl_V_uy_i;

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // LHS mass: (1/dt)(U^n, V)
                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    // Viscous: (ν(θ) D(U), D(V)) = (ν/4)(T(U), T(V))
                    local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_y) * JxW;

                    // Convection: B_h(U^{n-1}; U^n, V) skew-symmetric
                    if (include_convection)
                    {
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }

                    // b_stab(m^{n-1}; U^n, V) — Zhang Eq 3.11 stabilization
                    // Three terms, all on LHS with coefficient μ₀*δt:
                    //
                    // Term 1: μ₀δt ((ũ·∇)m, (v·∇)m)
                    // Term 2: 2μ₀δt ((∇·ũ)m, (∇·v)m)
                    // Term 3: μ₀/2 δt (m×∇×ũ, m×∇×v)
                    //
                    if (!params_.disable_bstab)
                    {
                    // For ũ_j = (φ_ux_j, 0) tested against V_i = (φ_ux_i, 0):
                    {
                        // Term 1: (ũ_j·∇)m · (V_i·∇)m
                        const double Ugrad_m_x_j = phi_ux_j * grad_Mx[0];
                        const double Ugrad_m_y_j = phi_ux_j * grad_My[0];
                        const double t1 = Ugrad_m_x_j * bstab_Vgrad_m_ux_x_i
                                        + Ugrad_m_y_j * bstab_Vgrad_m_ux_y_i;

                        // Term 2: 2((∇·ũ_j)m) · ((∇·V_i)m)
                        const double divU_m_x_j = grad_phi_ux_j[0] * M[0];
                        const double divU_m_y_j = grad_phi_ux_j[0] * M[1];
                        const double t2 = 2.0 * (divU_m_x_j * bstab_divV_m_ux_x_i
                                                + divU_m_y_j * bstab_divV_m_ux_y_i);

                        // Term 3: 0.5*(m×∇×ũ_j) · (m×∇×V_i)
                        const double curl_U_j = -grad_phi_ux_j[1];
                        const double mcurl_x_j =  M[1] * curl_U_j;
                        const double mcurl_y_j = -M[0] * curl_U_j;
                        const double t3 = 0.5 * (mcurl_x_j * bstab_mcurl_ux_x_i
                                                + mcurl_y_j * bstab_mcurl_ux_y_i);

                        local_ux_ux(i, j) += mu0 * dt * (t1 + t2 + t3) * JxW;
                    }

                    // For ũ_j = (0, φ_uy_j) tested against V_i = (φ_ux_i, 0):
                    {
                        const double Ugrad_m_x_j = phi_uy_j * grad_Mx[1];
                        const double Ugrad_m_y_j = phi_uy_j * grad_My[1];
                        const double t1 = Ugrad_m_x_j * bstab_Vgrad_m_ux_x_i
                                        + Ugrad_m_y_j * bstab_Vgrad_m_ux_y_i;

                        const double divU_m_x_j = grad_phi_uy_j[1] * M[0];
                        const double divU_m_y_j = grad_phi_uy_j[1] * M[1];
                        const double t2 = 2.0 * (divU_m_x_j * bstab_divV_m_ux_x_i
                                                + divU_m_y_j * bstab_divV_m_ux_y_i);

                        const double curl_U_j = grad_phi_uy_j[0];
                        const double mcurl_x_j =  M[1] * curl_U_j;
                        const double mcurl_y_j = -M[0] * curl_U_j;
                        const double t3 = 0.5 * (mcurl_x_j * bstab_mcurl_ux_x_i
                                                + mcurl_y_j * bstab_mcurl_ux_y_i);

                        local_ux_uy(i, j) += mu0 * dt * (t1 + t2 + t3) * JxW;
                    }

                    // For ũ_j = (φ_ux_j, 0) tested against V_i = (0, φ_uy_i):
                    {
                        const double Ugrad_m_x_j = phi_ux_j * grad_Mx[0];
                        const double Ugrad_m_y_j = phi_ux_j * grad_My[0];
                        const double t1 = Ugrad_m_x_j * bstab_Vgrad_m_uy_x_i
                                        + Ugrad_m_y_j * bstab_Vgrad_m_uy_y_i;

                        const double divU_m_x_j = grad_phi_ux_j[0] * M[0];
                        const double divU_m_y_j = grad_phi_ux_j[0] * M[1];
                        const double t2 = 2.0 * (divU_m_x_j * bstab_divV_m_uy_x_i
                                                + divU_m_y_j * bstab_divV_m_uy_y_i);

                        const double curl_U_j = -grad_phi_ux_j[1];
                        const double mcurl_x_j =  M[1] * curl_U_j;
                        const double mcurl_y_j = -M[0] * curl_U_j;
                        const double t3 = 0.5 * (mcurl_x_j * bstab_mcurl_uy_x_i
                                                + mcurl_y_j * bstab_mcurl_uy_y_i);

                        local_uy_ux(i, j) += mu0 * dt * (t1 + t2 + t3) * JxW;
                    }

                    // For ũ_j = (0, φ_uy_j) tested against V_i = (0, φ_uy_i):
                    {
                        const double Ugrad_m_x_j = phi_uy_j * grad_Mx[1];
                        const double Ugrad_m_y_j = phi_uy_j * grad_My[1];
                        const double t1 = Ugrad_m_x_j * bstab_Vgrad_m_uy_x_i
                                        + Ugrad_m_y_j * bstab_Vgrad_m_uy_y_i;

                        const double divU_m_x_j = grad_phi_uy_j[1] * M[0];
                        const double divU_m_y_j = grad_phi_uy_j[1] * M[1];
                        const double t2 = 2.0 * (divU_m_x_j * bstab_divV_m_uy_x_i
                                                + divU_m_y_j * bstab_divV_m_uy_y_i);

                        const double curl_U_j = grad_phi_uy_j[0];
                        const double mcurl_x_j =  M[1] * curl_U_j;
                        const double mcurl_y_j = -M[0] * curl_U_j;
                        const double t3 = 0.5 * (mcurl_x_j * bstab_mcurl_uy_x_i
                                                + mcurl_y_j * bstab_mcurl_uy_y_i);

                        local_uy_uy(i, j) += mu0 * dt * (t1 + t2 + t3) * JxW;
                    }
                    } // end if (!params_.disable_bstab)
                }

                // Pressure: −(p, ∇·V)
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
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    local_p_ux(i, j) += grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) += grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }  // end quadrature loop

        // Distribute to global
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map_[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map_[uy_local_dofs[i]];
        }
        for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            coupled_p_dofs[i] = p_to_ns_map_[p_local_dofs[i]];

        ns_constraints_.distribute_local_to_global(local_ux_ux, local_rhs_ux, coupled_ux_dofs, ns_matrix_, ns_rhs_);
        ns_constraints_.distribute_local_to_global(local_ux_uy, coupled_ux_dofs, coupled_uy_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_ux_p, coupled_ux_dofs, coupled_p_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_uy_ux, coupled_uy_dofs, coupled_ux_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_uy_uy, local_rhs_uy, coupled_uy_dofs, ns_matrix_, ns_rhs_);
        ns_constraints_.distribute_local_to_global(local_uy_p, coupled_uy_dofs, coupled_p_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_p_ux, coupled_p_dofs, coupled_ux_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_p_uy, coupled_p_dofs, coupled_uy_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_rhs_p, coupled_p_dofs, ns_rhs_);
    }  // end cell loop

    // ========================================================================
    // Kelvin force FACE loop: −μ₀(V·n⁻)[[H]]·{M}  (Eq. 38, second line)
    //
    // Iterate over all interior faces.  H = ∇φ + h_a on each side.
    // For CG φ, [[∇φ]] ≠ 0 across element faces (gradient is piecewise).
    // For DG M, {M} = ½(M⁻ + M⁺) averages the discontinuous magnetization.
    //
    // This face term is REQUIRED for the energy identity
    //   B_h^m(H, H, M) = 0   (Lemma 3.1, Nochetto et al. 2016)
    // Without it, spurious forces break the directional balance.
    // ========================================================================
    // Kelvin force FACE loop: −μ₀(V·n⁻)[[H]]·{M}  (Eq. 38, second line)
    // DISABLED: Zhang uses simple cell-only form μ₀(M·∇)H, not the full DG
    // skew form with face terms. The face loop implementation was incomplete
    // (missing neighbor cell test function contributions) and caused instability.
    // To implement the full Nochetto DG skew form correctly, both cell-side
    // test functions must be assembled for each face.
    if (false && params_.enable_magnetic)
    {
        QGauss<dim - 1> face_quadrature(fe_vel.degree + 2);
        const unsigned int n_face_q = face_quadrature.size();

        FEFaceValues<dim> ux_face_values(fe_vel, face_quadrature,
            update_values | update_JxW_values | update_normal_vectors);
        FEFaceValues<dim> uy_face_values(fe_vel, face_quadrature,
            update_values);

        FEFaceValues<dim> phi_face_values_here(phi_dof_handler.get_fe(), face_quadrature,
            update_gradients | update_quadrature_points);
        FEFaceValues<dim> phi_face_values_there(phi_dof_handler.get_fe(), face_quadrature,
            update_gradients);

        FEFaceValues<dim> M_face_values_here(M_dof_handler.get_fe(), face_quadrature,
            update_values);
        FEFaceValues<dim> M_face_values_there(M_dof_handler.get_fe(), face_quadrature,
            update_values);

        // Scratch arrays for face quadrature
        std::vector<Tensor<1, dim>> phi_grad_here(n_face_q);
        std::vector<Tensor<1, dim>> phi_grad_there(n_face_q);
        std::vector<double>         Mx_here(n_face_q), My_here(n_face_q);
        std::vector<double>         Mx_there(n_face_q), My_there(n_face_q);

        Vector<double> face_rhs_ux(dofs_per_cell_vel);
        Vector<double> face_rhs_uy(dofs_per_cell_vel);

        // Synchronized cell iteration for face loop
        auto ux_cell_f  = ux_dof_handler_.begin_active();
        auto uy_cell_f  = uy_dof_handler_.begin_active();
        auto phi_cell_f = phi_dof_handler.begin_active();
        auto M_cell_f   = M_dof_handler.begin_active();

        for (; ux_cell_f != ux_dof_handler_.end();
             ++ux_cell_f, ++uy_cell_f, ++phi_cell_f, ++M_cell_f)
        {
            if (!ux_cell_f->is_locally_owned())
                continue;

            for (unsigned int f = 0; f < ux_cell_f->n_faces(); ++f)
            {
                if (ux_cell_f->at_boundary(f))
                    continue;

                auto ux_neighbor  = ux_cell_f->neighbor(f);
                auto phi_neighbor = phi_cell_f->neighbor(f);
                auto M_neighbor   = M_cell_f->neighbor(f);

                // Skip coarser neighbors (handled from fine side)
                if (ux_cell_f->neighbor_is_coarser(f))
                    continue;

                // Process each face once: lower index cell handles it
                if (ux_neighbor->is_active() &&
                    ux_cell_f->index() > ux_neighbor->index())
                    continue;

                const unsigned int nf = ux_cell_f->neighbor_of_neighbor(f);

                // Reinit face FEValues
                ux_face_values.reinit(ux_cell_f, f);
                uy_face_values.reinit(uy_cell_f, f);
                phi_face_values_here.reinit(phi_cell_f, f);
                phi_face_values_there.reinit(phi_neighbor, nf);
                M_face_values_here.reinit(M_cell_f, f);
                M_face_values_there.reinit(M_neighbor, nf);

                // Get field values at face quadrature points
                phi_face_values_here.get_function_gradients(phi_relevant, phi_grad_here);
                phi_face_values_there.get_function_gradients(phi_relevant, phi_grad_there);
                M_face_values_here.get_function_values(Mx_relevant, Mx_here);
                M_face_values_here.get_function_values(My_relevant, My_here);
                M_face_values_there.get_function_values(Mx_relevant, Mx_there);
                M_face_values_there.get_function_values(My_relevant, My_there);

                face_rhs_ux = 0;
                face_rhs_uy = 0;

                for (unsigned int q = 0; q < n_face_q; ++q)
                {
                    const double JxW_face = ux_face_values.JxW(q);
                    const Tensor<1, dim>& normal = ux_face_values.normal_vector(q);
                    const Point<dim>& x_q = phi_face_values_here.quadrature_point(q);

                    // Applied field (same on both sides — spatially uniform)
                    Tensor<1, dim> h_a = compute_applied_field<dim>(
                        x_q, params_, current_time);

                    // H = ∇φ + h_a on each side
                    Tensor<1, dim> H_here  = phi_grad_here[q] + h_a;
                    Tensor<1, dim> H_there = phi_grad_there[q] + h_a;

                    // [[H]] = H⁻ − H⁺  (minus = current cell, plus = neighbor)
                    Tensor<1, dim> jump_H;
                    Tensor<1, dim> avg_M;
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        jump_H[d] = H_here[d] - H_there[d];
                        avg_M[d]  = 0.5 * ((d == 0 ? Mx_here[q] : My_here[q])
                                         + (d == 0 ? Mx_there[q] : My_there[q]));
                    }

                    // [[H]] · {M}
                    double jump_H_dot_avg_M = 0.0;
                    for (unsigned int d = 0; d < dim; ++d)
                        jump_H_dot_avg_M += jump_H[d] * avg_M[d];

                    // Assemble: −μ₀(V·n)[[H]]·{M}
                    for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
                    {
                        const double phi_ux_i = ux_face_values.shape_value(i, q);
                        const double phi_uy_i = uy_face_values.shape_value(i, q);

                        // V = (φ_ux, 0): V·n = φ_ux * n[0]
                        // V = (0, φ_uy): V·n = φ_uy * n[1]
                        double kelvin_face_ux, kelvin_face_uy;
                        KelvinForce::face_kernel<dim>(
                            phi_ux_i, phi_uy_i, normal, jump_H, avg_M,
                            kelvin_face_ux, kelvin_face_uy);

                        face_rhs_ux(i) += params_.physics.mu_0 * kelvin_face_ux * JxW_face;
                        face_rhs_uy(i) += params_.physics.mu_0 * kelvin_face_uy * JxW_face;
                    }
                }

                // Distribute face RHS to global (current cell only — face
                // contributes to test functions on the minus side)
                ux_cell_f->get_dof_indices(ux_local_dofs);
                uy_cell_f->get_dof_indices(uy_local_dofs);

                for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
                {
                    coupled_ux_dofs[i] = ux_to_ns_map_[ux_local_dofs[i]];
                    coupled_uy_dofs[i] = uy_to_ns_map_[uy_local_dofs[i]];
                }

                ns_constraints_.distribute_local_to_global(
                    face_rhs_ux, coupled_ux_dofs, ns_rhs_);
                ns_constraints_.distribute_local_to_global(
                    face_rhs_uy, coupled_uy_dofs, ns_rhs_);
            }
        }
    }  // end Kelvin face loop

    ns_matrix_.compress(VectorOperation::add);
    ns_rhs_.compress(VectorOperation::add);
}


// ============================================================================
// Explicit instantiations
// ============================================================================
template void NSSubsystem<2>::assemble_stokes(double, double, bool, bool,
    const std::function<dealii::Tensor<1, 2>(const dealii::Point<2>&, double)>*, double);
template void NSSubsystem<3>::assemble_stokes(double, double, bool, bool,
    const std::function<dealii::Tensor<1, 3>(const dealii::Point<3>&, double)>*, double);

template void NSSubsystem<2>::assemble_coupled(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&, double, bool);
template void NSSubsystem<3>::assemble_coupled(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&, double, bool);


// ============================================================================
// PUBLIC: assemble_coupled_algebraic_M()
//
// Zhang, He & Yang (SIAM J. Sci. Comput. 43, 2021), Eq 3.11:
//   - Algebraic m^{n-1} = chi(Φ^{n-1}) * (∇φ^{n-1} + h_a)
//   - Three Kelvin RHS terms + b_stab on LHS
//   - Viscosity ν(Φ^n) uses current theta, M uses old theta + old phi
//
// LHS: (1/δt)(ũ^n, v) + ν(Φ^n)(D(ũ^n), D(v))
//       + b(u^{n-1}, ũ^n, v) + b_stab(m^{n-1}, ũ^n, v) - (p, ∇·v)
// RHS: (1/δt)(u^{n-1}, v) + μ₀((m^{n-1}·∇)h̃^{n-1}, v)
//       + μ₀/2(m^{n-1}×h̃^{n-1}, ∇×v) + μ₀(m^{n-1}×∇×h̃^{n-1}, v)
//       + Φ^{n-1}∇W^n·v + ρ(Φ^n)g·v
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_coupled_algebraic_M(
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const dealii::DoFHandler<dim>&               theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>&               psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>&               phi_dof_handler,
    double current_time,
    bool include_convection)
{
    using namespace dealii;

    last_assembled_dt_ = dt;
    last_assembled_viscosity_ = 0.5 * (params_.physics.nu_water + params_.physics.nu_ferro);

    ns_matrix_ = 0;
    ns_rhs_    = 0;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const auto& fe_p   = p_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p   = fe_p.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_values | update_gradients);
    FEValues<dim> p_fe_values(fe_p, quadrature, update_values);

    // Cross-subsystem FE values (NO M — computed algebraically)
    FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> phi_fe_values(phi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients | update_hessians);

    // Local matrices
    FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_ux_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_ux_p(dofs_per_cell_vel, dofs_per_cell_p);
    FullMatrix<double> local_uy_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_p(dofs_per_cell_vel, dofs_per_cell_p);
    FullMatrix<double> local_p_ux(dofs_per_cell_p, dofs_per_cell_vel);
    FullMatrix<double> local_p_uy(dofs_per_cell_p, dofs_per_cell_vel);

    Vector<double> local_rhs_ux(dofs_per_cell_vel);
    Vector<double> local_rhs_uy(dofs_per_cell_vel);
    Vector<double> local_rhs_p(dofs_per_cell_p);

    std::vector<types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> p_local_dofs(dofs_per_cell_p);
    std::vector<types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> coupled_p_dofs(dofs_per_cell_p);

    // Old velocity values at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<Tensor<1, dim>> uy_old_gradients(n_q_points);

    // Cross-subsystem values at quadrature points
    // Zhang Eq 3.11: ν(Φ^n), ρ(Φ^n) use CURRENT theta (theta_relevant)
    //                m^{n-1} = χ(Φ^{n-1})h̃^{n-1} uses OLD theta (theta_old_relevant)
    //                capillary: Φ^{n-1}∇W^n uses OLD theta
    std::vector<double>         theta_values(n_q_points);          // Φ^n (for ν, ρ)
    std::vector<double>         theta_old_values(n_q_points);      // Φ^{n-1} (for χ, capillary)
    std::vector<Tensor<1,dim>>  theta_gradients(n_q_points);       // ∇Φ^n (unused, kept for generality)
    std::vector<Tensor<1,dim>>  theta_old_gradients(n_q_points);   // ∇Φ^{n-1} (for ∇χ in b_stab)
    std::vector<double>         psi_values(n_q_points);
    std::vector<Tensor<1,dim>>  psi_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  phi_gradients(n_q_points);
    std::vector<Tensor<2,dim>>  phi_hessians(n_q_points);

    // Physics
    const double eps    = params_.physics.epsilon;
    const double chi_0  = params_.physics.chi_0;
    const double mu_0   = params_.physics.mu_0;

    // Gravity vector
    Tensor<1, dim> gravity;
    if (params_.enable_gravity)
    {
        for (unsigned int d = 0; d < dim; ++d)
            gravity[d] = params_.physics.gravity_magnitude
                       * params_.physics.gravity_direction[d];
    }

    // Iterate over all cells
    auto ux_cell    = ux_dof_handler_.begin_active();
    auto uy_cell    = uy_dof_handler_.begin_active();
    auto p_cell     = p_dof_handler_.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell   = psi_dof_handler.begin_active();
    auto phi_cell   = phi_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler_.end();
         ++ux_cell, ++uy_cell, ++p_cell,
         ++theta_cell, ++psi_cell, ++phi_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);
        phi_fe_values.reinit(phi_cell);

        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux  = 0; local_p_uy  = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Extract values from all fields at quadrature points
        ux_fe_values.get_function_values(ux_old_relevant_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_relevant_, uy_old_values);

        if (include_convection)
        {
            ux_fe_values.get_function_gradients(ux_old_relevant_, ux_old_gradients);
            uy_fe_values.get_function_gradients(uy_old_relevant_, uy_old_gradients);
        }

        theta_fe_values.get_function_values(theta_relevant, theta_values);
        theta_fe_values.get_function_values(theta_old_relevant, theta_old_values);
        theta_fe_values.get_function_gradients(theta_relevant, theta_gradients);
        theta_fe_values.get_function_gradients(theta_old_relevant, theta_old_gradients);
        psi_fe_values.get_function_values(psi_relevant, psi_values);
        psi_fe_values.get_function_gradients(psi_relevant, psi_gradients);
        phi_fe_values.get_function_gradients(phi_relevant, phi_gradients);
        phi_fe_values.get_function_hessians(phi_relevant, phi_hessians);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Old velocity
            Tensor<1, dim> U_old;
            U_old[0] = ux_old_values[q];
            U_old[1] = uy_old_values[q];

            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            // Variable viscosity ν(Φ^n) — Zhang Eq 3.11: use CURRENT theta
            const double theta_q = theta_values[q];
            const double theta_old_q = theta_old_values[q];
            const double nu_q = viscosity(theta_q, eps,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // ============================================================
            // Algebraic magnetization: m^{n-1} = chi(Φ^{n-1}) * h̃^{n-1}
            // Zhang Eq 3.11: Kelvin force uses m^{n-1} from PREVIOUS step.
            // theta_old = Φ^{n-1}, phi = φ^{n-1} (not yet updated).
            // ============================================================
            const double chi_q = susceptibility(theta_old_q, eps, chi_0);

            // h̃ = ∇φ — Zhang Eq 3.11: Kelvin force uses h̃^{n-1} = ∇φ^{n-1}
            // CRITICAL: Do NOT add h_a! The Poisson equation already encodes h_a
            // in φ via (∇φ, ∇χ) = (h_a - m, ∇χ).
            const Tensor<1, dim>& H_total = phi_gradients[q];  // h̃ = ∇φ

            // M = chi(theta) * h̃ (algebraic, no PDE)
            Tensor<1, dim> M = chi_q * H_total;

            // ∇h̃ = Hess(φ) — gradient of the effective field
            const Tensor<2, dim>& hess_phi = phi_hessians[q];
            const Tensor<2, dim>& grad_H = hess_phi;

            // Kelvin RHS term 1: μ₀(m·∇)h̃
            Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Kelvin RHS term 3: μ₀(m × ∇×h̃, v)
            // h̃ = ∇φ, ∇×(∇φ) = 0 → this term is zero
            Tensor<1, dim> M_cross_curlH;
            M_cross_curlH[0] = 0.0;
            M_cross_curlH[1] = 0.0;

            // Kelvin RHS term 2: μ₀/2(m × h̃, ∇×v)
            const double M_cross_H = M[0] * H_total[1] - M[1] * H_total[0];

            // Algebraic M gradient: ∇m = ∇(χ·h̃) = (∇χ)h̃ + χ(∇h̃)
            // where h̃ = ∇φ and ∇h̃ = Hess(φ)
            const double chi_prime_q = susceptibility_derivative(theta_old_q, eps, chi_0);
            const Tensor<1, dim>& grad_theta = theta_old_gradients[q];
            // M_x = chi * h̃_x, so dM_x/dx = chi'*dtheta/dx*h̃_x + chi*dh̃_x/dx
            Tensor<1, dim> grad_Mx_alg, grad_My_alg;
            grad_Mx_alg[0] = chi_prime_q * grad_theta[0] * H_total[0] + chi_q * grad_H[0][0];
            grad_Mx_alg[1] = chi_prime_q * grad_theta[1] * H_total[0] + chi_q * grad_H[0][1];
            grad_My_alg[0] = chi_prime_q * grad_theta[0] * H_total[1] + chi_q * grad_H[1][0];
            grad_My_alg[1] = chi_prime_q * grad_theta[1] * H_total[1] + chi_q * grad_H[1][1];

            // Gravity body force: ρ(θ^n)g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, eps, params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: Φ^{n-1} ∇W^n — Zhang Eq 3.11
            // Zhang has +(Φ^{n-1}∇W^n, v) on the LHS.
            // Moved to RHS: -(Φ∇W, v) = -(θ·(-∇ψ), v) = +(θ∇ψ, v).
            // So F_capillary = +θ_old·∇ψ on the RHS.
            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_old_q * grad_psi_q;
            }

            // Mass coefficient: 1/dt (Zhang Eq 3.11: NO density on time derivative)
            const double mass_coeff = 1.0 / dt;

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                // RHS: capillary + gravity
                local_rhs_ux(i) += (F_capillary[0] + F_gravity[0]) * phi_ux_i * JxW;
                local_rhs_uy(i) += (F_capillary[1] + F_gravity[1]) * phi_uy_i * JxW;

                // RHS: MMS source (only active when set via set_mms_source)
                if (mms_source_)
                {
                    const auto F_mms = mms_source_(x_q, current_time);
                    local_rhs_ux(i) += F_mms[0] * phi_ux_i * JxW;
                    local_rhs_uy(i) += F_mms[1] * phi_uy_i * JxW;
                }

                // RHS: Kelvin term 1 — μ₀((m·∇)H̃, v)
                local_rhs_ux(i) += mu_0 * kelvin[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu_0 * kelvin[1] * phi_uy_i * JxW;

                // RHS: Kelvin term 2 — μ₀/2(m × H̃, ∇×v)
                local_rhs_ux(i) += 0.5 * mu_0 * M_cross_H * (-grad_phi_ux_i[1]) * JxW;
                local_rhs_uy(i) += 0.5 * mu_0 * M_cross_H * ( grad_phi_uy_i[0]) * JxW;

                // RHS: Kelvin term 3 — μ₀(m × ∇×H̃, v)
                local_rhs_ux(i) += mu_0 * M_cross_curlH[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu_0 * M_cross_curlH[1] * phi_uy_i * JxW;

                // RHS: (1/dt) * U^{n-1}
                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                // b_stab precompute for test function V_i
                const double bstab_Vgrad_m_ux_x_i = phi_ux_i * grad_Mx_alg[0];
                const double bstab_Vgrad_m_ux_y_i = phi_ux_i * grad_My_alg[0];
                const double bstab_divV_m_ux_x_i  = grad_phi_ux_i[0] * M[0];
                const double bstab_divV_m_ux_y_i  = grad_phi_ux_i[0] * M[1];
                const double curl_V_ux_i = -grad_phi_ux_i[1];
                const double bstab_mcurl_ux_x_i =  M[1] * curl_V_ux_i;
                const double bstab_mcurl_ux_y_i = -M[0] * curl_V_ux_i;

                const double bstab_Vgrad_m_uy_x_i = phi_uy_i * grad_Mx_alg[1];
                const double bstab_Vgrad_m_uy_y_i = phi_uy_i * grad_My_alg[1];
                const double bstab_divV_m_uy_x_i  = grad_phi_uy_i[1] * M[0];
                const double bstab_divV_m_uy_y_i  = grad_phi_uy_i[1] * M[1];
                const double curl_V_uy_i = grad_phi_uy_i[0];
                const double bstab_mcurl_uy_x_i =  M[1] * curl_V_uy_i;
                const double bstab_mcurl_uy_y_i = -M[0] * curl_V_uy_i;

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // LHS mass: (1/dt)(U^n, V)
                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    // Viscous
                    local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_y) * JxW;

                    // Convection
                    if (include_convection)
                    {
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }

                    // b_stab — same 4 blocks as assemble_coupled
                    // ũ_j=(φ_ux_j,0) vs V_i=(φ_ux_i,0)
                    {
                        const double t1 = phi_ux_j*grad_Mx_alg[0]*bstab_Vgrad_m_ux_x_i
                                        + phi_ux_j*grad_My_alg[0]*bstab_Vgrad_m_ux_y_i;
                        const double t2 = 2.0*(grad_phi_ux_j[0]*M[0]*bstab_divV_m_ux_x_i
                                              +grad_phi_ux_j[0]*M[1]*bstab_divV_m_ux_y_i);
                        const double cj = -grad_phi_ux_j[1];
                        const double t3 = 0.5*(M[1]*cj*bstab_mcurl_ux_x_i
                                              -M[0]*cj*bstab_mcurl_ux_y_i);
                        local_ux_ux(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                    // ũ_j=(0,φ_uy_j) vs V_i=(φ_ux_i,0)
                    {
                        const double t1 = phi_uy_j*grad_Mx_alg[1]*bstab_Vgrad_m_ux_x_i
                                        + phi_uy_j*grad_My_alg[1]*bstab_Vgrad_m_ux_y_i;
                        const double t2 = 2.0*(grad_phi_uy_j[1]*M[0]*bstab_divV_m_ux_x_i
                                              +grad_phi_uy_j[1]*M[1]*bstab_divV_m_ux_y_i);
                        const double cj = grad_phi_uy_j[0];
                        const double t3 = 0.5*(M[1]*cj*bstab_mcurl_ux_x_i
                                              -M[0]*cj*bstab_mcurl_ux_y_i);
                        local_ux_uy(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                    // ũ_j=(φ_ux_j,0) vs V_i=(0,φ_uy_i)
                    {
                        const double t1 = phi_ux_j*grad_Mx_alg[0]*bstab_Vgrad_m_uy_x_i
                                        + phi_ux_j*grad_My_alg[0]*bstab_Vgrad_m_uy_y_i;
                        const double t2 = 2.0*(grad_phi_ux_j[0]*M[0]*bstab_divV_m_uy_x_i
                                              +grad_phi_ux_j[0]*M[1]*bstab_divV_m_uy_y_i);
                        const double cj = -grad_phi_ux_j[1];
                        const double t3 = 0.5*(M[1]*cj*bstab_mcurl_uy_x_i
                                              -M[0]*cj*bstab_mcurl_uy_y_i);
                        local_uy_ux(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                    // ũ_j=(0,φ_uy_j) vs V_i=(0,φ_uy_i)
                    {
                        const double t1 = phi_uy_j*grad_Mx_alg[1]*bstab_Vgrad_m_uy_x_i
                                        + phi_uy_j*grad_My_alg[1]*bstab_Vgrad_m_uy_y_i;
                        const double t2 = 2.0*(grad_phi_uy_j[1]*M[0]*bstab_divV_m_uy_x_i
                                              +grad_phi_uy_j[1]*M[1]*bstab_divV_m_uy_y_i);
                        const double cj = grad_phi_uy_j[0];
                        const double t3 = 0.5*(M[1]*cj*bstab_mcurl_uy_x_i
                                              -M[0]*cj*bstab_mcurl_uy_y_i);
                        local_uy_uy(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                }

                // Pressure: -(p, div V)
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // Continuity: (div U, q) = 0
            for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);
                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    local_p_ux(i, j) += grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) += grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }  // end quadrature loop

        // Distribute to global
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map_[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map_[uy_local_dofs[i]];
        }
        for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            coupled_p_dofs[i] = p_to_ns_map_[p_local_dofs[i]];

        ns_constraints_.distribute_local_to_global(local_ux_ux, local_rhs_ux, coupled_ux_dofs, ns_matrix_, ns_rhs_);
        ns_constraints_.distribute_local_to_global(local_ux_uy, coupled_ux_dofs, coupled_uy_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_ux_p, coupled_ux_dofs, coupled_p_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_uy_ux, coupled_uy_dofs, coupled_ux_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_uy_uy, local_rhs_uy, coupled_uy_dofs, ns_matrix_, ns_rhs_);
        ns_constraints_.distribute_local_to_global(local_uy_p, coupled_uy_dofs, coupled_p_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_p_ux, coupled_p_dofs, coupled_ux_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_p_uy, coupled_p_dofs, coupled_uy_dofs, ns_matrix_);
        ns_constraints_.distribute_local_to_global(local_rhs_p, coupled_p_dofs, ns_rhs_);
    }  // end cell loop

    ns_matrix_.compress(VectorOperation::add);
    ns_rhs_.compress(VectorOperation::add);
}

// Explicit instantiations for assemble_coupled_algebraic_M
template void NSSubsystem<2>::assemble_coupled_algebraic_M(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    double, bool);
template void NSSubsystem<3>::assemble_coupled_algebraic_M(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    double, bool);