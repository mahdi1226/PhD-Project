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
// Variable viscosity ν(θ), Kelvin force μ₀ B_h^m(V, H, M), gravity ρ(θ)g,
// capillary force θ∇ψ, and S₂ stabilization.
//
// Full DG skew form for Kelvin force (Nochetto Eq. 38, Zhang Eq. 3.22):
//   Cell: μ₀[(M·∇)H · V + ½(∇·M)(H·V)]
//   Face: −μ₀(V·n⁻)[[H]]·{M}
//
// LHS: (1/τ + S₂)(U^n, V) + ν(θ)(D(U^n), D(V)) + B_h(U^{n-1}; U^n, V)
//       − (P, ∇·V) = 0
//      (∇·U^n, Q) = 0
// RHS: (1/τ + S₂)(U^{n-1}, V) + μ₀ B_h^m(V, H, M) + (ρ(θ)g, V) + (θ∇ψ, V)
//
// S₂ stabilization: Zhang Theorem 4.1 requires S₂ ≥ μ₀²C_M²/(4ν_min)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-f
//            Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) Theorem 4.1
// ============================================================================
#include "physics/material_properties.h"
#include "physics/kelvin_force.h"
#include "physics/applied_field.h"

template <int dim>
void NSSubsystem<dim>::assemble_coupled(
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::DoFHandler<dim>&               theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>&               psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>&               phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    const dealii::DoFHandler<dim>&               M_dof_handler,
    double current_time,
    double S2,
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
    std::vector<double>         theta_values(n_q_points);
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

    // Effective mass coefficient: 1/dt + S2
    // S2 stabilization per Zhang Theorem 4.1: S2 >= mu_0^2*C_M^2/(4*nu_min)
    const double mass_coeff = 1.0 / dt + S2;

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

            // Variable viscosity ν(θ)
            const double theta_q = theta_values[q];
            const double nu_q = viscosity(theta_q, params_.physics.epsilon,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // Magnetization (provided externally from DG field)
            Tensor<1, dim> M;
            M[0] = Mx_values[q];
            M[1] = My_values[q];

            // Kelvin force: full DG skew cell kernel (Eq. 38, first line)
            //   (M·∇)H · V + ½(∇·M)(H·V)
            //
            // H = ∇φ + h_a, so ∇H = Hess(φ) + ∇h_a
            // div(M) = ∂Mx/∂x + ∂My/∂y (elementwise, DG)
            const Tensor<2, dim>& hess_phi = phi_hessians[q];
            Tensor<2, dim> grad_H = hess_phi;
            {
                Tensor<2, dim> grad_h_a = compute_applied_field_gradient<dim>(
                    x_q, params_, current_time);
                grad_H += grad_h_a;
            }
            const Tensor<1, dim> M_grad_H = KelvinForce::compute_M_grad_H<dim>(M, grad_H);
            const double div_M = KelvinForce::compute_div_M<dim>(
                Mx_gradients[q], My_gradients[q]);

            // H for the ½(∇·M)(H·V) term
            Tensor<1, dim> H_total;
            {
                Tensor<1, dim> h_a_q = compute_applied_field<dim>(
                    x_q, params_, current_time);
                for (unsigned int d = 0; d < dim; ++d)
                    H_total[d] = phi_gradients[q][d] + h_a_q[d];
            }

            // Gravity body force: ρ(θ)g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, params_.physics.epsilon,
                                                   params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: θ ∇ψ
            //
            // Energy identity derivation:
            //   Our ψ = λ(ε∆θ - (1/ε)f(θ)) absorbs both λ and 1/ε.
            //   dE_CH/dt = -(ψ, θ_t), and θ_t = -u·∇θ + γ∆ψ, gives
            //   dE_CH/dt = -(θ∇ψ, u) + γ||∇ψ||².
            //   NS capillary = +θ∇ψ cancels the CH coupling term.
            //
            // Nochetto Eq 65d: (λ/ε)Θ∇Ψ with Ψ without λ or 1/ε.
            // Zhang Eq 2.6:    Φ∇W with W = λ(-ε∆Φ + f(Φ)).
            // Both equivalent. Our ψ has all factors baked in → just θ∇ψ.
            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_q * grad_psi_q;
            }

            // Non-magnetic body forces (capillary + gravity)
            Tensor<1, dim> F_non_magnetic;
            F_non_magnetic = F_capillary + F_gravity;

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                // RHS: non-magnetic body forces
                local_rhs_ux(i) += F_non_magnetic[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_non_magnetic[1] * phi_uy_i * JxW;

                // RHS: Kelvin force — full DG skew cell kernel (Eq. 38)
                //   μ₀ [(M·∇)H · V + ½(∇·M)(H·V)]
                double kelvin_ux, kelvin_uy;
                KelvinForce::cell_kernel<dim>(
                    M_grad_H, div_M, H_total,
                    phi_ux_i, phi_uy_i,
                    kelvin_ux, kelvin_uy);
                local_rhs_ux(i) += params_.physics.mu_0 * kelvin_ux * JxW;
                local_rhs_uy(i) += params_.physics.mu_0 * kelvin_uy * JxW;

                // RHS: time derivative + S2 stabilization: (1/dt + S2) * U^{n-1}
                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                // Kelvin force face contribution: −μ₀(V·n)[[H]]·{M}
                // assembled in dedicated face loop after cell loop

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // Mass + S2: (1/dt + S2)(U^n, V)
                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    // Viscous: (ν(θ) D(U), D(V)) = (ν/4)(T(U), T(V))
                    // Zhang Eq 2.6: D = ½(∇u+∇u^T), T = 2D
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
    // Required for energy identity B_h^m(H, H, M) = 0 (Lemma 3.1).
    // Zhang, He & Yang (2021) Eq. 3.22: full skew form with face terms.
    if (params_.enable_magnetic)
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
    double, const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&, double, double, bool);
template void NSSubsystem<3>::assemble_coupled(
    double, const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&, double, double, bool);


// ============================================================================
// PUBLIC: assemble_coupled_algebraic_M()
//
// Zhang, He & Yang (SIAM J. Sci. Comput. 43, 2021) scheme:
//   - Algebraic magnetization: M = chi(theta) * (grad phi + h_a)
//   - S2 stabilization: +S2(u^{n+1} - u^n, v)
//   - Kelvin force explicit: mu0*(M.grad)H on RHS
//
// LHS: (1/tau + S2)(U^n, V) + nu(theta)(D(U^n), D(V))
//       + B_h(U^{n-1}; U^n, V) - (P, div V) = 0
// RHS: (1/tau + S2)(U^{n-1}, V) + mu0*(M.grad)H * V + theta*grad_psi * V
//       + rho(theta)*g * V
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_coupled_algebraic_M(
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::DoFHandler<dim>&               theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>&               psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>&               phi_dof_handler,
    double current_time,
    double S2,
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
    std::vector<double>         theta_values(n_q_points);
    std::vector<Tensor<1,dim>>  theta_gradients(n_q_points);
    std::vector<double>         psi_values(n_q_points);
    std::vector<Tensor<1,dim>>  psi_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  phi_gradients(n_q_points);
    std::vector<Tensor<2,dim>>  phi_hessians(n_q_points);

    // Effective mass coefficient: 1/dt + S2
    const double mass_coeff = 1.0 / dt + S2;

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
        theta_fe_values.get_function_gradients(theta_relevant, theta_gradients);
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

            // Variable viscosity nu(theta)
            const double theta_q = theta_values[q];
            const double nu_q = viscosity(theta_q, eps,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // ============================================================
            // Algebraic magnetization: M = chi(theta) * H_total
            // ============================================================
            const double chi_q = susceptibility(theta_q, eps, chi_0);

            // H_total = grad phi + h_a
            const Tensor<1, dim>& grad_phi = phi_gradients[q];
            Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);
            Tensor<1, dim> H_total = grad_phi + h_a;

            // M = chi(theta) * H_total (algebraic, no PDE)
            Tensor<1, dim> M = chi_q * H_total;

            // Kelvin force: mu0 * (M . grad)H
            // H = grad phi + h_a, so grad H = Hess(phi) + grad(h_a)
            // For spatially varying h_a (dipoles), grad(h_a) != 0.
            const Tensor<2, dim>& hess_phi = phi_hessians[q];
            Tensor<2, dim> grad_H = hess_phi;
            {
                Tensor<2, dim> grad_h_a = compute_applied_field_gradient<dim>(
                    x_q, params_, current_time);
                grad_H += grad_h_a;
            }
            Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Gravity body force: rho(theta) * g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, eps, params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: θ ∇ψ
            //
            // Our ψ = λ(ε∆θ - (1/ε)f(θ)) absorbs λ and 1/ε factors.
            // Energy identity: dE_CH/dt = -(θ∇ψ, u) + γ||∇ψ||²,
            // so NS capillary = +θ∇ψ for energy cancellation.
            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_q * grad_psi_q;
            }

            // Total RHS force
            Tensor<1, dim> F_source;
            F_source = mu_0 * kelvin + F_capillary + F_gravity;


            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                // RHS: body forces
                local_rhs_ux(i) += F_source[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_source[1] * phi_uy_i * JxW;

                // RHS: time derivative + S2 stabilization: (1/dt + S2) * U^{n-1}
                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // Mass + S2: (1/dt + S2)(U^n, V)
                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    // Viscous: (ν(θ) D(U), D(V)) = (ν/4)(T(U), T(V))
                    // Zhang Eq 2.6: D = ½(∇u+∇u^T), T = 2D
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
    double, const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    double, double, bool);
template void NSSubsystem<3>::assemble_coupled_algebraic_M(
    double, const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    double, double, bool);