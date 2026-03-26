// ============================================================================
// navier_stokes/navier_stokes_assemble.cc — Projection Method Assembly
//
// Pressure-correction projection method (Zhang Algorithm 3.1, Steps 2-4).
//
// Implements:
//   assemble_stokes()               — Step 2 velocity predictor (standalone)
//   assemble_coupled()              — Step 2 velocity predictor (coupled, DG M)
//   assemble_coupled_algebraic_M()  — Step 2 velocity predictor (coupled, algebraic M)
//   assemble_pressure_poisson()     — Step 3 pressure Poisson
//   velocity_correction()           — Step 4 algebraic velocity update
//
// Key change from monolithic saddle-point:
//   - Viscous cross-terms (ux-uy coupling) → RHS using old velocity
//   - b_stab cross-terms (ux-uy coupling) → RHS using old velocity
//   - Old pressure gradient on RHS: +(p^{n-1}, ∇·v)
//   - No pressure-velocity coupling on LHS (separate pressure Poisson)
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include "physics/applied_field.h"
#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <cmath>

// ============================================================================
// Helper: T(V) for test function V with nonzero component `comp`
// T(U) = ∇U + (∇U)^T  (= 2D(U), where D = ½(∇U + ∇U^T))
//
// comp=0 → V = (φ, 0):  T[0][0] = 2∂φ/∂x,  T[0][1] = ∂φ/∂y
// comp=1 → V = (0, φ):  T[1][1] = 2∂φ/∂y,  T[0][1] = ∂φ/∂x
//
// The viscous bilinear form is (ν D(U), D(V)) = (ν/4)(T(U), T(V))
// ============================================================================
template <int dim, unsigned int comp>
static dealii::SymmetricTensor<2, dim> compute_T_test(
    const dealii::Tensor<1, dim>& grad_phi)
{
    static_assert(comp < dim, "Component must be < dim");
    dealii::SymmetricTensor<2, dim> T;
    T[comp][comp] = 2.0 * grad_phi[comp];
    T[0][1] = grad_phi[1 - comp];
    return T;
}


// ============================================================================
// PUBLIC: assemble_stokes() — Velocity predictor for standalone testing
//
// Builds ux_matrix_ and uy_matrix_ separately (diagonal blocks only).
// Cross-terms (viscous ux-uy coupling) → RHS using old velocity.
// Old pressure gradient on RHS.
// No pressure coupling on LHS (handled by pressure Poisson step).
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

    ux_matrix_ = 0;
    uy_matrix_ = 0;
    ux_rhs_ = 0;
    uy_rhs_ = 0;
    ux_amg_valid_ = false;
    uy_amg_valid_ = false;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients);

    // Pressure FE values for old pressure gradient on RHS
    dealii::FEValues<dim> p_fe_values(fe_pressure_, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> p_old_values(n_q_points);

    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();
    auto p_cell  = p_dof_handler_.begin_active();

    for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        local_ux_ux = 0;
        local_uy_uy = 0;
        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        ux_fe_values.get_function_values(ux_old_relevant_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_relevant_, uy_old_values);

        // Always get old velocity gradients (needed for viscous cross-terms)
        ux_fe_values.get_function_gradients(ux_old_relevant_, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old_relevant_, uy_old_gradients);

        // Old pressure at quadrature points (for RHS gradient term)
        p_fe_values.get_function_values(p_old_relevant_, p_old_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            const double ux_old_q = ux_old_values[q];
            const double uy_old_q = uy_old_values[q];
            const double p_old_q  = p_old_values[q];
            dealii::Tensor<1, dim> U_old;
            U_old[0] = ux_old_q;
            U_old[1] = uy_old_q;

            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            // Viscous cross-term: T(u_old) from old velocity field
            auto T_ux_old = compute_T_test<dim, 0>(ux_old_gradients[q]);
            auto T_uy_old = compute_T_test<dim, 1>(uy_old_gradients[q]);

            dealii::Tensor<1, dim> F_source;
            F_source = 0;
            if (body_force != nullptr)
                F_source += (*body_force)(x_q, body_force_time);

            // MMS source
            if (mms_source_)
            {
                const auto F_mms = mms_source_(x_q, body_force_time);
                F_source += F_mms;
            }

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test<dim, 0>(grad_phi_ux_i);
                auto T_V_y = compute_T_test<dim, 1>(grad_phi_uy_i);

                // RHS: body force
                local_rhs_ux(i) += F_source[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_source[1] * phi_uy_i * JxW;

                // RHS: old time derivative
                if (include_time_derivative)
                {
                    local_rhs_ux(i) += (ux_old_q / dt) * phi_ux_i * JxW;
                    local_rhs_uy(i) += (uy_old_q / dt) * phi_uy_i * JxW;
                }

                // RHS: old pressure gradient +(p^{n-1}, ∇·V)
                local_rhs_ux(i) += p_old_q * grad_phi_ux_i[0] * JxW;
                local_rhs_uy(i) += p_old_q * grad_phi_uy_i[1] * JxW;

                // RHS: viscous cross-term from old velocity
                // -(ν/4)(T(uy_old), T(Vx)) on ux RHS
                // -(ν/4)(T(ux_old), T(Vy)) on uy RHS
                local_rhs_ux(i) -= (nu / 4.0) * (T_uy_old * T_V_x) * JxW;
                local_rhs_uy(i) -= (nu / 4.0) * (T_ux_old * T_V_y) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test<dim, 0>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test<dim, 1>(grad_phi_uy_j);

                    // LHS mass
                    if (include_time_derivative)
                    {
                        local_ux_ux(i, j) += (1.0 / dt) * phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += (1.0 / dt) * phi_uy_j * phi_uy_i * JxW;
                    }

                    // LHS viscous diagonal: (ν/4)(T(Ux), T(Vx)) and (ν/4)(T(Uy), T(Vy))
                    local_ux_ux(i, j) += (nu / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu / 4.0) * (T_U_y * T_V_y) * JxW;

                    // LHS convection (diagonal only — no ux-uy cross in convection)
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
            }
        }  // end quadrature loop

        // Distribute to separate matrices
        ux_constraints_.distribute_local_to_global(
            local_ux_ux, local_rhs_ux, ux_local_dofs, ux_matrix_, ux_rhs_);
        uy_constraints_.distribute_local_to_global(
            local_uy_uy, local_rhs_uy, uy_local_dofs, uy_matrix_, uy_rhs_);
    }  // end cell loop

    ux_matrix_.compress(dealii::VectorOperation::add);
    uy_matrix_.compress(dealii::VectorOperation::add);
    ux_rhs_.compress(dealii::VectorOperation::add);
    uy_rhs_.compress(dealii::VectorOperation::add);
}


// ============================================================================
// PUBLIC: assemble_coupled() — Velocity predictor with full coupling
//
// Zhang Eq 3.11: variable viscosity, Kelvin force, b_stab, capillary, gravity.
// M and φ are passed from the previous time step.
// Viscous and b_stab cross-terms (ux-uy coupling) → RHS using old velocity.
// Old pressure gradient on RHS.
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

    ux_matrix_ = 0;
    uy_matrix_ = 0;
    ux_rhs_    = 0;
    uy_rhs_    = 0;
    ux_amg_valid_ = false;
    uy_amg_valid_ = false;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_values | update_gradients);

    // Pressure FE values for old pressure gradient on RHS
    FEValues<dim> p_fe_values(fe_pressure_, quadrature, update_values);

    // Cross-subsystem FE values
    FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> phi_fe_values(phi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients | update_hessians);
    FEValues<dim> M_fe_values(M_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);

    // Local matrices — diagonal blocks only
    FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);

    Vector<double> local_rhs_ux(dofs_per_cell_vel);
    Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    // Old velocity at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> p_old_values(n_q_points);

    // Cross-subsystem values at quadrature points
    std::vector<double>         theta_values(n_q_points);
    std::vector<double>         theta_old_values(n_q_points);
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

    const double mu0 = params_.physics.mu_0;

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

        local_ux_ux = 0;
        local_uy_uy = 0;
        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        // Extract values from all fields at quadrature points
        ux_fe_values.get_function_values(ux_old_relevant_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_relevant_, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old_relevant_, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old_relevant_, uy_old_gradients);
        p_fe_values.get_function_values(p_old_relevant_, p_old_values);

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

            const double p_old_q = p_old_values[q];

            // Variable viscosity ν(θ^n)
            const double theta_q = theta_values[q];
            const double theta_old_q = theta_old_values[q];
            const double nu_q = viscosity(theta_q, params_.physics.epsilon,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // Viscous cross-term: T(u_old) from old velocity field
            auto T_ux_old = compute_T_test<dim, 0>(ux_old_gradients[q]);
            auto T_uy_old = compute_T_test<dim, 1>(uy_old_gradients[q]);

            // Magnetization m^{n-1}
            Tensor<1, dim> M;
            M[0] = Mx_values[q];
            M[1] = My_values[q];

            // M gradients for b_stab
            const Tensor<1, dim>& grad_Mx = Mx_gradients[q];
            const Tensor<1, dim>& grad_My = My_gradients[q];

            // H and ∇H for Kelvin force
            Tensor<1, dim> H_vec;
            Tensor<2, dim> grad_H;

            if (params_.use_reduced_magnetic_field)
            {
                // Reduced mode: H = h_a (no demagnetizing field)
                H_vec = compute_applied_field<dim>(x_q, params_, current_time);
                grad_H = compute_applied_field_gradient<dim>(x_q, params_, current_time);
            }
            else
            {
                // Full mode: H = ∇φ (Poisson encodes h_a via natural BCs)
                H_vec[0] = phi_gradients[q][0];
                H_vec[1] = phi_gradients[q][1];
                grad_H = phi_hessians[q];
            }

            // Kelvin force: μ₀(m·∇)H
            const Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Gravity body force: ρ(θ^n)g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, params_.physics.epsilon,
                                                   params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: +θ_old·∇ψ on the RHS
            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_old_q * grad_psi_q;
            }

            const double mass_coeff = 1.0 / dt;

            // b_stab cross-term from old velocity (RHS):
            // For trial ũ = (0, uy_old) tested against V = (φ_ux_i, 0):
            const double uy_old_q = uy_old_values[q];
            const double ux_old_q = ux_old_values[q];

            // b_stab cross-term: (0, uy_old) contribution to ux RHS
            // Term 1: ((0,uy_old)·∇)m = uy_old * (∂Mx/∂y, ∂My/∂y)
            const double bstab_cross_uy_Ugrad_mx = uy_old_q * grad_Mx[1];
            const double bstab_cross_uy_Ugrad_my = uy_old_q * grad_My[1];
            // Term 2: div(0,uy_old) = ∂uy_old/∂y
            const double bstab_cross_uy_divU = uy_old_gradients[q][1];
            // Term 3: curl(0,uy_old) = ∂uy_old/∂x
            const double bstab_cross_uy_curlU = uy_old_gradients[q][0];

            // b_stab cross-term: (ux_old, 0) contribution to uy RHS
            const double bstab_cross_ux_Ugrad_mx = ux_old_q * grad_Mx[0];
            const double bstab_cross_ux_Ugrad_my = ux_old_q * grad_My[0];
            const double bstab_cross_ux_divU = ux_old_gradients[q][0];
            const double bstab_cross_ux_curlU = -ux_old_gradients[q][1];

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test<dim, 0>(grad_phi_ux_i);
                auto T_V_y = compute_T_test<dim, 1>(grad_phi_uy_i);

                // RHS: capillary + gravity
                local_rhs_ux(i) += (F_capillary[0] + F_gravity[0]) * phi_ux_i * JxW;
                local_rhs_uy(i) += (F_capillary[1] + F_gravity[1]) * phi_uy_i * JxW;

                // RHS: MMS source
                if (mms_source_)
                {
                    const auto F_mms = mms_source_(x_q, current_time);
                    local_rhs_ux(i) += F_mms[0] * phi_ux_i * JxW;
                    local_rhs_uy(i) += F_mms[1] * phi_uy_i * JxW;
                }

                // RHS: Kelvin — μ₀((m·∇)H̃, v)  (Zhang Eq 3.11, term 1)
                local_rhs_ux(i) += mu0 * kelvin[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu0 * kelvin[1] * phi_uy_i * JxW;

                // Spin torque — (μ₀/2)(m×H̃, ∇×v)  (Zhang Eq 3.11, term 2)
                // DISABLED: Creates spurious rotational velocity in dome test.
                // At equilibrium M ∝ H so m×H ≈ 0, but numerical misalignment
                // from transport creates feedback: misalignment → torque → velocity
                // → more misalignment. Dome blew up at t=1.45 vs t=1.66 without it.

                // RHS: Consistent term — μ₀(m×∇×H̃, v)  (Zhang Eq 3.11, term 3)
                // Vanishes when H = ∇φ since ∇×∇φ = 0 identically.
                // Needed only if using L² projected h̃ (where ∇×h̃ ≠ 0).

                // RHS: (1/dt) * U^{n-1}
                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                // RHS: old pressure gradient +(p^{n-1}, ∇·V)
                local_rhs_ux(i) += p_old_q * grad_phi_ux_i[0] * JxW;
                local_rhs_uy(i) += p_old_q * grad_phi_uy_i[1] * JxW;

                // RHS: viscous cross-term from old velocity
                local_rhs_ux(i) -= (nu_q / 4.0) * (T_uy_old * T_V_x) * JxW;
                local_rhs_uy(i) -= (nu_q / 4.0) * (T_ux_old * T_V_y) * JxW;

                // RHS: b_stab cross-terms from old velocity
                {
                    // b_stab cross: (0, uy_old) tested against V_i = (φ_ux_i, 0)
                    {
                        const double bstab_Vgrad_m_ux_x_i = phi_ux_i * grad_Mx[0];
                        const double bstab_Vgrad_m_ux_y_i = phi_ux_i * grad_My[0];
                        const double bstab_divV_m_ux_x_i  = grad_phi_ux_i[0] * M[0];
                        const double bstab_divV_m_ux_y_i  = grad_phi_ux_i[0] * M[1];
                        const double curl_V_ux_i = -grad_phi_ux_i[1];
                        const double bstab_mcurl_ux_x_i =  M[1] * curl_V_ux_i;
                        const double bstab_mcurl_ux_y_i = -M[0] * curl_V_ux_i;

                        const double t1 = bstab_cross_uy_Ugrad_mx * bstab_Vgrad_m_ux_x_i
                                        + bstab_cross_uy_Ugrad_my * bstab_Vgrad_m_ux_y_i;
                        const double t2 = 2.0 * (bstab_cross_uy_divU * M[0] * bstab_divV_m_ux_x_i
                                                + bstab_cross_uy_divU * M[1] * bstab_divV_m_ux_y_i);
                        const double mcurl_x_old =  M[1] * bstab_cross_uy_curlU;
                        const double mcurl_y_old = -M[0] * bstab_cross_uy_curlU;
                        const double t3 = 0.5 * (mcurl_x_old * bstab_mcurl_ux_x_i
                                                + mcurl_y_old * bstab_mcurl_ux_y_i);

                        local_rhs_ux(i) -= mu0 * dt * (t1 + t2 + t3) * JxW;
                    }

                    // b_stab cross: (ux_old, 0) tested against V_i = (0, φ_uy_i)
                    {
                        const double bstab_Vgrad_m_uy_x_i = phi_uy_i * grad_Mx[1];
                        const double bstab_Vgrad_m_uy_y_i = phi_uy_i * grad_My[1];
                        const double bstab_divV_m_uy_x_i  = grad_phi_uy_i[1] * M[0];
                        const double bstab_divV_m_uy_y_i  = grad_phi_uy_i[1] * M[1];
                        const double curl_V_uy_i = grad_phi_uy_i[0];
                        const double bstab_mcurl_uy_x_i =  M[1] * curl_V_uy_i;
                        const double bstab_mcurl_uy_y_i = -M[0] * curl_V_uy_i;

                        const double t1 = bstab_cross_ux_Ugrad_mx * bstab_Vgrad_m_uy_x_i
                                        + bstab_cross_ux_Ugrad_my * bstab_Vgrad_m_uy_y_i;
                        const double t2 = 2.0 * (bstab_cross_ux_divU * M[0] * bstab_divV_m_uy_x_i
                                                + bstab_cross_ux_divU * M[1] * bstab_divV_m_uy_y_i);
                        const double mcurl_x_old =  M[1] * bstab_cross_ux_curlU;
                        const double mcurl_y_old = -M[0] * bstab_cross_ux_curlU;
                        const double t3 = 0.5 * (mcurl_x_old * bstab_mcurl_uy_x_i
                                                + mcurl_y_old * bstab_mcurl_uy_y_i);

                        local_rhs_uy(i) -= mu0 * dt * (t1 + t2 + t3) * JxW;
                    }
                }

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test<dim, 0>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test<dim, 1>(grad_phi_uy_j);

                    // LHS mass: (1/dt)(U^n, V)
                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    // LHS viscous diagonal
                    local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_y) * JxW;

                    // LHS convection
                    if (include_convection)
                    {
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }

                    // LHS b_stab diagonal blocks only (ux-ux and uy-uy)
                    {
                        // ũ_j=(φ_ux_j,0) vs V_i=(φ_ux_i,0)
                        {
                            const double bstab_Vgrad_m_ux_x_i = phi_ux_i * grad_Mx[0];
                            const double bstab_Vgrad_m_ux_y_i = phi_ux_i * grad_My[0];
                            const double bstab_divV_m_ux_x_i  = grad_phi_ux_i[0] * M[0];
                            const double bstab_divV_m_ux_y_i  = grad_phi_ux_i[0] * M[1];
                            const double curl_V_ux_i = -grad_phi_ux_i[1];
                            const double bstab_mcurl_ux_x_i =  M[1] * curl_V_ux_i;
                            const double bstab_mcurl_ux_y_i = -M[0] * curl_V_ux_i;

                            const double Ugrad_m_x_j = phi_ux_j * grad_Mx[0];
                            const double Ugrad_m_y_j = phi_ux_j * grad_My[0];
                            const double t1 = Ugrad_m_x_j * bstab_Vgrad_m_ux_x_i
                                            + Ugrad_m_y_j * bstab_Vgrad_m_ux_y_i;
                            const double divU_m_x_j = grad_phi_ux_j[0] * M[0];
                            const double divU_m_y_j = grad_phi_ux_j[0] * M[1];
                            const double t2 = 2.0 * (divU_m_x_j * bstab_divV_m_ux_x_i
                                                    + divU_m_y_j * bstab_divV_m_ux_y_i);
                            const double curl_U_j = -grad_phi_ux_j[1];
                            const double mcurl_x_j =  M[1] * curl_U_j;
                            const double mcurl_y_j = -M[0] * curl_U_j;
                            const double t3 = 0.5 * (mcurl_x_j * bstab_mcurl_ux_x_i
                                                    + mcurl_y_j * bstab_mcurl_ux_y_i);

                            local_ux_ux(i, j) += mu0 * dt * (t1 + t2 + t3) * JxW;
                        }

                        // ũ_j=(0,φ_uy_j) vs V_i=(0,φ_uy_i)
                        {
                            const double bstab_Vgrad_m_uy_x_i = phi_uy_i * grad_Mx[1];
                            const double bstab_Vgrad_m_uy_y_i = phi_uy_i * grad_My[1];
                            const double bstab_divV_m_uy_x_i  = grad_phi_uy_i[1] * M[0];
                            const double bstab_divV_m_uy_y_i  = grad_phi_uy_i[1] * M[1];
                            const double curl_V_uy_i = grad_phi_uy_i[0];
                            const double bstab_mcurl_uy_x_i =  M[1] * curl_V_uy_i;
                            const double bstab_mcurl_uy_y_i = -M[0] * curl_V_uy_i;

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
                    } // end b_stab
                }
            }
        }  // end quadrature loop

        // Distribute to separate matrices
        ux_constraints_.distribute_local_to_global(
            local_ux_ux, local_rhs_ux, ux_local_dofs, ux_matrix_, ux_rhs_);
        uy_constraints_.distribute_local_to_global(
            local_uy_uy, local_rhs_uy, uy_local_dofs, uy_matrix_, uy_rhs_);
    }  // end cell loop

    ux_matrix_.compress(VectorOperation::add);
    uy_matrix_.compress(VectorOperation::add);
    ux_rhs_.compress(VectorOperation::add);
    uy_rhs_.compress(VectorOperation::add);
}


// ============================================================================
// PUBLIC: assemble_coupled_algebraic_M()
//
// Same as assemble_coupled but M is computed algebraically:
//   m^{n-1} = chi(Φ^{n-1}) * h̃^{n-1} = chi(Φ^{n-1}) * ∇φ^{n-1}
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

    ux_matrix_ = 0;
    uy_matrix_ = 0;
    ux_rhs_    = 0;
    uy_rhs_    = 0;
    ux_amg_valid_ = false;
    uy_amg_valid_ = false;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_values | update_gradients);
    FEValues<dim> p_fe_values(fe_pressure_, quadrature, update_values);

    // Cross-subsystem FE values (NO M DoFHandler — computed algebraically)
    FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> phi_fe_values(phi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients | update_hessians);

    FullMatrix<double> local_ux_ux(dofs_per_cell_vel, dofs_per_cell_vel);
    FullMatrix<double> local_uy_uy(dofs_per_cell_vel, dofs_per_cell_vel);
    Vector<double> local_rhs_ux(dofs_per_cell_vel);
    Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> p_old_values(n_q_points);

    std::vector<double>         theta_values(n_q_points);
    std::vector<double>         theta_old_values(n_q_points);
    std::vector<Tensor<1,dim>>  theta_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  theta_old_gradients(n_q_points);
    std::vector<double>         psi_values(n_q_points);
    std::vector<Tensor<1,dim>>  psi_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  phi_gradients(n_q_points);
    std::vector<Tensor<2,dim>>  phi_hessians(n_q_points);

    const double eps   = params_.physics.epsilon;
    const double chi_0 = params_.physics.chi_0;
    const double mu_0  = params_.physics.mu_0;

    Tensor<1, dim> gravity;
    if (params_.enable_gravity)
    {
        for (unsigned int d = 0; d < dim; ++d)
            gravity[d] = params_.physics.gravity_magnitude
                       * params_.physics.gravity_direction[d];
    }

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

        local_ux_ux = 0;
        local_uy_uy = 0;
        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        ux_fe_values.get_function_values(ux_old_relevant_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_relevant_, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old_relevant_, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old_relevant_, uy_old_gradients);
        p_fe_values.get_function_values(p_old_relevant_, p_old_values);

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

            Tensor<1, dim> U_old;
            U_old[0] = ux_old_values[q];
            U_old[1] = uy_old_values[q];

            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            const double p_old_q = p_old_values[q];

            const double theta_q = theta_values[q];
            const double theta_old_q = theta_old_values[q];
            const double nu_q = viscosity(theta_q, eps,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            auto T_ux_old = compute_T_test<dim, 0>(ux_old_gradients[q]);
            auto T_uy_old = compute_T_test<dim, 1>(uy_old_gradients[q]);

            // Algebraic magnetization: m = chi(θ_old) * ∇φ
            const double chi_q = susceptibility(theta_old_q, eps, chi_0);
            const Tensor<1, dim>& H_total = phi_gradients[q];
            Tensor<1, dim> M = chi_q * H_total;

            const Tensor<2, dim>& hess_phi = phi_hessians[q];
            const Tensor<2, dim>& grad_H = hess_phi;

            Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Algebraic M gradient: ∇m = (∇χ)h̃ + χ(∇h̃)
            const double chi_prime_q = susceptibility_derivative(theta_old_q, eps, chi_0);
            const Tensor<1, dim>& grad_theta = theta_old_gradients[q];
            Tensor<1, dim> grad_Mx_alg, grad_My_alg;
            grad_Mx_alg[0] = chi_prime_q * grad_theta[0] * H_total[0] + chi_q * grad_H[0][0];
            grad_Mx_alg[1] = chi_prime_q * grad_theta[1] * H_total[0] + chi_q * grad_H[0][1];
            grad_My_alg[0] = chi_prime_q * grad_theta[0] * H_total[1] + chi_q * grad_H[1][0];
            grad_My_alg[1] = chi_prime_q * grad_theta[1] * H_total[1] + chi_q * grad_H[1][1];

            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, eps, params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            Tensor<1, dim> F_capillary;
            {
                const Tensor<1, dim>& grad_psi_q = psi_gradients[q];
                F_capillary = theta_old_q * grad_psi_q;
            }

            const double mass_coeff = 1.0 / dt;

            const double ux_old_q = ux_old_values[q];
            const double uy_old_q = uy_old_values[q];

            // b_stab cross from old velocity (for RHS)
            const double bstab_cross_uy_Ugrad_mx = uy_old_q * grad_Mx_alg[1];
            const double bstab_cross_uy_Ugrad_my = uy_old_q * grad_My_alg[1];
            const double bstab_cross_uy_divU = uy_old_gradients[q][1];
            const double bstab_cross_uy_curlU = uy_old_gradients[q][0];

            const double bstab_cross_ux_Ugrad_mx = ux_old_q * grad_Mx_alg[0];
            const double bstab_cross_ux_Ugrad_my = ux_old_q * grad_My_alg[0];
            const double bstab_cross_ux_divU = ux_old_gradients[q][0];
            const double bstab_cross_ux_curlU = -ux_old_gradients[q][1];

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test<dim, 0>(grad_phi_ux_i);
                auto T_V_y = compute_T_test<dim, 1>(grad_phi_uy_i);

                local_rhs_ux(i) += (F_capillary[0] + F_gravity[0]) * phi_ux_i * JxW;
                local_rhs_uy(i) += (F_capillary[1] + F_gravity[1]) * phi_uy_i * JxW;

                if (mms_source_)
                {
                    const auto F_mms = mms_source_(x_q, current_time);
                    local_rhs_ux(i) += F_mms[0] * phi_ux_i * JxW;
                    local_rhs_uy(i) += F_mms[1] * phi_uy_i * JxW;
                }

                local_rhs_ux(i) += mu_0 * kelvin[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu_0 * kelvin[1] * phi_uy_i * JxW;

                local_rhs_ux(i) += mass_coeff * ux_old_values[q] * phi_ux_i * JxW;
                local_rhs_uy(i) += mass_coeff * uy_old_values[q] * phi_uy_i * JxW;

                local_rhs_ux(i) += p_old_q * grad_phi_ux_i[0] * JxW;
                local_rhs_uy(i) += p_old_q * grad_phi_uy_i[1] * JxW;

                local_rhs_ux(i) -= (nu_q / 4.0) * (T_uy_old * T_V_x) * JxW;
                local_rhs_uy(i) -= (nu_q / 4.0) * (T_ux_old * T_V_y) * JxW;

                // b_stab cross-terms from old velocity on RHS
                {
                    // (0, uy_old) cross on ux RHS
                    {
                        const double bVgm_x = phi_ux_i * grad_Mx_alg[0];
                        const double bVgm_y = phi_ux_i * grad_My_alg[0];
                        const double bdVm_x = grad_phi_ux_i[0] * M[0];
                        const double bdVm_y = grad_phi_ux_i[0] * M[1];
                        const double cVi = -grad_phi_ux_i[1];
                        const double bmcV_x =  M[1] * cVi;
                        const double bmcV_y = -M[0] * cVi;

                        const double t1 = bstab_cross_uy_Ugrad_mx * bVgm_x
                                        + bstab_cross_uy_Ugrad_my * bVgm_y;
                        const double t2 = 2.0 * (bstab_cross_uy_divU * M[0] * bdVm_x
                                                + bstab_cross_uy_divU * M[1] * bdVm_y);
                        const double mcx =  M[1] * bstab_cross_uy_curlU;
                        const double mcy = -M[0] * bstab_cross_uy_curlU;
                        const double t3 = 0.5 * (mcx * bmcV_x + mcy * bmcV_y);
                        local_rhs_ux(i) -= mu_0 * dt * (t1 + t2 + t3) * JxW;
                    }
                    // (ux_old, 0) cross on uy RHS
                    {
                        const double bVgm_x = phi_uy_i * grad_Mx_alg[1];
                        const double bVgm_y = phi_uy_i * grad_My_alg[1];
                        const double bdVm_x = grad_phi_uy_i[1] * M[0];
                        const double bdVm_y = grad_phi_uy_i[1] * M[1];
                        const double cVi = grad_phi_uy_i[0];
                        const double bmcV_x =  M[1] * cVi;
                        const double bmcV_y = -M[0] * cVi;

                        const double t1 = bstab_cross_ux_Ugrad_mx * bVgm_x
                                        + bstab_cross_ux_Ugrad_my * bVgm_y;
                        const double t2 = 2.0 * (bstab_cross_ux_divU * M[0] * bdVm_x
                                                + bstab_cross_ux_divU * M[1] * bdVm_y);
                        const double mcx =  M[1] * bstab_cross_ux_curlU;
                        const double mcy = -M[0] * bstab_cross_ux_curlU;
                        const double t3 = 0.5 * (mcx * bmcV_x + mcy * bmcV_y);
                        local_rhs_uy(i) -= mu_0 * dt * (t1 + t2 + t3) * JxW;
                    }
                }

                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test<dim, 0>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test<dim, 1>(grad_phi_uy_j);

                    local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;

                    local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_y) * JxW;

                    if (include_convection)
                    {
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }

                    // b_stab diagonal blocks
                    // ũ_j=(φ_ux_j,0) vs V_i=(φ_ux_i,0)
                    {
                        const double bVgm_x = phi_ux_i * grad_Mx_alg[0];
                        const double bVgm_y = phi_ux_i * grad_My_alg[0];
                        const double bdVm_x = grad_phi_ux_i[0] * M[0];
                        const double bdVm_y = grad_phi_ux_i[0] * M[1];
                        const double cVi = -grad_phi_ux_i[1];
                        const double bmcV_x =  M[1] * cVi;
                        const double bmcV_y = -M[0] * cVi;

                        const double t1 = phi_ux_j*grad_Mx_alg[0]*bVgm_x
                                        + phi_ux_j*grad_My_alg[0]*bVgm_y;
                        const double t2 = 2.0*(grad_phi_ux_j[0]*M[0]*bdVm_x
                                              +grad_phi_ux_j[0]*M[1]*bdVm_y);
                        const double cj = -grad_phi_ux_j[1];
                        const double t3 = 0.5*(M[1]*cj*bmcV_x - M[0]*cj*bmcV_y);
                        local_ux_ux(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                    // ũ_j=(0,φ_uy_j) vs V_i=(0,φ_uy_i)
                    {
                        const double bVgm_x = phi_uy_i * grad_Mx_alg[1];
                        const double bVgm_y = phi_uy_i * grad_My_alg[1];
                        const double bdVm_x = grad_phi_uy_i[1] * M[0];
                        const double bdVm_y = grad_phi_uy_i[1] * M[1];
                        const double cVi = grad_phi_uy_i[0];
                        const double bmcV_x =  M[1] * cVi;
                        const double bmcV_y = -M[0] * cVi;

                        const double t1 = phi_uy_j*grad_Mx_alg[1]*bVgm_x
                                        + phi_uy_j*grad_My_alg[1]*bVgm_y;
                        const double t2 = 2.0*(grad_phi_uy_j[1]*M[0]*bdVm_x
                                              +grad_phi_uy_j[1]*M[1]*bdVm_y);
                        const double cj = grad_phi_uy_j[0];
                        const double t3 = 0.5*(M[1]*cj*bmcV_x - M[0]*cj*bmcV_y);
                        local_uy_uy(i, j) += mu_0 * dt * (t1+t2+t3) * JxW;
                    }
                }
            }
        }  // end quadrature loop

        ux_constraints_.distribute_local_to_global(
            local_ux_ux, local_rhs_ux, ux_local_dofs, ux_matrix_, ux_rhs_);
        uy_constraints_.distribute_local_to_global(
            local_uy_uy, local_rhs_uy, uy_local_dofs, uy_matrix_, uy_rhs_);
    }  // end cell loop

    ux_matrix_.compress(VectorOperation::add);
    uy_matrix_.compress(VectorOperation::add);
    ux_rhs_.compress(VectorOperation::add);
    uy_rhs_.compress(VectorOperation::add);
}


// ============================================================================
// PUBLIC: assemble_pressure_poisson() — Zhang Step 3
//
// (∇p^n, ∇q) = -(1/δt)(∇·ū^n, q) + (∇p^{n-1}, ∇q)
//
// LHS: Laplacian on CG Q1 pressure space
// RHS: divergence of velocity predictor + old pressure Laplacian
//
// Must be called AFTER solve_velocity() so ū is available.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_pressure_poisson(double dt)
{
    using namespace dealii;

    // Pressure Laplacian matrix is constant — only build once
    const bool need_matrix = !p_matrix_assembled_;
    if (need_matrix)
    {
        p_matrix_ = 0;
        p_amg_valid_ = false;
    }
    p_rhs_ = 0;

    const auto& fe_p   = p_dof_handler_.get_fe();
    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_p = fe_p.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // Pressure FE values (need gradients for Laplacian)
    FEValues<dim> p_fe_values(fe_p, quadrature,
        update_values | update_gradients | update_JxW_values);

    // Velocity FE values (need gradients for divergence of ū)
    FEValues<dim> ux_fe_values(fe_vel, quadrature, update_gradients);
    FEValues<dim> uy_fe_values(fe_vel, quadrature, update_gradients);

    // Pressure FE values for old pressure gradient
    // (same FE as p, so reuse p_fe_values for shape functions)

    FullMatrix<double> local_p_p(dofs_per_cell_p, dofs_per_cell_p);
    Vector<double> local_rhs_p(dofs_per_cell_p);

    std::vector<types::global_dof_index> p_local_dofs(dofs_per_cell_p);

    // ū gradients at quadrature points (velocity predictor solution)
    std::vector<Tensor<1, dim>> ux_bar_gradients(n_q_points);
    std::vector<Tensor<1, dim>> uy_bar_gradients(n_q_points);
    // Old pressure gradients at quadrature points
    std::vector<Tensor<1, dim>> p_old_gradients(n_q_points);

    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();
    auto p_cell  = p_dof_handler_.begin_active();

    for (; p_cell != p_dof_handler_.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!p_cell->is_locally_owned())
            continue;

        p_fe_values.reinit(p_cell);
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        if (need_matrix)
            local_p_p = 0;
        local_rhs_p = 0;

        p_cell->get_dof_indices(p_local_dofs);

        // Get ū gradients (velocity predictor is already solved)
        ux_fe_values.get_function_gradients(ux_relevant_, ux_bar_gradients);
        uy_fe_values.get_function_gradients(uy_relevant_, uy_bar_gradients);
        // Get old pressure gradients
        p_fe_values.get_function_gradients(p_old_relevant_, p_old_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = p_fe_values.JxW(q);

            // div(ū) = ∂ūx/∂x + ∂ūy/∂y
            const double div_u_bar = ux_bar_gradients[q][0] + uy_bar_gradients[q][1];

            const Tensor<1, dim>& grad_p_old = p_old_gradients[q];

            for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            {
                const Tensor<1, dim>& grad_q_i = p_fe_values.shape_grad(i, q);
                const double q_i = p_fe_values.shape_value(i, q);

                // RHS: -(1/dt)(∇·ū, q)
                local_rhs_p(i) -= (1.0 / dt) * div_u_bar * q_i * JxW;

                // RHS: +(∇p^{n-1}, ∇q)
                local_rhs_p(i) += (grad_p_old * grad_q_i) * JxW;

                if (need_matrix)
                {
                    for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                    {
                        const Tensor<1, dim>& grad_q_j = p_fe_values.shape_grad(j, q);
                        local_p_p(i, j) += (grad_q_i * grad_q_j) * JxW;
                    }
                }
            }
        }  // end quadrature loop

        if (need_matrix)
            p_constraints_.distribute_local_to_global(
                local_p_p, local_rhs_p, p_local_dofs, p_matrix_, p_rhs_);
        else
            p_constraints_.distribute_local_to_global(
                local_rhs_p, p_local_dofs, p_rhs_);
    }  // end cell loop

    if (need_matrix)
    {
        p_matrix_.compress(VectorOperation::add);
        p_matrix_assembled_ = true;
    }
    p_rhs_.compress(VectorOperation::add);
}


// ============================================================================
// PUBLIC: velocity_correction() — Zhang Step 4 (algebraic)
//
// u^n = ū^n - δt ∇(p^n - p^{n-1})
//
// Weak form with consistent mass CG solve:
//   M * δu_x = δt * ∫(p^n - p_old) * ∂φ_i/∂x dx
//   M * δu_y = δt * ∫(p^n - p_old) * ∂φ_i/∂y dx
//   u_x^n = ū_x + δu_x,  u_y^n = ū_y + δu_y
//
// Using consistent mass (CG solve) instead of lumped mass (pointwise)
// avoids the O(δt/h) numerical boundary layer in H1 norm.
// The CG solve converges in ~5 iterations (mass matrix is well-conditioned).
//
// Must be called AFTER solve_pressure().
// ============================================================================
template <int dim>
void NSSubsystem<dim>::velocity_correction(double dt)
{
    using namespace dealii;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_gradients | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_gradients);
    FEValues<dim> p_fe_values(fe_pressure_, quadrature,
        update_values);

    std::vector<types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);

    // Pressure correction: p^n - p^{n-1}
    TrilinosWrappers::MPI::Vector dp(p_locally_owned_, mpi_comm_);
    dp = p_solution_;
    dp -= p_old_solution_;

    // Need ghosted version for quadrature evaluation
    TrilinosWrappers::MPI::Vector dp_relevant(
        p_locally_owned_, p_locally_relevant_, mpi_comm_);
    dp_relevant = dp;

    std::vector<double> dp_values(n_q_points);

    // RHS vectors for the mass CG solve: rhs = dt * ∫ δp ∂φ/∂x dx
    TrilinosWrappers::MPI::Vector mass_rhs_ux(ux_locally_owned_, mpi_comm_);
    TrilinosWrappers::MPI::Vector mass_rhs_uy(uy_locally_owned_, mpi_comm_);
    mass_rhs_ux = 0;
    mass_rhs_uy = 0;

    Vector<double> local_rhs_ux(dofs_per_cell_vel);
    Vector<double> local_rhs_uy(dofs_per_cell_vel);

    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();
    auto p_cell  = p_dof_handler_.begin_active();

    for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);

        // Get pressure correction at quadrature points
        p_fe_values.get_function_values(dp_relevant, dp_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const double dp_q = dp_values[q];

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const Tensor<1, dim>& grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const Tensor<1, dim>& grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                // dt * ∫(p^n - p_old) * ∂φ_i/∂x dx
                local_rhs_ux(i) += dt * dp_q * grad_phi_ux_i[0] * JxW;
                local_rhs_uy(i) += dt * dp_q * grad_phi_uy_i[1] * JxW;
            }
        }

        // Distribute with constraints (handles hanging nodes + zeros Dirichlet DOFs)
        ux_constraints_.distribute_local_to_global(
            local_rhs_ux, ux_local_dofs, mass_rhs_ux);
        ux_constraints_.distribute_local_to_global(
            local_rhs_uy, ux_local_dofs, mass_rhs_uy);
    }

    mass_rhs_ux.compress(VectorOperation::add);
    mass_rhs_uy.compress(VectorOperation::add);

    // Solve M * δu_x = mass_rhs_ux  and  M * δu_y = mass_rhs_uy
    // Mass matrix is SPD and well-conditioned → CG with Jacobi, ~5 iterations
    TrilinosWrappers::MPI::Vector delta_ux(ux_locally_owned_, mpi_comm_);
    TrilinosWrappers::MPI::Vector delta_uy(uy_locally_owned_, mpi_comm_);
    delta_ux = 0;
    delta_uy = 0;

    {
        const double rhs_ux_norm = mass_rhs_ux.l2_norm();
        const double rhs_uy_norm = mass_rhs_uy.l2_norm();

        TrilinosWrappers::PreconditionJacobi jacobi;
        jacobi.initialize(vel_mass_matrix_);

        if (rhs_ux_norm > 1e-14)
        {
            SolverControl control(200, 1e-12 * rhs_ux_norm);
            SolverCG<TrilinosWrappers::MPI::Vector> cg(control);
            cg.solve(vel_mass_matrix_, delta_ux, mass_rhs_ux, jacobi);
            ux_constraints_.distribute(delta_ux);
        }

        if (rhs_uy_norm > 1e-14)
        {
            SolverControl control(200, 1e-12 * rhs_uy_norm);
            SolverCG<TrilinosWrappers::MPI::Vector> cg(control);
            cg.solve(vel_mass_matrix_, delta_uy, mass_rhs_uy, jacobi);
            ux_constraints_.distribute(delta_uy);
        }
    }

    // Apply correction: u^n = ū^n + δu
    ux_solution_ += delta_ux;
    uy_solution_ += delta_uy;

    // Apply velocity boundary conditions (ensures Dirichlet BCs after correction)
    ux_constraints_.distribute(ux_solution_);
    uy_constraints_.distribute(uy_solution_);

    // Update ghosted vectors
    ux_relevant_ = ux_solution_;
    uy_relevant_ = uy_solution_;
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

template void NSSubsystem<2>::assemble_pressure_poisson(double);
template void NSSubsystem<3>::assemble_pressure_poisson(double);

template void NSSubsystem<2>::velocity_correction(double);
template void NSSubsystem<3>::velocity_correction(double);
