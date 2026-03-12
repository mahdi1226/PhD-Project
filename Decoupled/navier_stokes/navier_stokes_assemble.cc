// ============================================================================
// navier_stokes/navier_stokes_assemble.cc — Projection Method Assembly
//
// Pressure-correction projection method (Zhang Algorithm 3.1, Steps 2-4).
//
// Implements:
//   assemble_stokes()               — Step 2 velocity predictor (standalone)
//   assemble_coupled()              — Step 2 velocity predictor (coupled, explicit M)
//   assemble_coupled_algebraic_M()  — Step 2 velocity predictor (coupled, algebraic M)
//   assemble_coupled_internal()     — Unified implementation for both coupled variants
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

#include "physics/skew_forms.h"
#include "physics/material_properties.h"
#include "physics/kelvin_force.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <cmath>
#include <chrono>

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
// Helper: b_stab bilinear form for scalar velocity components
//
// b_stab(U, V) = [(U·∇M)·(V·∇M)] + 2[(∇·U)(∇·V)]|M|² + ½[(∇×U)(∇×V)]|M|²
//
// For a scalar component vector at index `comp`:
//   U = (φ,0) if comp=0,  U = (0,φ) if comp=1
//   ∇·U = ∂φ/∂x_{comp}
//   ∇×U = (-1)^{1-comp} · ∂φ/∂x_{1-comp}
//
// Used for both diagonal blocks (comp_u == comp_v) and
// RHS cross-terms (comp_u ≠ comp_v) in the decoupled formulation.
// ============================================================================
template <int dim>
static inline double compute_bstab(
    const dealii::Tensor<1,dim>& grad_Mx,
    const dealii::Tensor<1,dim>& grad_My,
    double M_sq,
    unsigned int comp_u, double u_val, const dealii::Tensor<1,dim>& grad_u,
    unsigned int comp_v, double v_val, const dealii::Tensor<1,dim>& grad_v)
{
    // Term 1: advective — ((U·∇)M, (V·∇)M)
    const double t1 = (u_val * grad_Mx[comp_u]) * (v_val * grad_Mx[comp_v])
                     + (u_val * grad_My[comp_u]) * (v_val * grad_My[comp_v]);

    // Term 2: divergence — 2(∇·U)(∇·V)|M|²
    const double t2 = 2.0 * grad_u[comp_u] * grad_v[comp_v] * M_sq;

    // Term 3: curl — ½(∇×U)(∇×V)|M|²
    const double curlU = (comp_u == 0) ? -grad_u[1] : grad_u[0];
    const double curlV = (comp_v == 0) ? -grad_v[1] : grad_v[0];
    const double t3 = 0.5 * curlU * curlV * M_sq;

    return t1 + t2 + t3;
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
    const auto t_start = std::chrono::steady_clock::now();
    last_assembled_viscosity_ = nu;
    last_assembled_dt_ = include_time_derivative ? dt : -1.0;

    ux_matrix_ = 0;
    uy_matrix_ = 0;
    ux_rhs_ = 0;
    uy_rhs_ = 0;

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

    last_assemble_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
}


// ============================================================================
// PRIVATE: assemble_coupled_internal() — Unified velocity predictor
//
// Zhang Eq 3.11: variable viscosity, Kelvin force, b_stab, capillary, gravity.
//
// M source determined by algebraic_M flag:
//   false → read M, ∇M from FE vectors (Mx_relevant, My_relevant, M_dof_handler)
//   true  → compute M = χ(θ_old)·∇φ and ∇M = (∇χ)H + χ(∇H) algebraically
//
// Viscous and b_stab cross-terms (ux-uy coupling) → RHS using old velocity.
// Old pressure gradient on RHS.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_coupled_internal(
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const dealii::DoFHandler<dim>&               theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>&               psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>&               phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector* Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector* My_relevant,
    const dealii::DoFHandler<dim>*               M_dof_handler,
    double current_time,
    bool include_convection)
{
    using namespace dealii;
    const auto t_start = std::chrono::steady_clock::now();

    const bool algebraic_M = (Mx_relevant == nullptr);

    last_assembled_dt_ = dt;
    last_assembled_viscosity_ = 0.5 * (params_.physics.nu_water + params_.physics.nu_ferro);

    ux_matrix_ = 0;
    uy_matrix_ = 0;
    ux_rhs_    = 0;
    uy_rhs_    = 0;

    const auto& fe_vel = ux_dof_handler_.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_vel, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_vel, quadrature,
        update_values | update_gradients);
    FEValues<dim> p_fe_values(fe_pressure_, quadrature, update_values);

    // Cross-subsystem FE values
    FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients);
    FEValues<dim> phi_fe_values(phi_dof_handler.get_fe(), quadrature,
        update_values | update_gradients | update_hessians);

    // M FE values — only created for explicit M source
    std::unique_ptr<FEValues<dim>> M_fe_values_ptr;
    if (!algebraic_M)
    {
        M_fe_values_ptr = std::make_unique<FEValues<dim>>(
            M_dof_handler->get_fe(), quadrature,
            update_values | update_gradients);
    }

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
    std::vector<Tensor<1,dim>>  theta_old_gradients(n_q_points);
    std::vector<double>         psi_values(n_q_points);
    std::vector<Tensor<1,dim>>  psi_gradients(n_q_points);
    std::vector<Tensor<1,dim>>  phi_gradients(n_q_points);
    std::vector<Tensor<2,dim>>  phi_hessians(n_q_points);
    std::vector<double>         Mx_vals(n_q_points);
    std::vector<double>         My_vals(n_q_points);
    std::vector<Tensor<1,dim>>  Mx_grads(n_q_points);
    std::vector<Tensor<1,dim>>  My_grads(n_q_points);

    const double eps   = params_.physics.epsilon;
    const double chi_0 = params_.physics.chi_0;
    const double mu0   = params_.physics.mu_0;

    Tensor<1, dim> gravity;
    if (params_.enable_gravity)
    {
        for (unsigned int d = 0; d < dim; ++d)
            gravity[d] = params_.physics.gravity_magnitude
                       * params_.physics.gravity_direction[d];
    }

    // Cell iterators
    auto ux_cell    = ux_dof_handler_.begin_active();
    auto uy_cell    = uy_dof_handler_.begin_active();
    auto p_cell     = p_dof_handler_.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell   = psi_dof_handler.begin_active();
    auto phi_cell   = phi_dof_handler.begin_active();

    // M cell iterator — only for explicit M
    typename DoFHandler<dim>::active_cell_iterator M_cell;
    if (!algebraic_M)
        M_cell = M_dof_handler->begin_active();

    for (; ux_cell != ux_dof_handler_.end();
         ++ux_cell, ++uy_cell, ++p_cell,
         ++theta_cell, ++psi_cell, ++phi_cell)
    {
        if (!ux_cell->is_locally_owned())
        {
            if (!algebraic_M) ++M_cell;
            continue;
        }

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);
        phi_fe_values.reinit(phi_cell);
        if (!algebraic_M)
            M_fe_values_ptr->reinit(M_cell);

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
        psi_fe_values.get_function_values(psi_relevant, psi_values);
        psi_fe_values.get_function_gradients(psi_relevant, psi_gradients);
        phi_fe_values.get_function_gradients(phi_relevant, phi_gradients);
        phi_fe_values.get_function_hessians(phi_relevant, phi_hessians);

        if (!algebraic_M)
        {
            // Read M from FE vectors
            M_fe_values_ptr->get_function_values(*Mx_relevant, Mx_vals);
            M_fe_values_ptr->get_function_values(*My_relevant, My_vals);
            M_fe_values_ptr->get_function_gradients(*Mx_relevant, Mx_grads);
            M_fe_values_ptr->get_function_gradients(*My_relevant, My_grads);
        }
        else
        {
            // Need θ_old gradients for algebraic ∇M
            theta_fe_values.get_function_gradients(theta_old_relevant, theta_old_gradients);
        }

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
            const double nu_q = viscosity(theta_q,
                                          params_.physics.nu_water,
                                          params_.physics.nu_ferro);

            // Viscous cross-term: T(u_old) from old velocity field
            auto T_ux_old = compute_T_test<dim, 0>(ux_old_gradients[q]);
            auto T_uy_old = compute_T_test<dim, 1>(uy_old_gradients[q]);

            // ============================================================
            // M source: either from FE vectors or computed algebraically
            // ============================================================
            Tensor<1, dim> M;
            Tensor<1, dim> grad_Mx, grad_My;

            const Tensor<1, dim>& H_vec = phi_gradients[q];
            const Tensor<2, dim>& grad_H = phi_hessians[q];

            if (!algebraic_M)
            {
                M[0] = Mx_vals[q];
                M[1] = My_vals[q];
                grad_Mx = Mx_grads[q];
                grad_My = My_grads[q];
            }
            else
            {
                // m = χ(θ_old)·∇φ
                const double chi_q = susceptibility(theta_old_q, chi_0);
                M = chi_q * H_vec;

                // ∇m = (∇χ)·h̃ + χ·(∇h̃)
                const double chi_prime = susceptibility_derivative(theta_old_q, chi_0);
                const Tensor<1, dim>& gt = theta_old_gradients[q];
                grad_Mx[0] = chi_prime * gt[0] * H_vec[0] + chi_q * grad_H[0][0];
                grad_Mx[1] = chi_prime * gt[1] * H_vec[0] + chi_q * grad_H[0][1];
                grad_My[0] = chi_prime * gt[0] * H_vec[1] + chi_q * grad_H[1][0];
                grad_My[1] = chi_prime * gt[1] * H_vec[1] + chi_q * grad_H[1][1];
            }

            const double M_sq = M[0] * M[0] + M[1] * M[1];

            // Kelvin RHS term 1: μ₀(m·∇)H
            const Tensor<1, dim> kelvin = KelvinForce::compute_M_grad_H<dim>(M, grad_H);

            // Kelvin RHS term 2: μ₀/2(m × H̃, ∇×v)
            const double M_cross_H = M[0] * H_vec[1] - M[1] * H_vec[0];

            // Gravity body force: ρ(θ^n)g
            Tensor<1, dim> F_gravity;
            if (params_.enable_gravity)
            {
                const double rho_q = density_ratio(theta_q, eps, params_.physics.r);
                F_gravity = rho_q * gravity;
            }

            // Capillary force: +θ_old·∇ψ on the RHS
            const Tensor<1, dim> F_capillary = theta_old_q * psi_gradients[q];

            const double mass_coeff = 1.0 / dt;

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

                // RHS: Kelvin term 1 — μ₀((m·∇)H̃, v)
                local_rhs_ux(i) += mu0 * kelvin[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += mu0 * kelvin[1] * phi_uy_i * JxW;

                // RHS: Kelvin term 2 — μ₀/2(m × H̃, ∇×v)
                local_rhs_ux(i) += 0.5 * mu0 * M_cross_H * (-grad_phi_ux_i[1]) * JxW;
                local_rhs_uy(i) += 0.5 * mu0 * M_cross_H * ( grad_phi_uy_i[0]) * JxW;

                // RHS: Kelvin term 3 = 0 (∇×∇φ = 0 for CG)

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
                // (0, uy_old) cross on ux RHS — comp_u=1, comp_v=0
                local_rhs_ux(i) -= mu0 * dt * compute_bstab<dim>(
                    grad_Mx, grad_My, M_sq,
                    1, uy_old_values[q], uy_old_gradients[q],
                    0, phi_ux_i, grad_phi_ux_i) * JxW;
                // (ux_old, 0) cross on uy RHS — comp_u=0, comp_v=1
                local_rhs_uy(i) -= mu0 * dt * compute_bstab<dim>(
                    grad_Mx, grad_My, M_sq,
                    0, ux_old_values[q], ux_old_gradients[q],
                    1, phi_uy_i, grad_phi_uy_i) * JxW;

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

                    // LHS b_stab diagonal blocks
                    // ũ_j=(φ_ux_j,0) vs V_i=(φ_ux_i,0) — comp=0
                    local_ux_ux(i, j) += mu0 * dt * compute_bstab<dim>(
                        grad_Mx, grad_My, M_sq,
                        0, phi_ux_j, grad_phi_ux_j,
                        0, phi_ux_i, grad_phi_ux_i) * JxW;
                    // ũ_j=(0,φ_uy_j) vs V_i=(0,φ_uy_i) — comp=1
                    local_uy_uy(i, j) += mu0 * dt * compute_bstab<dim>(
                        grad_Mx, grad_My, M_sq,
                        1, phi_uy_j, grad_phi_uy_j,
                        1, phi_uy_i, grad_phi_uy_i) * JxW;
                }
            }
        }  // end quadrature loop

        // Distribute to separate matrices
        ux_constraints_.distribute_local_to_global(
            local_ux_ux, local_rhs_ux, ux_local_dofs, ux_matrix_, ux_rhs_);
        uy_constraints_.distribute_local_to_global(
            local_uy_uy, local_rhs_uy, uy_local_dofs, uy_matrix_, uy_rhs_);

        if (!algebraic_M) ++M_cell;
    }  // end cell loop

    ux_matrix_.compress(VectorOperation::add);
    uy_matrix_.compress(VectorOperation::add);
    ux_rhs_.compress(VectorOperation::add);
    uy_rhs_.compress(VectorOperation::add);

    last_assemble_time_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
}


// ============================================================================
// PUBLIC: assemble_coupled() — wrapper with explicit M from FE vectors
// ============================================================================
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
    assemble_coupled_internal(dt,
        theta_relevant, theta_old_relevant, theta_dof_handler,
        psi_relevant, psi_dof_handler,
        phi_relevant, phi_dof_handler,
        &Mx_relevant, &My_relevant, &M_dof_handler,
        current_time, include_convection);
}


// ============================================================================
// PUBLIC: assemble_coupled_algebraic_M() — wrapper with M = χ(θ)·∇φ
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
    assemble_coupled_internal(dt,
        theta_relevant, theta_old_relevant, theta_dof_handler,
        psi_relevant, psi_dof_handler,
        phi_relevant, phi_dof_handler,
        nullptr, nullptr, nullptr,
        current_time, include_convection);
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

    p_matrix_ = 0;
    p_rhs_    = 0;

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

                for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                    const Tensor<1, dim>& grad_q_j = p_fe_values.shape_grad(j, q);

                    // LHS: (∇p, ∇q)
                    local_p_p(i, j) += (grad_q_i * grad_q_j) * JxW;
                }
            }
        }  // end quadrature loop

        p_constraints_.distribute_local_to_global(
            local_p_p, local_rhs_p, p_local_dofs, p_matrix_, p_rhs_);
    }  // end cell loop

    p_matrix_.compress(VectorOperation::add);
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
            uy_constraints_.distribute(delta_uy);
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

template void NSSubsystem<2>::assemble_coupled_internal(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector*, const dealii::TrilinosWrappers::MPI::Vector*,
    const dealii::DoFHandler<2>*, double, bool);
template void NSSubsystem<3>::assemble_coupled_internal(
    double, const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector*, const dealii::TrilinosWrappers::MPI::Vector*,
    const dealii::DoFHandler<3>*, double, bool);

template void NSSubsystem<2>::assemble_pressure_poisson(double);
template void NSSubsystem<3>::assemble_pressure_poisson(double);

template void NSSubsystem<2>::velocity_correction(double);
template void NSSubsystem<3>::velocity_correction(double);
