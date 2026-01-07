// ============================================================================
// assembly/ns_mms_assembler.cc - Navier-Stokes MMS Verification Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e-42f (discrete scheme)
//
// This is a MINIMAL assembler for MMS verification of the NS equations.
// See ns_mms_assembler.h for phase descriptions.
//
// ============================================================================

#include "assembly/ns_mms_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Helper: Compute symmetric gradient T(U) = ∇U + (∇U)^T
//
// For U = (ux, uy), T(U) is a 2x2 symmetric tensor:
//   T[0][0] = 2 * ∂ux/∂x
//   T[1][1] = 2 * ∂uy/∂y
//   T[0][1] = T[1][0] = ∂ux/∂y + ∂uy/∂x
// ============================================================================
template <int dim>
dealii::SymmetricTensor<2, dim> compute_symmetric_gradient_mms(
    const dealii::Tensor<1, dim>& grad_ux,
    const dealii::Tensor<1, dim>& grad_uy)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_ux[0];           // 2 * ∂ux/∂x
    T[1][1] = 2.0 * grad_uy[1];           // 2 * ∂uy/∂y
    T[0][1] = grad_ux[1] + grad_uy[0];    // ∂ux/∂y + ∂uy/∂x
    return T;
}

// ============================================================================
// Helper: T(V) for test function V = (φ_ux, 0)
// ============================================================================
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_ux(
    const dealii::Tensor<1, dim>& grad_phi_ux)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_phi_ux[0];   // 2 * ∂φ_ux/∂x
    T[1][1] = 0.0;                     // 0
    T[0][1] = grad_phi_ux[1];          // ∂φ_ux/∂y
    return T;
}

// ============================================================================
// Helper: T(V) for test function V = (0, φ_uy)
// ============================================================================
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_uy(
    const dealii::Tensor<1, dim>& grad_phi_uy)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 0.0;                     // 0
    T[1][1] = 2.0 * grad_phi_uy[1];   // 2 * ∂φ_uy/∂y
    T[0][1] = grad_phi_uy[0];          // ∂φ_uy/∂x
    return T;
}

// ============================================================================
// MMS Exact Solution (from stream function ψ = t·sin²(πx)·sin²(πy/L_y))
//
//   ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
//   uy = -t·π·sin(2πx)·sin²(πy/L_y)
//   p  = t·cos(πx)·cos(πy/L_y)
//
// Properties:
//   - ∇·U = 0 exactly (incompressible)
//   - U = 0 on all boundaries (no-slip)
// ============================================================================

// Get exact velocity at a point
template <int dim>
dealii::Tensor<1, dim> mms_exact_velocity(
    const dealii::Point<dim>& pt,
    double time,
    double L_y)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    dealii::Tensor<1, dim> U;
    U[0] = t * (M_PI / L_y) * sin_px * sin_px * sin_2py;
    U[1] = -t * M_PI * sin_2px * sin_py * sin_py;

    return U;
}

// Get exact velocity gradients at a point
template <int dim>
void mms_exact_velocity_gradients(
    const dealii::Point<dim>& pt,
    double time,
    double L_y,
    dealii::Tensor<1, dim>& grad_ux,
    dealii::Tensor<1, dim>& grad_uy)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;

    // ∂ux/∂x = t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
    grad_ux[0] = t * (pi2 / L_y) * sin_2px * sin_2py;

    // ∂ux/∂y = t·(2π²/L_y²)·sin²(πx)·cos(2πy/L_y)
    grad_ux[1] = t * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;

    // ∂uy/∂x = -t·2π²·cos(2πx)·sin²(πy/L_y)
    grad_uy[0] = -t * 2.0 * pi2 * cos_2px * sin_py * sin_py;

    // ∂uy/∂y = -t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
    grad_uy[1] = -t * (pi2 / L_y) * sin_2px * sin_2py;
}

// ============================================================================
// MMS Source Term: Steady Stokes
//
// f = -2ν∇²U + ∇p
//
// For the weak form ν(T(U), T(V)), the strong form is -ν∇·T(U) = -2ν∇²U
// (factor of 2 because T(U) = ∇U + (∇U)^T for incompressible flow)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_steady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;
    const double pi3 = M_PI * M_PI * M_PI;

    // ========================================================================
    // Laplacian of ux
    // ========================================================================
    // ∂²ux/∂x² = t·(2π³/L_y)·cos(2πx)·sin(2πy/L_y)
    const double d2ux_dx2 = t * (2.0 * pi3 / L_y) * cos_2px * sin_2py;

    // ∂²ux/∂y² = -t·(4π³/L_y³)·sin²(πx)·sin(2πy/L_y)
    const double d2ux_dy2 = -t * (4.0 * pi3 / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;

    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;

    // ========================================================================
    // Laplacian of uy
    // ========================================================================
    // ∂²uy/∂x² = t·4π³·sin(2πx)·sin²(πy/L_y)
    const double d2uy_dx2 = t * 4.0 * pi3 * sin_2px * sin_py * sin_py;

    // ∂²uy/∂y² = -t·(2π³/L_y²)·sin(2πx)·cos(2πy/L_y)
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);
    const double d2uy_dy2 = -t * (2.0 * pi3 / (L_y * L_y)) * sin_2px * cos_2py;

    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // ========================================================================
    // Pressure gradient
    // ========================================================================
    // p = t·cos(πx)·cos(πy/L_y)
    // ∂p/∂x = -t·π·sin(πx)·cos(πy/L_y)
    const double dp_dx = -t * pi * sin_px * cos_py;

    // ∂p/∂y = -t·(π/L_y)·cos(πx)·sin(πy/L_y)
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // ========================================================================
    // Total: f = -2ν∇²U + ∇p
    // ========================================================================
    dealii::Tensor<1, dim> f;
    f[0] = -2.0 * nu * laplacian_ux + dp_dx;
    f[1] = -2.0 * nu * laplacian_uy + dp_dy;

    return f;
}

// ============================================================================
// MMS Source Term: Unsteady Stokes
//
// f = ∂U/∂t - 2ν∇²U + ∇p
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y)
{
    const double x = pt[0];
    const double y = pt[1];

    const double sin_px = std::sin(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    const double pi = M_PI;

    // ∂U/∂t (since U = t * spatial_part, ∂U/∂t = spatial_part)
    const double dux_dt = (pi / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -pi * sin_2px * sin_py * sin_py;

    // Get steady Stokes source (-2ν∇²U + ∇p)
    dealii::Tensor<1, dim> f_steady = compute_steady_stokes_mms_source<dim>(pt, time, nu, L_y);

    // Add time derivative
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + f_steady[0];
    f[1] = duy_dt + f_steady[1];

    return f;
}

// ============================================================================
// MMS Source Term: Steady NS
//
// f = (U·∇)U - 2ν∇²U + ∇p
//
// Note: Skew term ½(∇·U)U = 0 for divergence-free exact solution.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_steady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y)
{
    // Get velocity and gradients
    dealii::Tensor<1, dim> U = mms_exact_velocity<dim>(pt, time, L_y);
    dealii::Tensor<1, dim> grad_ux, grad_uy;
    mms_exact_velocity_gradients<dim>(pt, time, L_y, grad_ux, grad_uy);

    // Convection: (U·∇)U
    const double convect_x = U[0] * grad_ux[0] + U[1] * grad_ux[1];
    const double convect_y = U[0] * grad_uy[0] + U[1] * grad_uy[1];

    // Get steady Stokes source (-2ν∇²U + ∇p)
    dealii::Tensor<1, dim> f_stokes = compute_steady_stokes_mms_source<dim>(pt, time, nu, L_y);

    // Add convection
    dealii::Tensor<1, dim> f;
    f[0] = convect_x + f_stokes[0];
    f[1] = convect_y + f_stokes[1];

    return f;
}

// ============================================================================
// MMS Source Term: Unsteady NS (semi-implicit discretization)
//
// f = (U^n - U^{n-1})/τ + (U^{n-1}·∇)U^n - 2ν∇²U^n + ∇p^n
//
// This matches the discrete scheme in ns_assembler.cc.
// Note: Skew term ½(∇·U^{n-1})U^n = 0 for divergence-free exact solution.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double L_y)
{
    const double dt = t_new - t_old;

    // Velocities at old and new times
    dealii::Tensor<1, dim> U_new = mms_exact_velocity<dim>(pt, t_new, L_y);
    dealii::Tensor<1, dim> U_old = mms_exact_velocity<dim>(pt, t_old, L_y);

    // Velocity gradients at new time (what's being convected)
    dealii::Tensor<1, dim> grad_ux_new, grad_uy_new;
    mms_exact_velocity_gradients<dim>(pt, t_new, L_y, grad_ux_new, grad_uy_new);

    // Discrete time derivative: (U^n - U^{n-1})/τ
    const double dux_dt = (U_new[0] - U_old[0]) / dt;
    const double duy_dt = (U_new[1] - U_old[1]) / dt;

    // Semi-implicit convection: (U^{n-1}·∇)U^n
    const double convect_x = U_old[0] * grad_ux_new[0] + U_old[1] * grad_ux_new[1];
    const double convect_y = U_old[0] * grad_uy_new[0] + U_old[1] * grad_uy_new[1];

    // Get steady Stokes source at t_new (-2ν∇²U + ∇p)
    dealii::Tensor<1, dim> f_stokes = compute_steady_stokes_mms_source<dim>(pt, t_new, nu, L_y);

    // Total source
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + convect_x + f_stokes[0];
    f[1] = duy_dt + convect_y + f_stokes[1];

    return f;
}

// ============================================================================
// Main Assembly Function
// ============================================================================
template <int dim>
void assemble_ns_mms_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::Vector<double>& ux_old,
    const dealii::Vector<double>& uy_old,
    double nu,
    double dt,
    double current_time,
    double L_y,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::SparseMatrix<double>& ns_matrix,
    dealii::Vector<double>& ns_rhs)
{
    ns_matrix = 0;
    ns_rhs = 0;

    const auto& fe_Q2 = ux_dof_handler.get_fe();
    const auto& fe_Q1 = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell_Q2 = fe_Q2.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = fe_Q1.n_dofs_per_cell();

    // Quadrature (degree + 2 for accuracy)
    const unsigned int quad_degree = fe_Q2.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for velocity (Q2)
    dealii::FEValues<dim> ux_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients);

    // FEValues for pressure (Q1)
    dealii::FEValues<dim> p_fe_values(fe_Q1, quadrature, dealii::update_values);

    // Local matrices (9 blocks for 3×3 coupled system)
    dealii::FullMatrix<double> local_ux_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_ux_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_uy_ux(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_uy(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_uy_p(dofs_per_cell_Q2, dofs_per_cell_Q1);
    dealii::FullMatrix<double> local_p_ux(dofs_per_cell_Q1, dofs_per_cell_Q2);
    dealii::FullMatrix<double> local_p_uy(dofs_per_cell_Q1, dofs_per_cell_Q2);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_p(dofs_per_cell_Q1);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_local_dofs(dofs_per_cell_Q1);

    // Solution values at quadrature points (for convection)
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);

    // Determine which source term to use
    const bool is_steady = !include_time_derivative;
    const bool has_convection = include_convection;

    // Cell iterators
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        local_ux_ux = 0;
        local_ux_uy = 0;
        local_ux_p = 0;
        local_uy_ux = 0;
        local_uy_uy = 0;
        local_uy_p = 0;
        local_p_ux = 0;
        local_p_uy = 0;
        local_rhs_ux = 0;
        local_rhs_uy = 0;
        local_rhs_p = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get U_old values if needed for convection
        if (include_convection)
        {
            ux_fe_values.get_function_values(ux_old, ux_old_values);
            ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
            uy_fe_values.get_function_values(uy_old, uy_old_values);
            uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);
        }

        // ====================================================================
        // QUADRATURE LOOP
        // ====================================================================
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // ================================================================
            // Compute MMS source term
            // ================================================================
            dealii::Tensor<1, dim> F_mms;
            if (is_steady && !has_convection)
            {
                // Phase A: Steady Stokes
                F_mms = compute_steady_stokes_mms_source<dim>(x_q, current_time, nu, L_y);
            }
            else if (!is_steady && !has_convection)
            {
                // Phase B: Unsteady Stokes
                F_mms = compute_unsteady_stokes_mms_source<dim>(x_q, current_time, nu, L_y);
            }
            else if (is_steady && has_convection)
            {
                // Phase C: Steady NS
                F_mms = compute_steady_ns_mms_source<dim>(x_q, current_time, nu, L_y);
            }
            else
            {
                // Phase D: Unsteady NS
                const double t_old = current_time - dt;
                F_mms = compute_unsteady_ns_mms_source<dim>(x_q, current_time, t_old, nu, L_y);
            }

            // ================================================================
            // Get U_old at quadrature point (for convection and time derivative)
            // ================================================================
            double ux_old_q = 0.0, uy_old_q = 0.0;
            dealii::Tensor<1, dim> U_old;
            double div_U_old = 0.0;

            if (include_time_derivative || include_convection)
            {
                // For time-dependent: use exact U at t_old = t_new - dt
                // For steady with convection: use exact U at current_time (same as source term)
                const double t_eval = include_time_derivative ? (current_time - dt) : current_time;
                U_old = mms_exact_velocity<dim>(x_q, t_eval, L_y);
                ux_old_q = U_old[0];
                uy_old_q = U_old[1];
                div_U_old = 0.0;  // Exact solution is divergence-free
            }

            // ================================================================
            // ASSEMBLE LOCAL MATRICES
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_ux<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_uy<dim>(grad_phi_uy_i);

                // ============================================================
                // RHS contributions
                // ============================================================
                // MMS source: (f, V)
                local_rhs_ux(i) += F_mms[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_mms[1] * phi_uy_i * JxW;

                // Time derivative RHS: (U^{k-1}/τ, V)
                if (include_time_derivative)
                {
                    local_rhs_ux(i) += (ux_old_q / dt) * phi_ux_i * JxW;
                    local_rhs_uy(i) += (uy_old_q / dt) * phi_uy_i * JxW;
                }

                // ============================================================
                // LHS contributions (loop over trial functions)
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // --------------------------------------------------------
                    // Time derivative LHS: (U/τ, V)
                    // --------------------------------------------------------
                    if (include_time_derivative)
                    {
                        local_ux_ux(i, j) += (1.0 / dt) * phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += (1.0 / dt) * phi_uy_j * phi_uy_i * JxW;
                    }

                    // --------------------------------------------------------
                    // Viscous term: ν(T(U), T(V))
                    // --------------------------------------------------------
                    // T(U_x) : T(V_x), T(U_y) : T(V_y), plus cross terms
                    local_ux_ux(i, j) += nu * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += nu * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += nu * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += nu * (T_U_x * T_V_y) * JxW;

                    // --------------------------------------------------------
                    // Convection: B_h(U^{k-1}, U, V) (skew-symmetric form, Eq. 37)
                    //   = (U^{k-1}·∇U, V) + ½(∇·U^{k-1})(U, V)
                    // --------------------------------------------------------
                    if (include_convection)
                    {
                        // Standard convection: (U^{k-1}·∇U, V)
                        const double U_dot_grad_phi_ux_j =
                            U_old[0] * grad_phi_ux_j[0] + U_old[1] * grad_phi_ux_j[1];
                        const double U_dot_grad_phi_uy_j =
                            U_old[0] * grad_phi_uy_j[0] + U_old[1] * grad_phi_uy_j[1];

                        local_ux_ux(i, j) += U_dot_grad_phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += U_dot_grad_phi_uy_j * phi_uy_i * JxW;

                        // Skew term: +0.5*(∇·U^{k-1})(U, V)
                        // Note: div_U_old = 0 for exact MMS solution
                        local_ux_ux(i, j) += 0.5 * div_U_old * phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += 0.5 * div_U_old * phi_uy_j * phi_uy_i * JxW;
                    }
                }

                // ============================================================
                // Pressure gradient: -(p, ∇·V)
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);

                    // -(p, ∇·V) where ∇·V = ∂φ_ux/∂x for V=(φ_ux, 0)
                    //                     = ∂φ_uy/∂y for V=(0, φ_uy)
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // ================================================================
            // Continuity equation: (∇·U, q) = 0
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // (∇·U, q) where ∇·U = ∂ux/∂x + ∂uy/∂y
                    local_p_ux(i, j) += grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) += grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }  // end quadrature loop

        // ====================================================================
        // Distribute to global matrix
        // ====================================================================
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto gi = ux_to_ns_map[ux_local_dofs[i]];
            ns_rhs(gi) += local_rhs_ux(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_ux_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_ux_uy(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                ns_matrix.add(gi, p_to_ns_map[p_local_dofs[j]], local_ux_p(i, j));
        }

        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto gi = uy_to_ns_map[uy_local_dofs[i]];
            ns_rhs(gi) += local_rhs_uy(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_uy_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_uy_uy(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                ns_matrix.add(gi, p_to_ns_map[p_local_dofs[j]], local_uy_p(i, j));
        }

        for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
        {
            const auto gi = p_to_ns_map[p_local_dofs[i]];
            ns_rhs(gi) += local_rhs_p(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, ux_to_ns_map[ux_local_dofs[j]], local_p_ux(i, j));
            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                ns_matrix.add(gi, uy_to_ns_map[uy_local_dofs[j]], local_p_uy(i, j));
        }
    }  // end cell loop

    // Apply constraints
    ns_constraints.condense(ns_matrix, ns_rhs);
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void assemble_ns_mms_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double, double, double, double,
    bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&);

template dealii::Tensor<1, 2> compute_steady_stokes_mms_source<2>(
    const dealii::Point<2>&, double, double, double);

template dealii::Tensor<1, 2> compute_unsteady_stokes_mms_source<2>(
    const dealii::Point<2>&, double, double, double);

template dealii::Tensor<1, 2> compute_steady_ns_mms_source<2>(
    const dealii::Point<2>&, double, double, double);

template dealii::Tensor<1, 2> compute_unsteady_ns_mms_source<2>(
    const dealii::Point<2>&, double, double, double, double);