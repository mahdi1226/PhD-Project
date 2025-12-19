// ============================================================================
// assembly/ns_assembler.cc - Navier-Stokes Assembly (CORRECTED)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e (discrete scheme), p.505
//
// CORRECTED to match paper:
//   1. θ^{k-1} (lagged) for capillary, viscosity, gravity
//   2. Symmetric gradient T(U) = ∇U + (∇U)^T
//   3. Skew convection B_h(U^{k-1}, U^k, V) for energy stability
//   4. Full Kelvin force B_h^m(V, H^k, M^k) with DG face terms
//   5. MMS source term support for verification
//
// ============================================================================

#include "assembly/ns_assembler.h"
#include "physics/material_properties.h"
#include "physics/skew_forms.h"
#include "physics/kelvin_force.h"
#include "mms/ns_mms.h"


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Helper: Compute symmetric gradient T(U) = ∇U + (∇U)^T
//
// In 2D: T = [T_xx  T_xy]   where T_xx = 2 * ∂u_x/∂x
//            [T_xy  T_yy]         T_yy = 2 * ∂u_y/∂y
//                                 T_xy = ∂u_x/∂y + ∂u_y/∂x
//
// Note: T = 2*D where D is the strain rate tensor
// ============================================================================
template <int dim>
dealii::SymmetricTensor<2, dim> compute_symmetric_gradient(
    const dealii::Tensor<1, dim>& grad_ux,
    const dealii::Tensor<1, dim>& grad_uy)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_ux[0];                    // 2 * ∂u_x/∂x
    T[1][1] = 2.0 * grad_uy[1];                    // 2 * ∂u_y/∂y
    T[0][1] = grad_ux[1] + grad_uy[0];             // ∂u_x/∂y + ∂u_y/∂x
    return T;
}

// Helper: T(φ_i) for velocity test function where V_i = (φ_ux, 0)
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_x(
    const dealii::Tensor<1, dim>& grad_phi_ux)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_phi_ux[0];     // 2 * ∂φ_ux/∂x
    T[1][1] = 0.0;                       // no y-component
    T[0][1] = grad_phi_ux[1];           // ∂φ_ux/∂y + 0
    return T;
}

// Helper: T(φ_i) for velocity test function where V_i = (0, φ_uy)
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_y(
    const dealii::Tensor<1, dim>& grad_phi_uy)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 0.0;                       // no x-component
    T[1][1] = 2.0 * grad_phi_uy[1];     // 2 * ∂φ_uy/∂y
    T[0][1] = grad_phi_uy[0];           // 0 + ∂φ_uy/∂x
    return T;
}

// ============================================================================
// Main assembly function (CORRECTED to match Paper Eq. 42e)
// ============================================================================
template <int dim>
void assemble_ns_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::DoFHandler<dim>* M_dof_handler,
    const dealii::Vector<double>& ux_old,
    const dealii::Vector<double>& uy_old,
    const dealii::Vector<double>& theta_old,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const dealii::Vector<double>* mx_solution,
    const dealii::Vector<double>* my_solution,
    const Parameters& params,
    double dt,
    double current_time,
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

    // Quadrature
    dealii::QGauss<dim> quadrature(params.fe.degree_velocity + 2);
    dealii::QGauss<dim - 1> face_quadrature(params.fe.degree_velocity + 2);
    const unsigned int n_q_points = quadrature.size();
    const unsigned int n_face_q_points = face_quadrature.size();

    // FEValues for velocity (Q2)
    dealii::FEValues<dim> ux_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_Q2, quadrature,
        dealii::update_values | dealii::update_gradients);

    // FEValues for pressure (Q1)
    dealii::FEValues<dim> p_fe_values(fe_Q1, quadrature,
        dealii::update_values);

    // FEValues for θ (for lagged θ^{k-1})
    dealii::FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        dealii::update_values);

    // FEValues for ψ (for ∇ψ^k in capillary)
    dealii::FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        dealii::update_gradients);

    // FEValues for φ (for H^k = ∇φ^k)
    std::unique_ptr<dealii::FEValues<dim>> phi_fe_values;
    if (params.enable_magnetic && phi_dof_handler != nullptr)
    {
        phi_fe_values = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler->get_fe(), quadrature,
            dealii::update_gradients | dealii::update_hessians);
    }

    // FEValues for M (DG) - CRITICAL: separate from θ!
    // Need update_gradients for div(M) in Kelvin skew form
    std::unique_ptr<dealii::FEValues<dim>> M_fe_values;
    const bool use_full_kelvin = params.enable_magnetic &&
                                  M_dof_handler != nullptr &&
                                  mx_solution != nullptr &&
                                  my_solution != nullptr;
    if (use_full_kelvin)
    {
        M_fe_values = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler->get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
    }

    // FEFaceValues for B_h^m face terms (velocity)
    dealii::FEFaceValues<dim> ux_fe_face_values(fe_Q2, face_quadrature,
        dealii::update_values | dealii::update_normal_vectors | dealii::update_JxW_values);
    dealii::FEFaceValues<dim> uy_fe_face_values(fe_Q2, face_quadrature,
        dealii::update_values);

    // FEFaceValues for φ (H traces)
    std::unique_ptr<dealii::FEFaceValues<dim>> phi_fe_face_values_here;
    std::unique_ptr<dealii::FEFaceValues<dim>> phi_fe_face_values_there;
    if (use_full_kelvin)
    {
        phi_fe_face_values_here = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature,
            dealii::update_gradients);
        phi_fe_face_values_there = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature,
            dealii::update_gradients);
    }

    // FEFaceValues for M (DG traces)
    std::unique_ptr<dealii::FEFaceValues<dim>> M_fe_face_values_here;
    std::unique_ptr<dealii::FEFaceValues<dim>> M_fe_face_values_there;
    if (use_full_kelvin)
    {
        M_fe_face_values_here = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature,
            dealii::update_values);
        M_fe_face_values_there = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature,
            dealii::update_values);
    }

    // Local matrices (9 blocks for 3×3 system)
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

    // Solution values at quadrature points
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);
    std::vector<double> theta_old_values(n_q_points);  // LAGGED θ^{k-1}
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);
    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> mx_gradients(n_q_points);  // For div(M) = ∂Mx/∂x + ∂My/∂y
    std::vector<dealii::Tensor<1, dim>> my_gradients(n_q_points);

    // Face values for B_h^m
    std::vector<dealii::Tensor<1, dim>> phi_grad_here(n_face_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_grad_there(n_face_q_points);
    std::vector<double> mx_here(n_face_q_points), mx_there(n_face_q_points);
    std::vector<double> my_here(n_face_q_points), my_there(n_face_q_points);

    // MMS parameters
    const bool mms_mode = params.enable_mms;
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double nu_mms = nu_water;  // Constant viscosity for MMS

    // ========================================================================
    // Gravity (optional extension, NOT in paper Eq. 42e)
    // WARNING: Gravity is not part of the energy proof in the paper.
    // Enable only for physical simulations, not paper replication.
    // Disabled in MMS mode.
    // ========================================================================
    const bool use_gravity = params.enable_gravity && !mms_mode;
    const double g_mag = gravity_dimensionless;
    dealii::Tensor<1, dim> g_direction = params.gravity_direction;

    if (use_gravity)
    {
        static bool gravity_warned = false;
        if (!gravity_warned)
        {
            std::cout << "[NS] WARNING: Gravity is ENABLED. This is NOT part of "
                      << "paper Eq. 42e and breaks the discrete energy law.\n";
            gravity_warned = true;
        }
    }

    // Diagnostic tracking
    double max_F_cap = 0.0, max_F_mag = 0.0, max_F_grav = 0.0;

    // Cell iterators (all share same mesh)
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    typename dealii::DoFHandler<dim>::active_cell_iterator phi_cell;
    typename dealii::DoFHandler<dim>::active_cell_iterator M_cell;

    if (params.enable_magnetic && phi_dof_handler != nullptr)
        phi_cell = phi_dof_handler->begin_active();

    if (use_full_kelvin)
        M_cell = M_dof_handler->begin_active();

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    for (; ux_cell != ux_dof_handler.end();
         ++ux_cell, ++uy_cell, ++p_cell, ++theta_cell, ++psi_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        // Track current phi/M cells for face loop
        typename dealii::DoFHandler<dim>::active_cell_iterator current_phi_cell;
        typename dealii::DoFHandler<dim>::active_cell_iterator current_M_cell;

        if (params.enable_magnetic && phi_dof_handler != nullptr)
        {
            current_phi_cell = phi_cell;
            phi_fe_values->reinit(phi_cell);
            ++phi_cell;
        }

        if (use_full_kelvin)
        {
            current_M_cell = M_cell;
            M_fe_values->reinit(M_cell);
            ++M_cell;
        }

        // Reset local matrices
        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux = 0;  local_p_uy = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        // Get DoF indices
        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get solution values at quadrature points
        ux_fe_values.get_function_values(ux_old, ux_old_values);
        uy_fe_values.get_function_values(uy_old, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);

        // CRITICAL: Use θ^{k-1} (lagged) for capillary, viscosity, gravity
        theta_fe_values.get_function_values(theta_old, theta_old_values);

        psi_fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (params.enable_magnetic && phi_dof_handler != nullptr && phi_solution != nullptr)
        {
            phi_fe_values->get_function_gradients(*phi_solution, phi_gradients);
            phi_fe_values->get_function_hessians(*phi_solution, phi_hessians);
        }

        // Get M^k values using M's FEValues (CORRECT for DG!)
        if (use_full_kelvin)
        {
            M_fe_values->get_function_values(*mx_solution, mx_values);
            M_fe_values->get_function_values(*my_solution, my_values);
            M_fe_values->get_function_gradients(*mx_solution, mx_gradients);
            M_fe_values->get_function_gradients(*my_solution, my_gradients);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // U^{k-1} at quadrature point
            const double ux_q = ux_old_values[q];
            const double uy_q = uy_old_values[q];
            dealii::Tensor<1, dim> U_old;
            U_old[0] = ux_q;
            U_old[1] = uy_q;

            const dealii::Tensor<1, dim>& grad_ux_old = ux_old_gradients[q];
            const dealii::Tensor<1, dim>& grad_uy_old = uy_old_gradients[q];

            // div(U^{k-1}) for skew form
            const double div_U_old = grad_ux_old[0] + grad_uy_old[1];

            // θ^{k-1} (LAGGED) for all material properties
            const double theta_old_q = theta_old_values[q];

            // ∇ψ^k for capillary
            const dealii::Tensor<1, dim>& grad_psi = psi_gradients[q];

            // Phase-dependent viscosity at θ^{k-1} [Eq. 17]
            // Use constant viscosity in MMS mode
            const double nu = mms_mode ? nu_mms : viscosity(theta_old_q);

            // ================================================================
            // Capillary force [Eq. 10]: F_cap = (λ/ε)θ^{k-1}∇ψ^k
            // CORRECTED: Uses lagged θ^{k-1}, not θ^k
            // Disabled in MMS mode
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            F_cap = 0;
            if (!mms_mode)
            {
                const double coeff = lambda / epsilon;
                F_cap[0] = coeff * theta_old_q * grad_psi[0];
                F_cap[1] = coeff * theta_old_q * grad_psi[1];
            }

            // ================================================================
            // Gravity [Eq. 19]: F_grav = (1 + r·H(θ^{k-1}/ε))g
            // NOTE: Not in Eq. 42e, optional extension
            // Disabled in MMS mode (use_gravity already includes !mms_mode)
            // ================================================================
            dealii::Tensor<1, dim> F_grav;
            F_grav = 0;
            if (use_gravity)
            {
const double H_theta = heaviside(theta_old_q / epsilon);
const double gravity_factor = r * H_theta;
                F_grav = gravity_factor * g_mag * g_direction;
            }

            // ================================================================
            // MMS source term: f = ∂U/∂t + (U·∇)U - ν∇²U + ∇p
            // Only active in MMS mode
            // ================================================================
            dealii::Tensor<1, dim> F_mms;
            F_mms = 0;
            if (mms_mode)
            {
                F_mms = compute_ns_mms_source<dim>(x_q, current_time, nu_mms, L_y);
            }

            // ================================================================
            // Kelvin force: CELL CONTRIBUTION of B_h^m(V, H^k, M^k)
            //
            // NOTE: Poisson is solved in "total-field" form:
            //       (∇φ, ∇χ) = (h_a - M, ∇χ)
            //       Hence H^k = ∇φ^k (no minus sign, no +h_a here).
            //
            // Paper's DG skew form (CORRECTED):
            //   B_h^m(V, H, M) cell = (M·∇)H · V + ½ (∇·M)(H·V)
            //
            // This is NOT the same as (V·∇)H · M + ½ div(V)(H·M)!
            // The skew is in M (for div(M)), not in V.
            //
            // Energy cancellation B_h^m(H, H, M) = 0 requires div(M).
            //
            // For V_i = (φ_ux, 0):
            //   (M·∇)H · V_i = (M·∇)H_x * φ_ux
            //   (H·V_i) = H_x * φ_ux
            //
            // For V_i = (0, φ_uy):
            //   (M·∇)H · V_i = (M·∇)H_y * φ_uy
            //   (H·V_i) = H_y * φ_uy
            //
            // Disabled in MMS mode
            // ================================================================

            // Pre-compute Kelvin quantities using kelvin_force.h helpers
            double div_M = 0.0;
            dealii::Tensor<1, dim> M_grad_H;
            dealii::Tensor<1, dim> H_field;
            M_grad_H = 0;
            H_field = 0;

            const bool compute_kelvin = !mms_mode && use_full_kelvin && phi_solution != nullptr;
            if (compute_kelvin)
            {
                H_field = phi_gradients[q];  // H = ∇φ
                const dealii::Tensor<2, dim>& hess_phi = phi_hessians[q];

                // Build M vector from scalar components
                dealii::Tensor<1, dim> M = KelvinForce::make_M_vector<dim>(mx_values[q], my_values[q]);

                // Compute div(M) = ∂Mx/∂x + ∂My/∂y
                div_M = KelvinForce::compute_div_M<dim>(mx_gradients[q], my_gradients[q]);

                // Compute (M·∇)H
                M_grad_H = KelvinForce::compute_M_grad_H<dim>(M, hess_phi);

                // Track max force for diagnostics
                max_F_mag = std::max(max_F_mag, mu_0 * M_grad_H.norm());
            }

            // Track max forces for diagnostics
            max_F_cap = std::max(max_F_cap, F_cap.norm());
            max_F_grav = std::max(max_F_grav, F_grav.norm());



            // DEBUG: Disable forces to isolate div U source
            const bool DEBUG_NO_CAPILLARY = false;
            const bool DEBUG_NO_GRAVITY = false;

            if (DEBUG_NO_CAPILLARY) F_cap = 0;
            if (DEBUG_NO_GRAVITY) F_grav = 0;
            // Kelvin is already controlled by compute_kelvin

            // ================================================================
            // Assemble local contributions
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                // Test function symmetric gradients T(V)
                dealii::SymmetricTensor<2, dim> T_V_x = compute_T_test_x<dim>(grad_phi_ux_i);
                dealii::SymmetricTensor<2, dim> T_V_y = compute_T_test_y<dim>(grad_phi_uy_i);

                // ============================================================
                // RHS: momentum equations
                // ============================================================

                // Time derivative: (1/τ)(U^{k-1}, V)
                double rhs_ux = (1.0 / dt) * ux_q * phi_ux_i;
                double rhs_uy = (1.0 / dt) * uy_q * phi_uy_i;

                // NOTE: Convection B_h(U^{k-1}, U^k, V) is on LHS only (semi-implicit)
                // No convection term on RHS per paper Eq. 42e

                // Capillary force: (λ/ε)(θ^{k-1}∇ψ^k, V)
                rhs_ux += F_cap[0] * phi_ux_i;
                rhs_uy += F_cap[1] * phi_uy_i;

                // Gravity (optional, not in Eq. 42e)
                rhs_ux += F_grav[0] * phi_ux_i;
                rhs_uy += F_grav[1] * phi_uy_i;

                // MMS source term
                rhs_ux += F_mms[0] * phi_ux_i;
                rhs_uy += F_mms[1] * phi_uy_i;

                // ============================================================
                // Kelvin cell term: μ₀ B_h^m(V, H, M) cell contribution
                // Uses kelvin_force.h::cell_kernel
                // ============================================================
                if (compute_kelvin)
                {
                    double kelvin_ux, kelvin_uy;
                    KelvinForce::cell_kernel<dim>(
                        M_grad_H, div_M, H_field,
                        phi_ux_i, phi_uy_i,
                        mu_0,
                        kelvin_ux, kelvin_uy);
                    rhs_ux += kelvin_ux;
                    rhs_uy += kelvin_uy;
                }

                local_rhs_ux(i) += rhs_ux * JxW;
                local_rhs_uy(i) += rhs_uy * JxW;

                // ============================================================
                // Matrix: velocity-velocity blocks
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // Trial function symmetric gradients T(U)
                    dealii::SymmetricTensor<2, dim> T_U_x = compute_T_test_x<dim>(grad_phi_ux_j);
                    dealii::SymmetricTensor<2, dim> T_U_y = compute_T_test_y<dim>(grad_phi_uy_j);

                    // Mass: (1/τ)(U, V)
                    local_ux_ux(i, j) += (1.0 / dt) * phi_ux_i * phi_ux_j * JxW;
                    local_uy_uy(i, j) += (1.0 / dt) * phi_uy_i * phi_uy_j * JxW;

                    // ========================================================
                    // Viscosity: (ν(θ^{k-1}) T(U), T(V))
                    // Paper Eq. 42e uses T = ∇u + (∇u)^T
                    // Inner product: T(U):T(V) = sum_ij T_U_ij * T_V_ij
                    // ========================================================
                    local_ux_ux(i, j) += nu * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += nu * (T_U_y * T_V_y) * JxW;

                    // Cross terms from symmetric gradient coupling
                    local_ux_uy(i, j) += nu * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += nu * (T_U_x * T_V_y) * JxW;

                    // ========================================================
                    // Skew convection LHS: B_h(U^{k-1}, U^k, V)
                    // B_h(w, u, v) = (w·∇u, v) + (1/2)(div(w)u, v)
                    //
                    // With w = U^{k-1} (known), u = U^k (unknown), v = V (test)
                    // Full vector form requires cross-component coupling
                    // ========================================================
                    const double U_dot_grad_phi_ux_j = U_old[0] * grad_phi_ux_j[0] + U_old[1] * grad_phi_ux_j[1];
                    const double U_dot_grad_phi_uy_j = U_old[0] * grad_phi_uy_j[0] + U_old[1] * grad_phi_uy_j[1];

                    // Diagonal blocks
                    local_ux_ux(i, j) += (U_dot_grad_phi_ux_j + 0.5 * div_U_old * phi_ux_j) * phi_ux_i * JxW;
                    local_uy_uy(i, j) += (U_dot_grad_phi_uy_j + 0.5 * div_U_old * phi_uy_j) * phi_uy_i * JxW;

                    // Cross blocks for full vector consistency (Eq. 42e)
                    //local_ux_uy(i, j) += (U_dot_grad_phi_uy_j + 0.5 * div_U_old * phi_uy_j) * phi_ux_i * JxW;
                    //local_uy_ux(i, j) += (U_dot_grad_phi_ux_j + 0.5 * div_U_old * phi_ux_j) * phi_uy_i * JxW;

                    // Grad-div stabilization (optional)
                    if (grad_div > 0.0)
                    {
                        local_ux_ux(i, j) += grad_div * grad_phi_ux_i[0] * grad_phi_ux_j[0] * JxW;
                        local_ux_uy(i, j) += grad_div * grad_phi_ux_i[0] * grad_phi_uy_j[1] * JxW;
                        local_uy_ux(i, j) += grad_div * grad_phi_uy_i[1] * grad_phi_ux_j[0] * JxW;
                        local_uy_uy(i, j) += grad_div * grad_phi_uy_i[1] * grad_phi_uy_j[1] * JxW;
                    }
                }

                // ============================================================
                // Matrix: velocity-pressure coupling -(P, div V)
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);

                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // ================================================================
            // Continuity equation: (div U, Q) = 0
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    local_p_ux(i, j) += grad_phi_ux_j[0] * phi_p_i * JxW;
                    local_p_uy(i, j) += grad_phi_uy_j[1] * phi_p_i * JxW;
                }
            }
        }

        // ====================================================================
        // Assemble cell contributions into global matrix
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

        // ====================================================================
        // FACE LOOP: B_h^m face terms for Kelvin force
        //
        // Face contribution: -μ₀ (V·n⁻) [[H]] · {M}
        // where [[H]] = H⁻ - H⁺, {M} = ½(M⁻ + M⁺)
        //
        // Disabled in MMS mode
        // ====================================================================
        if (!mms_mode && use_full_kelvin)
        {
            for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                // Only process interior faces once
                // AMR-SAFE: Use level + cell_id comparison, not index()
                // This preserves B_h^m(H,H,M) = 0 under adaptive refinement
                if (!ux_cell->at_boundary(face_no))
                {
                    const auto neighbor = ux_cell->neighbor(face_no);

                    // Process face only once: when cells are same level and our id < neighbor id
                    // This is robust under AMR (cell->index() is NOT)
                    if (neighbor->level() == ux_cell->level() &&
                        ux_cell->id() < neighbor->id())
                    {
                        const unsigned int neighbor_face_no = ux_cell->neighbor_of_neighbor(face_no);

                        // Reinit velocity face values
                        ux_fe_face_values.reinit(ux_cell, face_no);
                        uy_fe_face_values.reinit(uy_cell, face_no);

                        // Get phi neighbor cell
                        auto phi_neighbor = current_phi_cell->neighbor(face_no);

                        // Reinit phi face values (both sides)
                        phi_fe_face_values_here->reinit(current_phi_cell, face_no);
                        phi_fe_face_values_there->reinit(phi_neighbor, neighbor_face_no);

                        // Get M neighbor cell
                        auto M_neighbor = current_M_cell->neighbor(face_no);

                        // Reinit M face values (both sides)
                        M_fe_face_values_here->reinit(current_M_cell, face_no);
                        M_fe_face_values_there->reinit(M_neighbor, neighbor_face_no);

                        // Get H = ∇φ traces
                        phi_fe_face_values_here->get_function_gradients(*phi_solution, phi_grad_here);
                        phi_fe_face_values_there->get_function_gradients(*phi_solution, phi_grad_there);

                        // Get M traces
                        M_fe_face_values_here->get_function_values(*mx_solution, mx_here);
                        M_fe_face_values_there->get_function_values(*mx_solution, mx_there);
                        M_fe_face_values_here->get_function_values(*my_solution, my_here);
                        M_fe_face_values_there->get_function_values(*my_solution, my_there);

                        // Get neighbor velocity cell and DoFs
                        auto uy_neighbor = uy_cell->neighbor(face_no);
                        std::vector<dealii::types::global_dof_index> ux_there_dofs(dofs_per_cell_Q2);
                        std::vector<dealii::types::global_dof_index> uy_there_dofs(dofs_per_cell_Q2);
                        neighbor->get_dof_indices(ux_there_dofs);
                        uy_neighbor->get_dof_indices(uy_there_dofs);

                        // FEFaceValues for neighbor velocity
                        dealii::FEFaceValues<dim> ux_fe_face_values_there(fe_Q2, face_quadrature, dealii::update_values);
                        dealii::FEFaceValues<dim> uy_fe_face_values_there(fe_Q2, face_quadrature, dealii::update_values);
                        ux_fe_face_values_there.reinit(neighbor, neighbor_face_no);
                        uy_fe_face_values_there.reinit(uy_neighbor, neighbor_face_no);

                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            const double JxW_face = ux_fe_face_values.JxW(q);
                            const dealii::Tensor<1, dim>& normal_minus = ux_fe_face_values.normal_vector(q);

                            // Compute jump and average using helpers (one truth)
                            dealii::Tensor<1, dim> jump_H = KelvinForce::compute_jump_H<dim>(
                                phi_grad_here[q], phi_grad_there[q]);

                            dealii::Tensor<1, dim> M_minus = KelvinForce::make_M_vector<dim>(mx_here[q], my_here[q]);
                            dealii::Tensor<1, dim> M_plus = KelvinForce::make_M_vector<dim>(mx_there[q], my_there[q]);
                            dealii::Tensor<1, dim> avg_M = KelvinForce::compute_avg_M<dim>(M_minus, M_plus);

                            // Assemble face contribution for minus cell test functions
                            // Uses normal_minus and jump_H = H⁻ - H⁺
                            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
                            {
                                const double phi_ux_here = ux_fe_face_values.shape_value(i, q);
                                const double phi_uy_here = uy_fe_face_values.shape_value(i, q);

                                double kelvin_ux, kelvin_uy;
                                KelvinForce::face_kernel<dim>(
                                    phi_ux_here, phi_uy_here,
                                    normal_minus, jump_H, avg_M, mu_0,
                                    kelvin_ux, kelvin_uy);

                                const auto gi_ux = ux_to_ns_map[ux_local_dofs[i]];
                                const auto gi_uy = uy_to_ns_map[uy_local_dofs[i]];

                                ns_rhs(gi_ux) += kelvin_ux * JxW_face;
                                ns_rhs(gi_uy) += kelvin_uy * JxW_face;
                            }

                            // Assemble face contribution for plus cell test functions
                            // Same normal_minus and jump_H, different test functions
                            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
                            {
                                const double phi_ux_there = ux_fe_face_values_there.shape_value(i, q);
                                const double phi_uy_there = uy_fe_face_values_there.shape_value(i, q);

                                double kelvin_ux, kelvin_uy;
                                KelvinForce::face_kernel<dim>(
                                    phi_ux_there, phi_uy_there,
                                    normal_minus, jump_H, avg_M, mu_0,
                                    kelvin_ux, kelvin_uy);

                                const auto gi_ux = ux_to_ns_map[ux_there_dofs[i]];
                                const auto gi_uy = uy_to_ns_map[uy_there_dofs[i]];

                                ns_rhs(gi_ux) += kelvin_ux * JxW_face;
                                ns_rhs(gi_uy) += kelvin_uy * JxW_face;
                            }
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Apply constraints symmetrically to preserve saddle-point structure
    // This is CRITICAL for pressure-velocity adjoint consistency
    // ========================================================================
    ns_constraints.condense(ns_matrix, ns_rhs);

    // Diagnostic output (disabled in MMS mode)
    if (params.output.verbose && !mms_mode)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)
        {
            std::cout << "[NS] Forces: |F_cap|=" << max_F_cap
                      << ", |F_mag|=" << max_F_mag
                      << ", |F_grav|=" << max_F_grav
                      << (use_full_kelvin ? " (full B_h^m)" : " (equilibrium)")
                      << (use_gravity ? " [gravity ON]" : "") << "\n";
        }
        ++call_count;
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_ns_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>*,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const Parameters&,
    double,
    double,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&);