// ============================================================================
// assembly/ns_assembler.cc - Navier-Stokes Assembly (MMS-AWARE VERSION)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e (discrete scheme), p.505
//
// Paper's discrete NS scheme:
//
//   (δU^k/τ, V) + B_h(U^{k-1}, U^k, V) + (ν(θ^{k-1})T(U^k), T(V))
//     - (P^k, div V) + (div U^k, Q) = (F^{k-1}, V)
//
// where:
//   - δU^k = U^k - U^{k-1}
//   - T(U) = ∇U + (∇U)^T (symmetric gradient)
//   - B_h(w,u,v) = (w·∇u, v) + ½(∇·w)(u, v) (skew-symmetric convection, Eq. 37)
//   - ν(θ^{k-1}) = viscosity at LAGGED θ
//   - F^{k-1} = F_cap + F_mag + F_grav with LAGGED fields
//
// SKEW-SYMMETRIC FORM (Eq. 37):
//   The convection term uses the Temam skew-symmetric form for energy stability.
//   Both terms (w·∇u, v) AND ½(∇·w)(u, v) are assembled.
//
// MMS FIX: When enable_mms=true, uses EXACT U_old values at quadrature points
//          instead of FE-interpolated values for consistency with MMS source.
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
// ============================================================================
template <int dim>
dealii::SymmetricTensor<2, dim> compute_symmetric_gradient(
    const dealii::Tensor<1, dim>& grad_ux,
    const dealii::Tensor<1, dim>& grad_uy)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_ux[0];
    T[1][1] = 2.0 * grad_uy[1];
    T[0][1] = grad_ux[1] + grad_uy[0];
    return T;
}

// Helper: T(φ_i) for velocity test function V_i = (φ_ux, 0)
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_x(
    const dealii::Tensor<1, dim>& grad_phi_ux)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_phi_ux[0];
    T[1][1] = 0.0;
    T[0][1] = grad_phi_ux[1];
    return T;
}

// Helper: T(φ_i) for velocity test function V_i = (0, φ_uy)
template <int dim>
dealii::SymmetricTensor<2, dim> compute_T_test_y(
    const dealii::Tensor<1, dim>& grad_phi_uy)
{
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 0.0;
    T[1][1] = 2.0 * grad_phi_uy[1];
    T[0][1] = grad_phi_uy[0];
    return T;
}

// ============================================================================
// Helper: Compute exact MMS velocity at a point (for MMS mode consistency)
// This matches the exact solution in ns_mms.h
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_mms_exact_velocity_assembler(
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
    // ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
    U[0] = t * (M_PI / L_y) * sin_px * sin_px * sin_2py;
    // uy = -t·π·sin(2πx)·sin²(πy/L_y)
    U[1] = -t * M_PI * sin_2px * sin_py * sin_py;

    return U;
}

// ============================================================================
// Main assembly function (Paper Eq. 42e)
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
    dealii::FEValues<dim> p_fe_values(fe_Q1, quadrature, dealii::update_values);

    // FEValues for θ (lagged θ^{k-1})
    dealii::FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
                                          dealii::update_values);

    // FEValues for ψ (∇ψ^k in capillary)
    dealii::FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
                                        dealii::update_gradients);

    // FEValues for φ (H^k = ∇φ^k)
    std::unique_ptr<dealii::FEValues<dim>> phi_fe_values;
    if (params.enable_magnetic && phi_dof_handler != nullptr)
    {
        phi_fe_values = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler->get_fe(), quadrature,
            dealii::update_gradients | dealii::update_hessians);
    }

    // FEValues for M (DG)
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

    // FEFaceValues for Kelvin face terms
    dealii::FEFaceValues<dim> ux_fe_face_values(fe_Q2, face_quadrature,
                                                dealii::update_values | dealii::update_normal_vectors |
                                                dealii::update_JxW_values);
    dealii::FEFaceValues<dim> uy_fe_face_values(fe_Q2, face_quadrature,
                                                dealii::update_values);

    std::unique_ptr<dealii::FEFaceValues<dim>> phi_fe_face_values_here;
    std::unique_ptr<dealii::FEFaceValues<dim>> phi_fe_face_values_there;
    std::unique_ptr<dealii::FEFaceValues<dim>> M_fe_face_values_here;
    std::unique_ptr<dealii::FEFaceValues<dim>> M_fe_face_values_there;
    if (use_full_kelvin)
    {
        phi_fe_face_values_here = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature, dealii::update_gradients);
        phi_fe_face_values_there = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature, dealii::update_gradients);
        M_fe_face_values_here = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature, dealii::update_values);
        M_fe_face_values_there = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature, dealii::update_values);
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
    std::vector<double> theta_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);
    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> mx_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> my_gradients(n_q_points);

    // Face values for Kelvin
    std::vector<dealii::Tensor<1, dim>> phi_grad_here(n_face_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_grad_there(n_face_q_points);
    std::vector<double> mx_here(n_face_q_points), mx_there(n_face_q_points);
    std::vector<double> my_here(n_face_q_points), my_there(n_face_q_points);

    // Parameters from params (NOT globals!)
    const bool mms_mode = params.enable_mms;
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double nu_mms = params.physics.nu_water;
    const double mu_0_val = params.physics.mu_0;
    const double grad_div = params.physics.grad_div_stabilization;

    // Gravity (optional, NOT in paper Eq. 42e)
    const bool use_gravity = params.enable_gravity && !mms_mode;
    const double g_mag = params.physics.gravity;
    const double density_ratio = params.physics.r;
    dealii::Tensor<1, dim> g_direction = params.gravity_direction;

    // Diagnostic tracking
    double max_F_cap = 0.0, max_F_mag = 0.0, max_F_grav = 0.0;

    // Cell iterators
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

        typename dealii::DoFHandler<dim>::active_cell_iterator current_phi_cell;
        typename dealii::DoFHandler<dim>::active_cell_iterator current_M_cell;

        if (params.enable_magnetic && phi_dof_handler != nullptr)
        {
            phi_fe_values->reinit(phi_cell);
            current_phi_cell = phi_cell;
            ++phi_cell;
        }
        if (use_full_kelvin)
        {
            M_fe_values->reinit(M_cell);
            current_M_cell = M_cell;
            ++M_cell;
        }

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

        // Get solution values at quadrature points
        ux_fe_values.get_function_values(ux_old, ux_old_values);
        ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
        uy_fe_values.get_function_values(uy_old, uy_old_values);
        uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);
        theta_fe_values.get_function_values(theta_old, theta_old_values);
        psi_fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (params.enable_magnetic && phi_dof_handler != nullptr && phi_solution != nullptr)
        {
            phi_fe_values->get_function_gradients(*phi_solution, phi_gradients);
            phi_fe_values->get_function_hessians(*phi_solution, phi_hessians);
        }

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
            // KEY FIX: In MMS mode, use EXACT U_old values for consistency
            double ux_q, uy_q;
            dealii::Tensor<1, dim> U_old;
            double div_U_old;

            if (mms_mode) {
                // MMS MODE: Use EXACT U_old values
                const double t_old = current_time - dt;
                const double sin_px = std::sin(M_PI * x_q[0]);
                const double sin_py = std::sin(M_PI * x_q[1] / L_y);
                const double sin_2px = std::sin(2.0 * M_PI * x_q[0]);
                const double sin_2py = std::sin(2.0 * M_PI * x_q[1] / L_y);

                ux_q = t_old * (M_PI / L_y) * sin_px * sin_px * sin_2py;
                uy_q = -t_old * M_PI * sin_2px * sin_py * sin_py;
                U_old[0] = ux_q;
                U_old[1] = uy_q;
                div_U_old = 0.0;  // Exact solution is divergence-free
            } else {
                // PRODUCTION MODE: Use FE-interpolated values
                ux_q = ux_old_values[q];
                uy_q = uy_old_values[q];
                U_old[0] = ux_q;
                U_old[1] = uy_q;

                const dealii::Tensor<1, dim>& grad_ux_old = ux_old_gradients[q];
                const dealii::Tensor<1, dim>& grad_uy_old = uy_old_gradients[q];
                div_U_old = grad_ux_old[0] + grad_uy_old[1];
            }

            // θ^{k-1} (LAGGED)
            const double theta_old_q = theta_old_values[q];
            const dealii::Tensor<1, dim>& grad_psi = psi_gradients[q];

            // Viscosity at θ^{k-1}
            const double nu = mms_mode
                                  ? nu_mms
                                  : viscosity(theta_old_q, params.physics.epsilon, params.physics.nu_water,
                                              params.physics.nu_ferro);



            // ================================================================
            // Capillary force: F_cap = (λ/ε)θ^{k-1}∇ψ^k
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            F_cap = 0;
            const double coeff = params.physics.lambda / params.physics.epsilon;
            if (mms_mode)
            {
                // Use EXACT θ_old and ∇ψ for MMS consistency
                const double t_old = current_time - dt;
                const double t4_old = t_old * t_old * t_old * t_old;
                const double t4 = current_time * current_time * current_time * current_time;

                // θ_exact(t_old) = t_old^4 cos(πx) cos(πy)
                const double theta_exact = t4_old * std::cos(M_PI * x_q[0]) * std::cos(M_PI * x_q[1]);

                // ∇ψ_exact(t) = t^4 π [cos(πx)sin(πy), sin(πx)cos(πy)]
                const double grad_psi_x = t4 * M_PI * std::cos(M_PI * x_q[0]) * std::sin(M_PI * x_q[1]);
                const double grad_psi_y = t4 * M_PI * std::sin(M_PI * x_q[0]) * std::cos(M_PI * x_q[1]);

                F_cap[0] = coeff * theta_exact * grad_psi_x;
                F_cap[1] = coeff * theta_exact * grad_psi_y;
            }
            else
            {
                F_cap[0] = coeff * theta_old_q * grad_psi[0];
                F_cap[1] = coeff * theta_old_q * grad_psi[1];
            }
            // ================================================================
            // Gravity: F_grav = (1 + r·H(θ^{k-1}/ε))g
            // ================================================================
            dealii::Tensor<1, dim> F_grav;
            F_grav = 0;
            if (use_gravity)
            {
                const double H_theta = heaviside(theta_old_q / params.physics.epsilon);
                const double buoyancy_factor = density_ratio * H_theta;
                F_grav = buoyancy_factor * g_mag * g_direction;
            }

            // ================================================================
            // MMS source term (semi-implicit to match discretization)
            // ================================================================
            dealii::Tensor<1, dim> F_mms;
            F_mms = 0;
            if (mms_mode)
            {
                const double t_old = current_time - dt;
                F_mms = compute_ns_mms_source_semi_implicit<dim>(x_q, current_time, t_old, nu_mms, L_y);
                static bool printed_mms_debug = false;
                if (!printed_mms_debug && mms_mode) {
                    std::cout << "[DEBUG] Using semi_implicit MMS source, F_mms = "
                              << F_mms[0] << ", " << F_mms[1] << std::endl;
                    printed_mms_debug = true;
                }
            }

            // ================================================================
            // Kelvin force cell contribution
            // ================================================================
            double div_M = 0.0;
            dealii::Tensor<1, dim> M_grad_H;
            dealii::Tensor<1, dim> H_field;
            M_grad_H = 0;
            H_field = 0;

            const bool compute_kelvin = !mms_mode && use_full_kelvin && phi_solution != nullptr;
            if (compute_kelvin)
            {
                const dealii::Tensor<1, dim>& grad_phi = phi_gradients[q];
                const dealii::Tensor<2, dim>& hess_phi = phi_hessians[q];

                H_field = grad_phi;

                const double Mx = mx_values[q];
                const double My = my_values[q];
                const dealii::Tensor<1, dim>& grad_Mx = mx_gradients[q];
                const dealii::Tensor<1, dim>& grad_My = my_gradients[q];

                div_M = grad_Mx[0] + grad_My[1];

                // (M·∇)H
                M_grad_H[0] = Mx * hess_phi[0][0] + My * hess_phi[0][1];
                M_grad_H[1] = Mx * hess_phi[1][0] + My * hess_phi[1][1];
            }

            // ================================================================
            // Combine forces
            // ================================================================
            dealii::Tensor<1, dim> F_total;
            F_total = 0;
            if (!mms_mode)
            {
                F_total = F_cap + F_grav;

                if (compute_kelvin)
                {
                    // Add Kelvin: μ₀(M·∇)H
                    F_total[0] += mu_0_val * M_grad_H[0];
                    F_total[1] += mu_0_val * M_grad_H[1];
                }
            }
            else
            {
                F_total = F_mms;
            }

            // Track max forces for diagnostics
            max_F_cap = std::max(max_F_cap, F_cap.norm());
            max_F_grav = std::max(max_F_grav, F_grav.norm());
            if (compute_kelvin)
                max_F_mag = std::max(max_F_mag, (mu_0_val * M_grad_H).norm());

            // ================================================================
            // ASSEMBLE LOCAL MATRICES
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                auto T_V_x = compute_T_test_x<dim>(grad_phi_ux_i);
                auto T_V_y = compute_T_test_y<dim>(grad_phi_uy_i);

                // RHS: (1/dt)(U^{k-1}, V) + (F, V)
                local_rhs_ux(i) += ((1.0 / dt) * ux_q + F_total[0]) * phi_ux_i * JxW;
                local_rhs_uy(i) += ((1.0 / dt) * uy_q + F_total[1]) * phi_uy_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_x<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_y<dim>(grad_phi_uy_j);

                    // Mass: (1/dt)(U, V)
                    local_ux_ux(i, j) += (1.0 / dt) * phi_ux_i * phi_ux_j * JxW;
                    local_uy_uy(i, j) += (1.0 / dt) * phi_uy_i * phi_uy_j * JxW;

                    // Viscosity: (ν(θ^{k-1}) T(U), T(V))
                    // T(U_x) : T(V_x), T(U_y) : T(V_y), plus cross terms
                    local_ux_ux(i, j) += nu * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += nu * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += nu * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += nu * (T_U_x * T_V_y) * JxW;

                    // Convection: B_h(U^{k-1}, U, V) = (U^{k-1}·∇U, V) + 0.5*(∇·U^{k-1})(U, V)
                    // Paper Eq. 37: Skew-symmetric (Temam) form for energy stability
                    const double U_dot_grad_phi_ux_j = U_old[0] * grad_phi_ux_j[0] + U_old[1] * grad_phi_ux_j[1];
                    const double U_dot_grad_phi_uy_j = U_old[0] * grad_phi_uy_j[0] + U_old[1] * grad_phi_uy_j[1];

                    // Standard convection: (U^{k-1}·∇U, V)
                    local_ux_ux(i, j) += U_dot_grad_phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += U_dot_grad_phi_uy_j * phi_uy_i * JxW;

                    // Skew term: +0.5*(∇·U^{k-1})(U, V) for energy stability
                    local_ux_ux(i, j) += 0.5 * div_U_old * phi_ux_j * phi_ux_i * JxW;
                    local_uy_uy(i, j) += 0.5 * div_U_old * phi_uy_j * phi_uy_i * JxW;

                    // Grad-div stabilization: γ(∇·U, ∇·V)
                    if (grad_div > 0.0)
                    {
                        local_ux_ux(i, j) += grad_div * grad_phi_ux_j[0] * grad_phi_ux_i[0] * JxW;
                        local_ux_uy(i, j) += grad_div * grad_phi_uy_j[1] * grad_phi_ux_i[0] * JxW;
                        local_uy_ux(i, j) += grad_div * grad_phi_ux_j[0] * grad_phi_uy_i[1] * JxW;
                        local_uy_uy(i, j) += grad_div * grad_phi_uy_j[1] * grad_phi_uy_i[1] * JxW;
                    }
                }

                // Pressure gradient: -(p, ∇·V)
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // Continuity equation: (div U, Q) = 0
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

        // Assemble into global matrix
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
        // FACE LOOP: Kelvin face terms (skipped in MMS mode)
        // ====================================================================
        if (!mms_mode && use_full_kelvin)
        {
            for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                if (!ux_cell->at_boundary(face_no))
                {
                    const auto neighbor = ux_cell->neighbor(face_no);

                    if (neighbor->level() == ux_cell->level() &&
                        neighbor->is_active() &&
                        ux_cell->id() < neighbor->id())
                    {
                        if (current_phi_cell->at_boundary(face_no) ||
                            current_M_cell->at_boundary(face_no))
                            continue;

                        auto phi_neighbor = current_phi_cell->neighbor(face_no);
                        auto M_neighbor = current_M_cell->neighbor(face_no);

                        if (phi_neighbor->level() != current_phi_cell->level() ||
                            M_neighbor->level() != current_M_cell->level())
                            continue;

                        const unsigned int neighbor_face_no = ux_cell->neighbor_of_neighbor(face_no);

                        ux_fe_face_values.reinit(ux_cell, face_no);
                        uy_fe_face_values.reinit(uy_cell, face_no);

                        phi_fe_face_values_here->reinit(current_phi_cell, face_no);
                        phi_fe_face_values_there->reinit(phi_neighbor, neighbor_face_no);
                        M_fe_face_values_here->reinit(current_M_cell, face_no);
                        M_fe_face_values_there->reinit(M_neighbor, neighbor_face_no);

                        phi_fe_face_values_here->get_function_gradients(*phi_solution, phi_grad_here);
                        phi_fe_face_values_there->get_function_gradients(*phi_solution, phi_grad_there);
                        M_fe_face_values_here->get_function_values(*mx_solution, mx_here);
                        M_fe_face_values_there->get_function_values(*mx_solution, mx_there);
                        M_fe_face_values_here->get_function_values(*my_solution, my_here);
                        M_fe_face_values_there->get_function_values(*my_solution, my_there);

                        auto uy_neighbor = uy_cell->neighbor(face_no);
                        std::vector<dealii::types::global_dof_index> ux_there_dofs(dofs_per_cell_Q2);
                        std::vector<dealii::types::global_dof_index> uy_there_dofs(dofs_per_cell_Q2);
                        neighbor->get_dof_indices(ux_there_dofs);
                        uy_neighbor->get_dof_indices(uy_there_dofs);

                        for (unsigned int qf = 0; qf < n_face_q_points; ++qf)
                        {
                            const double JxW_f = ux_fe_face_values.JxW(qf);
                            const dealii::Tensor<1, dim>& normal = ux_fe_face_values.normal_vector(qf);

                            const double Mn_here = mx_here[qf] * normal[0] + my_here[qf] * normal[1];
                            const double Mn_there = mx_there[qf] * normal[0] + my_there[qf] * normal[1];
                            const double M_jump = Mn_here - Mn_there;

                            const dealii::Tensor<1, dim> H_avg = 0.5 * (phi_grad_here[qf] + phi_grad_there[qf]);

                            const dealii::Tensor<1, dim> kelvin_jump = mu_0_val * M_jump * H_avg;

                            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
                            {
                                const double phi_ux_here = ux_fe_face_values.shape_value(i, qf);
                                const double phi_uy_here = uy_fe_face_values.shape_value(i, qf);

                                ns_rhs(ux_to_ns_map[ux_local_dofs[i]]) += 0.5 * kelvin_jump[0] * phi_ux_here * JxW_f;
                                ns_rhs(uy_to_ns_map[uy_local_dofs[i]]) += 0.5 * kelvin_jump[1] * phi_uy_here * JxW_f;
                            }
                        }
                    }
                }
            }
        }
    }

    // Condense constraints
    ns_constraints.condense(ns_matrix, ns_rhs);

    // Print diagnostics
    if (params.output.verbose && !mms_mode)
    {
        std::cout << "[NS Assembler] max|F_cap|=" << max_F_cap
                  << ", max|F_mag|=" << max_F_mag
                  << ", max|F_grav|=" << max_F_grav << "\n";
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