// ============================================================================
// assembly/ns_assembler.cc - Parallel Navier-Stokes Assembler (Production)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e-42f (discrete scheme)
//
// PRODUCTION assembler with:
//   - Skew-symmetric convection via skew_forms.h (Eq. 37)
//   - Kelvin force μ₀(M·∇)H via kelvin_force.h (Eq. 38)
//   - Optional MMS support via enable_mms flag
//
// FIX: Kelvin force now uses total field H = h_a + h_d (not just h_d = ∇φ)
// FIX: Added missing face term to Kelvin force assembly (Eq. 38)
//
// ============================================================================

#include "assembly/ns_assembler.h"
#include "utilities/parameters.h"

// Skew-symmetric forms (Eq. 37 for convection)
#include "physics/skew_forms.h"

// Kelvin force helpers (Eq. 38)
#include "physics/kelvin_force.h"
#include <deal.II/dofs/dof_tools.h>
#include "physics/applied_field.h"
#include "physics/material_properties.h"

// MMS source terms - only used when enable_mms=true
#include "mms/ns/ns_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/full_matrix.h>


#include <cmath>
#include <memory>

// ============================================================================
// Helper: Compute symmetric gradient T(U) = ∇U + (∇U)^T
// ============================================================================
template <int dim>
static dealii::SymmetricTensor<2, dim> compute_symmetric_gradient(
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

// Helper: T(V) for test function V = (φ_ux, 0)
template <int dim>
static dealii::SymmetricTensor<2, dim> compute_T_test_ux(
    const dealii::Tensor<1, dim>& grad_phi_ux)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 2.0 * grad_phi_ux[0];
    T[1][1] = 0.0;
    T[0][1] = grad_phi_ux[1];
    return T;
}

// Helper: T(V) for test function V = (0, φ_uy)
template <int dim>
static dealii::SymmetricTensor<2, dim> compute_T_test_uy(
    const dealii::Tensor<1, dim>& grad_phi_uy)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::SymmetricTensor<2, dim> T;
    T[0][0] = 0.0;
    T[1][1] = 2.0 * grad_phi_uy[1];
    T[0][1] = grad_phi_uy[0];
    return T;
}

// ============================================================================
// Internal assembly - core NS without Kelvin force
//
// Variable viscosity support:
//   If theta_dof_handler is non-null, computes ν(θ) = ν_water + (ν_ferro - ν_water)·H(θ/ε)
//   Otherwise uses constant nu parameter.
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
    double body_force_time,
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y,
    // Variable viscosity (optional - pass nullptr for constant nu)
    const dealii::DoFHandler<dim>* theta_dof_handler = nullptr,
    const dealii::TrilinosWrappers::MPI::Vector* theta_solution = nullptr,
    double epsilon = 0.01,
    double nu_water = 1.0,
    double nu_ferro = 2.0,
    double r_density = 0.0,
    bool mms_analytical_dt = false)
{
    ns_matrix = 0;
    ns_rhs = 0;

    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_p = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p = fe_p.n_dofs_per_cell();

    // Quadrature (degree + 2 for sufficient accuracy)
    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> p_fe_values(fe_p, quadrature,
        dealii::update_values);

    // Local matrices and vectors
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

    // Pre-allocated coupled DoF index mappings
    std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> coupled_p_dofs(dofs_per_cell_p);

    // Quadrature point values
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);

    // Variable viscosity support
    const bool use_variable_viscosity = (theta_dof_handler != nullptr) && (theta_solution != nullptr);
    std::unique_ptr<dealii::FEValues<dim>> theta_fe_values_ptr;
    std::vector<double> theta_values(n_q_points);

    if (use_variable_viscosity)
    {
        theta_fe_values_ptr = std::make_unique<dealii::FEValues<dim>>(
            theta_dof_handler->get_fe(), quadrature, dealii::update_values);
    }

    // Cell loop
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    // Theta cell iterator for variable viscosity (only used if enabled)
    typename dealii::DoFHandler<dim>::active_cell_iterator theta_cell;
    if (use_variable_viscosity)
        theta_cell = theta_dof_handler->begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
        {
            if (use_variable_viscosity) ++theta_cell;
            continue;
        }

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        // Variable viscosity: reinit theta FEValues and get values
        if (use_variable_viscosity)
        {
            theta_fe_values_ptr->reinit(theta_cell);
            theta_fe_values_ptr->get_function_values(*theta_solution, theta_values);
        }

        // Zero local contributions
        local_ux_ux = 0; local_ux_uy = 0; local_ux_p = 0;
        local_uy_ux = 0; local_uy_uy = 0; local_uy_p = 0;
        local_p_ux = 0;  local_p_uy = 0;
        local_rhs_ux = 0; local_rhs_uy = 0; local_rhs_p = 0;

        // Get DoF indices
        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get old velocity values
        ux_fe_values.get_function_values(ux_old, ux_old_values);
        uy_fe_values.get_function_values(uy_old, uy_old_values);

        if (include_convection)
        {
            ux_fe_values.get_function_gradients(ux_old, ux_old_gradients);
            uy_fe_values.get_function_gradients(uy_old, uy_old_gradients);
        }

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Old velocity at quadrature point
            const double ux_old_q = ux_old_values[q];
            const double uy_old_q = uy_old_values[q];
            dealii::Tensor<1, dim> U_old;
            U_old[0] = ux_old_q;
            U_old[1] = uy_old_q;

            // Divergence of U_old (for skew-symmetric convection)
            double div_U_old = 0.0;
            if (include_convection)
                div_U_old = ux_old_gradients[q][0] + uy_old_gradients[q][1];

            // Variable viscosity: ν(θ) = ν_water + (ν_ferro - ν_water)·H(θ/ε)
            // Paper Eq. 17
            double nu_q = nu;  // Default to constant
            double rho_q = 1.0;  // Default to constant density
            if (use_variable_viscosity)
            {
                const double theta_q = theta_values[q];
                const double H_theta = heaviside(theta_q / epsilon);
                nu_q = nu_water + (nu_ferro - nu_water) * H_theta;
                // Variable density: ρ(θ) = 1 + r·H(θ/ε)  (Paper Eq. 17)
                rho_q = 1.0 + r_density * H_theta;
            }

            // Mass coefficient: ρ(θ^{k-1})/Δt (Paper Eq. 42e with Eq 19)
            // ρ(θ) = 1 + r·H(θ/ε), where r is the density ratio.
            const double mass_coeff = rho_q / dt;

            // ================================================================
            // Compute source terms
            // ================================================================
            dealii::Tensor<1, dim> F_source;
            F_source = 0;

            // Body force (if provided)
            if (body_force != nullptr)
                F_source += (*body_force)(x_q, body_force_time);

            // MMS source (if enabled)
            if (enable_mms)
            {
                if (!include_time_derivative && !include_convection)
                    F_source += compute_steady_stokes_mms_source<dim>(x_q, mms_time, nu, mms_L_y);
                else if (include_time_derivative && !include_convection)
                    F_source += compute_unsteady_stokes_mms_source<dim>(x_q, mms_time, nu, mms_L_y);
                else if (!include_time_derivative && include_convection)
                    F_source += compute_steady_ns_mms_source<dim>(x_q, mms_time, nu, mms_L_y);
                else
                    F_source += compute_unsteady_ns_mms_source<dim>(
                        x_q, mms_time, mms_time_old, nu, mms_L_y, mms_analytical_dt);
            }

            // ================================================================
            // Assemble local matrices
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
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
                // Source term: (f, V)
                local_rhs_ux(i) += F_source[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += F_source[1] * phi_uy_i * JxW;

                // Time derivative RHS: mass_coeff * (U^{n-1}, V)
                // Paper Eq. 42e: ½ρ(θ^{k-1})/Δt when variable density, 1/Δt otherwise
                if (include_time_derivative)
                {
                    local_rhs_ux(i) += mass_coeff * ux_old_q * phi_ux_i * JxW;
                    local_rhs_uy(i) += mass_coeff * uy_old_q * phi_uy_i * JxW;
                }

                // ============================================================
                // LHS contributions
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    auto T_U_x = compute_T_test_ux<dim>(grad_phi_ux_j);
                    auto T_U_y = compute_T_test_uy<dim>(grad_phi_uy_j);

                    // Time derivative LHS: mass_coeff * (U^k, V)
                    // Paper Eq. 42e: ½ρ(θ^{k-1})/Δt when variable density, 1/Δt otherwise
                    if (include_time_derivative)
                    {
                        local_ux_ux(i, j) += mass_coeff * phi_ux_j * phi_ux_i * JxW;
                        local_uy_uy(i, j) += mass_coeff * phi_uy_j * phi_uy_i * JxW;
                    }

                    // Viscous term: (ν_θ T(U), T(V)) where T = ½(∇u + (∇u)^T)
                    // Paper Eq. 42e: bilinear form is ν(T,T).
                    // Code's "T" helper returns D = ∇u + (∇u)^T = 2T_paper.
                    // So (ν/4)(D,D) = ν(T_paper, T_paper) matches the paper.
                    local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
                    local_uy_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_y) * JxW;
                    local_ux_uy(i, j) += (nu_q / 4.0) * (T_U_y * T_V_x) * JxW;
                    local_uy_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_y) * JxW;

                    // --------------------------------------------------------
                    // Convection: B_h(U^{n-1}, U, V) using skew_forms.h (Eq. 37)
                    //   = (U^{n-1}·∇U, V) + ½(∇·U^{n-1})(U, V)
                    // --------------------------------------------------------
                    if (include_convection)
                    {
                        // For componentwise: U = (φ_ux_j, 0), V = (φ_ux_i, 0)
                        // grad_U for this trial: [[∂φ_ux_j/∂x, ∂φ_ux_j/∂y], [0, 0]]
                        // Using scalar version: (U_old·∇φ_ux_j)·φ_ux_i + 0.5*div_U_old*φ_ux_j*φ_ux_i
                        const double convect_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        const double convect_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_uy_j, grad_phi_uy_j, phi_uy_i);

                        local_ux_ux(i, j) += convect_ux * JxW;
                        local_uy_uy(i, j) += convect_uy * JxW;
                    }
                }

                // ============================================================
                // Pressure gradient: -(p, ∇·V)
                // ============================================================
                for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);
                    local_ux_p(i, j) -= phi_p_j * grad_phi_ux_i[0] * JxW;
                    local_uy_p(i, j) -= phi_p_j * grad_phi_uy_i[1] * JxW;
                }
            }

            // ================================================================
            // Continuity equation: (∇·U, q) = 0
            // ================================================================
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

        // ====================================================================
        // Distribute to global matrix
        // ====================================================================
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }
        for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
            coupled_p_dofs[i] = p_to_ns_map[p_local_dofs[i]];

        // Distribute all blocks
        ns_constraints.distribute_local_to_global(local_ux_ux, local_rhs_ux, coupled_ux_dofs, ns_matrix, ns_rhs);
        ns_constraints.distribute_local_to_global(local_ux_uy, coupled_ux_dofs, coupled_uy_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_ux_p, coupled_ux_dofs, coupled_p_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_uy_ux, coupled_uy_dofs, coupled_ux_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_uy_uy, local_rhs_uy, coupled_uy_dofs, ns_matrix, ns_rhs);
        ns_constraints.distribute_local_to_global(local_uy_p, coupled_uy_dofs, coupled_p_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_p_ux, coupled_p_dofs, coupled_ux_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_p_uy, coupled_p_dofs, coupled_uy_dofs, ns_matrix);
        ns_constraints.distribute_local_to_global(local_rhs_p, coupled_p_dofs, ns_rhs);

        // Advance theta cell iterator (for variable viscosity)
        if (use_variable_viscosity)
            ++theta_cell;

    }  // end cell loop

    ns_matrix.compress(dealii::VectorOperation::add);
    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Kelvin force assembly — GRADIENT FORM (production)
//
// Uses the identity: χ(H·∇)H = (1/2)|H|²∇χ + χ∇(|H|²/2)
// where ∇(|H|²/2) = Hess(φ)·H since H = ∇φ.
//
// Term 1 (interface): (μ₀/2)|H|²∇χ(θ) = (μ₀/2)|H|²χ'(θ)∇θ
//   - Strong at interface (where ∇θ is large)
//   - Points normal to interface → drives Rosensweig spikes
//
// Term 2 (bulk): μ₀ χ(θ) Hess(φ)·H
//   - Nonzero in ferrofluid bulk
//   - Uses second derivatives (piecewise constant for Q2)
//
// This form is preferred over the DG skew form for physics runs because
// it uses ∇θ (well-resolved on the mesh) rather than relying solely on
// Hess(φ) (poorly resolved for Q2 elements).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016), Eq. 36
// ============================================================================
template <int dim>
static void assemble_kelvin_force_gradient(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double mu_0,
    double chi_0,
    double epsilon,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs)
{
    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_phi = phi_dof_handler.get_fe();
    const auto& fe_theta = theta_dof_handler.get_fe();

    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for velocity test functions
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values);

    // FEValues for φ — need gradients (H = ∇φ) and Hessians (for bulk term)
    dealii::FEValues<dim> phi_fe_values(fe_phi, quadrature,
        dealii::update_gradients | dealii::update_hessians);

    // FEValues for θ — need values (for χ, χ') and gradients (for ∇χ)
    dealii::FEValues<dim> theta_fe_values(fe_theta, quadrature,
        dealii::update_values | dealii::update_gradients);

    // Storage
    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    // Cell loop (synchronized across DoF handlers)
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end();
         ++ux_cell, ++uy_cell, ++phi_cell, ++theta_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        phi_fe_values.get_function_gradients(phi_solution, phi_gradients);
        phi_fe_values.get_function_hessians(phi_solution, phi_hessians);
        theta_fe_values.get_function_values(theta_solution, theta_values);
        theta_fe_values.get_function_gradients(theta_solution, theta_gradients);

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);

            // H = ∇φ (total magnetic field)
            const dealii::Tensor<1, dim>& H = phi_gradients[q];
            const double H_sq = H * H;  // |H|²

            const double theta_q = theta_values[q];

            // Term 1 (interface): (μ₀/2)|H|² ∇χ(θ) = (μ₀/2)|H|² χ'(θ) ∇θ
            const double chi_prime = chi_0 * heaviside_derivative(theta_q / epsilon) / epsilon;
            const dealii::Tensor<1, dim>& grad_theta = theta_gradients[q];
            const double interface_coeff = 0.5 * mu_0 * H_sq * chi_prime;

            // Term 2 (bulk): μ₀ χ(θ) ∇(|H|²/2) = μ₀ χ(θ) Hess(φ)·H
            const double chi_val = susceptibility(theta_q, epsilon, chi_0);
            const dealii::Tensor<2, dim>& Hess_phi = phi_hessians[q];
            dealii::Tensor<1, dim> Hess_phi_dot_H;
            for (unsigned int d1 = 0; d1 < dim; ++d1)
                for (unsigned int d2 = 0; d2 < dim; ++d2)
                    Hess_phi_dot_H[d1] += Hess_phi[d1][d2] * H[d2];
            const double bulk_coeff = mu_0 * chi_val;

            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);

                // f_x = interface + bulk
                local_rhs_ux(i) += (interface_coeff * grad_theta[0]
                                    + bulk_coeff * Hess_phi_dot_H[0])
                                   * phi_ux_i * JxW;
                // f_y = interface + bulk
                local_rhs_uy(i) += (interface_coeff * grad_theta[1]
                                    + bulk_coeff * Hess_phi_dot_H[1])
                                   * phi_uy_i * JxW;
            }
        }

        // Distribute to global RHS
        std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }

        ns_constraints.distribute_local_to_global(local_rhs_ux, coupled_ux_dofs, ns_rhs);
        ns_constraints.distribute_local_to_global(local_rhs_uy, coupled_uy_dofs, ns_rhs);
    }

    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Kelvin force assembly — DG SKEW FORM (used by MMS tests)
//
// DG skew form: μ₀ B_h^m(V, H, M) = (M·∇)H·V + ½ div(V)(H·M)
// Kept for MMS tests where theta is uniform (gradient form interface term = 0).
//
// H = ∇φ (total magnetic field from Poisson solve)
// The Poisson equation encodes h_a via the RHS, so ∇φ IS the total field.
// ============================================================================
template <int dim>
static void assemble_kelvin_force(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    const Parameters& params,
    double current_time)
{
    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_phi = phi_dof_handler.get_fe();
    const auto& fe_M = M_dof_handler.get_fe();

    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    // Create ghosted copies of solution vectors for face loop.
    // The face loop reads from neighbor (ghost) cells, so we need
    // vectors that include ghost DoF values.
    const MPI_Comm comm = phi_dof_handler.get_communicator();
    const dealii::IndexSet phi_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler);
    const dealii::IndexSet M_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(M_dof_handler);

    dealii::TrilinosWrappers::MPI::Vector phi_ghosted(
        phi_dof_handler.locally_owned_dofs(), phi_relevant, comm);
    phi_ghosted = phi_solution;

    dealii::TrilinosWrappers::MPI::Vector Mx_ghosted(
        M_dof_handler.locally_owned_dofs(), M_relevant, comm);
    Mx_ghosted = Mx_solution;

    dealii::TrilinosWrappers::MPI::Vector My_ghosted(
        M_dof_handler.locally_owned_dofs(), M_relevant, comm);
    My_ghosted = My_solution;

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    const bool use_h_a_only = params.use_h_a_only;

    // FEValues for velocity (test functions) - need gradients for div(V) in B_h^m
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values
        | (use_h_a_only ? dealii::update_quadrature_points : dealii::update_default));
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients);

    // FEValues for φ (need Hessian for (M·∇)H)
    dealii::FEValues<dim> phi_fe_values(fe_phi, quadrature,
        dealii::update_gradients | dealii::update_hessians);

    // FEValues for M (values only)
    dealii::FEValues<dim> M_fe_values(fe_M, quadrature,
        dealii::update_values);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    // Quadrature point storage
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);
    std::vector<double> Mx_values(n_q_points);
    std::vector<double> My_values(n_q_points);

    // Cell loop
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto phi_cell = phi_dof_handler.begin_active();
    auto M_cell = M_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++phi_cell, ++M_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        phi_fe_values.reinit(phi_cell);
        M_fe_values.reinit(M_cell);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        // Get field values at quadrature points (must use ghosted vectors
        // so cells at MPI boundaries see correct values for ghost DoFs)
        phi_fe_values.get_function_gradients(phi_ghosted, phi_gradients);
        phi_fe_values.get_function_hessians(phi_ghosted, phi_hessians);
        M_fe_values.get_function_values(Mx_ghosted, Mx_values);
        M_fe_values.get_function_values(My_ghosted, My_values);

        // ====================================================================
        // Cell term: μ₀ B_h^m(V, H, M) = μ₀[(M·∇)H·V + ½ div(V)(H·M)]
        // Paper Eq. 57 with V (test func) as first arg. See kelvin_force.h.
        // ====================================================================
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);

            // H and (M·∇)H
            dealii::Tensor<1, dim> H;
            dealii::Tensor<1, dim> M_grad_H;
            dealii::Tensor<1, dim> M = KelvinForce::make_M_vector<dim>(Mx_values[q], My_values[q]);

            if (use_h_a_only)
            {
                // H = h_a (Paper Section 5, Eq 66)
                const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);
                H = compute_applied_field<dim>(x_q, params, current_time);
                // (M·∇)h_a via central differences
                const double fd_eps = 1e-7;
                for (unsigned int d = 0; d < dim; ++d)
                {
                    dealii::Point<dim> xp = x_q, xm = x_q;
                    xp[d] += fd_eps;
                    xm[d] -= fd_eps;
                    auto Hp = compute_applied_field<dim>(xp, params, current_time);
                    auto Hm = compute_applied_field<dim>(xm, params, current_time);
                    M_grad_H += M[d] * (Hp - Hm) / (2.0 * fd_eps);
                }
            }
            else
            {
                // H = ∇φ (total field from Poisson solve, includes h_a + h_d)
                H[0] = phi_gradients[q][0];
                H[1] = phi_gradients[q][1];
                M_grad_H = KelvinForce::compute_M_grad_H<dim>(M, phi_hessians[q]);
            }

            // Assemble RHS contributions using test function gradients
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim>& grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim>& grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                double kelvin_ux, kelvin_uy;
                KelvinForce::cell_kernel<dim>(M_grad_H, H, M, phi_ux_i, phi_uy_i,
                                               grad_phi_ux_i, grad_phi_uy_i,
                                               mu_0, kelvin_ux, kelvin_uy);

                local_rhs_ux(i) += kelvin_ux * JxW;
                local_rhs_uy(i) += kelvin_uy * JxW;
            }
        }

        // Distribute to global RHS
        std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }

        ns_constraints.distribute_local_to_global(local_rhs_ux, coupled_ux_dofs, ns_rhs);
        ns_constraints.distribute_local_to_global(local_rhs_uy, coupled_uy_dofs, ns_rhs);

    }  // end cell loop

    // ========================================================================
    // Face loop: -μ₀ ([[H]]·{M})(V·n)   (Paper Eq. 57 face term)
    //
    // H = ∇φ is the gradient of CG φ — DISCONTINUOUS across element faces.
    // M is DG — also discontinuous. V is CG velocity test function.
    //
    // Process each interior face ONCE (from the minus-side cell).
    // V is CG so only minus-cell test functions needed (no CG double-counting).
    //
    // AMR handling: 3 cases per face (same as magnetic assembler):
    //   1. Same-level neighbor: standard face integral
    //   2. Finer neighbor (face->has_children): subface loop
    //   3. Coarser neighbor: skip (coarser cell handles via case 2)
    // ========================================================================
    if (!use_h_a_only)  // Skip face loop when H = h_a (smooth, [[H]] = 0)
    {
        dealii::QGauss<dim - 1> face_quadrature(fe_vel.degree + 2);
        const unsigned int n_face_q = face_quadrature.size();

        // FEFaceValues for velocity test functions (both sides — CG test funcs
        // on both cells have support at the face and must be assembled)
        dealii::FEFaceValues<dim> ux_fe_face_minus(fe_vel, face_quadrature,
            dealii::update_values | dealii::update_normal_vectors |
            dealii::update_JxW_values);
        dealii::FEFaceValues<dim> uy_fe_face_minus(fe_vel, face_quadrature,
            dealii::update_values);
        dealii::FEFaceValues<dim> ux_fe_face_plus(fe_vel, face_quadrature,
            dealii::update_values);
        dealii::FEFaceValues<dim> uy_fe_face_plus(fe_vel, face_quadrature,
            dealii::update_values);

        // FEFaceValues for φ (gradients on both sides for [[H]])
        dealii::FEFaceValues<dim> phi_fe_face_minus(fe_phi, face_quadrature,
            dealii::update_gradients);
        dealii::FEFaceValues<dim> phi_fe_face_plus(fe_phi, face_quadrature,
            dealii::update_gradients);

        // FEFaceValues for M (values on both sides for {M})
        dealii::FEFaceValues<dim> M_fe_face_minus(fe_M, face_quadrature,
            dealii::update_values);
        dealii::FEFaceValues<dim> M_fe_face_plus(fe_M, face_quadrature,
            dealii::update_values);

        // FESubfaceValues for AMR: coarser cell side when neighbor is finer
        dealii::FESubfaceValues<dim> ux_fe_subface_minus(fe_vel, face_quadrature,
            dealii::update_values | dealii::update_normal_vectors |
            dealii::update_JxW_values);
        dealii::FESubfaceValues<dim> uy_fe_subface_minus(fe_vel, face_quadrature,
            dealii::update_values);
        dealii::FESubfaceValues<dim> phi_fe_subface_minus(fe_phi, face_quadrature,
            dealii::update_gradients);
        dealii::FESubfaceValues<dim> M_fe_subface_minus(fe_M, face_quadrature,
            dealii::update_values);

        // Face quadrature point storage
        std::vector<dealii::Tensor<1, dim>> phi_grad_minus(n_face_q);
        std::vector<dealii::Tensor<1, dim>> phi_grad_plus(n_face_q);
        std::vector<double> Mx_minus(n_face_q), My_minus(n_face_q);
        std::vector<double> Mx_plus(n_face_q), My_plus(n_face_q);

        // Local RHS for both cells (CG test functions on both sides need assembly)
        dealii::Vector<double> face_rhs_ux_minus(dofs_per_cell_vel);
        dealii::Vector<double> face_rhs_uy_minus(dofs_per_cell_vel);
        dealii::Vector<double> face_rhs_ux_plus(dofs_per_cell_vel);
        dealii::Vector<double> face_rhs_uy_plus(dofs_per_cell_vel);

        std::vector<dealii::types::global_dof_index> ux_dofs_minus(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> uy_dofs_minus(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> ux_dofs_plus(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> uy_dofs_plus(dofs_per_cell_vel);

        // Lambda: Kelvin face kernel. Takes FEValuesBase refs for minus/plus sides.
        // CG velocity test functions on BOTH cells must be assembled at each face.
        // The face-fixed quantities ([[H]], {M}, n_minus) are computed once,
        // then used for test functions from both cells.
        auto kelvin_face_kernel = [&](
            const dealii::FEValuesBase<dim>& ux_minus,
            const dealii::FEValuesBase<dim>& uy_minus,
            const dealii::FEValuesBase<dim>& ux_plus_fv,
            const dealii::FEValuesBase<dim>& uy_plus_fv,
            const dealii::FEValuesBase<dim>& phi_minus,
            const dealii::FEValuesBase<dim>& phi_plus_fv,
            const dealii::FEValuesBase<dim>& M_minus_fv,
            const dealii::FEValuesBase<dim>& M_plus_fv)
        {
            // Get field values at face quadrature points
            phi_minus.get_function_gradients(phi_ghosted, phi_grad_minus);
            phi_plus_fv.get_function_gradients(phi_ghosted, phi_grad_plus);
            M_minus_fv.get_function_values(Mx_ghosted, Mx_minus);
            M_minus_fv.get_function_values(My_ghosted, My_minus);
            M_plus_fv.get_function_values(Mx_ghosted, Mx_plus);
            M_plus_fv.get_function_values(My_ghosted, My_plus);

            face_rhs_ux_minus = 0;
            face_rhs_uy_minus = 0;
            face_rhs_ux_plus = 0;
            face_rhs_uy_plus = 0;

            const unsigned int n_q = ux_minus.n_quadrature_points;
            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW_face = ux_minus.JxW(q);
                const dealii::Tensor<1, dim>& normal =
                    ux_minus.normal_vector(q);

                const dealii::Tensor<1, dim>& H_minus_q = phi_grad_minus[q];
                const dealii::Tensor<1, dim>& H_plus_q = phi_grad_plus[q];

                dealii::Tensor<1, dim> M_minus_vec =
                    KelvinForce::make_M_vector<dim>(Mx_minus[q], My_minus[q]);
                dealii::Tensor<1, dim> M_plus_vec =
                    KelvinForce::make_M_vector<dim>(Mx_plus[q], My_plus[q]);

                dealii::Tensor<1, dim> jump_H =
                    KelvinForce::compute_jump_H<dim>(H_minus_q, H_plus_q);
                dealii::Tensor<1, dim> avg_M =
                    KelvinForce::compute_avg_M<dim>(M_minus_vec, M_plus_vec);

                // Minus-cell CG test functions
                for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
                {
                    const double phi_ux = ux_minus.shape_value(i, q);
                    const double phi_uy = uy_minus.shape_value(i, q);

                    double k_ux, k_uy;
                    KelvinForce::face_kernel<dim>(
                        phi_ux, phi_uy, normal, jump_H, avg_M,
                        mu_0, k_ux, k_uy);

                    face_rhs_ux_minus(i) += k_ux * JxW_face;
                    face_rhs_uy_minus(i) += k_uy * JxW_face;
                }

                // Plus-cell CG test functions (same normal, same jump_H, same avg_M)
                for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
                {
                    const double phi_ux = ux_plus_fv.shape_value(i, q);
                    const double phi_uy = uy_plus_fv.shape_value(i, q);

                    double k_ux, k_uy;
                    KelvinForce::face_kernel<dim>(
                        phi_ux, phi_uy, normal, jump_H, avg_M,
                        mu_0, k_ux, k_uy);

                    face_rhs_ux_plus(i) += k_ux * JxW_face;
                    face_rhs_uy_plus(i) += k_uy * JxW_face;
                }
            }

            // Distribute minus cell to global RHS
            std::vector<dealii::types::global_dof_index> coupled_ux_m(dofs_per_cell_vel);
            std::vector<dealii::types::global_dof_index> coupled_uy_m(dofs_per_cell_vel);
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                coupled_ux_m[i] = ux_to_ns_map[ux_dofs_minus[i]];
                coupled_uy_m[i] = uy_to_ns_map[uy_dofs_minus[i]];
            }
            ns_constraints.distribute_local_to_global(
                face_rhs_ux_minus, coupled_ux_m, ns_rhs);
            ns_constraints.distribute_local_to_global(
                face_rhs_uy_minus, coupled_uy_m, ns_rhs);

            // Distribute plus cell to global RHS
            std::vector<dealii::types::global_dof_index> coupled_ux_p(dofs_per_cell_vel);
            std::vector<dealii::types::global_dof_index> coupled_uy_p(dofs_per_cell_vel);
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                coupled_ux_p[i] = ux_to_ns_map[ux_dofs_plus[i]];
                coupled_uy_p[i] = uy_to_ns_map[uy_dofs_plus[i]];
            }
            ns_constraints.distribute_local_to_global(
                face_rhs_ux_plus, coupled_ux_p, ns_rhs);
            ns_constraints.distribute_local_to_global(
                face_rhs_uy_plus, coupled_uy_p, ns_rhs);
        };

        // Synchronized cell iterators for all DoFHandlers
        auto ux_cell_f = ux_dof_handler.begin_active();
        auto uy_cell_f = uy_dof_handler.begin_active();
        auto phi_cell_f = phi_dof_handler.begin_active();
        auto M_cell_f = M_dof_handler.begin_active();

        for (; ux_cell_f != ux_dof_handler.end();
             ++ux_cell_f, ++uy_cell_f, ++phi_cell_f, ++M_cell_f)
        {
            if (!ux_cell_f->is_locally_owned())
                continue;

            for (unsigned int face_no = 0;
                 face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                if (ux_cell_f->at_boundary(face_no))
                    continue;

                const auto face = ux_cell_f->face(face_no);

                // ==============================================================
                // Case 1: Face has children → neighbor is FINER
                // Use FESubfaceValues on this (coarser) cell.
                // ==============================================================
                if (face->has_children())
                {
                    for (unsigned int subface = 0;
                         subface < face->n_children(); ++subface)
                    {
                        // Get finer neighbor child cells for all DoF handlers
                        const auto ux_child =
                            ux_cell_f->neighbor_child_on_subface(face_no, subface);
                        const auto phi_child =
                            phi_cell_f->neighbor_child_on_subface(face_no, subface);
                        const auto M_child =
                            M_cell_f->neighbor_child_on_subface(face_no, subface);

                        // Find which face of the child touches us
                        unsigned int child_face_no = 0;
                        for (unsigned int f = 0;
                             f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                        {
                            if (!ux_child->at_boundary(f) &&
                                ux_child->neighbor(f) == ux_cell_f)
                            {
                                child_face_no = f;
                                break;
                            }
                        }

                        const auto uy_child =
                            uy_cell_f->neighbor_child_on_subface(face_no, subface);

                        // Reinit: subface on coarser side, face on finer side
                        ux_fe_subface_minus.reinit(ux_cell_f, face_no, subface);
                        uy_fe_subface_minus.reinit(uy_cell_f, face_no, subface);
                        phi_fe_subface_minus.reinit(phi_cell_f, face_no, subface);
                        M_fe_subface_minus.reinit(M_cell_f, face_no, subface);

                        ux_fe_face_plus.reinit(ux_child, child_face_no);
                        uy_fe_face_plus.reinit(uy_child, child_face_no);
                        phi_fe_face_plus.reinit(phi_child, child_face_no);
                        M_fe_face_plus.reinit(M_child, child_face_no);

                        // Get DoF indices for both cells
                        ux_cell_f->get_dof_indices(ux_dofs_minus);
                        uy_cell_f->get_dof_indices(uy_dofs_minus);
                        ux_child->get_dof_indices(ux_dofs_plus);
                        uy_child->get_dof_indices(uy_dofs_plus);

                        kelvin_face_kernel(
                            ux_fe_subface_minus, uy_fe_subface_minus,
                            ux_fe_face_plus, uy_fe_face_plus,
                            phi_fe_subface_minus, phi_fe_face_plus,
                            M_fe_subface_minus, M_fe_face_plus);
                    }
                    continue;
                }

                // ==============================================================
                // Case 2: Neighbor is COARSER → skip
                // The coarser neighbor handles this face via Case 1.
                // ==============================================================
                if (ux_cell_f->neighbor_is_coarser(face_no))
                    continue;

                // ==============================================================
                // Case 3: SAME-LEVEL neighbor
                // Process each face once: CG test functions from the minus cell
                // are assembled. The plus cell's test functions at the face
                // give the SAME contribution (normal and jump both flip), so
                // we must also assemble plus-cell test functions here.
                // ==============================================================
                const auto ux_neighbor = ux_cell_f->neighbor(face_no);

                // MPI FIX: Process each face exactly ONCE using global cell
                // index comparison, regardless of MPI ownership. This prevents
                // double-assembly at MPI boundaries where both ranks see the face.
                if (ux_neighbor->level() < ux_cell_f->level() ||
                    (ux_neighbor->level() == ux_cell_f->level() &&
                     ux_neighbor->index() < ux_cell_f->index()))
                    continue;

                const unsigned int neighbor_face_no =
                    ux_cell_f->neighbor_of_neighbor(face_no);

                const auto phi_neighbor = phi_cell_f->neighbor(face_no);
                const auto M_neighbor = M_cell_f->neighbor(face_no);

                const auto uy_neighbor = uy_cell_f->neighbor(face_no);

                // Reinit FEFaceValues on minus side (this cell)
                ux_fe_face_minus.reinit(ux_cell_f, face_no);
                uy_fe_face_minus.reinit(uy_cell_f, face_no);
                phi_fe_face_minus.reinit(phi_cell_f, face_no);
                M_fe_face_minus.reinit(M_cell_f, face_no);

                // Reinit FEFaceValues on plus side (neighbor) — fields AND velocity
                ux_fe_face_plus.reinit(ux_neighbor, neighbor_face_no);
                uy_fe_face_plus.reinit(uy_neighbor, neighbor_face_no);
                phi_fe_face_plus.reinit(phi_neighbor, neighbor_face_no);
                M_fe_face_plus.reinit(M_neighbor, neighbor_face_no);

                // Get DoF indices for both cells
                ux_cell_f->get_dof_indices(ux_dofs_minus);
                uy_cell_f->get_dof_indices(uy_dofs_minus);
                ux_neighbor->get_dof_indices(ux_dofs_plus);
                uy_neighbor->get_dof_indices(uy_dofs_plus);

                kelvin_face_kernel(
                    ux_fe_face_minus, uy_fe_face_minus,
                    ux_fe_face_plus, uy_fe_face_plus,
                    phi_fe_face_minus, phi_fe_face_plus,
                    M_fe_face_minus, M_fe_face_plus);

            }  // end face loop
        }  // end cell loop (face pass)
    }

    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Capillary Force Assembly: -(λ/ε) θ∇ψ_code (Paper Eq. 42e: +(λ/ε)θ∇Ψ)
//
// Paper Eq. 1: Ψ = εΔθ - (1/ε)f(θ).
// Code solves: ψ_code = -εΔθ + (1/ε)f(θ) = -Ψ_paper.
// So: +(λ/ε)θ∇Ψ_paper = -(λ/ε)θ∇ψ_code.
//
// Unlike Kelvin force, this has NO face terms since θ and ψ are continuous.
// ============================================================================
template <int dim>
static void assemble_capillary_force(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double lambda,
    double epsilon,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs)
{
    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_theta = theta_dof_handler.get_fe();
    const auto& fe_psi = psi_dof_handler.get_fe();

    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for velocity (test functions)
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values);

    // FEValues for θ (need values)
    dealii::FEValues<dim> theta_fe_values(fe_theta, quadrature,
        dealii::update_values);

    // FEValues for ψ (need gradients)
    dealii::FEValues<dim> psi_fe_values(fe_psi, quadrature,
        dealii::update_gradients);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    // Quadrature point storage
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);

    // Capillary coefficient: -λ/ε  (Paper Eq. 42e: +(λ/ε)(θ∇Ψ,V))
    // Paper Eq. 1: Ψ = εΔθ - (1/ε)f(θ).
    // Code solves: (ψ,Υ) - ε(∇θ,∇Υ) - (1/ε)(f,Υ) = 0 → ψ_code = -εΔθ + (1/ε)f = -Ψ_paper.
    // So ∇Ψ_paper = -∇ψ_code, giving -(λ/ε)(θ∇ψ_code, V).
    const double capillary_coeff = -lambda / epsilon;

    // Cell loop
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++theta_cell, ++psi_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        // Get field values at quadrature points
        theta_fe_values.get_function_values(theta_solution, theta_values);
        psi_fe_values.get_function_gradients(psi_solution, psi_gradients);

        // ====================================================================
        // Cell term: -λ·θ·∇ψ · V  (capillary force, ψ_code = μ/λ)
        // ====================================================================
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const double theta_q = theta_values[q];
            const dealii::Tensor<1, dim>& grad_psi_q = psi_gradients[q];

            // Capillary force: F_cap = -λ·θ·∇ψ
            const double F_cap_x = capillary_coeff * theta_q * grad_psi_q[0];
            const double F_cap_y = capillary_coeff * theta_q * grad_psi_q[1];

            // Add to RHS: (F_cap, V) where V = (φ_ux, 0) or (0, φ_uy)
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);

                local_rhs_ux(i) += F_cap_x * phi_ux_i * JxW;
                local_rhs_uy(i) += F_cap_y * phi_uy_i * JxW;
            }
        }

        // Distribute to global RHS (use vector form for correct constraint handling)
        std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }

        ns_constraints.distribute_local_to_global(local_rhs_ux, coupled_ux_dofs, ns_rhs);
        ns_constraints.distribute_local_to_global(local_rhs_uy, coupled_uy_dofs, ns_rhs);

    }  // end cell loop

    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Gravity Force Assembly: r·H(θ/ε)·g (Paper Eq. 19 - Boussinesq approximation)
//
// This is the buoyancy force that couples the phase field (θ) to momentum.
// The density varies between phases: ρ(θ) = ρ_0(1 + r·H(θ/ε))
//
// Unlike capillary force, this uses Heaviside of θ/ε, not θ directly.
// ============================================================================
template <int dim>
static void assemble_gravity_force(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double r,           // density ratio
    double epsilon,     // interface thickness
    double gravity_mag, // gravity magnitude (e.g., 30000)
    const dealii::Tensor<1, dim>& gravity_dir,  // gravity direction (e.g., (0,-1))
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs)
{
    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_theta = theta_dof_handler.get_fe();

    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for velocity (test functions)
    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values);

    // FEValues for θ (need values)
    dealii::FEValues<dim> theta_fe_values(fe_theta, quadrature,
        dealii::update_values);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    // Quadrature point storage
    std::vector<double> theta_values(n_q_points);

    // Gravity vector: g = gravity_mag * gravity_dir
    dealii::Tensor<1, dim> g = gravity_mag * gravity_dir;

    // Cell loop
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++theta_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        theta_fe_values.reinit(theta_cell);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        // Get θ values at quadrature points
        theta_fe_values.get_function_values(theta_solution, theta_values);

        // ====================================================================
        // Cell term: r·H(θ/ε)·g · V
        // ====================================================================
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const double theta_q = theta_values[q];

            const double H_theta = heaviside(theta_q / epsilon);

            // Gravity force: F_g = r·H(θ/ε)·g
            const double density_factor = r * H_theta;
            const double F_g_x = density_factor * g[0];
            const double F_g_y = density_factor * g[1];

            // Add to RHS: (F_g, V)
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);

                local_rhs_ux(i) += F_g_x * phi_ux_i * JxW;
                local_rhs_uy(i) += F_g_y * phi_uy_i * JxW;
            }
        }

        // Distribute to global RHS (use vector form for correct constraint handling)
        std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }

        ns_constraints.distribute_local_to_global(local_rhs_ux, coupled_ux_dofs, ns_rhs);
        ns_constraints.distribute_local_to_global(local_rhs_uy, coupled_uy_dofs, ns_rhs);

    }  // end cell loop

    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Helper: Subtract exact Kelvin force for MMS correction
//
// When testing NS + Kelvin coupling with MMS:
//   - assemble_ns_core adds: f_NS (standalone MMS source)
//   - assemble_kelvin_force adds: F_K(M_h, H_h) (numerical Kelvin force)
//
// For convergence, we need: f_NS - F_K(M*, H*) + F_K(M_h, H_h)
// So we must SUBTRACT the exact Kelvin force F_K(M*, H*).
//
// As h->0: F_K(M_h, H_h) -> F_K(M*, H*), so total -> f_NS
// ============================================================================
template <int dim>
static void assemble_kelvin_mms_correction(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    double mu_0,
    double mms_time,
    double mms_L_y)
{
    const auto& fe_vel = ux_dof_handler.get_fe();
    const unsigned int dofs_per_cell_vel = fe_vel.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(fe_vel, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_vel, quadrature,
        dealii::update_values);

    dealii::Vector<double> local_rhs_ux(dofs_per_cell_vel);
    dealii::Vector<double> local_rhs_uy(dofs_per_cell_vel);

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_vel);

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        local_rhs_ux = 0;
        local_rhs_uy = 0;

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Exact Kelvin force F_K(M*, H*) at this quadrature point
            dealii::Tensor<1, dim> F_K_exact =
                compute_kelvin_force_mms_source<dim>(x_q, mms_time, mu_0, mms_L_y);

            // SUBTRACT from RHS: we need (f_NS - F_K_exact), and f_NS was already added
            for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);

                local_rhs_ux(i) -= F_K_exact[0] * phi_ux_i * JxW;
                local_rhs_uy(i) -= F_K_exact[1] * phi_uy_i * JxW;
            }
        }

        // Map to coupled system and distribute
        std::vector<dealii::types::global_dof_index> coupled_ux_dofs(dofs_per_cell_vel);
        std::vector<dealii::types::global_dof_index> coupled_uy_dofs(dofs_per_cell_vel);
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            coupled_ux_dofs[i] = ux_to_ns_map[ux_local_dofs[i]];
            coupled_uy_dofs[i] = uy_to_ns_map[uy_local_dofs[i]];
        }

        ns_constraints.distribute_local_to_global(local_rhs_ux, coupled_ux_dofs, ns_rhs);
        ns_constraints.distribute_local_to_global(local_rhs_uy, coupled_uy_dofs, ns_rhs);
    }

    ns_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Public API: Unified NS assembly function
//
// Single entry point for all NS assembly variants. Calls internal helpers
// based on what forces are enabled in the NSForceData struct.
//
// Execution order:
//   1. assemble_ns_core() — LHS matrix + core RHS (time, viscous, convection,
//      pressure, body force, MMS source). Variable viscosity if enabled.
//   2. Kelvin force — either gradient form or DG skew form (Eq. 38)
//      + MMS correction if enable_mms
//   3. Capillary force — (lambda/epsilon) theta grad(psi)
//   4. Gravity force — r*H(theta/epsilon)*g
// ============================================================================
template <int dim>
void assemble_ns_system_unified(
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
    const NSForceData<dim>& forces,
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y,
    bool mms_analytical_dt)
{
    // Step 1: Assemble core NS (time, viscous, convection, pressure, body/MMS source)
    // Optionally with variable viscosity nu(theta)
    if (forces.has_variable_viscosity)
    {
        assemble_ns_core<dim>(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_old, uy_old, nu, dt,
            include_time_derivative, include_convection,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_constraints, ns_matrix, ns_rhs,
            forces.body_force, forces.body_force_time,
            enable_mms, mms_time, mms_time_old, mms_L_y,
            forces.theta_visc_dof_handler, forces.theta_visc_solution,
            forces.epsilon_visc, forces.nu_water, forces.nu_ferro,
            forces.r, mms_analytical_dt);
    }
    else
    {
        assemble_ns_core<dim>(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_old, uy_old, nu, dt,
            include_time_derivative, include_convection,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_constraints, ns_matrix, ns_rhs,
            forces.body_force, forces.body_force_time,
            enable_mms, mms_time, mms_time_old, mms_L_y,
            /*theta_dof=*/nullptr, /*theta_sol=*/nullptr,
            /*epsilon=*/0.01, /*nu_water=*/1.0, /*nu_ferro=*/2.0,
            /*r=*/0.0, mms_analytical_dt);
    }

    // Step 2: Add Kelvin force to RHS (if enabled)
    if (forces.has_kelvin)
    {
        const Parameters& kp = *forces.kelvin_params;

        // ALWAYS use the paper's DG skew form: μ₀ B_h^m(V, H, M)
        //   = μ₀[(M·∇)H·V + ½div(V)(H·M)]  (Paper Eq. 42e, 57)
        //
        // WHY NOT the gradient form:
        //   The gradient form computes F = (μ₀/2)|H|²∇χ + μ₀χ Hess(φ)·H
        //   = ∇(μ₀χ|H|²/2), which is a PURE GRADIENT of a scalar.
        //   For incompressible flow with no-slip BCs, gradient forces are
        //   entirely absorbed by pressure and produce ZERO velocity effect.
        //   The Rosensweig instability requires the non-gradient interface
        //   force -(μ₀/2)|H|²∇χ that only B_h^m provides.
        assemble_kelvin_force<dim>(
            ux_dof_handler, uy_dof_handler,
            *forces.phi_dof_handler, *forces.M_dof_handler,
            *forces.phi_solution, *forces.Mx_solution, *forces.My_solution,
            forces.mu_0,
            ux_to_ns_map, uy_to_ns_map,
            ns_constraints, ns_rhs,
            kp, forces.kelvin_time);

        // MMS correction: subtract exact Kelvin force F_K(M*, H*)
        if (enable_mms)
        {
            assemble_kelvin_mms_correction<dim>(
                ux_dof_handler, uy_dof_handler,
                ux_to_ns_map, uy_to_ns_map,
                ns_constraints, ns_rhs,
                forces.mu_0, mms_time, mms_L_y);
        }
    }

    // Step 3: Add capillary force -lambda * theta * grad(psi) to RHS
    if (forces.has_capillary)
    {
        assemble_capillary_force<dim>(
            ux_dof_handler, uy_dof_handler,
            *forces.theta_cap_dof_handler, *forces.psi_dof_handler,
            *forces.theta_cap_solution, *forces.psi_solution,
            forces.lambda, forces.epsilon_cap,
            ux_to_ns_map, uy_to_ns_map,
            ns_constraints, ns_rhs);
    }

    // Step 4: Add gravity force r*H(theta/epsilon)*g to RHS
    if (forces.has_gravity)
    {
        // Use the capillary theta dof handler for gravity as well
        // (gravity uses theta for Heaviside density interpolation)
        assemble_gravity_force<dim>(
            ux_dof_handler, uy_dof_handler,
            *forces.theta_cap_dof_handler, *forces.theta_cap_solution,
            forces.r, forces.epsilon_cap, forces.gravity_mag, forces.gravity_dir,
            ux_to_ns_map, uy_to_ns_map,
            ns_constraints, ns_rhs);
    }
}

// ============================================================================
// Legacy wrapper: Basic NS assembly (no forces)
// ============================================================================
template <int dim>
void assemble_ns_system_parallel(
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
    const dealii::IndexSet& /*ns_owned*/,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm /*mpi_comm*/,
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y)
{
    assemble_ns_system_unified<dim>(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_old, uy_old, nu, dt,
        include_time_derivative, include_convection,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_constraints, ns_matrix, ns_rhs,
        NSForceData<dim>{},  // no forces
        enable_mms, mms_time, mms_time_old, mms_L_y);
}

// ============================================================================
// Legacy wrapper: NS assembly with body force
// ============================================================================
template <int dim>
void assemble_ns_system_with_body_force_parallel(
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
    const dealii::IndexSet& /*ns_owned*/,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm /*mpi_comm*/,
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>& body_force,
    double current_time,
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y)
{
    NSForceData<dim> forces;
    forces.set_body_force(&body_force, current_time);

    assemble_ns_system_unified<dim>(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_old, uy_old, nu, dt,
        include_time_derivative, include_convection,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_constraints, ns_matrix, ns_rhs,
        forces,
        enable_mms, mms_time, mms_time_old, mms_L_y);
}

// ============================================================================
// Legacy wrapper: NS assembly with Kelvin force only (used by MMS tests)
//
// NOTE: Uses H = grad(phi) only (no applied field h_a).
// Default Parameters has empty dipoles => h_a = 0, use_reduced = false => H = grad(phi).
// ============================================================================
template <int dim>
void assemble_ns_system_with_kelvin_force_parallel(
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
    const dealii::IndexSet& /*ns_owned*/,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm /*mpi_comm*/,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y,
    bool mms_analytical_dt)
{
    // Create default params for Kelvin force (empty dipoles => H = grad(phi))
    Parameters mms_params;
    mms_params.enable_mms = enable_mms;

    NSForceData<dim> forces;
    forces.enable_kelvin(phi_dof_handler, M_dof_handler,
                         phi_solution, Mx_solution, My_solution,
                         mu_0, mms_params, 0.0);

    assemble_ns_system_unified<dim>(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_old, uy_old, nu, dt,
        include_time_derivative, include_convection,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_constraints, ns_matrix, ns_rhs,
        forces,
        enable_mms, mms_time, mms_time_old, mms_L_y, mms_analytical_dt);
}

// ============================================================================
// Legacy wrapper: Full ferrofluid NS assembly (Kelvin + Capillary + Gravity)
//
// Paper Eq. 14e RHS: mu_0(M.grad)H + (lambda/epsilon) theta grad(psi) + r*H(theta/epsilon)*g
// ============================================================================
template <int dim>
void assemble_ns_system_ferrofluid_parallel(
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
    const dealii::IndexSet& /*ns_owned*/,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm /*mpi_comm*/,
    // Kelvin force inputs (magnetic)
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    // Capillary force inputs (phase field)
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double lambda,
    double epsilon,
    // Variable viscosity inputs (Paper Eq. 17)
    double nu_water,
    double nu_ferro,
    // Gravity force inputs
    bool enable_gravity_flag,
    double r,
    double gravity_mag,
    const dealii::Tensor<1, dim>& gravity_dir,
    // Simulation parameters (for dipole config, MMS, reduced field)
    const Parameters& params,
    double current_time,
    // MMS options
    bool enable_mms,
    double mms_time,
    double mms_time_old,
    double mms_L_y)
{
    NSForceData<dim> forces;

    // Kelvin force
    forces.enable_kelvin(phi_dof_handler, M_dof_handler,
                         phi_solution, Mx_solution, My_solution,
                         mu_0, params, current_time);

    // Capillary force
    forces.enable_capillary(theta_dof_handler, psi_dof_handler,
                            theta_solution, psi_solution,
                            lambda, epsilon);

    // Variable viscosity (uses same theta field as capillary)
    forces.enable_variable_viscosity(theta_dof_handler, theta_solution,
                                     epsilon, nu_water, nu_ferro);

    // Density ratio r is needed for variable density mass matrix (Eq. 42e)
    // even when gravity is disabled, so always set it.
    forces.r = r;

    // Gravity
    if (enable_gravity_flag)
        forces.enable_gravity_force(r, gravity_mag, gravity_dir);

    assemble_ns_system_unified<dim>(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_old, uy_old, nu, dt,
        include_time_derivative, include_convection,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_constraints, ns_matrix, ns_rhs,
        forces,
        enable_mms, mms_time, mms_time_old, mms_L_y);
}

// ============================================================================
// Explicit instantiations
// ============================================================================

// Unified function
template void assemble_ns_system_unified<2>(
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&, dealii::TrilinosWrappers::MPI::Vector&,
    const NSForceData<2>&,
    bool, double, double, double, bool);

// Legacy wrappers
template void assemble_ns_system_parallel<2>(
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::IndexSet&, const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&, dealii::TrilinosWrappers::MPI::Vector&,
    MPI_Comm, bool, double, double, double);

template void assemble_ns_system_with_body_force_parallel<2>(
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::IndexSet&, const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&, dealii::TrilinosWrappers::MPI::Vector&,
    MPI_Comm,
    const std::function<dealii::Tensor<1, 2>(const dealii::Point<2>&, double)>&, double,
    bool, double, double, double);

template void assemble_ns_system_with_kelvin_force_parallel<2>(
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::IndexSet&, const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&, dealii::TrilinosWrappers::MPI::Vector&,
    MPI_Comm,
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, bool, double, double, double, bool);

template void assemble_ns_system_ferrofluid_parallel<2>(
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool, bool,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    const dealii::IndexSet&, const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&, dealii::TrilinosWrappers::MPI::Vector&,
    MPI_Comm,
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    double,
    const dealii::DoFHandler<2>&, const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, double,
    double, double,
    bool, double, double, const dealii::Tensor<1, 2>&,
    const Parameters&, double,
    bool, double, double, double);