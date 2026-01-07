// ============================================================================
// assembly/ns_mms_assembler.h - Navier-Stokes MMS Verification Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e-42f (discrete scheme)
//
// This is a MINIMAL assembler for MMS verification of the NS equations.
// It builds up complexity in phases:
//
//   Phase A: Steady Stokes (no time, no convection)
//            ν(T(U), T(V)) - (p, ∇·V) + (∇·U, q) = (f, V)
//
//   Phase B: Unsteady Stokes (add time derivative)
//            (U^n/τ, V) + ν(T(U), T(V)) - (p, ∇·V) + (∇·U, q) = (U^{n-1}/τ + f, V)
//
//   Phase C: Steady NS (add convection, no time)
//            B_h(U_old, U, V) + ν(T(U), T(V)) - (p, ∇·V) + (∇·U, q) = (f, V)
//
//   Phase D: Unsteady NS (full equation, matches production ns_assembler)
//            (U^n/τ, V) + B_h(U^{n-1}, U^n, V) + ν(T(U^n), T(V))
//              - (p^n, ∇·V) + (∇·U^n, q) = (U^{n-1}/τ + f, V)
//
// where:
//   T(U) = ∇U + (∇U)^T (symmetric gradient)
//   B_h(w,u,v) = (w·∇u, v) + ½(∇·w)(u, v) (skew-symmetric convection, Eq. 37)
//
// ============================================================================
#ifndef NS_MMS_ASSEMBLER_H
#define NS_MMS_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <vector>

/**
 * @brief Assemble the Navier-Stokes MMS verification system
 *
 * Assembles the coupled (ux, uy, p) system for MMS verification.
 * Mode flags control which terms are included:
 *   - include_time_derivative: adds (U/τ, V) mass term
 *   - include_convection: adds B_h(U_old, U, V) skew convection
 *
 * @param ux_dof_handler    DoFHandler for velocity x (Q2)
 * @param uy_dof_handler    DoFHandler for velocity y (Q2)
 * @param p_dof_handler     DoFHandler for pressure (Q1)
 * @param ux_old            Previous velocity U^{k-1}_x (for time derivative and convection)
 * @param uy_old            Previous velocity U^{k-1}_y (for time derivative and convection)
 * @param nu                Kinematic viscosity (constant for MMS)
 * @param dt                Time step τ (ignored if include_time_derivative=false)
 * @param current_time      Current time t^n (for MMS source evaluation)
 * @param L_y               Domain height (for MMS exact solution)
 * @param include_time_derivative  If true, include (U/τ, V) term
 * @param include_convection       If true, include B_h(U_old, U, V) term
 * @param ux_to_ns_map      Index map: ux DoF → coupled index
 * @param uy_to_ns_map      Index map: uy DoF → coupled index
 * @param p_to_ns_map       Index map: p DoF → coupled index
 * @param ns_constraints    Combined constraints (hanging nodes + BCs)
 * @param ns_matrix         [OUT] Assembled system matrix
 * @param ns_rhs            [OUT] Assembled RHS vector
 */
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
    dealii::Vector<double>& ns_rhs);

// ============================================================================
// MMS Source Term Functions
// ============================================================================

/**
 * @brief Compute MMS source for steady Stokes
 *
 * f = -2ν∇²U + ∇p
 *
 * Evaluated at specified time (use t=1 for steady case).
 */
template <int dim>
dealii::Tensor<1, dim> compute_steady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y);

/**
 * @brief Compute MMS source for unsteady Stokes
 *
 * f = ∂U/∂t - 2ν∇²U + ∇p
 *
 * Uses continuous time derivative.
 */
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y);

/**
 * @brief Compute MMS source for steady NS
 *
 * f = (U·∇)U - 2ν∇²U + ∇p
 *
 * Note: Skew term ½(∇·U)U = 0 for divergence-free exact solution.
 */
template <int dim>
dealii::Tensor<1, dim> compute_steady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y);

/**
 * @brief Compute MMS source for unsteady NS (semi-implicit discretization)
 *
 * f = (U^n - U^{n-1})/τ + (U^{n-1}·∇)U^n - 2ν∇²U^n + ∇p^n
 *
 * This matches the discrete scheme in ns_assembler.cc.
 * Note: Skew term ½(∇·U^{n-1})U^n = 0 for divergence-free exact solution.
 */
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double L_y);

#endif // NS_MMS_ASSEMBLER_H