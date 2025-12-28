// ============================================================================
// setup/poisson_setup.h - Magnetostatic Poisson System Setup
//
// Free function for Poisson system setup:
//   - Constraints (hanging nodes + nullspace fix for pure Neumann)
//   - Sparsity pattern
//
// Architecture follows ch_setup.h / ns_setup.h pattern for consistency.
//
// Pure Neumann problem: (μ∇φ, ∇χ) = (h_a - M, ∇χ)
// Requires pinning one DoF to fix the constant (nullspace).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42c
// ============================================================================
#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

/**
 * @brief Set up constraints and sparsity pattern for Poisson system
 *
 * Creates:
 *   - Constraints: hanging nodes + pin DoF 0 (fixes Neumann nullspace)
 *   - Sparsity pattern with constrained DoFs eliminated
 *
 * For pure Neumann problems, the solution is unique only up to a constant.
 * Pinning DoF 0 to zero fixes this constant.
 *
 * Note: Named differently from PhaseFieldProblem::setup_poisson_system()
 * to avoid member/free function collision.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   [OUT] Constraints (hanging nodes + nullspace fix)
 * @param phi_sparsity      [OUT] Sparsity pattern
 * @param verbose           Print setup info
 */
template <int dim>
void setup_poisson_constraints_and_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints,
    dealii::SparsityPattern& phi_sparsity,
    bool verbose = false);

#endif // POISSON_SETUP_H