// ============================================================================
// setup/poisson_setup.h - Magnetostatic Poisson System Setup (CORRECTED)
//
// CORRECTED for pure Neumann problem:
//   - Sparsity pattern
//   - Constraints: hanging nodes + one pinned DoF (to fix constant)
//
// Equation (Eq. 42d): (∇φ, ∇χ) = (h_a - m, ∇χ)
// BC: ∂φ/∂n = (h_a - m)·n  (Neumann, natural BC)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

/**
 * @brief Setup constraints for pure Neumann Poisson problem
 *
 * For pure Neumann BC, the solution is unique only up to a constant.
 * We fix this by pinning one DoF to zero.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   [OUT] Constraints with hanging nodes + pinned DoF
 */
template <int dim>
void setup_poisson_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Build the Poisson system sparsity pattern
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   Constraints (hanging nodes + pinned DoF)
 * @param phi_sparsity      [OUT] Sparsity pattern
 * @param verbose           Print setup info
 */
template <int dim>
void build_poisson_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::SparsityPattern& phi_sparsity,
    bool verbose = false);

#endif // POISSON_SETUP_H