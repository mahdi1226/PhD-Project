// ============================================================================
// setup/poisson_setup.h - Magnetostatic Poisson System Setup
//
// Simple single-field setup:
//   - Sparsity pattern
//   - Constraints (hanging nodes + Dirichlet BCs)
//
// Equation: -∇·(μ(θ)∇φ) = 0 in Ω, φ = φ_dipole on ∂Ω [Eq. 14d, p.499]
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

/**
 * @brief Build the Poisson system sparsity pattern
 *
 * Simple single-field setup. Constraints should already be populated
 * with hanging nodes and Dirichlet BCs before calling this.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   Constraints (hanging nodes + BCs)
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