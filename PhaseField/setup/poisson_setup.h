// ============================================================================
// setup/poisson_setup.h - Magnetostatic Poisson System Setup
//
// Pure Neumann problem: -Δφ = ∇·(m - h_a), ∂φ/∂n = (h_a - m)·n
//
// NOTE: Constraint setup (hanging nodes + nullspace fix) is in
//       poisson_assembler.h: setup_poisson_neumann_constraints()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42d
// ============================================================================
#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

/**
 * @brief Build the Poisson system sparsity pattern
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   Constraints (from setup_poisson_neumann_constraints)
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