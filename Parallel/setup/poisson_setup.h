// ============================================================================
// setup/poisson_setup.h - Magnetostatic Poisson System Setup (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses TrilinosWrappers::SparseMatrix
//   - Takes IndexSets for owned/relevant DoFs
//   - Pins DoF 0 globally to fix Neumann nullspace
//
// Pure Neumann problem: (μ∇φ, ∇χ) = (h_a - M, ∇χ)
// Requires pinning one DoF to fix the constant (nullspace).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42c
// ============================================================================
#ifndef POISSON_SETUP_H
#define POISSON_SETUP_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/base/index_set.h>

#include <mpi.h>

/**
 * @brief Set up constraints and sparsity pattern for Poisson system (PARALLEL)
 *
 * Creates:
 *   - Constraints: hanging nodes + pin DoF 0 (fixes Neumann nullspace)
 *   - Trilinos sparsity pattern with constrained DoFs eliminated
 *
 * For pure Neumann problems, the solution is unique only up to a constant.
 * Pinning DoF 0 to zero fixes this constant.
 *
 * @param phi_dof_handler    DoFHandler for magnetic potential φ
 * @param phi_locally_owned  IndexSet of locally owned DoFs
 * @param phi_locally_relevant IndexSet of locally relevant DoFs
 * @param phi_constraints    [OUT] Constraints (hanging nodes + nullspace fix)
 * @param phi_matrix         [OUT] Trilinos sparse matrix (initialized with sparsity)
 * @param mpi_communicator   MPI communicator
 * @param pcout              Conditional output stream (rank 0 only)
 */
template <int dim>
void setup_poisson_constraints_and_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::IndexSet& phi_locally_owned,
    const dealii::IndexSet& phi_locally_relevant,
    dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout);

#endif // POISSON_SETUP_H