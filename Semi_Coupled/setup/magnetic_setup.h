// ============================================================================
// setup/magnetic_setup.h - Monolithic Magnetics Setup (PARALLEL)
//
// Combined FESystem for DG magnetization (vector) + CG potential (scalar):
//   FESystem<dim>(FE_DGQ(degree_M), dim, FE_Q(degree_phi), 1)
//
// Block structure (after DoFRenumbering::component_wise):
//   Components 0..dim-1: M (DG vector)
//   Component  dim:      phi (CG scalar)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42c-42d solved as a monolithic block system.
// ============================================================================
#ifndef MAGNETIC_SETUP_H
#define MAGNETIC_SETUP_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <mpi.h>
#include <vector>

/**
 * @brief Set up monolithic magnetics system (PARALLEL)
 *
 * Creates constraints and sparsity pattern for the combined M+phi system.
 *
 * Constraints:
 *   - Hanging nodes for CG phi component (AMR support)
 *   - Pin first phi DoF to zero (Neumann nullspace fix)
 *   - DG M components: unconstrained
 *
 * Sparsity:
 *   - Cell coupling: all components couple (M-M, M-phi, phi-M, phi-phi)
 *   - Face coupling: only M-M (DG transport upwind flux)
 *
 * PREREQUISITE: DoFRenumbering::component_wise must have been applied
 * to the DoFHandler BEFORE calling this function.
 *
 * @param mag_dof_handler   DoFHandler with FESystem (DG^dim + CG)
 * @param locally_owned     IndexSet of locally owned DoFs
 * @param locally_relevant  IndexSet of locally relevant DoFs
 * @param mag_constraints   [OUT] AffineConstraints (hanging nodes + phi pin)
 * @param mag_matrix        [OUT] Trilinos sparse matrix
 * @param mpi_communicator  MPI communicator
 * @param pcout             Conditional output stream
 */
template <int dim>
void setup_magnetic_system(
    const dealii::DoFHandler<dim>& mag_dof_handler,
    const dealii::IndexSet& locally_owned,
    const dealii::IndexSet& locally_relevant,
    dealii::AffineConstraints<double>& mag_constraints,
    dealii::TrilinosWrappers::SparseMatrix& mag_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout);

#endif // MAGNETIC_SETUP_H
