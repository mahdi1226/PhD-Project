// ============================================================================
// setup/poisson_setup.cc - Magnetostatic Poisson System Setup (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses Trilinos sparsity pattern
//   - Handles distributed DoFs
//   - Pins DoF 0 globally to fix Neumann nullspace
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42c
// ============================================================================

#include "setup/poisson_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

// ============================================================================
// setup_poisson_constraints_and_sparsity (PARALLEL)
// ============================================================================
template <int dim>
void setup_poisson_constraints_and_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::IndexSet& phi_locally_owned,
    const dealii::IndexSet& phi_locally_relevant,
    dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout)
{
    // ========================================================================
    // Step 1: Build constraints (hanging nodes + nullspace fix)
    // ========================================================================
    phi_constraints.clear();
    phi_constraints.reinit(phi_locally_owned, phi_locally_relevant);

    // Hanging node constraints (for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Pin DoF 0 to zero (fixes the constant for pure Neumann)
    // Only the rank that owns DoF 0 adds this constraint
    if (phi_locally_owned.is_element(0))
    {
        phi_constraints.add_line(0);
        phi_constraints.set_inhomogeneity(0, 0.0);
    }

    phi_constraints.close();

    // ========================================================================
    // Step 2: Build Trilinos sparsity pattern
    // ========================================================================
    dealii::TrilinosWrappers::SparsityPattern trilinos_sp(
        phi_locally_owned, phi_locally_owned, phi_locally_relevant,
        mpi_communicator);

    dealii::DoFTools::make_sparsity_pattern(
        phi_dof_handler, trilinos_sp, phi_constraints,
        /*keep_constrained_dofs=*/false);

    trilinos_sp.compress();

    // ========================================================================
    // Step 3: Initialize matrix with sparsity pattern
    // ========================================================================
    phi_matrix.reinit(trilinos_sp);

    pcout << "[Poisson Setup] n_dofs = " << phi_dof_handler.n_dofs()
          << ", locally_owned = " << phi_locally_owned.n_elements()
          << ", nnz = " << trilinos_sp.n_nonzero_elements() << "\n";
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_poisson_constraints_and_sparsity<2>(
    const dealii::DoFHandler<2>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm,
    dealii::ConditionalOStream&);

template void setup_poisson_constraints_and_sparsity<3>(
    const dealii::DoFHandler<3>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm,
    dealii::ConditionalOStream&);