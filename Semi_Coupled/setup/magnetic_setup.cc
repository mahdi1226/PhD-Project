// ============================================================================
// setup/magnetic_setup.cc - Monolithic Magnetics Setup (PARALLEL)
//
// Creates constraints and sparsity pattern for the combined M+phi system.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/magnetic_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

// ============================================================================
// setup_magnetic_system (PARALLEL)
// ============================================================================
template <int dim>
void setup_magnetic_system(
    const dealii::DoFHandler<dim>& mag_dof_handler,
    const dealii::IndexSet& locally_owned,
    const dealii::IndexSet& locally_relevant,
    dealii::AffineConstraints<double>& mag_constraints,
    dealii::TrilinosWrappers::SparseMatrix& mag_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout)
{
    // ========================================================================
    // Step 1: Count DoFs per component (requires component_wise renumbering)
    // Components: 0=Mx, 1=My, ..., dim=phi
    // ========================================================================
    const std::vector<dealii::types::global_dof_index> dofs_per_component =
        dealii::DoFTools::count_dofs_per_fe_component(mag_dof_handler);

    dealii::types::global_dof_index n_M_dofs = 0;
    for (unsigned int d = 0; d < dim; ++d)
        n_M_dofs += dofs_per_component[d];
    const dealii::types::global_dof_index phi_start = n_M_dofs;

    // ========================================================================
    // Step 2: Constraints (hanging nodes for CG phi + pin one phi DoF)
    // ========================================================================
    mag_constraints.clear();
    mag_constraints.reinit(locally_owned, locally_relevant);

    // Hanging node constraints (only CG phi component produces these)
    dealii::DoFTools::make_hanging_node_constraints(mag_dof_handler, mag_constraints);

    // Pin first phi DoF to zero (Neumann nullspace fix)
    // With component_wise renumbering, phi DoFs start at index phi_start
    if (locally_owned.is_element(phi_start))
    {
        mag_constraints.add_line(phi_start);
        mag_constraints.set_inhomogeneity(phi_start, 0.0);
    }

    mag_constraints.close();

    // ========================================================================
    // Step 3: Sparsity pattern (cell + face coupling for DG transport)
    //
    // Paper Eq. 42c: DG transport B_h^m(U; M, Z) requires face flux coupling
    // between neighboring cells for M (DG) components. The phi (CG) component
    // has no face terms (continuous), but make_flux_sparsity_pattern includes
    // entries for all components — the extra CG-CG face entries are zero.
    // ========================================================================
    dealii::DynamicSparsityPattern dsp(locally_relevant);
    dealii::DoFTools::make_flux_sparsity_pattern(
        mag_dof_handler, dsp, mag_constraints,
        /*keep_constrained_dofs=*/false);

    dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        mag_dof_handler.locally_owned_dofs(),
        mpi_communicator,
        locally_relevant);

    // ========================================================================
    // Step 4: Initialize matrix
    // ========================================================================
    mag_matrix.reinit(locally_owned, locally_owned, dsp, mpi_communicator);

    pcout << "[Magnetic Setup] n_dofs = " << mag_dof_handler.n_dofs()
          << " (M: " << n_M_dofs
          << ", phi: " << dofs_per_component[dim]
          << "), nnz = " << mag_matrix.n_nonzero_elements() << "\n";
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_magnetic_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm,
    dealii::ConditionalOStream&);
