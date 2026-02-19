// ============================================================================
// magnetization/magnetization_setup.cc — DG DoF Distribution, Sparsity, Vectors
//
// Private methods called by MagnetizationSubsystem::setup():
//   1. distribute_dofs()       — DG-Q1 DoF distribution + index sets
//   2. build_sparsity_pattern() — Face-coupled (flux) sparsity for DG
//   3. allocate_vectors()       — Owned + ghosted vectors for Mx, My
//
// DG SPECIFICS:
//   - No hanging-node constraints (DG has no inter-element continuity)
//   - Sparsity includes face coupling via make_flux_sparsity_pattern()
//   - Single matrix shared by Mx and My (same transport operator)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

using namespace dealii;

// ============================================================================
// distribute_dofs() — Attach FE_DGQ(1) and extract parallel index sets
//
// DG elements have no continuity constraints, so AffineConstraints is
// trivially empty.  We skip creating one entirely.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    locally_owned_dofs_    = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);

    pcout_ << "[Magnetization Setup] DoFs distributed: "
           << dof_handler_.n_dofs() << " total, "
           << locally_owned_dofs_.n_elements() << " local"
           << std::endl;
}

// ============================================================================
// build_sparsity_pattern() — DG flux sparsity (cell + face coupling)
//
// make_flux_sparsity_pattern() includes neighbor coupling across interior
// faces, which is required for the upwind DG flux terms in B_h^m (Eq. 57).
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::build_sparsity_pattern()
{
    TrilinosWrappers::SparsityPattern trilinos_sp(
        locally_owned_dofs_,
        locally_owned_dofs_,
        locally_relevant_dofs_,
        mpi_comm_);

    DoFTools::make_flux_sparsity_pattern(dof_handler_, trilinos_sp);

    trilinos_sp.compress();

    system_matrix_.reinit(trilinos_sp);

    pcout_ << "[Magnetization Setup] Sparsity: nnz = "
           << trilinos_sp.n_nonzero_elements() << " (DG flux)"
           << std::endl;
}

// ============================================================================
// allocate_vectors() — Owned (Mx, My solution + rhs) and ghosted (relevant)
//
// Owned vectors:  Mx/My_solution_, Mx/My_rhs_  (no ghost entries)
// Ghosted vectors: Mx/My_relevant_  (include locally_relevant for reads)
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::allocate_vectors()
{
    // RHS vectors (owned only — assembled locally, no ghost writes)
    Mx_rhs_.reinit(locally_owned_dofs_, mpi_comm_);
    My_rhs_.reinit(locally_owned_dofs_, mpi_comm_);

    // Solution vectors (owned — solver output)
    Mx_solution_.reinit(locally_owned_dofs_, mpi_comm_);
    My_solution_.reinit(locally_owned_dofs_, mpi_comm_);

    // Ghosted vectors (for cross-subsystem reads)
    Mx_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
    My_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);

    pcout_ << "[Magnetization Setup] Vectors allocated (Mx+My: "
           << "2×solution, 2×rhs, 2×ghosted)" << std::endl;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void MagnetizationSubsystem<2>::distribute_dofs();
template void MagnetizationSubsystem<2>::build_sparsity_pattern();
template void MagnetizationSubsystem<2>::allocate_vectors();

template void MagnetizationSubsystem<3>::distribute_dofs();
template void MagnetizationSubsystem<3>::build_sparsity_pattern();
template void MagnetizationSubsystem<3>::allocate_vectors();