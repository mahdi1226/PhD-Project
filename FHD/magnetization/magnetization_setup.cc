// ============================================================================
// magnetization/magnetization_setup.cc - DoF Distribution, Sparsity, Vectors
//
// FE space: M_h = DG Q_ℓ (Nochetto Section 4.3)
//   Discontinuous Galerkin: no hanging node constraints
//   Sparsity pattern includes face couplings for DG
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_tools.h>

template <int dim>
void MagnetizationSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);
    locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

template <int dim>
void MagnetizationSubsystem<dim>::build_sparsity_pattern()
{
    // DG: no constraints (discontinuous elements)
    constraints_.clear();
    constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);
    constraints_.close();

    // DG sparsity: includes face couplings between adjacent cells
    dealii::TrilinosWrappers::SparsityPattern sparsity_pattern(
        locally_owned_dofs_, locally_owned_dofs_,
        locally_relevant_dofs_, mpi_comm_);

    dealii::DoFTools::make_flux_sparsity_pattern(
        dof_handler_, sparsity_pattern, constraints_,
        /*keep_constrained_dofs=*/false);

    sparsity_pattern.compress();
    system_matrix_.reinit(sparsity_pattern);
}

template <int dim>
void MagnetizationSubsystem<dim>::allocate_vectors()
{
    // Owned vectors (solver I/O)
    Mx_solution_.reinit(locally_owned_dofs_, mpi_comm_);
    My_solution_.reinit(locally_owned_dofs_, mpi_comm_);
    Mx_rhs_.reinit(locally_owned_dofs_, mpi_comm_);
    My_rhs_.reinit(locally_owned_dofs_, mpi_comm_);

    // Ghosted vectors (cross-subsystem reads)
    Mx_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
    My_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
}

// Explicit instantiations
template void MagnetizationSubsystem<2>::distribute_dofs();
template void MagnetizationSubsystem<2>::build_sparsity_pattern();
template void MagnetizationSubsystem<2>::allocate_vectors();

template void MagnetizationSubsystem<3>::distribute_dofs();
template void MagnetizationSubsystem<3>::build_sparsity_pattern();
template void MagnetizationSubsystem<3>::allocate_vectors();
