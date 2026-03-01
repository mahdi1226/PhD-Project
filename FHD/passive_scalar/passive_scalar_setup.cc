// ============================================================================
// passive_scalar/passive_scalar_setup.cc - DoFs, Constraints, Sparsity
//
// CG Q_ℓ with homogeneous Neumann BCs (no-flux).
// Neumann = natural BC for CG → only hanging node constraints needed.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 104
// ============================================================================

#include "passive_scalar/passive_scalar.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparsity_tools.h>

template <int dim>
void PassiveScalarSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    locally_owned_ = dof_handler_.locally_owned_dofs();
    locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

template <int dim>
void PassiveScalarSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_, locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    // Homogeneous Neumann BCs: no boundary constraints needed (natural BC)
    constraints_.close();
}

template <int dim>
void PassiveScalarSubsystem<dim>::build_sparsity_pattern()
{
    dealii::TrilinosWrappers::SparsityPattern sparsity(
        locally_owned_, locally_owned_,
        locally_relevant_, mpi_comm_);

    dealii::DoFTools::make_sparsity_pattern(
        dof_handler_, sparsity, constraints_, false);

    sparsity.compress();
    system_matrix_.reinit(sparsity);
}

template <int dim>
void PassiveScalarSubsystem<dim>::allocate_vectors()
{
    system_rhs_.reinit(locally_owned_, mpi_comm_);
    c_solution_.reinit(locally_owned_, mpi_comm_);
    c_relevant_.reinit(locally_owned_, locally_relevant_, mpi_comm_);
}

// Explicit instantiations
template void PassiveScalarSubsystem<2>::distribute_dofs();
template void PassiveScalarSubsystem<2>::build_constraints();
template void PassiveScalarSubsystem<2>::build_sparsity_pattern();
template void PassiveScalarSubsystem<2>::allocate_vectors();

template void PassiveScalarSubsystem<3>::distribute_dofs();
template void PassiveScalarSubsystem<3>::build_constraints();
template void PassiveScalarSubsystem<3>::build_sparsity_pattern();
template void PassiveScalarSubsystem<3>::allocate_vectors();
