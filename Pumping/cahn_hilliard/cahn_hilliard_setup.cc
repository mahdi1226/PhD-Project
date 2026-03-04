// ============================================================================
// cahn_hilliard/cahn_hilliard_setup.cc - DoFs, Constraints, Sparsity
//
// FESystem(FE_Q(degree), 2): component 0 = phi, component 1 = mu
// Homogeneous Neumann BCs for both (natural BC, no boundary constraints).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparsity_tools.h>

template <int dim>
void CahnHilliardSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    // Note: component_wise renumbering removed — it makes locally_owned_dofs
    // non-contiguous on distributed meshes, breaking DataOut for VTK output.
    // The monolithic FESystem solver works fine with default (interleaved) numbering.

    locally_owned_ = dof_handler_.locally_owned_dofs();
    locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

template <int dim>
void CahnHilliardSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_, locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    // Homogeneous Neumann BCs: no boundary constraints needed (natural BC)
    constraints_.close();
}

template <int dim>
void CahnHilliardSubsystem<dim>::build_sparsity_pattern()
{
    dealii::TrilinosWrappers::SparsityPattern sparsity(
        locally_owned_, locally_owned_,
        locally_relevant_, mpi_comm_);

    // Full coupling: all 4 blocks (phi-phi, phi-mu, mu-phi, mu-mu)
    dealii::DoFTools::make_sparsity_pattern(
        dof_handler_, sparsity, constraints_, false);

    sparsity.compress();
    system_matrix_.reinit(sparsity);
}

template <int dim>
void CahnHilliardSubsystem<dim>::allocate_vectors()
{
    system_rhs_.reinit(locally_owned_, mpi_comm_);
    solution_.reinit(locally_owned_, mpi_comm_);
    solution_relevant_.reinit(locally_owned_, locally_relevant_, mpi_comm_);
    old_solution_relevant_.reinit(locally_owned_, locally_relevant_, mpi_comm_);
}

// Explicit instantiations
template void CahnHilliardSubsystem<2>::distribute_dofs();
template void CahnHilliardSubsystem<2>::build_constraints();
template void CahnHilliardSubsystem<2>::build_sparsity_pattern();
template void CahnHilliardSubsystem<2>::allocate_vectors();

template void CahnHilliardSubsystem<3>::distribute_dofs();
template void CahnHilliardSubsystem<3>::build_constraints();
template void CahnHilliardSubsystem<3>::build_sparsity_pattern();
template void CahnHilliardSubsystem<3>::allocate_vectors();
