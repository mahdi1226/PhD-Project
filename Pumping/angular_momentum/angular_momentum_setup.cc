// ============================================================================
// angular_momentum/angular_momentum_setup.cc - DoFs, Constraints, Sparsity
//
// CG Q_ℓ with homogeneous Dirichlet on all boundaries (no-spin BC).
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 5
// ============================================================================

#include "angular_momentum/angular_momentum.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

template <int dim>
void AngularMomentumSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    locally_owned_ = dof_handler_.locally_owned_dofs();
    locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

template <int dim>
void AngularMomentumSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_, locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    // Homogeneous Dirichlet on all boundaries (w = 0)
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler_, 0, dealii::Functions::ZeroFunction<dim>(), constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler_, 1, dealii::Functions::ZeroFunction<dim>(), constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler_, 2, dealii::Functions::ZeroFunction<dim>(), constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        dof_handler_, 3, dealii::Functions::ZeroFunction<dim>(), constraints_);

    constraints_.close();
}

template <int dim>
void AngularMomentumSubsystem<dim>::build_sparsity_pattern()
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
void AngularMomentumSubsystem<dim>::allocate_vectors()
{
    system_rhs_.reinit(locally_owned_, mpi_comm_);
    w_solution_.reinit(locally_owned_, mpi_comm_);
    w_relevant_.reinit(locally_owned_, locally_relevant_, mpi_comm_);
}

// Explicit instantiations
template void AngularMomentumSubsystem<2>::distribute_dofs();
template void AngularMomentumSubsystem<2>::build_constraints();
template void AngularMomentumSubsystem<2>::build_sparsity_pattern();
template void AngularMomentumSubsystem<2>::allocate_vectors();

template void AngularMomentumSubsystem<3>::distribute_dofs();
template void AngularMomentumSubsystem<3>::build_constraints();
template void AngularMomentumSubsystem<3>::build_sparsity_pattern();
template void AngularMomentumSubsystem<3>::allocate_vectors();
