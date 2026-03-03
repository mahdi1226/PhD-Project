// ============================================================================
// poisson/poisson_setup.cc - DoF Distribution, Constraints, Sparsity, Vectors
//
// FE space: X_h = CG Q_ℓ (Nochetto Section 4.3)
// BCs: pure Neumann (∇φ·n = 0 on ∂Ω)
// Null-space: pin DoF 0 = 0 to make the Laplacian invertible
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_tools.h>

template <int dim>
void PoissonSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);
    locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

template <int dim>
void PoissonSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);

    // Hanging node constraints (for AMR compatibility)
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    // Pin DoF 0 to eliminate Neumann null-space
    // Only the owning process adds this constraint
    if (locally_owned_dofs_.is_element(0))
        constraints_.add_constraint(0, {}, 0.0);

    constraints_.close();
}

template <int dim>
void PoissonSubsystem<dim>::build_sparsity_pattern()
{
    dealii::TrilinosWrappers::SparsityPattern sparsity_pattern(
        locally_owned_dofs_, locally_owned_dofs_,
        locally_relevant_dofs_, mpi_comm_);

    dealii::DoFTools::make_sparsity_pattern(
        dof_handler_, sparsity_pattern, constraints_,
        /*keep_constrained_dofs=*/false);

    sparsity_pattern.compress();
    system_matrix_.reinit(sparsity_pattern);
}

template <int dim>
void PoissonSubsystem<dim>::allocate_vectors()
{
    system_rhs_.reinit(locally_owned_dofs_, mpi_comm_);
    solution_.reinit(locally_owned_dofs_, mpi_comm_);
    solution_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_,
                              mpi_comm_);
}

// Explicit instantiations
template void PoissonSubsystem<2>::distribute_dofs();
template void PoissonSubsystem<2>::build_constraints();
template void PoissonSubsystem<2>::build_sparsity_pattern();
template void PoissonSubsystem<2>::allocate_vectors();

template void PoissonSubsystem<3>::distribute_dofs();
template void PoissonSubsystem<3>::build_constraints();
template void PoissonSubsystem<3>::build_sparsity_pattern();
template void PoissonSubsystem<3>::allocate_vectors();
