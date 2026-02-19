// ============================================================================
// poisson/poisson_setup.cc - DoFs, Constraints, Sparsity, Vectors
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   FE space X_h = CG Q1 (piecewise linear, continuous)
//   BCs: ∇φ·n = 0 on ∂Ω (pure Neumann) → pin DoF 0 = 0
//
// Called once from setup(), and again after AMR remeshing.
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

// ============================================================================
// distribute_dofs
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);
}

// ============================================================================
// build_constraints
//
// Two constraints:
//   1. Hanging node constraints (AMR compatibility)
//   2. Pin DoF 0 = 0 (fix Neumann null-space)
//      Only the rank that owns DoF 0 adds this line.
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);

    dealii::DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    if (locally_owned_dofs_.is_element(0))
    {
        constraints_.add_line(0);
        constraints_.set_inhomogeneity(0, 0.0);
    }

    constraints_.close();
}

// ============================================================================
// build_sparsity_pattern
//
// Trilinos distributed sparsity pattern respecting constraints.
// Initializes system_matrix_ with the pattern.
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::build_sparsity_pattern()
{
    dealii::TrilinosWrappers::SparsityPattern trilinos_sp(
        locally_owned_dofs_,
        locally_owned_dofs_,
        locally_relevant_dofs_,
        mpi_comm_);

    dealii::DoFTools::make_sparsity_pattern(
        dof_handler_, trilinos_sp, constraints_,
        /*keep_constrained_dofs=*/false);

    trilinos_sp.compress();

    system_matrix_.reinit(trilinos_sp);
}

// ============================================================================
// allocate_vectors
//
// Three vectors:
//   system_rhs_:        locally owned (assembled, passed to solver)
//   solution_:          locally owned (solver output)
//   solution_relevant_: ghosted (for other subsystems to read ∇φ)
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::allocate_vectors()
{
    system_rhs_.reinit(locally_owned_dofs_, mpi_comm_);
    solution_.reinit(locally_owned_dofs_, mpi_comm_);
    solution_relevant_.reinit(locally_owned_dofs_,
                              locally_relevant_dofs_,
                              mpi_comm_);
    ghosts_valid_ = false;
}

// ============================================================================
// Explicit instantiations (methods defined in THIS file)
// ============================================================================
template void PoissonSubsystem<2>::distribute_dofs();
template void PoissonSubsystem<3>::distribute_dofs();

template void PoissonSubsystem<2>::build_constraints();
template void PoissonSubsystem<3>::build_constraints();

template void PoissonSubsystem<2>::build_sparsity_pattern();
template void PoissonSubsystem<3>::build_sparsity_pattern();

template void PoissonSubsystem<2>::allocate_vectors();
template void PoissonSubsystem<3>::allocate_vectors();
