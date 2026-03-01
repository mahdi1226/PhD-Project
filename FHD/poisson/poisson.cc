// ============================================================================
// poisson/poisson.cc - Constructor, Setup Orchestration, Accessors
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42d
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/base/utilities.h>

template <int dim>
PoissonSubsystem<dim>::PoissonSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(params.fe.degree_potential)
    , dof_handler_(triangulation)
{
}

template <int dim>
void PoissonSubsystem<dim>::setup()
{
    distribute_dofs();
    build_constraints();
    build_sparsity_pattern();
    allocate_vectors();
    assemble_matrix();
    initialize_preconditioner();

    pcout_ << "  Poisson: " << dof_handler_.n_dofs() << " DoFs (CG Q"
           << params_.fe.degree_potential << ")\n";
}

template <int dim>
const dealii::DoFHandler<dim>&
PoissonSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PoissonSubsystem<dim>::get_solution() const
{
    return solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PoissonSubsystem<dim>::get_solution_relevant() const
{
    return solution_relevant_;
}

template <int dim>
void PoissonSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        solution_relevant_ = solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void PoissonSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void PoissonSubsystem<dim>::set_mms_source(MMSSourceFunction source)
{
    mms_source_ = source;
}

// Explicit instantiations
template class PoissonSubsystem<2>;
template class PoissonSubsystem<3>;
