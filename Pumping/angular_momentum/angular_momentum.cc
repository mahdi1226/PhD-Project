// ============================================================================
// angular_momentum/angular_momentum.cc - Constructor, Setup, Accessors
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42f
// ============================================================================

#include "angular_momentum/angular_momentum.h"

#include <deal.II/base/utilities.h>

template <int dim>
AngularMomentumSubsystem<dim>::AngularMomentumSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(params.fe.degree_angular)
    , dof_handler_(triangulation)
{
}

template <int dim>
void AngularMomentumSubsystem<dim>::setup()
{
    distribute_dofs();
    build_constraints();
    build_sparsity_pattern();
    allocate_vectors();

    pcout_ << "  Angular Momentum: "
           << dof_handler_.n_dofs() << " DoFs (CG Q"
           << params_.fe.degree_angular << ")\n";
}

template <int dim>
const dealii::DoFHandler<dim>&
AngularMomentumSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
AngularMomentumSubsystem<dim>::get_solution() const
{
    return w_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
AngularMomentumSubsystem<dim>::get_relevant() const
{
    return w_relevant_;
}

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
AngularMomentumSubsystem<dim>::get_solution_mutable()
{
    return w_solution_;
}

template <int dim>
void AngularMomentumSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        w_relevant_ = w_solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void AngularMomentumSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void AngularMomentumSubsystem<dim>::initialize_zero()
{
    w_solution_ = 0.0;
    ghosts_valid_ = false;
}

template <int dim>
void AngularMomentumSubsystem<dim>::set_mms_source(MmsSourceFunction source)
{
    mms_source_ = source;
}

// Explicit instantiations
template class AngularMomentumSubsystem<2>;
template class AngularMomentumSubsystem<3>;
