// ============================================================================
// passive_scalar/passive_scalar.cc - Constructor, Setup, Accessors
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 104
// ============================================================================

#include "passive_scalar/passive_scalar.h"

#include <deal.II/base/utilities.h>
#include <deal.II/numerics/vector_tools.h>

template <int dim>
PassiveScalarSubsystem<dim>::PassiveScalarSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(params.fe.degree_scalar)
    , dof_handler_(triangulation)
{
}

template <int dim>
void PassiveScalarSubsystem<dim>::setup()
{
    distribute_dofs();
    build_constraints();
    build_sparsity_pattern();
    allocate_vectors();

    pcout_ << "  Passive Scalar: "
           << dof_handler_.n_dofs() << " DoFs (CG Q"
           << params_.fe.degree_scalar << ")\n";
}

template <int dim>
void PassiveScalarSubsystem<dim>::initialize(
    const dealii::Function<dim>& ic_function)
{
    dealii::TrilinosWrappers::MPI::Vector tmp(locally_owned_, mpi_comm_);
    dealii::VectorTools::interpolate(dof_handler_, ic_function, tmp);
    constraints_.distribute(tmp);
    c_solution_ = tmp;
    ghosts_valid_ = false;
}

template <int dim>
const dealii::DoFHandler<dim>&
PassiveScalarSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PassiveScalarSubsystem<dim>::get_solution() const
{
    return c_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PassiveScalarSubsystem<dim>::get_relevant() const
{
    return c_relevant_;
}

template <int dim>
void PassiveScalarSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        c_relevant_ = c_solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void PassiveScalarSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void PassiveScalarSubsystem<dim>::initialize_zero()
{
    c_solution_ = 0.0;
    ghosts_valid_ = false;
}

// Explicit instantiations
template class PassiveScalarSubsystem<2>;
template class PassiveScalarSubsystem<3>;
