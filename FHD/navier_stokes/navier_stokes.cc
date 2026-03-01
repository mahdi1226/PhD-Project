// ============================================================================
// navier_stokes/navier_stokes.cc - Constructor, Setup, Accessors
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42e
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/base/utilities.h>

template <int dim>
NavierStokesSubsystem<dim>::NavierStokesSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_velocity_(params.fe.degree_velocity)
    , fe_pressure_(params.fe.degree_pressure)
    , ux_dof_handler_(triangulation)
    , uy_dof_handler_(triangulation)
    , p_dof_handler_(triangulation)
{
}

template <int dim>
void NavierStokesSubsystem<dim>::setup()
{
    distribute_dofs();
    build_constraints();
    build_coupled_system();
    allocate_vectors();

    pcout_ << "  Navier-Stokes: "
           << n_ux_ + n_uy_ + n_p_ << " DoFs (CG Q"
           << params_.fe.degree_velocity << " velocity + DG P"
           << params_.fe.degree_pressure << " pressure: "
           << n_ux_ << "+" << n_uy_ << "+" << n_p_ << ")\n";
}

template <int dim>
const dealii::DoFHandler<dim>&
NavierStokesSubsystem<dim>::get_ux_dof_handler() const
{
    return ux_dof_handler_;
}

template <int dim>
const dealii::DoFHandler<dim>&
NavierStokesSubsystem<dim>::get_uy_dof_handler() const
{
    return uy_dof_handler_;
}

template <int dim>
const dealii::DoFHandler<dim>&
NavierStokesSubsystem<dim>::get_p_dof_handler() const
{
    return p_dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_ux_solution() const
{
    return ux_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_uy_solution() const
{
    return uy_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_p_solution() const
{
    return p_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_ux_relevant() const
{
    return ux_relevant_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_uy_relevant() const
{
    return uy_relevant_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NavierStokesSubsystem<dim>::get_p_relevant() const
{
    return p_relevant_;
}

template <int dim>
void NavierStokesSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        ux_relevant_ = ux_solution_;
        uy_relevant_ = uy_solution_;
        p_relevant_ = p_solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void NavierStokesSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void NavierStokesSubsystem<dim>::initialize_zero()
{
    ux_solution_ = 0.0;
    uy_solution_ = 0.0;
    p_solution_ = 0.0;
    ghosts_valid_ = false;
}

template <int dim>
void NavierStokesSubsystem<dim>::set_mms_source(MmsSourceFunction source)
{
    mms_source_ = source;
}

// Explicit instantiations
template class NavierStokesSubsystem<2>;
template class NavierStokesSubsystem<3>;
