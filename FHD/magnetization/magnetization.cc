// ============================================================================
// magnetization/magnetization.cc - Constructor, Setup, Accessors
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42c
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/base/utilities.h>

template <int dim>
MagnetizationSubsystem<dim>::MagnetizationSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(params.fe.degree_magnetization)
    , dof_handler_(triangulation)
{
}

template <int dim>
void MagnetizationSubsystem<dim>::setup()
{
    distribute_dofs();
    build_sparsity_pattern();
    allocate_vectors();

    pcout_ << "  Magnetization: " << dof_handler_.n_dofs()
           << " DoFs (DG Q" << params_.fe.degree_magnetization
           << " x " << dim << " components)\n";
}

template <int dim>
const dealii::DoFHandler<dim>&
MagnetizationSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_solution() const
{
    return Mx_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_solution() const
{
    return My_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_relevant() const
{
    return Mx_relevant_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_relevant() const
{
    return My_relevant_;
}

template <int dim>
void MagnetizationSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        Mx_relevant_ = Mx_solution_;
        My_relevant_ = My_solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void MagnetizationSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void MagnetizationSubsystem<dim>::save_old_solution()
{
    // Caller is responsible for ensuring ghosts are valid before saving
    update_ghosts();
}

template <int dim>
void MagnetizationSubsystem<dim>::set_mms_source(MmsSourceFunction source)
{
    mms_source_ = source;
}

// Explicit instantiations
template class MagnetizationSubsystem<2>;
template class MagnetizationSubsystem<3>;
