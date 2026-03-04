// ============================================================================
// cahn_hilliard/cahn_hilliard.cc - Constructor, Setup, Accessors
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/base/utilities.h>
#include <deal.II/numerics/vector_tools.h>

template <int dim>
CahnHilliardSubsystem<dim>::CahnHilliardSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(dealii::FE_Q<dim>(params.fe.degree_cahn_hilliard), 2)
    , dof_handler_(triangulation)
{
}

template <int dim>
void CahnHilliardSubsystem<dim>::setup()
{
    distribute_dofs();
    build_constraints();
    build_sparsity_pattern();
    allocate_vectors();

    const unsigned int total_dofs = dof_handler_.n_dofs();
    pcout_ << "  Cahn-Hilliard: "
           << total_dofs << " DoFs (CG Q"
           << params_.fe.degree_cahn_hilliard
           << " x 2 components: phi + mu)\n";
}

template <int dim>
void CahnHilliardSubsystem<dim>::initialize(
    const dealii::Function<dim>& ic_function)
{
    dealii::TrilinosWrappers::MPI::Vector tmp(locally_owned_, mpi_comm_);
    dealii::VectorTools::interpolate(dof_handler_, ic_function, tmp);
    constraints_.distribute(tmp);
    solution_ = tmp;
    ghosts_valid_ = false;
}

template <int dim>
void CahnHilliardSubsystem<dim>::save_old_solution()
{
    update_ghosts();
    old_solution_relevant_ = solution_relevant_;
}

template <int dim>
const dealii::DoFHandler<dim>&
CahnHilliardSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::FESystem<dim>&
CahnHilliardSubsystem<dim>::get_fe() const
{
    return fe_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_solution() const
{
    return solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_relevant() const
{
    return solution_relevant_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_old_relevant() const
{
    return old_solution_relevant_;
}

template <int dim>
void CahnHilliardSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        solution_relevant_ = solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void CahnHilliardSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

template <int dim>
void CahnHilliardSubsystem<dim>::initialize_zero()
{
    solution_ = 0.0;
    ghosts_valid_ = false;
}

template <int dim>
void CahnHilliardSubsystem<dim>::set_mms_source(const MmsSourceFn& fn)
{
    mms_source_fn_ = fn;
}

// Assemble without convection (delegates to full assemble)
template <int dim>
void CahnHilliardSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& old_solution_relevant,
    double dt)
{
    dealii::TrilinosWrappers::MPI::Vector empty_ux, empty_uy;
    dealii::DoFHandler<dim> dummy_dof(triangulation_);
    dealii::FE_Q<dim> dummy_fe(1);
    dummy_dof.distribute_dofs(dummy_fe);

    assemble(old_solution_relevant, dt, empty_ux, empty_uy, dummy_dof);
}

// Explicit instantiations
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;
