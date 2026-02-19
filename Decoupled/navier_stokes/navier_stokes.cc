// ============================================================================
// navier_stokes/navier_stokes.cc — Orchestration, Accessors, Diagnostics
//
// Constructor, advance_time(), initialize_zero(), public accessors,
// ghost management, and diagnostics computation.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Equation 42e-42f
//
// Other implementation files:
//   navier_stokes_setup.cc    — setup(): DoFs, constraints, sparsity, vectors
//   navier_stokes_assemble.cc — assemble(), assemble_stokes()
//   navier_stokes_solve.cc    — solve(): direct or block-Schur
// ============================================================================

#include "navier_stokes.h"

#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <limits>
#include <cmath>
#include <algorithm>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
NSSubsystem<dim>::NSSubsystem(
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

// ============================================================================
// advance_time() — swap U^{n-1} ← U^n
//
// Must be called AFTER solve() and BEFORE the next timestep's assemble().
// ============================================================================
template <int dim>
void NSSubsystem<dim>::advance_time()
{
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;

    // Update old ghosted vectors for next step's assembly
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
}

// ============================================================================
// initialize_zero() — set all solution vectors to zero
//
// Call after setup() to initialize the system.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::initialize_zero()
{
    ux_solution_     = 0;
    ux_old_solution_ = 0;
    uy_solution_     = 0;
    uy_old_solution_ = 0;
    p_solution_      = 0;
    ns_solution_     = 0;

    ux_relevant_     = 0;
    uy_relevant_     = 0;
    p_relevant_      = 0;
    ux_old_relevant_ = 0;
    uy_old_relevant_ = 0;
}

// ============================================================================
// set_old_velocity() — initialize U^{n-1} from exact solution functions
// ============================================================================
template <int dim>
void NSSubsystem<dim>::set_old_velocity(
    const dealii::Function<dim>& ux_exact,
    const dealii::Function<dim>& uy_exact)
{
    // Interpolate onto owned vectors
    dealii::VectorTools::interpolate(ux_dof_handler_, ux_exact, ux_old_solution_);
    dealii::VectorTools::interpolate(uy_dof_handler_, uy_exact, uy_old_solution_);

    // Update ghosted copies
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
}

// ============================================================================
// initialize_velocity() — set velocity from analytic functions
//
// Interpolates ux_init, uy_init into both current and old solution vectors,
// then updates ghosted copies.  Pressure is zeroed.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::initialize_velocity(
    const dealii::Function<dim>& ux_init,
    const dealii::Function<dim>& uy_init)
{
    // Interpolate into owned temporary, then copy to owned + old
    dealii::TrilinosWrappers::MPI::Vector ux_tmp(ux_locally_owned_, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector uy_tmp(uy_locally_owned_, mpi_comm_);

    dealii::VectorTools::interpolate(ux_dof_handler_, ux_init, ux_tmp);
    dealii::VectorTools::interpolate(uy_dof_handler_, uy_init, uy_tmp);

    ux_solution_     = ux_tmp;
    ux_old_solution_ = ux_tmp;
    uy_solution_     = uy_tmp;
    uy_old_solution_ = uy_tmp;
    p_solution_      = 0;
    ns_solution_     = 0;

    // Update ghosts
    ux_relevant_     = ux_solution_;
    uy_relevant_     = uy_solution_;
    p_relevant_      = 0;
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
}

// ============================================================================
// Accessors
// ============================================================================
template <int dim>
const dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_ux_dof_handler() const { return ux_dof_handler_; }

template <int dim>
const dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_uy_dof_handler() const { return uy_dof_handler_; }

template <int dim>
const dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_p_dof_handler() const { return p_dof_handler_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_ux_solution() const { return ux_solution_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_uy_solution() const { return uy_solution_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_p_solution() const { return p_solution_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_ux_relevant() const { return ux_relevant_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_uy_relevant() const { return uy_relevant_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_p_relevant() const { return p_relevant_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_ux_old_relevant() const { return ux_old_relevant_; }

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_uy_old_relevant() const { return uy_old_relevant_; }

// ============================================================================
// Ghost management
// ============================================================================
template <int dim>
void NSSubsystem<dim>::update_ghosts()
{
    ux_relevant_     = ux_solution_;
    uy_relevant_     = uy_solution_;
    p_relevant_      = p_solution_;
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
}

template <int dim>
void NSSubsystem<dim>::invalidate_ghosts()
{
    // Mark ghosted vectors as stale.
    // Trilinos ghosted vectors don't have an explicit invalidate,
    // but setting to zero signals that update_ghosts() must be called
    // before any cross-DoFHandler evaluation.
    ux_relevant_     = 0;
    uy_relevant_     = 0;
    p_relevant_      = 0;
}

// ============================================================================
// compute_diagnostics()
//
// Computes velocity, pressure, incompressibility, and solver diagnostics
// after solve().
//
// NOTE: This is a placeholder skeleton. Full implementation requires
// quadrature evaluation. For now, it computes vector min/max and
// returns solver info from last solve.
// ============================================================================
template <int dim>
typename NSSubsystem<dim>::Diagnostics
NSSubsystem<dim>::compute_diagnostics(double dt) const
{
    Diagnostics diag;

    // --- Velocity bounds (Epetra global min/max across all MPI ranks) ---
    ux_solution_.trilinos_vector().MinValue(&diag.ux_min);
    ux_solution_.trilinos_vector().MaxValue(&diag.ux_max);
    uy_solution_.trilinos_vector().MinValue(&diag.uy_min);
    uy_solution_.trilinos_vector().MaxValue(&diag.uy_max);

    diag.U_max = std::max(std::max(std::abs(diag.ux_min), std::abs(diag.ux_max)),
                          std::max(std::abs(diag.uy_min), std::abs(diag.uy_max)));

    // --- CFL ---
    if (dt > 0)
    {
        const double h_min = dealii::GridTools::minimal_cell_diameter(triangulation_);
        diag.CFL = diag.U_max * dt / h_min;
    }

    // --- Pressure bounds ---
    p_solution_.trilinos_vector().MinValue(&diag.p_min);
    p_solution_.trilinos_vector().MaxValue(&diag.p_max);

    // --- Solver info ---
    diag.iterations = last_solve_info_.iterations;
    diag.residual   = last_solve_info_.residual;

    // --- TODO: Full quadrature-based diagnostics ---
    // E_kin, divU_L2, divU_Linf, p_mean, omega_L2, omega_Linf, enstrophy
    // These require FEValues evaluation and will be implemented when needed
    // for production monitoring.

    return diag;
}

// ============================================================================
// Explicit instantiations (for methods defined in THIS file only)
// ============================================================================
template class NSSubsystem<2>;
template class NSSubsystem<3>;