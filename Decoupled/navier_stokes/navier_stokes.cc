// ============================================================================
// navier_stokes/navier_stokes.cc — Orchestration, Accessors, Diagnostics
//
// Constructor, advance_time(), initialize_zero(), public accessors,
// ghost management, and diagnostics computation.
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021, B167-B193
//            Algorithm 3.1 Steps 2-4, Equations 3.11-3.13
//
// Other implementation files:
//   navier_stokes_setup.cc    — setup(): DoFs, constraints, sparsity, vectors
//   navier_stokes_assemble.cc — assemble(), assemble_stokes()
//   navier_stokes_solve.cc    — solve(): CG+AMG for velocity and pressure
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
// advance_time() — swap U^{n-1} ← U^n, P^{n-1} ← P^n
//
// Must be called AFTER solve() and BEFORE the next timestep's assemble().
// Projection method needs old pressure for both velocity predictor and
// pressure Poisson RHS.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::advance_time()
{
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;
    p_old_solution_  = p_solution_;

    // Update old ghosted vectors for next step's assembly
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
    p_old_relevant_  = p_old_solution_;
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
    p_old_solution_  = 0;

    ux_relevant_     = 0;
    uy_relevant_     = 0;
    p_relevant_      = 0;
    ux_old_relevant_ = 0;
    uy_old_relevant_ = 0;
    p_old_relevant_  = 0;
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
    p_old_solution_  = 0;

    // Update ghosts
    ux_relevant_     = ux_solution_;
    uy_relevant_     = uy_solution_;
    p_relevant_      = 0;
    ux_old_relevant_ = ux_old_solution_;
    uy_old_relevant_ = uy_old_solution_;
    p_old_relevant_  = 0;
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
// Mutable accessors — for AMR SolutionTransfer
// ============================================================================
template <int dim>
dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_ux_dof_handler_mutable() { return ux_dof_handler_; }

template <int dim>
dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_uy_dof_handler_mutable() { return uy_dof_handler_; }

template <int dim>
dealii::DoFHandler<dim>&
NSSubsystem<dim>::get_p_dof_handler_mutable() { return p_dof_handler_; }

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_ux_solution_mutable() { return ux_solution_; }

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_uy_solution_mutable() { return uy_solution_; }

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_p_solution_mutable() { return p_solution_; }

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_ux_old_solution_mutable() { return ux_old_solution_; }

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
NSSubsystem<dim>::get_uy_old_solution_mutable() { return uy_old_solution_; }

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
    p_old_relevant_  = p_old_solution_;
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
// set_mms_source()
// ============================================================================
template <int dim>
void NSSubsystem<dim>::set_mms_source(MmsSourceFunction source)
{
    mms_source_ = std::move(source);
}

// ============================================================================
// compute_diagnostics()
//
// Computes velocity, pressure, incompressibility, vorticity, energy, and
// solver diagnostics after solve().
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
    const double h_min = dealii::GridTools::minimal_cell_diameter(triangulation_);
    if (dt > 0)
        diag.CFL = diag.U_max * dt / h_min;

    // --- Pressure bounds ---
    p_solution_.trilinos_vector().MinValue(&diag.p_min);
    p_solution_.trilinos_vector().MaxValue(&diag.p_max);

    // --- Solver info + timing ---
    diag.iterations    = last_solve_info_.iterations;
    diag.residual      = last_solve_info_.residual;
    diag.solve_time    = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    // --- Quadrature-based diagnostics ---
    // E_kin = ½∫|u|² dΩ, divU = ∇·u, vorticity, p_mean
    {
        const dealii::QGauss<dim> quadrature(fe_velocity_.degree + 1);
        dealii::FEValues<dim> fe_ux(fe_velocity_, quadrature,
            dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
        dealii::FEValues<dim> fe_uy(fe_velocity_, quadrature,
            dealii::update_values | dealii::update_gradients);
        dealii::FEValues<dim> fe_p(fe_pressure_, quadrature,
            dealii::update_values | dealii::update_JxW_values);

        const unsigned int n_q = quadrature.size();
        std::vector<double> ux_vals(n_q), uy_vals(n_q), p_vals(n_q);
        std::vector<dealii::Tensor<1, dim>> grad_ux(n_q), grad_uy(n_q);

        double local_E_kin       = 0.0;
        double local_divU_L2_sq  = 0.0;
        double local_divU_Linf   = 0.0;
        double local_p_sum       = 0.0;
        double local_volume      = 0.0;
        double local_omega_Linf  = 0.0;
        double local_omega_L2_sq = 0.0;
        double local_enstrophy   = 0.0;

        auto cell_ux = ux_dof_handler_.begin_active();
        auto cell_uy = uy_dof_handler_.begin_active();
        auto cell_p  = p_dof_handler_.begin_active();
        const auto end_ux = ux_dof_handler_.end();

        for (; cell_ux != end_ux; ++cell_ux, ++cell_uy, ++cell_p)
        {
            if (!cell_ux->is_locally_owned())
                continue;

            fe_ux.reinit(cell_ux);
            fe_uy.reinit(cell_uy);
            fe_p.reinit(cell_p);

            fe_ux.get_function_values(ux_relevant_, ux_vals);
            fe_uy.get_function_values(uy_relevant_, uy_vals);
            fe_ux.get_function_gradients(ux_relevant_, grad_ux);
            fe_uy.get_function_gradients(uy_relevant_, grad_uy);
            fe_p.get_function_values(p_relevant_, p_vals);

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW = fe_ux.JxW(q);

                // Kinetic energy: ½∫|u|² dΩ
                local_E_kin += 0.5 * (ux_vals[q] * ux_vals[q]
                                    + uy_vals[q] * uy_vals[q]) * JxW;

                // Divergence: ∇·u = ∂ux/∂x + ∂uy/∂y
                const double div_u = grad_ux[q][0] + grad_uy[q][1];
                local_divU_L2_sq += div_u * div_u * JxW;
                local_divU_Linf = std::max(local_divU_Linf, std::abs(div_u));

                // Pressure mean
                local_p_sum += p_vals[q] * JxW;
                local_volume += JxW;

                // 2D vorticity: ω_z = ∂uy/∂x − ∂ux/∂y
                const double omega_z = grad_uy[q][0] - grad_ux[q][1];
                local_omega_Linf = std::max(local_omega_Linf, std::abs(omega_z));
                local_omega_L2_sq += omega_z * omega_z * JxW;
                local_enstrophy  += 0.5 * omega_z * omega_z * JxW;
            }
        }

        // MPI reduce across all ranks
        const auto& mpi_comm = triangulation_.get_communicator();

        // SUM reductions (5 values)
        double local_sums[5] = {local_E_kin, local_divU_L2_sq,
                                local_omega_L2_sq, local_enstrophy, local_p_sum};
        double global_sums[5];
        MPI_Allreduce(local_sums, global_sums, 5, MPI_DOUBLE, MPI_SUM, mpi_comm);

        double local_volume_arr[1] = {local_volume};
        double global_volume[1];
        MPI_Allreduce(local_volume_arr, global_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

        // MAX reductions (2 values)
        double local_maxes[2] = {local_omega_Linf, local_divU_Linf};
        double global_maxes[2];
        MPI_Allreduce(local_maxes, global_maxes, 2, MPI_DOUBLE, MPI_MAX, mpi_comm);

        diag.E_kin      = global_sums[0];
        diag.divU_L2    = std::sqrt(global_sums[1]);
        diag.omega_L2   = std::sqrt(global_sums[2]);
        diag.enstrophy  = global_sums[3];
        diag.p_mean     = (global_volume[0] > 0.0)
                        ? global_sums[4] / global_volume[0] : 0.0;
        diag.omega_Linf = global_maxes[0];
        diag.divU_Linf  = global_maxes[1];
    }

    // --- Reynolds number: Re = U_max * L / ν ---
    // Use average viscosity as characteristic ν
    const double nu_avg = 0.5 * (params_.physics.nu_water + params_.physics.nu_ferro);
    if (nu_avg > 0.0)
        diag.Re_max = diag.U_max * h_min / nu_avg;

    return diag;
}

// ============================================================================
// Explicit instantiations (for methods defined in THIS file only)
// ============================================================================
template class NSSubsystem<2>;
// template class NSSubsystem<3>;  // 2D only