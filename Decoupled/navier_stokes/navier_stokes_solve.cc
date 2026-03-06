// ============================================================================
// navier_stokes/navier_stokes_solve.cc — CG+AMG Solver Implementation
//
// Pressure-correction projection method (Zhang Algorithm 3.1).
// Each subsystem has its own CG+AMG solve:
//
//   solve_velocity(): CG+AMG for ux_matrix_ and uy_matrix_
//   solve_pressure(): CG+AMG for p_matrix_ + mean subtraction
//   solve():          backward-compat wrapper (calls solve_velocity)
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// solve_velocity() — CG+AMG for ux and uy velocity predictor
//
// Solves [A_ux][ūx] = [F_x] and [A_uy][ūy] = [F_y] independently.
// Each is SPD (mass + viscous + convection), so CG is appropriate.
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve_velocity()
{
    SolverInfo info;
    info.solver_name = "NS-Velocity-CG+AMG";
    info.matrix_size = ux_matrix_.m() + uy_matrix_.m();
    info.nnz = ux_matrix_.n_nonzero_elements() + uy_matrix_.n_nonzero_elements();
    info.used_direct = false;

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = params_.solvers.ns.rel_tolerance;
    const unsigned int max_iter = params_.solvers.ns.max_iterations;

    // --- Solve ux ---
    {
        const double rhs_norm = ux_rhs_.l2_norm();
        if (rhs_norm < 1e-14)
        {
            ux_solution_ = 0;
        }
        else
        {
            dealii::SolverControl control(max_iter, tol * rhs_norm);
            dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(control);

            dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
            amg_data.elliptic = true;
            amg_data.higher_order_elements = true;
            amg_data.smoother_sweeps = 2;
            amg_data.aggregation_threshold = 0.02;

            dealii::TrilinosWrappers::PreconditionAMG amg;
            amg.initialize(ux_matrix_, amg_data);

            cg.solve(ux_matrix_, ux_solution_, ux_rhs_, amg);
            ux_constraints_.distribute(ux_solution_);

            info.iterations = control.last_step();
            info.residual = control.last_value() / rhs_norm;
        }
    }

    // --- Solve uy ---
    {
        const double rhs_norm = uy_rhs_.l2_norm();
        if (rhs_norm < 1e-14)
        {
            uy_solution_ = 0;
        }
        else
        {
            dealii::SolverControl control(max_iter, tol * rhs_norm);
            dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(control);

            dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
            amg_data.elliptic = true;
            amg_data.higher_order_elements = true;
            amg_data.smoother_sweeps = 2;
            amg_data.aggregation_threshold = 0.02;

            dealii::TrilinosWrappers::PreconditionAMG amg;
            amg.initialize(uy_matrix_, amg_data);

            cg.solve(uy_matrix_, uy_solution_, uy_rhs_, amg);
            uy_constraints_.distribute(uy_solution_);

            info.iterations += control.last_step();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();
    info.converged = true;

    // Update ghosted velocity vectors (needed for pressure Poisson div(ū))
    ux_relevant_ = ux_solution_;
    uy_relevant_ = uy_solution_;

    if (params_.solvers.ns.verbose)
    {
        pcout_ << "  [NS Velocity] CG+AMG iters=" << info.iterations
               << ", time=" << std::fixed << std::setprecision(2)
               << info.solve_time << "s\n" << std::defaultfloat;
    }

    last_solve_info_ = info;
    return info;
}


// ============================================================================
// solve_pressure() — CG+AMG for pressure Poisson
//
// Solves [L_p][p^n] = [G].
// L_p = (∇p, ∇q) is SPD (after removing constant null space).
// Pressure uniqueness via post-solve mean subtraction.
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve_pressure()
{
    SolverInfo info;
    info.solver_name = "NS-Pressure-CG+AMG";
    info.matrix_size = p_matrix_.m();
    info.nnz = p_matrix_.n_nonzero_elements();
    info.used_direct = false;

    auto start = std::chrono::high_resolution_clock::now();

    const double rhs_norm = p_rhs_.l2_norm();
    if (rhs_norm < 1e-14)
    {
        p_solution_ = 0;
    }
    else
    {
        const double tol = params_.solvers.ns.rel_tolerance;
        const unsigned int max_iter = params_.solvers.ns.max_iterations;

        dealii::SolverControl control(max_iter, tol * rhs_norm);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(control);

        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = true;
        amg_data.higher_order_elements = false;  // Q1 pressure
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 0.02;

        dealii::TrilinosWrappers::PreconditionAMG amg;
        amg.initialize(p_matrix_, amg_data);

        cg.solve(p_matrix_, p_solution_, p_rhs_, amg);
        p_constraints_.distribute(p_solution_);

        info.iterations = control.last_step();
        info.residual = control.last_value() / rhs_norm;
    }

    // Mean subtraction for pressure uniqueness
    subtract_mean_pressure();

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();
    info.converged = true;

    // Update ghosted pressure
    p_relevant_ = p_solution_;

    if (params_.solvers.ns.verbose)
    {
        pcout_ << "  [NS Pressure] CG+AMG iters=" << info.iterations
               << ", time=" << std::fixed << std::setprecision(2)
               << info.solve_time << "s\n" << std::defaultfloat;
    }

    return info;
}


// ============================================================================
// solve() — Backward-compatible wrapper
//
// In the old monolithic system, solve() solved everything at once.
// Now it only calls solve_velocity() for backward compat with MMS tests.
// The full projection method sequence (solve_velocity → assemble_pressure
// → solve_pressure → velocity_correction) is done in the driver.
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve()
{
    return solve_velocity();
}


// ============================================================================
// Explicit instantiations
// ============================================================================
template SolverInfo NSSubsystem<2>::solve_velocity();
template SolverInfo NSSubsystem<3>::solve_velocity();
template SolverInfo NSSubsystem<2>::solve_pressure();
template SolverInfo NSSubsystem<3>::solve_pressure();
template SolverInfo NSSubsystem<2>::solve();
template SolverInfo NSSubsystem<3>::solve();
