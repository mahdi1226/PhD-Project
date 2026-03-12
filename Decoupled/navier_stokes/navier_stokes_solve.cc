// ============================================================================
// navier_stokes/navier_stokes_solve.cc — NS Solver (CG+AMG or Direct)
//
// Pressure-correction projection method (Zhang Algorithm 3.1).
// Each subsystem has its own solve:
//
//   solve_velocity(): ux_matrix_ and uy_matrix_
//   solve_pressure(): p_matrix_ + mean subtraction
//   solve():          backward-compat wrapper (calls solve_velocity)
//
// Solver modes:
//   use_iterative = true  → CG + AMG (default for projection method)
//   use_iterative = false → Direct (MUMPS → KLU), via --ns-direct
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// Helper: Solve a single scalar system with CG+AMG.
// Returns iteration count (0 if RHS was zero).
// If residual_out is non-null, stores the final solver residual.
// ============================================================================
template <int dim>
static unsigned int solve_scalar_cg_amg(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    double tol, unsigned int max_iter, bool higher_order,
    double* residual_out = nullptr)
{
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        if (residual_out) *residual_out = 0.0;
        return 0;
    }

    dealii::SolverControl control(max_iter, tol * rhs_norm);
    dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(control);

    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.elliptic = true;
    amg_data.higher_order_elements = higher_order;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    dealii::TrilinosWrappers::PreconditionAMG amg;
    amg.initialize(matrix, amg_data);

    cg.solve(matrix, solution, rhs, amg);
    constraints.distribute(solution);

    if (residual_out) *residual_out = control.last_value();
    return control.last_step();
}


// ============================================================================
// Helper: Solve a single scalar system with direct solver (MUMPS → KLU).
// Returns 1 on success, 0 if RHS was zero.
// Throws on failure (caller handles fallback).
// ============================================================================
static unsigned int solve_scalar_direct(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    double tol)
{
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        return 0;
    }

    dealii::SolverControl control(1, tol * rhs_norm);

    // Try MUMPS first
    try
    {
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = "Amesos_Mumps";
        dealii::TrilinosWrappers::SolverDirect solver(control, data);
        solver.solve(matrix, solution, rhs);
    }
    catch (std::exception&)
    {
        // KLU fallback
        dealii::TrilinosWrappers::SolverDirect solver(control);
        solver.solve(matrix, solution, rhs);
    }

    constraints.distribute(solution);
    return 1;
}


// ============================================================================
// solve_velocity() — solve ux and uy velocity predictor systems
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve_velocity()
{
    SolverInfo info;
    info.matrix_size = ux_matrix_.m() + uy_matrix_.m();
    info.nnz = ux_matrix_.n_nonzero_elements() + uy_matrix_.n_nonzero_elements();

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = params_.solvers.ns.rel_tolerance;
    const unsigned int max_iter = params_.solvers.ns.max_iterations;
    const bool use_direct = !params_.solvers.ns.use_iterative;

    double ux_residual = 0.0, uy_residual = 0.0;

    if (use_direct)
    {
        // Direct solver path (--ns-direct or --all-direct)
        info.solver_name = "NS-Velocity-Direct";
        info.used_direct = true;
        try
        {
            info.iterations  = solve_scalar_direct(ux_matrix_, ux_solution_, ux_rhs_,
                                                   ux_constraints_, tol);
            info.iterations += solve_scalar_direct(uy_matrix_, uy_solution_, uy_rhs_,
                                                   uy_constraints_, tol);
            info.converged = true;
            // Compute post-solve residual ‖Ax − b‖ for each component
            {
                dealii::TrilinosWrappers::MPI::Vector res(ux_rhs_);
                ux_matrix_.vmult(res, ux_solution_); res -= ux_rhs_;
                ux_residual = res.l2_norm();
                uy_matrix_.vmult(res, uy_solution_); res -= uy_rhs_;
                uy_residual = res.l2_norm();
                info.residual = std::max(ux_residual, uy_residual);
            }
        }
        catch (std::exception& e)
        {
            pcout_ << "  [NS Velocity] Direct solver failed: " << e.what()
                   << ", falling back to CG+AMG\n";
            // Fall through to iterative
            info.iterations  = solve_scalar_cg_amg<dim>(ux_matrix_, ux_solution_, ux_rhs_,
                                                        ux_constraints_, tol, max_iter, true,
                                                        &ux_residual);
            info.iterations += solve_scalar_cg_amg<dim>(uy_matrix_, uy_solution_, uy_rhs_,
                                                        uy_constraints_, tol, max_iter, true,
                                                        &uy_residual);
            info.solver_name = "NS-Velocity-CG+AMG(fallback)";
            info.used_direct = false;
            info.converged = true;
            info.residual = std::max(ux_residual, uy_residual);
        }
    }
    else
    {
        // CG+AMG path (default)
        info.solver_name = "NS-Velocity-CG+AMG";
        info.used_direct = false;
        info.iterations  = solve_scalar_cg_amg<dim>(ux_matrix_, ux_solution_, ux_rhs_,
                                                    ux_constraints_, tol, max_iter, true,
                                                    &ux_residual);
        info.iterations += solve_scalar_cg_amg<dim>(uy_matrix_, uy_solution_, uy_rhs_,
                                                    uy_constraints_, tol, max_iter, true,
                                                    &uy_residual);
        info.residual = std::max(ux_residual, uy_residual);
        info.converged = true;
    }

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    // Update ghosted velocity vectors (needed for pressure Poisson div(ū))
    ux_relevant_ = ux_solution_;
    uy_relevant_ = uy_solution_;

    if (params_.solvers.ns.verbose)
    {
        pcout_ << "  [NS Velocity] " << (use_direct ? "Direct" : "CG+AMG")
               << " iters=" << info.iterations
               << ", time=" << std::fixed << std::setprecision(2)
               << info.solve_time << "s\n" << std::defaultfloat;
    }

    last_solve_info_ = info;
    return info;
}


// ============================================================================
// solve_pressure() — solve pressure Poisson system
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve_pressure()
{
    SolverInfo info;
    info.matrix_size = p_matrix_.m();
    info.nnz = p_matrix_.n_nonzero_elements();

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = params_.solvers.ns.rel_tolerance;
    const unsigned int max_iter = params_.solvers.ns.max_iterations;
    const bool use_direct = !params_.solvers.ns.use_iterative;

    double p_residual = 0.0;

    if (use_direct)
    {
        // Direct solver path
        info.solver_name = "NS-Pressure-Direct";
        info.used_direct = true;
        try
        {
            info.iterations = solve_scalar_direct(p_matrix_, p_solution_, p_rhs_,
                                                  p_constraints_, tol);
            info.converged = true;
            // Compute post-solve residual ‖Ax − b‖
            {
                dealii::TrilinosWrappers::MPI::Vector res(p_rhs_);
                p_matrix_.vmult(res, p_solution_); res -= p_rhs_;
                info.residual = res.l2_norm();
            }
        }
        catch (std::exception& e)
        {
            pcout_ << "  [NS Pressure] Direct solver failed: " << e.what()
                   << ", falling back to CG+AMG\n";
            info.iterations = solve_scalar_cg_amg<dim>(p_matrix_, p_solution_, p_rhs_,
                                                       p_constraints_, tol, max_iter, false,
                                                       &p_residual);
            info.solver_name = "NS-Pressure-CG+AMG(fallback)";
            info.used_direct = false;
            info.converged = true;
            info.residual = p_residual;
        }
    }
    else
    {
        // CG+AMG path (default)
        info.solver_name = "NS-Pressure-CG+AMG";
        info.used_direct = false;
        info.iterations = solve_scalar_cg_amg<dim>(p_matrix_, p_solution_, p_rhs_,
                                                   p_constraints_, tol, max_iter, false,
                                                   &p_residual);
        info.residual = p_residual;
        info.converged = true;
    }

    // Mean subtraction for pressure uniqueness
    subtract_mean_pressure();

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    // Update ghosted pressure
    p_relevant_ = p_solution_;

    if (params_.solvers.ns.verbose)
    {
        pcout_ << "  [NS Pressure] " << (use_direct ? "Direct" : "CG+AMG")
               << " iters=" << info.iterations
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
