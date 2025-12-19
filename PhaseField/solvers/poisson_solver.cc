// ============================================================================
// solvers/poisson_solver.cc - Magnetostatic Poisson Solver
//
// UPDATED: Now returns SolverInfo with iterations/residual/time
// ============================================================================

#include "solvers/poisson_solver.h"
#include "utilities/parameters.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <chrono>

// ============================================================================
// Main solver with explicit parameters - RETURNS SolverInfo
// ============================================================================
SolverInfo solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const LinearSolverParams& params,
    bool log_output)
{
    SolverInfo info;
    info.solver_name = "Poisson";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        if (log_output)
            std::cout << "[Poisson] Zero RHS, solution set to zero\n";
        info.converged = true;
        return info;
    }

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();
    bool converged = false;
    unsigned int iterations = 0;
    double final_residual = 0.0;

    if (params.use_iterative)
    {
        const double tol = std::max(params.abs_tolerance,
                                    params.rel_tolerance * rhs_norm);

        dealii::SolverControl solver_control(params.max_iterations, tol);
        dealii::SolverCG<dealii::Vector<double>> solver(solver_control);

        dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(matrix, params.ssor_omega);

        try
        {
            solver.solve(matrix, solution, rhs, preconditioner);
            converged = true;
            iterations = solver_control.last_step();
            final_residual = solver_control.last_value();
        }
        catch (dealii::SolverControl::NoConvergence& e)
        {
            std::cerr << "[Poisson] WARNING: CG did not converge after "
                      << e.last_step << " iterations. "
                      << "Residual = " << e.last_residual << "\n";
            iterations = e.last_step;
            final_residual = e.last_residual;

            if (params.fallback_to_direct)
                std::cerr << "[Poisson] Falling back to direct solver.\n";
        }
    }

    if (!converged && params.fallback_to_direct)
    {
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(solution, rhs);
        converged = true;
        iterations = 1;
        final_residual = 0.0;
        info.used_direct = true;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    // Fill SolverInfo
    info.iterations = iterations;
    info.residual = final_residual;
    info.solve_time = solve_time;
    info.converged = converged;

    if (log_output || params.verbose)
    {
        std::cout << "[Poisson] Size: " << matrix.m()
                  << ", iterations: " << iterations
                  << ", residual: " << final_residual
                  << ", time: " << solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Legacy interface with default parameters
// ============================================================================
SolverInfo solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    // Default Poisson parameters (SPD system: CG + SSOR)
    LinearSolverParams default_params;
    default_params.type = LinearSolverParams::Type::CG;
    default_params.preconditioner = LinearSolverParams::Preconditioner::SSOR;
    default_params.rel_tolerance = 1e-8;
    default_params.abs_tolerance = 1e-12;
    default_params.max_iterations = 2000;
    default_params.ssor_omega = 1.2;
    default_params.use_iterative = true;
    default_params.fallback_to_direct = true;
    default_params.verbose = verbose;

    return solve_poisson_system(matrix, rhs, solution, constraints,
                                default_params, /*log_output=*/true);
}