// ============================================================================
// solvers/poisson_solver.cc - Magnetostatic Poisson Solver
//
// Solves: (∇φ, ∇χ) = (h_a - m, ∇χ)  with Neumann BC
//
// The system is SPD. CG + SSOR is efficient and robust.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/poisson_solver.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <chrono>

void solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    // ----- Solver configuration -----
    const bool use_iterative = true;
    const double rel_tolerance = 1e-8;
    const double abs_tolerance = 1e-12;
    const unsigned int max_iterations = 2000;
    const double ssor_omega = 1.2;
    const bool fallback_to_direct = true;
    const bool log_output = true;
    // --------------------------------

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        if (log_output)
            std::cout << "[Poisson] Zero RHS, solution set to zero\n";
        return;
    }

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();
    bool converged = false;
    unsigned int iterations = 0;
    double final_residual = 0.0;

    if (use_iterative)
    {
        const double tol = std::max(abs_tolerance, rel_tolerance * rhs_norm);

        dealii::SolverControl solver_control(max_iterations, tol);
        dealii::SolverCG<dealii::Vector<double>> solver(solver_control);

        dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(matrix, ssor_omega);

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

            if (fallback_to_direct)
                std::cerr << "[Poisson] Falling back to direct solver.\n";
        }
    }

    if (!converged)
    {
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(solution, rhs);
        converged = true;
        iterations = 1;
        final_residual = 0.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    if (log_output || verbose)
    {
        std::cout << "[Poisson] Size: " << matrix.m()
                  << ", iterations: " << iterations
                  << ", residual: " << final_residual
                  << ", time: " << solve_time << "s\n";
    }
}