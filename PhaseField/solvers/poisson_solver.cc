// ============================================================================
// solvers/poisson_solver.cc - Magnetostatic Poisson solver (CORRECTED)
//
// Solves: (∇φ, ∇χ) = (h_a - m, ∇χ)  with Neumann BC
//
// The system is SPD (simple Laplacian, after fixing constant).
// CG + SSOR is efficient and robust for this problem.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#include "solvers/poisson_solver.h"

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <iostream>

// Set to true for faster iterative solver, false for robust direct solver
#define USE_ITERATIVE_POISSON_SOLVER true

void solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    // Handle zero RHS case (can happen at t=0 when field is off)
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        if (verbose)
        {
            std::cout << "[Poisson] Zero RHS, solution set to zero\n";
        }
        return;
    }

#if USE_ITERATIVE_POISSON_SOLVER
    // ========================================================================
    // Iterative solver: CG + SSOR
    // Good for SPD systems like Poisson with Neumann BC (after fixing constant)
    // ========================================================================

    // Tolerance: 1e-8 relative to RHS norm is plenty accurate
    // (1e-10 was too strict and caused false "no convergence" warnings)
    const double tol = 1e-8 * rhs_norm;
    const unsigned int max_iter = 2000;
    dealii::SolverControl solver_control(max_iter, tol);
    dealii::SolverCG<dealii::Vector<double>> solver(solver_control);

    // SSOR preconditioner
    dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
    preconditioner.initialize(matrix, 1.2);  // relaxation parameter

    try
    {
        solver.solve(matrix, solution, rhs, preconditioner);

        if (verbose)
        {
            std::cout << "[Poisson] CG converged in "
                      << solver_control.last_step() << " iterations, "
                      << "residual = " << solver_control.last_value() << "\n";
        }
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        std::cerr << "[Poisson] WARNING: CG did not converge after "
                  << e.last_step << " iterations. "
                  << "Residual = " << e.last_residual << "\n";
        std::cerr << "          Falling back to direct solver.\n";

        // Fallback to direct solver
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(solution, rhs);
    }

#else
    // ========================================================================
    // Direct solver: UMFPACK (robust but slow)
    // ========================================================================
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(solution, rhs);

    if (verbose)
    {
        std::cout << "[Poisson] Direct solve (UMFPACK)\n";
    }
#endif

    // Apply constraints (distribute values from constrained DoFs)
    constraints.distribute(solution);
}