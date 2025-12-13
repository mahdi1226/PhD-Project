// ============================================================================
// solvers/poisson_solver.cc - Magnetostatic Poisson solver
//
// Options:
//   - CG + SSOR preconditioner (fast iterative, good for Poisson)
//   - UMFPACK direct solver (robust, slow for large problems)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#include "solvers/poisson_solver.h"

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

// Set to true for faster iterative solver, false for robust direct solver
#define USE_ITERATIVE_POISSON_SOLVER true

void solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints)
{
#if USE_ITERATIVE_POISSON_SOLVER
    // ========================================================================
    // Iterative solver: CG + SSOR (much faster for SPD systems like Poisson)
    // ========================================================================
    dealii::SolverControl solver_control(1000, 1e-10 * rhs.l2_norm());
    dealii::SolverCG<dealii::Vector<double>> solver(solver_control);

    // SSOR preconditioner (good for Poisson, cheap to setup)
    dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
    preconditioner.initialize(matrix, 1.2);  // relaxation parameter

    // Solve
    solver.solve(matrix, solution, rhs, preconditioner);

#else
    // ========================================================================
    // Direct solver: UMFPACK (robust but slow)
    // ========================================================================
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(solution, rhs);
#endif

    // Apply constraints (distribute Dirichlet values)
    constraints.distribute(solution);
}