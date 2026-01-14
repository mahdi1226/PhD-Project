// ============================================================================
// solvers/poisson_solver.h - Magnetostatic Poisson Solver
//
// UPDATED: Now returns SolverInfo with iterations/residual/time
// ============================================================================
#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "solvers/solver_info.h"

// Forward declaration
struct LinearSolverParams;

/**
 * @brief Solve Poisson system and return solver statistics
 */
SolverInfo solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const LinearSolverParams& params,
    bool log_output = true);

/**
 * @brief Legacy interface (default parameters)
 */
SolverInfo solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = true);

#endif // POISSON_SOLVER_H