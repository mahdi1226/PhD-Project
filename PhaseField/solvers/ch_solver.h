// ============================================================================
// solvers/ch_solver.h - Cahn-Hilliard System Solver
//
// UPDATED: Now returns SolverInfo with iterations/residual/time
// ============================================================================
#ifndef CH_SOLVER_H
#define CH_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "solvers/solver_info.h"

// Forward declaration
struct LinearSolverParams;

/**
 * @brief Solve the coupled CH system and return solver statistics
 *
 * @return SolverInfo with iterations, residual, time
 */
SolverInfo solve_ch_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution,
    const LinearSolverParams& params,
    bool log_output = true);

/**
 * @brief Legacy interface (default parameters)
 */
SolverInfo solve_ch_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution);

#endif // CH_SOLVER_H