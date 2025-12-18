// ============================================================================
// solvers/ch_solver.h - Cahn-Hilliard Coupled System Solver
//
// Solves the coupled (θ, ψ) system from CH equations.
// Uses GMRES + ILU (nonsymmetric system) with direct fallback.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_SOLVER_H
#define CH_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <vector>

// Forward declaration
struct LinearSolverParams;

/**
 * @brief Solve the coupled Cahn-Hilliard system
 *
 * @param matrix           Coupled system matrix
 * @param rhs              Right-hand side
 * @param constraints      Combined constraints for coupled system
 * @param theta_to_ch_map  DoF mapping: θ local → coupled global
 * @param psi_to_ch_map    DoF mapping: ψ local → coupled global
 * @param theta_solution   [OUT] Phase field θ
 * @param psi_solution     [OUT] Chemical potential ψ
 * @param params           Solver parameters
 * @param log_output       Print solver statistics
 */
void solve_ch_system(
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
 * @brief Legacy interface (uses default parameters)
 */
void solve_ch_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution);

#endif // CH_SOLVER_H