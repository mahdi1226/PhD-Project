// ============================================================================
// solvers/ns_solver.h - Navier-Stokes Linear Solver
//
// Implements three solvers:
//   1. Simple GMRES + ILU (baseline)
//   2. FGMRES + Block Schur preconditioner (following step-56)
//   3. Direct UMFPACK (fallback)
//
// Reference: deal.II step-22, step-56
// ============================================================================
#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>

#include <vector>

// Forward declaration
struct LinearSolverParams;

/**
 * @brief Simple GMRES + ILU solver (baseline)
 */
void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const LinearSolverParams& params,
    bool log_output = true);

/**
 * @brief Legacy interface (uses default parameters)
 */
void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief FGMRES + Block Schur preconditioner (following step-56)
 */
void solve_ns_system_schur(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::SparseMatrix<double>& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const LinearSolverParams& params,
    bool log_output = true);

/**
 * @brief Legacy Schur interface (uses default parameters)
 */
void solve_ns_system_schur(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::SparseMatrix<double>& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    bool verbose = false);

/**
 * @brief Direct solver (UMFPACK)
 */
void solve_ns_system_direct(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief Extract individual velocity/pressure solutions from coupled vector
 */
void extract_ns_solutions(
    const dealii::Vector<double>& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution);

/**
 * @brief Assemble pressure mass matrix (for Schur complement preconditioner)
 */
template <int dim>
void assemble_pressure_mass_matrix(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    dealii::SparsityPattern& sparsity,
    dealii::SparseMatrix<double>& mass_matrix);

#endif // NS_SOLVER_H