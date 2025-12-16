// ============================================================================
// solvers/ns_solver.h - Navier-Stokes Linear Solver
//
// Three solver options:
//   1. solve_ns_system(): GMRES + ILU (simple, small problems)
//   2. solve_ns_system_schur(): FGMRES + Block Schur (recommended)
//   3. solve_ns_system_direct(): UMFPACK (fallback)
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

/**
 * @brief Solve NS system with ILU preconditioner (simple baseline)
 */
void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief Solve NS system with FGMRES + Block Schur preconditioner
 *
 * Following deal.II step-56 pattern. Recommended for refinement 4+.
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
 * @brief Solve NS system with direct solver (UMFPACK)
 */
void solve_ns_system_direct(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief Extract individual field solutions from coupled NS solution
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
 * @brief Assemble pressure mass matrix for Schur complement
 */
template <int dim>
void assemble_pressure_mass_matrix(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,  // ADD
    dealii::SparsityPattern& sparsity,
    dealii::SparseMatrix<double>& mass_matrix);


#endif // NS_SOLVER_H