// ============================================================================
// solvers/ns_solver.h - Navier-Stokes Linear Solver
// ============================================================================
#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>

#include "solvers/solver_info.h"
#include "solvers/ns_block_preconditioner.h"

/**
 * @brief Simple GMRES + ILU solver
 */
SolverInfo solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    unsigned int max_iterations = 3000,
    double rel_tolerance = 1e-6,
    bool verbose = false);

/**
 * @brief FGMRES + Block Schur preconditioner (preconditioner passed in for caching)
 */
SolverInfo solve_ns_system_schur(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    BlockSchurPreconditioner& preconditioner,
    unsigned int max_iterations = 1000,
    double rel_tolerance = 1e-6,
    bool verbose = false);

/**
 * @brief Direct solver (UMFPACK)
 */
SolverInfo solve_ns_system_direct(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief Extract individual solutions from coupled NS solution
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
 * @brief Assemble pressure mass matrix for Schur preconditioner
 */
template <int dim>
void assemble_pressure_mass_matrix(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    dealii::SparsityPattern& sparsity,
    dealii::SparseMatrix<double>& mass_matrix);

#endif // NS_SOLVER_H