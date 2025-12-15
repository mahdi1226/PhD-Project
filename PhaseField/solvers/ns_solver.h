// ============================================================================
// solvers/ns_solver.h - Navier-Stokes Linear Solver
//
// Solves the saddle-point NS system:
//   [ A   B^T ] [ u ]   [ f ]
//   [ B   0   ] [ p ] = [ 0 ]
//
// Solver: UMFPACK direct solver (robust for saddle point systems)
//
// Workflow:
//   1. Assembler: condense(matrix, rhs) incorporates constraints
//   2. Solver: vmult() solves the modified system
//   3. Solver: distribute(solution) fixes constrained DoF values
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <vector>

/**
 * @brief Solve the Navier-Stokes linear system
 *
 * Uses UMFPACK direct solver for robustness with saddle-point systems.
 *
 * IMPORTANT: The matrix and rhs should already have constraints applied
 * via constraints.condense(matrix, rhs) in the assembler.
 *
 * @param matrix       System matrix (saddle point, already condensed)
 * @param rhs          Right-hand side (already condensed)
 * @param solution     [OUT] Solution (ux, uy, p concatenated)
 * @param constraints  Constraints for distribute() after solve
 * @param verbose      Print solver statistics
 */
void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

/**
 * @brief Extract individual field solutions from coupled NS solution
 *
 * @param ns_solution   Coupled solution vector
 * @param ux_to_ns_map  Index map for ux
 * @param uy_to_ns_map  Index map for uy
 * @param p_to_ns_map   Index map for p
 * @param ux_solution   [OUT] Velocity x-component
 * @param uy_solution   [OUT] Velocity y-component
 * @param p_solution    [OUT] Pressure
 */
void extract_ns_solutions(
    const dealii::Vector<double>& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution);

#endif // NS_SOLVER_H