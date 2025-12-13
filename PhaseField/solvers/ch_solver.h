// ============================================================================
// solvers/ch_solver.h - Cahn-Hilliard System Solver
//
// Free function interface - no circular dependencies.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_SOLVER_H
#define CH_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <vector>

/**
 * @brief Solve the coupled Cahn-Hilliard system
 *
 * Solves the linear system Ax = b using UMFPACK direct solver,
 * then extracts θ and ψ solutions from the coupled solution vector.
 *
 * @param matrix             System matrix (already condensed with constraints)
 * @param rhs                RHS vector (already condensed with constraints)
 * @param constraints        Combined constraints for the coupled system
 * @param theta_to_ch_map    Index mapping: θ DoF → coupled system index
 * @param psi_to_ch_map      Index mapping: ψ DoF → coupled system index
 * @param theta_solution     Output: phase field solution
 * @param psi_solution       Output: chemical potential solution
 */
void solve_ch_system(
    const dealii::SparseMatrix<double>&  matrix,
    const dealii::Vector<double>&        rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>&              theta_solution,
    dealii::Vector<double>&              psi_solution);

#endif // CH_SOLVER_H