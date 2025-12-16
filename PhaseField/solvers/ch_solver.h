// ============================================================================
// solvers/ch_solver.h - Cahn-Hilliard Linear Solver
//
// Solves the coupled (θ, ψ) system using GMRES + ILU iterative solver.
// Falls back to UMFPACK direct solver if iterative solver fails.
//
// The CH system is nonsymmetric due to convection terms, hence GMRES.
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
 * @brief Solve the Cahn-Hilliard linear system
 *
 * Solves the coupled system and extracts θ, ψ solutions.
 *
 * Solver: GMRES + ILU (with UMFPACK fallback)
 *
 * The system is nonsymmetric due to:
 *   - Convection terms in θ equation
 *   - Off-diagonal coupling between θ and ψ
 *
 * @param matrix           System matrix (already condensed)
 * @param rhs              Right-hand side (already condensed)
 * @param constraints      Constraints for distribute() after solve
 * @param theta_to_ch_map  Index mapping: θ DoF → coupled system index
 * @param psi_to_ch_map    Index mapping: ψ DoF → coupled system index
 * @param theta_solution   [OUT] θ solution vector
 * @param psi_solution     [OUT] ψ solution vector
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