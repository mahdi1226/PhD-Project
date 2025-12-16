// ============================================================================
// solvers/poisson_solver.h - Magnetostatic Poisson Solver
//
// Solves: (∇φ, ∇χ) = (h_a - m, ∇χ)  with Neumann BC
//
// Options: CG + SSOR (fast, SPD) with UMFPACK fallback
//
// Note: Pure Neumann problem - the constant is fixed by pinning one DoF.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

/**
 * @brief Solve the Poisson system for magnetic potential
 *
 * Solves: (∇φ, ∇χ) = (h_a - m, ∇χ)
 *
 * The system is SPD (after fixing the constant via constraints),
 * so CG with SSOR preconditioner is efficient.
 *
 * @param matrix       System matrix (∇φ, ∇χ) - simple Laplacian stiffness
 * @param rhs          Right-hand side (h_a - m, ∇χ)
 * @param solution     [OUT] Magnetic potential φ
 * @param constraints  Constraints (hanging nodes + pinned DoF)
 * @param verbose      Print solver info
 */
void solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose = false);

#endif // POISSON_SOLVER_H