// ============================================================================
// solvers/poisson_solver.h - Magnetostatic Poisson solver interface
//
// Options: CG+SSOR (fast) or UMFPACK (robust)
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
 * Uses CG+SSOR for SPD systems (fast), or UMFPACK for robustness.
 *
 * @param matrix       System matrix (μ(θ)∇φ, ∇ψ)
 * @param rhs          Right-hand side (typically zero)
 * @param solution     [OUT] Magnetic potential φ
 * @param constraints  Dirichlet BCs for applied field
 */
void solve_poisson_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints);

#endif // POISSON_SOLVER_H