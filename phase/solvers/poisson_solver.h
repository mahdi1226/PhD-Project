// ============================================================================
// solvers/poisson_solver.h - Magnetostatic Poisson solver interface
// ============================================================================
#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

/**
 * @brief Solve the Poisson system for magnetic potential
 *
 * Uses direct solver (UMFPACK) for robustness with variable coefficients
 *
 * @param matrix       System matrix (μ(c)∇φ, ∇ψ)
 * @param rhs          Right-hand side (typically zero)
 * @param solution     Output: magnetic potential φ
 * @param constraints  Dirichlet BCs for applied field
 */
void solve_poisson_system(
    const dealii::SparseMatrix<double>&      matrix,
    const dealii::Vector<double>&            rhs,
    dealii::Vector<double>&                  solution,
    const dealii::AffineConstraints<double>& constraints);

#endif // POISSON_SOLVER_H