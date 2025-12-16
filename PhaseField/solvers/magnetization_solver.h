// ============================================================================
// solvers/magnetization_solver.h - DG Magnetization System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c: Magnetization equation
//
// Solves TWO scalar DG systems (Mx and My) with the SAME matrix.
//
// The paper does not prescribe a specific solver. We provide:
//   - Direct (UMFPACK): robust, recommended for moderate problem sizes
//   - Iterative (GMRES): available for large problems
//
// NOTE: The system matrix includes face coupling from Eq. 57, so it is
// NOT block-diagonal. However, DG sparsity is still favorable.
//
// ============================================================================
#ifndef MAGNETIZATION_SOLVER_H
#define MAGNETIZATION_SOLVER_H

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/exceptions.h>

#include "utilities/parameters.h"

/**
 * @brief Solver for the DG magnetization system
 *
 * Solves Mx and My separately using the same system matrix.
 *
 * Usage:
 *   MagnetizationSolver solver(params);
 *   solver.initialize(system_matrix);  // Factorize once
 *   solver.solve(Mx, rhs_x);           // Solve for Mx
 *   solver.solve(My, rhs_y);           // Solve for My
 *
 * For time-dependent problems where U^{k-1} changes:
 *   - Call initialize() when matrix changes (new time step with different U)
 *   - Call solve() for Mx and My
 */
template <int dim>
class MagnetizationSolver
{
public:
    /**
     * @brief Constructor
     *
     * @param params  Simulation parameters (for solver settings)
     */
    explicit MagnetizationSolver(const Parameters& params);

    /**
     * @brief Initialize/factorize the system matrix
     *
     * For direct solver: performs LU factorization (UMFPACK)
     * For iterative: stores matrix reference
     *
     * Call this whenever the matrix changes (e.g., U^{k-1} updated).
     */
    void initialize(const dealii::SparseMatrix<double>& system_matrix);

    /**
     * @brief Solve system_matrix * solution = rhs
     *
     * @param solution  [OUT] Solution vector
     * @param rhs       Right-hand side vector
     */
    void solve(dealii::Vector<double>& solution,
               const dealii::Vector<double>& rhs) const;

    /**
     * @brief Get number of iterations (iterative solver only)
     *
     * Returns 1 for direct solver.
     */
    unsigned int last_n_iterations() const { return last_n_iterations_; }

private:
    const Parameters& params_;

    // Direct solver (UMFPACK)
    mutable dealii::SparseDirectUMFPACK direct_solver_;

    // Matrix pointer (non-owning; must outlive solver usage)
    const dealii::SparseMatrix<double>* matrix_ptr_;

    // Iteration count (meaningful for iterative solver only)
    mutable unsigned int last_n_iterations_;

    // Solver mode
    bool use_direct_;

    // Safety: ensures initialize() is called before solve()
    bool initialized_;
};

#endif // MAGNETIZATION_SOLVER_H