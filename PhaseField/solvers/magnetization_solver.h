// ============================================================================
// solvers/magnetization_solver.h - DG Magnetization System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c: Magnetization equation
//
// Solves TWO scalar DG systems (Mx and My) with the SAME matrix.
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
 *   MagnetizationSolver solver(params.solvers.magnetization);
 *   solver.initialize(system_matrix);
 *   solver.solve(Mx, rhs_x);
 *   solver.solve(My, rhs_y);
 */
template <int dim>
class MagnetizationSolver
{
public:
    /**
     * @brief Constructor
     *
     * @param params  LinearSolverParams for this solver
     */
    explicit MagnetizationSolver(const LinearSolverParams& params);

    /**
     * @brief Initialize/factorize the system matrix
     */
    void initialize(const dealii::SparseMatrix<double>& system_matrix);

    /**
     * @brief Solve system_matrix * solution = rhs
     */
    void solve(dealii::Vector<double>& solution,
               const dealii::Vector<double>& rhs) const;

    /**
     * @brief Get number of iterations (1 for direct solver)
     */
    unsigned int last_n_iterations() const { return last_n_iterations_; }

private:
    const LinearSolverParams& params_;
    mutable dealii::SparseDirectUMFPACK direct_solver_;
    const dealii::SparseMatrix<double>* matrix_ptr_;
    mutable unsigned int last_n_iterations_;
    bool use_direct_;
    bool initialized_;
};

#endif // MAGNETIZATION_SOLVER_H