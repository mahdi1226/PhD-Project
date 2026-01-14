// ============================================================================
// solvers/magnetization_solver.h - DG Magnetization Solver (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses Trilinos matrix/vectors
//   - GMRES + ILU(0) (exploits DG block-diagonal structure)
//   - Direct solver fallback (Amesos)
//
// OPTIMIZATION: DG mass matrices are block-diagonal. ILU(0) captures this
// structure well and provides much better convergence than point-Jacobi.
// Additionally, the preconditioner is initialized ONCE and reused for
// both Mx and My solves (since they share the same matrix).
//
// Solves TWO scalar DG systems (Mx and My) with the SAME matrix.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_SOLVER_H
#define MAGNETIZATION_SOLVER_H

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/index_set.h>

#include "utilities/parameters.h"

#include <mpi.h>

/**
 * @brief Solver for the DG magnetization system (PARALLEL)
 *
 * Solves Mx and My separately using the same system matrix.
 * Uses ILU(0) preconditioner which works well with DG block-diagonal
 * matrices and is initialized once and reused for both solves.
 *
 * Usage:
 *   MagnetizationSolver<2> solver(params, M_locally_owned, mpi_comm);
 *   solver.initialize(system_matrix);  // Initializes preconditioner once
 *   solver.solve(Mx, rhs_x);           // Reuses preconditioner
 *   solver.solve(My, rhs_y);           // Reuses preconditioner
 */
template <int dim>
class MagnetizationSolver
{
public:
    /**
     * @brief Constructor
     *
     * @param params            LinearSolverParams
     * @param locally_owned     IndexSet of locally owned DoFs
     * @param mpi_communicator  MPI communicator
     */
    MagnetizationSolver(
        const LinearSolverParams& params,
        const dealii::IndexSet& locally_owned,
        MPI_Comm mpi_communicator);

    /**
     * @brief Initialize solver and preconditioner with system matrix
     *
     * For iterative solver: initializes ILU(0) preconditioner
     * which is reused for both Mx and My solves.
     */
    void initialize(const dealii::TrilinosWrappers::SparseMatrix& system_matrix);

    /**
     * @brief Solve system_matrix * solution = rhs
     *
     * Reuses the preconditioner initialized in initialize().
     */
    void solve(dealii::TrilinosWrappers::MPI::Vector& solution,
               const dealii::TrilinosWrappers::MPI::Vector& rhs);

    /**
     * @brief Get number of iterations from last solve (1 for direct solver)
     */
    unsigned int last_n_iterations() const { return last_n_iterations_; }

private:
    const LinearSolverParams& params_;
    const dealii::IndexSet& locally_owned_;
    MPI_Comm mpi_communicator_;

    const dealii::TrilinosWrappers::SparseMatrix* matrix_ptr_;
    unsigned int last_n_iterations_;
    bool use_direct_;
    bool initialized_;

    // ILU preconditioner (initialized once, reused for Mx and My solves)
    dealii::TrilinosWrappers::PreconditionILU ilu_preconditioner_;
    bool preconditioner_initialized_;
};

#endif // MAGNETIZATION_SOLVER_H