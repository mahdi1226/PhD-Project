// ============================================================================
// solvers/poisson_solver.h - Parallel Magnetostatic Poisson Solver
//
// Uses CG + AMG (Trilinos) for distributed systems.
//
// OPTIMIZATION: Since the Poisson matrix is CONSTANT (only RHS changes),
// we cache the AMG preconditioner and reuse it across timesteps.
// ============================================================================
#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/index_set.h>

#include "solvers/solver_info.h"
#include "utilities/parameters.h"

#include <mpi.h>

/**
 * @brief Poisson solver with cached AMG preconditioner
 *
 * Since the Poisson Laplacian matrix is constant throughout the simulation,
 * we initialize the AMG preconditioner ONCE and reuse it for all solves.
 * This saves significant setup time per timestep.
 *
 * Usage:
 *   PoissonSolver solver(params, locally_owned, mpi_comm);
 *   solver.initialize(matrix);  // ONCE at setup - builds AMG
 *
 *   // Each timestep:
 *   solver.solve(rhs, solution, constraints);
 */
class PoissonSolver
{
public:
    /**
     * @brief Constructor
     */
    PoissonSolver(const LinearSolverParams& params,
                  const dealii::IndexSet& locally_owned,
                  MPI_Comm mpi_comm);

    /**
     * @brief Initialize preconditioner with system matrix (ONCE at setup)
     *
     * Builds AMG preconditioner. Call this once after assembling the matrix.
     */
    void initialize(const dealii::TrilinosWrappers::SparseMatrix& matrix);

    /**
     * @brief Solve the system (reuses cached preconditioner)
     */
    SolverInfo solve(const dealii::TrilinosWrappers::MPI::Vector& rhs,
                     dealii::TrilinosWrappers::MPI::Vector& solution,
                     const dealii::AffineConstraints<double>& constraints,
                     bool verbose = false);

    /**
     * @brief Check if preconditioner is initialized
     */
    bool is_initialized() const { return initialized_; }

private:
    const LinearSolverParams& params_;
    const dealii::IndexSet& locally_owned_;
    MPI_Comm mpi_comm_;

    const dealii::TrilinosWrappers::SparseMatrix* matrix_ptr_;
    dealii::TrilinosWrappers::PreconditionAMG amg_preconditioner_;
    bool initialized_;
};

/**
 * @brief Legacy free function interface (creates preconditioner each call)
 *
 * For backward compatibility. Prefer using PoissonSolver class for
 * better performance with cached preconditioner.
 */
SolverInfo solve_poisson_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& locally_owned,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm,
    bool verbose = false);

#endif // POISSON_SOLVER_H