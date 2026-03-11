// ============================================================================
// solvers/magnetic_solver.h - Monolithic Magnetics Solver (PARALLEL)
//
// MUMPS direct solver for the combined M+phi block system.
// No preconditioner needed — direct solve of the nonsymmetric system.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_SOLVER_H
#define MAGNETIC_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>

/**
 * @brief MUMPS direct solver for the monolithic magnetics system (PARALLEL)
 *
 * Single solve for the combined M+phi system (one RHS, one solution vector).
 * Falls back to SuperLU_DIST then KLU if MUMPS is unavailable.
 */
template <int dim>
class MagneticSolver
{
public:
    MagneticSolver(
        const dealii::IndexSet& locally_owned,
        MPI_Comm mpi_communicator);

    /**
     * @brief Solve system_matrix * solution = rhs
     */
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs);

    unsigned int last_n_iterations() const { return last_n_iterations_; }

private:
    dealii::IndexSet locally_owned_;
    MPI_Comm mpi_communicator_;
    unsigned int last_n_iterations_;
};

#endif // MAGNETIC_SOLVER_H
