// ============================================================================
// solvers/magnetic_solver.h - Monolithic Magnetics Solver (PARALLEL)
//
// Two solver paths:
//   1. Direct: MUMPS → SuperLU_DIST → KLU cascade (default)
//   2. Iterative: GMRES + ILU(k) on the full monolithic system, with the
//      preconditioner cached across non-AMR steps.
//
// ILU fill level k is configurable via LinearSolverParams::ilu_fill (set in
// utilities/parameters.h). For dome (h_a only, phi block trivial), k=0 works.
// For hedgehog/Rosensweig (full Laplacian phi block), k=4-5 is typically
// needed, with relative tolerance ~1e-7.
//
// Note: a pure block preconditioner with AMG on phi was attempted (see
// magnetic_block_preconditioner.{h,cc}) but the deal.II SparseMatrix iterator
// misbehaves at the M/phi boundary in this version, blocking sub-matrix
// extraction. ILU(k) on the monolithic system is the practical alternative
// and produces correct results when GMRES converges.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_SOLVER_H
#define MAGNETIC_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include "solvers/solver_info.h"

#include <memory>
#include <mpi.h>

template <int dim>
class MagneticSolver
{
public:
    MagneticSolver(
        const dealii::IndexSet& locally_owned,
        MPI_Comm mpi_communicator);

    /**
     * @brief Solve system_matrix * solution = rhs.
     */
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        bool rebuild_preconditioner);

    /// Backward-compatible overload (defaults to direct).
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs);

    unsigned int last_n_iterations() const { return last_n_iterations_; }
    bool last_used_direct() const { return last_used_direct_; }

    /// Drop the cached preconditioner (forces rebuild on next iterative solve).
    void invalidate_preconditioner() { cached_ilu_.reset(); }

private:
    bool solve_direct(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs);

    bool solve_iterative(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        bool rebuild_preconditioner);

    dealii::IndexSet locally_owned_;
    MPI_Comm mpi_communicator_;
    unsigned int last_n_iterations_;
    bool last_used_direct_;

    // Cached ILU preconditioner (rebuild on AMR; reuse otherwise).
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionILU> cached_ilu_;
};

#endif // MAGNETIC_SOLVER_H
