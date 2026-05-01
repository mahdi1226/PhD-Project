// ============================================================================
// solvers/magnetic_solver.h - Monolithic Magnetics Solver (PARALLEL)
//
// Direct (default) and iterative (GMRES + cached block preconditioner) paths.
//
// The iterative path uses MagneticBlockPreconditioner: ILU(0) on the M block
// (mass-coefficient dominated) and AMG on the phi block (Laplacian).
// Sub-block extraction goes through the Trilinos Epetra backend (see
// magnetic_block_preconditioner.cc) to avoid a deal.II 9.7.1 SparseMatrix
// iterator bug at the M/phi block boundary.
//
// The cached preconditioner is rebuilt at AMR steps (when the PhaseFieldProblem
// recreates magnetic_solver_ inside setup_magnetic_system()) and reused
// between AMR events.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_SOLVER_H
#define MAGNETIC_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "solvers/solver_info.h"
#include "solvers/magnetic_block_preconditioner.h"

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
     *
     * @param params                 Solver configuration.
     * @param n_M_dofs               Boundary between M (DG vector) and phi
     *                               (CG scalar) blocks in the component_wise
     *                               renumbered monolithic system. Required for
     *                               the iterative path; ignored for direct.
     * @param rebuild_preconditioner If true (or no cache exists), rebuild the
     *                               block preconditioner before solving.
     */
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        dealii::types::global_dof_index n_M_dofs,
        bool rebuild_preconditioner);

    /// Backward-compatible overload (always direct, no n_M_dofs needed).
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs);

    unsigned int last_n_iterations() const { return last_n_iterations_; }
    bool last_used_direct() const { return last_used_direct_; }

    /// Drop the cached preconditioner (forces rebuild on next iterative solve).
    void invalidate_preconditioner() { cached_block_prec_.reset(); }

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
        dealii::types::global_dof_index n_M_dofs,
        bool rebuild_preconditioner);

    dealii::IndexSet locally_owned_;
    MPI_Comm mpi_communicator_;
    unsigned int last_n_iterations_;
    bool last_used_direct_;

    std::unique_ptr<MagneticBlockPreconditioner> cached_block_prec_;
};

#endif // MAGNETIC_SOLVER_H
