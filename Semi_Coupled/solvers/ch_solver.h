// ============================================================================
// solvers/ch_solver.h - Parallel Cahn-Hilliard Solver Interface
//
// Uses Trilinos GMRES + AMG for distributed CH system.
// ============================================================================
#ifndef CH_SOLVER_H
#define CH_SOLVER_H

#include "solvers/solver_info.h"  // SolverInfo, LinearSolverParams

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <memory>
#include <vector>
#include <mpi.h>

/**
 * @brief Solve coupled CH system and extract θ, ψ solutions
 *
 * Solves the coupled Cahn-Hilliard system using GMRES + AMG preconditioner,
 * then extracts the phase field (θ) and chemical potential (ψ) components.
 *
 * @param matrix              Coupled CH system matrix
 * @param rhs                 Right-hand side vector
 * @param constraints         Affine constraints
 * @param ch_locally_owned    Locally owned DoFs for coupled system
 * @param theta_locally_owned Locally owned DoFs for θ
 * @param psi_locally_owned   Locally owned DoFs for ψ
 * @param theta_to_ch_map     Map from θ DoFs to coupled indices
 * @param psi_to_ch_map       Map from ψ DoFs to coupled indices
 * @param theta_solution      [OUT] Phase field solution
 * @param psi_solution        [OUT] Chemical potential solution
 * @param params              Solver parameters
 * @param mpi_comm            MPI communicator
 * @param verbose             Print diagnostics
 */
/**
 * @brief Solve CH system, optionally with cross-AMR preconditioner caching.
 *
 * @param cached_amg            in/out — caller-owned AMG handle. If null
 *                              or rebuild_preconditioner==true, AMG is
 *                              reinitialized from `matrix`. Otherwise the
 *                              cached preconditioner is reused (matrix
 *                              values may have changed but sparsity is
 *                              fixed between AMR events; GMRES absorbs
 *                              minor staleness via extra iterations).
 *                              Caller must reset() the unique_ptr after AMR.
 * @param rebuild_preconditioner if true, force AMG reinit even if cache exists.
 */
SolverInfo solve_ch_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& ch_locally_relevant,
    const dealii::IndexSet& theta_locally_owned,
    const dealii::IndexSet& psi_locally_owned,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const LinearSolverParams& params,
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionAMG>& cached_amg,
    bool rebuild_preconditioner,
    MPI_Comm mpi_comm,
    bool verbose = false);

/// Backward-compat overload for MMS tests and any callers that don't need
/// cross-AMR caching. Each call rebuilds AMG locally (matches pre-cache
/// behavior). Production callers should use the cached overload.
inline SolverInfo solve_ch_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& ch_locally_relevant,
    const dealii::IndexSet& theta_locally_owned,
    const dealii::IndexSet& psi_locally_owned,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm,
    bool verbose = false)
{
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionAMG> local_amg;
    return solve_ch_system(matrix, rhs, constraints,
        ch_locally_owned, ch_locally_relevant,
        theta_locally_owned, psi_locally_owned,
        theta_to_ch_map, psi_to_ch_map,
        theta_solution, psi_solution,
        params, local_amg, /*rebuild_preconditioner=*/true,
        mpi_comm, verbose);
}

#endif // CH_SOLVER_H