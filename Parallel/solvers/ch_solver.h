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
#include <deal.II/lac/affine_constraints.h>

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
SolverInfo solve_ch_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& theta_locally_owned,
    const dealii::IndexSet& psi_locally_owned,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm,
    bool verbose = false);

#endif // CH_SOLVER_H