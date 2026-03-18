// ============================================================================
// solvers/magnetic_solver.h - Monolithic Magnetics Solver (PARALLEL)
//
// Solver for the combined M+φ block system.
// Supports:
//   - Direct: MUMPS → SuperLU_DIST → KLU cascade
//   - Iterative: GMRES + AMG (or ILU)
//   - Block preconditioned: FGMRES + block-triangular PC (Phase 2)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_SOLVER_H
#define MAGNETIC_SOLVER_H

#include "solvers/solver_info.h"

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>

/**
 * @brief Solver for the monolithic magnetics system (M+φ, PARALLEL)
 *
 * Direct path: MUMPS → SuperLU_DIST → KLU cascade
 * Iterative path: GMRES + AMG (or ILU fallback)
 * Block PC path: FGMRES + block-triangular [A_M, C_Mφ; 0, A_φ]
 */
template <int dim>
class MagneticSolver
{
public:
    MagneticSolver(
        const dealii::IndexSet& locally_owned,
        MPI_Comm mpi_communicator);

    /**
     * @brief Set block structure information (required for block PC path)
     * @param n_M_dofs   Total M DoFs (Mx + My combined)
     * @param n_phi_dofs Total φ DoFs
     */
    void set_block_structure(dealii::types::global_dof_index n_M_dofs,
                             dealii::types::global_dof_index n_phi_dofs);

    /**
     * @brief Solve with solver selection via params
     */
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        bool verbose = false);

    /**
     * @brief Legacy interface (direct solver only)
     */
    void solve(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs);

    unsigned int last_n_iterations() const { return last_n_iterations_; }
    const SolverInfo& last_info() const { return last_info_; }

private:
    void solve_direct(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        bool verbose);

    void solve_iterative(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        bool verbose);

    void solve_block_preconditioned(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const LinearSolverParams& params,
        bool verbose);

    dealii::IndexSet locally_owned_;
    MPI_Comm mpi_communicator_;
    unsigned int last_n_iterations_;
    SolverInfo last_info_;

    // Block structure (set by set_block_structure)
    dealii::types::global_dof_index n_M_dofs_;
    dealii::types::global_dof_index n_phi_dofs_;
    bool block_structure_set_;
};

#endif // MAGNETIC_SOLVER_H
