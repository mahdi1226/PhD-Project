// ============================================================================
// solvers/ns_solver.h - Parallel Navier-Stokes Solver Interface
//
// UPDATE (2026-01-12): Added unified solve_ns_system() that auto-selects
// direct vs iterative based on problem size.
//
// Threshold: < 500K DoFs → Direct (MUMPS), >= 500K → Iterative (Block Schur)
// ============================================================================
#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include "solvers/solver_info.h"

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>

#include <vector>
#include <mpi.h>

// ============================================================================
// Default threshold for auto-selection (can be overridden)
// Based on benchmarks: direct is faster up to ~500K DoFs
// ============================================================================
constexpr unsigned int NS_DIRECT_THRESHOLD = 500000;

/**
 * @brief Unified NS solver with auto-selection (RECOMMENDED)
 *
 * Automatically selects:
 *   - Direct solver (MUMPS) for DoFs < threshold (default 500K)
 *   - Iterative solver (Block Schur) for DoFs >= threshold
 *
 * @param force_direct   If true, always use direct solver regardless of size
 * @param force_iterative If true, always use iterative solver regardless of size
 * @param direct_threshold DoF count below which direct solver is used (default 500K)
 */
SolverInfo solve_ns_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& vel_owned,
    const dealii::IndexSet& p_owned,
    MPI_Comm mpi_comm,
    double viscosity = 1.0,
    bool verbose = false,
    bool force_direct = false,
    bool force_iterative = false,
    unsigned int direct_threshold = NS_DIRECT_THRESHOLD);

/**
 * @brief Solve NS system with Block Schur preconditioner (ITERATIVE)
 *
 * Uses FGMRES with block Schur preconditioner:
 *   - AMG for velocity block
 *   - Pressure mass matrix for Schur approximation
 *
 * Recommended for refinement >= 5 where direct solver is too slow/OOM.
 */
SolverInfo solve_ns_system_schur_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& vel_owned,
    const dealii::IndexSet& p_owned,
    MPI_Comm mpi_comm,
    double viscosity = 1.0,
    bool verbose = false);

/**
 * @brief Solve NS system with direct solver + pressure pinning (DIRECT)
 *
 * Uses Trilinos Amesos direct solver with pressure pinning to remove null space.
 * Tries MUMPS first, then UMFPACK, SuperLU, KLU as fallbacks.
 *
 * Recommended for refinement levels <= 4 (< 500k DoFs)
 */
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    MPI_Comm mpi_comm,
    bool verbose = false);

/**
 * @brief Legacy direct solver (NO PRESSURE PINNING - may fail)
 */
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    MPI_Comm mpi_comm,
    bool verbose = false);

/**
 * @brief Extract individual field solutions from coupled NS solution
 */
void extract_ns_solutions_parallel(
    const dealii::TrilinosWrappers::MPI::Vector& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ux_owned,
    const dealii::IndexSet& uy_owned,
    const dealii::IndexSet& p_owned,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& ns_relevant,
    dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    dealii::TrilinosWrappers::MPI::Vector& p_solution,
    MPI_Comm mpi_comm);

/**
 * @brief Assemble pressure mass matrix for Schur preconditioner
 */
template <int dim>
void assemble_pressure_mass_matrix_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    const dealii::IndexSet& p_owned,
    const dealii::IndexSet& p_relevant,
    dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    MPI_Comm mpi_comm);

#endif // NS_SOLVER_H