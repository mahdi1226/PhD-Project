// ============================================================================
// solvers/ns_solver.h - Parallel Navier-Stokes Solver Interface
//
// FIX (2026-01-17): Added dt parameter for proper Schur complement scaling
// ============================================================================

#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include "solvers/solver_info.h"  // SolverInfo, LinearSolverParams

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>

#include <mpi.h>
#include <vector>

// ============================================================================
// Block Schur preconditioned FGMRES solver (recommended for saddle-point)
//
// Parameters:
//   dt > 0:  Unsteady problem, Schur scaling alpha = nu + 1/dt
//   dt <= 0: Steady problem, Schur scaling alpha = nu
// ============================================================================
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
    double viscosity,
    double dt = -1.0,      // NEW: time step for Schur scaling (negative = steady)
    bool verbose = true);

// ============================================================================
// Direct solver with pressure pinning (recommended)
// ============================================================================
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    MPI_Comm mpi_comm,
    bool verbose = true);

// ============================================================================
// Direct solver without pressure pinning (legacy, may fail on singular systems)
// ============================================================================
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    MPI_Comm mpi_comm,
    bool verbose = true);

// ============================================================================
// Extract component solutions from monolithic NS solution
// ============================================================================
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

// ============================================================================
// Assemble pressure mass matrix (works for both CG and DG pressure)
// ============================================================================
template <int dim>
void assemble_pressure_mass_matrix_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    const dealii::IndexSet& p_owned,
    const dealii::IndexSet& p_relevant,
    dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    MPI_Comm mpi_comm);

// ============================================================================
// Unified NS solver with auto-selection between direct and iterative
//
// Parameters:
//   viscosity:        Kinematic viscosity nu
//   dt:               Time step (dt > 0 for unsteady, dt <= 0 for steady)
//   verbose:          Print solver progress
//   force_direct:     Always use direct solver
//   force_iterative:  Always use iterative solver
//   direct_threshold: Use direct if n_dofs < threshold (default 50000)
// ============================================================================
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
    double viscosity,
    double dt = -1.0,                    // NEW: time step for Schur scaling
    bool verbose = true,
    bool force_direct = false,
    bool force_iterative = false,
    unsigned int direct_threshold = 50000);

#endif // NS_SOLVER_H