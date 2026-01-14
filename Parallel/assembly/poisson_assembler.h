// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson Assembly (PARALLEL)
//
// PAPER EQUATION 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// OPTIMIZATION: The Laplacian matrix (∇φ, ∇χ) is CONSTANT - it doesn't depend
// on θ, M, or time. Only the RHS (h_a - M, ∇χ) changes each timestep.
//
// Usage:
//   // At setup (once):
//   assemble_poisson_matrix<dim>(phi_dof, constraints, matrix);
//
//   // Each timestep:
//   assemble_poisson_rhs<dim>(phi_dof, M_dof, Mx, My, params, time, constraints, rhs);
//   solve_poisson_system(matrix, rhs, solution, ...);
//
// PARALLEL VERSION:
//   - Uses Trilinos matrix/vectors
//   - Only assembles locally owned cells
//   - Uses distribute_local_to_global for constraint handling
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_ASSEMBLER_H
#define POISSON_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/parameters.h"

/**
 * @brief Assemble the Poisson Laplacian matrix (ONCE at setup)
 *
 * Assembles: (∇φ, ∇χ) for all test/trial pairs
 *
 * This matrix is CONSTANT throughout the simulation since it only contains
 * the Laplacian operator with no dependence on solution fields.
 *
 * @param phi_dof_handler  DoFHandler for φ
 * @param phi_constraints  Constraints for φ (hanging nodes, BCs)
 * @param phi_matrix       [OUT] System matrix (Trilinos)
 */
template <int dim>
void assemble_poisson_matrix(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix);

/**
 * @brief Assemble the Poisson RHS only (EVERY timestep)
 *
 * Assembles: (h_a - M^k, ∇χ) + MMS source terms
 *
 * This is the only part that changes each timestep due to:
 *   - Applied field h_a (time-dependent ramp)
 *   - Magnetization M^k (from previous solve)
 *
 * @param phi_dof_handler  DoFHandler for φ
 * @param M_dof_handler    DoFHandler for M (DG)
 * @param mx_solution      Mx component (ghosted, for reading)
 * @param my_solution      My component (ghosted, for reading)
 * @param params           Simulation parameters
 * @param current_time     Current time (for applied field ramp, MMS)
 * @param phi_constraints  Constraints for φ
 * @param phi_rhs          [OUT] RHS vector (Trilinos, owned)
 */
template <int dim>
void assemble_poisson_rhs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& my_solution,
    const Parameters& params,
    double current_time,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::MPI::Vector& phi_rhs);

/**
 * @brief Assemble full Poisson system (LEGACY - assembles both matrix and RHS)
 *
 * For backward compatibility. Prefer using assemble_poisson_matrix() once
 * at setup and assemble_poisson_rhs() each timestep for better performance.
 *
 * @param phi_dof_handler  DoFHandler for φ
 * @param M_dof_handler    DoFHandler for M (DG)
 * @param mx_solution      Mx component (ghosted, for reading)
 * @param my_solution      My component (ghosted, for reading)
 * @param params           Simulation parameters
 * @param current_time     Current time (for MMS/applied field)
 * @param phi_constraints  Constraints for φ
 * @param phi_matrix       [OUT] System matrix (Trilinos)
 * @param phi_rhs          [OUT] RHS vector (Trilinos, owned)
 */
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& my_solution,
    const Parameters& params,
    double current_time,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix,
    dealii::TrilinosWrappers::MPI::Vector& phi_rhs);

#endif // POISSON_ASSEMBLER_H