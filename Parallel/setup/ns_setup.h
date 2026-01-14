// ============================================================================
// setup/ns_setup.h - Navier-Stokes Setup Functions (Parallel)
//
// Production setup functions for NS system:
//   - setup_ns_coupled_system_parallel: Build coupled saddle-point system
//   - setup_ns_velocity_constraints_parallel: Velocity BCs (homogeneous Dirichlet)
//   - setup_ns_pressure_constraints_parallel: Pressure constraints (pin DoF 0)
//
// These functions are used by BOTH production code and MMS tests.
// ============================================================================
#ifndef NS_SETUP_H
#define NS_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <vector>
#include <mpi.h>

/**
 * @brief Setup the coupled NS system (sparsity pattern, constraints, DoF maps)
 *
 * Creates the monolithic saddle-point system structure:
 *   [A   B^T] [u]   [f]
 *   [B   0  ] [p] = [0]
 *
 * @param ux_dof_handler DoF handler for x-velocity
 * @param uy_dof_handler DoF handler for y-velocity
 * @param p_dof_handler DoF handler for pressure
 * @param ux_constraints Constraints for x-velocity (input)
 * @param uy_constraints Constraints for y-velocity (input)
 * @param p_constraints Constraints for pressure (input)
 * @param ux_to_ns_map Output: maps local ux DoF to coupled system DoF
 * @param uy_to_ns_map Output: maps local uy DoF to coupled system DoF
 * @param p_to_ns_map Output: maps local p DoF to coupled system DoF
 * @param ns_owned Output: locally owned DoFs for coupled system
 * @param ns_relevant Output: locally relevant DoFs for coupled system
 * @param ns_constraints Output: constraints for coupled system
 * @param ns_sparsity Output: sparsity pattern for coupled system
 * @param mpi_comm MPI communicator
 * @param pcout Conditional output stream
 */
template <int dim>
void setup_ns_coupled_system_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& ux_constraints,
    const dealii::AffineConstraints<double>& uy_constraints,
    const dealii::AffineConstraints<double>& p_constraints,
    std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::IndexSet& ns_owned,
    dealii::IndexSet& ns_relevant,
    dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparsityPattern& ns_sparsity,
    MPI_Comm mpi_comm,
    dealii::ConditionalOStream& pcout);

/**
 * @brief Setup velocity constraints (hanging nodes + homogeneous Dirichlet on all boundaries)
 *
 * Applies u = 0 on boundary IDs 0-3 (all boundaries for rectangular domain).
 * Used by both production and MMS tests.
 */
template <int dim>
void setup_ns_velocity_constraints_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    dealii::AffineConstraints<double>& ux_constraints,
    dealii::AffineConstraints<double>& uy_constraints);

/**
 * @brief Setup pressure constraints (hanging nodes + pin DoF 0 for uniqueness)
 *
 * Pins pressure DoF 0 to zero to fix the constant.
 * Used by both production and MMS tests.
 */
template <int dim>
void setup_ns_pressure_constraints_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::AffineConstraints<double>& p_constraints);

#endif // NS_SETUP_H