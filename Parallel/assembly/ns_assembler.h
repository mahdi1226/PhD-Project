// ============================================================================
// assembly/ns_assembler.h - Parallel Navier-Stokes Assembler (Production)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e-42f (discrete scheme)
//
// PRODUCTION assembler with:
//   - Skew-symmetric convection via skew_forms.h (Eq. 37)
//   - Kelvin force μ₀(M·∇)H via kelvin_force.h (Eq. 38)
//   - Optional MMS support via enable_mms flag
//
// FIX: Kelvin force now uses total field H = h_a + h_d (not just h_d = ∇φ)
//
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <functional>
#include <vector>
#include <mpi.h>

/**
 * @brief Basic NS assembly (no Kelvin force)
 */
template <int dim>
void assemble_ns_system_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0);

/**
 * @brief NS assembly with body force
 */
template <int dim>
void assemble_ns_system_with_body_force_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>& body_force,
    double current_time,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0);

/**
 * @brief NS assembly with Kelvin force (legacy - use ferrofluid version)
 */
template <int dim>
void assemble_ns_system_with_kelvin_force_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0);

/**
 * @brief Full ferrofluid NS assembly (Kelvin + Capillary + Gravity + Variable viscosity)
 *
 * Paper Eq. 14e RHS: μ₀(M·∇)H + (λ/ε)θ∇ψ + r·H(θ/ε)·g
 *
 * FIX: Now correctly computes H = h_a + h_d (total field from dipoles + demagnetizing)
 *
 * @param dipole_positions     Dipole positions for h_a computation
 * @param dipole_direction     Dipole direction (typically {0, 1} for vertical)
 * @param dipole_intensity_max Maximum dipole intensity
 * @param dipole_ramp_time     Ramp time for field intensity
 * @param current_time         Current simulation time
 * @param use_reduced_magnetic_field  If true, use H = h_a only (dome mode)
 */
template <int dim>
void assemble_ns_system_ferrofluid_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    // Kelvin force inputs (magnetic)
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    // Capillary force inputs (phase field)
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double lambda,
    double epsilon,
    // Variable viscosity inputs (Paper Eq. 17)
    double nu_water,
    double nu_ferro,
    // Gravity force inputs
    bool enable_gravity,
    double r,
    double gravity_mag,
    const dealii::Tensor<1, dim>& gravity_dir,
    // NEW: Dipole/applied field inputs for H = h_a + h_d
    const std::vector<dealii::Point<2>>& dipole_positions,
    const std::vector<double>& dipole_direction,
    double dipole_intensity_max,
    double dipole_ramp_time,
    double current_time,
    bool use_reduced_magnetic_field,
    // MMS options
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0);

#endif // NS_ASSEMBLER_H