// ============================================================================
// mms/mms_runtime.h - MMS Runtime Helpers
//
// Runtime support functions for MMS verification mode.
// Free functions with explicit parameters - no coupling to PhaseFieldProblem.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MMS_RUNTIME_H
#define MMS_RUNTIME_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <vector>

// ============================================================================
// Constraint Updates
// ============================================================================

/// Update CH boundary constraints for MMS at new time level
template <int dim>
void update_ch_mms_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double time);

/// Rebuild combined CH constraints from individual θ, ψ constraints
template <int dim>
void rebuild_ch_combined_constraints(
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    unsigned int n_theta,
    unsigned int n_psi,
    dealii::AffineConstraints<double>& ch_combined_constraints);

/// Convenience: Update all CH MMS constraints in one call
template <int dim>
void update_all_ch_mms_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    double time);

// ============================================================================
// Error Computation
// ============================================================================

/// Compute and print all MMS errors (CH + Poisson + NS if enabled)
template <int dim>
void compute_mms_errors(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>* phi_solution,
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::DoFHandler<dim>* p_dof_handler,
    const dealii::Vector<double>* ux_solution,
    const dealii::Vector<double>* uy_solution,
    const dealii::Vector<double>* p_solution,
    double time,
    double L_y,
    double h_min,
    unsigned int refinement_level,
    bool enable_magnetic,
    bool enable_ns);

#endif // MMS_RUNTIME_H