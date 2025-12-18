// ============================================================================
// diagnostics/mms_runtime.h - MMS Runtime Helpers
//
// Runtime support functions for MMS verification mode.
// These are called from phase_field.cc during time stepping.
//
// Design: Free functions with explicit parameters to avoid coupling to
// PhaseFieldProblem internals. Core files only need minimal changes.
// ============================================================================
#ifndef MMS_RUNTIME_H
#define MMS_RUNTIME_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <vector>

// ============================================================================
// Update CH boundary constraints for MMS at new time level
//
// Call this at the START of each time step when MMS mode is enabled.
// Updates theta_constraints and psi_constraints with time-dependent BCs.
// ============================================================================
template <int dim>
void update_ch_mms_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double time);

// ============================================================================
// Rebuild combined CH constraints from individual θ, ψ constraints
//
// Call this AFTER update_ch_mms_constraints to sync the coupled system.
// ============================================================================
template <int dim>
void rebuild_ch_combined_constraints(
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    unsigned int n_theta,
    unsigned int n_psi,
    dealii::AffineConstraints<double>& ch_combined_constraints);

// ============================================================================
// Convenience function: Update all CH MMS constraints in one call
//
// Combines update_ch_mms_constraints + rebuild_ch_combined_constraints.
// This is the recommended function to call from phase_field.cc.
// ============================================================================
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

#endif // MMS_RUNTIME_H