// ============================================================================
// nsch_verification.h - Verification metrics for coupled NS-CH solver
// ============================================================================
#ifndef NSCH_VERIFICATION_H
#define NSCH_VERIFICATION_H

#include <deal.II/dofs/dof_handler.h>
#include "utilities/nsch_linear_algebra.h"

/**
 * @brief Verification metrics for coupled NS-CH system
 *
 * These metrics verify the solver WITHOUT requiring an exact solution:
 *
 * From Cahn-Hilliard:
 *   1. Mass conservation: ∫c dΩ = const
 *   2. CH energy: E_CH = ∫[½λ|∇c|² + W(c)] dΩ
 *   3. Bounds: c ∈ [-1, 1]
 *
 * From Navier-Stokes:
 *   4. Divergence: ‖∇·u‖_L2 (mass conservation for incompressible flow)
 *   5. Kinetic energy: E_K = ½∫|u|² dΩ
 *
 * Coupled:
 *   6. Total energy: E_total = E_K + E_CH (should dissipate)
 */
struct NSCHVerificationMetrics
{
    // Cahn-Hilliard metrics
    double mass;                // ∫c dΩ
    double ch_energy;           // E_CH = ∫[½λ|∇c|² + W(c)] dΩ
    double c_min;               // min(c)
    double c_max;               // max(c)

    // Navier-Stokes metrics
    double divergence_L2;       // ‖∇·u‖_L2
    double kinetic_energy;      // E_K = ½∫|u|² dΩ
    double u_max;               // max(|u|)

    // Coupled metrics
    double total_energy;        // E_total = E_K + E_CH
    double energy_rate;         // dE_total/dt (should be ≤ 0)

    // Interface metrics
    double interface_area;      // ∫|∇c| dΩ

    // Stability metrics
    double cfl_number;          // max(|u|) * dt / h
};

/**
 * @brief Compute all verification metrics
 */
template <int dim>
NSCHVerificationMetrics compute_nsch_metrics(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution,
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution,
    double                         lambda,
    double                         dt,
    double                         h,
    double                         old_total_energy);

// ============================================================================
// Individual metric functions
// ============================================================================

/**
 * @brief Compute mass: ∫c dΩ
 */
template <int dim>
double compute_mass(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution);

/**
 * @brief Compute Cahn-Hilliard energy: E_CH = ∫[½λ|∇c|² + W(c)] dΩ
 */
template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution,
    double                         lambda);

/**
 * @brief Compute min/max of concentration
 */
template <int dim>
std::pair<double, double> compute_c_bounds(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution);

/**
 * @brief Compute divergence: ‖∇·u‖_L2
 */
template <int dim>
double compute_divergence_L2(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution);

/**
 * @brief Compute kinetic energy: E_K = ½∫|u|² dΩ
 */
template <int dim>
double compute_kinetic_energy(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution);

/**
 * @brief Compute max velocity magnitude
 */
template <int dim>
double compute_max_velocity(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution);

/**
 * @brief Compute interface area: ∫|∇c| dΩ
 */
template <int dim>
double compute_interface_area(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution);

// ============================================================================
// Output functions
// ============================================================================

/**
 * @brief Print verification header for time stepping output
 */
void print_nsch_verification_header();

/**
 * @brief Print single line of verification metrics
 */
void print_nsch_verification_line(unsigned int step, double time,
                                   const NSCHVerificationMetrics& m);

/**
 * @brief Print detailed verification summary
 */
void print_nsch_verification_summary(const NSCHVerificationMetrics& m);

/**
 * @brief Check if solution is healthy (no NaN, no blow-up)
 */
bool check_nsch_health(const NSCHVerificationMetrics& m);

#endif // NSCH_VERIFICATION_H