// ============================================================================
// mms/mms_core/temporal_convergence.h - Temporal Convergence Tests
//
// Verifies O(τ) convergence for backward Euler time integration.
//
// Approach:
//   - Fix spatial mesh at a fine level (e.g., refinement 5)
//   - Run with increasing n_time_steps: {10, 20, 40, 80, 160}
//   - Fixed time interval [t_start, t_end] = [0.1, 0.2]
//   - dt = (t_end - t_start) / n_time_steps
//   - Measure error at final time T = 0.2
//   - Rate = log(e₁/e₂) / log(τ₁/τ₂), expect ≈ 1.0
//
// Tests:
//   CH_TEMPORAL    - CH standalone temporal convergence
//   NS_TEMPORAL    - NS standalone (Phase D: unsteady NS) temporal convergence
//   MAG_TEMPORAL   - Magnetization standalone temporal convergence
//   FULL_TEMPORAL  - Full system (all four subsystems) temporal convergence
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef TEMPORAL_CONVERGENCE_H
#define TEMPORAL_CONVERGENCE_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Temporal test level
// ============================================================================
enum class TemporalTestLevel
{
    CH_TEMPORAL,
    NS_TEMPORAL,
    MAG_TEMPORAL,
    FULL_TEMPORAL
};

std::string to_string(TemporalTestLevel level);

// ============================================================================
// Temporal convergence result
// ============================================================================
struct TemporalConvergenceResult
{
    TemporalTestLevel level;
    unsigned int refinement = 0;       // Fixed spatial mesh
    double h = 0.0;                    // Mesh size
    unsigned int n_dofs = 0;           // Total DOFs
    unsigned int n_mpi_ranks = 1;
    double total_wall_time = 0.0;

    // Per-run data
    std::vector<unsigned int> time_step_counts;
    std::vector<double> dt_values;
    std::vector<double> wall_times;

    // CH errors (populated for CH_TEMPORAL, FULL_TEMPORAL)
    std::vector<double> theta_L2;
    std::vector<double> theta_L2_rate;
    std::vector<double> theta_H1;
    std::vector<double> theta_H1_rate;

    // NS errors (populated for NS_TEMPORAL, FULL_TEMPORAL)
    std::vector<double> ux_L2;
    std::vector<double> ux_L2_rate;
    std::vector<double> p_L2;
    std::vector<double> p_L2_rate;

    // Magnetization errors (populated for MAG_TEMPORAL, FULL_TEMPORAL)
    std::vector<double> M_L2;
    std::vector<double> M_L2_rate;

    // Poisson errors (populated for FULL_TEMPORAL)
    std::vector<double> phi_L2;
    std::vector<double> phi_L2_rate;

    // Expected temporal rate (1.0 for backward Euler)
    double expected_rate = 1.0;

    void compute_rates();
    void print() const;
    void write_csv(const std::string& filename) const;
    bool passes(double tolerance = 0.3) const;
};

// ============================================================================
// Temporal test runners
// ============================================================================

/**
 * @brief Run CH temporal convergence test
 *
 * Fix spatial mesh at 'refinement', run with each n_time_steps.
 * Expect O(τ) = 1.0 for backward Euler.
 *
 * @param refinement       Spatial mesh refinement level (e.g., 5)
 * @param params           Physical/numerical parameters
 * @param time_step_counts Vector of n_time_steps to test (e.g., {10, 20, 40, 80, 160})
 * @param mpi_communicator MPI communicator
 * @return Temporal convergence results
 */
TemporalConvergenceResult run_ch_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator);

/**
 * @brief Run NS temporal convergence test (Phase D: unsteady NS)
 *
 * Fix spatial mesh at 'refinement', run with each n_time_steps.
 * Expect O(τ) = 1.0 for backward Euler.
 */
TemporalConvergenceResult run_ns_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator);

/**
 * @brief Run Magnetization temporal convergence test
 *
 * Fix spatial mesh at 'refinement', run with each n_time_steps.
 * Expect O(τ) = 1.0 for backward Euler.
 */
TemporalConvergenceResult run_magnetization_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator);

/**
 * @brief Run full system temporal convergence test
 *
 * Fix spatial mesh at 'refinement', run all four subsystems coupled
 * with each n_time_steps. Validates that block-Gauss-Seidel splitting
 * preserves O(τ) for backward Euler.
 */
TemporalConvergenceResult run_full_system_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator);

/**
 * @brief Main dispatcher for temporal tests
 */
TemporalConvergenceResult run_temporal_mms_test(
    TemporalTestLevel level,
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator);

#endif // TEMPORAL_CONVERGENCE_H
