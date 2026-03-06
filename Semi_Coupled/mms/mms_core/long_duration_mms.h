// ============================================================================
// mms/mms_core/long_duration_mms.h - Long-Duration MMS Stability Tests
//
// PURPOSE: Track error evolution over many time steps to detect:
//   1. Linear growth (expected for 1st-order backward Euler)
//   2. Exponential growth (instability in scheme or coupling)
//   3. Bounded error (ideal)
//
// Unlike temporal_convergence.h (which varies dt to check O(tau) rate),
// these tests run MANY steps at FIXED dt and record errors at each step.
//
// Key diagnostic: plot log(error) vs time. Linear → stable. Upward curve → trouble.
//
// Test levels:
//   CH_LONG     - CH standalone (no velocity), isolates CH time integration
//   CH_NS_LONG  - CH + NS coupled, tests convection coupling stability
//   FULL_LONG   - All subsystems, tests full coupling stability
//
// Output: CSV file with per-step errors for post-processing.
//
// Usage:
//   mpirun -np 1 parallel_test_mms --level CH_LONG --temporal-ref 4 --temporal-steps 500
//   mpirun -np 1 parallel_test_mms --level CH_NS_LONG --temporal-ref 4 --temporal-steps 500
//   mpirun -np 1 parallel_test_mms --level FULL_LONG --temporal-ref 3 --temporal-steps 200
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef LONG_DURATION_MMS_H
#define LONG_DURATION_MMS_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Long-duration test levels
// ============================================================================
enum class LongDurationLevel
{
    CH_LONG,        // CH standalone
    CH_NS_LONG,     // CH + NS (convection coupling)
    FULL_LONG       // All subsystems coupled
};

std::string to_string(LongDurationLevel level);

// ============================================================================
// Per-step error snapshot
// ============================================================================
struct StepErrorSnapshot
{
    unsigned int step = 0;
    double time = 0.0;
    double wall_time = 0.0;    // Cumulative wall time

    // CH errors (always populated)
    double theta_L2 = 0.0;
    double theta_H1 = 0.0;
    double psi_L2 = 0.0;

    // NS errors (populated for CH_NS_LONG, FULL_LONG)
    double ux_L2 = 0.0;
    double p_L2 = 0.0;

    // Magnetic errors (populated for FULL_LONG)
    double phi_L2 = 0.0;
    double M_L2 = 0.0;
};

// ============================================================================
// Long-duration test result
// ============================================================================
struct LongDurationResult
{
    LongDurationLevel level;
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;
    unsigned int n_steps = 0;
    double dt = 0.0;
    double t_start = 0.0;
    double t_end = 0.0;
    unsigned int n_mpi_ranks = 1;
    double total_wall_time = 0.0;

    // Per-step error history
    std::vector<StepErrorSnapshot> snapshots;

    // Error growth analysis (computed from snapshots)
    double theta_L2_growth_rate = 0.0;    // Fit: e(t) = e0 * exp(rate * t)
    double theta_L2_final_ratio = 0.0;    // e(t_end) / e(t_start)
    bool is_exponential_growth = false;   // true if growth faster than linear

    void analyze_growth();
    void print_summary() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test runners
// ============================================================================

/**
 * @brief Run CH standalone long-duration MMS test
 *
 * Isolates CH time integration stability. No velocity coupling.
 * Runs n_time_steps at fixed dt, records theta_L2, theta_H1 at every step.
 *
 * If error grows linearly: normal (1st-order backward Euler accumulation)
 * If error grows exponentially: CH scheme itself is unstable
 *
 * @param refinement       Spatial mesh refinement level (e.g., 4)
 * @param params           Physical/numerical parameters
 * @param n_time_steps     Number of time steps to run (e.g., 500)
 * @param log_interval     Record errors every N steps (default: 1 = every step)
 * @param mpi_communicator MPI communicator
 * @return Long-duration results with per-step error history
 */
LongDurationResult run_ch_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval = 1,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/**
 * @brief Run CH + NS coupled long-duration MMS test
 *
 * Tests whether the velocity-CH coupling causes error amplification.
 * Compare with CH_LONG to isolate coupling effects.
 *
 * CH is solved with velocity from NS (convection term).
 * NS is solved with MMS forcing.
 *
 * @param refinement       Spatial mesh refinement level
 * @param params           Physical/numerical parameters
 * @param n_time_steps     Number of time steps to run
 * @param log_interval     Record errors every N steps
 * @param mpi_communicator MPI communicator
 * @return Long-duration results
 */
LongDurationResult run_ch_ns_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval = 1,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/**
 * @brief Run full system long-duration MMS test
 *
 * Tests the complete coupling chain: CH → Poisson/Mag → NS.
 * Most expensive but most complete: detects coupling-induced instabilities.
 *
 * @param refinement       Spatial mesh refinement level
 * @param params           Physical/numerical parameters
 * @param n_time_steps     Number of time steps to run
 * @param log_interval     Record errors every N steps
 * @param mpi_communicator MPI communicator
 * @return Long-duration results
 */
LongDurationResult run_full_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval = 1,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/**
 * @brief Main dispatcher for long-duration tests
 */
LongDurationResult run_long_duration_mms_test(
    LongDurationLevel level,
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval = 1,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

#endif // LONG_DURATION_MMS_H
