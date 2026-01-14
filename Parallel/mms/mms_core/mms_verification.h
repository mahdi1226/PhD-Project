// ============================================================================
// mms/mms_core/mms_verification.h - MMS Verification Interface (PARALLEL VERSION)
//
// STANDALONE TESTS ONLY - Coupled tests are in coupled_mms_test.h
//
// PARALLEL STATUS:
//   - CH_STANDALONE: CONVERTED ✓
//   - POISSON_STANDALONE: CONVERTED ✓
//   - NS_STANDALONE: CONVERTED ✓
//   - MAGNETIZATION_STANDALONE: CONVERTED ✓
//
// For coupled tests, see mms/coupled/coupled_mms_test.h:
//   - CH_NS: Phase advection by velocity
//   - POISSON_MAGNETIZATION: φ ↔ M Picard iteration
//   - FULL_SYSTEM: All four subsystems coupled
// ============================================================================
#ifndef MMS_VERIFICATION_H
#define MMS_VERIFICATION_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// MMS Test Levels (STANDALONE ONLY)
// ============================================================================
enum class MMSLevel
{
    CH_STANDALONE,           // Cahn-Hilliard only
    POISSON_STANDALONE,      // Poisson only
    NS_STANDALONE,           // Navier-Stokes only
    MAGNETIZATION_STANDALONE // Magnetization only
};

std::string to_string(MMSLevel level);

// ============================================================================
// Convergence study result
// ============================================================================
struct MMSConvergenceResult
{
    MMSLevel level;
    unsigned int fe_degree = 2;
    unsigned int n_time_steps = 10;
    unsigned int n_mpi_ranks = 1;
    double total_wall_time = 0.0;
    double dt = 0.0;
    double expected_L2_rate = 3.0;
    double expected_H1_rate = 2.0;

    // Data for each refinement level
    std::vector<unsigned int> refinements;
    std::vector<double> h_values;
    std::vector<unsigned int> n_dofs;
    std::vector<double> wall_times;

    // CH errors
    std::vector<double> theta_L2, theta_H1, psi_L2;
    std::vector<double> theta_L2_rate, theta_H1_rate, psi_L2_rate;

    // Poisson errors
    std::vector<double> phi_L2, phi_H1;
    std::vector<double> phi_L2_rate, phi_H1_rate;

    // NS errors
    std::vector<double> ux_L2, ux_H1, uy_L2, uy_H1, p_L2, div_u_L2;
    std::vector<double> ux_L2_rate, ux_H1_rate, uy_L2_rate, uy_H1_rate, p_L2_rate, div_u_L2_rate;

    // Magnetization errors
    std::vector<double> M_L2;
    std::vector<double> M_L2_rate;

    void compute_rates();
    void print() const;
    void write_csv(const std::string& filename) const;
    bool passes(double tolerance = 0.3) const;

    // Table printers
    void print_ch_table() const;
    void print_poisson_table() const;
    void print_ns_table() const;
    void print_magnetization_table() const;
};

// ============================================================================
// Main test function (PARALLEL) - STANDALONE TESTS ONLY
// ============================================================================

/**
 * @brief Run standalone MMS convergence study
 *
 * Uses Parameters defaults from parameters.h - NO OVERRIDES
 *
 * @param level Which subsystem to test (standalone only)
 * @param refinements Vector of refinement levels
 * @param params Simulation parameters (production defaults)
 * @param n_time_steps Number of time steps
 * @param mpi_communicator MPI communicator
 * @return Convergence results
 *
 * For coupled tests, use functions from mms/coupled/coupled_mms_test.h:
 *   - run_ch_ns_mms()
 *   - run_poisson_magnetization_mms()
 *   - run_full_system_mms()
 */
MMSConvergenceResult run_mms_test(
    MMSLevel level,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // MMS_VERIFICATION_H