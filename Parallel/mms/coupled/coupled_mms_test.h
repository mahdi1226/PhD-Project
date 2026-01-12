// ============================================================================
// mms/coupled/coupled_mms_test.h - Coupled MMS Tests (PARALLEL)
//
// Implements systematic coupling verification:
//   Level 2a: Poisson + Magnetization (h = -∇φ drives M equation)
//   Level 2b: CH + NS (phase advection by velocity)
//   Level 2c: NS + Poisson + Mag (Kelvin force μ₀(m·∇)h in momentum)
//   Level 3:  Full system (all four subsystems coupled)
//
// Key insight: Each subsystem's manufactured solution becomes forcing for others.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef COUPLED_MMS_TEST_H
#define COUPLED_MMS_TEST_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Coupled test levels
// ============================================================================
enum class CoupledMMSLevel
{
    POISSON_MAGNETIZATION,   // φ → h = -∇φ → M relaxation
    CH_NS,                   // θ advected by U
    NS_POISSON_MAG,          // Kelvin force: μ₀(M·∇)H in NS
    FULL_SYSTEM              // All four subsystems
};

std::string to_string(CoupledMMSLevel level);

// ============================================================================
// Coupled convergence result
// ============================================================================
struct CoupledMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // CH errors
    double theta_L2 = 0.0;
    double theta_H1 = 0.0;
    double psi_L2 = 0.0;

    // NS errors
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;

    // Poisson errors
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;

    // Magnetization errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;

    // Timing
    double total_time = 0.0;
};

struct CoupledMMSConvergenceResult
{
    std::vector<CoupledMMSResult> results;
    CoupledMMSLevel level;

    // Computed rates
    std::vector<double> theta_L2_rate;
    std::vector<double> theta_H1_rate;
    std::vector<double> ux_L2_rate;
    std::vector<double> ux_H1_rate;
    std::vector<double> p_L2_rate;
    std::vector<double> phi_L2_rate;
    std::vector<double> phi_H1_rate;
    std::vector<double> M_L2_rate;

    // Expected rates
    double expected_L2_rate = 3.0;  // Q2 elements
    double expected_H1_rate = 2.0;

    void compute_rates();
    bool passes(double tol = 0.3) const;
    void print() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test runners
// ============================================================================

/**
 * @brief Run Poisson + Magnetization coupled MMS test
 *
 * Tests: φ equation with ∇·M source, M equation with h = -∇φ
 * Expected: Both subsystems should maintain optimal convergence rates
 */
CoupledMMSConvergenceResult run_poisson_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run CH + NS coupled MMS test
 *
 * Tests: θ advected by U (U·∇θ term in CH equation)
 * Expected: Both CH and NS should maintain optimal rates
 */
CoupledMMSConvergenceResult run_ch_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run NS + Poisson + Magnetization coupled MMS test
 *
 * Tests: Kelvin force μ₀(M·∇)H in NS momentum equation
 * Expected: All three subsystems maintain optimal rates
 */
CoupledMMSConvergenceResult run_ns_poisson_mag_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run full system coupled MMS test
 *
 * Tests: All couplings active
 * - CH advected by NS
 * - Poisson with ∇·M source
 * - Magnetization relaxation to χH
 * - NS with Kelvin force + variable viscosity
 */
CoupledMMSConvergenceResult run_full_system_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // COUPLED_MMS_TEST_H