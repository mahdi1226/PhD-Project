// ============================================================================
// mms/coupled/coupled_mms_test.h - Coupled MMS Tests (PARALLEL)
//
// Systematic coupling verification following paper's algorithm order:
//   CH → Magnetic → NS
//
// Two-way tests (paper's coupling direction):
//   CH_MAGNETIC:   CH provides θ → χ(θ) drives Magnetic (M+φ)
//   MAGNETIC_NS:   Magnetic provides M,H → Kelvin force drives NS
//   NS_CH:         NS provides U → advection drives CH
//
// Full system:
//   FULL_SYSTEM:   All subsystems coupled (CH + Magnetic + NS)
//
// All tests use PRODUCTION code paths (monolithic M+φ block system).
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
// Coupled test levels (paper's algorithm order)
// ============================================================================
enum class CoupledMMSLevel
{
    CH_MAGNETIC,    // CH → Magnetic: χ(θ) coupling
    MAGNETIC_NS,    // Magnetic → NS: Kelvin force coupling
    NS_CH,          // NS → CH: advection coupling
    FULL_SYSTEM     // All subsystems coupled
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

    // Magnetic errors
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double M_H1 = 0.0;

    // L-infinity errors
    double theta_Linf = 0.0;
    double psi_Linf = 0.0;
    double ux_Linf = 0.0;
    double uy_Linf = 0.0;
    double p_Linf = 0.0;
    double phi_Linf = 0.0;
    double M_Linf = 0.0;

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
    std::vector<double> M_H1_rate;

    // L-infinity convergence rates (reported but NOT in pass/fail)
    std::vector<double> theta_Linf_rate;
    std::vector<double> ux_Linf_rate;
    std::vector<double> p_Linf_rate;
    std::vector<double> phi_Linf_rate;
    std::vector<double> M_Linf_rate;

    // Expected rates
    double expected_L2_rate = 3.0;  // Q2 elements
    double expected_H1_rate = 2.0;
    double expected_DG_rate = 2.0;  // DG-Q1 elements

    void compute_rates();
    bool passes(double tol = 0.3) const;
    void print() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test runners (paper's algorithm order)
// ============================================================================

/**
 * @brief CH → Magnetic coupled MMS test
 *
 * Tests: χ(θ) coupling from CH to monolithic Magnetic (M+φ)
 *   - CH solved standalone (with MMS source, U=0)
 *   - Magnetic uses θ from CH solve for χ(θ)
 *   - Validates θ → χ(θ) → M+φ coupling path
 *
 * Production code paths:
 *   - assemble_ch_system(), solve_ch_system()
 *   - setup_magnetic_system(), MagneticAssembler, MagneticSolver
 *
 * Expected: M L2 = 2 (DG-Q1), φ L2 = 3 (Q2)
 */
CoupledMMSConvergenceResult run_ch_magnetic_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Magnetic → NS coupled MMS test (Kelvin force)
 *
 * Tests: Monolithic Magnetic solved, then M,H → Kelvin force → NS
 *   1. Solve monolithic M+φ (θ=1, U=0)
 *   2. Extract M and φ from monolithic solution
 *   3. Solve NS with Kelvin force μ₀[(M·∇)H + ½(∇·M)H]
 *
 * This is the CRITICAL coupling for Rosensweig instability!
 *
 * Production code paths:
 *   - setup_magnetic_system(), MagneticAssembler, MagneticSolver
 *   - setup_ns_*_parallel(), assemble_ns_system_with_kelvin_force_parallel()
 *   - solve_ns_system_direct_parallel(), extract_ns_solutions_parallel()
 *
 * Expected: U L2 = 3, U H1 = 2 (Q2), p L2 = 2 (Q1), M L2 = 2, φ L2 = 3
 */
CoupledMMSConvergenceResult run_magnetic_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief NS → CH coupled MMS test
 *
 * Tests: θ advected by U (U·∇θ term in CH equation)
 *
 * Production code paths:
 *   - assemble_ns_system_parallel(), solve_ns_system_schur_parallel()
 *   - assemble_ch_system(), solve_ch_system()
 *
 * Expected: θ L2 = 3, θ H1 = 2, U L2 = 3, U H1 = 2, p L2 = 2
 */
CoupledMMSConvergenceResult run_ns_ch_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Full system coupled MMS test
 *
 * Tests: All couplings active with monolithic Magnetic
 *   - NS → CH (advection U·∇θ)
 *   - CH → Magnetic (χ(θ))
 *   - Magnetic → NS (Kelvin force)
 *
 * Production code paths:
 *   - All subsystem assemblers and solvers
 *   - Monolithic MagneticAssembler + MagneticSolver
 *
 * Expected: All subsystems maintain optimal rates
 *   - θ: L2 = 3, H1 = 2
 *   - U: L2 = 3, H1 = 2
 *   - φ: L2 = 3, H1 = 2
 *   - M: L2 = 2
 */
CoupledMMSConvergenceResult run_full_system_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // COUPLED_MMS_TEST_H
