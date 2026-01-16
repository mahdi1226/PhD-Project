// ============================================================================
// mms/coupled/coupled_mms_test.h - Coupled MMS Tests (PARALLEL)
//
// Implements systematic coupling verification:
//   Level 2a: CH + NS (phase advection by velocity)
//   Level 2b: Poisson + Magnetization (h = -∇φ drives M equation, Picard iteration)
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
    CH_NS,                   // θ advected by U (cluster A)
    POISSON_MAGNETIZATION,   // φ ↔ M Picard iteration (cluster B)
    FULL_SYSTEM              // All four subsystems (bridges clusters A and B)
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
    double expected_DG_rate = 2.0;  // DG-Q1 elements

    void compute_rates();
    bool passes(double tol = 0.3) const;
    void print() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test runners
// ============================================================================

/**
 * @brief Run CH + NS coupled MMS test
 *
 * Tests: θ advected by U (U·∇θ term in CH equation)
 *
 * Production code paths:
 *   - assemble_ns_system_parallel() with enable_mms=true
 *   - solve_ns_system_schur_parallel() (Block Schur preconditioner)
 *   - assemble_ch_system()
 *   - solve_ch_system()
 *
 * Expected: Both CH and NS should maintain optimal rates
 *   - θ: L2 = 3, H1 = 2 (Q2)
 *   - U: L2 = 3, H1 = 2 (Q2)
 *   - p: L2 = 2 (Q1)
 */
CoupledMMSConvergenceResult run_ch_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run Poisson + Magnetization coupled MMS test
 *
 * Tests: φ equation with ∇·M source, M equation with h = -∇φ
 *        Uses PICARD ITERATION (matches production)
 *
 * Production code paths:
 *   - setup_poisson_constraints_and_sparsity()
 *   - assemble_poisson_matrix() ONCE
 *   - assemble_poisson_rhs() each Picard iteration
 *   - PoissonSolver with cached AMG
 *   - setup_magnetization_sparsity() (DG flux pattern)
 *   - MagnetizationAssembler (cached)
 *   - MagnetizationSolver (MUMPS direct)
 *   - Picard loop with under-relaxation ω=0.35
 *
 * Expected: Both subsystems should maintain optimal convergence rates
 *   - φ: L2 = 3, H1 = 2 (Q2)
 *   - M: L2 = 2 (DG-Q1)
 */
CoupledMMSConvergenceResult run_poisson_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run full system coupled MMS test
 *
 * Tests: All couplings active
 *   - CH advected by NS (U·∇θ)
 *   - Poisson with ∇·M source
 *   - Magnetization relaxation to χH with advection U·∇M
 *   - NS with Kelvin force μ₀(M·∇)H + capillary + variable viscosity
 *
 * Production code paths:
 *   - Instantiates PhaseFieldProblem directly
 *   - Uses PhaseFieldProblem::time_step() with all couplings
 *   - assemble_ns_system_ferrofluid_parallel() (ALL forces)
 *   - solve_poisson_magnetization_picard() (Picard iteration)
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