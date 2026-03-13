// ============================================================================
// mms/magnetic/magnetic_mms_test.h - Monolithic Magnetics MMS Test (PARALLEL)
//
// PARALLEL VERSION - Self-contained test that:
//   - Creates its own distributed mesh
//   - Uses PRODUCTION setup/assembler/solver for combined M+phi system
//   - Computes errors with MPI reductions
//
// Expected convergence rates:
//   M  (DG Q1): L2 rate = 2
//   phi (CG Q2): L2 rate = 3, H1 rate = 2
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_MMS_TEST_H
#define MAGNETIC_MMS_TEST_H

#include "utilities/parameters.h"

#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Single refinement result
// ============================================================================
struct MagneticMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // Magnetization errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double M_H1 = 0.0;
    double M_Linf = 0.0;

    // Poisson errors (mean-shift corrected)
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;
    double phi_Linf = 0.0;

    // Timing
    double total_time = 0.0;
};

// ============================================================================
// Convergence study result
// ============================================================================
struct MagneticMMSConvergenceResult
{
    std::vector<MagneticMMSResult> results;
    std::vector<double> M_L2_rates;
    std::vector<double> M_Linf_rates;
    std::vector<double> M_H1_rates;
    std::vector<double> phi_L2_rates;
    std::vector<double> phi_Linf_rates;
    std::vector<double> phi_H1_rates;

    unsigned int degree_M = 1;
    unsigned int degree_phi = 2;
    double expected_M_L2_rate = 2.0;     // DG Q1: degree+1
    double expected_phi_L2_rate = 3.0;   // CG Q2: degree+1
    double expected_phi_H1_rate = 2.0;   // CG Q2: degree

    void compute_rates();
    bool passes(double tolerance = 0.3) const;
    void print() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test Runners
// ============================================================================

/**
 * @brief Run monolithic magnetics MMS convergence study (PARALLEL, U=0)
 *
 * Tests the combined M+phi block system with:
 *   - Zero velocity (no transport)
 *   - Constant theta=1 (full ferrofluid, chi = chi_0)
 *   - MUMPS direct solver
 *
 * @param refinements     List of refinement levels
 * @param params          Physical and numerical parameters
 * @param n_time_steps    Number of time steps
 * @param mpi_communicator MPI communicator
 * @return Convergence results
 */
MagneticMMSConvergenceResult run_magnetic_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run single magnetics MMS test at one refinement level
 */
MagneticMMSResult run_magnetic_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // MAGNETIC_MMS_TEST_H
