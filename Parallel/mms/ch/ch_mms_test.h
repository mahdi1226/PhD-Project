// ============================================================================
// mms/ch/ch_mms_test.h - CH MMS Test Interface (Parallel Version)
//
// Uses PRODUCTION code directly (no MMSContext):
//   - setup_ch_coupled_system() from setup/ch_setup.h
//   - assemble_ch_system() from assembly/ch_assembler.h
//   - solve_ch_system() from solvers/ch_solver.h
//
// NO PARAMETER OVERRIDES - uses Parameters defaults from parameters.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_MMS_TEST_H
#define CH_MMS_TEST_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Solver type selection
// ============================================================================
enum class CHSolverType
{
    Direct,     // Trilinos direct solver (Amesos)
    GMRES_AMG   // GMRES with AMG preconditioner (production default)
};

// ============================================================================
// Single refinement result
// ============================================================================
struct CHMMSResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;

    // Errors
    double theta_L2_error = 0.0;  // Alias for mms_verification.cc compatibility
    double theta_H1_error = 0.0;
    double psi_L2_error = 0.0;

    // Direct access names
    double theta_L2 = 0.0;
    double theta_H1 = 0.0;
    double psi_L2 = 0.0;

    // Timing breakdown
    double setup_time = 0.0;
    double assembly_time = 0.0;
    double solve_time = 0.0;
    double total_time = 0.0;

    // Solver info
    unsigned int solver_iterations = 0;
    double solver_residual = 0.0;
};

// ============================================================================
// Convergence study result
// ============================================================================
struct CHMMSConvergenceResult
{
    std::vector<CHMMSResult> results;
    unsigned int fe_degree = 2;
    unsigned int n_time_steps = 10;
    double dt = 0.0;
    double expected_L2_rate = 3.0;  // p+1 for Q2
    double expected_H1_rate = 2.0;  // p for Q2

    /// Compute convergence rates from stored results
    void compute_rates();

    /// Print formatted table
    void print() const;

    /// Write to CSV file
    void write_csv(const std::string& filename) const;

    /// Check if rates meet expectations (within tolerance)
    bool passes(double tol = 0.3) const;

    // Computed rates (filled by compute_rates())
    std::vector<double> theta_L2_rates;
    std::vector<double> theta_H1_rates;
    std::vector<double> psi_L2_rates;
};

// ============================================================================
// Main test functions
// ============================================================================

/**
 * @brief Run CH MMS convergence study using PRODUCTION code
 *
 * PARALLEL VERSION: Takes MPI_Comm, uses distributed triangulation
 * Uses Parameters defaults - NO OVERRIDES
 *
 * @param refinements Vector of refinement levels to test
 * @param params Parameters from parameters.h (production defaults)
 * @param solver_type Which solver to use
 * @param n_time_steps Number of time steps
 * @param mpi_communicator MPI communicator
 * @return Convergence results with errors and rates
 */
CHMMSConvergenceResult run_ch_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    CHSolverType solver_type = CHSolverType::GMRES_AMG,
    unsigned int n_time_steps = 10,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/**
 * @brief Run CH MMS (wrapper for mms_verification.cc compatibility)
 *
 * Same as run_ch_mms_standalone but with simplified interface
 */
CHMMSConvergenceResult run_ch_mms_parallel(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run single refinement level
 */
CHMMSResult run_ch_mms_single(
    unsigned int refinement,
    const Parameters& params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/**
 * @brief Compare solver performance (direct vs iterative)
 */
void compare_ch_solvers(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps = 10,
    MPI_Comm mpi_communicator = MPI_COMM_WORLD);

#endif // CH_MMS_TEST_H