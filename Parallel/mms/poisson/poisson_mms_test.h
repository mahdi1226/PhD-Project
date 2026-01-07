// ============================================================================
// mms/poisson/poisson_mms_test.h - Poisson MMS Test Interface (PARALLEL)
//
// Tests PRODUCTION components:
//   - setup/poisson_setup.h (constraints, sparsity)
//   - assembly/poisson_assembler.h (system assembly)
//   - solvers/poisson_solver.h (AMG solve)
//
// PARALLEL VERSION:
//   - Uses MMSContext with distributed triangulation
//   - MPI reductions for global error norms
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_MMS_TEST_H
#define POISSON_MMS_TEST_H

#include "utilities/parameters.h"
#include "solvers/solver_info.h"

#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Solver type for A/B testing
// ============================================================================
enum class PoissonSolverType
{
    AMG,      // CG + Trilinos AMG (default, scalable)
    Direct    // Amesos direct solver
};

std::string to_string(PoissonSolverType type);

// ============================================================================
// Single refinement result with timing breakdown
// ============================================================================
struct PoissonMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // Errors
    double L2_error = 0.0;
    double H1_error = 0.0;
    double Linf_error = 0.0;

    // Timing breakdown (seconds)
    double setup_time = 0.0;
    double assembly_time = 0.0;
    double solve_time = 0.0;
    double total_time = 0.0;

    // Solver info
    unsigned int solver_iterations = 0;
    double solver_residual = 0.0;
    bool used_direct_fallback = false;
    PoissonSolverType solver_type = PoissonSolverType::AMG;
};

// ============================================================================
// Convergence study result
// ============================================================================
struct PoissonMMSConvergenceResult
{
    std::vector<PoissonMMSResult> results;

    // Computed convergence rates
    std::vector<double> L2_rates;
    std::vector<double> H1_rates;

    // Expected rates (based on FE degree)
    double expected_L2_rate = 0.0;  // p+1 for Q_p elements
    double expected_H1_rate = 0.0;  // p for Q_p elements

    // Test configuration
    unsigned int fe_degree = 1;
    PoissonSolverType solver_type = PoissonSolverType::AMG;
    bool standalone = true;

    /// Compute convergence rates from errors
    void compute_rates();

    /// Check if rates meet expectations (within tolerance)
    bool passes(double tolerance = 0.3) const;

    /// Print formatted results table
    void print() const;

    /// Write results to CSV file
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test Runners (PARALLEL)
// ============================================================================

/**
 * @brief Run Poisson MMS convergence study (PARALLEL, STANDALONE M=0)
 *
 * Tests production code path with distributed mesh and Trilinos types.
 *
 * @param refinements      List of refinement levels to test
 * @param params           Physical and numerical parameters
 * @param solver_type      Which solver to use (AMG, Direct)
 * @param mpi_communicator MPI communicator
 * @return Convergence study results with timing
 */
PoissonMMSConvergenceResult run_poisson_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    PoissonSolverType solver_type,
    MPI_Comm mpi_communicator);

/**
 * @brief Run single Poisson MMS test at one refinement level (PARALLEL)
 */
PoissonMMSResult run_poisson_mms_single(
    unsigned int refinement,
    const Parameters& params,
    PoissonSolverType solver_type,
    MPI_Comm mpi_communicator);

#endif // POISSON_MMS_TEST_H