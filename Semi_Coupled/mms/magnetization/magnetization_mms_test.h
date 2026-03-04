// ============================================================================
// mms/magnetization/magnetization_mms_test.h - Magnetization MMS Test (PARALLEL)
//
// PARALLEL VERSION - Self-contained test that:
//   - Creates its own distributed mesh
//   - Uses PRODUCTION setup/assembler/solver
//   - Computes errors with MPI reductions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_MMS_TEST_H
#define MAGNETIZATION_MMS_TEST_H

#include "utilities/parameters.h"

#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Solver type for testing
// ============================================================================
enum class MagSolverType
{
    Direct,     // Amesos direct solver
    GMRES       // GMRES + Jacobi preconditioner
};

std::string to_string(MagSolverType type);

// ============================================================================
// Single refinement result
// ============================================================================
struct MagMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // Errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double Mx_Linf = 0.0;
    double My_Linf = 0.0;
    double M_Linf = 0.0;

    // Timing
    double total_time = 0.0;

    MagSolverType solver_type = MagSolverType::GMRES;
};

// ============================================================================
// Convergence study result
// ============================================================================
struct MagMMSConvergenceResult
{
    std::vector<MagMMSResult> results;
    std::vector<double> M_L2_rates;
    std::vector<double> M_Linf_rates;

    double expected_L2_rate = 0.0;
    unsigned int fe_degree = 1;
    MagSolverType solver_type = MagSolverType::GMRES;

    void compute_rates();
    bool passes(double tolerance = 0.3) const;
    void print() const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Test Runners
// ============================================================================

/**
 * @brief Run Magnetization MMS convergence study (PARALLEL, STANDALONE U=0)
 *
 * @param refinements     List of refinement levels
 * @param params          Physical and numerical parameters
 * @param solver_type     Solver to use
 * @param mpi_communicator MPI communicator
 * @return Convergence results
 */
MagMMSConvergenceResult run_magnetization_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    MagSolverType solver_type,
    MPI_Comm mpi_communicator);

/**
 * @brief Run Magnetization MMS convergence study with variable time steps
 *
 * Overload for temporal convergence testing.
 */
MagMMSConvergenceResult run_magnetization_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    MagSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

/**
 * @brief Run single Magnetization MMS test (default 100 time steps)
 */
MagMMSResult run_magnetization_mms_single(
    unsigned int refinement,
    const Parameters& params,
    MagSolverType solver_type,
    MPI_Comm mpi_communicator);

/**
 * @brief Run single Magnetization MMS test with variable time steps
 *
 * Overload for temporal convergence testing.
 */
MagMMSResult run_magnetization_mms_single(
    unsigned int refinement,
    const Parameters& params,
    MagSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

// ============================================================================
// Transport Test: Magnetization with non-zero velocity (NS MMS)
// ============================================================================

/**
 * @brief Run Magnetization MMS with NS velocity (DG transport test)
 *
 * Isolates the DG transport assembly from all coupling:
 *   - Uses interpolated NS MMS velocity (solenoidal, zero on boundaries)
 *   - Uses chi=0 (no magnetic coupling)
 *   - Single time step (spatial convergence only)
 */
MagMMSConvergenceResult run_magnetization_transport_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // MAGNETIZATION_MMS_TEST_H