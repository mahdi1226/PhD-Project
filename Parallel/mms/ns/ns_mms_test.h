// ============================================================================
// mms/ns/ns_mms_test.h - NS MMS Test Interface (PARALLEL VERSION)
//
// Provides parallel NS MMS verification integrated with mms_verification.cc
//
// Uses PRODUCTION code:
//   - setup_ns_coupled_system() from setup/ns_setup.h
//   - BlockSchurPreconditionerParallel from solvers/ns_block_preconditioner.h
//   - solve_ns_system_schur_parallel() from solvers/ns_solver.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_MMS_TEST_H
#define NS_MMS_TEST_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>

// ============================================================================
// Solver type selection
// ============================================================================
enum class NSSolverType
{
    Direct,         // Direct solver (serial only)
    GMRES_ILU,      // GMRES with ILU (serial only)
    Schur           // Block Schur preconditioner (parallel)
};

inline std::string to_string(NSSolverType type)
{
    switch (type)
    {
        case NSSolverType::Direct:    return "Direct";
        case NSSolverType::GMRES_ILU: return "GMRES+ILU";
        case NSSolverType::Schur:     return "Block Schur";
        default:                      return "Unknown";
    }
}

// ============================================================================
// NS Phase for testing
// ============================================================================
enum class NSPhase
{
    A,  // Steady Stokes
    B,  // Unsteady Stokes
    C,  // Steady NS
    D   // Unsteady NS (full equation)
};

inline std::string to_string(NSPhase phase)
{
    switch (phase)
    {
        case NSPhase::A: return "A: Steady Stokes";
        case NSPhase::B: return "B: Unsteady Stokes";
        case NSPhase::C: return "C: Steady NS";
        case NSPhase::D: return "D: Unsteady NS";
        default:         return "Unknown";
    }
}

// ============================================================================
// Single refinement result
// ============================================================================
struct NSMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // Errors
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;

    // Timing breakdown
    double setup_time = 0.0;
    double assembly_time = 0.0;
    double solve_time = 0.0;
    double total_time = 0.0;

    // Solver statistics
    unsigned int solver_iterations = 0;
    double solver_residual = 0.0;
};

// ============================================================================
// Convergence study result
// ============================================================================
struct NSMMSConvergenceResult
{
    std::vector<NSMMSResult> results;

    // Metadata
    NSPhase phase = NSPhase::D;
    unsigned int fe_degree_velocity = 2;   // Q2 for velocity
    unsigned int fe_degree_pressure = 1;   // Q1 for pressure
    unsigned int n_time_steps = 0;
    double dt = 0.0;
    double nu = 1.0;                       // Viscosity
    double L_y = 1.0;                      // Domain height

    // Expected rates for Q2-Q1 Taylor-Hood
    double expected_vel_L2_rate = 3.0;     // p+1 for Q2
    double expected_vel_H1_rate = 2.0;     // p for Q2
    double expected_p_L2_rate = 2.0;       // p+1 for Q1

    // Computed rates (filled by compute_rates())
    std::vector<double> ux_L2_rate;
    std::vector<double> ux_H1_rate;
    std::vector<double> uy_L2_rate;
    std::vector<double> uy_H1_rate;
    std::vector<double> p_L2_rate;

    /// Compute convergence rates from error data
    void compute_rates();

    /// Print formatted table to console
    void print() const;

    /// Write results to CSV file
    void write_csv(const std::string& filename) const;

    /// Check if rates match expected values (within tolerance)
    bool passes(double tol = 0.3) const;
};

// ============================================================================
// Main test function - PARALLEL version with Block Schur preconditioner
//
// Tests the full NS production path with parallel infrastructure:
//   - Distributed triangulation
//   - Trilinos matrices and vectors
//   - BlockSchurPreconditionerParallel
//
// For standalone NS test, theta/psi/phi/M are not needed (no coupling)
// ============================================================================
NSMMSConvergenceResult run_ns_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    NSSolverType solver_type,
    MPI_Comm mpi_communicator);

// Convenience overload defaulting to Block Schur solver
NSMMSConvergenceResult run_ns_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    MPI_Comm mpi_communicator);

// ============================================================================
// Phase-specific test (for detailed debugging)
// ============================================================================
NSMMSConvergenceResult run_ns_mms_phase(
    NSPhase phase,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    MPI_Comm mpi_communicator);

#endif // NS_MMS_TEST_H