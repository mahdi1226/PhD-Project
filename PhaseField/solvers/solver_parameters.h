// ============================================================================
// solvers/solver_parameters.h - Solver Configuration Parameters
//
// Centralized configuration for iterative solvers.
// Add to your Parameters struct or use standalone.
//
// Reference: Phase 1 + 1.5 of performance improvement plan
// ============================================================================
#ifndef SOLVER_PARAMETERS_H
#define SOLVER_PARAMETERS_H

#include <string>

/**
 * @brief Configuration for a single linear solver
 */
struct LinearSolverParams
{
    // Solver type
    enum class Type { CG, GMRES, Direct };
    Type solver_type = Type::GMRES;

    // Preconditioner type
    enum class Preconditioner { None, Jacobi, SSOR, ILU };
    Preconditioner preconditioner = Preconditioner::ILU;

    // Tolerances
    double rel_tolerance = 1e-8;   // Relative to ||b||
    double abs_tolerance = 1e-12;  // Absolute floor

    // Iteration limits
    unsigned int max_iterations = 2000;
    unsigned int gmres_restart = 50;  // GMRES restart parameter

    // SSOR relaxation (if using SSOR preconditioner)
    double ssor_omega = 1.2;

    // Logging
    bool verbose = false;
    bool log_convergence = true;  // Log iteration count per solve

    // Fallback to direct solver on failure
    bool fallback_to_direct = true;
};

/**
 * @brief All solver configurations for the ferrofluid problem
 */
struct SolverConfiguration
{
    // Cahn-Hilliard (θ, ψ) - nonsymmetric coupled system
    LinearSolverParams ch = {
        LinearSolverParams::Type::GMRES,
        LinearSolverParams::Preconditioner::ILU,
        1e-8,   // rel_tol
        1e-12,  // abs_tol
        2000,   // max_iter
        50,     // gmres_restart
        1.2,    // ssor_omega (unused for ILU)
        false,  // verbose
        true,   // log_convergence
        true    // fallback
    };

    // Poisson (φ) - SPD system
    LinearSolverParams poisson = {
        LinearSolverParams::Type::CG,
        LinearSolverParams::Preconditioner::SSOR,
        1e-8,
        1e-12,
        2000,
        50,     // unused for CG
        1.2,    // ssor_omega
        false,
        true,
        true
    };

    // Navier-Stokes (u, p) - saddle-point system
    // Note: Using looser tolerance since it's inside a time-stepping scheme
    LinearSolverParams ns = {
        LinearSolverParams::Type::GMRES,
        LinearSolverParams::Preconditioner::ILU,
        1e-6,   // Looser tolerance for NS (avoid over-solving!)
        1e-10,
        3000,   // More iterations for saddle-point
        100,    // Larger restart for saddle-point
        1.2,
        false,
        true,
        true
    };

    // Magnetization (Mx, My) - DG system
    LinearSolverParams magnetization = {
        LinearSolverParams::Type::GMRES,
        LinearSolverParams::Preconditioner::Jacobi,
        1e-8,
        1e-12,
        1000,
        30,
        1.2,
        false,
        true,
        true
    };
};

#endif // SOLVER_PARAMETERS_H