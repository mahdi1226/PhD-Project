// ============================================================================
// solvers/solver_info.h - Solver Parameters and Statistics
//
// LinearSolverParams: Configuration for linear solvers
// SolverInfo: Statistics returned from linear solvers
// Used by CH, Poisson, NS, and Magnetization solvers.
// ============================================================================
#ifndef SOLVER_INFO_H
#define SOLVER_INFO_H

#include <string>

/**
 * @brief Linear solver configuration parameters
 */
struct LinearSolverParams
{
    enum class Type { CG, GMRES, FGMRES, Direct };
    Type type = Type::GMRES;

    enum class Preconditioner { None, Jacobi, SSOR, ILU, BlockSchur };
    Preconditioner preconditioner = Preconditioner::ILU;

    double rel_tolerance = 1e-8;
    double abs_tolerance = 1e-12;
    unsigned int max_iterations = 2000;
    unsigned int gmres_restart = 50;
    double ssor_omega = 1.2;

    bool use_iterative = true;
    bool fallback_to_direct = true;
    bool verbose = false;
};

/**
 * @brief Solver statistics returned from linear solvers
 */
struct SolverInfo
{
    unsigned int iterations = 0;
    double residual = 0.0;
    double solve_time = 0.0;        // seconds
    bool converged = true;
    bool used_direct = false;       // true if fell back to direct solver
    std::string solver_name;        // "CH", "Poisson", "NS", etc.
    unsigned int matrix_size = 0;
    unsigned int nnz = 0;           // number of nonzeros

    /// Check if solver succeeded
    bool success() const { return converged; }

    /// Reset to defaults
    void reset()
    {
        iterations = 0;
        residual = 0.0;
        solve_time = 0.0;
        converged = true;
        used_direct = false;
        matrix_size = 0;
        nnz = 0;
    }
};

#endif // SOLVER_INFO_H