// ============================================================================
// solvers/solver_info.h - Solver Statistics
//
// Struct to capture iterations, residual, and timing from linear solvers.
// Used by CH, Poisson, and NS solvers to return diagnostic info.
// ============================================================================
#ifndef SOLVER_INFO_H
#define SOLVER_INFO_H

#include <string>

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