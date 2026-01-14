// ============================================================================
// solvers/solver_info.h - Solver Parameters and Statistics
//
// LinearSolverParams: Configuration for linear solvers
// SolverInfo: Statistics returned from linear solvers
// Used by CH, Poisson, NS, and Magnetization solvers.
//
// UPDATE (2026-01-13):
//   - Changed default to direct solver (MUMPS) - 10-50x faster for ref 3-5
//   - Improved iterative settings for when direct is not available
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
    Type type = Type::Direct;  // Changed: Direct is default (MUMPS)

    enum class Preconditioner { None, Jacobi, SSOR, ILU, AMG, BlockSchur };
    Preconditioner preconditioner = Preconditioner::AMG;  // Changed: AMG better than ILU

    double rel_tolerance = 1e-8;
    double abs_tolerance = 1e-12;
    unsigned int max_iterations = 1000;  // Reduced from 2000
    unsigned int gmres_restart = 100;    // Increased from 50 for better convergence

    // Preconditioner-specific parameters
    double ssor_omega = 1.2;        // SSOR relaxation parameter
    double ilu_strengthen = 1.0;    // ILU diagonal strengthening (1.0 = none, >1 = strengthen)

    // =========================================================================
    // Solver selection
    // =========================================================================
    // use_iterative = false (default): Use direct solver (MUMPS/UMFPACK)
    //   - Much faster for problems < 2M DoFs
    //   - Machine precision accuracy (residual ~ 1e-14)
    //   - Higher memory usage
    //
    // use_iterative = true: Use iterative solver (GMRES/CG + preconditioner)
    //   - Better for large problems (> 2M DoFs) where direct runs out of memory
    //   - Scales better with MPI ranks
    //   - May need tuning for convergence
    // =========================================================================
    bool use_iterative = false;     // Changed: Direct (MUMPS) is default
    bool fallback_to_direct = true; // Fall back to direct if iterative fails
    bool verbose = false;

    // =========================================================================
    // Block Schur preconditioner settings (for NS iterative solver)
    // =========================================================================
    double schur_inner_tolerance = 1e-3;      // Loose - we're preconditioning, not solving
    unsigned int schur_max_inner_iters = 20;  // Few iterations sufficient
    unsigned int schur_gmres_restart = 30;    // Small Krylov space for inner solve

    // =========================================================================
    // Auto-selection threshold
    // =========================================================================
    // If n_dofs < direct_dof_threshold, use direct solver automatically
    // Set to 0 to disable auto-selection
    unsigned int direct_dof_threshold = 2000000;  // 2M DoFs
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