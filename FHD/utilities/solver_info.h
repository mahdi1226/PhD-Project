// ============================================================================
// utilities/solver_info.h - Solver Parameters and Statistics
//
// LinearSolverParams: Configuration for iterative/direct linear solvers
// SolverInfo: Statistics returned after each linear solve
//
// Used by all subsystems (Poisson, Magnetization, NS, Angular Momentum).
// No dependencies on other project files.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_SOLVER_INFO_H
#define FHD_SOLVER_INFO_H

#include <string>

// ============================================================================
// Linear solver configuration
// ============================================================================
struct LinearSolverParams
{
    enum class Type { CG, GMRES, FGMRES, Direct };
    Type type = Type::Direct;

    enum class Preconditioner { None, Jacobi, SSOR, ILU, AMG, BlockSchur };
    Preconditioner preconditioner = Preconditioner::AMG;

    double rel_tolerance = 1e-8;
    double abs_tolerance = 1e-12;
    unsigned int max_iterations = 1000;
    unsigned int gmres_restart = 100;

    // Preconditioner-specific
    double ssor_omega = 1.2;
    double ilu_strengthen = 1.0;

    // Solver selection
    bool use_iterative = false;
    bool fallback_to_direct = true;
    bool verbose = false;

    // Block Schur settings (NS only)
    double schur_inner_tolerance = 1e-2;
    unsigned int schur_max_inner_iters = 50;
    unsigned int schur_gmres_restart = 100;

    // Auto-selection threshold
    unsigned int direct_dof_threshold = 2000000;
};

// ============================================================================
// Solver statistics returned from each solve
// ============================================================================
struct SolverInfo
{
    unsigned int iterations = 0;
    double residual = 0.0;
    double solve_time = 0.0;
    bool converged = true;
    bool used_direct = false;
    std::string solver_name;
    unsigned int matrix_size = 0;
    unsigned int nnz = 0;

    bool success() const { return converged; }

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

#endif // FHD_SOLVER_INFO_H
