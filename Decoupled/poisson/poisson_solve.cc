// ============================================================================
// poisson/poisson_solve.cc - CG Solver with Cached AMG Preconditioner
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   Solve: A φ = b
//   where A = (∇φ, ∇X) is the constant Laplacian
//         b = (h_a − M, ∇X) changes per Picard iteration
//
// AMG preconditioner is built ONCE in setup() and reused for all solves.
// Falls back to Jacobi if AMG initialization failed.
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/utilities.h>

#include <chrono>

// ============================================================================
// solve — CG + cached AMG
//
// Returns SolverInfo with iteration count, residual, timing.
// ============================================================================
template <int dim>
SolverInfo PoissonSubsystem<dim>::solve()
{
    SolverInfo info;
    info.solver_name = "Poisson";
    info.matrix_size = system_matrix_.m();
    info.nnz = system_matrix_.n_nonzero_elements();

    auto start = std::chrono::high_resolution_clock::now();

    // Zero RHS shortcut
    const double rhs_norm = system_rhs_.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution_ = 0;
        constraints_.distribute(solution_);
        info.iterations = 0;
        info.residual = 0.0;
        info.converged = true;

        auto end = std::chrono::high_resolution_clock::now();
        info.solve_time = std::chrono::duration<double>(end - start).count();
        last_solve_info_ = info;
        ghosts_valid_ = false;
        return info;
    }

    // Tolerance: max(abs_tol, rel_tol × ‖rhs‖)
    const auto& sp = params_.solvers.poisson;
    const double tol = std::max(sp.abs_tolerance, sp.rel_tolerance * rhs_norm);
    dealii::SolverControl solver_control(sp.max_iterations, tol);

    dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

    try
    {
        if (amg_initialized_)
        {
            cg.solve(system_matrix_, solution_, system_rhs_,
                     amg_preconditioner_);
        }
        else
        {
            // Fallback: Jacobi (slower but always works)
            dealii::TrilinosWrappers::PreconditionJacobi jacobi;
            jacobi.initialize(system_matrix_);
            cg.solve(system_matrix_, solution_, system_rhs_, jacobi);
        }

        info.iterations = solver_control.last_step();
        info.residual = solver_control.last_value();
        info.converged = true;
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        info.iterations = solver_control.last_step();
        info.residual = solver_control.last_value();
        info.converged = false;

        pcout_ << "[Poisson] WARNING: CG did not converge after "
               << info.iterations << " iterations, residual = "
               << info.residual << "\n";
    }

    // Distribute constraints (enforces pinned DoF 0 = 0, hanging nodes)
    constraints_.distribute(solution_);

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    if (sp.verbose)
    {
        pcout_ << "[Poisson] CG: " << info.iterations << " its, "
               << std::scientific << info.residual << " res, "
               << std::fixed << info.solve_time << "s\n";
    }

    last_solve_info_ = info;
    ghosts_valid_ = false;
    return info;
}

// ============================================================================
// Explicit instantiations (methods defined in THIS file)
// ============================================================================
template SolverInfo PoissonSubsystem<2>::solve();
template SolverInfo PoissonSubsystem<3>::solve();
