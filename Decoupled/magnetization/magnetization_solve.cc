// ============================================================================
// magnetization/magnetization_solve.cc — DG Scalar Component Solver
//
// Private method:
//   solve_component()  — solve matrix * solution = rhs for one component
//
// STRATEGY:
//   1. Direct solver (MUMPS → SuperLU_DIST → KLU fallback chain)
//   2. Iterative GMRES + cached ILU (if direct fails or configured)
//
// The same matrix is shared by Mx and My. The preconditioner is
// initialized once in magnetization_assemble.cc and reused for both.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

using namespace dealii;

// ============================================================================
// solve_component() — solve one scalar DG system
//
// Returns SolverInfo with iteration count, residual, convergence flag.
// ============================================================================
template <int dim>
SolverInfo MagnetizationSubsystem<dim>::solve_component(
    TrilinosWrappers::MPI::Vector& solution,
    const TrilinosWrappers::MPI::Vector& rhs,
    const std::string& component_name)
{
    Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = component_name;

    // Zero RHS → zero solution, skip solver
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        info.iterations = 0;
        info.residual   = 0.0;
        info.converged  = true;
        timer.stop();
        info.solve_time = timer.wall_time();
        return info;
    }

    const bool use_direct = !params_.solvers.magnetization.use_iterative;
    bool converged = false;

    // ========================================================================
    // Direct solver: MUMPS → SuperLU_DIST → KLU fallback
    // ========================================================================
    if (use_direct)
    {
        const double tol = std::max(
            params_.solvers.magnetization.abs_tolerance,
            params_.solvers.magnetization.rel_tolerance * rhs_norm);
        SolverControl solver_control(1, tol);

        // Try MUMPS (parallel direct)
        try
        {
            TrilinosWrappers::SolverDirect::AdditionalData data;
            data.solver_type = "Amesos_Mumps";

            TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
            direct_solver.solve(system_matrix_, solution, rhs);

            info.iterations = 1;
            converged = true;
        }
        catch (std::exception&)
        {
            // Try SuperLU_DIST
            try
            {
                TrilinosWrappers::SolverDirect::AdditionalData data;
                data.solver_type = "Amesos_Superludist";

                TrilinosWrappers::SolverDirect direct_solver(
                    solver_control, data);
                direct_solver.solve(system_matrix_, solution, rhs);

                info.iterations = 1;
                converged = true;
            }
            catch (std::exception&)
            {
                // Fall back to KLU (default)
                try
                {
                    TrilinosWrappers::SolverDirect direct_solver(
                        solver_control);
                    direct_solver.solve(system_matrix_, solution, rhs);

                    info.iterations = 1;
                    converged = true;
                }
                catch (std::exception&)
                {
                    pcout_ << "[Magnetization] " << component_name
                           << ": all direct solvers failed, "
                           << "falling back to iterative" << std::endl;
                    converged = false;
                }
            }
        }
    }

    // ========================================================================
    // Iterative solver: GMRES + ILU (primary or fallback)
    // ========================================================================
    if (!converged)
    {
        const double tol = std::max(
            params_.solvers.magnetization.abs_tolerance,
            params_.solvers.magnetization.rel_tolerance * rhs_norm);
        SolverControl solver_control(
            params_.solvers.magnetization.max_iterations, tol);

        try
        {
            TrilinosWrappers::SolverGMRES solver(solver_control);

            if (preconditioner_initialized_)
            {
                solver.solve(system_matrix_, solution, rhs,
                             ilu_preconditioner_);
            }
            else
            {
                // Fallback: Jacobi (always available)
                TrilinosWrappers::PreconditionJacobi jacobi;
                jacobi.initialize(system_matrix_);
                solver.solve(system_matrix_, solution, rhs, jacobi);
            }

            info.iterations = solver_control.last_step();
            info.residual   = solver_control.last_value();
            converged = true;
        }
        catch (SolverControl::NoConvergence& e)
        {
            info.iterations = e.last_step;
            info.residual   = e.last_residual;

            pcout_ << "[Magnetization] " << component_name
                   << ": GMRES did not converge after "
                   << info.iterations << " iterations, "
                   << "residual = " << info.residual << std::endl;
        }
        catch (std::exception& e)
        {
            pcout_ << "[Magnetization] " << component_name
                   << ": solver failed: " << e.what() << std::endl;
        }
    }

    info.converged = converged;

    timer.stop();
    info.solve_time = timer.wall_time();

    return info;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template SolverInfo MagnetizationSubsystem<2>::solve_component(
    TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const std::string&);

template SolverInfo MagnetizationSubsystem<3>::solve_component(
    TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const std::string&);