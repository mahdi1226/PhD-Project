// ============================================================================
// poisson/poisson_solve.cc - Linear Solve and Diagnostics
//
// Solves the constant-coefficient Laplacian with AMG-preconditioned CG.
// Matrix and AMG are built ONCE in setup(); only RHS changes per iteration.
//
// Solver strategy:
//   1. CG + AMG (default, iterative)
//   2. Fallback to direct solver (UMFPACK via Trilinos) if CG fails
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42d
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>

template <int dim>
SolverInfo PoissonSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = "Poisson-CG-AMG";
    info.matrix_size = dof_handler_.n_dofs();

    const auto& solver_params = params_.solvers.poisson;

    if (solver_params.use_iterative && amg_initialized_)
    {
        dealii::SolverControl solver_control(
            solver_params.max_iterations,
            solver_params.abs_tolerance,
            /*log_history=*/false,
            /*log_result=*/false);

        dealii::TrilinosWrappers::SolverCG solver(solver_control);

        try
        {
            solver.solve(system_matrix_, solution_,
                         system_rhs_, amg_preconditioner_);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            pcout_ << "  Poisson CG failed after "
                   << e.last_step << " iterations (residual="
                   << e.last_residual << ")\n";

            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  Falling back to direct solver...\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(
                    direct_control);
                direct_solver.solve(system_matrix_, solution_, system_rhs_);

                info.iterations = 1;
                info.residual = 0.0;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "Poisson-Direct-Fallback";
            }
            else
            {
                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = false;
            }
        }
        catch (const std::exception& e)
        {
            // Catches Trilinos AztecOO errors (e.g., "loss of precision")
            // which are thrown as dealii::ExcMessage, not NoConvergence
            pcout_ << "  Poisson CG failed: " << e.what() << "\n";

            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  Falling back to direct solver...\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(
                    direct_control);
                direct_solver.solve(system_matrix_, solution_, system_rhs_);

                info.iterations = 1;
                info.residual = 0.0;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "Poisson-Direct-Fallback";
            }
            else
            {
                info.converged = false;
                throw;
            }
        }
    }
    else
    {
        // Direct solver
        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
        direct_solver.solve(system_matrix_, solution_, system_rhs_);

        info.iterations = 1;
        info.residual = 0.0;
        info.converged = true;
        info.used_direct = true;
        info.solver_name = "Poisson-Direct";
    }

    constraints_.distribute(solution_);
    ghosts_valid_ = false;

    timer.stop();
    info.solve_time = timer.wall_time();
    last_solve_info_ = info;

    return info;
}

template <int dim>
typename PoissonSubsystem<dim>::Diagnostics
PoissonSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    // Solution bounds
    diag.phi_min = solution_.min();
    diag.phi_max = solution_.max();

    // Demagnetizing field statistics: H = ∇φ
    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

    double local_H_L2_sq = 0.0;
    double local_H_max = 0.0;

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        std::vector<dealii::Tensor<1, dim>> grad_phi_values(quadrature.size());
        fe_values.get_function_gradients(solution_relevant_, grad_phi_values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double H_mag = grad_phi_values[q].norm();
            local_H_max = std::max(local_H_max, H_mag);
            local_H_L2_sq += H_mag * H_mag * fe_values.JxW(q);
        }
    }

    diag.H_max = dealii::Utilities::MPI::max(local_H_max, mpi_comm_);
    diag.H_L2 = std::sqrt(
        dealii::Utilities::MPI::sum(local_H_L2_sq, mpi_comm_));

    // Solver performance
    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// Explicit instantiations
template SolverInfo PoissonSubsystem<2>::solve();
template SolverInfo PoissonSubsystem<3>::solve();
template PoissonSubsystem<2>::Diagnostics PoissonSubsystem<2>::compute_diagnostics() const;
template PoissonSubsystem<3>::Diagnostics PoissonSubsystem<3>::compute_diagnostics() const;
