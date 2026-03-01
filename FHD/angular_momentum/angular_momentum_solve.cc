// ============================================================================
// angular_momentum/angular_momentum_solve.cc - Solver and Diagnostics
//
// SPD system: CG + AMG default, direct solver fallback.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42f
// ============================================================================

#include "angular_momentum/angular_momentum.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
SolverInfo AngularMomentumSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = "AngMom-CG-AMG";
    info.matrix_size = dof_handler_.n_dofs();

    const auto& solver_params = params_.solvers.angular_momentum;

    if (!solver_params.use_iterative)
    {
        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);

        direct_solver.solve(system_matrix_, w_solution_, system_rhs_);

        info.iterations = 1;
        info.residual = 0.0;
        info.converged = true;
        info.used_direct = true;
        info.solver_name = "AngMom-Direct";
    }
    else
    {
        // CG + AMG
        dealii::TrilinosWrappers::PreconditionAMG amg;
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg.initialize(system_matrix_, amg_data);

        dealii::SolverControl solver_control(
            solver_params.max_iterations,
            solver_params.abs_tolerance);

        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

        try
        {
            cg.solve(system_matrix_, w_solution_, system_rhs_, amg);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  AngMom CG failed, falling back to direct\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(system_matrix_, w_solution_, system_rhs_);

                info.iterations = 1;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "AngMom-Direct-Fallback";
            }
            else
            {
                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = false;
            }
        }
    }

    ghosts_valid_ = false;

    timer.stop();
    info.solve_time = timer.wall_time();
    last_solve_info_ = info;

    return info;
}

template <int dim>
typename AngularMomentumSubsystem<dim>::Diagnostics
AngularMomentumSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    diag.w_min = w_solution_.min();
    diag.w_max = w_solution_.max();

    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    double local_L2_sq = 0.0;
    double local_max = 0.0;

    std::vector<double> w_vals(quadrature.size());

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(w_relevant_, w_vals);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            local_L2_sq += w_vals[q] * w_vals[q] * fe_values.JxW(q);
            local_max = std::max(local_max, std::abs(w_vals[q]));
        }
    }

    diag.w_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_L2_sq, mpi_comm_));
    diag.w_max_abs = dealii::Utilities::MPI::max(local_max, mpi_comm_);

    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// Explicit instantiations
template SolverInfo AngularMomentumSubsystem<2>::solve();
template SolverInfo AngularMomentumSubsystem<3>::solve();
template AngularMomentumSubsystem<2>::Diagnostics AngularMomentumSubsystem<2>::compute_diagnostics() const;
template AngularMomentumSubsystem<3>::Diagnostics AngularMomentumSubsystem<3>::compute_diagnostics() const;
