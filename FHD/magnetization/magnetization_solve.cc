// ============================================================================
// magnetization/magnetization_solve.cc - Linear Solve and Diagnostics
//
// Solves Mx and My separately using the same matrix and preconditioner.
// Default: direct solver. Fallback: GMRES + ILU.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42c
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
SolverInfo MagnetizationSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = "Magnetization-Direct";
    info.matrix_size = dof_handler_.n_dofs();

    const auto& solver_params = params_.solvers.magnetization;

    if (!solver_params.use_iterative)
    {
        // Direct solver for both Mx and My
        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);

        direct_solver.solve(system_matrix_, Mx_solution_, Mx_rhs_);
        direct_solver.solve(system_matrix_, My_solution_, My_rhs_);

        info.iterations = 1;
        info.residual = 0.0;
        info.converged = true;
        info.used_direct = true;
    }
    else
    {
        // GMRES + ILU
        dealii::TrilinosWrappers::PreconditionILU ilu;
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = 1;
        ilu.initialize(system_matrix_, ilu_data);

        dealii::SolverControl solver_control(
            solver_params.max_iterations,
            solver_params.abs_tolerance);

        dealii::TrilinosWrappers::SolverGMRES solver(solver_control);

        try
        {
            solver.solve(system_matrix_, Mx_solution_, Mx_rhs_, ilu);
            solver.solve(system_matrix_, My_solution_, My_rhs_, ilu);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
            info.solver_name = "Magnetization-GMRES-ILU";
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  Magnetization GMRES failed, falling back to direct\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(system_matrix_, Mx_solution_, Mx_rhs_);
                direct_solver.solve(system_matrix_, My_solution_, My_rhs_);

                info.iterations = 1;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "Magnetization-Direct-Fallback";
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
typename MagnetizationSubsystem<dim>::Diagnostics
MagnetizationSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    diag.Mx_min = Mx_solution_.min();
    diag.Mx_max = Mx_solution_.max();
    diag.My_min = My_solution_.min();
    diag.My_max = My_solution_.max();

    // M_L2 = sqrt(||Mx||^2 + ||My||^2)
    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
                                    dealii::update_values |
                                    dealii::update_JxW_values);

    double local_M_L2_sq = 0.0;
    double local_M_max = 0.0;

    std::vector<double> mx_vals(quadrature.size());
    std::vector<double> my_vals(quadrature.size());

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(Mx_relevant_, mx_vals);
        fe_values.get_function_values(My_relevant_, my_vals);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double M_sq = mx_vals[q] * mx_vals[q] + my_vals[q] * my_vals[q];
            local_M_L2_sq += M_sq * fe_values.JxW(q);
            local_M_max = std::max(local_M_max, std::sqrt(M_sq));
        }
    }

    diag.M_L2 = std::sqrt(
        dealii::Utilities::MPI::sum(local_M_L2_sq, mpi_comm_));
    diag.M_max = dealii::Utilities::MPI::max(local_M_max, mpi_comm_);

    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// Explicit instantiations
template SolverInfo MagnetizationSubsystem<2>::solve();
template SolverInfo MagnetizationSubsystem<3>::solve();
template MagnetizationSubsystem<2>::Diagnostics MagnetizationSubsystem<2>::compute_diagnostics() const;
template MagnetizationSubsystem<3>::Diagnostics MagnetizationSubsystem<3>::compute_diagnostics() const;
