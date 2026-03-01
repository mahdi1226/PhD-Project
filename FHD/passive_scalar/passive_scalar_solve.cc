// ============================================================================
// passive_scalar/passive_scalar_solve.cc - Solver and Diagnostics
//
// SPD system: CG + AMG default, direct solver fallback.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 104
// ============================================================================

#include "passive_scalar/passive_scalar.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
SolverInfo PassiveScalarSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = "Scalar-CG-AMG";
    info.matrix_size = dof_handler_.n_dofs();

    const auto& solver_params = params_.solvers.passive_scalar;

    if (!solver_params.use_iterative)
    {
        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);

        direct_solver.solve(system_matrix_, c_solution_, system_rhs_);

        info.iterations = 1;
        info.residual = 0.0;
        info.converged = true;
        info.used_direct = true;
        info.solver_name = "Scalar-Direct";
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
            cg.solve(system_matrix_, c_solution_, system_rhs_, amg);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  Scalar CG failed, falling back to direct\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(system_matrix_, c_solution_, system_rhs_);

                info.iterations = 1;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "Scalar-Direct-Fallback";
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
typename PassiveScalarSubsystem<dim>::Diagnostics
PassiveScalarSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    diag.c_min = c_solution_.min();
    diag.c_max = c_solution_.max();

    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    double local_L2_sq = 0.0;
    double local_mass = 0.0;

    std::vector<double> c_vals(quadrature.size());

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(c_relevant_, c_vals);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            local_L2_sq += c_vals[q] * c_vals[q] * fe_values.JxW(q);
            local_mass += c_vals[q] * fe_values.JxW(q);
        }
    }

    diag.c_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_L2_sq, mpi_comm_));
    diag.c_mass = dealii::Utilities::MPI::sum(local_mass, mpi_comm_);

    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// Explicit instantiations
template SolverInfo PassiveScalarSubsystem<2>::solve();
template SolverInfo PassiveScalarSubsystem<3>::solve();
template PassiveScalarSubsystem<2>::Diagnostics PassiveScalarSubsystem<2>::compute_diagnostics() const;
template PassiveScalarSubsystem<3>::Diagnostics PassiveScalarSubsystem<3>::compute_diagnostics() const;
