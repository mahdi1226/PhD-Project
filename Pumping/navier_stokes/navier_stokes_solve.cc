// ============================================================================
// navier_stokes/navier_stokes_solve.cc - Iterative/Direct Solver
//
// Solver strategies:
//   1. Block-Schur: FGMRES + block-Schur complement preconditioner
//   2. GMRES + ILU: monolithic iterative (fallback)
//   3. Direct: Trilinos Amesos (default, paper's approach)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42e
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "navier_stokes/ns_block_schur_preconditioner.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
SolverInfo NavierStokesSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.matrix_size = ns_matrix_.m();

    const auto& solver_params = params_.solvers.navier_stokes;

    if (use_block_schur_)
    {
        // ================================================================
        // Block-Schur complement preconditioner + FGMRES
        // ================================================================
        info.solver_name = "NS-FGMRES-BlockSchur";

        const double nu_eff = params_.physics.nu + params_.physics.nu_r;

        NSBlockSchurPreconditioner<dim> schur_precond(
            A_ux_ux_, A_uy_uy_,
            Bt_ux_, Bt_uy_,
            B_ux_, B_uy_,
            M_p_, nu_eff,
            ux_locally_owned_, uy_locally_owned_, p_locally_owned_,
            n_ux_, n_uy_,
            solver_params, mpi_comm_);

        schur_precond.initialize();

        // Use relative tolerance (reduction) for outer FGMRES
        dealii::ReductionControl solver_control(
            solver_params.max_iterations,
            /*abs_tol=*/1e-12,
            /*reduction=*/solver_params.rel_tolerance);

        dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData
            fgmres_data(solver_params.schur_gmres_restart);

        dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector>
            solver(solver_control, fgmres_data);

        try
        {
            solver.solve(ns_matrix_, ns_solution_, ns_rhs_, schur_precond);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  NS FGMRES+BlockSchur failed after " << e.last_step
                       << " iters (res=" << e.last_residual
                       << "), falling back to direct\n";

                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(ns_matrix_, ns_solution_, ns_rhs_);

                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "NS-BlockSchur-Direct-Fallback";
            }
            else
            {
                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = false;
            }
        }
    }
    else if (solver_params.use_iterative)
    {
        // ================================================================
        // GMRES + ILU for the monolithic saddle-point system
        // ================================================================
        info.solver_name = "NS-GMRES-ILU";

        dealii::TrilinosWrappers::PreconditionILU ilu;
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = 2;
        ilu_data.ilu_atol = 1e-3;
        ilu_data.ilu_rtol = 1.0;
        ilu.initialize(ns_matrix_, ilu_data);

        dealii::SolverControl solver_control(
            solver_params.max_iterations,
            solver_params.abs_tolerance);

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData
            gmres_data(solver_params.gmres_restart);

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>
            solver(solver_control, gmres_data);

        try
        {
            solver.solve(ns_matrix_, ns_solution_, ns_rhs_, ilu);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  NS GMRES failed after " << e.last_step
                       << " iters (res=" << e.last_residual
                       << "), falling back to direct\n";

                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(ns_matrix_, ns_solution_, ns_rhs_);

                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "NS-Direct-Fallback";
            }
            else
            {
                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = false;
            }
        }
    }
    else
    {
        // ================================================================
        // Direct solver (paper's approach for validation)
        // ================================================================
        info.solver_name = "NS-Direct";

        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);

        try
        {
            direct_solver.solve(ns_matrix_, ns_solution_, ns_rhs_);
            info.iterations = 1;
            info.residual = 0.0;
            info.converged = true;
            info.used_direct = true;
        }
        catch (const std::exception& e)
        {
            pcout_ << "  NS direct solve failed: " << e.what() << "\n";
            info.converged = false;
        }
    }

    // Extract component solutions from monolithic vector
    extract_solutions();

    ghosts_valid_ = false;

    timer.stop();
    info.solve_time = timer.wall_time();
    last_solve_info_ = info;

    return info;
}

template <int dim>
void NavierStokesSubsystem<dim>::extract_solutions()
{
    // Split monolithic ns_solution_ into component vectors
    for (auto it = ux_locally_owned_.begin(); it != ux_locally_owned_.end(); ++it)
    {
        const auto dof = *it;
        ux_solution_[dof] = ns_solution_[dof];  // offset = 0
    }
    ux_solution_.compress(dealii::VectorOperation::insert);

    for (auto it = uy_locally_owned_.begin(); it != uy_locally_owned_.end(); ++it)
    {
        const auto dof = *it;
        uy_solution_[dof] = ns_solution_[n_ux_ + dof];
    }
    uy_solution_.compress(dealii::VectorOperation::insert);

    for (auto it = p_locally_owned_.begin(); it != p_locally_owned_.end(); ++it)
    {
        const auto dof = *it;
        p_solution_[dof] = ns_solution_[n_ux_ + n_uy_ + dof];
    }
    p_solution_.compress(dealii::VectorOperation::insert);
}

template <int dim>
typename NavierStokesSubsystem<dim>::Diagnostics
NavierStokesSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    diag.ux_min = ux_solution_.min();
    diag.ux_max = ux_solution_.max();
    diag.uy_min = uy_solution_.min();
    diag.uy_max = uy_solution_.max();
    diag.p_min  = p_solution_.min();
    diag.p_max  = p_solution_.max();

    // U_max, E_kin, divU via quadrature
    const dealii::QGauss<dim> quadrature(fe_velocity_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_velocity_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> ux_vals(n_q), uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q), grad_uy_vals(n_q);

    double local_E_kin = 0.0, local_U_max = 0.0, local_div_sq = 0.0;

    for (const auto& cell : ux_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(ux_relevant_, ux_vals);
        fe_values.get_function_values(uy_relevant_, uy_vals);
        fe_values.get_function_gradients(ux_relevant_, grad_ux_vals);
        fe_values.get_function_gradients(uy_relevant_, grad_uy_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double U_sq = ux_vals[q] * ux_vals[q] + uy_vals[q] * uy_vals[q];
            const double JxW = fe_values.JxW(q);

            local_E_kin += 0.5 * U_sq * JxW;
            local_U_max = std::max(local_U_max, std::sqrt(U_sq));

            const double div_U = grad_ux_vals[q][0] + grad_uy_vals[q][1];
            local_div_sq += div_U * div_U * JxW;
        }
    }

    diag.E_kin = dealii::Utilities::MPI::sum(local_E_kin, mpi_comm_);
    diag.U_max = dealii::Utilities::MPI::max(local_U_max, mpi_comm_);
    diag.divU_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_div_sq, mpi_comm_));

    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    // Kelvin force diagnostics (already MPI-reduced in assemble)
    diag.kelvin_cell_L2 = std::sqrt(last_kelvin_cell_L2_sq_);
    diag.kelvin_face_L2 = std::sqrt(last_kelvin_face_L2_sq_);
    diag.kelvin_Fx = last_kelvin_Fx_;
    diag.kelvin_Fy = last_kelvin_Fy_;

    return diag;
}

// Explicit instantiations
template SolverInfo NavierStokesSubsystem<2>::solve();
template SolverInfo NavierStokesSubsystem<3>::solve();
template void NavierStokesSubsystem<2>::extract_solutions();
template void NavierStokesSubsystem<3>::extract_solutions();
template NavierStokesSubsystem<2>::Diagnostics NavierStokesSubsystem<2>::compute_diagnostics() const;
template NavierStokesSubsystem<3>::Diagnostics NavierStokesSubsystem<3>::compute_diagnostics() const;
