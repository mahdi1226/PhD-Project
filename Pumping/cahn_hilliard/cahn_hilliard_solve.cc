// ============================================================================
// cahn_hilliard/cahn_hilliard_solve.cc - Solver and Diagnostics
//
// Non-symmetric 2x2 block system: GMRES + ILU default, direct fallback.
// The off-diagonal blocks have different structure (gamma*K vs -S*M - eps^2*K),
// so the system is inherently non-symmetric.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "physics/material_properties.h"

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

template <int dim>
SolverInfo CahnHilliardSubsystem<dim>::solve()
{
    dealii::Timer timer;
    timer.start();

    SolverInfo info;
    info.solver_name = "CH-GMRES-ILU";
    info.matrix_size = dof_handler_.n_dofs();

    const auto& solver_params = params_.solvers.cahn_hilliard;

    if (!solver_params.use_iterative)
    {
        dealii::SolverControl direct_control(1, 0.0);
        dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);

        direct_solver.solve(system_matrix_, solution_, system_rhs_);

        info.iterations = 1;
        info.residual = 0.0;
        info.converged = true;
        info.used_direct = true;
        info.solver_name = "CH-Direct";
    }
    else
    {
        // GMRES + ILU (non-symmetric block system)
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
            solver.solve(system_matrix_, solution_, system_rhs_, ilu);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            info.used_direct = false;
        }
        catch (const dealii::SolverControl::NoConvergence& e)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  CH GMRES failed (" << e.last_step
                       << " its), falling back to direct\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(system_matrix_, solution_, system_rhs_);

                info.iterations = 1;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "CH-Direct-Fallback";
            }
            else
            {
                info.iterations = e.last_step;
                info.residual = e.last_residual;
                info.converged = false;
            }
        }
        catch (const std::exception& /*e*/)
        {
            if (solver_params.fallback_to_direct)
            {
                pcout_ << "  CH GMRES exception, falling back to direct\n";
                dealii::SolverControl direct_control(1, 0.0);
                dealii::TrilinosWrappers::SolverDirect direct_solver(direct_control);
                direct_solver.solve(system_matrix_, solution_, system_rhs_);

                info.iterations = 1;
                info.converged = true;
                info.used_direct = true;
                info.solver_name = "CH-Direct-Fallback";
            }
            else
            {
                throw;
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
typename CahnHilliardSubsystem<dim>::Diagnostics
CahnHilliardSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    const unsigned int degree = params_.fe.degree_cahn_hilliard;
    const dealii::QGauss<dim> quadrature(degree + 1);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_JxW_values);

    const dealii::FEValuesExtractors::Scalar phi_extract(0);
    const dealii::FEValuesExtractors::Scalar mu_extract(1);

    std::vector<double> phi_vals(n_q), mu_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_phi_vals(n_q);

    double local_phi_L2_sq = 0.0, local_mu_L2_sq = 0.0;
    double local_phi_mass = 0.0;
    double local_energy = 0.0;
    double local_phi_min = 1e30, local_phi_max = -1e30;
    double local_mu_min = 1e30, local_mu_max = -1e30;

    const double eps2 = params_.cahn_hilliard_params.epsilon *
                        params_.cahn_hilliard_params.epsilon;

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values[phi_extract].get_function_values(solution_relevant_, phi_vals);
        fe_values[phi_extract].get_function_gradients(solution_relevant_, grad_phi_vals);
        fe_values[mu_extract].get_function_values(solution_relevant_, mu_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const double phi_q = phi_vals[q];
            const double mu_q = mu_vals[q];

            local_phi_L2_sq += phi_q * phi_q * JxW;
            local_mu_L2_sq += mu_q * mu_q * JxW;
            local_phi_mass += phi_q * JxW;

            // Free energy: Psi(phi) + (eps^2/2)|grad phi|^2
            local_energy += (double_well_potential(phi_q) +
                             0.5 * eps2 * (grad_phi_vals[q] * grad_phi_vals[q])) * JxW;

            local_phi_min = std::min(local_phi_min, phi_q);
            local_phi_max = std::max(local_phi_max, phi_q);
            local_mu_min = std::min(local_mu_min, mu_q);
            local_mu_max = std::max(local_mu_max, mu_q);
        }
    }

    diag.phi_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_phi_L2_sq, mpi_comm_));
    diag.mu_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_mu_L2_sq, mpi_comm_));
    diag.phi_mass = dealii::Utilities::MPI::sum(local_phi_mass, mpi_comm_);
    diag.free_energy = dealii::Utilities::MPI::sum(local_energy, mpi_comm_);
    diag.phi_min = dealii::Utilities::MPI::min(local_phi_min, mpi_comm_);
    diag.phi_max = dealii::Utilities::MPI::max(local_phi_max, mpi_comm_);
    diag.mu_min = dealii::Utilities::MPI::min(local_mu_min, mpi_comm_);
    diag.mu_max = dealii::Utilities::MPI::max(local_mu_max, mpi_comm_);

    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// Explicit instantiations
template SolverInfo CahnHilliardSubsystem<2>::solve();
template SolverInfo CahnHilliardSubsystem<3>::solve();
template CahnHilliardSubsystem<2>::Diagnostics CahnHilliardSubsystem<2>::compute_diagnostics() const;
template CahnHilliardSubsystem<3>::Diagnostics CahnHilliardSubsystem<3>::compute_diagnostics() const;
