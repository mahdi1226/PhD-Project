// ============================================================================
// poisson/poisson.cc - Magnetostatic Poisson Subsystem (Orchestration)
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   (∇φ^k, ∇X) = (h_a^k − M^k, ∇X)    ∀X ∈ X_h
//
// This file contains:
//   - Constructor
//   - setup() orchestration
//   - Accessors (dof_handler, solution, solution_relevant)
//   - Ghost management (update_ghosts, invalidate_ghosts)
//   - Diagnostics computation
//
// Other files:
//   poisson_setup.cc    — DoFs, constraints, sparsity, vector allocation
//   poisson_assemble.cc — matrix (once), RHS (per step), AMG init
//   poisson_solve.cc    — CG + cached AMG
// ============================================================================

#include "poisson/poisson.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <chrono>
#include <limits>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
PoissonSubsystem<dim>::PoissonSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , fe_(params.fe.degree_potential)
    , dof_handler_(triangulation)
{
}

// ============================================================================
// setup() — call once after mesh is ready
//
// Order matters:
//   1. Distribute DoFs         → dof_handler_ ready
//   2. Build constraints       → hanging nodes + pin DoF 0
//   3. Build sparsity pattern  → Trilinos sparsity
//   4. Allocate vectors        → RHS, solution, ghosted
//   5. Assemble matrix         → (∇φ, ∇X) — CONSTANT
//   6. Initialize AMG          → preconditioner cached for all solves
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::setup()
{
    pcout_ << "[Poisson] Setting up subsystem...\n";

    distribute_dofs();
    build_constraints();
    build_sparsity_pattern();
    allocate_vectors();
    assemble_matrix();
    initialize_preconditioner();

    pcout_ << "[Poisson] Setup complete: "
           << dof_handler_.n_dofs() << " DoFs, "
           << locally_owned_dofs_.n_elements() << " local\n";
}

// ============================================================================
// MMS source injection
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::set_mms_source(MMSSourceFunction source)
{
    mms_source_ = std::move(source);
}

// ============================================================================
// Accessors
// ============================================================================
template <int dim>
const dealii::DoFHandler<dim>&
PoissonSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PoissonSubsystem<dim>::get_solution() const
{
    return solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
PoissonSubsystem<dim>::get_solution_relevant() const
{
    return solution_relevant_;
}

// ============================================================================
// Ghost management
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        solution_relevant_ = solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void PoissonSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

// ============================================================================
// compute_diagnostics() — full version with θ for μ(θ)-dependent quantities
//
// Computes:
//   - φ bounds (min, max)
//   - H = ∇φ statistics (max, L2 norm)
//   - E_mag = ½∫μ(θ)|∇φ|² dΩ  (magnetic energy)
//   - μ(θ) range
//   - Gauss law residual: ‖Δφ − (f_source)‖ (post-solve check)
//   - Pinned DoF check: |φ(DoF 0)| ≈ 0
//   - Last solver stats
// ============================================================================
template <int dim>
typename PoissonSubsystem<dim>::Diagnostics
PoissonSubsystem<dim>::compute_diagnostics(
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    double /*current_time*/) const
{
    Diagnostics diag;

    // Solver stats from last solve
    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    // Pinned DoF check
    if (locally_owned_dofs_.is_element(0))
        diag.phi_pinned_value = std::abs(solution_[0]);
    MPI_Allreduce(MPI_IN_PLACE, &diag.phi_pinned_value, 1,
                  MPI_DOUBLE, MPI_MAX, mpi_comm_);

    // Quadrature for field quantities
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    const bool has_theta = (theta_relevant.size() > 0);
    std::unique_ptr<dealii::FEValues<dim>> theta_fe_values_ptr;
    if (has_theta)
    {
        theta_fe_values_ptr = std::make_unique<dealii::FEValues<dim>>(
            theta_dof_handler.get_fe(), quadrature,
            dealii::update_values);
    }

    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<double> theta_values(n_q_points);

    // Local accumulators
    double local_phi_min = std::numeric_limits<double>::max();
    double local_phi_max = std::numeric_limits<double>::lowest();
    double local_H_max = 0.0;
    double local_H_L2_sq = 0.0;
    double local_E_mag = 0.0;
    double local_mu_min = std::numeric_limits<double>::max();
    double local_mu_max = std::numeric_limits<double>::lowest();

    auto phi_cell = dof_handler_.begin_active();
    auto theta_cell = has_theta
        ? theta_dof_handler.begin_active()
        : decltype(theta_dof_handler.begin_active())();

    for (; phi_cell != dof_handler_.end(); ++phi_cell)
    {
        if (!phi_cell->is_locally_owned())
        {
            if (has_theta) ++theta_cell;
            continue;
        }

        phi_fe_values.reinit(phi_cell);
        phi_fe_values.get_function_values(solution_relevant_, phi_values);
        phi_fe_values.get_function_gradients(solution_relevant_, phi_gradients);

        if (has_theta)
        {
            theta_fe_values_ptr->reinit(theta_cell);
            theta_fe_values_ptr->get_function_values(theta_relevant, theta_values);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);

            // φ bounds
            local_phi_min = std::min(local_phi_min, phi_values[q]);
            local_phi_max = std::max(local_phi_max, phi_values[q]);

            // |∇φ| = |H|
            const double H_mag = phi_gradients[q].norm();
            local_H_max = std::max(local_H_max, H_mag);
            local_H_L2_sq += H_mag * H_mag * JxW;

            // μ(θ) and E_mag = ½∫μ|H|²
            double mu = 1.0;
            if (has_theta)
            {
                mu = permeability(theta_values[q],
                                  params_.physics.epsilon,
                                  params_.physics.chi_0);
                local_mu_min = std::min(local_mu_min, mu);
                local_mu_max = std::max(local_mu_max, mu);
            }
            local_E_mag += 0.5 * mu * H_mag * H_mag * JxW;
        }

        if (has_theta) ++theta_cell;
    }

    // Global reductions
    MPI_Allreduce(MPI_IN_PLACE, &local_phi_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm_);
    MPI_Allreduce(MPI_IN_PLACE, &local_phi_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_);
    MPI_Allreduce(MPI_IN_PLACE, &local_H_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_);
    MPI_Allreduce(MPI_IN_PLACE, &local_H_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(MPI_IN_PLACE, &local_E_mag, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    diag.phi_min = local_phi_min;
    diag.phi_max = local_phi_max;
    diag.H_max = local_H_max;
    diag.H_L2 = std::sqrt(local_H_L2_sq);
    diag.E_mag = local_E_mag;

    if (has_theta)
    {
        MPI_Allreduce(MPI_IN_PLACE, &local_mu_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm_);
        MPI_Allreduce(MPI_IN_PLACE, &local_mu_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_);
        diag.mu_min = local_mu_min;
        diag.mu_max = local_mu_max;
    }
    else
    {
        diag.mu_min = 1.0;
        diag.mu_max = 1.0;
    }

    return diag;
}

// ============================================================================
// compute_diagnostics() — lightweight version without θ (standalone test)
// ============================================================================
template <int dim>
typename PoissonSubsystem<dim>::Diagnostics
PoissonSubsystem<dim>::compute_diagnostics() const
{
    // Call full version with empty theta
    dealii::TrilinosWrappers::MPI::Vector empty_theta;
    dealii::DoFHandler<dim> empty_dof(triangulation_);
    return compute_diagnostics(empty_theta, empty_dof, 0.0);
}

// ============================================================================
// Explicit instantiations (methods defined in THIS file only)
// ============================================================================
template PoissonSubsystem<2>::PoissonSubsystem(
    const Parameters&, MPI_Comm,
    dealii::parallel::distributed::Triangulation<2>&);
template PoissonSubsystem<3>::PoissonSubsystem(
    const Parameters&, MPI_Comm,
    dealii::parallel::distributed::Triangulation<3>&);

template void PoissonSubsystem<2>::setup();
template void PoissonSubsystem<3>::setup();

template void PoissonSubsystem<2>::set_mms_source(PoissonSubsystem<2>::MMSSourceFunction);
template void PoissonSubsystem<3>::set_mms_source(PoissonSubsystem<3>::MMSSourceFunction);

template const dealii::DoFHandler<2>& PoissonSubsystem<2>::get_dof_handler() const;
template const dealii::DoFHandler<3>& PoissonSubsystem<3>::get_dof_handler() const;

template const dealii::TrilinosWrappers::MPI::Vector& PoissonSubsystem<2>::get_solution() const;
template const dealii::TrilinosWrappers::MPI::Vector& PoissonSubsystem<3>::get_solution() const;

template const dealii::TrilinosWrappers::MPI::Vector& PoissonSubsystem<2>::get_solution_relevant() const;
template const dealii::TrilinosWrappers::MPI::Vector& PoissonSubsystem<3>::get_solution_relevant() const;

template void PoissonSubsystem<2>::update_ghosts();
template void PoissonSubsystem<3>::update_ghosts();

template void PoissonSubsystem<2>::invalidate_ghosts();
template void PoissonSubsystem<3>::invalidate_ghosts();

template PoissonSubsystem<2>::Diagnostics PoissonSubsystem<2>::compute_diagnostics(
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<2>&, double) const;
template PoissonSubsystem<3>::Diagnostics PoissonSubsystem<3>::compute_diagnostics(
    const dealii::TrilinosWrappers::MPI::Vector&, const dealii::DoFHandler<3>&, double) const;

template PoissonSubsystem<2>::Diagnostics PoissonSubsystem<2>::compute_diagnostics() const;
template PoissonSubsystem<3>::Diagnostics PoissonSubsystem<3>::compute_diagnostics() const;