// ============================================================================
// magnetization/magnetization.cc — Orchestration, Accessors, Diagnostics
//
// Constructor, setup() orchestration, public method delegation,
// ghost management, equilibrium initialization, and diagnostics.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization/magnetization.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <algorithm>

using namespace dealii;

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagnetizationSubsystem<dim>::MagnetizationSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , fe_(1)                           // DG-Q1
    , dof_handler_(triangulation)
    , preconditioner_initialized_(false)
    , ghosts_valid_(false)
    , pcout_(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0)
{
}

// ============================================================================
// setup() — orchestrate DoF distribution, sparsity, allocation
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::setup()
{
    pcout_ << "[Magnetization] Setting up DG-Q" << fe_.degree
           << " subsystem..." << std::endl;

    distribute_dofs();
    build_sparsity_pattern();
    allocate_vectors();

    preconditioner_initialized_ = false;
    ghosts_valid_ = false;

    pcout_ << "[Magnetization] Setup complete: "
           << dof_handler_.n_dofs() << " DoFs ("
           << locally_owned_dofs_.n_elements() << " local)"
           << std::endl;
}

// ============================================================================
// assemble() — full matrix + RHS (delegates to internal)
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::assemble(
    const TrilinosWrappers::MPI::Vector& Mx_old_relevant,
    const TrilinosWrappers::MPI::Vector& My_old_relevant,
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    const TrilinosWrappers::MPI::Vector& ux_relevant,
    const TrilinosWrappers::MPI::Vector& uy_relevant,
    const DoFHandler<dim>&               u_dof_handler,
    double dt,
    double current_time)
{
    assemble_system_internal(
        Mx_old_relevant, My_old_relevant,
        phi_relevant, phi_dof_handler,
        theta_relevant, theta_dof_handler,
        ux_relevant, uy_relevant, u_dof_handler,
        dt, current_time,
        /*matrix_and_rhs=*/true);

    // Matrix changed → need new preconditioner
    initialize_preconditioner();
}

// ============================================================================
// assemble_rhs_only() — RHS only for Picard iteration (matrix fixed)
//
// During Picard: matrix is fixed (U^{n-1} unchanged), only H^k updates.
// No velocity or velocity DoFHandler needed (already baked into matrix).
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::assemble_rhs_only(
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    const TrilinosWrappers::MPI::Vector& Mx_old_relevant,
    const TrilinosWrappers::MPI::Vector& My_old_relevant,
    double dt,
    double current_time)
{
    // Dummy velocity vectors — not used when matrix_and_rhs=false.
    // The matrix (containing U-dependent terms) was already assembled.
    // We pass the old solution's own vectors just to satisfy the signature;
    // assemble_system_internal will skip all matrix + face assembly.
    TrilinosWrappers::MPI::Vector dummy_ux;
    TrilinosWrappers::MPI::Vector dummy_uy;

    assemble_system_internal(
        Mx_old_relevant, My_old_relevant,
        phi_relevant, phi_dof_handler,
        theta_relevant, theta_dof_handler,
        dummy_ux, dummy_uy, phi_dof_handler,  // dummies, not accessed
        dt, current_time,
        /*matrix_and_rhs=*/false);

    // Preconditioner stays valid (matrix unchanged)
}

// ============================================================================
// solve() — solve both Mx and My with shared preconditioner
// ============================================================================
template <int dim>
SolverInfo MagnetizationSubsystem<dim>::solve()
{
    Timer timer;
    timer.start();

    last_Mx_info_ = solve_component(Mx_solution_, Mx_rhs_, "Mx");
    last_My_info_ = solve_component(My_solution_, My_rhs_, "My");

    timer.stop();

    // Ghosts are now stale
    invalidate_ghosts();

    // Build combined info
    SolverInfo combined;
    combined.iterations    = last_Mx_info_.iterations + last_My_info_.iterations;
    combined.residual      = std::max(last_Mx_info_.residual, last_My_info_.residual);
    combined.converged     = last_Mx_info_.converged && last_My_info_.converged;
    combined.solve_time    = timer.wall_time();
    combined.solver_name   = "Magnetization (Mx+My)";

    return combined;
}

// ============================================================================
// set_mms_source() — inject manufactured source for testing
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::set_mms_source(MmsSourceFunction func)
{
    mms_source_ = std::move(func);
}

// ============================================================================
// Accessors
// ============================================================================
template <int dim>
const DoFHandler<dim>&
MagnetizationSubsystem<dim>::get_dof_handler() const
{
    return dof_handler_;
}

template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_solution() const
{
    return Mx_solution_;
}

template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_solution() const
{
    return My_solution_;
}

template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_relevant() const
{
    return Mx_relevant_;
}

template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_relevant() const
{
    return My_relevant_;
}

// ============================================================================
// Ghost management
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::update_ghosts()
{
    if (ghosts_valid_)
        return;

    Mx_relevant_ = Mx_solution_;
    My_relevant_ = My_solution_;
    ghosts_valid_ = true;
}

template <int dim>
void MagnetizationSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

// ============================================================================
// initialize_equilibrium() — M⁰ = χ(θ⁰)H⁰ via cell-local L² projection
//
// For DG elements, L² projection is cell-local: each cell independently
// inverts its small local mass matrix.  No global solve required.
//
// H = ∇φ + h_a (total field = demagnetizing + applied)
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::initialize_equilibrium(
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    double current_time)
{
    pcout_ << "[Magnetization] Initializing M⁰ = χ(θ⁰)H⁰ via L² projection..."
           << std::endl;

    const FiniteElement<dim>& fe_M     = dof_handler_.get_fe();
    const FiniteElement<dim>& fe_phi   = phi_dof_handler.get_fe();
    const FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> fe_values_M(fe_M, quadrature,
        update_values | update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature,
        update_gradients);
    FEValues<dim> fe_values_theta(fe_theta, quadrature,
        update_values);

    // Local data structures
    FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs_x(dofs_per_cell);
    Vector<double>     local_rhs_y(dofs_per_cell);
    Vector<double>     local_sol_x(dofs_per_cell);
    Vector<double>     local_sol_y(dofs_per_cell);

    std::vector<double>          theta_values(n_q_points);
    std::vector<Tensor<1, dim>>  grad_phi_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Synchronized cell iteration across M, φ, θ DoFHandlers
    auto cell_M     = dof_handler_.begin_active();
    auto cell_phi   = phi_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();

    for (; cell_M != dof_handler_.end(); ++cell_M, ++cell_phi, ++cell_theta)
    {
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        // Evaluate external fields at quadrature points
        fe_values_theta.get_function_values(theta_relevant, theta_values);
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_values);

        // Build cell-local mass matrix and RHS
        local_mass = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);
            const double theta_q = theta_values[q];

            // Total field H = h_a + ∇φ
            Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
            {
                H = h_a;
            }
            else
            {
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = h_a[d] + grad_phi_values[q][d];
            }

            // χ(θ) and target M = χ(θ)H
            const double chi = susceptibility(
                theta_q, params_.physics.epsilon, params_.physics.chi_0);
            const double target_Mx = chi * H[0];
            const double target_My = (dim > 1) ? chi * H[1] : 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                local_rhs_x(i) += target_Mx * phi_i * JxW;
                local_rhs_y(i) += target_My * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    local_mass(i, j) +=
                        phi_i * fe_values_M.shape_value(j, q) * JxW;
                }
            }
        }

        // Invert cell-local mass matrix and solve
        local_mass_inv.invert(local_mass);
        local_mass_inv.vmult(local_sol_x, local_rhs_x);
        local_mass_inv.vmult(local_sol_y, local_rhs_y);

        // Write directly to global vectors (DG — cell-local DoFs)
        cell_M->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_solution_[local_dof_indices[i]] = local_sol_x(i);
            My_solution_[local_dof_indices[i]] = local_sol_y(i);
        }
    }

    // Compress after direct writes
    Mx_solution_.compress(VectorOperation::insert);
    My_solution_.compress(VectorOperation::insert);

    // Ghosts are stale after direct modification
    invalidate_ghosts();

    pcout_ << "[Magnetization] Equilibrium initialization complete."
           << std::endl;
}

// ============================================================================
// project_initial_condition() — L² project arbitrary Functions onto Mx, My
//
// For MMS tests: inject exact initial conditions into DG space.
// Cell-local inversion (no global solve). Invalidates ghosts.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::project_initial_condition(
    const Function<dim>& Mx_exact,
    const Function<dim>& My_exact)
{
    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();

    QGauss<dim> quadrature(fe_.degree + 2);
    const unsigned int n_q = quadrature.size();

    FEValues<dim> fe_values(fe_, quadrature,
        update_values | update_quadrature_points | update_JxW_values);

    FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    dealii::Vector<double> local_sol_x(dofs_per_cell);
    dealii::Vector<double> local_sol_y(dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        local_mass  = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q  = fe_values.quadrature_point(q);

            const double Mx_val = Mx_exact.value(x_q);
            const double My_val = My_exact.value(x_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                local_rhs_x(i) += Mx_val * phi_i * JxW;
                local_rhs_y(i) += My_val * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    local_mass(i, j) += phi_i * fe_values.shape_value(j, q) * JxW;
            }
        }

        local_mass_inv.invert(local_mass);
        local_mass_inv.vmult(local_sol_x, local_rhs_x);
        local_mass_inv.vmult(local_sol_y, local_rhs_y);

        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_solution_[local_dofs[i]] = local_sol_x(i);
            My_solution_[local_dofs[i]] = local_sol_y(i);
        }
    }

    Mx_solution_.compress(VectorOperation::insert);
    My_solution_.compress(VectorOperation::insert);

    invalidate_ghosts();
}

// ============================================================================
// compute_diagnostics() — full diagnostics with θ
//
// Evaluates at DG quadrature points over locally owned cells, then
// reduces across all MPI ranks.
// ============================================================================
template <int dim>
typename MagnetizationSubsystem<dim>::Diagnostics
MagnetizationSubsystem<dim>::compute_diagnostics(
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    double current_time) const
{
    Diagnostics diag;

    const FiniteElement<dim>& fe_M     = dof_handler_.get_fe();
    const FiniteElement<dim>& fe_phi   = phi_dof_handler.get_fe();
    const FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();

    QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> fe_values_M(fe_M, quadrature,
        update_values | update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature,
        update_gradients);
    FEValues<dim> fe_values_theta(fe_theta, quadrature,
        update_values);

    std::vector<double>          Mx_values(n_q_points);
    std::vector<double>          My_values(n_q_points);
    std::vector<double>          theta_values(n_q_points);
    std::vector<Tensor<1, dim>>  grad_phi_values(n_q_points);

    // Local accumulators
    double local_M_mag_sum    = 0.0;
    double local_M_mag_min    = std::numeric_limits<double>::max();
    double local_M_mag_max    = 0.0;
    double local_Mx_sum       = 0.0;
    double local_My_sum       = 0.0;
    double local_equil_sq     = 0.0;   // ||M - χH||²
    double local_air_max      = 0.0;   // max|M| in air
    double local_Mx_integral  = 0.0;
    double local_My_integral  = 0.0;
    double local_MH_align_sum = 0.0;
    double local_McrossH_sq   = 0.0;   // ||M×H||²
    double local_volume       = 0.0;

    auto cell_M     = dof_handler_.begin_active();
    auto cell_phi   = phi_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();

    for (; cell_M != dof_handler_.end(); ++cell_M, ++cell_phi, ++cell_theta)
    {
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        fe_values_M.get_function_values(Mx_relevant_, Mx_values);
        fe_values_M.get_function_values(My_relevant_, My_values);
        fe_values_theta.get_function_values(theta_relevant, theta_values);
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);
            const double theta_q = theta_values[q];

            // M at quadrature point
            const double Mx_q = Mx_values[q];
            const double My_q = My_values[q];
            const double M_mag = std::sqrt(Mx_q * Mx_q + My_q * My_q);

            // H = ∇φ + h_a
            Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
                H = h_a;
            else
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = h_a[d] + grad_phi_values[q][d];

            const double H_mag = H.norm();

            // χ(θ)
            const double chi = susceptibility(
                theta_q, params_.physics.epsilon, params_.physics.chi_0);

            // -- Field statistics --
            local_M_mag_sum += M_mag * JxW;
            local_M_mag_min = std::min(local_M_mag_min, M_mag);
            local_M_mag_max = std::max(local_M_mag_max, M_mag);
            local_Mx_sum += Mx_q * JxW;
            local_My_sum += My_q * JxW;

            // -- Equilibrium departure: ||M - χH||² --
            const double diff_x = Mx_q - chi * H[0];
            const double diff_y = (dim > 1) ? My_q - chi * H[1] : 0.0;
            local_equil_sq += (diff_x * diff_x + diff_y * diff_y) * JxW;

            // -- Air-phase confinement (θ < -0.5 = air) --
            if (theta_q < -0.5)
                local_air_max = std::max(local_air_max, M_mag);

            // -- DG conservation integrals --
            local_Mx_integral += Mx_q * JxW;
            local_My_integral += My_q * JxW;

            // -- M·H alignment --
            const double MdotH = Mx_q * H[0] + ((dim > 1) ? My_q * H[1] : 0.0);
            if (M_mag > 1e-14 && H_mag > 1e-14)
                local_MH_align_sum += (MdotH / (M_mag * H_mag)) * JxW;

            // -- M×H (2D: scalar cross product) --
            const double McrossH = Mx_q * H[1] - My_q * H[0];
            local_McrossH_sq += McrossH * McrossH * JxW;

            local_volume += JxW;
        }
    }

    // MPI reductions
    double global_M_mag_sum, global_Mx_sum, global_My_sum, global_equil_sq;
    double global_McrossH_sq, global_volume;
    double global_Mx_integral, global_My_integral, global_MH_align_sum;
    double global_M_mag_min, global_M_mag_max, global_air_max;

    MPI_Allreduce(&local_M_mag_sum,    &global_M_mag_sum,    1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_Mx_sum,       &global_Mx_sum,       1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_My_sum,       &global_My_sum,       1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_equil_sq,     &global_equil_sq,     1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_McrossH_sq,   &global_McrossH_sq,   1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_volume,       &global_volume,       1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_Mx_integral,  &global_Mx_integral,  1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_My_integral,  &global_My_integral,  1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_MH_align_sum, &global_MH_align_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_M_mag_min,    &global_M_mag_min,    1, MPI_DOUBLE, MPI_MIN, mpi_comm_);
    MPI_Allreduce(&local_M_mag_max,    &global_M_mag_max,    1, MPI_DOUBLE, MPI_MAX, mpi_comm_);
    MPI_Allreduce(&local_air_max,      &global_air_max,      1, MPI_DOUBLE, MPI_MAX, mpi_comm_);

    // Pack results
    const double inv_vol = (global_volume > 0) ? 1.0 / global_volume : 0.0;

    diag.M_magnitude_mean = global_M_mag_sum * inv_vol;
    diag.M_magnitude_min  = global_M_mag_min;
    diag.M_magnitude_max  = global_M_mag_max;
    diag.Mx_mean = global_Mx_sum * inv_vol;
    diag.My_mean = global_My_sum * inv_vol;

    diag.M_equilibrium_departure_L2 = std::sqrt(global_equil_sq);
    diag.M_air_phase_max = global_air_max;

    diag.Mx_integral = global_Mx_integral;
    diag.My_integral = global_My_integral;

    diag.M_H_alignment_mean = global_MH_align_sum * inv_vol;
    diag.M_cross_H_L2       = std::sqrt(global_McrossH_sq);

    diag.Mx_iterations = last_Mx_info_.iterations;
    diag.My_iterations = last_My_info_.iterations;
    diag.Mx_residual   = last_Mx_info_.residual;
    diag.My_residual   = last_My_info_.residual;
    diag.solve_time    = last_Mx_info_.solve_time + last_My_info_.solve_time;

    return diag;
}

// ============================================================================
// compute_diagnostics_standalone() — MMS mode (no θ, χ = χ₀ everywhere)
// ============================================================================
template <int dim>
typename MagnetizationSubsystem<dim>::Diagnostics
MagnetizationSubsystem<dim>::compute_diagnostics_standalone(
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    double current_time) const
{
    Diagnostics diag;

    const FiniteElement<dim>& fe_M   = dof_handler_.get_fe();
    const FiniteElement<dim>& fe_phi = phi_dof_handler.get_fe();

    QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> fe_values_M(fe_M, quadrature,
        update_values | update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature,
        update_gradients);

    std::vector<double>          Mx_values(n_q_points);
    std::vector<double>          My_values(n_q_points);
    std::vector<Tensor<1, dim>>  grad_phi_values(n_q_points);

    // Standalone: χ = χ₀ (no θ dependence)
    const double chi_0 = params_.physics.chi_0;

    double local_M_mag_sum    = 0.0;
    double local_M_mag_min    = std::numeric_limits<double>::max();
    double local_M_mag_max    = 0.0;
    double local_Mx_integral  = 0.0;
    double local_My_integral  = 0.0;
    double local_equil_sq     = 0.0;
    double local_McrossH_sq   = 0.0;
    double local_volume       = 0.0;

    auto cell_M   = dof_handler_.begin_active();
    auto cell_phi = phi_dof_handler.begin_active();

    for (; cell_M != dof_handler_.end(); ++cell_M, ++cell_phi)
    {
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);

        fe_values_M.get_function_values(Mx_relevant_, Mx_values);
        fe_values_M.get_function_values(My_relevant_, My_values);
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);

            const double Mx_q = Mx_values[q];
            const double My_q = My_values[q];
            const double M_mag = std::sqrt(Mx_q * Mx_q + My_q * My_q);

            // H = ∇φ + h_a
            Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
                H = h_a;
            else
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = h_a[d] + grad_phi_values[q][d];

            // Field stats
            local_M_mag_sum += M_mag * JxW;
            local_M_mag_min = std::min(local_M_mag_min, M_mag);
            local_M_mag_max = std::max(local_M_mag_max, M_mag);

            // Conservation
            local_Mx_integral += Mx_q * JxW;
            local_My_integral += My_q * JxW;

            // Equilibrium departure with χ₀
            const double diff_x = Mx_q - chi_0 * H[0];
            const double diff_y = (dim > 1) ? My_q - chi_0 * H[1] : 0.0;
            local_equil_sq += (diff_x * diff_x + diff_y * diff_y) * JxW;

            // M×H
            const double McrossH = Mx_q * H[1] - My_q * H[0];
            local_McrossH_sq += McrossH * McrossH * JxW;

            local_volume += JxW;
        }
    }

    // MPI reductions
    double global_val;

    MPI_Allreduce(&local_M_mag_sum,   &global_val, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    double global_volume_sum;
    MPI_Allreduce(&local_volume, &global_volume_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    const double inv_vol = (global_volume_sum > 0) ? 1.0 / global_volume_sum : 0.0;

    diag.M_magnitude_mean = global_val * inv_vol;

    MPI_Allreduce(&local_M_mag_min, &diag.M_magnitude_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm_);
    MPI_Allreduce(&local_M_mag_max, &diag.M_magnitude_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm_);
    MPI_Allreduce(&local_Mx_integral, &diag.Mx_integral,   1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_My_integral, &diag.My_integral,   1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    MPI_Allreduce(&local_equil_sq, &global_val, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    diag.M_equilibrium_departure_L2 = std::sqrt(global_val);

    MPI_Allreduce(&local_McrossH_sq, &global_val, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    diag.M_cross_H_L2 = std::sqrt(global_val);

    // No air-phase check in standalone mode
    diag.M_air_phase_max = 0.0;

    // Solver info from last solve
    diag.Mx_iterations = last_Mx_info_.iterations;
    diag.My_iterations = last_My_info_.iterations;
    diag.Mx_residual   = last_Mx_info_.residual;
    diag.My_residual   = last_My_info_.residual;
    diag.solve_time    = last_Mx_info_.solve_time + last_My_info_.solve_time;

    return diag;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class MagnetizationSubsystem<2>;
template class MagnetizationSubsystem<3>;