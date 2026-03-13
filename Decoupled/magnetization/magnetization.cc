// ============================================================================
// magnetization/magnetization.cc — Orchestration, Accessors, Diagnostics
//
// Constructor, setup() orchestration, public method delegation,
// ghost management, equilibrium initialization, and diagnostics.
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) B167-B193
// ============================================================================

#include "magnetization/magnetization.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
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
    , fe_(1)                           // CG Q1 (Zhang Eq 3.6: N_h ∈ C⁰)
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
    pcout_ << "[Magnetization] Setting up CG-Q" << fe_.degree
           << " subsystem..." << std::endl;

    distribute_dofs();
    build_constraints();
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
    double current_time,
    bool explicit_transport)
{
    assemble_system_internal(
        Mx_old_relevant, My_old_relevant,
        phi_relevant, phi_dof_handler,
        theta_relevant, theta_dof_handler,
        ux_relevant, uy_relevant, u_dof_handler,
        dt, current_time,
        /*matrix_and_rhs=*/true,
        explicit_transport);

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
    double current_time,
    bool explicit_transport)
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
        /*matrix_and_rhs=*/false,
        explicit_transport);

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

// Mutable accessors — for AMR SolutionTransfer
template <int dim>
DoFHandler<dim>&
MagnetizationSubsystem<dim>::get_dof_handler_mutable()
{
    return dof_handler_;
}

template <int dim>
TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_solution_mutable()
{
    return Mx_solution_;
}

template <int dim>
TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_solution_mutable()
{
    return My_solution_;
}

// ============================================================================
// apply_under_relaxation() — blend M_solve with M^k for Picard stability
//
// After solve(), Mx/My_solution_ contain M_solve (owned, non-ghosted).
// The ghosted Mx/My_relevant_ still hold M^k from the PREVIOUS update_ghosts().
// (solve() only writes to solution_, not relevant_.)
//
// We blend on the owned partition: solution = ω·solution + (1-ω)·relevant
// Since relevant_ is ghosted and solution_ is owned, we iterate over
// locally_owned_dofs only (both agree on these entries).
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::apply_under_relaxation(double omega)
{
    Assert(omega > 0.0 && omega <= 1.0,
           dealii::ExcMessage("omega must be in (0, 1]"));

    if (std::abs(omega - 1.0) < 1e-15)
        return;  // No relaxation needed

    const double one_minus_omega = 1.0 - omega;

    // Iterate over locally owned entries
    for (const auto idx : locally_owned_dofs_)
    {
        Mx_solution_[idx] = omega * Mx_solution_[idx]
                          + one_minus_omega * Mx_relevant_[idx];
        My_solution_[idx] = omega * My_solution_[idx]
                          + one_minus_omega * My_relevant_[idx];
    }

    // Compress after direct writes
    Mx_solution_.compress(VectorOperation::insert);
    My_solution_.compress(VectorOperation::insert);

    // Ghosts are stale (solution changed)
    invalidate_ghosts();
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
// save_old_solution() — snapshot M^{n-1} for Picard sub-iteration
//
// Copies current ghosted M → old ghosted M.
// Must be called after update_ghosts() and before the first Picard iteration.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::save_old_solution()
{
    Assert(ghosts_valid_,
           dealii::ExcMessage("Ghosts must be valid before save_old_solution()."));

    Mx_old_relevant_ = Mx_relevant_;
    My_old_relevant_ = My_relevant_;
}

// ============================================================================
// Old-time accessors (for Picard: M^{n-1})
// ============================================================================
template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_old_relevant() const
{
    return Mx_old_relevant_;
}

template <int dim>
const TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_old_relevant() const
{
    return My_old_relevant_;
}

template <int dim>
TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_Mx_old_relevant_mutable()
{
    return Mx_old_relevant_;
}

template <int dim>
TrilinosWrappers::MPI::Vector&
MagnetizationSubsystem<dim>::get_My_old_relevant_mutable()
{
    return My_old_relevant_;
}

// ============================================================================
// initialize_equilibrium() — M⁰ = χ(θ⁰)H⁰ via global L² projection
//
// For CG, we use VectorTools::project which performs a global L² projection
// with constraint application. This produces nodal values that are globally
// optimal in the L² sense (unlike DG cell-local inversion).
//
// H = ∇φ (total field — Poisson encodes h_a via natural BCs)
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
    Vector<double>     local_rhs_x(dofs_per_cell);
    Vector<double>     local_rhs_y(dofs_per_cell);
    std::vector<double>          theta_values(n_q_points);
    std::vector<Tensor<1, dim>>  grad_phi_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Build global mass matrix and RHS, then solve via CG
    // For simplicity and consistency with constraints, we assemble and solve
    // a global L² projection: M_mass * Mx = Mx_rhs, M_mass * My = My_rhs

    // Reuse the system_matrix_ (will be overwritten during first timestep anyway)
    system_matrix_ = 0;
    Mx_rhs_ = 0;
    My_rhs_ = 0;

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

        fe_values_theta.get_function_values(theta_relevant, theta_values);
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_values);

        local_mass  = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);
            const double theta_q = theta_values[q];

            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
            {
                H = compute_applied_field<dim>(x_q, params_, current_time);
            }
            else
            {
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = grad_phi_values[q][d];
            }

            const double chi = susceptibility(theta_q, params_.physics.chi_0);
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

        cell_M->get_dof_indices(local_dof_indices);

        // Distribute with constraints
        constraints_.distribute_local_to_global(
            local_mass, local_rhs_x, local_dof_indices,
            system_matrix_, Mx_rhs_);
        constraints_.distribute_local_to_global(
            local_rhs_y, local_dof_indices, My_rhs_);
    }

    system_matrix_.compress(VectorOperation::add);
    Mx_rhs_.compress(VectorOperation::add);
    My_rhs_.compress(VectorOperation::add);

    // Solve mass system for Mx and My (reuse solve_component —
    // handles zero-RHS guard and constraint distribution)
    solve_component(Mx_solution_, Mx_rhs_, "Mx_init");
    solve_component(My_solution_, My_rhs_, "My_init");

    invalidate_ghosts();

    pcout_ << "[Magnetization] Equilibrium initialization complete."
           << std::endl;
}

// ============================================================================
// project_initial_condition() — L² project arbitrary Functions onto Mx, My
//
// For MMS tests: inject exact initial conditions onto CG space.
// Uses VectorTools::project (global L² projection). Invalidates ghosts.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::project_initial_condition(
    const Function<dim>& Mx_exact,
    const Function<dim>& My_exact)
{
    VectorTools::interpolate(dof_handler_, Mx_exact, Mx_solution_);
    constraints_.distribute(Mx_solution_);
    VectorTools::interpolate(dof_handler_, My_exact, My_solution_);
    constraints_.distribute(My_solution_);

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

            // H = ∇φ (total field — Poisson encodes h_a via natural BCs)
            // CRITICAL: Do NOT add h_a here — ∇φ IS the total field.
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
                H = compute_applied_field<dim>(x_q, params_, current_time);
            else
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = grad_phi_values[q][d];

            const double H_mag = H.norm();

            // χ(θ)
            const double chi = susceptibility(theta_q, params_.physics.chi_0);

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

    // MPI reductions — batched to reduce latency
    // SUM reductions (9 values)
    double local_sums[9] = {local_M_mag_sum, local_Mx_sum, local_My_sum,
                            local_equil_sq, local_McrossH_sq, local_volume,
                            local_Mx_integral, local_My_integral, local_MH_align_sum};
    double global_sums[9];
    MPI_Allreduce(local_sums, global_sums, 9, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    // MIN reduction (1 value)
    double global_M_mag_min;
    MPI_Allreduce(&local_M_mag_min, &global_M_mag_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm_);

    // MAX reductions (2 values)
    double local_maxes[2] = {local_M_mag_max, local_air_max};
    double global_maxes[2];
    MPI_Allreduce(local_maxes, global_maxes, 2, MPI_DOUBLE, MPI_MAX, mpi_comm_);

    // Unpack SUM results
    const double global_M_mag_sum    = global_sums[0];
    const double global_Mx_sum       = global_sums[1];
    const double global_My_sum       = global_sums[2];
    const double global_equil_sq     = global_sums[3];
    const double global_McrossH_sq   = global_sums[4];
    const double global_volume       = global_sums[5];
    const double global_Mx_integral  = global_sums[6];
    const double global_My_integral  = global_sums[7];
    const double global_MH_align_sum = global_sums[8];
    const double global_M_mag_max    = global_maxes[0];
    const double global_air_max      = global_maxes[1];

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
    diag.assemble_time = last_assemble_time_;

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

            // H = ∇φ (total field — Poisson encodes h_a via natural BCs)
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
                H = compute_applied_field<dim>(x_q, params_, current_time);
            else
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = grad_phi_values[q][d];

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

    // MPI reductions — batched by operation type
    // SUM: M_mag_sum, volume, Mx_integral, My_integral, equil_sq, McrossH_sq
    double local_sums[6] = {local_M_mag_sum, local_volume,
                            local_Mx_integral, local_My_integral,
                            local_equil_sq, local_McrossH_sq};
    double global_sums[6];
    MPI_Allreduce(local_sums, global_sums, 6, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    const double inv_vol = (global_sums[1] > 0) ? 1.0 / global_sums[1] : 0.0;
    diag.M_magnitude_mean = global_sums[0] * inv_vol;
    diag.Mx_integral      = global_sums[2];
    diag.My_integral      = global_sums[3];
    diag.M_equilibrium_departure_L2 = std::sqrt(global_sums[4]);
    diag.M_cross_H_L2              = std::sqrt(global_sums[5]);

    // MIN
    MPI_Allreduce(&local_M_mag_min, &diag.M_magnitude_min, 1,
                  MPI_DOUBLE, MPI_MIN, mpi_comm_);

    // MAX
    MPI_Allreduce(&local_M_mag_max, &diag.M_magnitude_max, 1,
                  MPI_DOUBLE, MPI_MAX, mpi_comm_);

    // No air-phase check in standalone mode
    diag.M_air_phase_max = 0.0;

    // Solver info from last solve
    diag.Mx_iterations = last_Mx_info_.iterations;
    diag.My_iterations = last_My_info_.iterations;
    diag.Mx_residual   = last_Mx_info_.residual;
    diag.My_residual   = last_My_info_.residual;
    diag.solve_time    = last_Mx_info_.solve_time + last_My_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class MagnetizationSubsystem<2>;
// template class MagnetizationSubsystem<3>;  // 2D only