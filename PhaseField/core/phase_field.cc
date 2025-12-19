// ============================================================================
// core/phase_field.cc - Phase Field Problem Main Implementation
//
// Contains run(), do_time_step(), and all solve methods.
//
// Time stepping follows Paper Algorithm 1:
//   1. Solve CH → θ^k, ψ^k
//   2. Solve Poisson → φ^k (H^k = ∇φ^k)
//   3. Solve Magnetization → M^k (DG transport) [if enabled]
//   4. Solve NS → u^k, p^k (uses θ^{k-1}, ψ^k, H^k, M^k)
//
// CRITICAL: θ is LAGGED in NS (θ^{k-1}) for energy stability!
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "assembly/ch_assembler.h"
#include "assembly/poisson_assembler.h"
#include "assembly/ns_assembler.h"
#include "solvers/ch_solver.h"
#include "solvers/poisson_solver.h"
#include "solvers/ns_solver.h"
#include "utilities/tools.h"
#include "diagnostics/ch_diagnostics.h"
#include "../mms/mms_runtime.h"
#include "diagnostics/poisson_diagnostics.h"
#include "diagnostics/ns_diagnostics.h"
#include "mms/ch_mms.h"
#include "mms/poisson_mms.h"
#include "mms/ns_mms.h"
#include "physics/material_properties.h"


#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <limits>
#include <algorithm>

// ============================================================================
// run() - Main entry point
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::run()
{
    const bool mms_mode = params_.enable_mms;

    std::cout << "============================================================\n";
    if (mms_mode)
        std::cout << "  Ferrofluid Solver - MMS VERIFICATION MODE\n";
    else if (params_.enable_ns)
        std::cout << "  Ferrofluid Solver (CH + Poisson + NS)\n";
    else
        std::cout << "  Cahn-Hilliard Solver (Standalone Test)\n";
    std::cout << "  Reference: Nochetto et al. CMAME 309 (2016)\n";
    std::cout << "============================================================\n\n";

    // Create output directory with timestamp
    std::string output_dir = timestamped_folder(params_.output.folder);
    std::filesystem::create_directories(output_dir);
    std::cout << "[Info] Output directory: " << output_dir << "\n\n";

    // Store output directory for later use
    const_cast<Parameters&>(params_).output.folder = output_dir;

    // ========================================================================
    // Setup
    // ========================================================================
    std::cout << "--- Setup Phase ---\n";
    setup_mesh();
    setup_dof_handlers();
    setup_constraints();
    setup_ch_system();
    if (params_.enable_magnetic)
    {
        setup_poisson_system();
        if (params_.use_dg_transport)
            setup_magnetization_system();
    }
    if (params_.enable_ns)
        setup_ns_system();
    initialize_solutions();

    const double h_min = get_min_h();
    std::cout << "[Info] Mesh h_min = " << h_min << "\n";

    // Print parameters
    std::cout << "\n--- Parameters ---\n";
    std::cout << "  epsilon = " << epsilon << "\n";
    std::cout << "  gamma   = " << mobility << "\n";
    std::cout << "  dt      = " << params_.time.dt << "\n";
    std::cout << "  t_final = " << params_.time.t_final << "\n";
    if (params_.enable_magnetic)
    {
        std::cout << "  Magnetic: ENABLED\n";
        std::cout << "  chi_0 (susceptibility) = " << chi_0 << "\n";
        if (params_.use_dg_transport)
            std::cout << "  DG transport: ENABLED (tau_M = " << tau_M << ")\n";
        else
            std::cout << "  DG transport: disabled (quasi-equilibrium M = chi*H)\n";
    }
    if (params_.enable_ns)
    {
        std::cout << "  Navier-Stokes: ENABLED\n";
        std::cout << "  nu_water = " << nu_water << "\n";
        std::cout << "  nu_ferro = " << nu_ferro << "\n";
        if (params_.enable_gravity)
std::cout << "  Gravity: ENABLED (g = " << gravity_dimensionless << ")\n";    }
    if (mms_mode)
    {
        std::cout << "  MMS mode: ENABLED\n";
        std::cout << "  t_init  = " << params_.mms_t_init << "\n";
    }

    // ========================================================================
    // Initial field solve (to populate phi for output)
    // ========================================================================
    if (params_.enable_magnetic)
        solve_poisson();
    output_results(0);

    // ========================================================================
    // Time loop
    // ========================================================================
    std::cout << "\n--- Time Integration ---\n";
    std::cout << std::setw(8) << "Step"
        << std::setw(12) << "Time"
        << std::setw(14) << "Mass"
        << std::setw(14) << "E_CH"
        << std::setw(14) << "E_kin" << "\n";
    std::cout << std::string(62, '-') << "\n";

    const double dt = params_.time.dt;
    unsigned int output_counter = 0;

    // Checkpoint for mms
    if (params_.enable_mms)
    {
        auto ic_errors = compute_ch_mms_errors<dim>(
            theta_dof_handler_,
            psi_dof_handler_,
            theta_solution_,
            psi_solution_,
            time_);
        std::cout << "[MMS IC CHECK] θ L2 = " << std::scientific << ic_errors.theta_L2
            << ", ψ L2 = " << ic_errors.psi_L2 << "\n";

        // NS MMS IC check
        if (params_.enable_ns)
        {
            const double L_y = params_.domain.y_max - params_.domain.y_min;
            auto ns_errors = compute_ns_mms_error<dim>(
                ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
                ux_solution_, uy_solution_, p_solution_,
                time_, L_y);
            std::cout << "[MMS IC CHECK] ux L2 = " << std::scientific << ns_errors.ux_L2
                << ", uy L2 = " << ns_errors.uy_L2
                << ", p L2 = " << ns_errors.p_L2 << "\n";
        }
    }

    while (time_ < params_.time.t_final - 1e-12)
    {
        double current_dt = dt;
        if (time_ + current_dt > params_.time.t_final)
            current_dt = params_.time.t_final - time_;

        do_time_step(current_dt);

        // AMR: Refine mesh every amr_interval steps (Paper Section 6.1)
        if (params_.mesh.use_amr &&
            timestep_number_ % params_.mesh.amr_interval == 0 &&
            timestep_number_ > 0)
        {
            refine_mesh();
        }
        // ==============================

        // Output periodically
        if (timestep_number_ % params_.output.frequency == 0 ||
            time_ >= params_.time.t_final - 1e-12)
        {
            ++output_counter;
            output_results(output_counter);

            // Compute diagnostics
            const double mass = compute_mass();
            const double E_ch = compute_ch_energy();
            const double E_kin = params_.enable_ns ? compute_kinetic_energy() : 0.0;

            std::cout << std::setw(8) << timestep_number_
                << std::setw(12) << std::fixed << std::setprecision(4) << time_
                << std::setw(14) << std::scientific << std::setprecision(4) << mass
                << std::setw(14) << E_ch
                << std::setw(14) << E_kin << "\n";
        }
    }

    std::cout << "\n--- Simulation Complete ---\n";
    std::cout << "[Info] Final time: " << time_ << "\n";
    std::cout << "[Info] Total steps: " << timestep_number_ << "\n";

    // MMS error computation at final time
    if (mms_mode)
    {
        std::cout << "\n--- MMS ERROR ANALYSIS ---\n";
        compute_mms_errors();
    }

    std::cout << "[Info] Output saved to: " << params_.output.folder << "\n";
    // NS MMS errors
    if (params_.enable_ns)
    {
        const double L_y = params_.domain.y_max - params_.domain.y_min;
        auto ns_errors = compute_ns_mms_error<dim>(
            ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
            ux_solution_, uy_solution_, p_solution_,
            time_, L_y);

        std::cout << "\n--- NS MMS Errors ---\n";
        ns_errors.print(params_.mesh.initial_refinement, get_min_h());

        std::cout << "\nNS Convergence (h, ux_L2, ux_H1, uy_L2, uy_H1, p_L2):\n";
        ns_errors.print_for_convergence(get_min_h());
    }
}

// ============================================================================
// do_time_step() - Single time step (staggered approach)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::do_time_step(double dt)
{
    std::cout << "[DEBUG do_time_step START] use_direct_after_amr_ = "
        << use_direct_after_amr_ << "\n";

    // Save old solutions BEFORE updating (for lagging)
    theta_old_solution_ = theta_solution_;

    if (params_.enable_ns)
    {
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;
    }

    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        mx_old_solution_ = mx_solution_;
        my_old_solution_ = my_solution_;
    }

    const double time_new = time_ + dt;

    if (params_.enable_mms)
        update_mms_boundary_constraints(time_new);

    // Step 1: Cahn-Hilliard
    solve_ch();

    // CH diagnostics
    CHDiagnosticData ch_data = compute_ch_diagnostics<dim>(
        theta_dof_handler_, theta_solution_, epsilon,
        timestep_number_, time_ + dt, dt, ch_energy_prev_);
    print_ch_diagnostics(ch_data, params_.output.verbose);
    ch_energy_prev_ = ch_data.energy;

    // Step 2: Poisson
    if (params_.enable_magnetic)
        solve_poisson();

    // Poisson diagnostics
    if (params_.enable_magnetic)
    {
        auto poisson_diag = compute_poisson_diagnostics<dim>(
            phi_dof_handler_, phi_solution_,
            theta_dof_handler_, theta_solution_,
            params_, time_);
        poisson_diag.print(timestep_number_, time_);
    }

    if (params_.enable_mms && params_.enable_magnetic)
    {
        auto poisson_error = compute_poisson_mms_error<dim>(
            phi_dof_handler_, phi_solution_, time_,
            params_.domain.y_max - params_.domain.y_min);
        poisson_error.print(params_.mesh.initial_refinement, get_min_h());
    }

    // Step 3: Magnetization (DG transport)
    if (params_.enable_magnetic && params_.use_dg_transport)
        solve_magnetization();

    // Step 4: Navier-Stokes (uses theta^{k-1}!)
    if (params_.enable_ns)
        solve_ns();

    // Compute NS MMS errors
    if (params_.enable_mms && params_.enable_ns)
    {
        const double L_y = params_.domain.y_max - params_.domain.y_min;
        auto ns_error = compute_ns_mms_error<dim>(
            ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
            ux_solution_, uy_solution_, p_solution_,
            time_, L_y);
        ns_error.print(params_.mesh.initial_refinement, get_min_h());
    }

    if (params_.enable_ns)
    {
        auto ns_diag = compute_ns_diagnostics<dim>(
            ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
            ux_solution_, uy_solution_, p_solution_,
            params_.time.dt, get_min_h());
        ns_diag.print(timestep_number_, time_);
    }


    time_ = time_new;
    ++timestep_number_;
}

// ============================================================================
// solve_ch() - Solve Cahn-Hilliard system
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ch()
{
    const dealii::Vector<double>& ux_for_ch = params_.enable_ns ? ux_solution_ : ux_old_solution_;
    const dealii::Vector<double>& uy_for_ch = params_.enable_ns ? uy_solution_ : uy_old_solution_;

    ch_matrix_ = 0;
    ch_rhs_ = 0;

    assemble_ch_system<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_old_solution_,
        ux_for_ch,
        uy_for_ch,
        params_,
        params_.time.dt,
        time_,
        theta_to_ch_map_,
        psi_to_ch_map_,
        ch_matrix_,
        ch_rhs_);

    ch_combined_constraints_.condense(ch_matrix_, ch_rhs_);

    solve_ch_system(ch_matrix_, ch_rhs_,
                    ch_combined_constraints_,
                    theta_to_ch_map_,
                    psi_to_ch_map_,
                    theta_solution_,
                    psi_solution_);

    theta_constraints_.distribute(theta_solution_);
    psi_constraints_.distribute(psi_solution_);
}

// ============================================================================
// ============================================================================
// solve_poisson() - Solve magnetostatic Poisson equation
//
// QUASI-EQUILIBRIUM MODEL:
//   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)
//
// where μ(θ) = 1 + χ(θ) is the phase-dependent permeability.
//
// For quasi-equilibrium (M = χH), the M term in Eq. 42d is absorbed into
// the permeability μ(θ). This 8-arg signature (using θ, not M) is correct.
//
// The 9-arg signature with explicit M_dof_handler is only needed for the
// full DG transport model (Eq. 42c), which is not yet implemented.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_poisson()
{
    phi_matrix_ = 0;
    phi_rhs_ = 0;

    if (params_.enable_mms)
    {
        // MMS mode: use manufactured source h_a = μ(θ)∇φ_exact
        assemble_poisson_system_mms_quasi_equilibrium<dim>(
            phi_dof_handler_,
            theta_dof_handler_,
            theta_solution_,
            params_,
            time_,
            params_.domain.y_max - params_.domain.y_min,
            phi_matrix_,
            phi_rhs_,
            phi_constraints_);
    }
    else
    {
        // Physical mode: use dipole field h_a
        assemble_poisson_system<dim>(
            phi_dof_handler_,
            theta_dof_handler_,
            theta_solution_,
            params_,
            time_,
            phi_matrix_,
            phi_rhs_,
            phi_constraints_);
    }

    solve_poisson_system(phi_matrix_, phi_rhs_, phi_solution_,
                         phi_constraints_,
                         params_.output.verbose);
}

// ============================================================================
// ============================================================================
// solve_magnetization() - Compute magnetization
//
// QUASI-EQUILIBRIUM MODEL (τ_M → 0):
//   M = χ(θ) H  where H = ∇φ
//
// This is NOT the full DG transport equation (Eq. 42c). The flag
// `use_dg_transport` controls whether M is pre-computed and stored (true)
// or computed inline in NS assembler (false). Both give quasi-equilibrium.
//
// Full Eq. 42c: ∂M/∂t + B_h^m(U,Z,M) = (1/τ_M)(χH - M)
// is NOT YET IMPLEMENTED.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_magnetization()
{
    // If not using stored magnetization, NS assembler computes χH directly
    if (!params_.use_dg_transport)
        return;

    // Warn once that this is quasi-equilibrium, not full DG transport
    if (tau_M > 0.0)
    {
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "[Magnetization] WARNING: tau_M = "
                << tau_M << " > 0, but full DG transport\n"
                << "                (Eq. 42c) is not implemented. "
                << "Using quasi-equilibrium M = χH.\n";
            warned = true;
        }
    }

    // Quasi-equilibrium: M = χ(θ) H via L² projection onto DG0
    dealii::QGauss<dim> quadrature(2);
    dealii::FEValues<dim> phi_fe_values(phi_dof_handler_.get_fe(), quadrature,
                                        dealii::update_gradients | dealii::update_JxW_values |
                                        dealii::update_quadrature_points);
    dealii::FEValues<dim> theta_fe_values(theta_dof_handler_.get_fe(), quadrature,
                                          dealii::update_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    unsigned int cell_index = 0;
    auto theta_cell = theta_dof_handler_.begin_active();
    auto phi_cell = phi_dof_handler_.begin_active();

    for (auto& mx_cell : mx_dof_handler_.active_cell_iterators())
    {
        (void)mx_cell; // Silence unused warning

        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        phi_fe_values.get_function_gradients(phi_solution_, phi_gradients);
        theta_fe_values.get_function_values(theta_solution_, theta_values);

        double mx_avg = 0.0, my_avg = 0.0, vol = 0.0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double chi = compute_susceptibility(theta, epsilon,
                                                      chi_0);
            const dealii::Tensor<1, dim>& H = phi_gradients[q];

            mx_avg += chi * H[0] * phi_fe_values.JxW(q);
            my_avg += chi * H[1] * phi_fe_values.JxW(q);
            vol += phi_fe_values.JxW(q);
        }

        if (vol > 0)
        {
            mx_solution_[cell_index] = mx_avg / vol;
            my_solution_[cell_index] = my_avg / vol;
        }

        ++cell_index;
        ++theta_cell;
        ++phi_cell;
    }
}

// ============================================================================
// solve_ns() - Solve Navier-Stokes system
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ns()
{
    ns_matrix_ = 0;
    ns_rhs_ = 0;

    assemble_ns_system<dim>(
        ux_dof_handler_,
        uy_dof_handler_,
        p_dof_handler_,
        theta_dof_handler_,
        psi_dof_handler_,
        params_.enable_magnetic ? &phi_dof_handler_ : nullptr,
        params_.enable_magnetic && params_.use_dg_transport ? &mx_dof_handler_ : nullptr,
        ux_old_solution_,
        uy_old_solution_,
        theta_old_solution_,
        psi_solution_,
        params_.enable_magnetic ? &phi_solution_ : nullptr,
        params_.enable_magnetic && params_.use_dg_transport ? &mx_solution_ : nullptr,
        params_.enable_magnetic && params_.use_dg_transport ? &my_solution_ : nullptr,
        params_,
        params_.time.dt,
        time_,
        ux_to_ns_map_,
        uy_to_ns_map_,
        p_to_ns_map_,
        ns_combined_constraints_,
        ns_matrix_,
        ns_rhs_);

    // Use FGMRES + Block Schur preconditioner (following deal.II step-56)
    // Use direct solver if:
    // 1. After AMR (use_direct_after_amr_), OR
    // 2. User requested direct solver (--direct flag)
    if (use_direct_after_amr_ || !params_.solvers.ns.use_iterative)
    {
        solve_ns_system_direct(
            ns_matrix_,
            ns_rhs_,
            ns_solution_,
            ns_combined_constraints_,
            params_.output.verbose);

        // Only reset if it was the AMR flag (not user preference)
        if (use_direct_after_amr_)
            use_direct_after_amr_ = false;
    }
    else
    {
        solve_ns_system_schur(
            ns_matrix_,
            ns_rhs_,
            ns_solution_,
            ns_combined_constraints_,
            pressure_mass_matrix_,
            ux_to_ns_map_,
            uy_to_ns_map_,
            p_to_ns_map_,
            params_.output.verbose);
    }

    extract_ns_solutions(
        ns_solution_,
        ux_to_ns_map_,
        uy_to_ns_map_,
        p_to_ns_map_,
        ux_solution_,
        uy_solution_,
        p_solution_);

    ux_constraints_.distribute(ux_solution_);
    uy_constraints_.distribute(uy_solution_);
    p_constraints_.distribute(p_solution_);
}

// ============================================================================
// update_mms_boundary_constraints()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::update_mms_boundary_constraints(double time)
{
    theta_constraints_.clear();
    psi_constraints_.clear();

    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);

    apply_ch_mms_boundary_constraints<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_constraints_,
        psi_constraints_,
        time);

    theta_constraints_.close();
    psi_constraints_.close();

    // Rebuild combined constraints
    ch_combined_constraints_.clear();

    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_psi = psi_dof_handler_.n_dofs();

    for (unsigned int i = 0; i < n_theta; ++i)
    {
        if (theta_constraints_.is_constrained(i))
        {
            const auto coupled_i = theta_to_ch_map_[i];
            const auto* entries = theta_constraints_.get_constraint_entries(i);
            const double inhom = theta_constraints_.get_inhomogeneity(i);

            ch_combined_constraints_.add_line(coupled_i);
            if (entries != nullptr && !entries->empty())
            {
                for (const auto& entry : *entries)
                    ch_combined_constraints_.add_entry(coupled_i,
                                                       theta_to_ch_map_[entry.first], entry.second);
            }
            ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
        }
    }

    for (unsigned int i = 0; i < n_psi; ++i)
    {
        if (psi_constraints_.is_constrained(i))
        {
            const auto coupled_i = psi_to_ch_map_[i];
            const auto* entries = psi_constraints_.get_constraint_entries(i);
            const double inhom = psi_constraints_.get_inhomogeneity(i);

            ch_combined_constraints_.add_line(coupled_i);
            if (entries != nullptr && !entries->empty())
            {
                for (const auto& entry : *entries)
                    ch_combined_constraints_.add_entry(coupled_i,
                                                       psi_to_ch_map_[entry.first], entry.second);
            }
            ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
        }
    }

    ch_combined_constraints_.close();
}


// ============================================================================
// compute_mass() - Total mass integral of theta
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::compute_mass() const
{
    dealii::QGauss<dim> quadrature(fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_Q2_, quadrature,
                                    dealii::update_values | dealii::update_JxW_values);

    std::vector<double> theta_values(quadrature.size());
    double mass = 0.0;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution_, theta_values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
            mass += theta_values[q] * fe_values.JxW(q);
    }

    return mass;
}

// ============================================================================
// compute_ch_energy() - Cahn-Hilliard free energy
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::compute_ch_energy() const
{


    dealii::QGauss<dim> quadrature(fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_Q2_, quadrature,
                                    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    std::vector<double> theta_values(quadrature.size());
    std::vector<dealii::Tensor<1, dim>> theta_gradients(quadrature.size());
    double energy = 0.0;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution_, theta_values);
        fe_values.get_function_gradients(theta_solution_, theta_gradients);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double theta = theta_values[q];
            const double grad_theta_sq = theta_gradients[q] * theta_gradients[q];

            const double E_grad = 0.5 * mobility * epsilon * grad_theta_sq;
            const double W = 0.25 * (theta * theta - 1.0) * (theta * theta - 1.0);
            const double E_bulk = (mobility / epsilon) * W;

            energy += (E_grad + E_bulk) * fe_values.JxW(q);
        }
    }

    return energy;
}

// ============================================================================
// compute_kinetic_energy() - Kinetic energy (1/2) integral |u|^2
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::compute_kinetic_energy() const
{
    if (!params_.enable_ns)
        return 0.0;

    dealii::QGauss<dim> quadrature(fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values_ux(fe_Q2_, quadrature,
                                       dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_uy(fe_Q2_, quadrature,
                                       dealii::update_values);

    std::vector<double> ux_values(quadrature.size());
    std::vector<double> uy_values(quadrature.size());
    double energy = 0.0;

    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();

    for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++uy_cell)
    {
        fe_values_ux.reinit(ux_cell);
        fe_values_uy.reinit(uy_cell);

        fe_values_ux.get_function_values(ux_solution_, ux_values);
        fe_values_uy.get_function_values(uy_solution_, uy_values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double u_sq = ux_values[q] * ux_values[q] +
                uy_values[q] * uy_values[q];
            energy += 0.5 * u_sq * fe_values_ux.JxW(q);
        }
    }

    return energy;
}

// ============================================================================
// compute_magnetic_energy() - Magnetic energy (mu_0/2) integral |H|^2
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::compute_magnetic_energy() const
{
    if (!params_.enable_magnetic)
        return 0.0;



    dealii::QGauss<dim> quadrature(fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_Q2_, quadrature,
                                    dealii::update_gradients | dealii::update_JxW_values);

    std::vector<dealii::Tensor<1, dim>> phi_gradients(quadrature.size());
    double energy = 0.0;

    for (const auto& cell : phi_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(phi_solution_, phi_gradients);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double H_sq = phi_gradients[q] * phi_gradients[q];
            energy += 0.5 * mu_0 * H_sq * fe_values.JxW(q);
        }
    }

    return energy;
}

// ============================================================================
// output_results() - Write VTK output
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::output_results(unsigned int step) const
{
    // Theta
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(theta_dof_handler_);
        data_out.add_data_vector(theta_solution_, "theta");
        data_out.build_patches();

        std::string filename = params_.output.folder + "/theta_" + std::to_string(step) + ".vtk";
        std::ofstream output(filename);
        data_out.write_vtk(output);
    }

    // Psi
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(psi_dof_handler_);
        data_out.add_data_vector(psi_solution_, "psi");
        data_out.build_patches();

        std::string filename = params_.output.folder + "/psi_" + std::to_string(step) + ".vtk";
        std::ofstream output(filename);
        data_out.write_vtk(output);
    }

    // Phi (if magnetic)
    if (params_.enable_magnetic)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(phi_dof_handler_);
        data_out.add_data_vector(phi_solution_, "phi");
        data_out.build_patches();

        std::string filename = params_.output.folder + "/phi_" + std::to_string(step) + ".vtk";
        std::ofstream output(filename);
        data_out.write_vtk(output);

        // Mx, My (if DG transport)
        if (params_.use_dg_transport)
        {
            {
                dealii::DataOut<dim> data_out_mx;
                data_out_mx.attach_dof_handler(mx_dof_handler_);
                data_out_mx.add_data_vector(mx_solution_, "Mx",
                                            dealii::DataOut<dim>::type_dof_data);
                data_out_mx.build_patches();

                std::string fn = params_.output.folder + "/mx_" + std::to_string(step) + ".vtk";
                std::ofstream out(fn);
                data_out_mx.write_vtk(out);
            }
            {
                dealii::DataOut<dim> data_out_my;
                data_out_my.attach_dof_handler(my_dof_handler_);
                data_out_my.add_data_vector(my_solution_, "My",
                                            dealii::DataOut<dim>::type_dof_data);
                data_out_my.build_patches();

                std::string fn = params_.output.folder + "/my_" + std::to_string(step) + ".vtk";
                std::ofstream out(fn);
                data_out_my.write_vtk(out);
            }
        }
    }

    // Velocity and pressure (if NS)
    if (params_.enable_ns)
    {
        {
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler(ux_dof_handler_);
            data_out.add_data_vector(ux_solution_, "ux");
            data_out.build_patches();

            std::string filename = params_.output.folder + "/ux_" + std::to_string(step) + ".vtk";
            std::ofstream output(filename);
            data_out.write_vtk(output);
        }
        {
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler(uy_dof_handler_);
            data_out.add_data_vector(uy_solution_, "uy");
            data_out.build_patches();

            std::string filename = params_.output.folder + "/uy_" + std::to_string(step) + ".vtk";
            std::ofstream output(filename);
            data_out.write_vtk(output);
        }
        {
            dealii::DataOut<dim> data_out;
            data_out.attach_dof_handler(p_dof_handler_);
            data_out.add_data_vector(p_solution_, "pressure");
            data_out.build_patches();

            std::string filename = params_.output.folder + "/pressure_" + std::to_string(step) + ".vtk";
            std::ofstream output(filename);
            data_out.write_vtk(output);
        }
    }
}

// ============================================================================
// compute_mms_errors()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::compute_mms_errors() const
{
    auto errors = compute_ch_mms_errors<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_solution_,
        psi_solution_,
        time_);

    errors.print();

    std::cout << "\nConvergence data (tab-separated for plotting):\n";
    std::cout << "h\ttheta_L2\ttheta_H1\tpsi_L2\n";
    errors.print_for_convergence();
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;
