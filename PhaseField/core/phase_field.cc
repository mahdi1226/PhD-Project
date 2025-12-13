// ============================================================================
// core/phase_field.cc - Phase Field Problem Main Implementation
//
// Contains constructor, run(), do_time_step(), output_results()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "assembly/ch_assembler.h"
#include "assembly/poisson_assembler.h"
#include "solvers/ch_solver.h"
#include "solvers/poisson_solver.h"
#include "utilities/tools.h"
#include "diagnostics/ch_mms.h"

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
// Constructor
// ============================================================================
template <int dim>
PhaseFieldProblem<dim>::PhaseFieldProblem(const Parameters& params)
    : params_(params)
    , fe_phase_(params.fe.degree_phase)
    , theta_dof_handler_(triangulation_)
    , psi_dof_handler_(triangulation_)
    , phi_dof_handler_(triangulation_)
    , time_(0.0)
    , timestep_number_(0)
{
}

// ============================================================================
// run() - Main entry point
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::run()
{
    const bool mms_mode = params_.mms.enabled;

    std::cout << "============================================================\n";
    if (mms_mode)
        std::cout << "  Cahn-Hilliard Solver - MMS VERIFICATION MODE\n";
    else
        std::cout << "  Cahn-Hilliard Solver (Standalone Test)\n";
    std::cout << "  Reference: Nochetto et al. CMAME 309 (2016)\n";
    std::cout << "============================================================\n\n";

    // Create output directory with timestamp
    std::string output_dir = timestamped_folder(params_.output.folder);
    std::filesystem::create_directories(output_dir);
    std::cout << "[Info] Output directory: " << output_dir << "\n\n";

    // Store output directory for later use
    // Note: we cast away const because output_dir is runtime state
    const_cast<Parameters&>(params_).output.folder = output_dir;

    // Setup
    std::cout << "--- Setup Phase ---\n";
    setup_mesh();
    setup_dof_handlers();
    setup_constraints();  // Will apply MMS Dirichlet BCs if enabled
    setup_ch_system();
    if (params_.magnetic.enabled)
        setup_poisson_system();
    initialize_solutions();  // Will use MMS ICs if enabled

    const double h_min = get_min_h();
    std::cout << "[Info] Mesh h_min = " << h_min << "\n";

    // Print parameters
    std::cout << "\n--- Parameters ---\n";
    std::cout << "  epsilon = " << params_.ch.epsilon << "\n";
    std::cout << "  gamma   = " << params_.ch.gamma << "\n";
    std::cout << "  dt      = " << params_.time.dt << "\n";
    std::cout << "  t_final = " << params_.time.t_final << "\n";
    if (params_.magnetic.enabled)
    {
        std::cout << "  Magnetic: ENABLED\n";
        std::cout << "  χ₀ (susceptibility) = " << params_.magnetization.chi_0 << "\n";
        std::cout << "  Dipole intensity    = " << params_.dipoles.intensity_max << "\n";
        std::cout << "  Dipole ramp time    = " << params_.dipoles.ramp_time << "\n";
    }
    if (mms_mode)
    {
        std::cout << "  MMS mode: ENABLED (Dirichlet BCs)\n";
        std::cout << "  t_init  = " << params_.mms.t_init << "\n";
    }
    else
    {
        std::cout << "  IC type = " << params_.ic.type << "\n";
    }

    // Initial output
    if (params_.magnetic.enabled)
    {
        // Solve Poisson once at t=0 to populate φ for initial output
        solve_poisson();
    }
    output_results(0);

    // Time loop
    std::cout << "\n--- Time Integration ---\n";
    std::cout << std::setw(8) << "Step"
              << std::setw(12) << "Time"
              << std::setw(14) << "Mass"
              << std::setw(12) << "θ_min"
              << std::setw(12) << "θ_max" << "\n";
    std::cout << std::string(58, '-') << "\n";

    const double dt = params_.time.dt;
    unsigned int output_counter = 0;

    while (time_ < params_.time.t_final - 1e-12)
    {
        double current_dt = dt;
        if (time_ + current_dt > params_.time.t_final)
            current_dt = params_.time.t_final - time_;

        do_time_step(current_dt);

        // Output periodically
        if (timestep_number_ % params_.output.frequency == 0 ||
            time_ >= params_.time.t_final - 1e-12)
        {
            ++output_counter;
            output_results(output_counter);

            // Compute diagnostics
            const double mass = compute_mass();
            double theta_min = theta_solution_.linfty_norm();
            double theta_max = -theta_min;
            for (unsigned int i = 0; i < theta_solution_.size(); ++i)
            {
                theta_min = std::min(theta_min, theta_solution_[i]);
                theta_max = std::max(theta_max, theta_solution_[i]);
            }

            std::cout << std::setw(8) << timestep_number_
                      << std::setw(12) << std::fixed << std::setprecision(4) << time_
                      << std::setw(14) << std::scientific << std::setprecision(6) << mass
                      << std::setw(12) << std::fixed << std::setprecision(4) << theta_min
                      << std::setw(12) << theta_max << "\n";
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

    std::cout << "[Info] Output saved to: " << output_dir << "\n";
}

// ============================================================================
// do_time_step() - Single time step
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::do_time_step(double dt)
{
    // Save old solution
    theta_old_solution_ = theta_solution_;

    // Time at end of this step (for MMS source terms and BCs)
    const double time_new = time_ + dt;

    // For MMS: update BCs at new time before assembly
    if (params_.mms.enabled)
    {
        update_mms_boundary_constraints(time_new);
    }

    // Assemble CH system
    assemble_ch_system<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_old_solution_,
        ux_dummy_,  // Zero velocity for standalone CH
        uy_dummy_,
        params_,
        dt,
        time_new,  // Time for MMS source terms
        theta_to_ch_map_,
        psi_to_ch_map_,
        ch_matrix_,
        ch_rhs_);

    // Apply combined constraints (critical for AMR!)
    ch_combined_constraints_.condense(ch_matrix_, ch_rhs_);

    // Solve CH
    solve_ch_system(
        ch_matrix_,
        ch_rhs_,
        ch_combined_constraints_,
        theta_to_ch_map_,
        psi_to_ch_map_,
        theta_solution_,
        psi_solution_);

    // Apply individual constraints for consistency
    theta_constraints_.distribute(theta_solution_);
    psi_constraints_.distribute(psi_solution_);

    // Solve Poisson (if magnetic enabled)
    // θ changed → μ(θ) changed → solve for new φ
    if (params_.magnetic.enabled)
    {
        solve_poisson();
    }

    // Update time
    time_ += dt;
    ++timestep_number_;
}

// ============================================================================
// output_results() - VTK output
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::output_results(unsigned int step) const
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(theta_dof_handler_);

    data_out.add_data_vector(theta_solution_, "theta");
    data_out.add_data_vector(psi_solution_, "psi");

    // Add φ if magnetic enabled (same mesh/FE as θ)
    if (params_.magnetic.enabled)
    {
        data_out.add_data_vector(phi_solution_, "phi");

        // Diagnostic: print φ range
        double phi_min = *std::min_element(phi_solution_.begin(), phi_solution_.end());
        double phi_max = *std::max_element(phi_solution_.begin(), phi_solution_.end());
        std::cout << "[Poisson] φ range: [" << phi_min << ", " << phi_max << "]\n";
    }

    data_out.build_patches(params_.fe.degree_phase);

    std::string filename = params_.output.folder + "/solution_" +
                           dealii::Utilities::int_to_string(step, 4) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);
}

// ============================================================================
// compute_mass() - Integral of θ over domain
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::compute_mass() const
{
    dealii::QGauss<dim> quadrature(params_.fe.degree_phase + 1);
    dealii::FEValues<dim> fe_values(fe_phase_, quadrature,
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
// get_min_h() - Minimum cell diameter
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::get_min_h() const
{
    double h_min = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation_.active_cell_iterators())
        h_min = std::min(h_min, cell->diameter());
    return h_min;
}

// ============================================================================
// compute_mms_errors() - Compute and print MMS errors
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::compute_mms_errors() const
{
    CHMMSErrors errors = compute_ch_mms_errors<dim>(
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
// update_mms_boundary_constraints() - Update Dirichlet BCs for current time
//
// For MMS: boundary values change with time, so constraints must be rebuilt
// each time step. This is more expensive but necessary for MMS verification.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::update_mms_boundary_constraints(double time)
{
    // Clear and rebuild individual constraints
    theta_constraints_.clear();
    psi_constraints_.clear();

    // Hanging node constraints (always needed)
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);

    // MMS Dirichlet BCs at specified time
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

    // Map θ constraints to coupled system
    for (unsigned int i = 0; i < n_theta; ++i)
    {
        if (theta_constraints_.is_constrained(i))
        {
            const auto coupled_i = theta_to_ch_map_[i];
            const auto* entries = theta_constraints_.get_constraint_entries(i);
            const double inhom = theta_constraints_.get_inhomogeneity(i);

            if (entries == nullptr || entries->empty())
            {
                // Constrained to inhomogeneity only (Dirichlet)
                ch_combined_constraints_.add_line(coupled_i);
                ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
            }
            else
            {
                // Hanging node constraint
                ch_combined_constraints_.add_line(coupled_i);
                for (const auto& entry : *entries)
                {
                    const auto coupled_j = theta_to_ch_map_[entry.first];
                    ch_combined_constraints_.add_entry(coupled_i, coupled_j, entry.second);
                }
                ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
            }
        }
    }

    // Map ψ constraints to coupled system
    for (unsigned int i = 0; i < n_psi; ++i)
    {
        if (psi_constraints_.is_constrained(i))
        {
            const auto coupled_i = psi_to_ch_map_[i];
            const auto* entries = psi_constraints_.get_constraint_entries(i);
            const double inhom = psi_constraints_.get_inhomogeneity(i);

            if (entries == nullptr || entries->empty())
            {
                ch_combined_constraints_.add_line(coupled_i);
                ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
            }
            else
            {
                ch_combined_constraints_.add_line(coupled_i);
                for (const auto& entry : *entries)
                {
                    const auto coupled_j = psi_to_ch_map_[entry.first];
                    ch_combined_constraints_.add_entry(coupled_i, coupled_j, entry.second);
                }
                ch_combined_constraints_.set_inhomogeneity(coupled_i, inhom);
            }
        }
    }

    ch_combined_constraints_.close();
}

// ============================================================================
// solve_poisson() - Solve magnetostatic Poisson equation
//
// After CH step updates θ, we solve -∇·(μ(θ)∇φ) = 0 with dipole BCs.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_poisson()
{
    // Update BCs for current time (dipole field may be ramping)
    update_poisson_constraints(time_);

    // Assemble system
    assemble_poisson_system<dim>(
        phi_dof_handler_,
        theta_dof_handler_,
        theta_solution_,
        params_,
        phi_matrix_,
        phi_rhs_,
        phi_constraints_);

    // Solve
    solve_poisson_system(
        phi_matrix_,
        phi_rhs_,
        phi_solution_,
        phi_constraints_);
}

// ============================================================================
// update_poisson_constraints() - Update Dirichlet BCs for current time
//
// Dipole field ramps up over time, so BCs change.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::update_poisson_constraints(double time)
{
    phi_constraints_.clear();

    // Hanging node constraints
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);

    // Dirichlet BCs from dipole field
    apply_poisson_dirichlet_bcs<dim>(
        phi_dof_handler_,
        params_,
        time,
        phi_constraints_);

    phi_constraints_.close();
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;