// ============================================================================
// core/nsch_problem.cc - Main implementation for NS/CH/Poisson ferrofluid solver
//
// REFACTORED VERSION: Separate DoFHandlers for each scalar field
// Enables proper SolutionTransfer during AMR (fixes deal.II 9.7 issue)
//
// Based on: Nochetto, Salgado & Tomas (2016)
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#include "core/nsch_problem.h"

#include <deal.II/base/multithread_info.h>

#include "diagnostics/nsch_verification.h"
#include "output/nsch_output.h"
#include "diagnostics/interface_tracker.h"
#include "utilities/nsch_adaptive_dt.h"
#include "diagnostics/nsch_diagnostics.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <limits>
#include <chrono>
#include <sstream>
#include <ctime>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
NSCHProblem<dim>::NSCHProblem(const NSCHParameters& params)
    : params_(params)
    , fe_Q2_(params.fe_degree_velocity)  // Q2 for velocity, c, mu, phi
    , fe_Q1_(params.fe_degree_pressure)  // Q1 for pressure
    , c_dof_handler_(triangulation_)
    , mu_dof_handler_(triangulation_)
    , ux_dof_handler_(triangulation_)
    , uy_dof_handler_(triangulation_)
    , p_dof_handler_(triangulation_)
    , phi_dof_handler_(triangulation_)
    , time_(0.0)
    , timestep_number_(0)
{}

// ============================================================================
// Get minimum mesh size
// ============================================================================
template <int dim>
double NSCHProblem<dim>::get_h() const
{
    double h_min = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation_.active_cell_iterators())
        h_min = std::min(h_min, cell->diameter());
    return h_min;
}

// ============================================================================
// Single time step (staggered approach)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::do_time_step(double dt)
{
    if (params_.mms_mode)
        update_mms_boundary_conditions(time_ + dt);

    // Save old solutions
    c_old_solution_ = c_solution_;
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;

    // Staggered scheme:
    // 1. Solve CH with current velocity -> new c, mu
    // 2. Solve Poisson with new c -> new φ (if magnetic enabled)
    // 3. Solve NS with new c (and φ if magnetic) -> new u, p

    solve_cahn_hilliard(dt);
    solve_poisson();
    solve_navier_stokes(dt);

    // Optional Picard iteration for nonlinear coupling
    if (params_.use_picard)
    {
        for (unsigned int iter = 0; iter < params_.picard_max_iter; ++iter)
        {
            // Store previous iteration
            dealii::Vector<double> c_prev = c_solution_;
            dealii::Vector<double> ux_prev = ux_solution_;
            dealii::Vector<double> uy_prev = uy_solution_;

            solve_cahn_hilliard(dt);
            solve_poisson();
            solve_navier_stokes(dt);

            // Compute differences
            c_prev -= c_solution_;
            ux_prev -= ux_solution_;
            uy_prev -= uy_solution_;

            double c_diff = c_prev.l2_norm();
            double u_diff = std::sqrt(ux_prev.l2_norm() * ux_prev.l2_norm() +
                                      uy_prev.l2_norm() * uy_prev.l2_norm());

            if (params_.verbose && iter > 0)
            {
                std::cout << "    Picard iter " << iter
                          << ": |delta_c| = " << c_diff
                          << ", |delta_u| = " << u_diff << "\n";
            }

            if (c_diff < params_.picard_tol && u_diff < params_.picard_tol)
                break;
        }
    }

    time_ += dt;
    ++timestep_number_;
}

// ============================================================================
// Helper: Fill StepMetrics from NSCHVerificationMetrics
// ============================================================================
namespace {
    StepMetrics fill_step_metrics(unsigned int step, double time,
                                  const NSCHVerificationMetrics& metrics)
    {
        StepMetrics m;
        m.step = step;
        m.time = time;
        m.mass = metrics.mass;
        m.total_energy = metrics.total_energy;
        m.kinetic_energy = metrics.kinetic_energy;
        m.ch_energy = metrics.ch_energy;
        m.divergence_L2 = metrics.divergence_L2;
        m.c_min = metrics.c_min;
        m.c_max = metrics.c_max;
        m.u_max = metrics.u_max;
        m.cfl_number = metrics.cfl_number;
        // Additional fields if available in your metrics struct:
        // m.mu_min = metrics.mu_min;
        // m.mu_max = metrics.mu_max;
        // m.p_min = metrics.p_min;
        // m.p_max = metrics.p_max;
        return m;
    }
}

// ============================================================================
// Main run method
// ============================================================================
template <int dim>
void NSCHProblem<dim>::run()
{
    // Start wall clock timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize output handler FIRST - it creates ../ReSuLtS/run_TIMESTAMP/
    // This folder is in PROJECT ROOT (not cmake-build-debug)
    NSCHOutput output;
    output.initialize("../ReSuLtS");
    output.set_intervals(10, 1);  // Console every 10 steps, CSV every step

    // Update params_.output_dir so VTK files go to the same folder
    const_cast<NSCHParameters&>(params_).output_dir = output.get_run_dir();
    std::cout << "[INFO] Output directory: " << params_.output_dir << "\n";

    // Print system/parallel info
    std::cout << "\n[System Info]\n";
    std::cout << "  deal.II version: " << DEAL_II_VERSION_MAJOR << "."
              << DEAL_II_VERSION_MINOR << "." << DEAL_II_VERSION_SUBMINOR << "\n";
#ifdef DEAL_II_WITH_MPI
    std::cout << "  MPI: enabled\n";
#else
    std::cout << "  MPI: not available\n";
#endif
#ifdef DEAL_II_WITH_THREADS
    std::cout << "  Threading: enabled (TBB)\n";
    std::cout << "  Max threads: " << dealii::MultithreadInfo::n_threads() << "\n";
#else
    std::cout << "  Threading: not available\n";
#endif
#ifdef DEAL_II_WITH_UMFPACK
    std::cout << "  UMFPACK: available (direct solver)\n";
#endif

    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "  Coupled Navier-Stokes / Cahn-Hilliard Solver\n";
    std::cout << "  [REFACTORED: Separate DoFHandlers for AMR]\n";
    if (params_.enable_magnetic)
        std::cout << "  (WITH MAGNETOSTATICS - Ferrofluid Model)\n";
    if (params_.use_adaptive_dt)
        std::cout << "  (Staggered approach with ADAPTIVE TIME-STEPPING)\n";
    else
        std::cout << "  (Staggered approach with FIXED TIME-STEPPING)\n";
    std::cout << "============================================================\n\n";

    // Print physical parameters
    std::cout << "[Physical Parameters]\n";
    std::cout << "  nu_water     : " << params_.nu_water << "\n";
    std::cout << "  nu_ferro     : " << params_.nu_ferro << "\n";
    std::cout << "  Epsilon eps  : " << params_.epsilon << "\n";
    std::cout << "  Lambda lam   : " << params_.lambda << "\n";
    std::cout << "  Mobility M   : " << params_.mobility << "\n";
    if (params_.enable_magnetic)
    {
        std::cout << "  chi_m (susc) : " << params_.chi_m << "\n";
        std::cout << "\n[Dipole Configuration] (Nochetto Section 6.2)\n";
        std::cout << "  Dipole intensity : " << params_.dipole_intensity << "\n";
        std::cout << "  Dipole ramp time : " << params_.dipole_ramp_time << "\n";
        std::cout << "  Dipole y-position: " << params_.dipole_y_position << "\n";
        std::cout << "  Dipole positions : (-0.5, 0, 0.5, 1, 1.5) at y="
                  << params_.dipole_y_position << "\n";
        std::cout << "  Dipole direction : (0, 1) upward\n";
    }

    if (params_.enable_gravity)
    {
        std::cout << "\n[Gravity] ENABLED\n";
        std::cout << "  g magnitude  : " << params_.gravity << "\n";
        std::cout << "  g angle      : " << params_.gravity_angle << " deg (-90 = down)\n";
    }
    std::cout << "\n";

    std::cout << "[Domain] [" << params_.x_min << ", " << params_.x_max << "] x ["
              << params_.y_min << ", " << params_.y_max << "]\n";

    // Initial condition info
    std::cout << "\n[Initial Condition] ";
    if (params_.ic_type == 0)
        std::cout << "Circular droplet (centered)\n";
    else if (params_.ic_type == 1)
        std::cout << "Rosensweig flat layer (no perturbation)\n";
    else if (params_.ic_type == 2)
    {
        std::cout << "Rosensweig layer with perturbation\n";
        std::cout << "  Layer height  : " << params_.rosensweig_layer_height << " (fraction)\n";
        std::cout << "  Perturbation  : " << params_.rosensweig_perturbation << "\n";
        std::cout << "  Pert. modes   : " << params_.rosensweig_perturbation_modes << "\n";
    }

    std::cout << "\n[Time] dt = " << params_.dt
              << ", t_final = " << params_.t_final << "\n";

    if (params_.use_amr)
    {
        std::cout << "\n[Adaptive Mesh Refinement] ENABLED\n";
        std::cout << "  Interval     : every " << params_.amr_interval << " steps\n";
        std::cout << "  Level range  : [" << params_.amr_min_level << ", " << params_.amr_max_level << "]\n";
        std::cout << "  Refine frac  : " << params_.amr_refine_fraction << "\n";
        std::cout << "  Coarsen frac : " << params_.amr_coarsen_fraction << "\n";
    }

    std::cout << "\n";

    // Setup
    std::cout << "--- Setup Phase ---\n";
    setup_mesh();
    setup_all_systems();
    initialize_all();

    double h_min = get_h();
    std::cout << "[INFO] Mesh size h_min = " << h_min << "\n";
    std::cout << "[INFO] Total DoFs: " << get_total_dofs() << "\n";

    // Adaptive time-stepping
    AdaptiveTimeStep adaptive_dt(params_.dt, 1e-8, 1e-3);
    adaptive_dt.cfl_target = 0.3;
    adaptive_dt.cfl_max = 0.8;
    adaptive_dt.energy_tol = 1e-4;

    output_results(0);

    double dt = params_.dt;

    // Compute initial metrics using adapter to pack solutions
    NSCHVerificationMetrics metrics = compute_nsch_metrics_from_scalars(
        ux_dof_handler_, ux_solution_, uy_solution_,
        c_dof_handler_, c_solution_, mu_solution_,
        params_.lambda, dt, h_min, 0.0);

    double old_total_energy = metrics.total_energy;

    if (params_.use_adaptive_dt)
        adaptive_dt.initialize(metrics.total_energy);

    // Time loop
    std::cout << "\n--- Time Integration ---\n";
    output.record(fill_step_metrics(0, time_, metrics));

    unsigned int output_counter = 0;

    while (time_ < params_.t_final - 1e-12)
    {
        if (params_.use_adaptive_dt)
            dt = adaptive_dt.get_dt();
        else
            dt = params_.dt;

        if (time_ + dt > params_.t_final)
            dt = params_.t_final - time_;

        do_time_step(dt);

        // Adaptive mesh refinement
        if (params_.use_amr && timestep_number_ % params_.amr_interval == 0)
        {
            refine_mesh();
            h_min = get_h();
        }

        metrics = compute_nsch_metrics_from_scalars(
            ux_dof_handler_, ux_solution_, uy_solution_,
            c_dof_handler_, c_solution_, mu_solution_,
            params_.lambda, dt, h_min, old_total_energy);

        if (params_.use_adaptive_dt)
            dt = adaptive_dt.compute_new_dt(metrics, h_min);

        old_total_energy = metrics.total_energy;

        if (timestep_number_ % params_.output_interval == 0 ||
            time_ >= params_.t_final - 1e-12)
        {
            output_counter++;
            output.record(fill_step_metrics(timestep_number_, time_, metrics));
            output_results(output_counter);

            // Interface tracking (spike detection)
            InterfaceProfile profile = compute_interface_profile_from_scalar(
                c_dof_handler_, c_solution_, params_.fe_degree_phase);
            std::cout << "       Interface: amp=" << std::fixed << std::setprecision(4)
                      << profile.amplitude << "  y=[" << profile.y_min << "," << profile.y_max
                      << "]  spikes=" << profile.spikes.size() << "\n";

            if (!check_nsch_health(metrics))
            {
                std::cerr << "\n[ERROR] Solution health check failed! Aborting.\n";
                break;
            }
        }
    }

    // Finalize output (prints summary, closes CSV)
    output.finalize();

    if (params_.use_adaptive_dt)
        adaptive_dt.print_summary();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\n[INFO] Total wall time: " << duration.count() << " seconds\n";
    std::cout << "[INFO] Output saved to: " << params_.output_dir << "\n";
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class NSCHProblem<2>;