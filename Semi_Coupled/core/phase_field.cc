// ============================================================================
// core/phase_field.cc - Time Stepping and Solve Methods (PARALLEL)
//
// Time stepping algorithm:
//   Block-Gauss-Seidel global iteration (outer loop):
//     1. Solve CH: (θ, ψ) using current U
//     2. Solve Magnetics: monolithic (M, φ) block system (no Picard)
//     3. Solve NS: (U, P) using updated θ, M, H
//     Repeat 1-3 until all fields converge.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "assembly/ch_assembler.h"
#include "assembly/magnetic_assembler.h"
#include "assembly/ns_assembler.h"
#include "solvers/ch_solver.h"
#include "solvers/magnetic_solver.h"
#include "solvers/ns_solver.h"

// Utilities - MUST come before diagnostics/loggers that use tools.h functions
#include "utilities/tools.h"
#include "utilities/run_tracker.h"

// Diagnostics and logging system (these use functions from tools.h)
#include "diagnostics/system_diagnostics.h"
#include "diagnostics/magnetic_diagnostics.h"
#include "diagnostics/validation_diagnostics.h"
#include "output/console_logger.h"
#include "output/metrics_logger.h"
#include "output/timing_logger.h"
#include "output/parallel_diagnostics_logger.h"
#include "diagnostics/parallel_data.h"
#include "utilities/sparsity_export.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

// ============================================================================
// run() - Main time-stepping loop with integrated logging
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::run()
{
    // ========================================================================
    // SETUP PHASE
    // ========================================================================
    pcout_ << "========================================\n";
    pcout_ << "  Ferrofluid Phase Field Solver\n";
    pcout_ << "  (Monolithic Magnetics)\n";
    pcout_ << "========================================\n\n";

    pcout_ << "[1/5] Setting up mesh...\n";
    setup_mesh();

    pcout_ << "[2/5] Setting up DoF handlers...\n";
    setup_dof_handlers();

    pcout_ << "[3/5] Setting up CH system...\n";
    setup_ch_system();

    if (params_.enable_magnetic)
    {
        pcout_ << "[3/5] Setting up Magnetic system (monolithic M+φ)...\n";
        setup_magnetic_system();
    }

    if (params_.enable_ns)
    {
        pcout_ << "[3/5] Setting up NS system...\n";
        setup_ns_system();
    }

    pcout_ << "[4/5] Initializing solutions...\n";
    initialize_solutions();

    // ========================================================================
    // CREATE OUTPUT DIRECTORY WITH TIMESTAMP
    // ========================================================================
    const std::string output_dir = timestamped_folder(params_.output.folder, params_);
    if (MPIUtils::is_root(mpi_communicator_))
        ensure_directory(output_dir);
    MPI_Barrier(mpi_communicator_);

    // Write run configuration
    if (MPIUtils::is_root(mpi_communicator_))
        write_run_info(output_dir, params_, MPIUtils::size(mpi_communicator_));

    // ========================================================================
    // INITIALIZE LOGGERS
    // ========================================================================
    RunTracker tracker;
    tracker.start(output_dir, mpi_communicator_);

    ConsoleLogger console(params_, mpi_communicator_);
    MetricsLogger metrics(output_dir, params_, mpi_communicator_);
    TimingLogger timing(output_dir, params_, mpi_communicator_);

    // Validation logger for Rosensweig wavelength tracking (CSV only)
    ValidationLogger validation_logger;
    validation_logger.open(output_dir + "/validation_metrics.csv", mpi_communicator_);

    // Optional: Magnetization/field distribution logger
    MagnetizationLogger mag_logger;

    // Parallel diagnostics logger (optional, --parallel-diag flag)
    std::unique_ptr<ParallelDiagnosticsLogger> parallel_diag;
    if (params_.enable_parallel_diagnostics)
    {
        parallel_diag = std::make_unique<ParallelDiagnosticsLogger>(
            output_dir, params_, mpi_communicator_,
            params_.parallel_diag_all_ranks);
    }

    // ========================================================================
    // SPARSITY PATTERN EXPORT (optional, --dump-sparsity flag)
    // Exports SVG + gnuplot + bandwidth CSV for each matrix at step 0
    // ========================================================================
    if (params_.dump_sparsity)
    {
        pcout_ << "\n[Sparsity Export] Dumping sparsity patterns...\n";
        pcout_ << "  Cuthill-McKee renumbering: "
               << (params_.renumber_dofs ? "ON" : "OFF") << "\n";

        std::vector<SparsityAnalysis> all_analyses;

        // CH matrix
        {
            auto a = analyze_sparsity(ch_matrix_, "CH");
            unsigned int global_bw = 0;
            MPI_Reduce(&a.bandwidth, &global_bw, 1, MPI_UNSIGNED, MPI_MAX, 0, mpi_communicator_);
            if (MPIUtils::is_root(mpi_communicator_))
                a.bandwidth = global_bw;
            all_analyses.push_back(a);
            export_sparsity_pattern(ch_matrix_, "ch", output_dir, mpi_communicator_, pcout_);
        }

        // Monolithic magnetics matrix (M+phi)
        if (params_.enable_magnetic)
        {
            auto a = analyze_sparsity(mag_matrix_, "Magnetic(M+φ)");
            unsigned int global_bw = 0;
            MPI_Reduce(&a.bandwidth, &global_bw, 1, MPI_UNSIGNED, MPI_MAX, 0, mpi_communicator_);
            if (MPIUtils::is_root(mpi_communicator_))
                a.bandwidth = global_bw;
            all_analyses.push_back(a);
            export_sparsity_pattern(mag_matrix_, "magnetic", output_dir, mpi_communicator_, pcout_);
        }

        // NS matrix (need to assemble first — it's assembled each step)
        // Skip: NS matrix is zero at setup. It will be dumped after first step solve.

        // Write summary
        if (MPIUtils::is_root(mpi_communicator_))
            write_sparsity_summary(all_analyses, output_dir, params_.renumber_dofs, pcout_);

        pcout_ << "[Sparsity Export] Done.\n\n";
    }

    // ========================================================================
    // PRINT ROSENSWEIG VALIDATION REFERENCE (one-time at startup)
    // ========================================================================
    if (params_.enable_magnetic)
    {
        // Effective surface tension for phase-field: σ_eff = λ/ε
        // (not λ alone — the capillary force is (λ/ε)θ∇ψ)
        double sigma_eff = params_.physics.lambda / params_.physics.epsilon;
        double rho_g_eff = params_.physics.r * params_.physics.gravity;
        double lambda_theory = RosensweigTheory::critical_wavelength(
            sigma_eff, 1.0, rho_g_eff);
        pcout_ << "\n[Rosensweig Validation Reference]\n";
        pcout_ << "  Effective surface tension σ = λ/ε = " << sigma_eff << "\n";
        pcout_ << "  Critical wavelength λ_c = " << lambda_theory << "\n";
        pcout_ << "  Expected spikes in [0,1]: ~" << 1.0/lambda_theory << "\n";
        pcout_ << "  Magnetics: monolithic (no Picard)\n\n";
    }

    // ========================================================================
    // PRINT CONSOLE HEADER
    // ========================================================================
    pcout_ << "[5/5] Starting time stepping...\n";
    console.print_header();

    // Initialize previous step data for energy rate computation
    StepData prev_data;

    // Time stepping parameters
    const double dt = params_.time.dt;
    const double t_final = params_.time.t_final;
    const unsigned int output_frequency = params_.output.frequency;

    // ========================================================================
    // INITIAL OUTPUT (step 0)
    // ========================================================================
    {
        StepData data = compute_system_diagnostics<dim>(
            theta_dof_handler_, theta_relevant_,
            params_.enable_magnetic ? &phi_dof_handler_ : nullptr,
            params_.enable_magnetic ? &phi_relevant_ : nullptr,
            params_.enable_ns ? &ux_dof_handler_ : nullptr,
            params_.enable_ns ? &uy_dof_handler_ : nullptr,
            params_.enable_ns ? &p_dof_handler_ : nullptr,
            params_.enable_ns ? &ux_relevant_ : nullptr,
            params_.enable_ns ? &uy_relevant_ : nullptr,
            params_.enable_ns ? &p_relevant_ : nullptr,
            params_, 0, time_, dt, get_min_h(),
            prev_data, mpi_communicator_);

        // Add force diagnostics if we have psi
        update_force_diagnostics<dim>(data,
                                      theta_dof_handler_, theta_relevant_, psi_relevant_,
                                      params_.enable_magnetic ? &phi_relevant_ : nullptr,
                                      params_, time_, mpi_communicator_);

        // Add validation diagnostics (CSV logging only - no console output)
        if (params_.enable_magnetic)
        {
            update_validation_diagnostics<dim>(data,
                                               theta_dof_handler_, theta_relevant_,
                                               params_, mpi_communicator_);
            validation_logger.log(data);
        }

        update_timing_info(data, 0.0, 0.0);
        update_mesh_info(data, triangulation_.n_global_active_cells(),
                         theta_dof_handler_.n_dofs());

        // Set initial interface position for delta tracking
        console.set_initial_interface(data.interface_y_max);

        // Log initial state
        console.print_step(data);
        metrics.log_step(data);

        if (params_.enable_magnetic)
            mag_logger.write(compute_field_distribution(params_, time_, 0));

        // Output VTK
        output_results(output_dir);

        prev_data = data;
    }

    ++timestep_number_;

    // ========================================================================
    // MAIN TIME LOOP
    // ========================================================================
    while (time_ < t_final - 1e-12 && timestep_number_ <= params_.time.max_steps)
    {
        // Adaptive mesh refinement (before solving on this step)
        double amr_time_this_step = 0.0;
        if (params_.mesh.use_amr && timestep_number_ > 0 &&
            timestep_number_ % params_.mesh.amr_interval == 0)
        {
            CumulativeTimer t_amr;
            t_amr.start();
            refine_mesh();
            t_amr.stop();
            amr_time_this_step = t_amr.last();
        }

        time_ += dt;

        // Start step timer
        CumulativeTimer step_timer;
        step_timer.start();

        // Timing for each subsystem
        StepTiming step_timing;

        // ====================================================================
        // SOLVE SUBSYSTEMS: Block-Gauss-Seidel global iteration
        // Paper (CMAME 2016, Section 6, p.520):
        //   [CH] -> [Mag+Poisson (Picard)] -> [NS], repeat until convergence
        // ====================================================================
        {
            const unsigned int max_bgs = params_.bgs_max_iterations;
            unsigned int bgs_iter = 0;
            double bgs_residual = 1.0;

            // Storage for previous Block-GS iteration (for convergence check)
            // We check convergence on theta and velocity (the fields that couple blocks)
            dealii::TrilinosWrappers::MPI::Vector theta_bgs_prev;
            dealii::TrilinosWrappers::MPI::Vector ux_bgs_prev;
            dealii::TrilinosWrappers::MPI::Vector uy_bgs_prev;
            theta_bgs_prev.reinit(theta_locally_owned_, mpi_communicator_);
            if (params_.enable_ns)
            {
                ux_bgs_prev.reinit(ux_locally_owned_, mpi_communicator_);
                uy_bgs_prev.reinit(uy_locally_owned_, mpi_communicator_);
            }

            for (bgs_iter = 0; bgs_iter < max_bgs; ++bgs_iter)
            {
                // Store current fields for convergence check
                if (bgs_iter > 0)
                {
                    theta_bgs_prev = theta_solution_;
                    if (params_.enable_ns)
                    {
                        ux_bgs_prev = ux_solution_;
                        uy_bgs_prev = uy_solution_;
                    }
                }

                // Block 1: Solve CH (θ, ψ) using current U
                {
                    CumulativeTimer t;
                    t.start();
                    solve_ch(dt);
                    t.stop();
                    step_timing.ch_time += t.last();
                }

                // Block 2: Solve Monolithic Magnetics (M + φ) — no Picard
                if (params_.enable_magnetic)
                {
                    CumulativeTimer t;
                    t.start();
                    solve_magnetics(dt);
                    t.stop();
                    step_timing.mag_time += t.last();
                }

                // Block 3: Solve NS (U, P) using updated θ, M, H
                if (params_.enable_ns)
                {
                    CumulativeTimer t;
                    t.start();
                    solve_ns(dt);
                    t.stop();
                    step_timing.ns_time += t.last();
                }

                // ============================================================
                // Check Block-GS convergence (skip on first iteration)
                // ============================================================
                if (max_bgs == 1)
                    break;  // Single pass mode

                if (bgs_iter == 0)
                {
                    // First iteration: store fields, can't check convergence yet
                    theta_bgs_prev = theta_solution_;
                    if (params_.enable_ns)
                    {
                        ux_bgs_prev = ux_solution_;
                        uy_bgs_prev = uy_solution_;
                    }
                    continue;
                }

                // Compute relative change: max over all fields
                // ||f^{i} - f^{i-1}|| / ||f^{i}||
                double max_rel_change = 0.0;

                // Check theta
                {
                    dealii::TrilinosWrappers::MPI::Vector diff(theta_locally_owned_, mpi_communicator_);
                    diff = theta_solution_;
                    diff -= theta_bgs_prev;
                    double diff_norm = diff.l2_norm();
                    double sol_norm = theta_solution_.l2_norm();
                    double rel = (sol_norm > 1e-12) ? diff_norm / sol_norm : 0.0;
                    max_rel_change = std::max(max_rel_change, rel);
                }

                // Check velocity
                if (params_.enable_ns)
                {
                    dealii::TrilinosWrappers::MPI::Vector diff_ux(ux_locally_owned_, mpi_communicator_);
                    dealii::TrilinosWrappers::MPI::Vector diff_uy(uy_locally_owned_, mpi_communicator_);
                    diff_ux = ux_solution_;
                    diff_ux -= ux_bgs_prev;
                    diff_uy = uy_solution_;
                    diff_uy -= uy_bgs_prev;

                    double diff_norm_sq = diff_ux.l2_norm() * diff_ux.l2_norm() +
                                          diff_uy.l2_norm() * diff_uy.l2_norm();
                    double sol_norm_sq = ux_solution_.l2_norm() * ux_solution_.l2_norm() +
                                         uy_solution_.l2_norm() * uy_solution_.l2_norm();
                    double rel = (sol_norm_sq > 1e-12) ? std::sqrt(diff_norm_sq / sol_norm_sq) : 0.0;
                    max_rel_change = std::max(max_rel_change, rel);
                }

                bgs_residual = max_rel_change;

                if (params_.output.verbose)
                {
                    pcout_ << "    [BGS " << bgs_iter + 1 << "/" << max_bgs
                           << "] residual = " << std::scientific << bgs_residual << "\n";
                }

                if (bgs_residual < params_.bgs_tolerance)
                    break;  // Converged
            }

            // bgs_iter == max_bgs when loop exhausts (not via break)
            last_bgs_iterations_ = (bgs_iter < max_bgs) ? bgs_iter + 1 : max_bgs;
            last_bgs_residual_ = bgs_residual;
        }

        // ====================================================================
        // Update old solutions for next time step
        // ====================================================================
        theta_old_solution_ = theta_solution_;
        theta_old_relevant_ = theta_relevant_;

        if (params_.enable_ns)
        {
            ux_old_solution_ = ux_solution_;
            uy_old_solution_ = uy_solution_;
        }

        // Stop step timer
        step_timer.stop();
        step_timing.step_total = step_timer.last();

        // ====================================================================
        // COMPUTE DIAGNOSTICS (timed for parallel diagnostics)
        // Controlled by --diagnostics_frequency: 0=off, 1=every step, N=every N steps
        // ====================================================================
        const bool run_diagnostics =
            params_.diagnostics_frequency > 0 &&
            timestep_number_ % params_.diagnostics_frequency == 0;

        CumulativeTimer t_diagnostics;
        StepData data;

        if (run_diagnostics)
        {
        t_diagnostics.start();

        data = compute_system_diagnostics<dim>(
            theta_dof_handler_, theta_relevant_,
            params_.enable_magnetic ? &phi_dof_handler_ : nullptr,
            params_.enable_magnetic ? &phi_relevant_ : nullptr,
            params_.enable_ns ? &ux_dof_handler_ : nullptr,
            params_.enable_ns ? &uy_dof_handler_ : nullptr,
            params_.enable_ns ? &p_dof_handler_ : nullptr,
            params_.enable_ns ? &ux_relevant_ : nullptr,
            params_.enable_ns ? &uy_relevant_ : nullptr,
            params_.enable_ns ? &p_relevant_ : nullptr,
            params_, timestep_number_, time_, dt, get_min_h(),
            prev_data, mpi_communicator_);

        // Add force diagnostics
        update_force_diagnostics<dim>(data,
                                      theta_dof_handler_, theta_relevant_, psi_relevant_,
                                      params_.enable_magnetic ? &phi_relevant_ : nullptr,
                                      params_, time_, mpi_communicator_);

        // Add validation diagnostics (CSV logging only - NO console output in loop)
        if (params_.enable_magnetic)
        {
            update_validation_diagnostics<dim>(data,
                                               theta_dof_handler_, theta_relevant_,
                                               params_, mpi_communicator_);
            validation_logger.log(data);
        }
        // Add solver info
        update_ch_solver_info(data, last_ch_info_.iterations,
                              last_ch_info_.residual, step_timing.ch_time,
                              !last_ch_info_.converged);
        if (params_.enable_magnetic)
        {
            update_magnetic_solver_info(data, last_mag_info_.iterations,
                                       last_mag_info_.residual, step_timing.mag_time);
        }
        if (params_.enable_ns)
            update_ns_solver_info(data, last_ns_info_.iterations, 0,
                                  last_ns_info_.residual, step_timing.ns_time,
                                  !last_ns_info_.converged);

        // Add Block-GS info
        data.bgs_iterations = last_bgs_iterations_;
        data.bgs_residual = last_bgs_residual_;

        // Add timing and mesh info
        update_timing_info(data, step_timing.step_total, tracker.elapsed_seconds());
        update_mesh_info(data, triangulation_.n_global_active_cells(),
                         theta_dof_handler_.n_dofs());

        t_diagnostics.stop();

        // ====================================================================
        // LOGGING
        // ====================================================================

        // CSV logging (only when diagnostics were computed)
        metrics.log_step(data);
        timing.log_step(timestep_number_, time_, step_timing);
        } // end if (run_diagnostics)

        // ====================================================================
        // PARALLEL DIAGNOSTICS (optional, --parallel-diag flag)
        // Only logged when diagnostics were computed this step
        // ====================================================================
        if (run_diagnostics && parallel_diag)
        {
            ParallelStepData pdata;
            pdata.step = timestep_number_;
            pdata.time = time_;

            // --- Timing breakdown (from instrumented solve methods) ---
            pdata.ch_assemble_time = last_ch_assemble_time_;
            pdata.ch_solve_time = last_ch_solve_time_;
            pdata.poisson_assemble_time = 0.0;  // Monolithic: no separate Poisson
            pdata.poisson_solve_time = 0.0;
            pdata.mag_time = last_mag_assemble_time_ + last_mag_solve_time_;
            pdata.ns_assemble_time = last_ns_assemble_time_;
            pdata.ns_solve_time = last_ns_solve_time_;
            pdata.diagnostics_time = t_diagnostics.last();
            pdata.amr_time = amr_time_this_step;
            pdata.step_total = step_timing.step_total;

            // --- BGS iteration counts ---
            pdata.bgs_iterations = last_bgs_iterations_;

            // --- Solver iterations ---
            pdata.ch_solver_iters = last_ch_info_.iterations;
            pdata.poisson_solver_iters = 0;  // Monolithic: no separate Poisson
            pdata.mag_solver_iters = last_mag_info_.iterations;
            pdata.ns_solver_iters = last_ns_info_.iterations;

            // --- Sparsity metrics (local nnz on this rank) ---
            pdata.ch_nnz = ch_matrix_.n_nonzero_elements();
            if (params_.enable_magnetic)
            {
                pdata.poisson_nnz = 0;  // Monolithic: no separate Poisson matrix
                pdata.mag_nnz = mag_matrix_.n_nonzero_elements();
            }
            if (params_.enable_ns)
                pdata.ns_nnz = ns_matrix_.n_nonzero_elements();

            // --- Bandwidth (only computed at step 1; cached for subsequent steps) ---
            // Computing bandwidth requires iterating all entries — expensive every step
            {
                static unsigned int cached_ch_bw = 0, cached_poi_bw = 0;
                static unsigned int cached_mag_bw = 0, cached_ns_bw = 0;
                static bool bandwidth_computed = false;

                if (!bandwidth_computed)
                {
                    auto compute_bandwidth = [](const dealii::TrilinosWrappers::SparseMatrix& mat)
                        -> unsigned int
                    {
                        unsigned int max_bw = 0;
                        const auto range = mat.local_range();
                        for (unsigned int i = range.first; i < range.second; ++i)
                        {
                            for (auto entry = mat.begin(i); entry != mat.end(i); ++entry)
                            {
                                const unsigned int j = entry->column();
                                const unsigned int dist = (i > j) ? (i - j) : (j - i);
                                if (dist > max_bw) max_bw = dist;
                            }
                        }
                        return max_bw;
                    };

                    cached_ch_bw = compute_bandwidth(ch_matrix_);
                    if (params_.enable_magnetic)
                        cached_mag_bw = compute_bandwidth(mag_matrix_);
                    if (params_.enable_ns && ns_matrix_.m() > 0
                        && ns_matrix_.n_nonzero_elements() > 0)
                        cached_ns_bw = compute_bandwidth(ns_matrix_);

                    bandwidth_computed = true;
                }

                pdata.ch_bandwidth = cached_ch_bw;
                pdata.poisson_bandwidth = cached_poi_bw;
                pdata.mag_bandwidth = cached_mag_bw;
                pdata.ns_bandwidth = cached_ns_bw;
            }

            // --- Load balance (per-rank) ---
            pdata.local_cells = triangulation_.n_locally_owned_active_cells();
            // Ghost cells = total active cells on this partition - locally owned
            pdata.ghost_cells = 0; // Not easily available; p4est ghost count is internal
            pdata.local_dofs_theta = theta_locally_owned_.n_elements();
            pdata.local_dofs_phi = params_.enable_magnetic ?
                mag_locally_owned_.n_elements() : 0;  // Combined M+phi
            pdata.local_dofs_M = 0;  // Included in mag
            pdata.local_dofs_ns = params_.enable_ns ?
                ns_locally_owned_.n_elements() : 0;
            pdata.total_local_dofs = pdata.local_dofs_theta + pdata.local_dofs_phi
                                   + pdata.local_dofs_M + pdata.local_dofs_ns;
            pdata.global_cells = triangulation_.n_global_active_cells();
            pdata.global_dofs = theta_dof_handler_.n_dofs()
                + (params_.enable_magnetic ? mag_dof_handler_.n_dofs() : 0)
                + (params_.enable_ns ?
                    (ux_dof_handler_.n_dofs() + uy_dof_handler_.n_dofs()
                     + p_dof_handler_.n_dofs()) : 0);

            // --- AMR levels ---
            pdata.amr_min_level = triangulation_.n_global_levels() > 0 ? 0 : 0;
            pdata.amr_max_level = triangulation_.n_global_levels() - 1;

            // --- Memory ---
            pdata.memory_mb = ParallelDiagnosticsLogger::get_memory_usage_mb();

            // --- MPI reductions (computes imbalance ratios) ---
            pdata.compute_mpi_reductions(mpi_communicator_);

            // --- Log ---
            parallel_diag->log_step(pdata);
        }

        // Sparsity export of NS matrix (only after step 1, when NS matrix is assembled)
        if (params_.dump_sparsity && timestep_number_ == 1 && params_.enable_ns)
        {
            pcout_ << "[Sparsity Export] Dumping NS matrix (after step 1)...\n";
            export_sparsity_pattern(ns_matrix_, "ns", output_dir, mpi_communicator_, pcout_);

            // Append NS to summary
            auto ns_analysis = analyze_sparsity(ns_matrix_, "NS");
            unsigned int global_bw = 0;
            MPI_Reduce(&ns_analysis.bandwidth, &global_bw, 1, MPI_UNSIGNED, MPI_MAX, 0, mpi_communicator_);
            if (MPIUtils::is_root(mpi_communicator_))
            {
                ns_analysis.bandwidth = global_bw;

                // Append to summary file
                std::string path = output_dir + "/sparsity_summary.csv";
                std::ofstream f(path, std::ios::app);
                if (f.is_open())
                {
                    f << ns_analysis.name << ","
                      << ns_analysis.n_rows << "," << ns_analysis.n_cols << ","
                      << ns_analysis.total_nnz << ","
                      << ns_analysis.bandwidth << ","
                      << std::fixed << std::setprecision(2) << ns_analysis.avg_bandwidth << ","
                      << ns_analysis.min_nnz_per_row << "," << ns_analysis.max_nnz_per_row << ","
                      << std::setprecision(2) << ns_analysis.avg_nnz_per_row << ","
                      << std::setprecision(2) << ns_analysis.std_nnz_per_row << ","
                      << std::scientific << std::setprecision(4) << ns_analysis.density << "\n";
                    f.close();
                }
            }
            pcout_ << "[Sparsity Export] NS matrix done.\n";
        }

        // Console output (every N steps, only if diagnostics were computed)
        if (run_diagnostics && timestep_number_ % output_frequency == 0)
        {
            console.print_step(data);

            // Field distribution (for magnetic runs)
            if (params_.enable_magnetic)
                mag_logger.write(compute_field_distribution(params_, time_, timestep_number_));
        }

        // Warnings and notes (only if diagnostics were computed)
        if (run_diagnostics)
        {
            console.print_warnings(data);
            console.print_notes(data);
        }

        // VTK output (every N steps)
        if (timestep_number_ % output_frequency == 0)
        {
            CumulativeTimer t;
            t.start();
            output_results(output_dir);
            t.stop();
            step_timing.output_time = t.last();
        }

        // ====================================================================
        // TERMINATION CHECKS
        // ====================================================================

        // Termination and safety checks (only when diagnostics were computed)
        if (run_diagnostics)
        {
            // Check for NaN
            if (std::isnan(data.E_total) || std::isnan(data.mass))
            {
                console.error("NaN detected in solution!");
                tracker.end("error: NaN detected");
                console.print_footer("error: NaN detected", data);
                validation_logger.close();
                return;
            }

            // Check for severe bounds violation
            if (data.theta_min < -1.5 || data.theta_max > 1.5)
            {
                console.warning("Severe theta bounds violation - simulation may be unstable");
            }

            // Store for next iteration (energy rate computation)
            prev_data = data;
        }

        ++timestep_number_;
    }

    // ========================================================================
    // FINAL OUTPUT (only if last step wasn't already output)
    // ========================================================================
    if ((timestep_number_ - 1) % output_frequency != 0)
        output_results(output_dir);

    // ========================================================================
    // FINAL ROSENSWEIG VALIDATION SUMMARY (printed ONCE at end)
    // ========================================================================
    if (params_.enable_magnetic)
    {
        InterfaceProfile final_profile = analyze_interface<dim>(
            theta_dof_handler_, theta_relevant_, params_, mpi_communicator_);

        RosensweigValidation validation = validate_rosensweig(final_profile, params_);

        if (MPIUtils::is_root(mpi_communicator_))
        {
            validation.print_summary(std::cout);
            validation.write_csv(output_dir + "/rosensweig_validation.csv");
            write_interface_profile_csv(final_profile,
                                        output_dir + "/interface_profile_final.csv", mpi_communicator_);
        }
    }

    // Close validation logger
    validation_logger.close();

    // Determine termination reason
    const std::string reason = (timestep_number_ > params_.time.max_steps)
        ? "max_steps" : "complete";
    tracker.end(reason);
    console.print_footer(reason, prev_data);
}

// ============================================================================
// solve_magnetics() - Monolithic M+φ block system (no Picard)
//
// Single assemble + solve for the combined (M, φ) system.
// Replaces the old Picard iteration between Poisson and Magnetization.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_magnetics(double dt)
{
    CumulativeTimer t_assemble, t_solve;

    // Update ghosted theta (from CH solve) — but use θ^{k-1} for χ(θ)
    // per paper Theorem 4.1: material coefficients at OLD theta for stability
    theta_relevant_ = theta_solution_;

    // Update ghosted M_old for transport time derivative (Eq 42c)
    mag_old_relevant_ = mag_solution_old_;

    // Assemble monolithic system (M transport + Poisson, Eq 42c-42d)
    // NOTE: χ(θ) must use θ^{k-1} (theta_old_relevant_) per paper Eq 42c
    t_assemble.start();
    magnetic_assembler_->assemble(
        mag_matrix_,
        mag_rhs_,
        ux_relevant_,
        uy_relevant_,
        theta_old_relevant_,  // θ^{k-1} for χ(θ) — paper Theorem 4.1
        mag_old_relevant_,  // M^{k-1} for time derivative in Eq 42c
        dt,
        time_);
    t_assemble.stop();

    // Solve monolithic system.
    // The solver dispatches on params_.solvers.magnetic.use_iterative:
    //   - false: MUMPS direct cascade
    //   - true:  GMRES + cached MagneticBlockPreconditioner
    //           (ILU on M block, AMG on phi block, extracted via Trilinos)
    // setup_magnetic_system() (called after AMR) recreates magnetic_solver_,
    // which drops the cached preconditioner — so rebuild is implicit on AMR.
    //
    // Compute n_M_dofs (split point between M and phi in component_wise layout).
    const auto dofs_per_component =
        dealii::DoFTools::count_dofs_per_fe_component(mag_dof_handler_);
    dealii::types::global_dof_index n_M_dofs = 0;
    for (unsigned int d = 0; d < dim; ++d)
        n_M_dofs += dofs_per_component[d];

    // Plan A: explicit cache-rebuild policy with adaptive staleness trigger.
    // Rebuild on AMR (flag set in refine_mesh) or when last iter count
    // exceeded the staleness threshold. Otherwise reuse cached ILU/AMG.
    const bool rebuild_mag_prec = needs_mag_preconditioner_rebuild_;

    t_solve.start();
    magnetic_solver_->solve(
        mag_matrix_,
        mag_solution_,
        mag_rhs_,
        params_.solvers.magnetic,
        n_M_dofs,
        rebuild_mag_prec);
    t_solve.stop();

    // Apply constraints
    mag_constraints_.distribute(mag_solution_);

    // Record timing
    last_mag_assemble_time_ = t_assemble.last();
    last_mag_solve_time_ = t_solve.last();
    last_mag_info_.iterations = magnetic_solver_->last_n_iterations();
    last_mag_info_.residual   = magnetic_solver_->last_residual();
    last_mag_info_.converged  = true;

    // Plan A — adaptive rebuild policy.
    // After a successful solve, the cache is fresh (whether we just rebuilt
    // it or reused). Default: don't rebuild next call. Override if iter
    // count exceeded the staleness threshold (cached preconditioner is
    // drifting from matrix values), or if direct fallback fired (cache
    // got reset inside the solver).
    if (params_.solvers.magnetic.use_iterative
        && !magnetic_solver_->last_used_direct())
    {
        needs_mag_preconditioner_rebuild_ =
            (magnetic_solver_->last_n_iterations() > mag_iter_rebuild_threshold_);
    }
    else
    {
        // Direct path: no cached preconditioner state to track; ensure the
        // first iterative call after a switch still rebuilds.
        needs_mag_preconditioner_rebuild_ = true;
    }

    // Save current solution as M_old for next time step (Eq 42c δM^k/τ)
    mag_solution_old_ = mag_solution_;

    // Update ghosted mag solution
    mag_relevant_ = mag_solution_;

    // Extract individual Mx, My, phi to auxiliary vectors
    // (needed for NS assembly, output, diagnostics)
    extract_magnetic_components();

    if (params_.output.verbose)
    {
        pcout_ << "  [Mag] monolithic solve, iterations="
               << last_mag_info_.iterations << "\n";
    }
}

// ============================================================================
// solve_ch() - Cahn-Hilliard solver
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ch(double dt)
{
    CumulativeTimer t_assemble, t_solve;

    // Assemble CH system
    t_assemble.start();
    assemble_ch_system<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_relevant_,   // θ^{n-1}
        ux_dof_handler_,
        uy_dof_handler_,
        params_.enable_ns ? ux_relevant_ : theta_relevant_,   // u_x^{n-1} (dummy if no NS)
        params_.enable_ns ? uy_relevant_ : theta_relevant_,   // u_y^{n-1} (dummy if no NS)
        params_,
        dt,
        time_,
        theta_to_ch_map_,
        psi_to_ch_map_,
        ch_constraints_,
        ch_matrix_,
        ch_rhs_);
    t_assemble.stop();

    // Solve and extract θ, ψ
    t_solve.start();
    last_ch_info_ = solve_ch_system(
        ch_matrix_,
        ch_rhs_,
        ch_constraints_,
        ch_locally_owned_,
        ch_locally_relevant_,
        theta_locally_owned_,
        psi_locally_owned_,
        theta_to_ch_map_,
        psi_to_ch_map_,
        theta_solution_,
        psi_solution_,
        params_.solvers.ch,
        mpi_communicator_,
        params_.output.verbose);
    t_solve.stop();

    // Record assembly/solve split for parallel diagnostics
    last_ch_assemble_time_ = t_assemble.last();
    last_ch_solve_time_ = t_solve.last();

    // Update ghosted vectors
    theta_relevant_ = theta_solution_;
    psi_relevant_ = psi_solution_;

    if (params_.output.verbose)
    {
        pcout_ << "  [CH] iterations=" << last_ch_info_.iterations
               << ", residual=" << std::scientific << last_ch_info_.residual << "\n";
    }
}

// (Old solve_poisson, solve_magnetization, project_equilibrium_magnetization
//  have been replaced by the monolithic solve_magnetics() above)

// ============================================================================
// solve_ns() - Navier-Stokes solver with Kelvin force
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ns(double dt)
{
    CumulativeTimer t_assemble, t_solve;

    // Compute spatially-varying viscosity: ν(θ) = ν_w + (ν_f - ν_w) * H(θ)
    const double nu = 0.5 * (params_.physics.nu_water + params_.physics.nu_ferro);

    const bool include_time = true;
    const bool include_convection = true;

    // Assemble with magnetic forces if enabled
    t_assemble.start();
    if (params_.enable_magnetic)
    {
        assemble_ns_system_ferrofluid_parallel<dim>(
            ux_dof_handler_,
            uy_dof_handler_,
            p_dof_handler_,
            ux_relevant_,
            uy_relevant_,
            nu,
            dt,
            include_time,
            include_convection,
            ux_to_ns_map_,
            uy_to_ns_map_,
            p_to_ns_map_,
            ns_locally_owned_,
            ns_constraints_,
            ns_matrix_,
            ns_rhs_,
            mpi_communicator_,
            // Kelvin force inputs
            phi_dof_handler_,
            M_dof_handler_,
            phi_relevant_,
            Mx_relevant_,
            My_relevant_,
            params_.physics.mu_0,
            // Capillary force inputs
            theta_dof_handler_,
            psi_dof_handler_,
            theta_old_relevant_,  // θ^{k-1} — material coefficients lagged per Theorem 4.1
            psi_relevant_,
            params_.physics.lambda,
            params_.physics.epsilon,
            // Variable viscosity
            params_.physics.nu_water,
            params_.physics.nu_ferro,
            // Gravity
            params_.enable_gravity,
            params_.physics.r,
            params_.physics.gravity,
            params_.gravity_direction,
            // Simulation parameters
            params_,
            time_,
            // MMS options
            params_.enable_mms,
            time_,
            time_ - dt,
            params_.domain.y_max - params_.domain.y_min);
    }
    else
    {
        // Basic NS assembly without magnetic forces
        assemble_ns_system_parallel<dim>(
            ux_dof_handler_,
            uy_dof_handler_,
            p_dof_handler_,
            ux_relevant_,
            uy_relevant_,
            nu,
            dt,
            include_time,
            include_convection,
            ux_to_ns_map_,
            uy_to_ns_map_,
            p_to_ns_map_,
            ns_locally_owned_,
            ns_constraints_,
            ns_matrix_,
            ns_rhs_,
            mpi_communicator_,
            // MMS options
            params_.enable_mms,
            time_,
            time_ - dt,
            params_.domain.y_max - params_.domain.y_min);
    }
    t_assemble.stop();

    // Solve NS system - select direct or iterative based on --direct flag
    t_solve.start();
    if (!params_.solvers.ns.use_iterative)
    {
        // Direct solver with pressure pinning (fast for refinement <= 4)
        last_ns_info_ = solve_ns_system_direct_parallel(
            ns_matrix_,
            ns_rhs_,
            ns_solution_,
            ns_constraints_,
            p_to_ns_map_,
            ns_locally_owned_,
            mpi_communicator_,
            params_.output.verbose);
    }
    else
    {
        // Iterative Block Schur solver
        last_ns_info_ = solve_ns_system_schur_parallel(
            ns_matrix_,
            ns_rhs_,
            ns_solution_,
            ns_constraints_,
            pressure_mass_matrix_,
            ux_to_ns_map_,
            uy_to_ns_map_,
            p_to_ns_map_,
            ns_locally_owned_,
            ux_locally_owned_,
            p_locally_owned_,
            mpi_communicator_,
            nu,
            dt,
            params_.output.verbose);
    }

    // Extract individual solutions
    extract_ns_solutions_parallel(
        ns_solution_,
        ux_to_ns_map_,
        uy_to_ns_map_,
        p_to_ns_map_,
        ux_locally_owned_,
        uy_locally_owned_,
        p_locally_owned_,
        ns_locally_owned_,
        ns_locally_relevant_,
        ux_solution_,
        uy_solution_,
        p_solution_,
        mpi_communicator_);
    t_solve.stop();

    // Record assembly/solve split for parallel diagnostics
    last_ns_assemble_time_ = t_assemble.last();
    last_ns_solve_time_ = t_solve.last();

    // Update ghosted vectors
    ux_relevant_ = ux_solution_;
    uy_relevant_ = uy_solution_;
    p_relevant_ = p_solution_;

    if (params_.output.verbose)
    {
        pcout_ << "  [NS] iterations=" << last_ns_info_.iterations
            << ", residual=" << std::scientific << last_ns_info_.residual << "\n";
    }
}

// ============================================================================
// output_results() - Unified VTU output (all fields in single file)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::output_results(const std::string& output_dir)
{
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(theta_dof_handler_);

    // Phase field (θ, ψ) — primary DoFHandler
    dealii::TrilinosWrappers::MPI::Vector theta_out(
        theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
    theta_out = theta_solution_;
    data_out.add_data_vector(theta_out, "theta");

    dealii::TrilinosWrappers::MPI::Vector psi_out(
        psi_locally_owned_, psi_locally_relevant_, mpi_communicator_);
    psi_out = psi_solution_;
    data_out.add_data_vector(psi_out, "psi");

    // Velocity (CG Q2, same FE as θ)
    dealii::TrilinosWrappers::MPI::Vector ux_out(
        ux_locally_owned_, ux_locally_relevant_, mpi_communicator_);
    dealii::TrilinosWrappers::MPI::Vector uy_out(
        uy_locally_owned_, uy_locally_relevant_, mpi_communicator_);
    if (params_.enable_ns)
    {
        ux_out = ux_solution_;
        uy_out = uy_solution_;
        data_out.add_data_vector(ux_dof_handler_, ux_out, "ux");
        data_out.add_data_vector(ux_dof_handler_, uy_out, "uy");

        // Pressure (DG P1, different DoFHandler)
        dealii::TrilinosWrappers::MPI::Vector p_out(
            p_locally_owned_, p_locally_relevant_, mpi_communicator_);
        p_out = p_solution_;
        data_out.add_data_vector(p_dof_handler_, p_out, "p");
    }

    // Magnetic potential (CG Q2)
    dealii::TrilinosWrappers::MPI::Vector phi_out(
        phi_locally_owned_, phi_locally_relevant_, mpi_communicator_);
    if (params_.enable_magnetic)
    {
        phi_out = phi_solution_;
        data_out.add_data_vector(phi_dof_handler_, phi_out, "phi");
    }

    // Magnetization (DG Q2)
    dealii::TrilinosWrappers::MPI::Vector Mx_out(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    dealii::TrilinosWrappers::MPI::Vector My_out(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    if (params_.enable_magnetic)
    {
        Mx_out = Mx_solution_;
        My_out = My_solution_;
        data_out.add_data_vector(M_dof_handler_, Mx_out, "Mx");
        data_out.add_data_vector(M_dof_handler_, My_out, "My");
    }

    // Cell-averaged derived quantities
    const unsigned int n_cells = triangulation_.n_active_cells();
    dealii::Vector<float> U_mag_cell(n_cells);
    dealii::Vector<float> M_mag_cell(n_cells);
    dealii::Vector<float> H_mag_cell(n_cells);
    dealii::Vector<float> H_x_cell(n_cells);
    dealii::Vector<float> H_y_cell(n_cells);

    {
        const dealii::QMidpoint<dim> q_mid;

        if (params_.enable_ns)
        {
            dealii::FEValues<dim> fe_vel(ux_dof_handler_.get_fe(),
                                          q_mid, dealii::update_values);
            unsigned int idx = 0;
            for (const auto& cell : ux_dof_handler_.active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_vel.reinit(cell);
                    std::vector<double> ux_val(1), uy_val(1);
                    fe_vel.get_function_values(ux_out, ux_val);
                    fe_vel.get_function_values(uy_out, uy_val);
                    U_mag_cell[idx] = static_cast<float>(
                        std::sqrt(ux_val[0]*ux_val[0] + uy_val[0]*uy_val[0]));
                }
                ++idx;
            }
            data_out.add_data_vector(U_mag_cell, "U_mag",
                                     dealii::DataOut<dim>::type_cell_data);
        }

        if (params_.enable_magnetic)
        {
            dealii::FEValues<dim> fe_mag(M_dof_handler_.get_fe(),
                                          q_mid, dealii::update_values);
            unsigned int idx = 0;
            for (const auto& cell : M_dof_handler_.active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_mag.reinit(cell);
                    std::vector<double> mx_val(1), my_val(1);
                    fe_mag.get_function_values(Mx_out, mx_val);
                    fe_mag.get_function_values(My_out, my_val);
                    M_mag_cell[idx] = static_cast<float>(
                        std::sqrt(mx_val[0]*mx_val[0] + my_val[0]*my_val[0]));
                }
                ++idx;
            }
            data_out.add_data_vector(M_mag_cell, "M_mag",
                                     dealii::DataOut<dim>::type_cell_data);
        }

        if (params_.enable_magnetic)
        {
            dealii::FEValues<dim> fe_phi(phi_dof_handler_.get_fe(),
                                          q_mid, dealii::update_gradients);
            unsigned int idx = 0;
            for (const auto& cell : phi_dof_handler_.active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_phi.reinit(cell);
                    std::vector<dealii::Tensor<1, dim>> grad_phi(1);
                    fe_phi.get_function_gradients(phi_out, grad_phi);
                    H_x_cell[idx] = static_cast<float>(grad_phi[0][0]);
                    H_y_cell[idx] = static_cast<float>(grad_phi[0][1]);
                    H_mag_cell[idx] = static_cast<float>(grad_phi[0].norm());
                }
                ++idx;
            }
            data_out.add_data_vector(H_x_cell, "H_x",
                                     dealii::DataOut<dim>::type_cell_data);
            data_out.add_data_vector(H_y_cell, "H_y",
                                     dealii::DataOut<dim>::type_cell_data);
            data_out.add_data_vector(H_mag_cell, "H_mag",
                                     dealii::DataOut<dim>::type_cell_data);
        }
    }

    // Subdomain for parallel visualization
    dealii::Vector<float> subdomain(n_cells);
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain",
                             dealii::DataOut<dim>::type_cell_data);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "solution_",
        timestep_number_, mpi_communicator_);

    if (params_.output.verbose)
        pcout_ << "  [Output] Wrote step " << timestep_number_ << "\n";
}

// ============================================================================
// get_min_h() - Minimum cell diameter for CFL computation
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::get_min_h() const
{
    return dealii::GridTools::minimal_cell_diameter(triangulation_);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;