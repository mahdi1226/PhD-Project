// ============================================================================
// core/phase_field.cc - Time Stepping and Solve Methods (PARALLEL)
//
// Time stepping algorithm (per Paper Section 6):
//   1. Solve CH ONCE (uses U^{n-1} from previous timestep)
//   2. Picard iteration for Poisson ↔ Magnetization only:
//      - Solve Poisson (42d): φ depends on M
//      - Solve Magnetization (42c): M evolves with H = ∇φ
//      - Iterate until M converges
//   3. Solve NS ONCE (uses θ^{n-1}, converged M^n, H^n)
//
// The "Block-Gauss-Seidel" refers to the sequential dependency structure,
// NOT that all systems iterate together. Only Poisson ↔ Magnetization
// needs iteration due to their tight coupling.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "assembly/ch_assembler.h"
#include "assembly/poisson_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "assembly/ns_assembler.h"
#include "solvers/ch_solver.h"
#include "solvers/poisson_solver.h"
#include "solvers/magnetization_solver.h"
#include "solvers/ns_solver.h"

// New diagnostics and logging system
#include "diagnostics/system_diagnostics.h"
#include "diagnostics/magnetization_diagnostics.h"
#include "diagnostics/validation_diagnostics.h"
#include "output/console_logger.h"
#include "output/metrics_logger.h"
#include "output/timing_logger.h"
#include "utilities/run_tracker.h"
#include "utilities/tools.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>
#include "solvers/ns_solver.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    pcout_ << "  (Optimized with Picard Iteration)\n";
    pcout_ << "========================================\n\n";

    pcout_ << "[1/5] Setting up mesh...\n";
    setup_mesh();

    pcout_ << "[2/5] Setting up DoF handlers...\n";
    setup_dof_handlers();

    pcout_ << "[3/5] Setting up CH system...\n";
    setup_ch_system();
/*
    if (params_.physics.beta > 0) {
        pcout_ << "[WARNING] Beta is active - potential infinite loop source\n";
        // UNCOMMENT TO TEST: params_.physics.beta = 0.0;
    } */
    // pcout_ << "\n";

    if (params_.enable_magnetic)
    {
        pcout_ << "[3/5] Setting up Poisson system...\n";
        setup_poisson_system();

        if (params_.use_dg_transport)
        {
            pcout_ << "[3/5] Setting up Magnetization system...\n";
            setup_magnetization_system();
        }
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

    // ========================================================================
    // PRINT ROSENSWEIG VALIDATION REFERENCE (one-time at startup)
    // ========================================================================
    if (params_.enable_magnetic)
    {
        double lambda_theory = RosensweigTheory::critical_wavelength(
            params_.physics.lambda,
            1.0 + params_.physics.r,
            params_.physics.gravity);
        pcout_ << "\n[Rosensweig Validation Reference]\n";
        pcout_ << "  Critical wavelength λ_c = " << lambda_theory << "\n";
        pcout_ << "  Picard iterations: " << params_.picard_iterations << "\n";
        pcout_ << "  Picard tolerance: " << params_.picard_tolerance << "\n\n";
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
        // Compute initial diagnostics
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
            params_, mpi_communicator_);

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
    while (time_ < t_final - 1e-12)
    {
        time_ += dt;

        // Start step timer
        CumulativeTimer step_timer;
        step_timer.start();

        // Timing for each subsystem
        StepTiming step_timing;

        // ====================================================================
        // SOLVE SUBSYSTEMS (Sequential with Picard iteration for Mag+Poisson)
        //
        // The paper's "Block-Gauss-Seidel" refers to the SEQUENTIAL DEPENDENCY:
        //   Step 1: Solve CH (uses U^{n-1} from PREVIOUS timestep)
        //   Step 2: Solve Mag+Poisson (uses θ^n from Step 1) - ITERATE HERE
        //   Step 3: Solve NS (uses θ^{n-1}, converged M^n, H^n)
        //
        // Only Poisson ↔ Magnetization needs iteration because they're tightly
        // coupled (φ depends on M, M depends on H=∇φ).
        //
        // CH and NS are already decoupled by the semi-implicit time discretization.
        // ====================================================================


        // Step 1: Solve CH ONCE (uses U^{n-1} from previous timestep)
        {
            CumulativeTimer t;
            t.start();
            solve_ch(dt);
            t.stop();
            step_timing.ch_time = t.last();
        }


        // Step 2: Solve Magnetic subsystem (Picard iteration for Poisson ↔ Mag)
        if (params_.enable_magnetic)
        {
            CumulativeTimer t;
            t.start();

            unsigned int picard_iters = solve_poisson_magnetization_picard(dt);
            last_picard_iterations_ = picard_iters;

            t.stop();
            // Split time between Poisson and Magnetization
            step_timing.poisson_time = t.last() * 0.3;
            step_timing.mag_time = t.last() * 0.7;


        }

        // Step 3: Solve NS ONCE (uses θ^{n-1}, converged M^n, H^n)
        if (params_.enable_ns)
        {

            CumulativeTimer t;
            t.start();
            solve_ns(dt);
            t.stop();
            step_timing.ns_time = t.last();

        }

        // ====================================================================
        // Update old solutions for next time step
        // ====================================================================
        theta_old_solution_ = theta_solution_;
        theta_old_relevant_ = theta_relevant_;

        if (params_.enable_magnetic && params_.use_dg_transport)
        {
            Mx_old_solution_ = Mx_solution_;
            My_old_solution_ = My_solution_;
        }

        if (params_.enable_ns)
        {
            ux_old_solution_ = ux_solution_;
            uy_old_solution_ = uy_solution_;
        }

        // Stop step timer
        step_timer.stop();
        step_timing.step_total = step_timer.last();

        // ====================================================================
        // COMPUTE DIAGNOSTICS
        // ====================================================================
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
            params_, timestep_number_, time_, dt, get_min_h(),
            prev_data, mpi_communicator_);

        // Add force diagnostics
        update_force_diagnostics<dim>(data,
            theta_dof_handler_, theta_relevant_, psi_relevant_,
            params_.enable_magnetic ? &phi_relevant_ : nullptr,
            params_, mpi_communicator_);

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
            update_poisson_solver_info(data, last_poisson_info_.iterations,
                                       last_poisson_info_.residual, step_timing.poisson_time);
            // Add Picard iteration count to magnetization info
            if (params_.use_dg_transport)
                update_mag_solver_info(data, last_M_info_.iterations,
                                       last_M_info_.residual, step_timing.mag_time);
        }
        if (params_.enable_ns)
            update_ns_solver_info(data, last_ns_info_.iterations, 0,
                                  last_ns_info_.residual, step_timing.ns_time,
                                  !last_ns_info_.converged);

        // Add timing and mesh info
        update_timing_info(data, step_timing.step_total, tracker.elapsed_seconds());
        update_mesh_info(data, triangulation_.n_global_active_cells(),
                         theta_dof_handler_.n_dofs());

        // ====================================================================
        // LOGGING
        // ====================================================================

        // CSV logging (every step)
        metrics.log_step(data);
        timing.log_step(timestep_number_, time_, step_timing);

        // Console output (every N steps)
        if (timestep_number_ % output_frequency == 0)
        {
            console.print_step(data);

            // Field distribution (for magnetic runs)
            if (params_.enable_magnetic)
                mag_logger.write(compute_field_distribution(params_, time_, timestep_number_));
        }

        // Always check for warnings
        console.print_warnings(data);

        // Interface/spike notes (only on change)
        console.print_notes(data);

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

        // Store for next iteration
        prev_data = data;

        ++timestep_number_;
    }

    // ========================================================================
    // FINAL OUTPUT
    // ========================================================================
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

    tracker.end("complete");
    console.print_footer("complete", prev_data);
}

// ============================================================================
// solve_poisson_magnetization_picard() - Picard iteration for Poisson ↔ Mag
//
// Only iterates over Poisson and Magnetization because they are tightly coupled:
//   - Poisson (42d): φ depends on M
//   - Magnetization (42c): M evolves with H = ∇φ
//
// CH and NS are solved ONCE per timestep (not inside this loop).
// ============================================================================
template <int dim>
unsigned int PhaseFieldProblem<dim>::solve_poisson_magnetization_picard(double dt)
{

    //DEBUG ONLY:
    pcout_ << "[DEBUG] Entering Picard, time_ = " << std::setprecision(10) << time_ << "\n";

    // Update ghosted theta for magnetization (from CH solve)
    theta_relevant_ = theta_solution_;

    if (!params_.use_dg_transport)
    {
        // No DG transport - just solve Poisson once
        solve_poisson();
        return 1;
    }

    const double picard_tol = params_.picard_tolerance;
    const double omega = 0.35;  // Under-relaxation factor for M
    const unsigned int max_picard = params_.picard_iterations;

    // Store previous M for convergence check
    dealii::TrilinosWrappers::MPI::Vector Mx_prev(M_locally_owned_, mpi_communicator_);
    dealii::TrilinosWrappers::MPI::Vector My_prev(M_locally_owned_, mpi_communicator_);

    unsigned int picard_iter = 0;
    double picard_residual = 1.0;

    for (picard_iter = 0; picard_iter < max_picard; ++picard_iter)
    {
        // Store current M for convergence check
        Mx_prev = Mx_solution_;
        My_prev = My_solution_;

        // ================================================================
        // Solve Poisson: uses current M
        // ================================================================
        Mx_relevant_ = Mx_solution_;
        My_relevant_ = My_solution_;

        assemble_poisson_rhs<dim>(
            phi_dof_handler_, M_dof_handler_,
            Mx_relevant_, My_relevant_,
            params_, time_, phi_constraints_, phi_rhs_);

        //DEBUG ONLY:
        pcout_ << "[DEBUG] dipoles.positions.size() = " << params_.dipoles.positions.size() << "\n";
        pcout_ << "[DEBUG] time = " << time_ << ", ramp_time = " << params_.dipoles.ramp_time << "\n";

        //DEBUG ONLY:
        // Around line 505, add:
        pcout_ << "[DEBUG] phi_rhs L2 norm = " << phi_rhs_.l2_norm() << "\n";

        last_poisson_info_ = poisson_solver_->solve(
            phi_rhs_, phi_solution_, phi_constraints_, false);

        phi_relevant_ = phi_solution_;

        // ================================================================
        // Solve Magnetization: uses just-computed φ
        // ================================================================
        if (picard_iter == 0)
            solve_magnetization(dt, true);  // Full assembly + factorization
        else
            solve_magnetization_rhs_only(dt);  // RHS only, reuse factorization

        // ================================================================
        // Apply under-relaxation to M (helps convergence)
        // M^{k+1} = ω * M_new + (1-ω) * M_old
        //
        // CRITICAL: Apply from iteration 0 to prevent initial overshoot
        // that causes divergence at iteration 2 (seen as 500%+ residual)
        // ================================================================
        if (omega < 1.0)
        {
            Mx_solution_.sadd(omega, 1.0 - omega, Mx_prev);
            My_solution_.sadd(omega, 1.0 - omega, My_prev);
        }

        // ================================================================
        // Check convergence of M
        // ================================================================
        dealii::TrilinosWrappers::MPI::Vector Mx_diff(M_locally_owned_, mpi_communicator_);
        dealii::TrilinosWrappers::MPI::Vector My_diff(M_locally_owned_, mpi_communicator_);
        Mx_diff = Mx_solution_;
        My_diff = My_solution_;
        Mx_diff -= Mx_prev;
        My_diff -= My_prev;

        double M_change = Mx_diff.l2_norm() + My_diff.l2_norm();
        double M_norm = Mx_solution_.l2_norm() + My_solution_.l2_norm() + 1e-14;
        picard_residual = M_change / M_norm;

        if (params_.output.verbose)
        {
            pcout_ << "    [Picard " << picard_iter + 1 << "/" << max_picard
                   << "] ΔM/|M| = " << std::scientific << std::setprecision(2)
                   << picard_residual << std::fixed << "\n";
        }

        // Early exit if converged
        if (picard_residual < picard_tol)
        {
            if (params_.output.verbose)
                pcout_ << "    [Picard] Converged in " << picard_iter + 1 << " iterations\n";
            break;
        }
    }

    // Update ghosted M vectors
    Mx_relevant_ = Mx_solution_;
    My_relevant_ = My_solution_;

    // Store final residual for diagnostics
    last_picard_residual_ = picard_residual;

    // Warn if not converged
    if (picard_iter == max_picard && picard_residual >= picard_tol)
    {
        pcout_ << "  [WARNING] Picard did not converge: residual = "
               << std::scientific << std::setprecision(1) << picard_residual
               << " > tol = " << picard_tol << std::fixed << "\n";
    }

    return picard_iter + 1;
}

// ============================================================================
// time_step() - Single time step (backward compatibility wrapper)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::time_step(double dt)
{
    // Step 1: Solve Cahn-Hilliard
    solve_ch(dt);

    // Steps 2-3: Picard iteration for Poisson ↔ Magnetization
    if (params_.enable_magnetic)
        solve_poisson_magnetization_picard(dt);

    // Step 4: Solve Navier-Stokes (if enabled)
    if (params_.enable_ns)
        solve_ns(dt);

    // Update old solutions for next time step
    theta_old_solution_ = theta_solution_;
    theta_old_relevant_ = theta_relevant_;

    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        Mx_old_solution_ = Mx_solution_;
        My_old_solution_ = My_solution_;
    }

    if (params_.enable_ns)
    {
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;
    }
}

// ============================================================================
// solve_ch() - Cahn-Hilliard solve
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ch(double dt)
{
    // Update ghosted vectors for assembly
    theta_old_relevant_ = theta_old_solution_;
    if (params_.enable_ns)
    {
        ux_relevant_ = ux_solution_;
        uy_relevant_ = uy_solution_;
    }

    // Assemble system
    assemble_ch_system<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_old_relevant_,        // θ^{k-1}
        ux_dof_handler_,            // velocity x DoFHandler
        uy_dof_handler_,            // velocity y DoFHandler
        ux_relevant_,               // U_x (or zero)
        uy_relevant_,               // U_y (or zero)
        params_,
        dt,
        time_,                      // current_time for MMS
        theta_to_ch_map_,
        psi_to_ch_map_,
        ch_constraints_,
        ch_matrix_,
        ch_rhs_);


    // Solve coupled system and extract θ, ψ
    last_ch_info_ = solve_ch_system(
    ch_matrix_, ch_rhs_, ch_constraints_,
    ch_locally_owned_,
    ch_locally_relevant_,   // <-- ADD THIS
    theta_locally_owned_, psi_locally_owned_,
    theta_to_ch_map_, psi_to_ch_map_,
    theta_solution_, psi_solution_,
    params_.solvers.ch, mpi_communicator_,
    params_.output.verbose);

    // Update ghosted vectors
    theta_relevant_ = theta_solution_;
    psi_relevant_ = psi_solution_;

    if (params_.output.verbose)
    {
        pcout_ << "  [CH] iterations=" << last_ch_info_.iterations
               << ", residual=" << std::scientific << last_ch_info_.residual << "\n";
    }
}

// ============================================================================
// solve_poisson() - Magnetostatic potential (Paper Eq. 42d)
//
// Solves: -Δφ = -∇·(h_a - M)
// Weak form: (∇φ, ∇χ) = (h_a - M, ∇χ)
//
// OPTIMIZATION: Matrix is assembled ONCE in setup_poisson_system().
// Only the RHS changes each timestep (depends on h_a ramp and M^k).
// AMG preconditioner is also cached and reused.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_poisson()
{
    // Update ghosted vectors for assembly
    if (params_.use_dg_transport)
    {
        Mx_relevant_ = Mx_solution_;
        My_relevant_ = My_solution_;
    }

    // Assemble RHS ONLY (matrix was assembled once in setup)
    assemble_poisson_rhs<dim>(
        phi_dof_handler_,
        M_dof_handler_,
        Mx_relevant_,
        My_relevant_,
        params_,
        time_,
        phi_constraints_,
        phi_rhs_);

    // Solve using cached Poisson solver (AMG preconditioner reused)
    last_poisson_info_ = poisson_solver_->solve(
        phi_rhs_,
        phi_solution_,
        phi_constraints_,
        params_.output.verbose);

    // Update ghosted vector
    phi_relevant_ = phi_solution_;
}

// ============================================================================
// solve_magnetization() - DG transport for M (Paper Eq. 42c)
//
// Equation 42c (rearranged):
//   (1/τ + 1/τ_M)(M^k, Z) - B_h^m(U^{k-1}, Z, M^k) = (1/τ_M)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// Uses upwind DG for the transport term B_h^m
// Solves TWO scalar systems (Mx and My) with the SAME matrix
//
// OPTIMIZATION: Uses cached assembler and solver
// - matrix_changed=true: full assembly + new factorization
// - matrix_changed=false: RHS only, reuse factorization
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_magnetization(double dt, bool matrix_changed)
{
    // Update ghosted vectors for assembly
    phi_relevant_ = phi_solution_;
    theta_relevant_ = theta_solution_;

    if (params_.enable_ns)
    {
        ux_relevant_ = ux_solution_;
        uy_relevant_ = uy_solution_;
    }

    // Update ghosted old magnetization
    dealii::TrilinosWrappers::MPI::Vector Mx_old_relevant(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    dealii::TrilinosWrappers::MPI::Vector My_old_relevant(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    Mx_old_relevant = Mx_old_solution_;
    My_old_relevant = My_old_solution_;

    // Assemble system using CACHED assembler
    magnetization_assembler_->assemble(
        M_matrix_,
        Mx_rhs_,
        My_rhs_,
        ux_relevant_,
        uy_relevant_,
        phi_relevant_,
        theta_relevant_,
        Mx_old_relevant,
        My_old_relevant,
        dt,
        time_);

    // Initialize solver with new matrix (factorization)
    if (matrix_changed)
    {
        magnetization_solver_->initialize(M_matrix_);
    }

    // Solve for Mx
    magnetization_solver_->solve(Mx_solution_, Mx_rhs_);
    const unsigned int mx_iters = magnetization_solver_->last_n_iterations();

    // Solve for My (reuses same factorization)
    magnetization_solver_->solve(My_solution_, My_rhs_);
    const unsigned int my_iters = magnetization_solver_->last_n_iterations();

    // Update ghosted vectors
    Mx_relevant_ = Mx_solution_;
    My_relevant_ = My_solution_;

    // Store solver info
    last_M_info_.solver_name = "Magnetization-DG";
    last_M_info_.iterations = mx_iters + my_iters;
    last_M_info_.converged = true;

    if (params_.output.verbose)
    {
        pcout_ << "  [Magnetization] Mx_iters=" << mx_iters
               << ", My_iters=" << my_iters << "\n";
    }
}

// ============================================================================
// solve_magnetization_rhs_only() - RHS-only assembly for Picard iterations 2+
//
// OPTIMIZATION: Matrix structure is unchanged within a timestep (U is fixed),
// so we only need to reassemble the RHS which depends on:
//   - χ(θ)H (relaxation target)
//   - M^old (time derivative)
//
// The matrix factorization from iteration 1 is reused!
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_magnetization_rhs_only(double dt)
{
    // Update ghosted vectors
    phi_relevant_ = phi_solution_;
    theta_relevant_ = theta_solution_;

    // Update ghosted old magnetization
    dealii::TrilinosWrappers::MPI::Vector Mx_old_relevant(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    dealii::TrilinosWrappers::MPI::Vector My_old_relevant(
        M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    Mx_old_relevant = Mx_old_solution_;
    My_old_relevant = My_old_solution_;

    // Assemble RHS ONLY using cached assembler
    magnetization_assembler_->assemble_rhs_only(
        Mx_rhs_,
        My_rhs_,
        phi_relevant_,
        theta_relevant_,
        Mx_old_relevant,
        My_old_relevant,
        dt,
    time_);

    // Solve using CACHED factorization (no re-initialization!)
    magnetization_solver_->solve(Mx_solution_, Mx_rhs_);
    magnetization_solver_->solve(My_solution_, My_rhs_);

    // Update ghosted vectors
    Mx_relevant_ = Mx_solution_;
    My_relevant_ = My_solution_;
}

// ============================================================================
// solve_ns() - Navier-Stokes with Kelvin force (Paper Eq. 42e-42f)
//
// Momentum equation includes:
//   - Time derivative: ρ(∂u/∂t, v)
//   - Viscous: ν(T(u), T(v)) where T(u) = ∇u + (∇u)^T
//   - Convection: B_h(u^{k-1}, u, v) (skew-symmetric)
//   - Pressure: -(p, ∇·v)
//   - Kelvin force: B_h^m(θ, H, M) - magnetic body force
//
// CRITICAL: θ is LAGGED (θ^{k-1}) for energy stability!
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::solve_ns(double dt)
{
    // Update ghosted vectors
    // CRITICAL: Use LAGGED θ^{k-1} for energy stability per paper!
    theta_old_relevant_ = theta_old_solution_;
    ux_relevant_ = ux_old_solution_;
    uy_relevant_ = uy_old_solution_;

    if (params_.enable_magnetic)
    {
        phi_relevant_ = phi_solution_;
        Mx_relevant_ = Mx_solution_;
        My_relevant_ = My_solution_;
    }

    // Assemble NS system (free function from ns_assembler.h)
    const double L_y = params_.domain.y_max - params_.domain.y_min;
    // Use nu_water as reference viscosity for Schur preconditioner
    const double nu = params_.physics.nu_water;
    const bool include_time = true;
    const bool include_convection = true;

    // Choose assembler based on whether magnetic forces are enabled
    if (params_.enable_magnetic)
    {
        // Assemble NS with FULL ferrofluid forces:
        //   - Variable viscosity: ν(θ) = ν_water + (ν_ferro - ν_water)·H(θ/ε)
        //   - Kelvin force: μ₀(M·∇)H (magnetic body force)
        //   - Capillary force: (λ/ε)θ∇ψ (surface tension)
        //   - Gravity force: r·H(θ/ε)·g (Boussinesq buoyancy)
        // This is the CRITICAL coupling for ferrofluid simulations (Paper Eq. 14e, 17, 19)
        const double lambda = params_.physics.lambda;
        const double epsilon = params_.physics.epsilon;

        assemble_ns_system_ferrofluid_parallel<dim>(
              ux_dof_handler_,
              uy_dof_handler_,
              p_dof_handler_,
              ux_relevant_,
              uy_relevant_,
              nu,  // Reference viscosity for MMS and Schur preconditioner
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
              // Kelvin force inputs (magnetic)
              phi_dof_handler_,
              M_dof_handler_,
              phi_relevant_,
              Mx_relevant_,
              My_relevant_,
              params_.physics.mu_0,
              // Capillary force inputs (phase field)
              theta_dof_handler_,
              psi_dof_handler_,
              theta_old_relevant_,  // Use LAGGED θ^{k-1} per paper
              psi_relevant_,
              lambda,
              epsilon,
              // Variable viscosity (Paper Eq. 17)
              params_.physics.nu_water,
              params_.physics.nu_ferro,
              // Gravity force inputs
              params_.enable_gravity,
              params_.physics.r,
              params_.physics.gravity,
              params_.gravity_direction,
              // NEW: Dipole/applied field inputs for H = h_a + h_d
              params_.dipoles.positions,
              params_.dipoles.direction,
              params_.dipoles.intensity_max,
              params_.dipoles.ramp_time,
              time_,
              params_.use_reduced_magnetic_field,
              // MMS options
              params_.enable_mms,
              time_,
              time_ - dt,
              L_y);
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
            L_y);
    }

    // Solve NS system - select direct or iterative based on --direct flag
    if (!params_.solvers.ns.use_iterative)
    {
        // Direct solver with pressure pinning (fast for refinement <= 4)
        last_ns_info_ = solve_ns_system_direct_parallel(
            ns_matrix_,
            ns_rhs_,
            ns_solution_,
            ns_constraints_,
            p_to_ns_map_,           // For pressure pinning (removes null space)
            ns_locally_owned_,      // For parallel ownership
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
            params_.output.verbose);
    }

    // Extract individual solutions (free function)
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
// output_results() - Parallel VTU output
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::output_results(const std::string& output_dir)
{
    // Phase field output (θ and ψ)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(theta_dof_handler_);

        // Need ghosted vector for output
        dealii::TrilinosWrappers::MPI::Vector theta_out(
            theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
        theta_out = theta_solution_;
        data_out.add_data_vector(theta_out, "theta");

        dealii::TrilinosWrappers::MPI::Vector psi_out(
            psi_locally_owned_, psi_locally_relevant_, mpi_communicator_);
        psi_out = psi_solution_;
        data_out.add_data_vector(psi_out, "psi");

        // Subdomain for parallel visualization
        dealii::Vector<float> subdomain(triangulation_.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = triangulation_.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "phase_field",
            timestep_number_, mpi_communicator_);
    }

    // Velocity output (if NS enabled)
    if (params_.enable_ns)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(ux_dof_handler_);

        dealii::TrilinosWrappers::MPI::Vector ux_out(
            ux_locally_owned_, ux_locally_relevant_, mpi_communicator_);
        dealii::TrilinosWrappers::MPI::Vector uy_out(
            uy_locally_owned_, uy_locally_relevant_, mpi_communicator_);
        ux_out = ux_solution_;
        uy_out = uy_solution_;

        data_out.add_data_vector(ux_out, "ux");
        data_out.add_data_vector(uy_out, "uy");

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "velocity",
            timestep_number_, mpi_communicator_);

        // Pressure output (separate DoFHandler)
        dealii::DataOut<dim> data_out_p;
        data_out_p.attach_dof_handler(p_dof_handler_);

        dealii::TrilinosWrappers::MPI::Vector p_out(
            p_locally_owned_, p_locally_relevant_, mpi_communicator_);
        p_out = p_solution_;
        data_out_p.add_data_vector(p_out, "pressure");

        data_out_p.build_patches();
        data_out_p.write_vtu_with_pvtu_record(
            output_dir + "/", "pressure",
            timestep_number_, mpi_communicator_);
    }

    // Magnetic potential output
    if (params_.enable_magnetic)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(phi_dof_handler_);

        dealii::TrilinosWrappers::MPI::Vector phi_out(
            phi_locally_owned_, phi_locally_relevant_, mpi_communicator_);
        phi_out = phi_solution_;
        data_out.add_data_vector(phi_out, "phi");

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "magnetic_potential",
            timestep_number_, mpi_communicator_);
    }

    // Magnetization output (DG)
    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(M_dof_handler_);

        dealii::TrilinosWrappers::MPI::Vector Mx_out(
            M_locally_owned_, M_locally_relevant_, mpi_communicator_);
        dealii::TrilinosWrappers::MPI::Vector My_out(
            M_locally_owned_, M_locally_relevant_, mpi_communicator_);
        Mx_out = Mx_solution_;
        My_out = My_solution_;

        data_out.add_data_vector(Mx_out, "Mx");
        data_out.add_data_vector(My_out, "My");

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "magnetization",
            timestep_number_, mpi_communicator_);
    }

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