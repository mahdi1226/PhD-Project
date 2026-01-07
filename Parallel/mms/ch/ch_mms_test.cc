// ============================================================================
// mms/ch/ch_mms_test.cc - CH MMS Test Implementation (Parallel Version)
//
// Uses MMSContext for setup, which calls PRODUCTION code:
//   - setup_ch_coupled_system() from setup/ch_setup.h
//   - assemble_ch_system() from assembly/ch_assembler.h
//   - solve_ch_system() from solvers/ch_solver.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ch/ch_mms_test.h"
#include "mms/ch/ch_mms.h"
#include "mms/mms_context.h"

// PRODUCTION components
#include "assembly/ch_assembler.h"
#include "solvers/ch_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

// ============================================================================
// CHMMSConvergenceResult Implementation (unchanged - just data)
// ============================================================================

void CHMMSConvergenceResult::compute_rates()
{
    theta_L2_rates.clear();
    theta_H1_rates.clear();
    psi_L2_rates.clear();

    for (size_t i = 1; i < results.size(); ++i)
    {
        const double h_ratio = results[i-1].h / results[i].h;
        const double log_h_ratio = std::log(h_ratio);

        if (results[i-1].theta_L2 > 1e-15 && results[i].theta_L2 > 1e-15)
        {
            theta_L2_rates.push_back(
                std::log(results[i-1].theta_L2 / results[i].theta_L2) / log_h_ratio);
        }
        else
        {
            theta_L2_rates.push_back(0.0);
        }

        if (results[i-1].theta_H1 > 1e-15 && results[i].theta_H1 > 1e-15)
        {
            theta_H1_rates.push_back(
                std::log(results[i-1].theta_H1 / results[i].theta_H1) / log_h_ratio);
        }
        else
        {
            theta_H1_rates.push_back(0.0);
        }

        if (results[i-1].psi_L2 > 1e-15 && results[i].psi_L2 > 1e-15)
        {
            psi_L2_rates.push_back(
                std::log(results[i-1].psi_L2 / results[i].psi_L2) / log_h_ratio);
        }
        else
        {
            psi_L2_rates.push_back(0.0);
        }
    }
}

void CHMMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: CH_STANDALONE\n";
    std::cout << "========================================\n";

    std::cout << std::left
              << std::setw(6) << "Ref"
              << std::setw(12) << "h"
              << std::setw(12) << "θ_L2"
              << std::setw(8) << "rate"
              << std::setw(12) << "θ_H1"
              << std::setw(8) << "rate"
              << std::setw(12) << "ψ_L2"
              << std::setw(8) << "rate"
              << std::setw(10) << "time(s)"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(6) << r.refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.h
                  << std::setw(12) << r.theta_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << (i > 0 ? theta_L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.theta_H1
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << (i > 0 ? theta_H1_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.psi_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << (i > 0 ? psi_L2_rates[i-1] : 0.0)
                  << std::setw(10) << r.total_time
                  << "\n";
    }
    std::cout << "========================================\n";

    if (passes())
        std::cout << "[PASS] All convergence rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some convergence rates below expected!\n";
}

void CHMMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[CH MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    file << "refinement,h,n_dofs,theta_L2,theta_L2_rate,theta_H1,theta_H1_rate,"
         << "psi_L2,psi_L2_rate,setup_time,assembly_time,solve_time,total_time,"
         << "solver_iterations,solver_residual\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        file << r.refinement << ","
             << std::scientific << std::setprecision(6) << r.h << ","
             << r.n_dofs << ","
             << r.theta_L2 << ","
             << (i > 0 ? theta_L2_rates[i-1] : 0.0) << ","
             << r.theta_H1 << ","
             << (i > 0 ? theta_H1_rates[i-1] : 0.0) << ","
             << r.psi_L2 << ","
             << (i > 0 ? psi_L2_rates[i-1] : 0.0) << ","
             << std::fixed << std::setprecision(4)
             << r.setup_time << ","
             << r.assembly_time << ","
             << r.solve_time << ","
             << r.total_time << ","
             << r.solver_iterations << ","
             << std::scientific << r.solver_residual << "\n";
    }

    file.close();
    std::cout << "[CH MMS] Results written to " << filename << "\n";
}

bool CHMMSConvergenceResult::passes(double tol) const
{
    if (theta_L2_rates.empty())
        return false;

    // Check last rate is within tolerance of expected
    const double last_L2_rate = theta_L2_rates.back();
    const double last_H1_rate = theta_H1_rates.back();

    return (last_L2_rate >= expected_L2_rate - tol) &&
           (last_H1_rate >= expected_H1_rate - tol);
}

// ============================================================================
// Parallel error computation helper
// ============================================================================
template <int dim>
static CHMMSErrors compute_ch_mms_errors_parallel(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double time,
    MPI_Comm mpi_communicator)
{
    CHMMSErrors errors;

    // Exact solutions
    CHExactTheta<dim> exact_theta;
    CHExactPsi<dim> exact_psi;
    exact_theta.set_time(time);
    exact_psi.set_time(time);

    // Quadrature for error integration
    const unsigned int quad_degree = theta_dof_handler.get_fe().degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);

    // Compute local error contributions
    dealii::Vector<double> theta_diff_L2(theta_dof_handler.get_triangulation().n_active_cells());
    dealii::Vector<double> theta_diff_H1(theta_dof_handler.get_triangulation().n_active_cells());
    dealii::Vector<double> psi_diff_L2(psi_dof_handler.get_triangulation().n_active_cells());

    dealii::VectorTools::integrate_difference(
        theta_dof_handler, theta_solution, exact_theta,
        theta_diff_L2, quadrature, dealii::VectorTools::L2_norm);

    dealii::VectorTools::integrate_difference(
        theta_dof_handler, theta_solution, exact_theta,
        theta_diff_H1, quadrature, dealii::VectorTools::H1_seminorm);

    dealii::VectorTools::integrate_difference(
        psi_dof_handler, psi_solution, exact_psi,
        psi_diff_L2, quadrature, dealii::VectorTools::L2_norm);

    // Compute global norms via MPI reduction
    // integrate_difference returns cell-wise norm^2, so we sum and sqrt
    double local_theta_L2_sq = theta_diff_L2.norm_sqr();
    double local_theta_H1_sq = theta_diff_H1.norm_sqr();
    double local_psi_L2_sq = psi_diff_L2.norm_sqr();

    double global_theta_L2_sq = 0.0, global_theta_H1_sq = 0.0, global_psi_L2_sq = 0.0;

    MPI_Allreduce(&local_theta_L2_sq, &global_theta_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_theta_H1_sq, &global_theta_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_psi_L2_sq, &global_psi_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    errors.theta_L2 = std::sqrt(global_theta_L2_sq);
    errors.theta_H1 = std::sqrt(global_theta_H1_sq);
    errors.psi_L2 = std::sqrt(global_psi_L2_sq);

    return errors;
}

// ============================================================================
// run_ch_mms_single - Single refinement test using MMSContext (Parallel)
// ============================================================================

template <int dim>
static CHMMSResult run_ch_mms_single_impl(
    unsigned int refinement,
    Parameters params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CHMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // MMS Parameters
    // ========================================================================
    const double t_init = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;

    params.enable_mms = true;
    params.enable_ns = false;  // Standalone CH
    params.time.dt = dt;

    // Domain: unit square for MMS
    params.domain.x_min = 0.0;
    params.domain.x_max = 1.0;
    params.domain.y_min = 0.0;
    params.domain.y_max = 1.0;
    params.domain.initial_cells_x = 1;
    params.domain.initial_cells_y = 1;

    // ========================================================================
    // Setup using MMSContext - USES PRODUCTION CODE
    // ========================================================================
    auto setup_start = std::chrono::high_resolution_clock::now();

    MMSContext<dim> ctx(mpi_communicator);
    ctx.setup_mesh(params, refinement);
    ctx.setup_ch(params, t_init);
    ctx.apply_ch_initial_conditions(params, t_init);

    auto setup_end = std::chrono::high_resolution_clock::now();
    double total_setup_time = std::chrono::duration<double>(setup_end - setup_start).count();

    // Zero velocity for standalone CH (no NS coupling) - using Trilinos vectors
    dealii::TrilinosWrappers::MPI::Vector ux_zero(ctx.theta_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_zero(ctx.theta_locally_owned, mpi_communicator);
    ux_zero = 0;
    uy_zero = 0;

    // ========================================================================
    // Time stepping loop
    // ========================================================================
    double current_time = t_init;
    double total_assembly_time = 0.0;
    double total_solve_time = 0.0;
    unsigned int total_iterations = 0;
    double last_residual = 0.0;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Copy solution to old
        ctx.theta_old_owned = ctx.theta_owned;
        ctx.update_theta_old_ghosts();

        // Update boundary constraints for new time
        ctx.update_ch_constraints(params, current_time);

        // ====================================================================
        // ASSEMBLY - Using PRODUCTION code
        // ====================================================================
        auto assembly_start = std::chrono::high_resolution_clock::now();

        ctx.ch_matrix = 0;
        ctx.ch_rhs = 0;

        // Assembly reads from ghosted vectors
        // Uses distribute_local_to_global for constraint handling
        assemble_ch_system<dim>(
            ctx.theta_dof_handler, ctx.psi_dof_handler,
            ctx.theta_old_relevant, ux_zero, uy_zero,
            params, dt, current_time,
            ctx.theta_to_ch_map, ctx.psi_to_ch_map,
            ctx.ch_constraints,  // ADDED: constraints for BC handling
            ctx.ch_matrix, ctx.ch_rhs);

        auto assembly_end = std::chrono::high_resolution_clock::now();
        total_assembly_time += std::chrono::duration<double>(assembly_end - assembly_start).count();

        // ====================================================================
        // SOLVE - Using PRODUCTION code
        // ====================================================================
        auto solve_start = std::chrono::high_resolution_clock::now();

        // PRODUCTION solver (GMRES + AMG or Direct)
        LinearSolverParams solver_params = params.solvers.ch;
        if (solver_type == CHSolverType::Direct)
        {
            solver_params.use_iterative = false;
            solver_params.fallback_to_direct = true;
        }

        SolverInfo info = solve_ch_system(
            ctx.ch_matrix, ctx.ch_rhs, ctx.ch_constraints,
            ctx.ch_locally_owned, ctx.theta_locally_owned, ctx.psi_locally_owned,
            ctx.theta_to_ch_map, ctx.psi_to_ch_map,
            ctx.theta_owned, ctx.psi_owned,
            solver_params, mpi_communicator, false);

        total_iterations += info.iterations;
        last_residual = info.residual;

        // Update ghost values for next time step
        ctx.update_theta_ghosts();
        ctx.update_psi_ghosts();

        auto solve_end = std::chrono::high_resolution_clock::now();
        total_solve_time += std::chrono::duration<double>(solve_end - solve_start).count();
    }

    // ========================================================================
    // Compute errors (parallel reduction)
    // ========================================================================
    CHMMSErrors errors = compute_ch_mms_errors_parallel<dim>(
        ctx.theta_dof_handler, ctx.psi_dof_handler,
        ctx.theta_relevant, ctx.psi_relevant,
        current_time, mpi_communicator);

    auto total_end = std::chrono::high_resolution_clock::now();

    // Fill result
    result.h = ctx.get_min_h();
    result.n_dofs = ctx.n_ch_dofs();
    result.theta_L2 = errors.theta_L2;
    result.theta_H1 = errors.theta_H1;
    result.psi_L2 = errors.psi_L2;
    result.setup_time = total_setup_time;
    result.assembly_time = total_assembly_time;
    result.solve_time = total_solve_time;
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();
    result.solver_iterations = total_iterations;
    result.solver_residual = last_residual;

    return result;
}

CHMMSResult run_ch_mms_single(
    unsigned int refinement,
    const Parameters& params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    return run_ch_mms_single_impl<2>(refinement, params, solver_type, n_time_steps, mpi_communicator);
}

// ============================================================================
// run_ch_mms_standalone - Full convergence study
// ============================================================================

CHMMSConvergenceResult run_ch_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    CHMMSConvergenceResult result;
    result.fe_degree = params.fe.degree_phase;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = params.fe.degree_phase + 1;
    result.expected_H1_rate = params.fe.degree_phase;

    const double t_init = 0.1;
    const double t_final = 0.2;
    result.dt = (t_final - t_init) / n_time_steps;

    if (this_rank == 0)
    {
        std::cout << "\n[CH_STANDALONE] Running parallel convergence study...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  t ∈ [" << t_init << ", " << t_final << "], dt = " << result.dt << "\n";
        std::cout << "  ε = " << params.physics.epsilon
                  << ", γ = " << params.physics.mobility << "\n";
        std::cout << "  FE degree = Q" << params.fe.degree_phase << "\n";
        std::cout << "  Solver = " << (solver_type == CHSolverType::Direct ? "Direct" : "GMRES+AMG") << "\n";
        std::cout << "  Using MMSContext with PRODUCTION: ch_setup + ch_assembler + ch_solver\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        CHMMSResult single = run_ch_mms_single(ref, params, solver_type, n_time_steps, mpi_communicator);
        result.results.push_back(single);

        if (this_rank == 0)
        {
            std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << single.theta_L2
                      << ", θ_H1=" << single.theta_H1
                      << ", time=" << std::fixed << std::setprecision(1) << single.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// compare_ch_solvers - Compare direct vs iterative
// ============================================================================

void compare_ch_solvers(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[CH Solver Comparison] Refinement " << refinement << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    // Direct solver
    if (this_rank == 0)
        std::cout << "Direct (Amesos):\n";
    CHMMSResult direct = run_ch_mms_single(refinement, params, CHSolverType::Direct, n_time_steps, mpi_communicator);
    if (this_rank == 0)
    {
        std::cout << "  θ_L2 = " << std::scientific << direct.theta_L2
                  << ", time = " << std::fixed << std::setprecision(3) << direct.total_time << "s\n";
    }

    // Iterative solver
    if (this_rank == 0)
        std::cout << "Iterative (GMRES+AMG):\n";
    CHMMSResult iterative = run_ch_mms_single(refinement, params, CHSolverType::GMRES_AMG, n_time_steps, mpi_communicator);
    if (this_rank == 0)
    {
        std::cout << "  θ_L2 = " << std::scientific << iterative.theta_L2
                  << ", iters = " << iterative.solver_iterations
                  << ", time = " << std::fixed << std::setprecision(3) << iterative.total_time << "s\n";

        std::cout << "\nSpeedup (direct/iterative): "
                  << std::setprecision(2) << direct.total_time / iterative.total_time << "x\n";
    }
}