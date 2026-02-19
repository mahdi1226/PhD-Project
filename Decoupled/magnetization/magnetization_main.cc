// ============================================================================
// magnetization/magnetization_main.cc — Standalone Magnetization Driver
//
// Modes:
//   mms       MMS spatial convergence (2D), refs 2-6
//   2d        Relaxation M=0 → χH with VTK output
//   3d        (Not yet implemented)
//   temporal  Temporal convergence study (2D), sweep dt
//
// Usage:
//   mpirun -np 4 ./magnetization_main --mode mms
//   mpirun -np 4 ./magnetization_main --mode 2d --refinement 5
//   mpirun -np 4 ./magnetization_main --mode 3d
//   mpirun -np 4 ./magnetization_main --mode temporal
//   mpirun -np 4 ./magnetization_main --ref 2 3 4 5
//   mpirun -np 4 ./magnetization_main --steps 100
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42c / 56-57
// ============================================================================

#include "magnetization/magnetization.h"
#include "magnetization/tests/magnetization_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"
#include "physics/applied_field.h"
#include "physics/material_properties.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace dealii;


// ============================================================================
// Helper: create dummy CG Q1 field (uniform scalar) on its own DoFHandler
// ============================================================================
template <int dim>
void create_uniform_cg_field(
    DoFHandler<dim>&                    dof_handler,
    FE_Q<dim>&                          fe,
    TrilinosWrappers::MPI::Vector&      owned_vec,
    TrilinosWrappers::MPI::Vector&      relevant_vec,
    double                              value,
    MPI_Comm                            mpi_comm)
{
    dof_handler.distribute_dofs(fe);

    IndexSet locally_owned   = dof_handler.locally_owned_dofs();
    IndexSet locally_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

    owned_vec.reinit(locally_owned, mpi_comm);
    relevant_vec.reinit(locally_relevant, mpi_comm);

    owned_vec = value;
    owned_vec.compress(VectorOperation::insert);
    relevant_vec = owned_vec;
}


// ============================================================================
// 2D Relaxation: M=0 → χH under applied field
// ============================================================================
template <int dim>
void run_relaxation(const Parameters& params, MPI_Comm mpi_comm,
                    int requested_steps = -1)
{
    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    pcout << "============================================\n"
          << "  Magnetization Standalone — Relaxation (" << dim << "D)\n"
          << "  M = 0 → χH under applied h_a\n"
          << "============================================\n\n";

    // ── Mesh ──
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1(params.domain.x_min, params.domain.y_min);
    Point<dim> p2(params.domain.x_max, params.domain.y_max);
    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y};
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(params.mesh.initial_refinement);

    pcout << "  Mesh: " << triangulation.n_global_active_cells() << " cells, "
          << Utilities::MPI::n_mpi_processes(mpi_comm) << " MPI ranks\n";

    // ── Dummy fields: φ=0, θ=+1, U=0 ──
    FE_Q<dim> fe_phi(1), fe_theta(1), fe_vel(1);
    DoFHandler<dim> phi_dof(triangulation), theta_dof(triangulation), u_dof(triangulation);
    TrilinosWrappers::MPI::Vector phi_own, phi_rel;
    TrilinosWrappers::MPI::Vector theta_own, theta_rel;
    TrilinosWrappers::MPI::Vector ux_own, ux_rel, uy_own, uy_rel;

    create_uniform_cg_field(phi_dof, fe_phi, phi_own, phi_rel, 0.0, mpi_comm);
    create_uniform_cg_field(theta_dof, fe_theta, theta_own, theta_rel, 1.0, mpi_comm);
    create_uniform_cg_field(u_dof, fe_vel, ux_own, ux_rel, 0.0, mpi_comm);
    {
        IndexSet lo = u_dof.locally_owned_dofs();
        IndexSet lr = DoFTools::extract_locally_relevant_dofs(u_dof);
        uy_own.reinit(lo, mpi_comm);
        uy_rel.reinit(lr, mpi_comm);
        uy_own = 0.0;
        uy_own.compress(VectorOperation::insert);
        uy_rel = uy_own;
    }

    pcout << "  φ DoFs: " << phi_dof.n_dofs()
          << ", θ DoFs: " << theta_dof.n_dofs()
          << ", U DoFs: " << u_dof.n_dofs() << "\n";

    // ── Magnetization subsystem ──
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    mag.setup();
    mag.update_ghosts();

    pcout << "  M DoFs: " << mag.get_dof_handler().n_dofs() << " (DG Q1)\n\n";

    // ── Time stepping parameters ──
    const double tau_M = params.physics.tau_M;
    const double dt    = tau_M / 10.0;
    const unsigned int n_steps      = (requested_steps > 0)
                                        ? static_cast<unsigned int>(requested_steps)
                                        : 100;   // ~10 τ_M total
    const unsigned int output_every = std::max(1u, n_steps / 20);

    pcout << "  τ_M = " << tau_M << "\n"
          << "  dt  = " << dt << " (τ_M/10)\n"
          << "  n_steps = " << n_steps << " (total time = "
          << n_steps * dt << ")\n"
          << "  Output every " << output_every << " steps\n\n";

    // ── Output directory ──
    const std::string output_dir =
        "../magnetization_results/vtk/" +
        timestamped_filename_mpi("magnetization", "", mpi_comm);

    pcout << "  Output: " << output_dir << "\n\n";

    // ── Initial VTK (step 0: M = 0) ──
    mag.write_vtu(output_dir, 0, 0.0);

    // ── Console header ──
    pcout << std::string(85, '-') << "\n"
          << std::setw(5) << "Step"
          << std::setw(12) << "time"
          << std::setw(12) << "|M|_mean"
          << std::setw(12) << "|M|_max"
          << std::setw(14) << "||M-χH||_L2"
          << std::setw(10) << "Mx_its"
          << std::setw(10) << "My_its"
          << std::setw(10) << "wall(s)"
          << "\n"
          << std::string(85, '-') << "\n";

    // ── Time loop ──
    Timer timer;
    double current_time = 0.0;

    for (unsigned int step = 1; step <= n_steps; ++step)
    {
        current_time += dt;
        timer.restart();

        mag.assemble(
            mag.get_Mx_relevant(), mag.get_My_relevant(),
            phi_rel, phi_dof,
            theta_rel, theta_dof,
            ux_rel, uy_rel, u_dof,
            dt, current_time);

        SolverInfo info = mag.solve();
        mag.update_ghosts();

        timer.stop();

        auto diag = mag.compute_diagnostics_standalone(phi_rel, phi_dof, current_time);

        pcout << std::setw(5)  << step
              << std::setw(12) << std::scientific << std::setprecision(2) << current_time
              << std::setw(12) << diag.M_magnitude_mean
              << std::setw(12) << diag.M_magnitude_max
              << std::setw(14) << diag.M_equilibrium_departure_L2
              << std::setw(10) << std::fixed << diag.Mx_iterations
              << std::setw(10) << diag.My_iterations
              << std::setw(10) << std::setprecision(3) << timer.wall_time()
              << "\n";

        if (step % output_every == 0 || step == n_steps)
        {
            mag.write_vtu(output_dir, step, current_time);
        }
    }

    pcout << std::string(85, '-') << "\n\n";

    // ── Final summary ──
    auto diag = mag.compute_diagnostics_standalone(phi_rel, phi_dof, current_time);

    // Compute χ₀·|h_a| at domain center for reference
    Point<dim> center(0.5 * (params.domain.x_min + params.domain.x_max),
                      0.5 * (params.domain.y_min + params.domain.y_max));
    Tensor<1, dim> ha_center = compute_applied_field(center, params, current_time);
    const double target_center = params.physics.chi_0 * ha_center.norm();

    pcout << "=== Final State ===\n"
          << "  |M| mean      = " << diag.M_magnitude_mean << "\n"
          << "  χ₀·|h_a| at center = " << target_center
          << "  (χ₀=" << params.physics.chi_0
          << ", |h_a|=" << ha_center.norm() << ")\n"
          << "  ||M - χH||_L2 = " << diag.M_equilibrium_departure_L2 << "\n"
          << "  ∫Mx dΩ        = " << diag.Mx_integral << "\n"
          << "  ∫My dΩ        = " << diag.My_integral << "\n"
          << "  VTK files in: " << output_dir << "\n\n";
}


// ============================================================================
// MMS convergence study (same logic as test file, uses production facade)
// ============================================================================
struct MagMMSResultMain
{
    unsigned int refinement = 0;
    unsigned int n_dofs     = 0;
    double h        = 0.0;
    double Mx_L2    = 0.0;
    double My_L2    = 0.0;
    double M_L2     = 0.0;
    double Mx_Linf  = 0.0;
    double My_Linf  = 0.0;
    double M_Linf   = 0.0;
    double time_s   = 0.0;
};

template <int dim>
MagMMSResultMain run_mag_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    MagMMSResultMain result;
    result.refinement = refinement;

    auto t_start_wall = std::chrono::high_resolution_clock::now();

    const double L_y    = params.domain.y_max - params.domain.y_min;
    const double t0     = 0.1;
    const double t_end  = 0.2;
    const double dt     = (t_end - t0) / n_time_steps;

    // ── Mesh ──
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1(params.domain.x_min, params.domain.y_min);
    Point<dim> p2(params.domain.x_max, params.domain.y_max);
    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y};
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ── Facade ──
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    mag.setup();
    result.n_dofs = 2 * mag.get_dof_handler().n_dofs();

    // ── Dummy CG DoFHandlers: φ=0, θ=1, U=0 ──
    FE_Q<dim> fe_CG(params.fe.degree_velocity);
    DoFHandler<dim> phi_dof_handler(triangulation);
    DoFHandler<dim> theta_dof_handler(triangulation);
    DoFHandler<dim> u_dof_handler(triangulation);

    phi_dof_handler.distribute_dofs(fe_CG);
    theta_dof_handler.distribute_dofs(fe_CG);
    u_dof_handler.distribute_dofs(fe_CG);

    auto phi_owned   = phi_dof_handler.locally_owned_dofs();
    auto phi_relevant = DoFTools::extract_locally_relevant_dofs(phi_dof_handler);
    auto theta_owned   = theta_dof_handler.locally_owned_dofs();
    auto theta_relevant = DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    auto u_owned   = u_dof_handler.locally_owned_dofs();
    auto u_relevant = DoFTools::extract_locally_relevant_dofs(u_dof_handler);

    TrilinosWrappers::MPI::Vector phi_vec(phi_owned, phi_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector theta_vec(theta_owned, theta_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector ux_vec(u_owned, u_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_vec(u_owned, u_relevant, mpi_comm);

    phi_vec   = 0.0;
    theta_vec = 1.0;
    ux_vec    = 0.0;
    uy_vec    = 0.0;

    // ── MMS source callback ──
    mag.set_mms_source(
        [L_y](const Point<dim>& point,
              double t_new, double t_old, double tau_M,
              double chi_val,
              const Tensor<1, dim>& H,
              const Tensor<1, dim>& U,
              double div_U)
              -> Tensor<1, dim>
        {
            return compute_mag_mms_source_with_transport<dim>(
                point, t_new, t_old, tau_M, chi_val, H, U, div_U, L_y);
        });

    // ── Project exact IC at t=t0 ──
    {
        MagExactMx<dim> exact_Mx(t0, L_y);
        MagExactMy<dim> exact_My(t0, L_y);
        mag.project_initial_condition(exact_Mx, exact_My);
    }

    // ── Mesh size ──
    {
        double local_min_h = std::numeric_limits<double>::max();
        for (const auto& cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                local_min_h = std::min(local_min_h, cell->diameter());
        MPI_Allreduce(&local_min_h, &result.h, 1,
                      MPI_DOUBLE, MPI_MIN, mpi_comm);
    }

    // ── Time stepping ──
    double current_time = t0;
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;
        mag.update_ghosts();

        const auto& Mx_old = mag.get_Mx_relevant();
        const auto& My_old = mag.get_My_relevant();

        mag.assemble(
            Mx_old, My_old,
            phi_vec, phi_dof_handler,
            theta_vec, theta_dof_handler,
            ux_vec, uy_vec, u_dof_handler,
            dt, current_time);

        mag.solve();
    }

    // ── Errors ──
    mag.update_ghosts();
    MagMMSError errors = compute_mag_mms_errors_parallel<dim>(
        mag.get_dof_handler(),
        mag.get_Mx_relevant(), mag.get_My_relevant(),
        current_time, L_y, mpi_comm);

    result.Mx_L2   = errors.Mx_L2;
    result.My_L2   = errors.My_L2;
    result.M_L2    = errors.M_L2;
    result.Mx_Linf = errors.Mx_Linf;
    result.My_Linf = errors.My_Linf;
    result.M_Linf  = errors.M_Linf;

    auto t_end_wall = std::chrono::high_resolution_clock::now();
    result.time_s = std::chrono::duration<double>(t_end_wall - t_start_wall).count();

    return result;
}


// ============================================================================
// Temporal convergence test
// ============================================================================
template <int dim>
bool run_mag_temporal_convergence(const Parameters& params, MPI_Comm mpi_comm)
{
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int ref = 4;
    const std::vector<unsigned int> steps_list = {5, 10, 20, 40, 80};

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   Magnetization Temporal Convergence (ref=" << ref << ")\n";
        std::cout << "============================================================\n";
    }

    std::vector<double> dts, M_L2_errors;

    for (unsigned int n_steps : steps_list)
    {
        if (rank == 0)
            std::cout << "  n_steps=" << n_steps << "... " << std::flush;

        MagMMSResultMain r = run_mag_mms_single<dim>(ref, params, n_steps, mpi_comm);
        const double dt = 0.1 / n_steps;
        dts.push_back(dt);
        M_L2_errors.push_back(r.M_L2);

        if (rank == 0)
            std::cout << "M_L2=" << std::scientific << std::setprecision(2)
                      << r.M_L2 << "\n";
    }

    if (rank == 0)
    {
        std::cout << "\n  dt-convergence rates:\n";
        std::cout << std::setw(12) << "dt" << std::setw(14) << "M_L2"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(34, '-') << "\n";

        bool pass = false;
        for (size_t i = 0; i < dts.size(); ++i)
        {
            double rate = 0.0;
            if (i > 0 && M_L2_errors[i] > 1e-15 && M_L2_errors[i-1] > 1e-15)
                rate = std::log(M_L2_errors[i-1] / M_L2_errors[i])
                     / std::log(dts[i-1] / dts[i]);

            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << dts[i]
                      << std::setw(14) << M_L2_errors[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << rate << "\n";

            if (i == dts.size() - 1)
                pass = (rate >= 0.7);
        }

        std::cout << "\n  Expected: O(dt^1) for backward Euler\n";
        if (pass) std::cout << "  [PASS] Temporal convergence rate within tolerance!\n";
        else      std::cout << "  [FAIL] Temporal convergence rate below expected!\n";

        return pass;
    }
    return true;
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        deallog.depth_console(0);

        Parameters params = Parameters::parse_command_line(argc, argv);
        const std::string& mode = params.run.mode;
        const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

        constexpr int dim = 2;

        // ================================================================
        // Mode: mms — spatial convergence study (2D)
        // ================================================================
        if (mode == "mms")
        {
            const unsigned int n_time_steps =
                (params.run.steps > 0) ? static_cast<unsigned int>(params.run.steps) : 1;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Magnetization MMS Convergence (DG"
                          << params.fe.degree_magnetization << ")\n";
                std::cout << "============================================================\n";
                std::cout << "  Refinements: ";
                for (auto r : params.run.refs) std::cout << r << " ";
                std::cout << "\n  Time steps: " << n_time_steps << "\n\n";
            }

            std::vector<MagMMSResultMain> results;
            for (unsigned int ref : params.run.refs)
            {
                if (rank == 0)
                    std::cout << "  Refinement " << ref << "... " << std::flush;

                MagMMSResultMain r = run_mag_mms_single<dim>(ref, params, n_time_steps, mpi_comm);
                results.push_back(r);

                if (rank == 0)
                    std::cout << "M_L2=" << std::scientific << std::setprecision(2)
                              << r.M_L2 << ", time="
                              << std::fixed << std::setprecision(1)
                              << r.time_s << "s\n";
            }

            // Compute and check rates
            const double expected_L2_rate = params.fe.degree_magnetization + 1;  // DG1 → 2.0
            bool pass = false;

            if (rank == 0 && results.size() >= 2)
            {
                std::cout << "\n"
                          << std::left
                          << std::setw(6)  << "Ref"
                          << std::setw(12) << "h"
                          << std::setw(12) << "M_L2"
                          << std::setw(8)  << "L2rate"
                          << std::setw(12) << "M_Linf"
                          << std::setw(8)  << "Lfrate"
                          << std::setw(10) << "DoFs"
                          << std::setw(10) << "time(s)"
                          << "\n" << std::string(78, '-') << "\n";

                double last_L2_rate = 0.0;
                for (size_t i = 0; i < results.size(); ++i)
                {
                    const auto& r = results[i];
                    double L2_rate = 0.0, Linf_rate = 0.0;
                    if (i > 0)
                    {
                        const auto& prev = results[i - 1];
                        const double log_h = std::log(prev.h / r.h);
                        if (prev.M_L2 > 1e-15 && r.M_L2 > 1e-15)
                            L2_rate = std::log(prev.M_L2 / r.M_L2) / log_h;
                        if (prev.M_Linf > 1e-15 && r.M_Linf > 1e-15)
                            Linf_rate = std::log(prev.M_Linf / r.M_Linf) / log_h;
                        last_L2_rate = L2_rate;
                    }

                    std::cout << std::left << std::setw(6) << r.refinement
                              << std::scientific << std::setprecision(2)
                              << std::setw(12) << r.h
                              << std::setw(12) << r.M_L2
                              << std::fixed << std::setprecision(2)
                              << std::setw(8) << L2_rate
                              << std::scientific << std::setprecision(2)
                              << std::setw(12) << r.M_Linf
                              << std::fixed << std::setprecision(2)
                              << std::setw(8) << Linf_rate
                              << std::setw(10) << r.n_dofs
                              << std::setprecision(1)
                              << std::setw(10) << r.time_s
                              << "\n";
                }

                std::cout << std::string(78, '-') << "\n";

                const double tolerance = 0.3;
                pass = (last_L2_rate >= expected_L2_rate - tolerance);

                if (pass)
                    std::cout << "[PASS] L2 rate " << std::fixed << std::setprecision(2)
                              << last_L2_rate << " >= "
                              << expected_L2_rate - tolerance << "\n";
                else
                    std::cout << "[FAIL] L2 rate " << std::fixed << std::setprecision(2)
                              << last_L2_rate << " < "
                              << expected_L2_rate - tolerance << "\n";
            }
            return pass ? 0 : 1;
        }
        // ================================================================
        // Mode: 2d — relaxation M=0 → χH
        // ================================================================
        else if (mode == "2d")
        {
            // Default parameters for relaxation test
            params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
            params.domain.y_min = 0.0;  params.domain.y_max = 0.6;
            params.domain.initial_cells_x = 10;
            params.domain.initial_cells_y = 6;
            if (params.mesh.initial_refinement == 0)
                params.mesh.initial_refinement = 4;
            params.physics.chi_0   = 0.5;
            params.physics.tau_M   = 1e-4;
            params.physics.epsilon = 0.01;
            params.enable_magnetic = true;

            // Standalone test: instant field
            params.dipoles.ramp_time = 0.0;

            // Default dipole below domain
            if (params.dipoles.positions.empty())
            {
                params.dipoles.positions.push_back(Point<2>(0.5, -0.3));
                params.dipoles.intensity_max = 4.3;
                params.dipoles.direction = {0.0, 1.0};
            }

            run_relaxation<2>(params, mpi_comm, params.run.steps);
            return 0;
        }
        // ================================================================
        // Mode: 3d — not yet implemented
        // ================================================================
        else if (mode == "3d")
        {
            if (rank == 0)
                std::cout << "3D Magnetization not yet implemented.\n";
            return 0;
        }
        // ================================================================
        // Mode: temporal — temporal convergence study
        // ================================================================
        else if (mode == "temporal")
        {
            bool pass = run_mag_temporal_convergence<dim>(params, mpi_comm);
            return pass ? 0 : 1;
        }
        else
        {
            if (rank == 0)
                std::cerr << "Unknown mode: " << mode
                          << " (use mms, 2d, 3d, or temporal)\n";
            return 1;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "\n[Error] " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n[Error] Unknown exception!\n";
        return 1;
    }
}
