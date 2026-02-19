// ============================================================================
// cahn_hilliard/cahn_hilliard_main.cc — Standalone CH Driver
//
// Modes:
//   mms       MMS spatial convergence (2D), refs 2-6
//   2d        Spinodal decomposition 2D with VTK output
//   3d        Spinodal decomposition 3D with VTK output
//   temporal  Temporal convergence study (2D), sweep dt
//
// Usage:
//   mpirun -np 4 ./cahn_hilliard_main --mode mms
//   mpirun -np 4 ./cahn_hilliard_main --mode 2d --refinement 5
//   mpirun -np 4 ./cahn_hilliard_main --mode 3d --refinement 0
//   mpirun -np 4 ./cahn_hilliard_main --mode temporal
//   mpirun -np 4 ./cahn_hilliard_main --ref 2 3 4 5
//   mpirun -np 4 ./cahn_hilliard_main --steps 500
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42b
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "cahn_hilliard/tests/cahn_hilliard_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
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
#include <random>
#include <string>
#include <vector>

using namespace dealii;


// ============================================================================
// Random initial condition: θ = mean + amplitude * random[-1,1]
// ============================================================================
template <int dim>
class RandomPhaseIC : public Function<dim>
{
public:
    RandomPhaseIC(double mean, double amplitude, unsigned int seed)
        : Function<dim>(1), mean_(mean), amplitude_(amplitude),
          rng_(seed), dist_(-1.0, 1.0) {}

    double value(const Point<dim>& /*p*/,
                 const unsigned int /*component*/ = 0) const override
    {
        return mean_ + amplitude_ * dist_(rng_);
    }

private:
    double mean_, amplitude_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> dist_;
};


// ============================================================================
// Spinodal decomposition simulation (2D or 3D)
// ============================================================================
template <int dim>
void run_spinodal_decomposition(const Parameters& params, MPI_Comm mpi_comm,
                                int requested_steps = -1)
{
    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    pcout << "============================================\n"
          << "  Cahn-Hilliard Standalone — Spinodal Decomposition (" << dim << "D)\n"
          << "  θ ≈ 0 (random) → phase separation ±1\n"
          << "============================================\n\n";

    // ── Mesh ──
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1, p2;
    std::vector<unsigned int> subdivisions(dim);

    p1[0] = params.domain.x_min;
    p2[0] = params.domain.x_max;
    subdivisions[0] = params.domain.initial_cells_x;

    if constexpr (dim >= 2)
    {
        p1[1] = params.domain.y_min;
        p2[1] = params.domain.y_max;
        subdivisions[1] = params.domain.initial_cells_y;
    }
    if constexpr (dim >= 3)
    {
        p1[2] = 0.0;
        p2[2] = 1.0;
        subdivisions[2] = params.domain.initial_cells_x;
    }

    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(params.mesh.initial_refinement);

    pcout << "  Mesh: " << triangulation.n_global_active_cells() << " cells, "
          << Utilities::MPI::n_mpi_processes(mpi_comm) << " MPI ranks\n";

    // ── CH subsystem ──
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);
    ch.setup();

    pcout << "  θ DoFs: " << ch.get_theta_dof_handler().n_dofs() << " (CG Q"
          << params.fe.degree_phase << ")\n"
          << "  ψ DoFs: " << ch.get_psi_dof_handler().n_dofs() << "\n\n";

    // ── Dummy velocity (zero — no convection) ──
    FE_Q<dim> fe_vel(params.fe.degree_velocity);
    DoFHandler<dim> vel_dof(triangulation);
    vel_dof.distribute_dofs(fe_vel);

    IndexSet vel_owned = vel_dof.locally_owned_dofs();
    IndexSet vel_relevant = DoFTools::extract_locally_relevant_dofs(vel_dof);

    std::vector<TrilinosWrappers::MPI::Vector> vel_vecs(dim);
    std::vector<const TrilinosWrappers::MPI::Vector*> vel_ptrs(dim);
    for (unsigned int d = 0; d < dim; ++d)
    {
        vel_vecs[d].reinit(vel_owned, vel_relevant, mpi_comm);
        vel_vecs[d] = 0;
        vel_ptrs[d] = &vel_vecs[d];
    }

    // ── Random initial condition ──
    const unsigned int seed = 42 + Utilities::MPI::this_mpi_process(mpi_comm);
    RandomPhaseIC<dim> theta_ic(0.0, 0.05, seed);
    Functions::ZeroFunction<dim> psi_ic;
    ch.project_initial_condition(theta_ic, psi_ic);

    ch.update_ghosts();

    // ── Time stepping parameters ──
    const double eps = params.physics.epsilon;
    const double dt  = 0.1 * eps * eps;
    const unsigned int n_steps      = (requested_steps > 0)
                                        ? static_cast<unsigned int>(requested_steps)
                                        : 500;
    const unsigned int output_every = std::max(1u, n_steps / 20);

    pcout << "  ε  = " << eps << "\n"
          << "  γ  = " << params.physics.mobility << "\n"
          << "  dt = " << std::scientific << std::setprecision(2) << dt
          << " (0.1 ε²)\n"
          << "  n_steps = " << n_steps
          << " (total time = " << n_steps * dt << ")\n"
          << "  Output every " << output_every << " steps\n\n";

    // ── Output directory ──
    const std::string output_dir =
        "../cahn_hilliard_results/vtk/" +
        timestamped_filename_mpi("cahn_hilliard", "", mpi_comm);

    pcout << "  Output: " << output_dir << "\n\n";

    // ── Initial VTK (step 0) ──
    ch.write_vtu(output_dir, 0, 0.0);

    // ── Console header ──
    pcout << std::string(90, '-') << "\n"
          << std::setw(6) << "Step"
          << std::setw(12) << "time"
          << std::setw(10) << "θ_min"
          << std::setw(10) << "θ_max"
          << std::setw(12) << "mass"
          << std::setw(14) << "E_total"
          << std::setw(12) << "intf_len"
          << std::setw(10) << "wall(s)"
          << "\n"
          << std::string(90, '-') << "\n";

    // ── Time loop ──
    Timer timer;
    double current_time = 0.0;

    for (unsigned int step = 1; step <= n_steps; ++step)
    {
        current_time += dt;
        timer.restart();

        ch.assemble(ch.get_theta_relevant(),
                    vel_ptrs, vel_dof,
                    dt, current_time);
        ch.solve();
        ch.update_ghosts();

        timer.stop();

        auto diag = ch.compute_diagnostics();

        pcout << std::setw(6) << step
              << std::setw(12) << std::scientific << std::setprecision(2) << current_time
              << std::setw(10) << std::fixed << std::setprecision(4) << diag.theta_min
              << std::setw(10) << diag.theta_max
              << std::setw(12) << std::scientific << std::setprecision(3) << diag.mass_integral
              << std::setw(14) << (diag.E_gradient + diag.E_bulk)
              << std::setw(12) << diag.interface_length
              << std::setw(10) << std::fixed << std::setprecision(3) << timer.wall_time()
              << "\n";

        if (step % output_every == 0 || step == n_steps)
            ch.write_vtu(output_dir, step, current_time);
    }

    pcout << std::string(90, '-') << "\n\n";

    auto diag = ch.compute_diagnostics();
    const double E_total = diag.E_gradient + diag.E_bulk;

    pcout << "=== Final State ===\n"
          << "  θ range    = [" << diag.theta_min << ", " << diag.theta_max << "]\n"
          << "  θ mean     = " << diag.theta_mean << "\n"
          << "  Mass ∫θ    = " << diag.mass_integral << "\n"
          << "  E_total    = " << E_total << "\n"
          << "  Intf len   = " << diag.interface_length << "\n"
          << "  VTK files: " << output_dir << "\n\n";
}


// ============================================================================
// MMS convergence study (same as test file, uses production facade)
// ============================================================================
struct CHMMSResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;
    double theta_L2 = 0.0, theta_H1 = 0.0, theta_Linf = 0.0;
    double psi_L2 = 0.0, psi_Linf = 0.0;
    double total_time = 0.0;
};

template <int dim>
CHMMSResult run_ch_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    CHMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    const double t_init  = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;

    double L[dim];
    L[0] = params.domain.x_max - params.domain.x_min;
    if constexpr (dim >= 2) L[1] = params.domain.y_max - params.domain.y_min;
    if constexpr (dim >= 3) L[2] = 1.0;

    // Mesh
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1, p2;
    std::vector<unsigned int> subdivisions(dim);
    p1[0] = params.domain.x_min;
    p2[0] = params.domain.x_max;
    subdivisions[0] = params.domain.initial_cells_x;
    if constexpr (dim >= 2)
    {
        p1[1] = params.domain.y_min;
        p2[1] = params.domain.y_max;
        subdivisions[1] = params.domain.initial_cells_y;
    }
    if constexpr (dim >= 3)
    {
        p1[2] = 0.0;
        p2[2] = L[2];
        subdivisions[2] = params.domain.initial_cells_x;
    }

    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // Setup facade
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);
    ch.setup();
    result.n_dofs = ch.get_theta_dof_handler().n_dofs() * 2;

    // Zero velocity
    FE_Q<dim> fe_vel(params.fe.degree_velocity);
    DoFHandler<dim> vel_dof(triangulation);
    vel_dof.distribute_dofs(fe_vel);
    IndexSet vel_owned = vel_dof.locally_owned_dofs();
    IndexSet vel_relevant = DoFTools::extract_locally_relevant_dofs(vel_dof);

    std::vector<TrilinosWrappers::MPI::Vector> vel_vecs(dim);
    std::vector<const TrilinosWrappers::MPI::Vector*> vel_ptrs(dim);
    for (unsigned int d = 0; d < dim; ++d)
    {
        vel_vecs[d].reinit(vel_owned, vel_relevant, mpi_comm);
        vel_vecs[d] = 0;
        vel_ptrs[d] = &vel_vecs[d];
    }

    // MMS source terms
    CHSourceTheta<dim> src_theta(params.physics.mobility, dt, L);
    CHSourcePsi<dim>   src_psi(params.physics.epsilon, dt, L);

    ch.set_mms_source(
        [&](const Point<dim>& p, double t) -> double {
            src_theta.set_time(t);
            return src_theta.value(p);
        },
        [&](const Point<dim>& p, double t) -> double {
            src_psi.set_time(t);
            return src_psi.value(p);
        });

    // Initial condition
    CHMMSInitialTheta<dim> theta_ic(t_init, L);
    CHMMSInitialPsi<dim>   psi_ic(t_init, L);
    ch.project_initial_condition(theta_ic, psi_ic);

    CHMMSBoundaryTheta<dim> theta_bc(L);
    CHMMSBoundaryPsi<dim>   psi_bc(L);
    theta_bc.set_time(t_init);
    psi_bc.set_time(t_init);
    ch.apply_dirichlet_boundary(theta_bc, psi_bc);
    ch.update_ghosts();

    // Time stepping
    double current_time = t_init;
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;
        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);
        ch.apply_dirichlet_boundary(theta_bc, psi_bc);

        ch.assemble(ch.get_theta_relevant(), vel_ptrs, vel_dof, dt, current_time);
        ch.solve();
        ch.update_ghosts();
    }

    // Errors
    CHMMSErrors errors = compute_ch_mms_errors<dim>(
        ch.get_theta_dof_handler(), ch.get_psi_dof_handler(),
        ch.get_theta_relevant(), ch.get_psi_relevant(),
        current_time, L, mpi_comm);

    result.theta_L2   = errors.theta_L2;
    result.theta_H1   = errors.theta_H1;
    result.theta_Linf = errors.theta_Linf;
    result.psi_L2     = errors.psi_L2;
    result.psi_Linf   = errors.psi_Linf;

    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}


// ============================================================================
// Temporal convergence test
// ============================================================================
template <int dim>
bool run_ch_temporal_convergence(const Parameters& params, MPI_Comm mpi_comm)
{
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int ref = 4;
    const std::vector<unsigned int> steps_list = {5, 10, 20, 40, 80};

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   CH Temporal Convergence (ref=" << ref << ")\n";
        std::cout << "============================================================\n";
    }

    std::vector<double> dts, theta_L2_errors;

    for (unsigned int n_steps : steps_list)
    {
        if (rank == 0)
            std::cout << "  n_steps=" << n_steps << "... " << std::flush;

        CHMMSResult r = run_ch_mms_single<dim>(ref, params, n_steps, mpi_comm);
        const double dt = 0.1 / n_steps;
        dts.push_back(dt);
        theta_L2_errors.push_back(r.theta_L2);

        if (rank == 0)
            std::cout << "theta_L2=" << std::scientific << std::setprecision(2)
                      << r.theta_L2 << "\n";
    }

    if (rank == 0)
    {
        std::cout << "\n  dt-convergence rates:\n";
        std::cout << std::setw(12) << "dt" << std::setw(14) << "theta_L2"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(34, '-') << "\n";

        bool pass = false;
        for (size_t i = 0; i < dts.size(); ++i)
        {
            double rate = 0.0;
            if (i > 0 && theta_L2_errors[i] > 1e-15 && theta_L2_errors[i-1] > 1e-15)
                rate = std::log(theta_L2_errors[i-1] / theta_L2_errors[i])
                     / std::log(dts[i-1] / dts[i]);

            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << dts[i]
                      << std::setw(14) << theta_L2_errors[i]
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
                (params.run.steps > 0) ? static_cast<unsigned int>(params.run.steps) : 10;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Cahn-Hilliard MMS Convergence (CG Q"
                          << params.fe.degree_phase << ")\n";
                std::cout << "============================================================\n";
                std::cout << "  Refinements: ";
                for (auto r : params.run.refs) std::cout << r << " ";
                std::cout << "\n  Time steps: " << n_time_steps << "\n\n";
            }

            std::vector<CHMMSResult> results;
            for (unsigned int ref : params.run.refs)
            {
                if (rank == 0)
                    std::cout << "  Refinement " << ref << "... " << std::flush;

                CHMMSResult r = run_ch_mms_single<dim>(ref, params, n_time_steps, mpi_comm);
                results.push_back(r);

                if (rank == 0)
                    std::cout << "theta_L2=" << std::scientific << std::setprecision(2)
                              << r.theta_L2 << ", theta_H1=" << r.theta_H1
                              << ", time=" << std::fixed << std::setprecision(1)
                              << r.total_time << "s\n";
            }

            // Compute and check rates
            bool pass = false;
            if (rank == 0 && results.size() >= 2)
            {
                std::cout << "\n  Convergence rates:\n";
                for (size_t i = 1; i < results.size(); ++i)
                {
                    double log_h = std::log(results[i-1].h / results[i].h);
                    double L2_rate = std::log(results[i-1].theta_L2 / results[i].theta_L2) / log_h;
                    double H1_rate = std::log(results[i-1].theta_H1 / results[i].theta_H1) / log_h;
                    std::cout << "    ref " << results[i].refinement
                              << ": theta_L2_rate=" << std::fixed << std::setprecision(2) << L2_rate
                              << ", theta_H1_rate=" << H1_rate << "\n";
                    if (i == results.size() - 1)
                        pass = (L2_rate >= params.fe.degree_phase + 1 - 0.3)
                            && (H1_rate >= params.fe.degree_phase - 0.3);
                }

                if (pass) std::cout << "  [PASS]\n";
                else      std::cout << "  [FAIL]\n";
            }
            return pass ? 0 : 1;
        }
        // ================================================================
        // Mode: 2d — spinodal decomposition
        // ================================================================
        else if (mode == "2d")
        {
            // Default parameters for spinodal
            params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
            params.domain.y_min = 0.0;  params.domain.y_max = 0.6;
            params.domain.initial_cells_x = 10;
            params.domain.initial_cells_y = 6;
            if (params.mesh.initial_refinement == 0)
                params.mesh.initial_refinement = 5;
            if (params.physics.epsilon < 1e-10)
                params.physics.epsilon = 0.01;
            params.physics.mobility = 1.0;

            run_spinodal_decomposition<2>(params, mpi_comm, params.run.steps);
            return 0;
        }
        // ================================================================
        // Mode: 3d — spinodal decomposition 3D
        // ================================================================
        else if (mode == "3d")
        {
            params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
            params.domain.y_min = 0.0;  params.domain.y_max = 0.6;
            params.domain.initial_cells_x = 6;
            params.domain.initial_cells_y = 4;
            if (params.mesh.initial_refinement == 0)
                params.mesh.initial_refinement = 0;
            params.physics.epsilon = 0.1;
            params.physics.mobility = 1.0;

            run_spinodal_decomposition<3>(params, mpi_comm, params.run.steps);
            return 0;
        }
        // ================================================================
        // Mode: temporal — temporal convergence study
        // ================================================================
        else if (mode == "temporal")
        {
            bool pass = run_ch_temporal_convergence<dim>(params, mpi_comm);
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
