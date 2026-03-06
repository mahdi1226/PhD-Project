// ============================================================================
// navier_stokes/navier_stokes_main.cc — NS Standalone Driver
//
// Pressure-correction projection method (Zhang Algorithm 3.1):
//   Step 2: Velocity predictor (assemble + solve ux, uy separately)
//   Step 3: Pressure Poisson   (assemble + solve p)
//   Step 4: Velocity correction (algebraic, consistent mass CG)
//
// Modes:
//   mms       MMS spatial convergence (2D), refs 2-6
//   2d        Single 2D run with VTK output
//   3d        3D NS (not yet implemented)
//   temporal  Temporal convergence study (2D), sweep dt
//
// Usage:
//   mpirun -np 4 ./navier_stokes_main --mode mms
//   mpirun -np 4 ./navier_stokes_main --mode 2d --refinement 4
//   mpirun -np 4 ./navier_stokes_main --mode 3d
//   mpirun -np 4 ./navier_stokes_main --mode temporal
//   mpirun -np 4 ./navier_stokes_main --ref 2 3 4 5
//   mpirun -np 4 ./navier_stokes_main --steps 20
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "navier_stokes/tests/navier_stokes_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <string>
#include <functional>

// ============================================================================
// Result structures
// ============================================================================
struct NSMMSResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;
    double ux_L2  = 0.0, ux_H1  = 0.0;
    double uy_L2  = 0.0, uy_H1  = 0.0;
    double p_L2   = 0.0, div_L2 = 0.0;
    double total_time = 0.0;
};

struct NSMMSConvergenceResult
{
    std::vector<NSMMSResult> results;
    std::vector<double> ux_L2_rates, ux_H1_rates;
    std::vector<double> uy_L2_rates, uy_H1_rates;
    std::vector<double> p_L2_rates,  div_L2_rates;
    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;
    unsigned int n_time_steps = 10;

    void compute_rates()
    {
        ux_L2_rates.clear();  ux_H1_rates.clear();
        uy_L2_rates.clear();  uy_H1_rates.clear();
        p_L2_rates.clear();   div_L2_rates.clear();

        for (size_t i = 1; i < results.size(); ++i)
        {
            const double log_h = std::log(results[i-1].h / results[i].h);
            auto rate = [&](double e_coarse, double e_fine) {
                return (e_coarse > 1e-15 && e_fine > 1e-15)
                    ? std::log(e_coarse / e_fine) / log_h : 0.0;
            };
            ux_L2_rates.push_back(rate(results[i-1].ux_L2, results[i].ux_L2));
            ux_H1_rates.push_back(rate(results[i-1].ux_H1, results[i].ux_H1));
            uy_L2_rates.push_back(rate(results[i-1].uy_L2, results[i].uy_L2));
            uy_H1_rates.push_back(rate(results[i-1].uy_H1, results[i].uy_H1));
            p_L2_rates.push_back(rate(results[i-1].p_L2,   results[i].p_L2));
            div_L2_rates.push_back(rate(results[i-1].div_L2, results[i].div_L2));
        }
    }

    void print() const
    {
        std::cout << "\n--- NS MMS Convergence (Q"
                  << degree_velocity << "/Q" << degree_pressure
                  << ", Projection Method) ---\n";
        std::cout << std::left
                  << std::setw(5)  << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "ux_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ux_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "uy_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "uy_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "p_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "div_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(10) << "wall(s)"
                  << "\n";
        std::cout << std::string(155, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            std::cout << std::left << std::setw(5) << r.refinement
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.h
                      << std::setw(12) << r.ux_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.ux_H1
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_H1_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.uy_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? uy_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.uy_H1
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? uy_H1_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.p_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? p_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.div_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? div_L2_rates[i-1] : 0.0)
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.total_time
                      << "\n";
        }
    }

    void write_csv(const std::string& filepath) const
    {
        std::ofstream f(filepath);
        f << "refinement,h,n_dofs,"
          << "ux_L2,ux_L2_rate,ux_H1,ux_H1_rate,"
          << "uy_L2,uy_L2_rate,uy_H1,uy_H1_rate,"
          << "p_L2,p_L2_rate,div_L2,div_L2_rate,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            f << r.refinement << ","
              << std::scientific << std::setprecision(6) << r.h << ","
              << r.n_dofs << ","
              << r.ux_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? ux_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.ux_H1 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? ux_H1_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.uy_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? uy_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.uy_H1 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? uy_H1_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.p_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? p_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.div_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? div_L2_rates[i-1] : 0.0) << ","
              << std::fixed << std::setprecision(4) << r.total_time << "\n";
        }
        std::cout << "  CSV written: " << filepath << "\n";
    }

    bool passes(double tol = 0.3) const
    {
        if (ux_H1_rates.empty()) return false;
        const double expected_H1 = static_cast<double>(degree_velocity);
        const double expected_p  = static_cast<double>(degree_pressure + 1);
        return (ux_H1_rates.back() >= expected_H1 - tol)
            && (uy_H1_rates.back() >= expected_H1 - tol)
            && (p_L2_rates.back()  >= expected_p  - tol);
    }
};


// ============================================================================
// Single refinement test
//
// Pressure-correction projection method:
//   1. Assemble velocity predictor (ux, uy matrices + RHS)
//   2. Solve velocity predictor (CG+AMG for ux, uy)
//   3. Assemble pressure Poisson
//   4. Solve pressure Poisson (CG+AMG for p)
//   5. Velocity correction (algebraic, mass-lumped)
//   6. Advance time (swap u→u_old, p→p_old)
// ============================================================================
template <int dim>
NSMMSResult run_ns_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    NSMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    const double t_init  = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double nu = params.physics.nu_water;
    const double Ly = params.domain.y_max - params.domain.y_min;

    // ========================================================================
    // Create distributed mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1, p2;
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

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                       subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Create facade and setup
    // ========================================================================
    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    // ========================================================================
    // Initialize velocity to exact solution at t_init
    // ========================================================================
    NSMMSInitialUx<dim> ic_ux(t_init, Ly);
    NSMMSInitialUy<dim> ic_uy(t_init, Ly);
    ns.initialize_velocity(ic_ux, ic_uy);

    // ========================================================================
    // Time stepping loop — Projection method (Zhang Algorithm 3.1)
    // ========================================================================
    double current_time = t_init;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        const double t_old = current_time;
        current_time += dt;

        // MMS source: captures t_old in closure
        std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
            body_force = [&](const dealii::Point<dim>& p, double t) {
                return NSMMS::source_phase_D<dim>(p, t, t_old, nu, Ly);
            };

        // Step 2: Assemble + solve velocity predictor
        ns.assemble_stokes(dt, nu,
                           /*include_time_derivative=*/ true,
                           /*include_convection=*/ true,
                           &body_force,
                           current_time);
        ns.solve_velocity();

        // Step 3: Assemble + solve pressure Poisson
        ns.assemble_pressure_poisson(dt);
        ns.solve_pressure();

        // Step 4: Velocity correction (algebraic, mass-lumped)
        ns.velocity_correction(dt);

        // Advance time: swap u→u_old, p→p_old
        ns.advance_time();
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    ns.update_ghosts();

    // Write final solution to VTK for visualization
    ns.write_vtu("../navier_stokes_results/vtk", n_time_steps, current_time);

    NSMMSErrors errors = compute_ns_mms_errors<dim>(
        ns, current_time, Ly, mpi_comm);

    result.ux_L2  = errors.ux_L2;   result.ux_H1  = errors.ux_H1;
    result.uy_L2  = errors.uy_L2;   result.uy_H1  = errors.uy_H1;
    result.p_L2   = errors.p_L2;    result.div_L2 = errors.div_U_L2;

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
//
// Fix spatial ref=4, sweep dt with increasing n_time_steps.
// Measure MMS error at t_final, compute dt convergence rate.
// Expected: O(dt^1) for backward Euler.
// ============================================================================
template <int dim>
bool run_ns_temporal_convergence(
    const Parameters& params,
    MPI_Comm mpi_comm)
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int ref = 4;
    const std::vector<unsigned int> steps_list = {5, 10, 20, 40, 80};

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   NS Temporal Convergence (ref=" << ref << ")\n";
        std::cout << "============================================================\n";
    }

    std::vector<double> dts, vel_L2_errors;

    for (unsigned int n_steps : steps_list)
    {
        const double dt = 0.1 / n_steps;  // t_final - t_init = 0.1

        if (rank == 0)
            std::cout << "  n_steps=" << n_steps << " (dt=" << std::scientific
                      << std::setprecision(2) << dt << ")... " << std::flush;

        NSMMSResult r = run_ns_mms_single<dim>(ref, params, n_steps, mpi_comm);

        dts.push_back(dt);
        vel_L2_errors.push_back(std::sqrt(r.ux_L2 * r.ux_L2 + r.uy_L2 * r.uy_L2));

        if (rank == 0)
            std::cout << "vel_L2=" << std::scientific << std::setprecision(2)
                      << vel_L2_errors.back() << "\n";
    }

    // Compute rates
    if (rank == 0)
    {
        std::cout << "\n  dt-convergence rates:\n";
        std::cout << std::setw(12) << "dt" << std::setw(14) << "vel_L2"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(34, '-') << "\n";

        bool pass = false;
        for (size_t i = 0; i < dts.size(); ++i)
        {
            double rate = 0.0;
            if (i > 0 && vel_L2_errors[i] > 1e-15 && vel_L2_errors[i-1] > 1e-15)
                rate = std::log(vel_L2_errors[i-1] / vel_L2_errors[i])
                     / std::log(dts[i-1] / dts[i]);

            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << dts[i]
                      << std::setw(14) << vel_L2_errors[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << rate << "\n";

            if (i == dts.size() - 1)
                pass = (rate >= 0.7);  // expect O(dt^1) for BDF1
        }

        std::cout << "\n  Expected: O(dt^1) for backward Euler\n";
        if (pass)
            std::cout << "  [PASS] Temporal convergence rate within tolerance!\n";
        else
            std::cout << "  [FAIL] Temporal convergence rate below expected!\n";

        return pass;
    }
    return true;
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    dealii::deallog.depth_console(0);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    try
    {
        Parameters params = Parameters::parse_command_line(argc, argv);
        const std::string& mode = params.run.mode;
        const unsigned int n_time_steps =
            (params.run.steps > 0) ? static_cast<unsigned int>(params.run.steps) : 10;

        constexpr int dim = 2;

        // ================================================================
        // Mode: mms — spatial convergence study (2D)
        // ================================================================
        if (mode == "mms")
        {
            NSMMSConvergenceResult conv;
            conv.degree_velocity = params.fe.degree_velocity;
            conv.degree_pressure = params.fe.degree_pressure;
            conv.n_time_steps = n_time_steps;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes MMS Verification (Projection Method)\n";
                std::cout << "============================================================\n";
                std::cout << "  MPI ranks:    "
                          << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
                std::cout << "  FE degrees:   velocity Q" << params.fe.degree_velocity
                          << ", pressure Q" << params.fe.degree_pressure << "\n";
                std::cout << "  nu = " << params.physics.nu_water << "\n";
                std::cout << "  Base steps:   " << n_time_steps
                          << " (×4 per ref), t in [0.1, 0.2]\n";
                std::cout << "  Refinements:  ";
                for (auto r : params.run.refs) std::cout << r << " ";
                std::cout << "\n";
                std::cout << "============================================================\n\n";
            }

            // Scale dt ∝ h² across refinement levels so that the O(dt)
            // splitting error from the projection method doesn't dominate
            // the spatial error.  Steps quadruple per refinement level.
            const unsigned int ref_base = params.run.refs.front();

            for (unsigned int ref : params.run.refs)
            {
                // steps = N_base * 4^(ref - ref_base)
                unsigned int steps_for_ref = n_time_steps;
                for (unsigned int r = ref_base; r < ref; ++r)
                    steps_for_ref *= 4;

                if (rank == 0)
                    std::cout << "  Refinement " << ref
                              << " (" << steps_for_ref << " steps)... " << std::flush;

                NSMMSResult r = run_ns_mms_single<dim>(ref, params, steps_for_ref, mpi_comm);
                conv.results.push_back(r);

                if (rank == 0)
                {
                    std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                              << ", ux_H1=" << r.ux_H1
                              << ", uy_L2=" << r.uy_L2
                              << ", p_L2=" << r.p_L2
                              << ", div=" << r.div_L2
                              << ", time=" << std::fixed << std::setprecision(1)
                              << r.total_time << "s\n";
                }
            }

            conv.compute_rates();

            if (rank == 0)
            {
                conv.print();

                const std::string out_dir = "../navier_stokes_results/mms";
                std::system(("mkdir -p " + out_dir).c_str());

                const std::string csv_name = timestamped_filename(
                    "ns_mms_convergence", ".csv");
                conv.write_csv(out_dir + "/" + csv_name);

                double total_wall = 0.0;
                for (const auto& r : conv.results) total_wall += r.total_time;

                std::cout << "\nExpected: vel_H1 ~ O(h^" << params.fe.degree_velocity
                          << "), p_L2 ~ O(h^" << (params.fe.degree_pressure + 1) << ")"
                          << "  |  Total wall time: " << std::fixed << std::setprecision(1)
                          << total_wall << "s\n";
                std::cout << "  (First-order time stepping may limit velocity L2 rate)\n";
                std::cout << "  (Projection method introduces O(dt) splitting error in pressure)\n";

                if (conv.passes())
                    std::cout << "[PASS] Convergence rates within tolerance!\n";
                else
                    std::cout << "[FAIL] Convergence rates below expected!\n";
            }

            return conv.passes() ? 0 : 1;
        }
        // ================================================================
        // Mode: 2d — single run with VTK output
        // ================================================================
        else if (mode == "2d")
        {
            const unsigned int ref = params.mesh.initial_refinement;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes 2D — Single MMS run with VTK\n";
                std::cout << "  Refinement: " << ref
                          << ", time steps: " << n_time_steps << "\n";
                std::cout << "============================================================\n\n";
            }

            NSMMSResult r = run_ns_mms_single<dim>(ref, params, n_time_steps, mpi_comm);

            if (rank == 0)
            {
                std::cout << "  ux_L2=" << std::scientific << std::setprecision(3) << r.ux_L2
                          << ", ux_H1=" << r.ux_H1
                          << ", p_L2=" << r.p_L2
                          << ", wall=" << std::fixed << std::setprecision(1)
                          << r.total_time << "s\n";
                std::cout << "  VTK output: ../navier_stokes_results/vtk/\n";
            }
            return 0;
        }
        // ================================================================
        // Mode: 3d — not yet implemented
        // ================================================================
        else if (mode == "3d")
        {
            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes 3D — Not yet implemented\n";
                std::cout << "============================================================\n";
                std::cout << "  3D NS requires uz_dof_handler + 3 more CG+AMG solves.\n";
                std::cout << "  This will be implemented in a future update.\n\n";
            }
            return 0;
        }
        // ================================================================
        // Mode: temporal — temporal convergence study
        // ================================================================
        else if (mode == "temporal")
        {
            bool pass = run_ns_temporal_convergence<dim>(params, mpi_comm);
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
