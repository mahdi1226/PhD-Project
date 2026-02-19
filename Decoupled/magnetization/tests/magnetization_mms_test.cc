// ============================================================================
// magnetization/tests/magnetization_mms_test.cc — Standalone MMS Test
//
// Self-contained parallel convergence test using the FACADE API:
//   MagnetizationSubsystem::setup()
//   MagnetizationSubsystem::set_mms_source()
//   MagnetizationSubsystem::assemble()
//   MagnetizationSubsystem::solve()
//
// STANDALONE TEST (U=0, φ=0, θ=1):
//   Mx* = t·sin(πx)·sin(πy/L_y)
//   My* = t·cos(πx)·sin(πy/L_y)
//
// Expected: DG1 → L2 rate ≈ 2.0
//
// Usage:
//   mpirun -np 2 ./test_magnetization_mms
//   mpirun -np 4 ./test_magnetization_mms --refs 2 3 4 5
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization/magnetization.h"
#include "magnetization/tests/magnetization_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <sys/stat.h>

constexpr int dim = 2;

// ============================================================================
// Single-refinement MMS result
// ============================================================================
struct MagMMSResult
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

// ============================================================================
// Run single refinement level
// ============================================================================
static MagMMSResult run_single(
    unsigned int refinement,
    const Parameters& params,
    MPI_Comm mpi_comm)
{
    MagMMSResult result;
    result.refinement = refinement;

    auto t_start_wall = std::chrono::high_resolution_clock::now();

    const double L_y    = params.domain.y_max - params.domain.y_min;
    const double t0     = 0.1;
    const double t_end  = 0.2;
    const unsigned int n_steps = 1;
    const double dt     = (t_end - t0) / n_steps;

    // ========================================================================
    // Mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y};

    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Facade
    // ========================================================================
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    mag.setup();

    result.n_dofs = 2 * mag.get_dof_handler().n_dofs();  // Mx + My

    // ========================================================================
    // Dummy CG DoFHandlers for φ, θ, U (all zero / constant)
    // ========================================================================
    dealii::FE_Q<dim> fe_CG(params.fe.degree_velocity);

    dealii::DoFHandler<dim> phi_dof_handler(triangulation);
    dealii::DoFHandler<dim> theta_dof_handler(triangulation);
    dealii::DoFHandler<dim> u_dof_handler(triangulation);

    phi_dof_handler.distribute_dofs(fe_CG);
    theta_dof_handler.distribute_dofs(fe_CG);
    u_dof_handler.distribute_dofs(fe_CG);

    auto phi_owned   = phi_dof_handler.locally_owned_dofs();
    auto phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler);
    auto theta_owned   = theta_dof_handler.locally_owned_dofs();
    auto theta_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    auto u_owned   = u_dof_handler.locally_owned_dofs();
    auto u_relevant = dealii::DoFTools::extract_locally_relevant_dofs(u_dof_handler);

    // φ=0, θ=1 (ferrofluid everywhere), U=0
    dealii::TrilinosWrappers::MPI::Vector phi_vec(phi_owned, phi_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_owned, theta_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector ux_vec(u_owned, u_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector uy_vec(u_owned, u_relevant, mpi_comm);

    phi_vec   = 0.0;
    theta_vec = 1.0;
    ux_vec    = 0.0;
    uy_vec    = 0.0;

    // ========================================================================
    // MMS source callback
    //
    // Lambda captures L_y and calls the MMS source function.
    // The assembler passes (point, t_new, t_old, tau_M, chi, H, U, div_U).
    // ========================================================================
    mag.set_mms_source(
        [L_y](const dealii::Point<dim>& point,
              double t_new, double t_old, double tau_M,
              double chi_val,
              const dealii::Tensor<1, dim>& H,
              const dealii::Tensor<1, dim>& U,
              double div_U)
              -> dealii::Tensor<1, dim>
        {
            return compute_mag_mms_source_with_transport<dim>(
                point, t_new, t_old, tau_M, chi_val, H, U, div_U, L_y);
        });

    // ========================================================================
    // Initialize Mx, My with exact solution at t=t0 via L² projection
    // ========================================================================
    {
        MagExactMx<dim> exact_Mx(t0, L_y);
        MagExactMy<dim> exact_My(t0, L_y);

        // Project exact MMS solution at t=t0 onto DG space
        mag.project_initial_condition(exact_Mx, exact_My);
    }

    // ========================================================================
    // Mesh size
    // ========================================================================
    {
        double local_min_h = std::numeric_limits<double>::max();
        for (const auto& cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                local_min_h = std::min(local_min_h, cell->diameter());
        MPI_Allreduce(&local_min_h, &result.h, 1,
                      MPI_DOUBLE, MPI_MIN, mpi_comm);
    }

    // ========================================================================
    // Time stepping
    // ========================================================================
    double current_time = t0;

    for (unsigned int step = 0; step < n_steps; ++step)
    {
        current_time += dt;

        // Ghost update before assembler reads old M
        mag.update_ghosts();

        // Ghosted old M for assembly
        const auto& Mx_old = mag.get_Mx_relevant();
        const auto& My_old = mag.get_My_relevant();

        // Assemble (matrix + RHS, U=0 so face terms vanish)
        mag.assemble(
            Mx_old, My_old,
            phi_vec, phi_dof_handler,
            theta_vec, theta_dof_handler,
            ux_vec, uy_vec, u_dof_handler,
            dt, current_time);

        // Solve Mx then My
        mag.solve();
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
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
// main — convergence study
// ============================================================================
int main(int argc, char** argv)
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int this_rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks =
        dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

    // Default refinements
    std::vector<unsigned int> refinements = {2, 3, 4, 5, 6};

    // Parse --refs
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--refs")
        {
            refinements.clear();
            for (int j = i + 1; j < argc; ++j)
            {
                try { refinements.push_back(std::stoul(argv[j])); }
                catch (...) { break; }
            }
            break;
        }
    }

    // Parameters (defaults: DG1, [0,1]², τ_M=1e-6, χ₀=3.0)
    Parameters params;

    const double expected_L2_rate = params.fe.degree_magnetization + 1;  // DG1 → 2.0

    if (this_rank == 0)
    {
        std::cout << "\n========================================"
                  << "\nMagnetization MMS Convergence Study"
                  << "\n========================================"
                  << "\n  MPI ranks:       " << n_ranks
                  << "\n  FE degree:       DG" << params.fe.degree_magnetization
                  << "\n  Expected L2:     " << expected_L2_rate
                  << "\n  Using FACADE:    MagnetizationSubsystem"
                  << "\n  Time steps:      1"
                  << "\n  t ∈ [0.1, 0.2]"
                  << "\n========================================\n"
                  << std::endl;
    }

    // ========================================================================
    // Run refinements
    // ========================================================================
    std::vector<MagMMSResult> results;

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "..." << std::flush;

        MagMMSResult r = run_single(ref, params, mpi_comm);
        results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << " M_L2=" << std::scientific << std::setprecision(2)
                      << r.M_L2 << ", time="
                      << std::fixed << std::setprecision(1)
                      << r.time_s << "s" << std::endl;
        }
    }

    // ========================================================================
    // Convergence rates + pass/fail
    // ========================================================================
    if (this_rank == 0)
    {
        // -- Console table --
        std::cout << "\n"
                  << std::left
                  << std::setw(6)  << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "M_L2"
                  << std::setw(8)  << "L2rate"
                  << std::setw(12) << "M_Linf"
                  << std::setw(8)  << "Lfrate"
                  << std::setw(12) << "Mx_L2"
                  << std::setw(12) << "My_L2"
                  << std::setw(10) << "DoFs"
                  << std::setw(10) << "time(s)"
                  << "\n" << std::string(102, '-') << "\n";

        double last_L2_rate = 0.0;
        double last_Linf_rate = 0.0;
        std::vector<double> L2_rates, Linf_rates;

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];

            double L2_rate = 0.0;
            double Linf_rate = 0.0;
            if (i > 0)
            {
                const auto& prev = results[i - 1];
                const double log_h = std::log(prev.h / r.h);
                if (prev.M_L2 > 1e-15 && r.M_L2 > 1e-15)
                    L2_rate = std::log(prev.M_L2 / r.M_L2) / log_h;
                if (prev.M_Linf > 1e-15 && r.M_Linf > 1e-15)
                    Linf_rate = std::log(prev.M_Linf / r.M_Linf) / log_h;
                last_L2_rate = L2_rate;
                last_Linf_rate = Linf_rate;
            }
            L2_rates.push_back(L2_rate);
            Linf_rates.push_back(Linf_rate);

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
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.Mx_L2
                      << std::setw(12) << r.My_L2
                      << std::fixed << std::setprecision(0)
                      << std::setw(10) << r.n_dofs
                      << std::setprecision(1)
                      << std::setw(10) << r.time_s
                      << "\n";
        }

        std::cout << std::string(102, '-') << "\n";

        const double tolerance = 0.3;
        const bool pass = (last_L2_rate >= expected_L2_rate - tolerance);

        if (pass)
            std::cout << "[PASS] L2 rate " << std::fixed << std::setprecision(2)
                      << last_L2_rate << " >= "
                      << expected_L2_rate - tolerance << " (expected "
                      << expected_L2_rate << " +/- " << tolerance << ")\n";
        else
            std::cout << "[FAIL] L2 rate " << std::fixed << std::setprecision(2)
                      << last_L2_rate << " < "
                      << expected_L2_rate - tolerance << "\n";

        std::cout << "========================================\n";

        // -- CSV output --
        const std::string csv_dir = "../magnetization_results/mms";
        mkdir("../magnetization_results", 0755);
        mkdir(csv_dir.c_str(), 0755);

        const std::string csv_filename = timestamped_filename(
            "magnetization_convergence", ".csv");
        const std::string csv_path = csv_dir + "/" + csv_filename;
        std::ofstream csv(csv_path);
        if (csv.is_open())
        {
            csv << "refinement,h,n_dofs,"
                << "M_L2,Mx_L2,My_L2,L2_rate,"
                << "M_Linf,Mx_Linf,My_Linf,Linf_rate,"
                << "time_s\n";

            for (size_t i = 0; i < results.size(); ++i)
            {
                const auto& r = results[i];
                csv << r.refinement << ","
                    << std::scientific << std::setprecision(6)
                    << r.h << ","
                    << r.n_dofs << ","
                    << r.M_L2 << ","
                    << r.Mx_L2 << ","
                    << r.My_L2 << ","
                    << std::fixed << std::setprecision(4)
                    << L2_rates[i] << ","
                    << std::scientific << std::setprecision(6)
                    << r.M_Linf << ","
                    << r.Mx_Linf << ","
                    << r.My_Linf << ","
                    << std::fixed << std::setprecision(4)
                    << Linf_rates[i] << ","
                    << std::setprecision(2)
                    << r.time_s << "\n";
            }
            csv.close();
            std::cout << "\nCSV written to: " << csv_path << "\n";
        }
        else
        {
            std::cerr << "WARNING: Could not open " << csv_path << " for writing\n";
        }
    }

    return 0;
}