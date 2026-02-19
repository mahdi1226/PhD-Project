// ============================================================================
// poisson/tests/poisson_mms_test.cc - Standalone MMS Convergence Test
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   (∇φ, ∇X) = (f_mms, X)    (standalone, M = 0, h_a = 0)
//
// Tests: PoissonSubsystem facade using PRODUCTION code paths
//   setup() → assemble_rhs() → solve() → compute errors
//
// Expected convergence (Q1):
//   L2: O(h²) — rate ≈ 2.0
//   H1: O(h¹) — rate ≈ 1.0
//
// Usage:
//   mpirun -np 2 ./test_poisson_mms
//   mpirun -np 4 ./test_poisson_mms --refs 2 3 4 5 6
// ============================================================================

#include "poisson/poisson.h"
#include "poisson/tests/poisson_mms.h"
#include "utilities/timestamp.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_dgq.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>


constexpr int dim = 2;

// ============================================================================
// Single refinement level test
// ============================================================================
struct MMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;
    double L2 = 0.0;
    double H1 = 0.0;
    double Linf = 0.0;
    int iterations = 0;
    double walltime = 0.0;
};

MMSResult run_single(unsigned int refinement,
                     const Parameters& params,
                     MPI_Comm mpi_comm)
{
    MMSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double mms_time = 1.0;

    // ================================================================
    // Create distributed mesh
    // ================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y
    };

    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // Compute min h (global)
    double local_h_min = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h_min = std::min(local_h_min, cell->diameter());
    MPI_Allreduce(&local_h_min, &result.h, 1,
                  MPI_DOUBLE, MPI_MIN, mpi_comm);

    // ================================================================
    // Create and setup Poisson subsystem (PRODUCTION CODE)
    // ================================================================
    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.enable_magnetic = false;  // no h_a in standalone test

    PoissonSubsystem<dim> poisson(mms_params, mpi_comm, triangulation);
    poisson.setup();

    result.n_dofs = poisson.get_dof_handler().n_dofs();

    // ================================================================
    // Inject MMS source: f_mms = −Δφ_exact
    // ================================================================
    poisson.set_mms_source(
        [L_y, mms_time](const dealii::Point<dim>& p, double /*t*/) -> double {
            return compute_poisson_mms_source_standalone<dim>(p, mms_time, L_y);
        });

    // ================================================================
    // Assemble RHS with M = 0 (standalone: empty vectors)
    // ================================================================
    dealii::TrilinosWrappers::MPI::Vector empty_Mx, empty_My;
    dealii::FE_DGQ<dim> dummy_fe(0);
    dealii::DoFHandler<dim> dummy_dof(triangulation);
    dummy_dof.distribute_dofs(dummy_fe);

    poisson.assemble_rhs(empty_Mx, empty_My, dummy_dof, mms_time);

    // ================================================================
    // Solve (PRODUCTION CODE)
    // ================================================================
    SolverInfo info = poisson.solve();
    result.iterations = info.iterations;

    // ================================================================
    // Compute errors (needs ghosted solution)
    // ================================================================
    poisson.update_ghosts();

    PoissonMMSErrors errors = compute_poisson_mms_errors<dim>(
        poisson.get_dof_handler(),
        poisson.get_solution_relevant(),
        mms_time, L_y, mpi_comm);

    result.L2 = errors.L2;
    result.H1 = errors.H1;
    result.Linf = errors.Linf;

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    return result;
}

// ============================================================================
// Main: convergence study over multiple refinement levels
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    // ================================================================
    // Parse refinement levels from command line
    // ================================================================
    std::vector<unsigned int> refinements = {2, 3, 4, 5, 6};

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--refs") == 0)
        {
            refinements.clear();
            for (int j = i + 1; j < argc; ++j)
            {
                if (argv[j][0] == '-') break;
                refinements.push_back(std::stoul(argv[j]));
            }
            break;
        }
    }

    // ================================================================
    // Setup parameters
    // ================================================================
    Parameters params;
    params.domain.x_min = 0.0;
    params.domain.x_max = 1.0;
    params.domain.y_min = 0.0;
    params.domain.y_max = 0.6;
    params.domain.initial_cells_x = 10;
    params.domain.initial_cells_y = 6;
    params.fe.degree_potential = 1;  // Q1

    const unsigned int fe_degree = params.fe.degree_potential;
    const double expected_L2 = fe_degree + 1;  // 2.0 for Q1
    const double expected_H1 = fe_degree;      // 1.0 for Q1

    // ================================================================
    // Run convergence study
    // ================================================================
    pcout << "\n"
          << "================================================================\n"
          << "  POISSON MMS CONVERGENCE STUDY (Eq. 42d, Standalone)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  FE degree:      Q" << fe_degree << "\n"
          << "  Expected rates: L2 = " << expected_L2
          << ", H1 = " << expected_H1 << "\n"
          << "  Domain:         [" << params.domain.x_min << ","
          << params.domain.x_max << "] x ["
          << params.domain.y_min << "," << params.domain.y_max << "]\n"
          << "  Refinements:    ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n\n";

    std::vector<MMSResult> results;

    for (unsigned int ref : refinements)
    {
        pcout << "  Refinement " << ref << "... " << std::flush;

        MMSResult r = run_single(ref, params, mpi_comm);
        results.push_back(r);

        pcout << "DoFs=" << r.n_dofs
              << ", L2=" << std::scientific << std::setprecision(2) << r.L2
              << ", H1=" << r.H1
              << ", Linf=" << r.Linf
              << ", its=" << r.iterations
              << ", wall=" << std::fixed << std::setprecision(2)
              << r.walltime << "s\n";
    }

    // ================================================================
    // Compute and display convergence rates
    // ================================================================
    std::vector<double> L2_rates, H1_rates, Linf_rates;
    for (size_t i = 1; i < results.size(); ++i)
    {
        if (results[i-1].L2 > 1e-15 && results[i].L2 > 1e-15)
            L2_rates.push_back(
                std::log(results[i-1].L2 / results[i].L2)
                / std::log(results[i-1].h / results[i].h));
        else
            L2_rates.push_back(0.0);

        if (results[i-1].H1 > 1e-15 && results[i].H1 > 1e-15)
            H1_rates.push_back(
                std::log(results[i-1].H1 / results[i].H1)
                / std::log(results[i-1].h / results[i].h));
        else
            H1_rates.push_back(0.0);

        if (results[i-1].Linf > 1e-15 && results[i].Linf > 1e-15)
            Linf_rates.push_back(
                std::log(results[i-1].Linf / results[i].Linf)
                / std::log(results[i-1].h / results[i].h));
        else
            Linf_rates.push_back(0.0);
    }

    // ================================================================
    // Print table
    // ================================================================
    pcout << "\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "L2 error"
          << std::setw(8)  << "rate"
          << std::setw(12) << "H1 error"
          << std::setw(8)  << "rate"
          << std::setw(12) << "Linf"
          << std::setw(8)  << "rate"
          << std::setw(8)  << "iters"
          << std::setw(10) << "wall(s)"
          << "\n"
          << std::string(105, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        pcout << std::left
              << std::setw(5)  << r.refinement
              << std::setw(10) << r.n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.h
              << std::setw(12) << r.L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? H1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.Linf
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? Linf_rates[i-1] : 0.0)
              << std::setw(8)  << r.iterations
              << std::fixed << std::setprecision(3)
              << std::setw(10) << r.walltime
              << "\n";
    }

    // ================================================================
    // Summary + Pass/fail check
    // ================================================================
    const double tolerance = 0.3;
    bool pass = false;

    if (!L2_rates.empty() && !H1_rates.empty())
    {
        const double final_L2_rate = L2_rates.back();
        const double final_H1_rate = H1_rates.back();
        const double final_Linf_rate = Linf_rates.back();

        pass = (final_L2_rate >= expected_L2 - tolerance) &&
               (final_H1_rate >= expected_H1 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  MPI ranks:       " << n_ranks << "\n"
              << "  FE degree:       Q" << fe_degree << "\n"
              << "  Refinement range:" << results.front().refinement
              << " → " << results.back().refinement
              << " (" << results.front().n_dofs << " → "
              << results.back().n_dofs << " DoFs)\n"
              << "  Total walltime:  " << std::fixed << std::setprecision(3);
        double total_wall = 0.0;
        for (const auto& r : results) total_wall += r.walltime;
        pcout << total_wall << "s\n"
              << "\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    L2:   " << std::fixed << std::setprecision(2)
              << final_L2_rate << "  (expected " << expected_L2 << ")\n"
              << "    H1:   " << final_H1_rate << "  (expected " << expected_H1 << ")\n"
              << "    Linf: " << final_Linf_rate << "  (expected " << expected_L2 << ")\n"
              << "\n"
              << "  STATUS: " << (pass ? "PASS ✓" : "FAIL ✗") << "\n"
              << "================================================================\n\n";
    }

    // ================================================================
    // Write CSV to poisson_results/mms/ (relative to source, not build)
    // ================================================================
    if (rank == 0)
    {
        const std::string out_dir = "../poisson_results/mms";
        std::system(("mkdir -p " + out_dir).c_str());

        const std::string csv_name = timestamped_filename(
            "poisson_convergence", ".csv");
        const std::string csv_path = out_dir + "/" + csv_name;

        std::ofstream csv(csv_path);
        csv << "refinement,n_dofs,h,L2_error,L2_rate,H1_error,H1_rate,"
            << "Linf_error,Linf_rate,iterations,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? H1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.Linf << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? Linf_rates[i-1] : 0.0) << ","
                << r.iterations << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }
        pcout << "  Results written to " << csv_path << "\n\n";
    }

    return pass ? 0 : 1;
}