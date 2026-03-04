// ============================================================================
// cahn_hilliard/tests/cahn_hilliard_mms_test.cc - Standalone MMS Convergence
//
// SPLIT CH (Eyre's convex-concave, backward Euler, no convection):
//
//   Single backward Euler step from t_old = 1.0, dt = h^2
//   Initialize phi_old to interpolation of phi*(t_old)
//   Solve monolithic (phi, mu) system with MMS source
//   Compare with analytical (phi*(t_new), mu*(t_new))
//
// Expected convergence (CG Q2):
//   phi_L2: rate ~= 3.0,  phi_H1: rate ~= 2.0
//   mu_L2:  rate ~= 3.0,  mu_H1:  rate ~= 2.0
//
// Usage:
//   mpirun -np 4 ./test_cahn_hilliard_mms
//   mpirun -np 4 ./test_cahn_hilliard_mms --refs 2 3 4 5 6
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "cahn_hilliard/tests/cahn_hilliard_mms.h"
#include "mesh/mesh.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
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
    double phi_L2 = 0.0, phi_H1 = 0.0;
    double mu_L2 = 0.0, mu_H1 = 0.0;
    int iterations = 0;
    double walltime = 0.0;
};

MMSResult run_single(unsigned int refinement,
                     const Parameters& base_params,
                     MPI_Comm mpi_comm)
{
    MMSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ================================================================
    // Create distributed mesh
    // ================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    Parameters params = base_params;
    params.mesh.initial_refinement = refinement;

    FHDMesh::create_mesh<dim>(triangulation, params);

    result.h = dealii::GridTools::minimal_cell_diameter(triangulation);

    // ================================================================
    // Time stepping: single step with dt = h^2
    // ================================================================
    const double t_old = 1.0;
    const double dt = result.h * result.h;
    const double t_new = t_old + dt;

    params.time.dt = dt;
    params.enable_mms = true;

    // ================================================================
    // Create and setup CH subsystem
    // ================================================================
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);
    ch.setup();

    result.n_dofs = ch.get_dof_handler().n_dofs();

    // ================================================================
    // Initialize old solution: phi = phi*(t_old), mu = 0
    // ================================================================
    CHExactSolution<dim> ic(t_old);
    ch.initialize(ic);
    ch.save_old_solution();

    // ================================================================
    // Inject MMS source
    // ================================================================
    const double epsilon = params.cahn_hilliard_params.epsilon;
    const double gamma   = params.cahn_hilliard_params.gamma;

    ch.set_mms_source(
        [epsilon, gamma, t_new](
            const dealii::Point<dim>& p,
            double /*t*/, double dt_local,
            double phi_old_disc) -> std::pair<double, double>
        {
            return compute_ch_mms_source<dim>(
                p, t_new, dt_local, phi_old_disc, epsilon, gamma);
        });

    // ================================================================
    // Assemble and solve (no convection)
    // ================================================================
    ch.assemble(ch.get_old_relevant(), dt);

    SolverInfo info = ch.solve();
    result.iterations = info.iterations;

    // ================================================================
    // Compute errors
    // ================================================================
    ch.update_ghosts();

    CHMMSErrors errors = compute_ch_mms_errors<dim>(
        ch.get_dof_handler(),
        ch.get_relevant(),
        t_new, mpi_comm);

    result.phi_L2 = errors.phi_L2;
    result.phi_H1 = errors.phi_H1;
    result.mu_L2  = errors.mu_L2;
    result.mu_H1  = errors.mu_H1;

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    return result;
}

// ============================================================================
// Main: convergence study
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    // ================================================================
    // Parse refinement levels
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
    params.setup_mms_validation();

    // CH-specific MMS parameters
    params.cahn_hilliard_params.epsilon = 1.0;  // moderate for MMS
    params.cahn_hilliard_params.gamma   = 1.0;
    params.enable_cahn_hilliard = true;

    const unsigned int fe_degree = params.fe.degree_cahn_hilliard;
    const double expected_L2 = fe_degree + 1;  // Q2 -> 3.0
    const double expected_H1 = fe_degree;      // Q2 -> 2.0

    // ================================================================
    // Run convergence study
    // ================================================================
    pcout << "\n"
          << "================================================================\n"
          << "  CAHN-HILLIARD MMS CONVERGENCE STUDY (Standalone)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  FE degree:      Q" << fe_degree << " x 2 (phi + mu)\n"
          << "  Expected rates: L2 = " << expected_L2
          << ", H1 = " << expected_H1 << "\n"
          << "  Domain:         [0,1]^2\n"
          << "  epsilon:        " << params.cahn_hilliard_params.epsilon << "\n"
          << "  gamma:          " << params.cahn_hilliard_params.gamma << "\n"
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
              << ", phi_L2=" << std::scientific << std::setprecision(2) << r.phi_L2
              << ", mu_L2=" << r.mu_L2
              << ", its=" << r.iterations
              << ", wall=" << std::fixed << std::setprecision(2)
              << r.walltime << "s\n";
    }

    // ================================================================
    // Compute convergence rates
    // ================================================================
    auto rate = [](double e_f, double e_c, double h_f, double h_c) -> double {
        if (e_f < 1e-15 || e_c < 1e-15 || h_f >= h_c) return 0.0;
        return std::log(e_c / e_f) / std::log(h_c / h_f);
    };

    std::vector<double> phi_L2_rates, phi_H1_rates, mu_L2_rates, mu_H1_rates;
    for (size_t i = 1; i < results.size(); ++i)
    {
        phi_L2_rates.push_back(rate(results[i].phi_L2, results[i-1].phi_L2,
                                     results[i].h, results[i-1].h));
        phi_H1_rates.push_back(rate(results[i].phi_H1, results[i-1].phi_H1,
                                     results[i].h, results[i-1].h));
        mu_L2_rates.push_back(rate(results[i].mu_L2, results[i-1].mu_L2,
                                    results[i].h, results[i-1].h));
        mu_H1_rates.push_back(rate(results[i].mu_H1, results[i-1].mu_H1,
                                    results[i].h, results[i-1].h));
    }

    // ================================================================
    // Print table
    // ================================================================
    pcout << "\n"
          << std::left
          << std::setw(5) << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "phi_L2"
          << std::setw(8) << "rate"
          << std::setw(12) << "phi_H1"
          << std::setw(8) << "rate"
          << std::setw(12) << "mu_L2"
          << std::setw(8) << "rate"
          << std::setw(12) << "mu_H1"
          << std::setw(8) << "rate"
          << std::setw(8) << "iters"
          << "\n"
          << std::string(113, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        pcout << std::left
              << std::setw(5)  << r.refinement
              << std::setw(10) << r.n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.h
              << std::setw(12) << r.phi_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (i > 0 ? phi_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.phi_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (i > 0 ? phi_H1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.mu_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (i > 0 ? mu_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.mu_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8) << (i > 0 ? mu_H1_rates[i-1] : 0.0)
              << std::setw(8) << r.iterations
              << "\n";
    }

    // ================================================================
    // Summary + Pass/fail check
    // ================================================================
    const double tolerance = 0.3;
    bool pass = false;

    if (!phi_L2_rates.empty())
    {
        const double final_phi_L2 = phi_L2_rates.back();
        const double final_phi_H1 = phi_H1_rates.back();
        const double final_mu_L2  = mu_L2_rates.back();
        const double final_mu_H1  = mu_H1_rates.back();

        pass = (final_phi_L2 >= expected_L2 - tolerance) &&
               (final_phi_H1 >= expected_H1 - tolerance) &&
               (final_mu_L2  >= expected_L2 - tolerance) &&
               (final_mu_H1  >= expected_H1 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  MPI ranks:       " << n_ranks << "\n"
              << "  FE degree:       Q" << fe_degree << " x 2 (phi + mu)\n"
              << "  Refinements:     " << results.front().refinement
              << " -> " << results.back().refinement
              << " (" << results.front().n_dofs << " -> "
              << results.back().n_dofs << " DoFs)\n"
              << "\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    phi_L2: " << std::fixed << std::setprecision(2)
              << final_phi_L2 << "  (expected " << expected_L2 << ")\n"
              << "    phi_H1: " << final_phi_H1 << "  (expected " << expected_H1 << ")\n"
              << "    mu_L2:  " << final_mu_L2  << "  (expected " << expected_L2 << ")\n"
              << "    mu_H1:  " << final_mu_H1  << "  (expected " << expected_H1 << ")\n"
              << "\n"
              << "  STATUS: " << (pass ? "PASS" : "FAIL") << "\n"
              << "================================================================\n\n";
    }

    // ================================================================
    // Write CSV
    // ================================================================
    if (rank == 0)
    {
        std::system("mkdir -p " SOURCE_DIR "/Results/mms");

        std::ofstream csv(SOURCE_DIR "/Results/mms/cahn_hilliard_mms.csv");
        csv << "refinement,n_dofs,h,phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,"
            << "mu_L2,mu_L2_rate,mu_H1,mu_H1_rate,iterations,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.phi_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phi_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.phi_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phi_H1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.mu_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? mu_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.mu_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? mu_H1_rates[i-1] : 0.0) << ","
                << r.iterations << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }
        pcout << "  Results written to Results/mms/cahn_hilliard_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
