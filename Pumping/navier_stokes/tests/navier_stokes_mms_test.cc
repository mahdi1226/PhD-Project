// ============================================================================
// navier_stokes/tests/navier_stokes_mms_test.cc - Steady Stokes MMS Test
//
// PAPER EQUATION 42e (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//   Standalone steady Stokes: -ν_eff Δu + ∇p = f,  ∇·u = 0
//
// Tests: NavierStokesSubsystem facade using PRODUCTION code paths
//   setup() → assemble(steady Stokes) → solve() → compute errors
//
// Expected convergence (CG Q2 / DG P1):
//   U_L2: O(h³) — rate ≈ 3.0
//   U_H1: O(h²) — rate ≈ 2.0
//   p_L2: O(h²) — rate ≈ 2.0
//
// Usage:
//   mpirun -np 2 ./test_navier_stokes_mms
//   mpirun -np 4 ./test_navier_stokes_mms --refs 2 3 4 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "navier_stokes/tests/navier_stokes_mms.h"
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

struct MMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;
    double U_L2 = 0.0;
    double U_H1 = 0.0;
    double p_L2 = 0.0;
    double U_Linf = 0.0;
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

    const double mms_time = 1.0;

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
    params.enable_mms = true;

    FHDMesh::create_mesh<dim>(triangulation, params);

    result.h = dealii::GridTools::minimal_cell_diameter(triangulation);

    // ================================================================
    // Create and setup NS subsystem (PRODUCTION CODE)
    // ================================================================
    NavierStokesSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs() * 2
                  + ns.get_p_dof_handler().n_dofs();

    // ================================================================
    // Inject MMS source: steady Stokes (f = -ν_eff Δu* + ∇p*)
    // ================================================================
    ns.set_mms_source(compute_ns_mms_source_stokes<dim>);

    // ================================================================
    // Assemble steady Stokes (large dt → no mass term, no convection)
    // ================================================================
    dealii::TrilinosWrappers::MPI::Vector empty_ux, empty_uy, empty_w;

    ns.assemble(empty_ux, empty_uy,
                /*dt=*/1e30, mms_time,
                /*include_convection=*/false,
                empty_w, ns.get_ux_dof_handler());

    // ================================================================
    // Solve (PRODUCTION CODE)
    // ================================================================
    SolverInfo info = ns.solve();
    result.iterations = info.iterations;

    // ================================================================
    // Compute errors
    // ================================================================
    ns.update_ghosts();

    NSMMSErrors errors = compute_ns_mms_errors<dim>(
        ns.get_ux_dof_handler(),
        ns.get_ux_relevant(),
        ns.get_uy_relevant(),
        ns.get_p_dof_handler(),
        ns.get_p_relevant(),
        mms_time, mpi_comm);

    result.U_L2 = errors.U_L2;
    result.U_H1 = errors.U_H1;
    result.p_L2 = errors.p_L2;
    result.U_Linf = errors.U_Linf;

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    return result;
}

int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    // Parse refinement levels and options
    std::vector<unsigned int> refinements = {2, 3, 4, 5, 6};
    bool use_block_schur = false;

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
        }
        else if (std::strcmp(argv[i], "--block-schur") == 0)
        {
            use_block_schur = true;
        }
    }

    Parameters params;
    params.setup_mms_validation();

    if (use_block_schur)
    {
        params.solvers.navier_stokes.use_iterative = true;
        params.solvers.navier_stokes.preconditioner =
            LinearSolverParams::Preconditioner::BlockSchur;
    }

    const unsigned int vel_degree = params.fe.degree_velocity;

    pcout << "\n"
          << "================================================================\n"
          << "  NAVIER-STOKES MMS CONVERGENCE (Nochetto Eq. 42e, Steady Stokes)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  FE space:       CG Q" << vel_degree << " / DG P"
          << params.fe.degree_pressure << "\n"
          << "  Expected rates: U_L2 = " << vel_degree + 1
          << ", U_H1 = " << vel_degree
          << ", p_L2 = " << params.fe.degree_pressure + 1 << "\n"
          << "  Domain:         [0,1]^2\n"
          << "  Standalone:     no coupling, steady Stokes\n"
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
              << ", U_L2=" << std::scientific << std::setprecision(2) << r.U_L2
              << ", U_H1=" << r.U_H1
              << ", p_L2=" << r.p_L2
              << ", wall=" << std::fixed << std::setprecision(2)
              << r.walltime << "s\n";
    }

    // Compute rates
    std::vector<double> UL2_rates, UH1_rates, pL2_rates;
    for (size_t i = 1; i < results.size(); ++i)
    {
        auto rate = [](double ef, double ec, double hf, double hc) -> double {
            if (ef < 1e-15 || ec < 1e-15 || hf >= hc) return 0.0;
            return std::log(ec / ef) / std::log(hc / hf);
        };
        UL2_rates.push_back(rate(results[i].U_L2, results[i-1].U_L2,
                                  results[i].h, results[i-1].h));
        UH1_rates.push_back(rate(results[i].U_H1, results[i-1].U_H1,
                                  results[i].h, results[i-1].h));
        pL2_rates.push_back(rate(results[i].p_L2, results[i-1].p_L2,
                                  results[i].h, results[i-1].h));
    }

    // Print table
    pcout << "\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "U_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "U_H1"
          << std::setw(8)  << "rate"
          << std::setw(12) << "p_L2"
          << std::setw(8)  << "rate"
          << "\n"
          << std::string(85, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        pcout << std::left
              << std::setw(5)  << r.refinement
              << std::setw(10) << r.n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.h
              << std::setw(12) << r.U_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? UL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.U_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? UH1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.p_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? pL2_rates[i-1] : 0.0)
              << "\n";
    }

    // Pass/fail
    const double tolerance = 0.3;
    bool pass = false;

    if (!UL2_rates.empty())
    {
        const double final_UL2 = UL2_rates.back();
        const double final_UH1 = UH1_rates.back();
        const double final_pL2 = pL2_rates.back();

        const double expected_UL2 = vel_degree + 1;
        const double expected_UH1 = vel_degree;
        const double expected_pL2 = params.fe.degree_pressure + 1;

        pass = (final_UL2 >= expected_UL2 - tolerance) &&
               (final_UH1 >= expected_UH1 - tolerance) &&
               (final_pL2 >= expected_pL2 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    U_L2: " << std::fixed << std::setprecision(2)
              << final_UL2 << "  (expected " << expected_UL2 << ")\n"
              << "    U_H1: " << final_UH1 << "  (expected " << expected_UH1 << ")\n"
              << "    p_L2: " << final_pL2 << "  (expected " << expected_pL2 << ")\n"
              << "\n"
              << "  STATUS: " << (pass ? "PASS" : "FAIL") << "\n"
              << "================================================================\n\n";
    }

    // Write CSV
    if (rank == 0)
    {
        std::system("mkdir -p " SOURCE_DIR "/Results/mms");

        std::ofstream csv(SOURCE_DIR "/Results/mms/navier_stokes_mms.csv");
        csv << "refinement,n_dofs,h,U_L2,U_L2_rate,U_H1,U_H1_rate,"
            << "p_L2,p_L2_rate,iterations,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.U_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? UL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.U_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? UH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.p_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? pL2_rates[i-1] : 0.0) << ","
                << r.iterations << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }
        pcout << "  Results written to Results/mms/navier_stokes_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
