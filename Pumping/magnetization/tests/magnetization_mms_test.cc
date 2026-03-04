// ============================================================================
// magnetization/tests/magnetization_mms_test.cc - Standalone MMS Convergence
//
// PAPER EQUATION 42c (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//   Standalone: U = 0, σ = 0, h_a = 0, ∇φ = 0
//   Reduces to: (1/τ + 1/𝒯)(m, z) = (1/τ)(m^{n-1}, z) + (f_mms, z)
//
// Tests: MagnetizationSubsystem facade using PRODUCTION code paths
//   setup() → project IC → assemble() → solve() → compute errors
//
// Strategy: single backward-Euler step from t_old=1.0 to t_new=1.0+dt
//   with dt = h². The MMS source cancels temporal error, so only spatial
//   discretization error remains.
//
// Expected convergence (DG Q_ℓ with ℓ=2):
//   L2: O(h^{ℓ+1}) — rate ≈ 3.0
//
// Usage:
//   mpirun -np 2 ./test_magnetization_mms
//   mpirun -np 4 ./test_magnetization_mms --refs 2 3 4 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================

#include "magnetization/magnetization.h"
#include "magnetization/tests/magnetization_mms.h"
#include "mesh/mesh.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

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
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double M_Linf = 0.0;
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
    params.enable_mms = true;

    FHDMesh::create_mesh<dim>(triangulation, params);

    result.h = dealii::GridTools::minimal_cell_diameter(triangulation);

    // Time stepping: single step from t_old to t_new with dt = h²
    const double dt = result.h * result.h;
    const double t_old = 1.0;
    const double t_new = t_old + dt;

    // ================================================================
    // Create and setup Magnetization subsystem (PRODUCTION CODE)
    // ================================================================
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    mag.setup();

    result.n_dofs = mag.get_dof_handler().n_dofs();

    // ================================================================
    // Inject MMS source (standalone: U=0, σ=0, h=0)
    // ================================================================
    mag.set_mms_source(compute_mag_mms_source_standalone<dim>);

    // ================================================================
    // Project initial condition M*(t_old) into DG space
    // ================================================================
    // We need owned vectors for projection, then copy to ghosted
    const auto& dof_handler = mag.get_dof_handler();
    dealii::IndexSet locally_owned = dof_handler.locally_owned_dofs();
    dealii::IndexSet locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    dealii::TrilinosWrappers::MPI::Vector Mx_old_owned(locally_owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector My_old_owned(locally_owned, mpi_comm);

    project_magnetization_exact<dim>(dof_handler, Mx_old_owned, My_old_owned, t_old);

    // Copy to ghosted vectors for assembly
    dealii::TrilinosWrappers::MPI::Vector Mx_old_relevant(
        locally_owned, locally_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector My_old_relevant(
        locally_owned, locally_relevant, mpi_comm);
    Mx_old_relevant = Mx_old_owned;
    My_old_relevant = My_old_owned;

    // ================================================================
    // Empty vectors for Poisson and velocity (standalone: not used)
    // ================================================================
    dealii::TrilinosWrappers::MPI::Vector empty_phi;
    dealii::FE_Q<dim> dummy_cg_fe(1);
    dealii::DoFHandler<dim> dummy_phi_dof(triangulation);
    dummy_phi_dof.distribute_dofs(dummy_cg_fe);

    dealii::TrilinosWrappers::MPI::Vector empty_ux, empty_uy;
    dealii::DoFHandler<dim> dummy_u_dof(triangulation);
    dummy_u_dof.distribute_dofs(dummy_cg_fe);

    // ================================================================
    // Assemble and solve (PRODUCTION CODE)
    // ================================================================
    mag.assemble(Mx_old_relevant, My_old_relevant,
                 empty_phi, dummy_phi_dof,
                 empty_ux, empty_uy, dummy_u_dof,
                 dt, t_new);

    SolverInfo info = mag.solve();
    result.iterations = info.iterations;

    // ================================================================
    // Compute errors at t_new
    // ================================================================
    mag.update_ghosts();

    MagnetizationMMSErrors errors = compute_mag_mms_errors<dim>(
        dof_handler,
        mag.get_Mx_relevant(),
        mag.get_My_relevant(),
        t_new, mpi_comm);

    result.Mx_L2 = errors.Mx_L2;
    result.My_L2 = errors.My_L2;
    result.M_L2 = errors.M_L2;
    result.M_Linf = errors.M_Linf;

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
    // Setup parameters (Nochetto Section 6: Ω = (0,1)², DG Q2)
    // ================================================================
    Parameters params;
    params.setup_mms_validation();

    const unsigned int fe_degree = params.fe.degree_magnetization;
    const double expected_L2 = fe_degree + 1;  // Q2 → 3.0

    // ================================================================
    // Run convergence study
    // ================================================================
    pcout << "\n"
          << "================================================================\n"
          << "  MAGNETIZATION MMS CONVERGENCE (Nochetto Eq. 42c, Standalone)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  FE space:       DG Q" << fe_degree << "\n"
          << "  Expected rate:  L2 = " << expected_L2 << "\n"
          << "  Domain:         [0,1]^2\n"
          << "  Standalone:     U = 0, sigma = 0, h_a = 0\n"
          << "  Time stepping:  1 backward-Euler step, dt = h^2\n"
          << "  Exact solution: Mx = t*sin(pi*x)*sin(pi*y)\n"
          << "                  My = t*cos(pi*x)*sin(pi*y)\n"
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
              << ", M_L2=" << std::scientific << std::setprecision(2) << r.M_L2
              << ", Linf=" << r.M_Linf
              << ", its=" << r.iterations
              << ", wall=" << std::fixed << std::setprecision(2)
              << r.walltime << "s\n";
    }

    // ================================================================
    // Compute and display convergence rates
    // ================================================================
    std::vector<double> M_L2_rates, Mx_L2_rates, My_L2_rates, Linf_rates;
    for (size_t i = 1; i < results.size(); ++i)
    {
        auto rate = [](double e_f, double e_c, double h_f, double h_c) -> double {
            if (e_f < 1e-15 || e_c < 1e-15 || h_f >= h_c) return 0.0;
            return std::log(e_c / e_f) / std::log(h_c / h_f);
        };

        M_L2_rates.push_back(rate(results[i].M_L2, results[i-1].M_L2,
                                   results[i].h, results[i-1].h));
        Mx_L2_rates.push_back(rate(results[i].Mx_L2, results[i-1].Mx_L2,
                                    results[i].h, results[i-1].h));
        My_L2_rates.push_back(rate(results[i].My_L2, results[i-1].My_L2,
                                    results[i].h, results[i-1].h));
        Linf_rates.push_back(rate(results[i].M_Linf, results[i-1].M_Linf,
                                   results[i].h, results[i-1].h));
    }

    // ================================================================
    // Print table
    // ================================================================
    pcout << "\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "M_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "Mx_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "My_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "Linf"
          << std::setw(8)  << "rate"
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
              << std::setw(12) << r.M_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? M_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.Mx_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? Mx_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.My_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? My_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.M_Linf
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? Linf_rates[i-1] : 0.0)
              << "\n";
    }

    // ================================================================
    // Summary + Pass/fail check
    // ================================================================
    const double tolerance = 0.3;
    bool pass = false;

    if (!M_L2_rates.empty())
    {
        const double final_M_L2_rate = M_L2_rates.back();

        pass = (final_M_L2_rate >= expected_L2 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  MPI ranks:       " << n_ranks << "\n"
              << "  FE space:        DG Q" << fe_degree << "\n"
              << "  Refinement range:" << results.front().refinement
              << " -> " << results.back().refinement
              << " (" << results.front().n_dofs << " -> "
              << results.back().n_dofs << " DoFs)\n"
              << "\n"
              << "  Asymptotic rate (finest pair):\n"
              << "    M_L2:  " << std::fixed << std::setprecision(2)
              << final_M_L2_rate << "  (expected " << expected_L2 << ")\n"
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

        std::ofstream csv(SOURCE_DIR "/Results/mms/magnetization_mms.csv");
        csv << "refinement,n_dofs,h,M_L2,M_L2_rate,Mx_L2,Mx_L2_rate,"
            << "My_L2,My_L2_rate,Linf,Linf_rate,iterations,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.M_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? M_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.Mx_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? Mx_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.My_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? My_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.M_Linf << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? Linf_rates[i-1] : 0.0) << ","
                << r.iterations << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }
        pcout << "  Results written to Results/mms/magnetization_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
