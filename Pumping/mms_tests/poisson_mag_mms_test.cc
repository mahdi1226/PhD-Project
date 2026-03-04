// ============================================================================
// mms_tests/poisson_mag_mms_test.cc — Coupled Poisson↔Magnetization MMS
//
// Verifies bidirectional coupling via Picard iteration with under-relaxation:
//   Poisson (Eq. 42d):  (∇φ^k, ∇X) = (h_a − M^k, ∇X) + (f_φ, X)
//   Magnetization (Eq. 42c):
//     (1/τ + 1/𝒯)(M^k, z) = (1/τ)(M^{n-1}, z) + (κ₀/𝒯)(h^k, z) + (f_M, z)
//
// Under-relaxation (ω ∈ (0,1]):
//   M_relaxed^k = ω·M_raw^k + (1−ω)·M_relaxed^{k-1}
//   Poisson sees M_relaxed, not M_raw → damps the M→φ→H→M feedback
//
// Configuration: U = 0, h_a = 0, σ = 0
//
// Usage:
//   mpirun -np 2 ./test_poisson_mag_mms
//   mpirun -np 4 ./test_poisson_mag_mms --refs 2 3 4 5 --steps 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================

#include "mms_tests/poisson_mag_mms.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "mesh/mesh.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>

constexpr int dim = 2;

// ============================================================================
// Single refinement level
// ============================================================================
static PoissonMagMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    PoissonMagMMSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------------
    // Time stepping
    // ----------------------------------------------------------------
    const double t_start = 1.0;
    const double t_end = 1.1;
    const double dt = (t_end - t_start) / n_time_steps;

    pcout << "\n  [ref=" << refinement << "] dt=" << std::scientific
          << std::setprecision(3) << dt
          << ", tau_M=" << params.physics.T_relax
          << ", kappa_0=" << params.physics.kappa_0 << "\n";

    // ----------------------------------------------------------------
    // 1. Shared triangulation
    // ----------------------------------------------------------------
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Parameters local_params = params;
    local_params.mesh.initial_refinement = refinement;
    local_params.enable_mms = true;

    FHDMesh::create_mesh<dim>(triangulation, local_params);

    result.h = GridTools::minimal_cell_diameter(triangulation);

    // ----------------------------------------------------------------
    // 2. Setup subsystems
    // ----------------------------------------------------------------
    PoissonSubsystem<dim> poisson(local_params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(local_params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs = mag.get_dof_handler().n_dofs();
    result.n_dofs = phi_dofs + 2 * M_dofs;

    pcout << "  DoFs: phi=" << phi_dofs << " M=" << M_dofs
          << " (2x" << M_dofs << ") total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. Dummy fields: U = 0 (no flow)
    // ----------------------------------------------------------------
    FE_Q<dim> dummy_cg_fe(1);

    DoFHandler<dim> u_dof(triangulation);
    u_dof.distribute_dofs(dummy_cg_fe);
    IndexSet u_owned = u_dof.locally_owned_dofs();
    IndexSet u_relevant = DoFTools::extract_locally_relevant_dofs(u_dof);

    TrilinosWrappers::MPI::Vector ux_vec(u_owned, u_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_vec(u_owned, u_relevant, mpi_comm);
    ux_vec = 0;
    uy_vec = 0;

    // ----------------------------------------------------------------
    // 4. MMS sources
    // ----------------------------------------------------------------
    poisson.set_mms_source(
        [](const Point<dim>& pt, double time) -> double
        {
            return compute_poisson_mms_source_coupled<dim>(pt, time);
        });

    mag.set_mms_source(compute_mag_mms_source_coupled<dim>);

    // ----------------------------------------------------------------
    // 5. Initial conditions at t_start
    // ----------------------------------------------------------------
    {
        const auto& dof_handler = mag.get_dof_handler();
        IndexSet M_owned = dof_handler.locally_owned_dofs();

        TrilinosWrappers::MPI::Vector Mx_init(M_owned, mpi_comm);
        TrilinosWrappers::MPI::Vector My_init(M_owned, mpi_comm);

        project_magnetization_exact<dim>(dof_handler, Mx_init, My_init, t_start);

        // Copy into the subsystem's internal solution (via assemble + set)
        // We need owned → relevant copy via the subsystem's solve path,
        // but there's no set_solution. Instead, do one trivial solve:
        //   Assemble with M_old = M*(t_start), φ = 0, then Poisson to get φ_init
        IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
        TrilinosWrappers::MPI::Vector Mx_init_rel(M_owned, M_relevant, mpi_comm);
        TrilinosWrappers::MPI::Vector My_init_rel(M_owned, M_relevant, mpi_comm);
        Mx_init_rel = Mx_init;
        My_init_rel = My_init;

        // Initial Poisson solve consistent with M*(t_start)
        poisson.assemble_rhs(Mx_init_rel, My_init_rel,
                             mag.get_dof_handler(), t_start);
        poisson.solve();
        poisson.update_ghosts();

        // Initial magnetization solve to populate internal state
        mag.assemble(Mx_init_rel, My_init_rel,
                     poisson.get_solution_relevant(), poisson.get_dof_handler(),
                     ux_vec, uy_vec, u_dof,
                     dt, t_start + dt);
        mag.solve();
        mag.update_ghosts();
    }

    // Re-project the IC since the solve overwrote the solution.
    // For time stepping, we'll track M_old externally.

    // ----------------------------------------------------------------
    // 6. Workspace vectors
    // ----------------------------------------------------------------
    IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
    IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(
        mag.get_dof_handler());

    // M^{n-1}: snapshot at start of each time step (ghosted)
    TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_comm);

    // M_relaxed: under-relaxed M fed to Poisson (ghosted)
    TrilinosWrappers::MPI::Vector Mx_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed(M_owned, M_relevant, mpi_comm);

    // Owned temporaries for blending
    TrilinosWrappers::MPI::Vector Mx_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_comm);

    // Project true initial condition
    TrilinosWrappers::MPI::Vector Mx_ic(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_ic(M_owned, mpi_comm);
    project_magnetization_exact<dim>(mag.get_dof_handler(), Mx_ic, My_ic, t_start);

    Mx_old = Mx_ic;
    My_old = My_ic;
    Mx_relaxed = Mx_ic;
    My_relaxed = My_ic;

    // ----------------------------------------------------------------
    // 7. Time stepping with under-relaxed Picard iteration
    // ----------------------------------------------------------------
    const unsigned int max_picard = 50;
    const double picard_tol = 1e-10;
    const double omega = 0.35;

    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Reset relaxed M to M^{n-1} at start of each step
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Step A: Poisson → φ^k using RELAXED M
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Step B: Magnetization → M_raw^k using φ^k
            mag.assemble(Mx_old, My_old,
                         poisson.get_solution_relevant(),
                         poisson.get_dof_handler(),
                         ux_vec, uy_vec, u_dof,
                         dt, current_time);
            mag.solve();
            mag.update_ghosts();

            // Step C: Under-relax M
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

            // M_relaxed = ω·M_raw + (1−ω)·M_prev
            Mx_relaxed_owned = mag.get_Mx_solution();
            Mx_relaxed_owned *= omega;
            Mx_relaxed_owned.add(1.0 - omega, Mx_prev);

            My_relaxed_owned = mag.get_My_solution();
            My_relaxed_owned *= omega;
            My_relaxed_owned.add(1.0 - omega, My_prev);

            Mx_relaxed = Mx_relaxed_owned;
            My_relaxed = My_relaxed_owned;

            // Step D: Convergence check
            TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_comm);
            TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_comm);
            Mx_diff = Mx_relaxed_owned;
            Mx_diff -= Mx_prev;
            My_diff = My_relaxed_owned;
            My_diff -= My_prev;

            const double change = Mx_diff.l2_norm() + My_diff.l2_norm();
            const double norm = Mx_relaxed_owned.l2_norm()
                              + My_relaxed_owned.l2_norm() + 1e-14;
            const double residual = change / norm;

            if (residual < picard_tol)
            {
                result.picard_iters = k + 1;
                break;
            }

            if (k == max_picard - 1)
            {
                result.picard_iters = max_picard;
                pcout << "    step " << step
                      << ": WARNING Picard did not converge, res="
                      << std::scientific << std::setprecision(2)
                      << residual << "\n";
            }
        }

        // Advance M_old for next time step using relaxed solution
        Mx_old = Mx_relaxed;
        My_old = My_relaxed;
    }

    // ----------------------------------------------------------------
    // 8. Final Poisson solve consistent with final M
    // ----------------------------------------------------------------
    poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                         mag.get_dof_handler(), current_time);
    poisson.solve();
    poisson.update_ghosts();

    // ----------------------------------------------------------------
    // 9. Compute errors at final time
    // ----------------------------------------------------------------
    {
        PoissonMMSErrors phi_err = compute_poisson_mms_errors<dim>(
            poisson.get_dof_handler(),
            poisson.get_solution_relevant(),
            current_time, mpi_comm);

        result.phi_L2 = phi_err.L2;
        result.phi_H1 = phi_err.H1;
        result.phi_Linf = phi_err.Linf;
    }

    {
        MagnetizationMMSErrors mag_err = compute_mag_mms_errors<dim>(
            mag.get_dof_handler(),
            Mx_relaxed, My_relaxed,
            current_time, mpi_comm);

        result.Mx_L2 = mag_err.Mx_L2;
        result.My_L2 = mag_err.My_L2;
        result.M_L2 = mag_err.M_L2;
        result.M_Linf = mag_err.M_Linf;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results: phi_L2=" << std::scientific << std::setprecision(2)
          << result.phi_L2 << "  phi_H1=" << result.phi_H1
          << "  M_L2=" << result.M_L2 << "  M_Linf=" << result.M_Linf
          << "  picard=" << result.picard_iters
          << "  time=" << std::fixed << std::setprecision(1)
          << result.walltime << "s\n";

    return result;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    // Defaults
    std::vector<unsigned int> refinements = {2, 3, 4, 5};
    unsigned int n_time_steps = 5;

    // Parse args
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
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            n_time_steps = std::stoul(argv[++i]);
        }
    }

    // MMS parameters
    Parameters params;
    params.setup_mms_validation();

    // Override: use_simplified_model = false so Poisson is actually solved
    // and H = h_a + ∇φ is used in magnetization assembly
    params.use_simplified_model = false;

    const unsigned int phi_deg = params.fe.degree_potential;
    const unsigned int M_deg = params.fe.degree_magnetization;
    const double expected_phi_L2 = phi_deg + 1;
    const double expected_phi_H1 = phi_deg;
    const double expected_M_L2 = M_deg + 1;

    pcout << "\n"
          << "================================================================\n"
          << "  COUPLED POISSON + MAGNETIZATION MMS (Picard)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Time steps:     " << n_time_steps << "\n"
          << "  Picard:         max=50  tol=1e-10  omega=0.35\n"
          << "  FE:             phi=CG Q" << phi_deg
          << "  M=DG Q" << M_deg << "\n"
          << "  Physics:        tau_M=" << params.physics.T_relax
          << "  kappa_0=" << params.physics.kappa_0 << "\n"
          << "  Expected:       phi_L2=" << expected_phi_L2
          << "  phi_H1=" << expected_phi_H1
          << "  M_L2=" << expected_M_L2 << "\n"
          << "  Refs:           ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n";

    // ================================================================
    // Run convergence study
    // ================================================================
    std::vector<PoissonMagMMSResult> results;

    for (unsigned int ref : refinements)
    {
        pcout << "\n  Refinement " << ref << "...\n";
        results.push_back(
            run_single_level(ref, params, n_time_steps, mpi_comm));
    }

    // ================================================================
    // Compute rates
    // ================================================================
    auto rate = [](double e_f, double e_c, double h_f, double h_c) -> double {
        if (e_f < 1e-15 || e_c < 1e-15 || h_f >= h_c) return 0.0;
        return std::log(e_c / e_f) / std::log(h_c / h_f);
    };

    std::vector<double> phi_L2_rates, phi_H1_rates;
    std::vector<double> M_L2_rates;

    for (size_t i = 1; i < results.size(); ++i)
    {
        phi_L2_rates.push_back(rate(results[i].phi_L2, results[i-1].phi_L2,
                                     results[i].h, results[i-1].h));
        phi_H1_rates.push_back(rate(results[i].phi_H1, results[i-1].phi_H1,
                                     results[i].h, results[i-1].h));
        M_L2_rates.push_back(rate(results[i].M_L2, results[i-1].M_L2,
                                   results[i].h, results[i-1].h));
    }

    // ================================================================
    // Print Poisson table
    // ================================================================
    pcout << "\n--- Poisson (φ) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "phi_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "phi_H1"
          << std::setw(8)  << "rate"
          << "\n"
          << std::string(67, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::setw(10) << results[i].n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].phi_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phi_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].phi_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phi_H1_rates[i-1] : 0.0)
              << "\n";
    }

    // ================================================================
    // Print Magnetization table
    // ================================================================
    pcout << "\n--- Magnetization (M) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "M_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "Mx_L2"
          << std::setw(12) << "My_L2"
          << std::setw(12) << "M_Linf"
          << std::setw(8)  << "picard"
          << std::setw(10) << "time(s)"
          << "\n"
          << std::string(91, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].M_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? M_L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].Mx_L2
              << std::setw(12) << results[i].My_L2
              << std::setw(12) << results[i].M_Linf
              << std::fixed << std::setprecision(0)
              << std::setw(8)  << static_cast<double>(results[i].picard_iters)
              << std::setprecision(1)
              << std::setw(10) << results[i].walltime
              << "\n";
    }

    // ================================================================
    // Pass/fail
    // ================================================================
    const double tolerance = 0.3;
    bool pass = false;

    if (!phi_L2_rates.empty())
    {
        const double final_phi_L2 = phi_L2_rates.back();
        const double final_phi_H1 = phi_H1_rates.back();
        const double final_M_L2 = M_L2_rates.back();

        pass = (final_phi_L2 >= expected_phi_L2 - tolerance)
            && (final_phi_H1 >= expected_phi_H1 - tolerance)
            && (final_M_L2 >= expected_M_L2 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    phi_L2: " << std::fixed << std::setprecision(2)
              << final_phi_L2 << "  (expected " << expected_phi_L2 << ")\n"
              << "    phi_H1: " << final_phi_H1
              << "  (expected " << expected_phi_H1 << ")\n"
              << "    M_L2:   " << final_M_L2
              << "  (expected " << expected_M_L2 << ")\n"
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

        std::ofstream csv(SOURCE_DIR "/Results/mms/poisson_mag_coupled_mms.csv");
        csv << "refinement,n_dofs,h,"
            << "phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,"
            << "M_L2,M_L2_rate,Mx_L2,My_L2,M_Linf,"
            << "picard_iters,walltime\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.phi_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phi_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.phi_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phi_H1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.M_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? M_L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.Mx_L2 << "," << r.My_L2 << "," << r.M_Linf << ","
                << r.picard_iters << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }

        pcout << "  Results written to Results/mms/poisson_mag_coupled_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
