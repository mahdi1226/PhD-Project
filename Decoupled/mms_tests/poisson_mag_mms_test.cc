// ============================================================================
// mms_tests/poisson_mag_mms_test.cc — Coupled Poisson↔Magnetization MMS
//
// Verifies bidirectional coupling via Picard iteration with under-relaxation:
//   Poisson (Eq. 42d):  (∇φ^k, ∇X) = (h_a − M^k, ∇X)
//   Magnetization (Eq. 42c/56-57):
//     (1/τ + 1/τ_M)(M^k, Z) + B_h^m(U, M^k, Z)
//       = (χ(θ)H^k/τ_M, Z) + (M^{n-1}/τ, Z)
//
// Under-relaxation (ω ∈ (0,1]):
//   M_relaxed^k = ω·M_raw^k + (1−ω)·M_relaxed^{k-1}
//   Poisson sees M_relaxed, not M_raw → damps the M→φ→H→M feedback
//
// Configuration: θ = +1, U = 0, h_a = 0
//
// Usage:
//   mpirun -np 4 ./test_poisson_mag_mms [--refs 2 3 4 5] [--steps 10]
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms_tests/poisson_mag_mms.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

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
    // Domain and time
    // ----------------------------------------------------------------
    const double L_y     = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end   = 0.2;
    const double dt      = (t_end - t_start) / n_time_steps;

    pcout << "\n  [ref=" << refinement << "] dt=" << dt
          << ", tau_M=" << params.physics.tau_M
          << ", chi_0=" << params.physics.chi_0 << "\n";

    // ----------------------------------------------------------------
    // 1. Shared triangulation
    // ----------------------------------------------------------------
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        {params.domain.initial_cells_x, params.domain.initial_cells_y},
        Point<dim>(params.domain.x_min, params.domain.y_min),
        Point<dim>(params.domain.x_max, params.domain.y_max));

    triangulation.refine_global(refinement);
    result.h = 1.0 / (params.domain.initial_cells_x * std::pow(2.0, refinement));

    // ----------------------------------------------------------------
    // 2. Setup subsystems
    // ----------------------------------------------------------------
    PoissonSubsystem<dim>        poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim>  mag(params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs   = mag.get_dof_handler().n_dofs();
    result.n_dofs = phi_dofs + 2 * M_dofs;

    pcout << "  DoFs: phi=" << phi_dofs << " M=" << M_dofs
          << " (2x" << M_dofs << ") total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. Dummy fields: theta = +1 (ferrofluid), U = 0 (no flow)
    // ----------------------------------------------------------------
    FE_Q<dim> fe_dummy(1);

    DoFHandler<dim> theta_dof(triangulation);
    theta_dof.distribute_dofs(fe_dummy);
    IndexSet theta_owned    = theta_dof.locally_owned_dofs();
    IndexSet theta_relevant = DoFTools::extract_locally_relevant_dofs(theta_dof);

    DoFHandler<dim> u_dof(triangulation);
    u_dof.distribute_dofs(fe_dummy);
    IndexSet u_owned    = u_dof.locally_owned_dofs();
    IndexSet u_relevant = DoFTools::extract_locally_relevant_dofs(u_dof);

    // theta = +1
    TrilinosWrappers::MPI::Vector theta_owned_vec(theta_owned, mpi_comm);
    theta_owned_vec = 1.0;
    TrilinosWrappers::MPI::Vector theta_vec(theta_owned, theta_relevant, mpi_comm);
    theta_vec = theta_owned_vec;

    // U = 0
    TrilinosWrappers::MPI::Vector ux_vec(u_owned, u_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_vec(u_owned, u_relevant, mpi_comm);
    ux_vec = 0;
    uy_vec = 0;

    // ----------------------------------------------------------------
    // 4. MMS sources
    // ----------------------------------------------------------------
    poisson.set_mms_source(
        [L_y](const Point<dim>& pt, double time) -> double
        {
            return compute_poisson_mms_source_coupled<dim>(pt, time, L_y);
        });

    mag.set_mms_source(
        [L_y](const Point<dim>& pt,
              double t_new, double t_old,
              double tau_M, double chi_val,
              const Tensor<1, dim>& H,
              const Tensor<1, dim>& U,
              double div_U) -> Tensor<1, dim>
        {
            return compute_mag_mms_source_with_transport<dim>(
                pt, t_new, t_old, tau_M, chi_val, H, U, div_U, L_y);
        });

    // ----------------------------------------------------------------
    // 5. Initial conditions at t_start
    // ----------------------------------------------------------------
    {
        MagExactMx<dim> exact_Mx(t_start, L_y);
        MagExactMy<dim> exact_My(t_start, L_y);
        mag.project_initial_condition(exact_Mx, exact_My);
        mag.update_ghosts();
    }

    // Initial Poisson solve consistent with M*(t_start)
    poisson.assemble_rhs(mag.get_Mx_relevant(), mag.get_My_relevant(),
                         mag.get_dof_handler(), t_start);
    poisson.solve();
    poisson.update_ghosts();

    // ----------------------------------------------------------------
    // 6. Index sets and workspace vectors
    // ----------------------------------------------------------------
    IndexSet M_owned    = mag.get_dof_handler().locally_owned_dofs();
    IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(mag.get_dof_handler());

    // M^{n-1}: snapshot at start of each time step (ghosted, fixed within step)
    TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_comm);

    // M_relaxed: under-relaxed M fed to Poisson (ghosted)
    TrilinosWrappers::MPI::Vector Mx_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed(M_owned, M_relevant, mpi_comm);
    Mx_relaxed = mag.get_Mx_relevant();
    My_relaxed = mag.get_My_relevant();

    // Owned temporaries for blending and convergence check
    TrilinosWrappers::MPI::Vector Mx_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_comm);

    // ----------------------------------------------------------------
    // 7. Time stepping with under-relaxed Picard iteration
    //
    // Under-relaxation parameter ω ∈ (0,1]:
    //   M_relaxed^k = ω·M_raw^k + (1−ω)·M_relaxed^{k-1}
    //
    // Poisson sees M_relaxed (damped), Magnetization sees raw φ.
    // This stabilizes the M→φ→H→M feedback for large χ₀.
    // ----------------------------------------------------------------
    const unsigned int max_picard = 50;
    const double picard_tol       = 1e-10;
    const double omega             = 0.35;

    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Snapshot M^{n-1}
        Mx_old = mag.get_Mx_relevant();
        My_old = mag.get_My_relevant();

        // Reset relaxed M to M^{n-1} at start of each time step
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
            if (k == 0)
            {
                mag.assemble(
                    Mx_old, My_old,
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    theta_vec, theta_dof,
                    ux_vec, uy_vec, u_dof,
                    dt, current_time);
            }
            else
            {
                mag.assemble_rhs_only(
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    theta_vec, theta_dof,
                    Mx_old, My_old,
                    dt, current_time);
            }

            mag.solve();
            mag.update_ghosts();

            // Step C: Under-relax
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

            Mx_relaxed_owned  = mag.get_Mx_solution();
            Mx_relaxed_owned *= omega;
            Mx_relaxed_owned.add(1.0 - omega, Mx_prev);

            My_relaxed_owned  = mag.get_My_solution();
            My_relaxed_owned *= omega;
            My_relaxed_owned.add(1.0 - omega, My_prev);

            Mx_relaxed = Mx_relaxed_owned;
            My_relaxed = My_relaxed_owned;

            // Step D: Convergence check on relaxed M
            TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_comm);
            TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_comm);
            Mx_diff  = Mx_relaxed_owned;
            Mx_diff -= Mx_prev;
            My_diff  = My_relaxed_owned;
            My_diff -= My_prev;

            const double change = Mx_diff.l2_norm() + My_diff.l2_norm();
            const double norm   = Mx_relaxed_owned.l2_norm()
                                + My_relaxed_owned.l2_norm() + 1e-14;
            const double residual = change / norm;

            if (residual < picard_tol)
            {
                result.picard_iters = k + 1;
                pcout << "    step " << step << ": Picard converged, iter="
                      << k + 1 << ", res=" << std::scientific
                      << std::setprecision(2) << residual << "\n";
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
    }

    // ----------------------------------------------------------------
    // 8. Compute errors at final time
    // ----------------------------------------------------------------
    {
        PoissonMMSErrors phi_err = compute_poisson_mms_errors<dim>(
            poisson.get_dof_handler(),
            poisson.get_solution_relevant(),
            current_time, L_y, mpi_comm);

        result.phi_L2 = phi_err.L2;
        result.phi_H1 = phi_err.H1;
    }

    {
        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            mag.get_dof_handler(),
            mag.get_Mx_relevant(),
            mag.get_My_relevant(),
            current_time, L_y, mpi_comm);

        result.Mx_L2 = mag_err.Mx_L2;
        result.My_L2 = mag_err.My_L2;
        result.M_L2  = mag_err.M_L2;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.time_s = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results: phi_L2=" << std::scientific << std::setprecision(2)
          << result.phi_L2 << "  phi_H1=" << result.phi_H1
          << "  M_L2=" << result.M_L2
          << "  time=" << std::fixed << std::setprecision(1)
          << result.time_s << "s\n";

    return result;
}


// ============================================================================
// Convergence study across refinement levels
// ============================================================================
PoissonMagMMSConvergenceResult run_poisson_mag_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Local copy with MMS overrides
    Parameters p = params;
    p.enable_mms = true;
    p.dipoles.intensity_max = 0.0;
    p.dipoles.positions.clear();

    PoissonMagMMSConvergenceResult result;

    const unsigned int deg = p.fe.degree_potential;
    result.expected_phi_L2_rate = static_cast<double>(deg + 1);
    result.expected_phi_H1_rate = static_cast<double>(deg);
    result.expected_M_L2_rate   = 2.0;

    if (rank == 0)
    {
        std::cout << "\n========================================================\n";
        std::cout << "  Coupled Poisson + Magnetization MMS (Picard)\n";
        std::cout << "========================================================\n";
        std::cout << "  MPI ranks:  "
                  << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Picard:     max=50  tol=1e-10  omega=0.35\n";
        std::cout << "  FE:         phi=Q" << deg
                  << "  M=DG-Q" << p.fe.degree_magnetization << "\n";
        std::cout << "  Physics:    tau_M=" << p.physics.tau_M
                  << "  chi_0=" << p.physics.chi_0 << "\n";
        std::cout << "  Expected:   phi_L2=" << result.expected_phi_L2_rate
                  << "  phi_H1=" << result.expected_phi_H1_rate
                  << "  M_L2=" << result.expected_M_L2_rate << "\n";
        std::cout << "  Refs:      ";
        for (auto r : refinements) std::cout << " " << r;
        std::cout << "\n";
    }

    for (unsigned int ref : refinements)
        result.results.push_back(
            run_single_level(ref, p, n_time_steps, mpi_comm));

    result.compute_rates();
    return result;
}


// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    // Defaults
    std::vector<unsigned int> refinements = {2, 3, 4, 5};
    unsigned int n_time_steps = 10;

    // Parse args
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--refs" && i + 1 < argc)
        {
            refinements.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                refinements.push_back(std::stoi(argv[++i]));
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            n_time_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            if (rank == 0)
                std::cout << "Usage: mpirun -np N " << argv[0]
                          << " [--refs 2 3 4 5] [--steps 10]\n";
            return 0;
        }
    }

    Parameters params;

    try
    {
        auto result = run_poisson_mag_mms(
            refinements, params, n_time_steps, MPI_COMM_WORLD);

        if (rank == 0)
        {
            result.print();

            std::cout << "\n========================================================\n";
            if (result.passes())
                std::cout << "  [PASS] All convergence rates within tolerance.\n";
            else
                std::cout << "  [FAIL] Some convergence rates below expected.\n";
            std::cout << "========================================================\n";

            result.write_csv("poisson_mag_mms_rates.csv");
        }

        return result.passes() ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}