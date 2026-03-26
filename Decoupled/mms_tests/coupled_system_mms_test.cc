// ============================================================================
// mms_tests/coupled_system_mms_test.cc — Full Coupled System MMS Test
//
// Verifies the complete decoupled algorithm:
//   Step 1: CH  → θ^n, ψ^n  (using U^{n-1})
//   Step 2: NS  → U^n, p^n  (using θ^n, θ^{n-1}, ψ^n, φ^{n-1}, M^{n-1})
//   Step 3: Poisson/Mag → φ^n, M^n  (using U^n, θ^n, Picard iteration)
//
// Tests spatial convergence: h-refinement at fixed dt.
// Splitting error O(dt) limits convergence at fine meshes.
//
// Usage:
//   mpirun -np 4 ./test_coupled_system_mms [--refs 2 3 4] [--steps 10]
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021)
// ============================================================================

#include "mms_tests/coupled_system_mms.h"

#include "cahn_hilliard/cahn_hilliard.h"
#include "navier_stokes/navier_stokes.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"

#include "utilities/amr.h"

#include "cahn_hilliard/tests/cahn_hilliard_mms.h"
#include "navier_stokes/tests/navier_stokes_mms.h"
#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
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
static CoupledMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    bool use_amr,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    CoupledMMSResult result;
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
          << "  n_steps=" << n_time_steps << "\n";

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
    // 2. Setup all 4 subsystems
    // ----------------------------------------------------------------
    CahnHilliardSubsystem<dim>  ch(params, mpi_comm, triangulation);
    NSSubsystem<dim>            ns(params, mpi_comm, triangulation);
    PoissonSubsystem<dim>       poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);

    ch.setup();
    ns.setup();
    poisson.setup();
    mag.setup();

    const unsigned int total_dofs =
        ch.get_theta_dof_handler().n_dofs()
        + ch.get_psi_dof_handler().n_dofs()
        + ns.get_ux_dof_handler().n_dofs()
        + ns.get_uy_dof_handler().n_dofs()
        + ns.get_p_dof_handler().n_dofs()
        + poisson.get_dof_handler().n_dofs()
        + 2 * mag.get_dof_handler().n_dofs();
    result.n_dofs = total_dofs;

    pcout << "  Total DoFs: " << total_dofs << "\n";

    // ----------------------------------------------------------------
    // 2b. AMR parameters (if enabled)
    // ----------------------------------------------------------------
    Parameters amr_params = params;
    const unsigned int amr_interval = 3;
    if (use_amr)
    {
        amr_params.mesh.use_amr = true;
        amr_params.mesh.amr_interval = amr_interval;
        amr_params.mesh.amr_upper_fraction = 0.3;
        amr_params.mesh.amr_lower_fraction = 0.1;
        amr_params.mesh.amr_max_level = refinement + 2;
        amr_params.mesh.amr_min_level = (refinement > 1) ? refinement - 1 : 0;
        amr_params.mesh.initial_refinement = refinement;
        amr_params.mesh.interface_coarsen_threshold = 0.9;
        pcout << "  AMR: interval=" << amr_interval
              << "  max_level=" << amr_params.mesh.amr_max_level
              << "  min_level=" << amr_params.mesh.amr_min_level << "\n";
    }

    // ----------------------------------------------------------------
    // 3. MMS source injection
    //
    // CH: Uses standard (non-SAV) assembler with convection by U*
    // NS: Uses assemble_coupled with MMS source for force residual
    // Poisson: Uses set_mms_source for −Δφ* − ∇·M*
    // Mag: Uses set_mms_source for transport + relaxation residual
    // ----------------------------------------------------------------

    // CH source: θ-equation with convection
    const double L[dim] = {1.0, L_y};
    CoupledCHSourceTheta<dim> coupled_ch_theta_src(
        params.physics.mobility, dt, L_y, L_y);

    // CH source: ψ-equation (Zhang Eq 3.10, {0,1} convention)
    CoupledCHSourcePsi<dim> coupled_ch_psi_src(
        params.physics.epsilon, params.physics.lambda, dt, L_y);

    ch.set_mms_source(
        [&](const Point<dim>& pt, double t) -> double {
            return coupled_ch_theta_src(pt, t);
        },
        [&](const Point<dim>& pt, double t) -> double {
            return coupled_ch_psi_src(pt, t);
        });

    // NS source
    CoupledNSSource<dim> ns_source(dt, L_y, params);
    ns.set_mms_source(
        [&](const Point<dim>& pt, double t) -> Tensor<1, dim> {
            return ns_source(pt, t);
        });

    // Poisson source: −Δφ* − ∇·M*  (coupled with magnetization)
    poisson.set_mms_source(
        [L_y](const Point<dim>& pt, double time) -> double {
            return compute_poisson_mms_source_coupled<dim>(pt, time, L_y);
        });

    // Magnetization source: with transport + spin-vorticity correction
    mag.set_mms_source(
        [L_y](const Point<dim>& pt,
              double t_new, double t_old,
              double tau_M, double chi_val,
              const Tensor<1, dim>& H,
              const Tensor<1, dim>& U,
              double div_U) -> Tensor<1, dim>
        {
            auto f = compute_mag_mms_source_with_transport<dim>(
                pt, t_new, t_old, tau_M, chi_val, H, U, div_U, L_y);

            // Spin-vorticity correction (D3 fix):
            // Assembler adds +½·ω_z·(-My_old, Mx_old) to RHS.
            // MMS source must subtract this exact contribution.
            // ω_z = curl(U*) at t_new, M* at t_old.
            const auto gux = NSMMS::ux_grad<dim>(pt, t_new, L_y);
            const auto guy = NSMMS::uy_grad<dim>(pt, t_new, L_y);
            const double omega_z = guy[0] - gux[1];
            const auto M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);
            f[0] -= 0.5 * omega_z * (-M_old[1]);
            f[1] -= 0.5 * omega_z * ( M_old[0]);

            return f;
        });

    // ----------------------------------------------------------------
    // 4. Initial conditions at t = t_start
    // ----------------------------------------------------------------
    // CH: project θ*, ψ*
    {
        CHMMSInitialTheta<dim> theta_ic(t_start, L);
        CHMMSInitialPsi<dim>   psi_ic(t_start, L);
        ch.project_initial_condition(theta_ic, psi_ic);
    }

    // CH Dirichlet BCs
    CHMMSBoundaryTheta<dim> theta_bc(L);
    CHMMSBoundaryPsi<dim>   psi_bc(L);
    theta_bc.set_time(t_start);
    psi_bc.set_time(t_start);
    ch.apply_dirichlet_boundary(theta_bc, psi_bc);
    ch.update_ghosts();

    // NS: initialize velocity from exact solution at t_start
    {
        NSMMSInitialUx<dim> ux_ic(t_start, L_y);
        NSMMSInitialUy<dim> uy_ic(t_start, L_y);
        ns.initialize_velocity(ux_ic, uy_ic);
    }
    ns.update_ghosts();

    // Magnetization: project M*(t_start)
    {
        MagExactMx<dim> exact_Mx(t_start, L_y);
        MagExactMy<dim> exact_My(t_start, L_y);
        mag.project_initial_condition(exact_Mx, exact_My);
    }
    mag.update_ghosts();

    // Poisson: initial solve consistent with M*(t_start)
    poisson.assemble_rhs(mag.get_Mx_relevant(), mag.get_My_relevant(),
                         mag.get_dof_handler(), t_start);
    poisson.solve();
    poisson.update_ghosts();

    // ----------------------------------------------------------------
    // 5. Index sets and workspace vectors for Picard iteration
    // ----------------------------------------------------------------
    IndexSet M_owned    = mag.get_dof_handler().locally_owned_dofs();
    IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(
        mag.get_dof_handler());

    TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_comm);

    Mx_relaxed = mag.get_Mx_relevant();
    My_relaxed = mag.get_My_relevant();

    // ----------------------------------------------------------------
    // 6. Time stepping — mirrors production driver
    //
    // Step 1: CH (θ^n, ψ^n using U^{n-1})
    // Step 2: NS (U^n, p^n using θ^n, θ^{n-1}, ψ^n, φ^{n-1}, M^{n-1})
    // Step 3: Poisson/Mag Picard (φ^n, M^n using U^n, θ^n)
    // ----------------------------------------------------------------
    const unsigned int max_picard = 50;
    const double picard_tol       = 1e-10;
    const double picard_omega     = 0.35;

    double current_time = t_start;

    // Save θ^{n-1} for NS
    TrilinosWrappers::MPI::Vector theta_old_relevant(
        ch.get_theta_dof_handler().locally_owned_dofs(),
        DoFTools::extract_locally_relevant_dofs(ch.get_theta_dof_handler()),
        mpi_comm);

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Save θ^{n-1} before CH solve
        theta_old_relevant = ch.get_theta_relevant();

        // ==== Step 1: Cahn-Hilliard ====
        // Update Dirichlet BCs
        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);
        ch.apply_dirichlet_boundary(theta_bc, psi_bc);

        // Get velocity for CH convection (U^{n-1})
        std::vector<const TrilinosWrappers::MPI::Vector*> vel_ptrs = {
            &ns.get_ux_old_relevant(),
            &ns.get_uy_old_relevant()
        };

        ch.assemble(ch.get_theta_relevant(),
                    vel_ptrs, ns.get_ux_dof_handler(),
                    dt, current_time);
        ch.solve();
        ch.update_ghosts();

        // ==== Step 2: Navier-Stokes ====
        // Uses θ^n (just computed), θ^{n-1} (saved), φ^{n-1}, M^{n-1}
        ns.assemble_coupled(
            dt,
            ch.get_theta_relevant(),           // θ^n for ν, ρ
            theta_old_relevant,                // θ^{n-1} for capillary
            ch.get_theta_dof_handler(),
            ch.get_psi_relevant(),             // ψ^n
            ch.get_psi_dof_handler(),
            poisson.get_solution_relevant(),   // φ^{n-1}
            poisson.get_dof_handler(),
            mag.get_Mx_relevant(),             // M^{n-1}
            mag.get_My_relevant(),
            mag.get_dof_handler(),
            current_time,                      // current time for MMS source
            true);                             // include convection

        ns.solve_velocity();

        // Step 3a: Pressure Poisson (projection method)
        ns.assemble_pressure_poisson(dt);
        ns.solve_pressure();

        // Step 4: Velocity correction
        ns.velocity_correction(dt);

        ns.advance_time();
        ns.update_ghosts();

        // ==== Step 3: Poisson/Mag Picard ====
        // Snapshot M^{n-1}
        Mx_old = mag.get_Mx_relevant();
        My_old = mag.get_My_relevant();
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Poisson: φ^k using relaxed M
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Magnetization: M^k using φ^k and U^n
            if (k == 0)
            {
                mag.assemble(
                    Mx_old, My_old,
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                    ns.get_ux_old_relevant(), ns.get_uy_old_relevant(),
                    ns.get_ux_dof_handler(),
                    dt, current_time);
            }
            else
            {
                mag.assemble_rhs_only(
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                    Mx_old, My_old,
                    dt, current_time);
            }
            mag.solve();
            mag.update_ghosts();

            // Under-relax
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

            Mx_relaxed_owned  = mag.get_Mx_solution();
            Mx_relaxed_owned *= picard_omega;
            Mx_relaxed_owned.add(1.0 - picard_omega, Mx_prev);

            My_relaxed_owned  = mag.get_My_solution();
            My_relaxed_owned *= picard_omega;
            My_relaxed_owned.add(1.0 - picard_omega, My_prev);

            Mx_relaxed = Mx_relaxed_owned;
            My_relaxed = My_relaxed_owned;

            // Convergence check
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
                break;
            }
            if (k == max_picard - 1)
                result.picard_iters = max_picard;
        }

        // ==== AMR (if enabled) ====
        if (use_amr && (step + 1) % amr_interval == 0 && step + 1 < n_time_steps)
        {
            perform_amr(triangulation, amr_params, mpi_comm,
                        ch, ns, poisson, mag,
                        /*ns_enabled=*/true, /*mag_enabled=*/true);

            // Recreate workspace vectors with new index sets
            M_owned    = mag.get_dof_handler().locally_owned_dofs();
            M_relevant = DoFTools::extract_locally_relevant_dofs(
                mag.get_dof_handler());

            Mx_old.reinit(M_owned, M_relevant, mpi_comm);
            My_old.reinit(M_owned, M_relevant, mpi_comm);
            Mx_relaxed.reinit(M_owned, M_relevant, mpi_comm);
            My_relaxed.reinit(M_owned, M_relevant, mpi_comm);
            Mx_relaxed_owned.reinit(M_owned, mpi_comm);
            My_relaxed_owned.reinit(M_owned, mpi_comm);
            Mx_prev.reinit(M_owned, mpi_comm);
            My_prev.reinit(M_owned, mpi_comm);

            // Copy current M state into workspace
            Mx_relaxed = mag.get_Mx_relevant();
            My_relaxed = mag.get_My_relevant();

            // Recreate theta_old_relevant with new CH index sets
            theta_old_relevant.reinit(
                ch.get_theta_dof_handler().locally_owned_dofs(),
                DoFTools::extract_locally_relevant_dofs(ch.get_theta_dof_handler()),
                mpi_comm);
            theta_old_relevant = ch.get_theta_relevant();

            // Re-apply Dirichlet BCs on new mesh
            ch.apply_dirichlet_boundary(theta_bc, psi_bc);
            ch.update_ghosts();
        }
    }

    // ----------------------------------------------------------------
    // 7. Compute errors at final time
    // ----------------------------------------------------------------
    ch.update_ghosts();
    ns.update_ghosts();
    poisson.update_ghosts();
    mag.update_ghosts();

    // CH errors
    {
        CHMMSErrors ch_err = compute_ch_mms_errors<dim>(
            ch.get_theta_dof_handler(),
            ch.get_psi_dof_handler(),
            ch.get_theta_relevant(),
            ch.get_psi_relevant(),
            current_time, L, mpi_comm);

        result.theta_L2 = ch_err.theta_L2;
        result.theta_H1 = ch_err.theta_H1;
        result.psi_L2   = ch_err.psi_L2;
    }

    // NS errors
    {
        NSMMSErrors ns_err = compute_ns_mms_errors<dim>(
            ns, current_time, L_y, mpi_comm);

        result.ux_L2 = ns_err.ux_L2;
        result.uy_L2 = ns_err.uy_L2;
        result.p_L2  = ns_err.p_L2;
    }

    // Poisson errors
    {
        PoissonMMSErrors phi_err = compute_poisson_mms_errors<dim>(
            poisson.get_dof_handler(),
            poisson.get_solution_relevant(),
            current_time, L_y, mpi_comm);

        result.phi_L2 = phi_err.L2;
        result.phi_H1 = phi_err.H1;
    }

    // Magnetization errors
    {
        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            mag.get_dof_handler(),
            mag.get_Mx_relevant(),
            mag.get_My_relevant(),
            current_time, L_y, mpi_comm);

        result.M_L2  = mag_err.M_L2;
        result.Mx_L2 = mag_err.Mx_L2;
        result.My_L2 = mag_err.My_L2;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.time_s = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  θ_L2=" << std::scientific << std::setprecision(2) << result.theta_L2
          << "  |U|_L2=" << result.ux_L2
          << "  φ_L2=" << result.phi_L2
          << "  M_L2=" << result.M_L2
          << "  time=" << std::fixed << std::setprecision(1)
          << result.time_s << "s\n";

    return result;
}


// ============================================================================
// Convergence study
// ============================================================================
CoupledMMSConvergenceResult run_coupled_system_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    bool use_amr,
    MPI_Comm mpi_comm)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Local copy with MMS overrides
    Parameters p = params;
    p.enable_mms = true;
    p.enable_ns = true;
    p.enable_magnetic = true;
    p.use_algebraic_magnetization = false;
    p.enable_gravity = false;
    p.dipoles.intensity_max = 0.0;
    p.dipoles.positions.clear();

    // ------------------------------------------------------------------
    // Simplifications to avoid strong-form MMS source inaccuracies:
    //
    // 1. Set μ₀ = 0: eliminates Kelvin force and b_stab in NS assembly.
    //    The NS source can't represent: (a) the Kelvin-2 weak-form term
    //    μ₀/2(M×H, ∇×V) which requires curl conversion to strong form,
    //    and (b) b_stab bilinear forms μ₀dt[(U·∇)m, (V·∇)m] which
    //    can't be represented as a simple (f, V) source.
    //
    // 2. Equal viscosities: removes the variable-viscosity correction
    //    -(∇ν)·D(U*) that the strong-form source omits (the weak form
    //    (ν/4)(T(U):T(V)) → strong form -∇·(ν D(U)) = -ν/2 ∇²U
    //    only when ν is constant).
    //
    // This still tests:
    //   - CH convection by NS velocity
    //   - NS capillary force from θ·∇ψ
    //   - NS viscous + convection + pressure
    //   - Poisson↔Magnetization coupling
    //   - Magnetization transport by NS velocity
    // ------------------------------------------------------------------
    p.physics.mu_0     = 0.0;  // disable Kelvin + b_stab in NS
    p.physics.nu_water = 1.0;  // constant viscosity: no ∇ν correction
    p.physics.nu_ferro = 1.0;  //   ν = 1.0 everywhere

    CoupledMMSConvergenceResult result;

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "  Full Coupled System MMS: CH → NS → Poisson/Mag"
                  << (use_amr ? " [AMR]" : "") << "\n";
        std::cout << "============================================================\n";
        std::cout << "  MPI ranks:  "
                  << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Picard:     max=50  tol=1e-10  omega=0.35\n";
        std::cout << "  FE:         θ,ψ,u=Q" << p.fe.degree_velocity
                  << "  p=DG-P" << p.fe.degree_pressure
                  << "  φ=Q" << p.fe.degree_potential
                  << "  M=DG-Q" << p.fe.degree_magnetization << "\n";
        std::cout << "  Expected:   θ_L2=" << result.expected_theta_L2
                  << "  u_L2=" << result.expected_ux_L2
                  << "  φ_L2=" << result.expected_phi_L2
                  << "  M_L2=" << result.expected_M_L2
                  << "  p_L2=" << result.expected_p_L2 << "\n";
        std::cout << "  Refs:      ";
        for (auto r : refinements) std::cout << " " << r;
        std::cout << "\n";
    }

    // Scale dt ∝ h² across refinement levels so the O(dt) projection
    // method splitting error doesn't dominate the spatial error.
    const unsigned int ref_base = refinements.front();
    for (unsigned int ref : refinements)
    {
        unsigned int steps_for_ref = n_time_steps;
        for (unsigned int r = ref_base; r < ref; ++r)
            steps_for_ref *= 4;
        result.results.push_back(
            run_single_level(ref, p, steps_for_ref, use_amr, mpi_comm));
    }

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

    // Defaults: use refs {1,2,3} so spatial error dominates over
    // transport/splitting errors at the coarsest pair
    std::vector<unsigned int> refinements = {1, 2, 3};
    unsigned int n_time_steps = 10;
    bool use_amr = false;

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
        else if (arg == "--amr")
        {
            use_amr = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            if (rank == 0)
                std::cout << "Usage: mpirun -np N " << argv[0]
                          << " [--refs 2 3 4] [--steps 10] [--amr]\n";
            return 0;
        }
    }

    Parameters params;

    try
    {
        auto result = run_coupled_system_mms(
            refinements, params, n_time_steps, use_amr, MPI_COMM_WORLD);

        if (rank == 0)
        {
            result.print();

            std::cout << "\n============================================================\n";
            if (result.passes())
                std::cout << "  [PASS] All coupled convergence rates within tolerance.\n";
            else
                std::cout << "  [FAIL] Some convergence rates below expected.\n";
            std::cout << "============================================================\n";

            result.write_csv("coupled_system_mms_rates.csv");
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
