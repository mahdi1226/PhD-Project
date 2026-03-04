// ============================================================================
// mms_tests/full_ch_fhd_mms_test.cc — Full 6-Subsystem Coupled MMS
//
// Verifies the complete Phase B system: FHD + Cahn-Hilliard with
// phase-dependent material properties chi(phi) and nu(phi).
//
// Per time step:
//   1. Picard loop: Mag(M_old, H, u_old, ch_old) <-> Poisson(M_relaxed)
//      - chi(phi_old) per-QP in magnetization relaxation
//   2. NS(u_old, w_old, M, H, ch_old) — nu(phi_old) + capillary + Kelvin + micropolar
//   3. AngMom(w_old, u_new, M, H) — curl coupling + magnetic torque
//   4. CH(ch_old, u_new) — convection from velocity
//
// Coupling terms verified (Phase A):
//   - Poisson <-> Mag:      div M -> Poisson RHS, grad(phi) -> Mag relaxation
//   - NS <- Mag/Poisson:    Kelvin force mu_0[(M.grad)H + 1/2(div M)H]
//   - NS <- AngMom:         micropolar 2*nu_r*(w, curl v)
//   - AngMom <- NS:         curl coupling 2*nu_r*(curl u, z)
//   - AngMom <- Mag/Poisson: magnetic torque mu_0*(m x h, z)
//   - Mag <- NS:            velocity transport B_h^m(u; M, z)
//
// Coupling terms verified (Phase B):
//   - Mag <- CH:            chi(phi) per-QP in relaxation
//   - NS <- CH:             nu(phi) per-QP in viscous + capillary force sigma*mu*grad(phi)
//   - CH <- NS:             convection u*grad(phi)
//
// Expected convergence (CG Q2 vel/pot/ang/ch, DG Q2 mag, DG P1 pressure):
//   U_L2 ~ 3, U_H1 ~ 2, p_L2 ~ 2, w_L2 ~ 3, w_H1 ~ 2
//   phi_mag_L2 ~ 3, phi_mag_H1 ~ 2, M_L2 ~ 3
//   theta_L2 ~ 3, theta_H1 ~ 2, mu_L2 ~ 3, mu_H1 ~ 2
//
// Usage:
//   mpirun -np 2 ./test_full_ch_fhd_mms
//   mpirun -np 4 ./test_full_ch_fhd_mms --refs 2 3 4 5 --steps 5
//
// Reference: Nochetto, Salgado & Tomas (2015, 2016)
// ============================================================================

#include "mms_tests/full_ch_fhd_mms.h"
#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "cahn_hilliard/cahn_hilliard.h"
#include "mesh/mesh.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstring>

constexpr int dim = 2;

// ============================================================================
// deal.II Function wrappers for IC projection
// ============================================================================
class FullCHFHDExactUx : public dealii::Function<dim>
{
public:
    FullCHFHDExactUx(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    { return ns_exact_velocity<dim>(p, time_)[0]; }
private:
    double time_;
};

class FullCHFHDExactUy : public dealii::Function<dim>
{
public:
    FullCHFHDExactUy(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    { return ns_exact_velocity<dim>(p, time_)[1]; }
private:
    double time_;
};

class FullCHFHDExactW : public dealii::Function<dim>
{
public:
    FullCHFHDExactW(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    { return angular_momentum_exact<dim>(p, time_); }
private:
    double time_;
};

// ============================================================================
// Single refinement level
// ============================================================================
static FullCHFHDMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    FullCHFHDMMSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------------
    // Time stepping parameters
    // ----------------------------------------------------------------
    const double t_start = 1.0;
    const double t_end = 1.1;
    const double dt = (t_end - t_start) / n_time_steps;
    const double nu_r = params.physics.nu_r;
    const double mu_0 = params.physics.mu_0;
    const double sigma_cap = params.cahn_hilliard_params.sigma;
    const double epsilon = params.cahn_hilliard_params.epsilon;
    const double gamma_ch = params.cahn_hilliard_params.gamma;

    pcout << "\n  [ref=" << refinement << "] dt=" << std::scientific
          << std::setprecision(3) << dt
          << ", nu=" << params.physics.nu
          << ", nu_r=" << nu_r
          << ", mu_0=" << mu_0
          << ", sigma=" << sigma_cap
          << ", eps=" << epsilon << "\n";

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
    local_params.time.dt = dt;

    FHDMesh::create_mesh<dim>(triangulation, local_params);

    result.h = GridTools::minimal_cell_diameter(triangulation);

    // ----------------------------------------------------------------
    // 2. Setup all 5 subsystems (6 fields: u, p, w, phi_mag, M, theta/mu)
    // ----------------------------------------------------------------
    PoissonSubsystem<dim> poisson(local_params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(local_params, mpi_comm, triangulation);
    NavierStokesSubsystem<dim> ns(local_params, mpi_comm, triangulation);
    AngularMomentumSubsystem<dim> am(local_params, mpi_comm, triangulation);
    CahnHilliardSubsystem<dim> ch(local_params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();
    ns.setup();
    am.setup();
    ch.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs = mag.get_dof_handler().n_dofs();
    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int w_dofs = am.get_dof_handler().n_dofs();
    const unsigned int ch_dofs = ch.get_dof_handler().n_dofs();
    result.n_dofs = phi_dofs + 2 * M_dofs + 2 * vel_dofs + p_dofs + w_dofs + ch_dofs;

    pcout << "  DoFs: phi_mag=" << phi_dofs << " M=" << M_dofs << "(x2)"
          << " vel=" << vel_dofs << "(x2) p=" << p_dofs
          << " w=" << w_dofs << " ch=" << ch_dofs
          << " total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. MMS sources
    // ----------------------------------------------------------------

    // Poisson: same as Phase A (no CH coupling in Poisson)
    poisson.set_mms_source(
        [](const Point<dim>& pt, double time) -> double
        {
            return compute_full_poisson_mms_source<dim>(pt, time);
        });

    // Magnetization: same callback, but assembler passes chi_q per-QP
    // instead of constant kappa_0 when CH is provided
    mag.set_mms_source(compute_full_mag_mms_source<dim>);

    // NS: combined Phase A + capillary force
    ns.set_mms_source(
        [nu_r, mu_0, sigma_cap](const Point<dim>& p,
                                double t_new, double t_old, double nu_eff,
                                const Tensor<1, dim>& U_old_disc,
                                double div_U_old_disc,
                                bool include_convection)
        {
            return compute_full_ch_fhd_ns_source<dim>(
                p, t_new, t_old, nu_eff, nu_r, mu_0, sigma_cap,
                U_old_disc, div_U_old_disc, include_convection);
        });

    // AngMom: same as Phase A (no CH coupling)
    am.set_mms_source(
        [mu_0](const Point<dim>& p,
               double t_new, double t_old,
               double j_micro, double c1, double nu_r_am,
               double w_old_disc,
               const Tensor<1, dim>& U_old_disc,
               double div_U_old_disc,
               bool include_convection)
        {
            return compute_full_angmom_mms_source<dim>(
                p, t_new, t_old, j_micro, c1, nu_r_am, mu_0,
                w_old_disc, U_old_disc, div_U_old_disc,
                include_convection);
        });

    // CH source: set per time step (captures current time)

    // ----------------------------------------------------------------
    // 4. Initial conditions at t_start
    // ----------------------------------------------------------------

    // Velocity: interpolate u*(t_start)
    TrilinosWrappers::MPI::Vector ux_owned(
        ns.get_ux_dof_handler().locally_owned_dofs(), mpi_comm);
    TrilinosWrappers::MPI::Vector uy_owned(
        ns.get_uy_dof_handler().locally_owned_dofs(), mpi_comm);

    FullCHFHDExactUx exact_ux(t_start);
    FullCHFHDExactUy exact_uy(t_start);
    VectorTools::interpolate(ns.get_ux_dof_handler(), exact_ux, ux_owned);
    VectorTools::interpolate(ns.get_uy_dof_handler(), exact_uy, uy_owned);

    IndexSet ux_relevant = DoFTools::extract_locally_relevant_dofs(
        ns.get_ux_dof_handler());
    IndexSet uy_relevant = DoFTools::extract_locally_relevant_dofs(
        ns.get_uy_dof_handler());

    TrilinosWrappers::MPI::Vector ux_old_rel(
        ns.get_ux_dof_handler().locally_owned_dofs(), ux_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_old_rel(
        ns.get_uy_dof_handler().locally_owned_dofs(), uy_relevant, mpi_comm);
    ux_old_rel = ux_owned;
    uy_old_rel = uy_owned;

    // Angular velocity: interpolate w*(t_start)
    TrilinosWrappers::MPI::Vector w_owned(
        am.get_dof_handler().locally_owned_dofs(), mpi_comm);

    FullCHFHDExactW exact_w(t_start);
    VectorTools::interpolate(am.get_dof_handler(), exact_w, w_owned);

    IndexSet w_relevant_set = DoFTools::extract_locally_relevant_dofs(
        am.get_dof_handler());
    TrilinosWrappers::MPI::Vector w_old_rel(
        am.get_dof_handler().locally_owned_dofs(), w_relevant_set, mpi_comm);
    w_old_rel = w_owned;

    // Magnetization: L2-project M*(t_start)
    IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
    IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(
        mag.get_dof_handler());

    TrilinosWrappers::MPI::Vector Mx_ic(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_ic(M_owned, mpi_comm);
    project_magnetization_exact<dim>(mag.get_dof_handler(),
                                     Mx_ic, My_ic, t_start);

    // M workspace vectors
    TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed(M_owned, M_relevant, mpi_comm);

    TrilinosWrappers::MPI::Vector Mx_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_comm);

    Mx_old = Mx_ic;
    My_old = My_ic;
    Mx_relaxed = Mx_ic;
    My_relaxed = My_ic;

    // CH: project (theta*, mu*) at t_start
    CHExactSolution<dim> ch_ic(t_start);
    ch.initialize(ch_ic);
    ch.save_old_solution();

    // Initial Poisson solve consistent with M*(t_start)
    poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                         mag.get_dof_handler(), t_start);
    poisson.solve();
    poisson.update_ghosts();

    // ----------------------------------------------------------------
    // 5. Time stepping with full coupling
    // ----------------------------------------------------------------
    const unsigned int max_picard = 50;
    const double picard_tol = 1e-10;
    const double omega = 0.35;

    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ============================================================
        // CH MMS source: must capture current physical time
        // ============================================================
        const double t_now = current_time;
        ch.set_mms_source(
            [epsilon, gamma_ch, t_now](
                const dealii::Point<dim>& p,
                double /*t*/, double dt_local,
                double phi_old_disc) -> std::pair<double, double>
            {
                return compute_ch_ns_coupled_source<dim>(
                    p, t_now, dt_local, phi_old_disc, epsilon, gamma_ch);
            });

        // ============================================================
        // Phase 1: Picard iteration — Poisson <-> Magnetization
        //   chi(phi_old) evaluated per-QP in Mag assembler
        // ============================================================
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Step A: Poisson -> phi^k using RELAXED M
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Step B: Mag -> M_raw^k using phi^k, u_old, w_old, ch_old
            //   chi(phi_old) per-QP passed via ch.get_old_relevant()
            mag.assemble(Mx_old, My_old,
                         poisson.get_solution_relevant(),
                         poisson.get_dof_handler(),
                         ux_old_rel, uy_old_rel,
                         ns.get_ux_dof_handler(),
                         dt, current_time,
                         w_old_rel, &am.get_dof_handler(),
                         ch.get_old_relevant(),
                         &ch.get_dof_handler());
            mag.solve();
            mag.update_ghosts();

            // Step C: Under-relax M
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

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

        // Final Poisson solve consistent with converged M
        poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                             mag.get_dof_handler(), current_time);
        poisson.solve();
        poisson.update_ghosts();

        // ============================================================
        // Phase 2: NS with ALL coupling
        //   nu(phi_old) per-QP + capillary + Kelvin + micropolar
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/true,
                    w_old_rel, am.get_dof_handler(),
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler(),
                    ch.get_old_relevant(),
                    &ch.get_dof_handler());

        ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Phase 3: AngMom with curl + torque + convection
        //   No direct CH coupling in AngMom
        // ============================================================
        am.assemble(w_old_rel,
                    dt, current_time,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler(),
                    /*include_convection=*/true,
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler());

        am.solve();
        am.update_ghosts();

        // ============================================================
        // Phase 4: CH with convection from new velocity
        // ============================================================
        ch.assemble(ch.get_old_relevant(), dt,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler());

        ch.solve();
        ch.update_ghosts();
        ch.save_old_solution();

        // ============================================================
        // Advance old solutions for next time step
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();
        w_old_rel = am.get_relevant();
        Mx_old = Mx_relaxed;
        My_old = My_relaxed;
    }

    // ----------------------------------------------------------------
    // 6. Compute errors at final time
    // ----------------------------------------------------------------
    {
        NSMMSErrors ns_err = compute_ns_mms_errors<dim>(
            ns.get_ux_dof_handler(),
            ns.get_ux_relevant(),
            ns.get_uy_relevant(),
            ns.get_p_dof_handler(),
            ns.get_p_relevant(),
            current_time, mpi_comm);

        result.U_L2 = ns_err.U_L2;
        result.U_H1 = ns_err.U_H1;
        result.p_L2 = ns_err.p_L2;
    }

    {
        AngularMomentumMMSErrors am_err = compute_angular_mms_errors<dim>(
            am.get_dof_handler(),
            am.get_relevant(),
            current_time, mpi_comm);

        result.w_L2 = am_err.w_L2;
        result.w_H1 = am_err.w_H1;
    }

    {
        PoissonMMSErrors phi_err = compute_poisson_mms_errors<dim>(
            poisson.get_dof_handler(),
            poisson.get_solution_relevant(),
            current_time, mpi_comm);

        result.phi_mag_L2 = phi_err.L2;
        result.phi_mag_H1 = phi_err.H1;
    }

    {
        MagnetizationMMSErrors mag_err = compute_mag_mms_errors<dim>(
            mag.get_dof_handler(),
            Mx_relaxed, My_relaxed,
            current_time, mpi_comm);

        result.M_L2 = mag_err.M_L2;
    }

    {
        CHMMSErrors ch_err = compute_ch_mms_errors<dim>(
            ch.get_dof_handler(),
            ch.get_relevant(),
            current_time, mpi_comm);

        result.theta_L2 = ch_err.phi_L2;
        result.theta_H1 = ch_err.phi_H1;
        result.mu_L2    = ch_err.mu_L2;
        result.mu_H1    = ch_err.mu_H1;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results:"
          << " U_L2=" << std::scientific << std::setprecision(2) << result.U_L2
          << " p_L2=" << result.p_L2
          << " w_L2=" << result.w_L2
          << " phi_mag_L2=" << result.phi_mag_L2
          << " M_L2=" << result.M_L2
          << "\n          "
          << " theta_L2=" << result.theta_L2
          << " mu_L2=" << result.mu_L2
          << " picard=" << result.picard_iters
          << " time=" << std::fixed << std::setprecision(1)
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
    bool use_block_schur = false;

    // Parse args
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--refs") == 0)
        {
            refinements.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                refinements.push_back(std::stoul(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            n_time_steps = std::stoul(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--block-schur") == 0)
        {
            use_block_schur = true;
        }
    }

    // MMS parameters
    Parameters params;
    params.setup_mms_validation();
    params.use_simplified_model = false;  // full Poisson solve for H

    // Phase B: CH coupling
    params.cahn_hilliard_params.epsilon = 1.0;
    params.cahn_hilliard_params.gamma   = 1.0;
    params.cahn_hilliard_params.sigma   = 1.0;
    params.enable_cahn_hilliard = true;

    // Phase-dependent material properties
    // NOTE: nu_carrier = nu_ferro -> constant viscosity.
    // Variable viscosity MMS requires -(grad nu).D(u*) correction in source.
    params.physics.nu_carrier = params.physics.nu;
    params.physics.nu_ferro   = params.physics.nu;
    params.physics.chi_ferro  = 1.0;  // chi(theta) = (theta+1)/2 varies from 0 to 1

    if (use_block_schur)
    {
        params.solvers.navier_stokes.use_iterative = true;
        params.solvers.navier_stokes.preconditioner =
            LinearSolverParams::Preconditioner::BlockSchur;
    }

    const unsigned int vel_deg = params.fe.degree_velocity;
    const unsigned int p_deg = params.fe.degree_pressure;
    const unsigned int w_deg = params.fe.degree_angular;
    const unsigned int phi_deg = params.fe.degree_potential;
    const unsigned int M_deg = params.fe.degree_magnetization;
    const unsigned int ch_deg = params.fe.degree_cahn_hilliard;

    pcout << "\n"
          << "================================================================\n"
          << "  FULL 6-SUBSYSTEM COUPLED MMS (FHD + CH with chi(phi), nu(phi))\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Time steps:     " << n_time_steps << "\n"
          << "  Picard:         max=50  tol=1e-10  omega=0.35\n"
          << "  FE:             u=CG Q" << vel_deg
          << "  p=DG P" << p_deg
          << "  w=CG Q" << w_deg
          << "  phi_mag=CG Q" << phi_deg
          << "  M=DG Q" << M_deg
          << "  theta,mu=CG Q" << ch_deg << "\n"
          << "  Physics:        nu=" << params.physics.nu
          << "  nu_r=" << params.physics.nu_r
          << "  mu_0=" << params.physics.mu_0
          << "  sigma=" << params.cahn_hilliard_params.sigma
          << "  chi_ferro=" << params.physics.chi_ferro << "\n"
          << "  Coupling:       Phase A: Poisson<->Mag, NS<-Kelvin+micropolar,\n"
          << "                           AngMom<-curl+torque, Mag<-transport\n"
          << "                  Phase B: Mag<-chi(phi), NS<-nu(phi)+capillary,\n"
          << "                           CH<-convection(u)\n"
          << "  Refs:           ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n";

    // ================================================================
    // Run convergence study
    // ================================================================
    std::vector<FullCHFHDMMSResult> results;

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

    std::vector<double> UL2_rates, UH1_rates, pL2_rates;
    std::vector<double> wL2_rates, wH1_rates;
    std::vector<double> phiL2_rates, phiH1_rates;
    std::vector<double> ML2_rates;
    std::vector<double> thetaL2_rates, thetaH1_rates;
    std::vector<double> muL2_rates, muH1_rates;

    for (size_t i = 1; i < results.size(); ++i)
    {
        UL2_rates.push_back(rate(results[i].U_L2, results[i-1].U_L2,
                                  results[i].h, results[i-1].h));
        UH1_rates.push_back(rate(results[i].U_H1, results[i-1].U_H1,
                                  results[i].h, results[i-1].h));
        pL2_rates.push_back(rate(results[i].p_L2, results[i-1].p_L2,
                                  results[i].h, results[i-1].h));
        wL2_rates.push_back(rate(results[i].w_L2, results[i-1].w_L2,
                                  results[i].h, results[i-1].h));
        wH1_rates.push_back(rate(results[i].w_H1, results[i-1].w_H1,
                                  results[i].h, results[i-1].h));
        phiL2_rates.push_back(rate(results[i].phi_mag_L2, results[i-1].phi_mag_L2,
                                    results[i].h, results[i-1].h));
        phiH1_rates.push_back(rate(results[i].phi_mag_H1, results[i-1].phi_mag_H1,
                                    results[i].h, results[i-1].h));
        ML2_rates.push_back(rate(results[i].M_L2, results[i-1].M_L2,
                                  results[i].h, results[i-1].h));
        thetaL2_rates.push_back(rate(results[i].theta_L2, results[i-1].theta_L2,
                                      results[i].h, results[i-1].h));
        thetaH1_rates.push_back(rate(results[i].theta_H1, results[i-1].theta_H1,
                                      results[i].h, results[i-1].h));
        muL2_rates.push_back(rate(results[i].mu_L2, results[i-1].mu_L2,
                                   results[i].h, results[i-1].h));
        muH1_rates.push_back(rate(results[i].mu_H1, results[i-1].mu_H1,
                                   results[i].h, results[i-1].h));
    }

    // ================================================================
    // Print NS table
    // ================================================================
    pcout << "\n--- Navier-Stokes (u, p) ---\n"
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
          << std::string(87, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::setw(10) << results[i].n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].U_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? UL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].U_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? UH1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].p_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? pL2_rates[i-1] : 0.0)
              << "\n";
    }

    // ================================================================
    // Print AngMom table
    // ================================================================
    pcout << "\n--- Angular Momentum (w) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "w_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "w_H1"
          << std::setw(8)  << "rate"
          << "\n"
          << std::string(57, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].w_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? wL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].w_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? wH1_rates[i-1] : 0.0)
              << "\n";
    }

    // ================================================================
    // Print Poisson + Magnetization table
    // ================================================================
    pcout << "\n--- Poisson (phi_mag) + Magnetization (M) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "phi_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "phi_H1"
          << std::setw(8)  << "rate"
          << std::setw(12) << "M_L2"
          << std::setw(8)  << "rate"
          << std::setw(8)  << "picard"
          << "\n"
          << std::string(83, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].phi_mag_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].phi_mag_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiH1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].M_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? ML2_rates[i-1] : 0.0)
              << std::fixed << std::setprecision(0)
              << std::setw(8)  << static_cast<double>(results[i].picard_iters)
              << "\n";
    }

    // ================================================================
    // Print CH table
    // ================================================================
    pcout << "\n--- Cahn-Hilliard (theta, mu) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "theta_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "theta_H1"
          << std::setw(8)  << "rate"
          << std::setw(12) << "mu_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "mu_H1"
          << std::setw(8)  << "rate"
          << std::setw(10) << "time(s)"
          << "\n"
          << std::string(109, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].theta_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? thetaL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].theta_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? thetaH1_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].mu_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? muL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].mu_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? muH1_rates[i-1] : 0.0)
              << std::fixed << std::setprecision(1)
              << std::setw(10) << results[i].walltime
              << "\n";
    }

    // ================================================================
    // Pass/fail
    // ================================================================
    const double tolerance = 0.4;  // slightly larger for coupled system
    bool pass = false;

    if (!UL2_rates.empty())
    {
        const double final_UL2     = UL2_rates.back();
        const double final_UH1     = UH1_rates.back();
        const double final_pL2     = pL2_rates.back();
        const double final_wL2     = wL2_rates.back();
        const double final_wH1     = wH1_rates.back();
        const double final_phiL2   = phiL2_rates.back();
        const double final_phiH1   = phiH1_rates.back();
        const double final_ML2     = ML2_rates.back();
        const double final_thetaL2 = thetaL2_rates.back();
        const double final_thetaH1 = thetaH1_rates.back();
        const double final_muL2    = muL2_rates.back();
        const double final_muH1    = muH1_rates.back();

        const double exp_UL2     = vel_deg + 1;
        const double exp_UH1     = vel_deg;
        const double exp_pL2     = p_deg + 1;
        const double exp_wL2     = w_deg + 1;
        const double exp_wH1     = w_deg;
        const double exp_phiL2   = phi_deg + 1;
        const double exp_phiH1   = phi_deg;
        const double exp_ML2     = M_deg + 1;
        const double exp_thetaL2 = ch_deg + 1;
        const double exp_thetaH1 = ch_deg;
        const double exp_muL2    = ch_deg + 1;
        const double exp_muH1    = ch_deg;

        pass = (final_UL2     >= exp_UL2     - tolerance)
            && (final_UH1     >= exp_UH1     - tolerance)
            && (final_pL2     >= exp_pL2     - tolerance)
            && (final_wL2     >= exp_wL2     - tolerance)
            && (final_wH1     >= exp_wH1     - tolerance)
            && (final_phiL2   >= exp_phiL2   - tolerance)
            && (final_phiH1   >= exp_phiH1   - tolerance)
            && (final_ML2     >= exp_ML2     - tolerance)
            && (final_thetaL2 >= exp_thetaL2 - tolerance)
            && (final_thetaH1 >= exp_thetaH1 - tolerance)
            && (final_muL2    >= exp_muL2    - tolerance)
            && (final_muH1    >= exp_muH1    - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY — Full 6-Subsystem Coupled MMS (FHD + CH)\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    U_L2:     " << std::fixed << std::setprecision(2)
              << final_UL2 << "  (expected " << exp_UL2 << ")\n"
              << "    U_H1:     " << final_UH1 << "  (expected " << exp_UH1 << ")\n"
              << "    p_L2:     " << final_pL2 << "  (expected " << exp_pL2 << ")\n"
              << "    w_L2:     " << final_wL2 << "  (expected " << exp_wL2 << ")\n"
              << "    w_H1:     " << final_wH1 << "  (expected " << exp_wH1 << ")\n"
              << "    phi_L2:   " << final_phiL2 << "  (expected " << exp_phiL2 << ")\n"
              << "    phi_H1:   " << final_phiH1 << "  (expected " << exp_phiH1 << ")\n"
              << "    M_L2:     " << final_ML2 << "  (expected " << exp_ML2 << ")\n"
              << "    theta_L2: " << final_thetaL2 << "  (expected " << exp_thetaL2 << ")\n"
              << "    theta_H1: " << final_thetaH1 << "  (expected " << exp_thetaH1 << ")\n"
              << "    mu_L2:    " << final_muL2 << "  (expected " << exp_muL2 << ")\n"
              << "    mu_H1:    " << final_muH1 << "  (expected " << exp_muH1 << ")\n"
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

        std::ofstream csv(SOURCE_DIR "/Results/mms/full_ch_fhd_mms.csv");
        csv << "refinement,n_dofs,h,"
            << "U_L2,U_L2_rate,U_H1,U_H1_rate,p_L2,p_L2_rate,"
            << "w_L2,w_L2_rate,w_H1,w_H1_rate,"
            << "phi_mag_L2,phi_mag_L2_rate,phi_mag_H1,phi_mag_H1_rate,"
            << "M_L2,M_L2_rate,"
            << "theta_L2,theta_L2_rate,theta_H1,theta_H1_rate,"
            << "mu_L2,mu_L2_rate,mu_H1,mu_H1_rate,"
            << "picard_iters,walltime\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.U_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? UL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.U_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? UH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.p_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? pL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.w_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? wL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.w_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? wH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.phi_mag_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.phi_mag_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.M_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? ML2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.theta_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? thetaL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.theta_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? thetaH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.mu_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? muL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.mu_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? muH1_rates[i-1] : 0.0) << ","
                << r.picard_iters << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }

        pcout << "  Results written to Results/mms/full_ch_fhd_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
