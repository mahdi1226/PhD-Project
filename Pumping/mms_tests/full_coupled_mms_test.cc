// ============================================================================
// mms_tests/full_coupled_mms_test.cc — Full 4-System Coupled MMS
//
// Verifies the complete FHD coupling (Nochetto, Salgado & Tomas, Algorithm 42):
//
// Per time step:
//   1. Picard loop: Poisson(M_relaxed) ↔ Mag(M_old, H, u_old)
//      Under-relaxation: M_relaxed = ω·M_raw + (1−ω)·M_prev
//   2. NS(u_old, w_old, M_conv, φ_conv) — Kelvin force + micropolar
//   3. AngMom(w_old, u_new, M_conv, φ_conv) — curl coupling + magnetic torque
//
// Coupling terms verified:
//   - Poisson ↔ Mag:   div M → Poisson RHS, ∇φ → Mag relaxation
//   - NS ← Mag/Poisson: Kelvin force μ₀[(M·∇)H + ½(∇·M)H]
//   - NS ← AngMom:      micropolar 2ν_r(w, ∇×v)
//   - AngMom ← NS:      curl coupling 2ν_r(∇×u, z)
//   - AngMom ← Mag/Poisson: magnetic torque μ₀(m × h, z)
//   - Mag ← NS:         velocity transport B_h^m(u; M, z)
//
// Expected convergence (CG Q₂ velocity/potential/angular, DG Q₂ mag, DG P₁ pressure):
//   U_L2 ≈ 3, U_H1 ≈ 2, p_L2 ≈ 2, w_L2 ≈ 3, w_H1 ≈ 2
//   φ_L2 ≈ 3, φ_H1 ≈ 2, M_L2 ≈ 3
//
// Usage:
//   mpirun -np 2 ./test_full_coupled_mms
//   mpirun -np 4 ./test_full_coupled_mms --refs 2 3 4 5 --steps 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "mms_tests/full_coupled_mms.h"
#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
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
class ExactUx : public dealii::Function<dim>
{
public:
    ExactUx(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    {
        return ns_exact_velocity<dim>(p, time_)[0];
    }
private:
    double time_;
};

class ExactUy : public dealii::Function<dim>
{
public:
    ExactUy(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    {
        return ns_exact_velocity<dim>(p, time_)[1];
    }
private:
    double time_;
};

class ExactW : public dealii::Function<dim>
{
public:
    ExactW(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    {
        return angular_momentum_exact<dim>(p, time_);
    }
private:
    double time_;
};

// ============================================================================
// Single refinement level
// ============================================================================
static FullCoupledMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    FullCoupledMMSResult result;
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

    pcout << "\n  [ref=" << refinement << "] dt=" << std::scientific
          << std::setprecision(3) << dt
          << ", nu=" << params.physics.nu
          << ", nu_r=" << nu_r
          << ", mu_0=" << mu_0 << "\n";

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
    // 2. Setup all 4 subsystems
    // ----------------------------------------------------------------
    PoissonSubsystem<dim> poisson(local_params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(local_params, mpi_comm, triangulation);
    NavierStokesSubsystem<dim> ns(local_params, mpi_comm, triangulation);
    AngularMomentumSubsystem<dim> am(local_params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();
    ns.setup();
    am.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs = mag.get_dof_handler().n_dofs();
    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int w_dofs = am.get_dof_handler().n_dofs();
    result.n_dofs = phi_dofs + 2 * M_dofs + 2 * vel_dofs + p_dofs + w_dofs;

    pcout << "  DoFs: phi=" << phi_dofs << " M=" << M_dofs << "(x2)"
          << " vel=" << vel_dofs << "(x2) p=" << p_dofs
          << " w=" << w_dofs << " total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. MMS sources (full coupling — captures extra physics params)
    // ----------------------------------------------------------------
    poisson.set_mms_source(
        [](const Point<dim>& pt, double time) -> double
        {
            return compute_full_poisson_mms_source<dim>(pt, time);
        });

    mag.set_mms_source(compute_full_mag_mms_source<dim>);

    ns.set_mms_source(
        [nu_r, mu_0](const Point<dim>& p,
                     double t_new, double t_old, double nu_eff,
                     const Tensor<1, dim>& U_old_disc,
                     double div_U_old_disc,
                     bool include_convection)
        {
            return compute_full_ns_mms_source<dim>(
                p, t_new, t_old, nu_eff, nu_r, mu_0,
                U_old_disc, div_U_old_disc, include_convection);
        });

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

    // ----------------------------------------------------------------
    // 4. Initial conditions at t_start
    // ----------------------------------------------------------------

    // Velocity: interpolate u*(t_start)
    TrilinosWrappers::MPI::Vector ux_owned(
        ns.get_ux_dof_handler().locally_owned_dofs(), mpi_comm);
    TrilinosWrappers::MPI::Vector uy_owned(
        ns.get_uy_dof_handler().locally_owned_dofs(), mpi_comm);

    ExactUx exact_ux(t_start);
    ExactUy exact_uy(t_start);
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

    ExactW exact_w(t_start);
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
        // Phase 1: Picard iteration — Poisson ↔ Magnetization
        //   Uses u_old from previous time step for transport in Mag
        // ============================================================
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

            // Step B: Magnetization → M_raw^k using φ^k, u_old, w_old
            mag.assemble(Mx_old, My_old,
                         poisson.get_solution_relevant(),
                         poisson.get_dof_handler(),
                         ux_old_rel, uy_old_rel,
                         ns.get_ux_dof_handler(),
                         dt, current_time,
                         w_old_rel, &am.get_dof_handler());
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
        // Phase 2: NS with Kelvin force + micropolar coupling
        //   Inputs: u_old, w_old, M_converged, φ_converged
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/true,
                    w_old_rel, am.get_dof_handler(),
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler());

        ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Phase 3: AngMom with curl coupling + magnetic torque + convection
        //   Inputs: w_old, u_new, M_converged, φ_converged
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

        result.phi_L2 = phi_err.L2;
        result.phi_H1 = phi_err.H1;
    }

    {
        MagnetizationMMSErrors mag_err = compute_mag_mms_errors<dim>(
            mag.get_dof_handler(),
            Mx_relaxed, My_relaxed,
            current_time, mpi_comm);

        result.M_L2 = mag_err.M_L2;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results:"
          << " U_L2=" << std::scientific << std::setprecision(2) << result.U_L2
          << " U_H1=" << result.U_H1
          << " p_L2=" << result.p_L2
          << " w_L2=" << result.w_L2
          << " w_H1=" << result.w_H1
          << "\n          "
          << " phi_L2=" << result.phi_L2
          << " phi_H1=" << result.phi_H1
          << " M_L2=" << result.M_L2
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

    pcout << "\n"
          << "================================================================\n"
          << "  FULL 4-SYSTEM COUPLED MMS (Nochetto Algorithm 42)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Time steps:     " << n_time_steps << "\n"
          << "  Picard:         max=50  tol=1e-10  omega=0.35\n"
          << "  FE:             u=CG Q" << vel_deg
          << "  p=DG P" << p_deg
          << "  w=CG Q" << w_deg
          << "  phi=CG Q" << phi_deg
          << "  M=DG Q" << M_deg << "\n"
          << "  Physics:        nu=" << params.physics.nu
          << "  nu_r=" << params.physics.nu_r
          << "  mu_0=" << params.physics.mu_0
          << "  j=" << params.physics.j_micro
          << "  c1=" << params.physics.c_1 << "\n"
          << "  Expected rates: U_L2=" << vel_deg + 1
          << "  U_H1=" << vel_deg
          << "  p_L2=" << p_deg + 1
          << "  w_L2=" << w_deg + 1
          << "  w_H1=" << w_deg << "\n"
          << "                  phi_L2=" << phi_deg + 1
          << "  phi_H1=" << phi_deg
          << "  M_L2=" << M_deg + 1 << "\n"
          << "  Coupling:       Poisson<->Mag (Picard), NS<-Kelvin+micropolar,\n"
          << "                  AngMom<-curl+torque, Mag<-transport(u)\n"
          << "  Refs:           ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n";

    // ================================================================
    // Run convergence study
    // ================================================================
    std::vector<FullCoupledMMSResult> results;

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
        phiL2_rates.push_back(rate(results[i].phi_L2, results[i-1].phi_L2,
                                    results[i].h, results[i-1].h));
        phiH1_rates.push_back(rate(results[i].phi_H1, results[i-1].phi_H1,
                                    results[i].h, results[i-1].h));
        ML2_rates.push_back(rate(results[i].M_L2, results[i-1].M_L2,
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
    // Print Poisson table
    // ================================================================
    pcout << "\n--- Poisson (phi) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "phi_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "phi_H1"
          << std::setw(8)  << "rate"
          << "\n"
          << std::string(57, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].phi_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].phi_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiH1_rates[i-1] : 0.0)
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
          << std::setw(8)  << "picard"
          << std::setw(10) << "time(s)"
          << "\n"
          << std::string(55, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        pcout << std::left
              << std::setw(5)  << results[i].refinement
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].h
              << std::setw(12) << results[i].M_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? ML2_rates[i-1] : 0.0)
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

    if (!UL2_rates.empty())
    {
        const double final_UL2 = UL2_rates.back();
        const double final_UH1 = UH1_rates.back();
        const double final_pL2 = pL2_rates.back();
        const double final_wL2 = wL2_rates.back();
        const double final_wH1 = wH1_rates.back();
        const double final_phiL2 = phiL2_rates.back();
        const double final_phiH1 = phiH1_rates.back();
        const double final_ML2 = ML2_rates.back();

        const double exp_UL2 = vel_deg + 1;
        const double exp_UH1 = vel_deg;
        const double exp_pL2 = p_deg + 1;
        const double exp_wL2 = w_deg + 1;
        const double exp_wH1 = w_deg;
        const double exp_phiL2 = phi_deg + 1;
        const double exp_phiH1 = phi_deg;
        const double exp_ML2 = M_deg + 1;

        pass = (final_UL2 >= exp_UL2 - tolerance)
            && (final_UH1 >= exp_UH1 - tolerance)
            && (final_pL2 >= exp_pL2 - tolerance)
            && (final_wL2 >= exp_wL2 - tolerance)
            && (final_wH1 >= exp_wH1 - tolerance)
            && (final_phiL2 >= exp_phiL2 - tolerance)
            && (final_phiH1 >= exp_phiH1 - tolerance)
            && (final_ML2 >= exp_ML2 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY — Full 4-System Coupled MMS\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    U_L2:   " << std::fixed << std::setprecision(2)
              << final_UL2 << "  (expected " << exp_UL2 << ")\n"
              << "    U_H1:   " << final_UH1 << "  (expected " << exp_UH1 << ")\n"
              << "    p_L2:   " << final_pL2 << "  (expected " << exp_pL2 << ")\n"
              << "    w_L2:   " << final_wL2 << "  (expected " << exp_wL2 << ")\n"
              << "    w_H1:   " << final_wH1 << "  (expected " << exp_wH1 << ")\n"
              << "    phi_L2: " << final_phiL2 << "  (expected " << exp_phiL2 << ")\n"
              << "    phi_H1: " << final_phiH1 << "  (expected " << exp_phiH1 << ")\n"
              << "    M_L2:   " << final_ML2 << "  (expected " << exp_ML2 << ")\n"
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

        std::ofstream csv(SOURCE_DIR "/Results/mms/full_coupled_mms.csv");
        csv << "refinement,n_dofs,h,"
            << "U_L2,U_L2_rate,U_H1,U_H1_rate,p_L2,p_L2_rate,"
            << "w_L2,w_L2_rate,w_H1,w_H1_rate,"
            << "phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,"
            << "M_L2,M_L2_rate,"
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
                << r.phi_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.phi_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6)
                << r.M_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? ML2_rates[i-1] : 0.0) << ","
                << r.picard_iters << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }

        pcout << "  Results written to Results/mms/full_coupled_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
