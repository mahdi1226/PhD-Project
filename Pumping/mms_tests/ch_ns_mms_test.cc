// ============================================================================
// mms_tests/ch_ns_mms_test.cc — Coupled CH + NS MMS Convergence Test
//
// Verifies two-phase coupling (Phase B):
//   NS ← CH:  Capillary force σ μ ∇φ on RHS
//   CH ← NS:  Convection u · ∇φ in CH transport
//
// Algorithm per time step (sequential):
//   1. Solve NS for (u^k, p^k) using capillary force from (φ^{k-1}, μ^{k-1})
//   2. Solve CH for (φ^k, μ^k) using velocity u^k for convection
//
// Configuration: no Kelvin force, no micropolar, no magnetization
//
// Usage:
//   mpirun -np 4 ./test_ch_ns_mms
//   mpirun -np 4 ./test_ch_ns_mms --refs 2 3 4 5 --steps 5
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "mms_tests/ch_ns_mms.h"
#include "navier_stokes/navier_stokes.h"
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
class NSExactUx : public dealii::Function<dim>
{
public:
    NSExactUx(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    { return ns_exact_velocity<dim>(p, time_)[0]; }
private:
    double time_;
};

class NSExactUy : public dealii::Function<dim>
{
public:
    NSExactUy(double t) : dealii::Function<dim>(1), time_(t) {}
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    { return ns_exact_velocity<dim>(p, time_)[1]; }
private:
    double time_;
};

// ============================================================================
// Single refinement level
// ============================================================================
static CHNSMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    CHNSMMSResult result;
    result.refinement = refinement;
    result.n_steps = n_time_steps;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------------
    // Time stepping parameters
    // ----------------------------------------------------------------
    const double t_start = 1.0;
    const double t_end = 1.1;
    const double dt = (t_end - t_start) / n_time_steps;

    const double sigma_cap = params.cahn_hilliard_params.sigma;
    const double epsilon = params.cahn_hilliard_params.epsilon;
    const double gamma_ch = params.cahn_hilliard_params.gamma;
    const double nu_eff = params.physics.nu + params.physics.nu_r;

    pcout << "\n  [ref=" << refinement << "] dt=" << std::scientific
          << std::setprecision(3) << dt
          << ", nu_eff=" << nu_eff
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
    // 2. Setup subsystems
    // ----------------------------------------------------------------
    NavierStokesSubsystem<dim> ns(local_params, mpi_comm, triangulation);
    CahnHilliardSubsystem<dim> ch(local_params, mpi_comm, triangulation);

    ns.setup();
    ch.setup();

    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int ch_dofs = ch.get_dof_handler().n_dofs();
    result.n_dofs = 2 * vel_dofs + p_dofs + ch_dofs;

    pcout << "  DoFs: vel=" << vel_dofs << "(x2) p=" << p_dofs
          << " ch=" << ch_dofs << " total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. NS MMS source (coupled — capillary force analytically)
    //
    // NS assembler passes correct (t_new, t_old) to the callback,
    // so we can set this once outside the loop.
    // ----------------------------------------------------------------
    ns.set_mms_source(
        [sigma_cap](
            const dealii::Point<dim>& p,
            double t_new, double t_old, double nu,
            const dealii::Tensor<1, dim>& U_old_disc,
            double div_U_old_disc, bool include_convection)
        {
            return compute_ns_ch_coupled_source<dim>(
                p, t_new, t_old, nu, sigma_cap,
                U_old_disc, div_U_old_disc, include_convection);
        });

    // ----------------------------------------------------------------
    // 4. Initial conditions at t_start
    // ----------------------------------------------------------------
    // Velocity: project u*(t_start)
    TrilinosWrappers::MPI::Vector ux_owned(
        ns.get_ux_dof_handler().locally_owned_dofs(), mpi_comm);
    TrilinosWrappers::MPI::Vector uy_owned(
        ns.get_uy_dof_handler().locally_owned_dofs(), mpi_comm);

    NSExactUx exact_ux(t_start);
    NSExactUy exact_uy(t_start);
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

    // CH: project (phi*, mu*) at t_start
    CHExactSolution<dim> ch_ic(t_start);
    ch.initialize(ch_ic);
    ch.save_old_solution();

    // ----------------------------------------------------------------
    // 5. Dummy w DoFHandler for NS (no micropolar coupling)
    // ----------------------------------------------------------------
    FE_Q<dim> fe_dummy(params.fe.degree_angular);
    DoFHandler<dim> w_dof_handler(triangulation);
    w_dof_handler.distribute_dofs(fe_dummy);

    IndexSet w_owned = w_dof_handler.locally_owned_dofs();
    IndexSet w_relevant_set = DoFTools::extract_locally_relevant_dofs(
        w_dof_handler);
    TrilinosWrappers::MPI::Vector w_dummy(w_owned, w_relevant_set, mpi_comm);
    w_dummy = 0.0;

    // ----------------------------------------------------------------
    // 6. Time stepping: NS(ch_old) → CH(u_new)
    // ----------------------------------------------------------------
    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ============================================================
        // CH MMS source must capture the current physical time.
        // The CH assembler passes params_.time.dt (not current time)
        // to the callback, so we capture t_now and ignore that param.
        // (Same pattern as standalone CH MMS test.)
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
        // Step 1: Solve NS with capillary force from old CH
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/false,
                    w_dummy, w_dof_handler,
                    /*Mx=*/{}, /*My=*/{}, /*M_dh=*/nullptr,
                    /*phi=*/{}, /*phi_dh=*/nullptr,
                    ch.get_old_relevant(),
                    &ch.get_dof_handler());

        ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Step 2: Solve CH with convection from new velocity
        // ============================================================
        ch.assemble(ch.get_old_relevant(), dt,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler());

        ch.solve();
        ch.update_ghosts();
        ch.save_old_solution();

        // ============================================================
        // Advance old NS solution for next step
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();
    }

    // ----------------------------------------------------------------
    // 7. Compute errors at final time
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
        CHMMSErrors ch_err = compute_ch_mms_errors<dim>(
            ch.get_dof_handler(),
            ch.get_relevant(),
            current_time, mpi_comm);

        result.phi_L2 = ch_err.phi_L2;
        result.phi_H1 = ch_err.phi_H1;
        result.mu_L2  = ch_err.mu_L2;
        result.mu_H1  = ch_err.mu_H1;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results: U_L2=" << std::scientific << std::setprecision(2)
          << result.U_L2 << "  p_L2=" << result.p_L2
          << "  phi_L2=" << result.phi_L2 << "  mu_L2=" << result.mu_L2
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
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            n_time_steps = std::stoul(argv[++i]);
        }
    }

    // MMS parameters
    Parameters params;
    params.setup_mms_validation();

    // CH coupling parameters (moderate for MMS)
    params.cahn_hilliard_params.epsilon = 1.0;
    params.cahn_hilliard_params.gamma   = 1.0;
    params.cahn_hilliard_params.sigma   = 1.0;
    params.enable_cahn_hilliard = true;

    const unsigned int vel_degree = params.fe.degree_velocity;
    const unsigned int p_degree   = params.fe.degree_pressure;
    const unsigned int ch_degree  = params.fe.degree_cahn_hilliard;

    pcout << "\n"
          << "================================================================\n"
          << "  COUPLED CH + NS MMS (Capillary + Convection)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Time steps:     " << n_time_steps << "\n"
          << "  FE:             u=CG Q" << vel_degree
          << "  p=DG P" << p_degree
          << "  phi,mu=CG Q" << ch_degree << "\n"
          << "  Physics:        nu=" << params.physics.nu
          << "  sigma=" << params.cahn_hilliard_params.sigma
          << "  eps=" << params.cahn_hilliard_params.epsilon
          << "  gamma=" << params.cahn_hilliard_params.gamma << "\n"
          << "  Coupling:       NS: +sigma*mu*grad(phi)  "
          << "CH: +u*grad(phi)\n"
          << "  Refs:           ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n";

    // ================================================================
    // Run convergence study
    // ================================================================
    std::vector<CHNSMMSResult> results;

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
    std::vector<double> phiL2_rates, phiH1_rates, muL2_rates, muH1_rates;

    for (size_t i = 1; i < results.size(); ++i)
    {
        UL2_rates.push_back(rate(results[i].U_L2, results[i-1].U_L2,
                                  results[i].h, results[i-1].h));
        UH1_rates.push_back(rate(results[i].U_H1, results[i-1].U_H1,
                                  results[i].h, results[i-1].h));
        pL2_rates.push_back(rate(results[i].p_L2, results[i-1].p_L2,
                                  results[i].h, results[i-1].h));
        phiL2_rates.push_back(rate(results[i].phi_L2, results[i-1].phi_L2,
                                    results[i].h, results[i-1].h));
        phiH1_rates.push_back(rate(results[i].phi_H1, results[i-1].phi_H1,
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
    // Print CH table
    // ================================================================
    pcout << "\n--- Cahn-Hilliard (phi, mu) ---\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "phi_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "phi_H1"
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
              << std::setw(12) << results[i].phi_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiL2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].phi_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? phiH1_rates[i-1] : 0.0)
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
        const double final_UL2    = UL2_rates.back();
        const double final_UH1    = UH1_rates.back();
        const double final_pL2    = pL2_rates.back();
        const double final_phiL2  = phiL2_rates.back();
        const double final_phiH1  = phiH1_rates.back();
        const double final_muL2   = muL2_rates.back();
        const double final_muH1   = muH1_rates.back();

        const double exp_UL2   = vel_degree + 1;   // 3.0
        const double exp_UH1   = vel_degree;        // 2.0
        const double exp_pL2   = p_degree + 1;      // 2.0
        const double exp_phiL2 = ch_degree + 1;     // 3.0
        const double exp_phiH1 = ch_degree;          // 2.0
        const double exp_muL2  = ch_degree + 1;     // 3.0
        const double exp_muH1  = ch_degree;          // 2.0

        pass = (final_UL2   >= exp_UL2   - tolerance)
            && (final_UH1   >= exp_UH1   - tolerance)
            && (final_pL2   >= exp_pL2   - tolerance)
            && (final_phiL2 >= exp_phiL2 - tolerance)
            && (final_phiH1 >= exp_phiH1 - tolerance)
            && (final_muL2  >= exp_muL2  - tolerance)
            && (final_muH1  >= exp_muH1  - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  MPI ranks:       " << n_ranks << "\n"
              << "  Time steps:      " << n_time_steps << "\n"
              << "  Refinements:     " << results.front().refinement
              << " -> " << results.back().refinement
              << " (" << results.front().n_dofs << " -> "
              << results.back().n_dofs << " DoFs)\n"
              << "\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    U_L2:   " << std::fixed << std::setprecision(2)
              << final_UL2 << "  (expected " << exp_UL2 << ")\n"
              << "    U_H1:   " << final_UH1 << "  (expected " << exp_UH1 << ")\n"
              << "    p_L2:   " << final_pL2 << "  (expected " << exp_pL2 << ")\n"
              << "    phi_L2: " << final_phiL2 << "  (expected " << exp_phiL2 << ")\n"
              << "    phi_H1: " << final_phiH1 << "  (expected " << exp_phiH1 << ")\n"
              << "    mu_L2:  " << final_muL2 << "  (expected " << exp_muL2 << ")\n"
              << "    mu_H1:  " << final_muH1 << "  (expected " << exp_muH1 << ")\n"
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

        std::ofstream csv(SOURCE_DIR "/Results/mms/ch_ns_coupled_mms.csv");
        csv << "refinement,n_dofs,h,"
            << "U_L2,U_L2_rate,U_H1,U_H1_rate,p_L2,p_L2_rate,"
            << "phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,"
            << "mu_L2,mu_L2_rate,mu_H1,mu_H1_rate,"
            << "n_steps,walltime\n";

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
                << std::scientific << std::setprecision(6) << r.phi_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.phi_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? phiH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.mu_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? muL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.mu_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? muH1_rates[i-1] : 0.0) << ","
                << r.n_steps << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }

        pcout << "  Results written to Results/mms/ch_ns_coupled_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
