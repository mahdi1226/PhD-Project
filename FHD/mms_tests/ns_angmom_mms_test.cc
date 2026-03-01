// ============================================================================
// mms_tests/ns_angmom_mms_test.cc — Coupled NS + Angular Momentum MMS
//
// Verifies micropolar coupling (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//   NS (Eq. 42e):      2ν_r(w, ∇×v)   on RHS
//   AngMom (Eq. 42f):  2ν_r(∇×u, z)   on RHS
//
// Algorithm per time step (sequential, no Picard):
//   1. Solve NS for (u^k, p^k) using w^{k-1}
//   2. Solve AngMom for w^k using u^k
//
// Configuration: no convection, no Kelvin force, no magnetization
//
// Usage:
//   mpirun -np 2 ./test_ns_angmom_mms
//   mpirun -np 4 ./test_ns_angmom_mms --refs 2 3 4 5 --steps 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================

#include "mms_tests/ns_angmom_mms.h"
#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
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
static NSAngMomMMSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    NSAngMomMMSResult result;
    result.refinement = refinement;
    result.n_steps = n_time_steps;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------------
    // Time stepping parameters
    // ----------------------------------------------------------------
    const double t_start = 1.0;
    const double t_end = 1.1;
    const double dt = (t_end - t_start) / n_time_steps;
    const double nu_r = params.physics.nu_r;

    pcout << "\n  [ref=" << refinement << "] dt=" << std::scientific
          << std::setprecision(3) << dt
          << ", nu=" << params.physics.nu
          << ", nu_r=" << nu_r << "\n";

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
    NavierStokesSubsystem<dim> ns(local_params, mpi_comm, triangulation);
    AngularMomentumSubsystem<dim> am(local_params, mpi_comm, triangulation);

    ns.setup();
    am.setup();

    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int w_dofs = am.get_dof_handler().n_dofs();
    result.n_dofs = 2 * vel_dofs + p_dofs + w_dofs;

    pcout << "  DoFs: vel=" << vel_dofs << "(x2) p=" << p_dofs
          << " w=" << w_dofs << " total=" << result.n_dofs << "\n";

    // ----------------------------------------------------------------
    // 3. MMS sources (coupled — include cross-coupling analytically)
    // ----------------------------------------------------------------
    ns.set_mms_source(
        [nu_r](const dealii::Point<dim>& p,
               double t_new, double t_old, double nu_eff,
               const dealii::Tensor<1, dim>& U_old_disc,
               double div_U_old_disc, bool include_convection)
        {
            return compute_ns_mms_source_coupled<dim>(
                p, t_new, t_old, nu_eff, nu_r, U_old_disc,
                div_U_old_disc, include_convection);
        });

    am.set_mms_source(
        [](const dealii::Point<dim>& p,
           double t_new, double t_old,
           double j_micro, double c1, double nu_r_am,
           double w_old_disc,
           const dealii::Tensor<1, dim>& U_old_disc,
           double div_U_old_disc,
           bool include_convection)
        {
            return compute_angmom_mms_source_coupled<dim>(
                p, t_new, t_old, j_micro, c1, nu_r_am, w_old_disc,
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

    // Angular velocity: project w*(t_start)
    TrilinosWrappers::MPI::Vector w_owned(
        am.get_dof_handler().locally_owned_dofs(), mpi_comm);

    ExactW exact_w(t_start);
    VectorTools::interpolate(am.get_dof_handler(), exact_w, w_owned);

    IndexSet w_relevant_set = DoFTools::extract_locally_relevant_dofs(
        am.get_dof_handler());
    TrilinosWrappers::MPI::Vector w_old_rel(
        am.get_dof_handler().locally_owned_dofs(), w_relevant_set, mpi_comm);
    w_old_rel = w_owned;

    // ----------------------------------------------------------------
    // 5. Time stepping: NS(w_old) → AngMom(u_new)
    // ----------------------------------------------------------------
    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ============================================================
        // Step 1: Solve NS for (u^k, p^k) using w^{k-1}
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/false,
                    w_old_rel, am.get_dof_handler());

        ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Step 2: Solve AngMom for w^k using u^k (just solved)
        // ============================================================
        am.assemble(w_old_rel,
                    dt, current_time,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler(),
                    /*include_convection=*/false);

        am.solve();
        am.update_ghosts();

        // ============================================================
        // Advance old solutions for next step
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();
        w_old_rel = am.get_relevant();
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
        result.U_Linf = ns_err.U_Linf;
        result.p_L2 = ns_err.p_L2;
    }

    {
        AngularMomentumMMSErrors am_err = compute_angular_mms_errors<dim>(
            am.get_dof_handler(),
            am.get_relevant(),
            current_time, mpi_comm);

        result.w_L2 = am_err.w_L2;
        result.w_H1 = am_err.w_H1;
        result.w_Linf = am_err.w_Linf;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  Results: U_L2=" << std::scientific << std::setprecision(2)
          << result.U_L2 << "  U_H1=" << result.U_H1
          << "  p_L2=" << result.p_L2
          << "  w_L2=" << result.w_L2 << "  w_H1=" << result.w_H1
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

    const unsigned int vel_degree = params.fe.degree_velocity;
    const unsigned int w_degree = params.fe.degree_angular;
    const unsigned int p_degree = params.fe.degree_pressure;

    pcout << "\n"
          << "================================================================\n"
          << "  COUPLED NS + ANGULAR MOMENTUM MMS (Micropolar Coupling)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Time steps:     " << n_time_steps << "\n"
          << "  FE:             u=CG Q" << vel_degree
          << "  p=DG P" << p_degree
          << "  w=CG Q" << w_degree << "\n"
          << "  Physics:        nu=" << params.physics.nu
          << "  nu_r=" << params.physics.nu_r
          << "  j=" << params.physics.j_micro
          << "  c1=" << params.physics.c_1 << "\n"
          << "  Expected:       U_L2=" << vel_degree + 1
          << "  U_H1=" << vel_degree
          << "  p_L2=" << p_degree + 1
          << "  w_L2=" << w_degree + 1
          << "  w_H1=" << w_degree << "\n"
          << "  Coupling:       NS: 2nu_r(w, curl v)  "
          << "AM: 2nu_r(curl u, z)\n"
          << "  Refs:           ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n";

    // ================================================================
    // Run convergence study
    // ================================================================
    std::vector<NSAngMomMMSResult> results;

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
          << std::setw(12) << "w_Linf"
          << std::setw(10) << "time(s)"
          << "\n"
          << std::string(79, '-') << "\n";

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
              << std::scientific << std::setprecision(2)
              << std::setw(12) << results[i].w_Linf
              << std::fixed << std::setprecision(1)
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

        const double expected_UL2 = vel_degree + 1;
        const double expected_UH1 = vel_degree;
        const double expected_pL2 = p_degree + 1;
        const double expected_wL2 = w_degree + 1;
        const double expected_wH1 = w_degree;

        pass = (final_UL2 >= expected_UL2 - tolerance)
            && (final_UH1 >= expected_UH1 - tolerance)
            && (final_pL2 >= expected_pL2 - tolerance)
            && (final_wL2 >= expected_wL2 - tolerance)
            && (final_wH1 >= expected_wH1 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    U_L2: " << std::fixed << std::setprecision(2)
              << final_UL2 << "  (expected " << expected_UL2 << ")\n"
              << "    U_H1: " << final_UH1 << "  (expected " << expected_UH1 << ")\n"
              << "    p_L2: " << final_pL2 << "  (expected " << expected_pL2 << ")\n"
              << "    w_L2: " << final_wL2 << "  (expected " << expected_wL2 << ")\n"
              << "    w_H1: " << final_wH1 << "  (expected " << expected_wH1 << ")\n"
              << "\n"
              << "  STATUS: " << (pass ? "PASS" : "FAIL") << "\n"
              << "================================================================\n\n";
    }

    // ================================================================
    // Write CSV
    // ================================================================
    if (rank == 0)
    {
        std::system("mkdir -p Results/mms");

        std::ofstream csv("Results/mms/ns_angmom_coupled_mms.csv");
        csv << "refinement,n_dofs,h,"
            << "U_L2,U_L2_rate,U_H1,U_H1_rate,p_L2,p_L2_rate,"
            << "w_L2,w_L2_rate,w_H1,w_H1_rate,"
            << "w_Linf,n_steps,walltime\n";

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
                << std::scientific << std::setprecision(6) << r.w_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? wL2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.w_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? wH1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.w_Linf << ","
                << r.n_steps << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }

        pcout << "  Results written to Results/mms/ns_angmom_coupled_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
