// ============================================================================
// navier_stokes/navier_stokes_main.cc — Comprehensive NS Validation Driver
//
// Runs all MMS verification phases in 2D (and 3D when facade is extended):
//
//   Phase A: Steady Stokes        — viscous + pressure only
//   Phase B: Unsteady Stokes      — time derivative + viscous + pressure
//   Phase C: Steady Navier-Stokes — viscous + convection + pressure
//   Phase D: Unsteady NS          — full production (all terms)
//
// Each phase performs h-convergence study over multiple refinement levels,
// checking that convergence rates match theoretical expectations:
//   Q2/DG-Q1:  vel L2 ~ O(h³), vel H1 ~ O(h²), p L2 ~ O(h²)
//   (Unsteady phases: vel L2 limited by O(dt) at fine h)
//
// Exit code: 0 if all phases pass, 1 if any fail.
//
// Usage:
//   mpirun -np 4 ./navier_stokes_main                    # all phases, 2D
//   mpirun -np 4 ./navier_stokes_main --phase A          # steady Stokes only
//   mpirun -np 4 ./navier_stokes_main --phase A B C D    # all phases explicitly
//   mpirun -np 4 ./navier_stokes_main --ref 2 3 4 5 6    # custom refinements
//   mpirun -np 4 ./navier_stokes_main --steps 20         # 20 time steps
//   mpirun -np 4 ./navier_stokes_main --dim 3            # 3D (when supported)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-42f
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "navier_stokes/tests/navier_stokes_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>


// ============================================================================
// Result structures (with L∞)
// ============================================================================
struct RefinementResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;

    double ux_L2   = 0.0, ux_H1   = 0.0, ux_Linf = 0.0;
    double uy_L2   = 0.0, uy_H1   = 0.0, uy_Linf = 0.0;
    double p_L2    = 0.0, p_Linf  = 0.0;
    double div_L2  = 0.0;

    double total_time = 0.0;
};

struct PhaseResult
{
    std::string phase_id;          // "A", "B", "C", "D"
    std::string phase_label;
    bool is_steady = false;
    std::vector<RefinementResult> results;

    std::vector<double> ux_L2_rates,  ux_H1_rates,  ux_Linf_rates;
    std::vector<double> uy_L2_rates,  uy_H1_rates,  uy_Linf_rates;
    std::vector<double> p_L2_rates,   p_Linf_rates;
    std::vector<double> div_L2_rates;

    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;
    unsigned int n_time_steps = 0;

    void compute_rates()
    {
        ux_L2_rates.clear();   ux_H1_rates.clear();   ux_Linf_rates.clear();
        uy_L2_rates.clear();   uy_H1_rates.clear();   uy_Linf_rates.clear();
        p_L2_rates.clear();    p_Linf_rates.clear();
        div_L2_rates.clear();

        for (size_t i = 1; i < results.size(); ++i)
        {
            const double log_h = std::log(results[i-1].h / results[i].h);
            auto rate = [&](double e_coarse, double e_fine) {
                return (e_coarse > 1e-15 && e_fine > 1e-15)
                    ? std::log(e_coarse / e_fine) / log_h : 0.0;
            };
            ux_L2_rates.push_back(rate(results[i-1].ux_L2, results[i].ux_L2));
            ux_H1_rates.push_back(rate(results[i-1].ux_H1, results[i].ux_H1));
            ux_Linf_rates.push_back(rate(results[i-1].ux_Linf, results[i].ux_Linf));
            uy_L2_rates.push_back(rate(results[i-1].uy_L2, results[i].uy_L2));
            uy_H1_rates.push_back(rate(results[i-1].uy_H1, results[i].uy_H1));
            uy_Linf_rates.push_back(rate(results[i-1].uy_Linf, results[i].uy_Linf));
            p_L2_rates.push_back(rate(results[i-1].p_L2, results[i].p_L2));
            p_Linf_rates.push_back(rate(results[i-1].p_Linf, results[i].p_Linf));
            div_L2_rates.push_back(rate(results[i-1].div_L2, results[i].div_L2));
        }
    }

    void print() const
    {
        std::cout << "\n--- " << phase_label
                  << " (Q" << degree_velocity << "/DG-Q" << degree_pressure
                  << ") ---\n";
        if (n_time_steps > 0)
            std::cout << "  Time steps: " << n_time_steps << "\n";

        std::cout << std::left
                  << std::setw(5)  << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "ux_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ux_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ux_Linf"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "uy_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "p_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "p_Linf"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "div_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(10) << "wall(s)"
                  << "\n";
        std::cout << std::string(175, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            auto pr = [&](double val, const std::vector<double>& rates) {
                std::cout << std::scientific << std::setprecision(2)
                          << std::setw(12) << val
                          << std::fixed << std::setprecision(2)
                          << std::setw(8) << (i > 0 ? rates[i-1] : 0.0);
            };

            std::cout << std::left << std::setw(5) << r.refinement
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.h;
            pr(r.ux_L2,   ux_L2_rates);
            pr(r.ux_H1,   ux_H1_rates);
            pr(r.ux_Linf, ux_Linf_rates);
            pr(r.uy_L2,   uy_L2_rates);
            pr(r.p_L2,    p_L2_rates);
            pr(r.p_Linf,  p_Linf_rates);
            pr(r.div_L2,  div_L2_rates);
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.total_time
                      << "\n";
        }
    }

    void write_csv(const std::string& filepath) const
    {
        std::ofstream f(filepath);
        f << "refinement,h,n_dofs,"
          << "ux_L2,ux_L2_rate,ux_H1,ux_H1_rate,ux_Linf,ux_Linf_rate,"
          << "uy_L2,uy_L2_rate,uy_H1,uy_H1_rate,uy_Linf,uy_Linf_rate,"
          << "p_L2,p_L2_rate,p_Linf,p_Linf_rate,"
          << "div_L2,div_L2_rate,walltime\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            auto csv_val = [&](double val, const std::vector<double>& rates) {
                f << std::scientific << std::setprecision(6) << val << ","
                  << std::fixed << std::setprecision(3)
                  << (i > 0 ? rates[i-1] : 0.0) << ",";
            };

            f << r.refinement << ","
              << std::scientific << std::setprecision(6) << r.h << ","
              << r.n_dofs << ",";
            csv_val(r.ux_L2,   ux_L2_rates);
            csv_val(r.ux_H1,   ux_H1_rates);
            csv_val(r.ux_Linf, ux_Linf_rates);
            csv_val(r.uy_L2,   uy_L2_rates);
            csv_val(r.uy_H1,   uy_H1_rates);
            csv_val(r.uy_Linf, uy_Linf_rates);
            csv_val(r.p_L2,    p_L2_rates);
            csv_val(r.p_Linf,  p_Linf_rates);
            csv_val(r.div_L2,  div_L2_rates);
            f << std::fixed << std::setprecision(4) << r.total_time << "\n";
        }
        std::cout << "  CSV written: " << filepath << "\n";
    }

    bool passes(double tol = 0.3) const
    {
        if (ux_H1_rates.empty()) return false;

        const double expected_H1 = static_cast<double>(degree_velocity);      // 2.0
        const double expected_p  = static_cast<double>(degree_pressure + 1);   // 2.0

        bool ok = (ux_H1_rates.back() >= expected_H1 - tol)
               && (uy_H1_rates.back() >= expected_H1 - tol)
               && (p_L2_rates.back()  >= expected_p  - tol);

        if (is_steady)
        {
            const double expected_L2 = static_cast<double>(degree_velocity + 1);  // 3.0
            ok = ok && (ux_L2_rates.back() >= expected_L2 - tol)
                    && (uy_L2_rates.back() >= expected_L2 - tol);
        }

        return ok;
    }
};


// ============================================================================
// Helpers
// ============================================================================
template <int dim>
void fill_domain_lengths(const Parameters& params, double L[dim])
{
    L[0] = params.domain.x_max - params.domain.x_min;
    if constexpr (dim >= 2) L[1] = params.domain.y_max - params.domain.y_min;
    if constexpr (dim >= 3) L[2] = 1.0;
}

template <int dim>
void create_mesh(
    dealii::parallel::distributed::Triangulation<dim>& triangulation,
    const Parameters& params,
    unsigned int refinement)
{
    dealii::Point<dim> p1, p2;
    std::vector<unsigned int> subdivisions(dim);

    p1[0] = params.domain.x_min;
    p2[0] = params.domain.x_max;
    subdivisions[0] = params.domain.initial_cells_x;

    if constexpr (dim >= 2)
    {
        p1[1] = params.domain.y_min;
        p2[1] = params.domain.y_max;
        subdivisions[1] = params.domain.initial_cells_y;
    }
    if constexpr (dim >= 3)
    {
        p1[2] = 0.0;
        p2[2] = 1.0;
        subdivisions[2] = params.domain.initial_cells_x;
    }

    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);
}

template <int dim>
double compute_min_h(
    const dealii::parallel::distributed::Triangulation<dim>& triangulation,
    MPI_Comm mpi_comm)
{
    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    double global_min_h;
    MPI_Allreduce(&local_min_h, &global_min_h, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);
    return global_min_h;
}


// ============================================================================
// collect_errors() — unified error computation via facade
// ============================================================================
template <int dim>
void collect_errors(
    NSSubsystem<dim>& ns,
    double current_time,
    double Ly,
    unsigned int refinement,
    const dealii::parallel::distributed::Triangulation<dim>& triangulation,
    MPI_Comm mpi_comm,
    RefinementResult& result)
{
    ns.update_ghosts();

    // Write VTK snapshot of final solution
    {
        std::string vtk_dir = "../navier_stokes_results/vtk/"
                            + get_timestamp() + "_navier_stokes";
        ns.write_vtu(vtk_dir, refinement, current_time);
    }

    NSMMSErrors errors = compute_ns_mms_errors<dim>(
        ns, current_time, Ly, mpi_comm);

    result.ux_L2   = errors.ux_L2;
    result.ux_H1   = errors.ux_H1;
    result.ux_Linf = errors.ux_Linf;
    result.uy_L2   = errors.uy_L2;
    result.uy_H1   = errors.uy_H1;
    result.uy_Linf = errors.uy_Linf;
    result.p_L2    = errors.p_L2;
    result.p_Linf  = errors.p_Linf;
    result.div_L2  = errors.div_U_L2;

    result.h = compute_min_h<dim>(triangulation, mpi_comm);
}


// ============================================================================
// Phase runners — each returns RefinementResult for a single refinement level
//
// All use NSSubsystem facade: setup() → assemble_stokes() → solve()
// ============================================================================

// --- Phase A: Steady Stokes (no time, no convection) ---
template <int dim>
RefinementResult run_phase_A(unsigned int refinement, const Parameters& params,
                             MPI_Comm mpi_comm)
{
    RefinementResult result;
    result.refinement = refinement;
    auto t0 = std::chrono::high_resolution_clock::now();

    const double t_eval = 1.0;
    const double nu = params.physics.nu_water;
    double L[dim];
    fill_domain_lengths<dim>(params, L);
    const double Ly = (dim >= 2) ? L[1] : 1.0;

    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    create_mesh<dim>(triangulation, params, refinement);

    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    ns.initialize_zero();

    std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
        body_force = [&](const dealii::Point<dim>& p, double t) {
            return NSMMS::source_phase_A<dim>(p, t, nu, Ly);
        };

    ns.assemble_stokes(1.0, nu, false, false, &body_force, t_eval);
    ns.solve();

    collect_errors<dim>(ns, t_eval, Ly, refinement, triangulation, mpi_comm, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// --- Phase B: Unsteady Stokes (time stepping, no convection) ---
template <int dim>
RefinementResult run_phase_B(unsigned int refinement, const Parameters& params,
                             unsigned int n_time_steps, MPI_Comm mpi_comm)
{
    RefinementResult result;
    result.refinement = refinement;
    auto t0 = std::chrono::high_resolution_clock::now();

    const double t_init = 0.1, t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double nu = params.physics.nu_water;
    double L[dim];
    fill_domain_lengths<dim>(params, L);
    const double Ly = (dim >= 2) ? L[1] : 1.0;

    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    create_mesh<dim>(triangulation, params, refinement);

    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    NSMMSInitialUx<dim> ic_ux(t_init, Ly);
    NSMMSInitialUy<dim> ic_uy(t_init, Ly);
    ns.initialize_velocity(ic_ux, ic_uy);

    double current_time = t_init;
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        const double t_old = current_time;
        current_time += dt;

        std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
            body_force = [&](const dealii::Point<dim>& p, double t) {
                return NSMMS::source_phase_B<dim>(p, t, t_old, nu, Ly);
            };

        ns.assemble_stokes(dt, nu, true, false, &body_force, current_time);
        ns.solve();
        ns.advance_time();
    }

    collect_errors<dim>(ns, current_time, Ly, refinement, triangulation, mpi_comm, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// --- Phase C: Steady NS (convection, no time stepping) ---
template <int dim>
RefinementResult run_phase_C(unsigned int refinement, const Parameters& params,
                             MPI_Comm mpi_comm)
{
    RefinementResult result;
    result.refinement = refinement;
    auto t0 = std::chrono::high_resolution_clock::now();

    const double t_eval = 1.0;
    const double nu = params.physics.nu_water;
    double L[dim];
    fill_domain_lengths<dim>(params, L);
    const double Ly = (dim >= 2) ? L[1] : 1.0;

    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    create_mesh<dim>(triangulation, params, refinement);

    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    ns.initialize_zero();

    NSMMSInitialUx<dim> ux_exact(t_eval, Ly);
    NSMMSInitialUy<dim> uy_exact(t_eval, Ly);
    ns.set_old_velocity(ux_exact, uy_exact);

    std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
        body_force = [&](const dealii::Point<dim>& p, double t) {
            return NSMMS::source_phase_C<dim>(p, t, nu, Ly);
        };

    ns.assemble_stokes(1.0, nu, false, true, &body_force, t_eval);
    ns.solve();

    collect_errors<dim>(ns, t_eval, Ly, refinement, triangulation, mpi_comm, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// --- Phase D: Unsteady NS (full production) ---
template <int dim>
RefinementResult run_phase_D(unsigned int refinement, const Parameters& params,
                             unsigned int n_time_steps, MPI_Comm mpi_comm)
{
    RefinementResult result;
    result.refinement = refinement;
    auto t0 = std::chrono::high_resolution_clock::now();

    const double t_init = 0.1, t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double nu = params.physics.nu_water;
    double L[dim];
    fill_domain_lengths<dim>(params, L);
    const double Ly = (dim >= 2) ? L[1] : 1.0;

    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
    create_mesh<dim>(triangulation, params, refinement);

    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    NSMMSInitialUx<dim> ic_ux(t_init, Ly);
    NSMMSInitialUy<dim> ic_uy(t_init, Ly);
    ns.initialize_velocity(ic_ux, ic_uy);

    double current_time = t_init;
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        const double t_old = current_time;
        current_time += dt;

        std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
            body_force = [&](const dealii::Point<dim>& p, double t) {
                return NSMMS::source_phase_D<dim>(p, t, t_old, nu, Ly);
            };

        ns.assemble_stokes(dt, nu, true, true, &body_force, current_time);
        ns.solve();
        ns.advance_time();
    }

    collect_errors<dim>(ns, current_time, Ly, refinement, triangulation, mpi_comm, result);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}


// ============================================================================
// run_phase() — Run a single phase over all refinement levels
// ============================================================================
template <int dim>
PhaseResult run_phase(
    const std::string& phase,
    const std::vector<unsigned int>& refinements,
    unsigned int n_time_steps,
    const Parameters& params,
    MPI_Comm mpi_comm)
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    PhaseResult pr;
    pr.phase_id        = phase;
    pr.degree_velocity = params.fe.degree_velocity;
    pr.degree_pressure = params.fe.degree_pressure;

    if (phase == "A")
    {
        pr.phase_label  = "Phase A: Steady Stokes";
        pr.is_steady    = true;
        pr.n_time_steps = 0;
    }
    else if (phase == "B")
    {
        pr.phase_label  = "Phase B: Unsteady Stokes";
        pr.is_steady    = false;
        pr.n_time_steps = n_time_steps;
    }
    else if (phase == "C")
    {
        pr.phase_label  = "Phase C: Steady NS";
        pr.is_steady    = true;
        pr.n_time_steps = 0;
    }
    else
    {
        pr.phase_label  = "Phase D: Unsteady NS";
        pr.is_steady    = false;
        pr.n_time_steps = n_time_steps;
    }

    if (rank == 0)
    {
        std::cout << "\n========================================\n";
        std::cout << " " << pr.phase_label << " (" << dim << "D)\n";
        std::cout << "========================================\n";
    }

    for (unsigned int ref : refinements)
    {
        if (rank == 0)
            std::cout << "  Refinement " << ref << "..." << std::flush;

        RefinementResult r;

        if (phase == "A")
            r = run_phase_A<dim>(ref, params, mpi_comm);
        else if (phase == "B")
            r = run_phase_B<dim>(ref, params, n_time_steps, mpi_comm);
        else if (phase == "C")
            r = run_phase_C<dim>(ref, params, mpi_comm);
        else
            r = run_phase_D<dim>(ref, params, n_time_steps, mpi_comm);

        pr.results.push_back(r);

        if (rank == 0)
        {
            std::cout << " ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                      << ", ux_H1=" << r.ux_H1
                      << ", p_L2=" << r.p_L2
                      << ", div=" << r.div_L2
                      << std::fixed << std::setprecision(1)
                      << ", time=" << r.total_time << "s\n";
        }
    }

    pr.compute_rates();
    return pr;
}


// ============================================================================
// run_all_phases() — Run selected phases for a given dimension
//
// Returns true if all phases pass convergence checks.
// ============================================================================
template <int dim>
bool run_all_phases(
    const std::vector<std::string>& phases,
    const std::vector<unsigned int>& refinements,
    unsigned int n_time_steps,
    const Parameters& params,
    MPI_Comm mpi_comm,
    const std::string& out_dir)
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   Navier-Stokes MMS Validation (" << dim << "D)\n";
        std::cout << "============================================================\n";
        std::cout << "  MPI ranks:    "
                  << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
        std::cout << "  FE degrees:   velocity Q" << params.fe.degree_velocity
                  << ", pressure DG-Q" << params.fe.degree_pressure << "\n";
        std::cout << "  ν = " << params.physics.nu_water << "\n";
        std::cout << "  Phases:       ";
        for (const auto& p : phases) std::cout << p << " ";
        std::cout << "\n";
        std::cout << "  Refinements:  ";
        for (auto r : refinements) std::cout << r << " ";
        std::cout << "\n";

        bool has_unsteady = std::find(phases.begin(), phases.end(), "B") != phases.end()
                         || std::find(phases.begin(), phases.end(), "D") != phases.end();
        if (has_unsteady)
            std::cout << "  Time steps:   " << n_time_steps
                      << ", t ∈ [0.1, 0.2]\n";
        std::cout << "============================================================\n";
    }

    std::vector<PhaseResult> all_results;
    bool all_pass = true;

    for (const auto& phase : phases)
    {
        PhaseResult pr = run_phase<dim>(
            phase, refinements, n_time_steps, params, mpi_comm);
        all_results.push_back(pr);

        if (rank == 0)
        {
            pr.print();

            // Write CSV
            std::string phase_lower = phase;
            std::transform(phase_lower.begin(), phase_lower.end(),
                           phase_lower.begin(), ::tolower);
            const std::string csv_name = timestamped_filename(
                "ns_mms_" + std::to_string(dim) + "d_phase_" + phase_lower, ".csv");
            pr.write_csv(out_dir + "/" + csv_name);

            // Expected rates
            double total_wall = 0.0;
            for (const auto& r : pr.results) total_wall += r.total_time;

            std::cout << "\nExpected (" << pr.phase_label << "): "
                      << "vel_H1 ~ O(h^" << params.fe.degree_velocity
                      << "), p_L2 ~ O(h^" << (params.fe.degree_pressure + 1) << ")";
            if (pr.is_steady)
                std::cout << ", vel_L2 ~ O(h^"
                          << (params.fe.degree_velocity + 1) << ")";
            else
                std::cout << "  (vel_L2 limited by O(dt))";
            std::cout << "  |  Wall time: " << std::fixed
                      << std::setprecision(1) << total_wall << "s\n";

            if (pr.passes())
                std::cout << "[PASS] " << pr.phase_label << "\n";
            else
            {
                std::cout << "[FAIL] " << pr.phase_label << "\n";
                all_pass = false;
            }
        }

        // Broadcast pass/fail
        {
            int local_pass = pr.passes() ? 1 : 0;
            int global_pass;
            MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, mpi_comm);
            if (!global_pass) all_pass = false;
        }
    }

    // ========================================================================
    // Summary table
    // ========================================================================
    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   SUMMARY (" << dim << "D)\n";
        std::cout << "============================================================\n";
        std::cout << std::left
                  << std::setw(30) << "Phase"
                  << std::setw(12) << "ux_L2_rate"
                  << std::setw(12) << "ux_H1_rate"
                  << std::setw(13) << "ux_Linf_rate"
                  << std::setw(12) << "uy_H1_rate"
                  << std::setw(12) << "p_L2_rate"
                  << std::setw(13) << "p_Linf_rate"
                  << std::setw(8)  << "Result"
                  << "\n";
        std::cout << std::string(112, '-') << "\n";

        for (const auto& pr : all_results)
        {
            std::cout << std::left << std::setw(30) << pr.phase_label
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << (pr.ux_L2_rates.empty()   ? 0.0 : pr.ux_L2_rates.back())
                      << std::setw(12) << (pr.ux_H1_rates.empty()   ? 0.0 : pr.ux_H1_rates.back())
                      << std::setw(13) << (pr.ux_Linf_rates.empty() ? 0.0 : pr.ux_Linf_rates.back())
                      << std::setw(12) << (pr.uy_H1_rates.empty()   ? 0.0 : pr.uy_H1_rates.back())
                      << std::setw(12) << (pr.p_L2_rates.empty()    ? 0.0 : pr.p_L2_rates.back())
                      << std::setw(13) << (pr.p_Linf_rates.empty()  ? 0.0 : pr.p_Linf_rates.back())
                      << (pr.passes() ? "[PASS]" : "[FAIL]")
                      << "\n";
        }

        std::cout << std::string(112, '-') << "\n";
        std::cout << "Expected:  vel_H1 ≥ " << std::fixed << std::setprecision(1)
                  << static_cast<double>(params.fe.degree_velocity) - 0.3
                  << ",  p_L2 ≥ "
                  << static_cast<double>(params.fe.degree_pressure + 1) - 0.3
                  << "\n";
    }

    return all_pass;
}


// ============================================================================
// CLI argument parsing
// ============================================================================
static std::vector<std::string> get_args_after(
    int argc, char* argv[], const std::string& flag)
{
    std::vector<std::string> result;
    bool found = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == flag) { found = true; continue; }
        if (found)
        {
            if (argv[i][0] == '-' && !std::isdigit(argv[i][1]))
                break;
            result.push_back(argv[i]);
        }
    }
    return result;
}

static unsigned int get_uint_arg(
    int argc, char* argv[], const std::string& flag, unsigned int default_val)
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag)
            return static_cast<unsigned int>(std::atoi(argv[i + 1]));
    return default_val;
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    dealii::deallog.depth_console(0);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    try
    {
        // ====================================================================
        // Parse command line
        // ====================================================================
        Parameters params;  // MMS tests use defaults (nu=1, domain [0,1]x[0,0.6])

        // Phases
        std::vector<std::string> phases;
        {
            auto args = get_args_after(argc, argv, "--phase");
            if (args.empty())
                phases = {"A", "B", "C", "D"};
            else
                phases = args;
        }

        // Refinements
        std::vector<unsigned int> refinements;
        {
            auto args = get_args_after(argc, argv, "--ref");
            if (args.empty())
                refinements = {2, 3, 4, 5, 6};
            else
                for (const auto& s : args)
                    refinements.push_back(static_cast<unsigned int>(std::atoi(s.c_str())));
        }

        // Time steps
        const unsigned int n_time_steps = get_uint_arg(argc, argv, "--steps", 10);

        // Dimension
        const unsigned int requested_dim = get_uint_arg(argc, argv, "--dim", 2);

        // Output directory
        const std::string out_dir = "../navier_stokes_results/mms";
        if (rank == 0)
            std::system(("mkdir -p " + out_dir).c_str());

        // ====================================================================
        // Help
        // ====================================================================
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--help")
            {
                if (rank == 0)
                {
                    std::cout << "Usage: mpirun -np N ./navier_stokes_main [options]\n"
                              << "\n"
                              << "Options:\n"
                              << "  --phase A B C D   Phases to run (default: all four)\n"
                              << "  --ref 2 3 4 5     Refinement levels (default: 2 3 4 5)\n"
                              << "  --steps N         Time steps for unsteady phases (default: 10)\n"
                              << "  --dim D           Spatial dimension: 2 or 3 (default: 2)\n"
                              << "\n"
                              << "Phases:\n"
                              << "  A  Steady Stokes        (viscous + pressure)\n"
                              << "  B  Unsteady Stokes      (+ time derivative)\n"
                              << "  C  Steady Navier-Stokes (+ convection)\n"
                              << "  D  Unsteady NS          (full production)\n"
                              << "\n"
                              << "Expected rates (Q2/DG-Q1):\n"
                              << "  vel L2 ~ O(h^3), vel H1 ~ O(h^2), p L2 ~ O(h^2)\n"
                              << "  (Unsteady: vel L2 limited by O(dt) at fine h)\n";
                }
                return 0;
            }
        }

        // ====================================================================
        // Run validation
        // ====================================================================
        bool all_pass = true;

        // --- 2D validation ---
        if (requested_dim == 2 || requested_dim == 0)
        {
            bool pass_2d = run_all_phases<2>(
                phases, refinements, n_time_steps, params, mpi_comm, out_dir);
            all_pass = all_pass && pass_2d;
        }

        // --- 3D validation ---
        // NOTE: NSSubsystem currently supports 2 velocity components (ux, uy).
        // For proper 3D validation, the facade needs uz support:
        //   - uz_dof_handler_, uz_solution_, uz_old_solution_, uz_relevant_
        //   - uz_constraints_, uz_to_ns_map_
        //   - 4-block saddle-point system [ux, uy, uz, p]
        //   - Assembly of all cross-coupling blocks
        //   - MMS exact solution for uz (e.g., uz=0 for z-independent test)
        //
        // When facade is extended, uncomment the block below and add
        // uz_val/uz_grad to navier_stokes_mms.h.
        if (requested_dim == 3)
        {
            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   3D Validation: NOT YET SUPPORTED\n";
                std::cout << "============================================================\n";
                std::cout << "  NSSubsystem currently has 2 velocity components (ux, uy).\n";
                std::cout << "  3D requires uz_dof_handler + 4-block saddle-point system.\n";
                std::cout << "\n";
                std::cout << "  To enable 3D:\n";
                std::cout << "    1. Add uz members to NSSubsystem (header + setup + assemble + solve)\n";
                std::cout << "    2. Add uz_val/uz_grad to navier_stokes_mms.h\n";
                std::cout << "    3. Extend compute_ns_mms_errors() for uz component\n";
                std::cout << "    4. Uncomment 3D runner in this file\n";
                std::cout << "============================================================\n";
            }

            // When ready:
            // bool pass_3d = run_all_phases<3>(
            //     phases, refinements, n_time_steps, params, mpi_comm, out_dir);
            // all_pass = all_pass && pass_3d;
        }

        // ====================================================================
        // Final verdict
        // ====================================================================
        if (rank == 0)
        {
            std::cout << "\n============================================================\n";
            if (all_pass)
                std::cout << "   [ALL PASS] NS MMS Validation PASSED\n";
            else
                std::cout << "   [FAIL] NS MMS Validation FAILED\n";
            std::cout << "============================================================\n";
        }

        return all_pass ? 0 : 1;
    }
    catch (std::exception& e)
    {
        std::cerr << "\n[Error] " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n[Error] Unknown exception!\n";
        return 1;
    }
}