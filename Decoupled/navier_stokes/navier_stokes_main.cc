// ============================================================================
// navier_stokes/navier_stokes_main.cc — NS Standalone Driver
//
// Modes:
//   mms       MMS spatial convergence (2D), refs 2-6
//   2d        Single 2D run with VTK output
//   3d        3D NS (not yet implemented)
//   temporal  Temporal convergence study (2D), sweep dt
//
// Usage:
//   mpirun -np 4 ./navier_stokes_main --mode mms
//   mpirun -np 4 ./navier_stokes_main --mode 2d --refinement 4
//   mpirun -np 4 ./navier_stokes_main --mode 3d
//   mpirun -np 4 ./navier_stokes_main --mode temporal
//   mpirun -np 4 ./navier_stokes_main --ref 2 3 4 5
//   mpirun -np 4 ./navier_stokes_main --steps 20
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-42f
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "navier_stokes/tests/navier_stokes_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/read_write_vector.h>

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

// ============================================================================
// Result structures
// ============================================================================
struct NSMMSResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;
    double ux_L2  = 0.0, ux_H1  = 0.0;
    double uy_L2  = 0.0, uy_H1  = 0.0;
    double p_L2   = 0.0, div_L2 = 0.0;
    double total_time = 0.0;
};

struct NSMMSConvergenceResult
{
    std::vector<NSMMSResult> results;
    std::vector<double> ux_L2_rates, ux_H1_rates;
    std::vector<double> uy_L2_rates, uy_H1_rates;
    std::vector<double> p_L2_rates,  div_L2_rates;
    unsigned int degree_velocity = 2;
    unsigned int degree_pressure = 1;
    unsigned int n_time_steps = 10;

    void compute_rates()
    {
        ux_L2_rates.clear();  ux_H1_rates.clear();
        uy_L2_rates.clear();  uy_H1_rates.clear();
        p_L2_rates.clear();   div_L2_rates.clear();

        for (size_t i = 1; i < results.size(); ++i)
        {
            const double log_h = std::log(results[i-1].h / results[i].h);
            auto rate = [&](double e_coarse, double e_fine) {
                return (e_coarse > 1e-15 && e_fine > 1e-15)
                    ? std::log(e_coarse / e_fine) / log_h : 0.0;
            };
            ux_L2_rates.push_back(rate(results[i-1].ux_L2, results[i].ux_L2));
            ux_H1_rates.push_back(rate(results[i-1].ux_H1, results[i].ux_H1));
            uy_L2_rates.push_back(rate(results[i-1].uy_L2, results[i].uy_L2));
            uy_H1_rates.push_back(rate(results[i-1].uy_H1, results[i].uy_H1));
            p_L2_rates.push_back(rate(results[i-1].p_L2,   results[i].p_L2));
            div_L2_rates.push_back(rate(results[i-1].div_L2, results[i].div_L2));
        }
    }

    void print() const
    {
        std::cout << "\n--- NS MMS Convergence (Q"
                  << degree_velocity << "/DG-Q" << degree_pressure
                  << ", Unsteady NS) ---\n";
        std::cout << std::left
                  << std::setw(5)  << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "ux_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ux_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "uy_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "uy_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "p_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "div_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(10) << "wall(s)"
                  << "\n";
        std::cout << std::string(155, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            std::cout << std::left << std::setw(5) << r.refinement
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.h
                      << std::setw(12) << r.ux_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.ux_H1
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_H1_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.uy_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? uy_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.uy_H1
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? uy_H1_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.p_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? p_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.div_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? div_L2_rates[i-1] : 0.0)
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.total_time
                      << "\n";
        }
    }

    void write_csv(const std::string& filepath) const
    {
        std::ofstream f(filepath);
        f << "refinement,h,n_dofs,"
          << "ux_L2,ux_L2_rate,ux_H1,ux_H1_rate,"
          << "uy_L2,uy_L2_rate,uy_H1,uy_H1_rate,"
          << "p_L2,p_L2_rate,div_L2,div_L2_rate,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            f << r.refinement << ","
              << std::scientific << std::setprecision(6) << r.h << ","
              << r.n_dofs << ","
              << r.ux_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? ux_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.ux_H1 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? ux_H1_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.uy_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? uy_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.uy_H1 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? uy_H1_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.p_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? p_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.div_L2 << ","
              << std::fixed << std::setprecision(3) << (i > 0 ? div_L2_rates[i-1] : 0.0) << ","
              << std::fixed << std::setprecision(4) << r.total_time << "\n";
        }
        std::cout << "  CSV written: " << filepath << "\n";
    }

    bool passes(double tol = 0.3) const
    {
        if (ux_H1_rates.empty()) return false;
        const double expected_H1 = static_cast<double>(degree_velocity);
        const double expected_p  = static_cast<double>(degree_pressure + 1);
        return (ux_H1_rates.back() >= expected_H1 - tol)
            && (uy_H1_rates.back() >= expected_H1 - tol)
            && (p_L2_rates.back()  >= expected_p  - tol);
    }
};


// ============================================================================
// Assembly verification diagnostic
//
// Injects the exact MMS solution into the assembled system and computes
//   r = A * x_exact - b
// to isolate whether the bug is in assembly vs. error extraction.
//
// This is called AFTER assemble_stokes() but BEFORE solve(), so the
// matrix is in its raw (un-pinned) saddle-point form.
//
// Interpretation:
//   r_ux, r_uy small (< h²): Momentum equation assembled correctly
//   r_p small (< h²):         Continuity (B block) assembled correctly
//   All small → bug is in error computation/extraction, not assembly
//   r_ux, r_uy large → momentum assembly bug (A, B^T, or f)
//   r_p large → continuity assembly bug (B block)
// ============================================================================
template <int dim>
void assembly_verification_diagnostic(
    const NSSubsystem<dim>& ns,
    double current_time,
    double Ly,
    unsigned int refinement,
    MPI_Comm mpi_comm)
{
    const int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Accessor aliases
    const auto& ux_dh = ns.get_ux_dof_handler();
    const auto& uy_dh = ns.get_uy_dof_handler();
    const auto& p_dh  = ns.get_p_dof_handler();

    const auto& ux_owned = ns.get_ux_locally_owned();
    const auto& uy_owned = ns.get_uy_locally_owned();
    const auto& p_owned  = ns.get_p_locally_owned();
    const auto& ns_owned = ns.get_ns_locally_owned();

    const auto& ux_map = ns.get_ux_to_ns_map();
    const auto& uy_map = ns.get_uy_to_ns_map();
    const auto& p_map  = ns.get_p_to_ns_map();

    const auto& ns_constraints = ns.get_ns_constraints();
    const auto& ns_matrix      = ns.get_ns_matrix();
    const auto& ns_rhs         = ns.get_ns_rhs();

    // ========================================================================
    // 1. Project/interpolate exact solutions onto FE spaces
    //    Velocity (FE_Q): use interpolate (has support points)
    //    Pressure (FE_DGP): use project (no support points)
    // ========================================================================
    NSMMSInitialUx<dim> ux_exact_func(current_time, Ly);
    NSMMSInitialUy<dim> uy_exact_func(current_time, Ly);
    NSMMSExactP<dim>    p_exact_func(current_time, Ly);

    dealii::TrilinosWrappers::MPI::Vector ux_exact(ux_owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector uy_exact(uy_owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector p_exact(p_owned, mpi_comm);

    dealii::VectorTools::interpolate(ux_dh, ux_exact_func, ux_exact);
    dealii::VectorTools::interpolate(uy_dh, uy_exact_func, uy_exact);

    // FE_DGP has no support points — use L2 projection instead
    {
        dealii::AffineConstraints<double> empty_constraints;
        empty_constraints.close();
        dealii::QGauss<dim> quad(p_dh.get_fe().degree + 2);
        dealii::VectorTools::project(p_dh, empty_constraints, quad,
                                     p_exact_func, p_exact);
    }

    // ========================================================================
    // 2. Map component vectors into monolithic x_exact
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector x_exact(ns_owned, mpi_comm);
    x_exact = 0;

    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
        x_exact[ux_map[*it]] = ux_exact[*it];

    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
        x_exact[uy_map[*it]] = uy_exact[*it];

    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        x_exact[p_map[*it]] = p_exact[*it];

    x_exact.compress(dealii::VectorOperation::insert);

    // ========================================================================
    // 3. Apply constraints to x_exact
    //    (sets boundary velocity DoFs to 0 = exact BC value)
    // ========================================================================
    ns_constraints.distribute(x_exact);

    // ========================================================================
    // 4. Compute residual: r = A * x_exact - b
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector residual(ns_owned, mpi_comm);
    ns_matrix.vmult(residual, x_exact);
    residual -= ns_rhs;

    // ========================================================================
    // 5. Extract component residual norms
    // ========================================================================
    double loc_r_ux_sq = 0.0, loc_r_uy_sq = 0.0, loc_r_p_sq = 0.0;
    double loc_r_ux_max = 0.0, loc_r_uy_max = 0.0, loc_r_p_max = 0.0;

    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
    {
        const double val = residual[ux_map[*it]];
        loc_r_ux_sq += val * val;
        loc_r_ux_max = std::max(loc_r_ux_max, std::abs(val));
    }
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
    {
        const double val = residual[uy_map[*it]];
        loc_r_uy_sq += val * val;
        loc_r_uy_max = std::max(loc_r_uy_max, std::abs(val));
    }
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
    {
        const double val = residual[p_map[*it]];
        loc_r_p_sq += val * val;
        loc_r_p_max = std::max(loc_r_p_max, std::abs(val));
    }

    // Global reductions
    double g_r_ux_sq = 0, g_r_uy_sq = 0, g_r_p_sq = 0;
    double g_r_ux_max = 0, g_r_uy_max = 0, g_r_p_max = 0;
    MPI_Allreduce(&loc_r_ux_sq,  &g_r_ux_sq,  1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_r_uy_sq,  &g_r_uy_sq,  1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_r_p_sq,   &g_r_p_sq,   1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_r_ux_max, &g_r_ux_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(&loc_r_uy_max, &g_r_uy_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(&loc_r_p_max,  &g_r_p_max,  1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    const double r_ux_l2 = std::sqrt(g_r_ux_sq);
    const double r_uy_l2 = std::sqrt(g_r_uy_sq);
    const double r_p_l2  = std::sqrt(g_r_p_sq);
    const double r_full  = residual.l2_norm();

    // Also report norms of RHS and x_exact for context
    const double rhs_norm = ns_rhs.l2_norm();
    const double x_norm   = x_exact.l2_norm();

    // ========================================================================
    // 6. Report
    // ========================================================================
    if (rank == 0)
    {
        std::cout << "\n"
                  << "  ============================================================\n"
                  << "  ASSEMBLY VERIFICATION DIAGNOSTIC (ref " << refinement << ")\n"
                  << "  ============================================================\n"
                  << "  Injecting exact MMS solution at t = " << current_time << "\n"
                  << "  Computing r = A * x_exact - b (BEFORE pressure pinning)\n"
                  << "\n"
                  << "  ||x_exact||_l2  = " << std::scientific << std::setprecision(3) << x_norm << "\n"
                  << "  ||rhs||_l2      = " << rhs_norm << "\n"
                  << "  ||r||_l2 (full) = " << r_full << "\n"
                  << "\n"
                  << "  Component residuals (l2 / Linf):\n"
                  << "    r_ux:  " << std::setw(12) << r_ux_l2
                  << "  /  " << std::setw(12) << g_r_ux_max << "\n"
                  << "    r_uy:  " << std::setw(12) << r_uy_l2
                  << "  /  " << std::setw(12) << g_r_uy_max << "\n"
                  << "    r_p:   " << std::setw(12) << r_p_l2
                  << "  /  " << std::setw(12) << g_r_p_max << "\n"
                  << "\n"
                  << "  Relative residuals (component / full):\n"
                  << "    r_ux / ||rhs|| = " << (rhs_norm > 1e-14 ? r_ux_l2 / rhs_norm : 0.0) << "\n"
                  << "    r_uy / ||rhs|| = " << (rhs_norm > 1e-14 ? r_uy_l2 / rhs_norm : 0.0) << "\n"
                  << "    r_p  / ||rhs|| = " << (rhs_norm > 1e-14 ? r_p_l2 / rhs_norm : 0.0) << "\n"
                  << "\n"
                  << "  Interpretation:\n"
                  << "    Small r_ux, r_uy, r_p → assembly correct, bug in error extraction\n"
                  << "    Large r_ux or r_uy    → momentum assembly bug (A, B^T, or f)\n"
                  << "    Large r_p             → continuity assembly bug (B block)\n"
                  << "  ============================================================\n\n"
                  << std::defaultfloat;
    }
}


// ============================================================================
// Post-solve pressure diagnostic
//
// Called AFTER solve() on the last time step.  Three-layer check:
//   Layer 1: DoF-level comparison (monolithic → extracted → exact)
//   Layer 2: Independent L2 error via quadrature (bypass compute_ns_mms_errors)
//   Layer 3: Mean-corrected L2 error via quadrature
//
// This isolates whether the bug is in:
//   - solve/pinning (monolithic values wrong)
//   - extract_solutions() (extracted ≠ monolithic)
//   - compute_ns_mms_errors() (L2 norm computation wrong)
// ============================================================================
template <int dim>
void post_solve_pressure_diagnostic(
    const NSSubsystem<dim>& ns,
    double current_time,
    double Ly,
    unsigned int refinement,
    MPI_Comm mpi_comm)
{
    const int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    const auto& p_dh     = ns.get_p_dof_handler();
    const auto& p_owned  = ns.get_p_locally_owned();
    const auto& p_rel_is = ns.get_p_locally_relevant();
    const auto& p_map    = ns.get_p_to_ns_map();
    const auto& ns_owned = ns.get_ns_locally_owned();

    // ========================================================================
    // Layer 1: DoF-level comparison — monolithic vs extracted vs exact
    // ========================================================================
    // Get monolithic solution values at pressure indices
    const auto& ns_sol = ns.get_ns_solution();  // owned monolithic
    const auto& p_sol  = ns.get_p_solution();    // owned extracted

    // Project exact pressure (FE_DGP has no support points for interpolate)
    NSMMSExactP<dim> p_exact_func(current_time, Ly);
    dealii::TrilinosWrappers::MPI::Vector p_exact(p_owned, mpi_comm);
    {
        dealii::AffineConstraints<double> empty_constraints;
        empty_constraints.close();
        dealii::QGauss<dim> quad(p_dh.get_fe().degree + 2);
        dealii::VectorTools::project(p_dh, empty_constraints, quad,
                                     p_exact_func, p_exact);
    }

    // Extract monolithic pressure values using ReadWriteVector
    dealii::IndexSet p_ns_indices(ns_sol.size());
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        p_ns_indices.add_index(p_map[*it]);
    p_ns_indices.compress();

    dealii::LinearAlgebra::ReadWriteVector<double> ns_vals(p_ns_indices);
    ns_vals.import_elements(ns_sol, dealii::VectorOperation::insert);

    // Print first 20 pressure DoFs on rank 0
    if (rank == 0)
    {
        std::cout << "\n"
                  << "  ============================================================\n"
                  << "  POST-SOLVE PRESSURE DIAGNOSTIC (ref " << refinement << ")\n"
                  << "  ============================================================\n"
                  << "\n  Layer 1: DoF-level comparison (first 20 pressure DoFs)\n"
                  << "  " << std::left
                  << std::setw(8)  << "p_dof"
                  << std::setw(10) << "ns_idx"
                  << std::setw(14) << "monolithic"
                  << std::setw(14) << "extracted"
                  << std::setw(14) << "exact"
                  << std::setw(14) << "mono-exact"
                  << std::setw(14) << "extr-exact"
                  << std::setw(14) << "mono-extr"
                  << "\n"
                  << "  " << std::string(100, '-') << "\n";

        unsigned int count = 0;
        for (auto it = p_owned.begin(); it != p_owned.end() && count < 20; ++it, ++count)
        {
            const auto p_dof  = *it;
            const auto ns_idx = p_map[p_dof];
            const double mono = ns_vals[ns_idx];
            const double extr = p_sol[p_dof];
            const double exact = p_exact[p_dof];

            std::cout << "  " << std::left
                      << std::setw(8)  << p_dof
                      << std::setw(10) << ns_idx
                      << std::scientific << std::setprecision(4)
                      << std::setw(14) << mono
                      << std::setw(14) << extr
                      << std::setw(14) << exact
                      << std::setw(14) << (mono - exact)
                      << std::setw(14) << (extr - exact)
                      << std::setw(14) << (mono - extr)
                      << "\n";
        }
    }

    // Global stats: max |mono - extr| and max |extr - exact|
    double loc_max_mono_extr = 0, loc_max_extr_exact = 0, loc_max_mono_exact = 0;
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
    {
        const double mono = ns_vals[p_map[*it]];
        const double extr = p_sol[*it];
        const double exact = p_exact[*it];
        loc_max_mono_extr  = std::max(loc_max_mono_extr,  std::abs(mono - extr));
        loc_max_extr_exact = std::max(loc_max_extr_exact, std::abs(extr - exact));
        loc_max_mono_exact = std::max(loc_max_mono_exact, std::abs(mono - exact));
    }
    double g_max_mono_extr = 0, g_max_extr_exact = 0, g_max_mono_exact = 0;
    MPI_Allreduce(&loc_max_mono_extr,  &g_max_mono_extr,  1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(&loc_max_extr_exact, &g_max_extr_exact, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(&loc_max_mono_exact, &g_max_mono_exact, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    if (rank == 0)
    {
        std::cout << "\n  Global max differences (all " << p_owned.n_elements() << " pressure DoFs):\n"
                  << "    max|monolithic - extracted| = " << std::scientific << std::setprecision(3) << g_max_mono_extr << "\n"
                  << "    max|extracted  - exact|     = " << g_max_extr_exact << "\n"
                  << "    max|monolithic - exact|     = " << g_max_mono_exact << "\n";

        if (g_max_mono_extr < 1e-12)
            std::cout << "    => extract_solutions() is CORRECT (mono == extr)\n";
        else
            std::cout << "    => BUG IN extract_solutions()! (mono != extr)\n";

        if (g_max_mono_exact < 1e-2)
            std::cout << "    => Solver produced correct pressure\n";
        else
            std::cout << "    => Solver/pinning produced WRONG pressure (mono != exact)\n";
    }

    // ========================================================================
    // Layer 2: Independent L2 error via quadrature (bypass compute_ns_mms_errors)
    // ========================================================================
    const auto& p_relevant = ns.get_p_relevant();  // ghosted

    dealii::QGauss<dim> quad(p_dh.get_fe().degree + 2);
    const unsigned int nq = quad.size();
    dealii::FEValues<dim> fv_p(p_dh.get_fe(), quad,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> p_vals(nq);

    // Pass 1: means
    double loc_p_num = 0, loc_p_exact = 0, loc_vol = 0;
    for (auto cell = p_dh.begin_active(); cell != p_dh.end(); ++cell)
    {
        if (!cell->is_locally_owned()) continue;
        fv_p.reinit(cell);
        fv_p.get_function_values(p_relevant, p_vals);
        for (unsigned int q = 0; q < nq; ++q)
        {
            const double w = fv_p.JxW(q);
            loc_p_num   += p_vals[q] * w;
            loc_p_exact += NSMMS::p_val<dim>(fv_p.quadrature_point(q), current_time, Ly) * w;
            loc_vol     += w;
        }
    }
    double g_p_num = 0, g_p_exact = 0, g_vol = 0;
    MPI_Allreduce(&loc_p_num,   &g_p_num,   1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_p_exact, &g_p_exact, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_vol,     &g_vol,     1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    const double mean_num = g_p_num / g_vol;
    const double mean_exact = g_p_exact / g_vol;

    // Pass 2: L2 errors (raw and mean-corrected)
    double loc_raw_sq = 0, loc_corr_sq = 0;
    for (auto cell = p_dh.begin_active(); cell != p_dh.end(); ++cell)
    {
        if (!cell->is_locally_owned()) continue;
        fv_p.reinit(cell);
        fv_p.get_function_values(p_relevant, p_vals);
        for (unsigned int q = 0; q < nq; ++q)
        {
            const double w = fv_p.JxW(q);
            const double pe = NSMMS::p_val<dim>(fv_p.quadrature_point(q), current_time, Ly);
            const double raw_diff  = p_vals[q] - pe;
            const double corr_diff = (p_vals[q] - mean_num) - (pe - mean_exact);
            loc_raw_sq  += raw_diff * raw_diff * w;
            loc_corr_sq += corr_diff * corr_diff * w;
        }
    }
    double g_raw_sq = 0, g_corr_sq = 0;
    MPI_Allreduce(&loc_raw_sq,  &g_raw_sq,  1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&loc_corr_sq, &g_corr_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    if (rank == 0)
    {
        std::cout << "\n  Layer 2: Independent L2 pressure error (quadrature)\n"
                  << "    mean(p_h)     = " << std::scientific << std::setprecision(6) << mean_num << "\n"
                  << "    mean(p_exact) = " << mean_exact << "\n"
                  << "    mean offset   = " << (mean_num - mean_exact) << "\n"
                  << "    ||p_h - p*||_L2 (raw)            = " << std::sqrt(g_raw_sq) << "\n"
                  << "    ||p_h - p*||_L2 (mean-corrected) = " << std::sqrt(g_corr_sq) << "\n"
                  << "  ============================================================\n\n"
                  << std::defaultfloat;
    }
}


// ============================================================================
// Single refinement test
// ============================================================================
template <int dim>
NSMMSResult run_ns_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    NSMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    const double t_init  = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double nu = params.physics.nu_water;
    const double Ly = params.domain.y_max - params.domain.y_min;

    // ========================================================================
    // Create distributed mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

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

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                       subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Create facade and setup
    // ========================================================================
    NSSubsystem<dim> ns(params, mpi_comm, triangulation);
    ns.setup();

    result.n_dofs = ns.get_ux_dof_handler().n_dofs()
                  + ns.get_uy_dof_handler().n_dofs()
                  + ns.get_p_dof_handler().n_dofs();

    // ========================================================================
    // Initialize velocity to exact solution at t_init
    // ========================================================================
    NSMMSInitialUx<dim> ic_ux(t_init, Ly);
    NSMMSInitialUy<dim> ic_uy(t_init, Ly);
    ns.initialize_velocity(ic_ux, ic_uy);

    // ========================================================================
    // Time stepping loop
    // ========================================================================
    double current_time = t_init;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        const double t_old = current_time;
        current_time += dt;

        // MMS source: captures t_old in closure
        std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>
            body_force = [&](const dealii::Point<dim>& p, double t) {
                return NSMMS::source_phase_D<dim>(p, t, t_old, nu, Ly);
            };

        ns.assemble_stokes(dt, nu,
                           /*include_time_derivative=*/ true,
                           /*include_convection=*/ true,
                           &body_force,
                           current_time);

        // ================================================================
        // Assembly verification diagnostic (last step only)
        // ================================================================
        if (step == n_time_steps - 1)
        {
            assembly_verification_diagnostic<dim>(
                ns, current_time, Ly, refinement, mpi_comm);
        }

        ns.solve();

        // ================================================================
        // Post-solve pressure diagnostic (last step only)
        // ================================================================
        if (step == n_time_steps - 1)
        {
            post_solve_pressure_diagnostic<dim>(
                ns, current_time, Ly, refinement, mpi_comm);
        }

        ns.advance_time();
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    ns.update_ghosts();

    // Write final solution to VTK for visualization
    ns.write_vtu("../navier_stokes_results/vtk", n_time_steps, current_time);

    NSMMSErrors errors = compute_ns_mms_errors<dim>(
        ns, current_time, Ly, mpi_comm);

    result.ux_L2  = errors.ux_L2;   result.ux_H1  = errors.ux_H1;
    result.uy_L2  = errors.uy_L2;   result.uy_H1  = errors.uy_H1;
    result.p_L2   = errors.p_L2;    result.div_L2 = errors.div_U_L2;

    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}


// ============================================================================
// Temporal convergence test
//
// Fix spatial ref=4, sweep dt with increasing n_time_steps.
// Measure MMS error at t_final, compute dt convergence rate.
// Expected: O(dt^1) for backward Euler.
// ============================================================================
template <int dim>
bool run_ns_temporal_convergence(
    const Parameters& params,
    MPI_Comm mpi_comm)
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int ref = 4;
    const std::vector<unsigned int> steps_list = {5, 10, 20, 40, 80};

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "   NS Temporal Convergence (ref=" << ref << ")\n";
        std::cout << "============================================================\n";
    }

    std::vector<double> dts, vel_L2_errors;

    for (unsigned int n_steps : steps_list)
    {
        const double dt = 0.1 / n_steps;  // t_final - t_init = 0.1

        if (rank == 0)
            std::cout << "  n_steps=" << n_steps << " (dt=" << std::scientific
                      << std::setprecision(2) << dt << ")... " << std::flush;

        NSMMSResult r = run_ns_mms_single<dim>(ref, params, n_steps, mpi_comm);

        dts.push_back(dt);
        vel_L2_errors.push_back(std::sqrt(r.ux_L2 * r.ux_L2 + r.uy_L2 * r.uy_L2));

        if (rank == 0)
            std::cout << "vel_L2=" << std::scientific << std::setprecision(2)
                      << vel_L2_errors.back() << "\n";
    }

    // Compute rates
    if (rank == 0)
    {
        std::cout << "\n  dt-convergence rates:\n";
        std::cout << std::setw(12) << "dt" << std::setw(14) << "vel_L2"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(34, '-') << "\n";

        bool pass = false;
        for (size_t i = 0; i < dts.size(); ++i)
        {
            double rate = 0.0;
            if (i > 0 && vel_L2_errors[i] > 1e-15 && vel_L2_errors[i-1] > 1e-15)
                rate = std::log(vel_L2_errors[i-1] / vel_L2_errors[i])
                     / std::log(dts[i-1] / dts[i]);

            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << dts[i]
                      << std::setw(14) << vel_L2_errors[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << rate << "\n";

            if (i == dts.size() - 1)
                pass = (rate >= 0.7);  // expect O(dt^1) for BDF1
        }

        std::cout << "\n  Expected: O(dt^1) for backward Euler\n";
        if (pass)
            std::cout << "  [PASS] Temporal convergence rate within tolerance!\n";
        else
            std::cout << "  [FAIL] Temporal convergence rate below expected!\n";

        return pass;
    }
    return true;
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
        Parameters params = Parameters::parse_command_line(argc, argv);
        const std::string& mode = params.run.mode;
        const unsigned int n_time_steps =
            (params.run.steps > 0) ? static_cast<unsigned int>(params.run.steps) : 10;

        constexpr int dim = 2;

        // ================================================================
        // Mode: mms — spatial convergence study (2D)
        // ================================================================
        if (mode == "mms")
        {
            NSMMSConvergenceResult conv;
            conv.degree_velocity = params.fe.degree_velocity;
            conv.degree_pressure = params.fe.degree_pressure;
            conv.n_time_steps = n_time_steps;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes MMS Verification (Unsteady NS)\n";
                std::cout << "============================================================\n";
                std::cout << "  MPI ranks:    "
                          << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
                std::cout << "  FE degrees:   velocity Q" << params.fe.degree_velocity
                          << ", pressure DG-Q" << params.fe.degree_pressure << "\n";
                std::cout << "  nu = " << params.physics.nu_water << "\n";
                std::cout << "  Time steps:   " << n_time_steps
                          << ", t in [0.1, 0.2]\n";
                std::cout << "  Refinements:  ";
                for (auto r : params.run.refs) std::cout << r << " ";
                std::cout << "\n";
                std::cout << "============================================================\n\n";
            }

            for (unsigned int ref : params.run.refs)
            {
                if (rank == 0)
                    std::cout << "  Refinement " << ref << "... " << std::flush;

                NSMMSResult r = run_ns_mms_single<dim>(ref, params, n_time_steps, mpi_comm);
                conv.results.push_back(r);

                if (rank == 0)
                {
                    std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                              << ", ux_H1=" << r.ux_H1
                              << ", uy_L2=" << r.uy_L2
                              << ", p_L2=" << r.p_L2
                              << ", div=" << r.div_L2
                              << ", time=" << std::fixed << std::setprecision(1)
                              << r.total_time << "s\n";
                }
            }

            conv.compute_rates();

            if (rank == 0)
            {
                conv.print();

                const std::string out_dir = "../navier_stokes_results/mms";
                std::system(("mkdir -p " + out_dir).c_str());

                const std::string csv_name = timestamped_filename(
                    "ns_mms_convergence", ".csv");
                conv.write_csv(out_dir + "/" + csv_name);

                double total_wall = 0.0;
                for (const auto& r : conv.results) total_wall += r.total_time;

                std::cout << "\nExpected: vel_H1 ~ O(h^" << params.fe.degree_velocity
                          << "), p_L2 ~ O(h^" << (params.fe.degree_pressure + 1) << ")"
                          << "  |  Total wall time: " << std::fixed << std::setprecision(1)
                          << total_wall << "s\n";
                std::cout << "  (First-order time stepping may limit velocity L2 rate)\n";

                if (conv.passes())
                    std::cout << "[PASS] Convergence rates within tolerance!\n";
                else
                    std::cout << "[FAIL] Convergence rates below expected!\n";
            }

            return conv.passes() ? 0 : 1;
        }
        // ================================================================
        // Mode: 2d — single run with VTK output
        // ================================================================
        else if (mode == "2d")
        {
            const unsigned int ref = params.mesh.initial_refinement;

            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes 2D — Single MMS run with VTK\n";
                std::cout << "  Refinement: " << ref
                          << ", time steps: " << n_time_steps << "\n";
                std::cout << "============================================================\n\n";
            }

            NSMMSResult r = run_ns_mms_single<dim>(ref, params, n_time_steps, mpi_comm);

            if (rank == 0)
            {
                std::cout << "  ux_L2=" << std::scientific << std::setprecision(3) << r.ux_L2
                          << ", ux_H1=" << r.ux_H1
                          << ", p_L2=" << r.p_L2
                          << ", wall=" << std::fixed << std::setprecision(1)
                          << r.total_time << "s\n";
                std::cout << "  VTK output: ../navier_stokes_results/vtk/\n";
            }
            return 0;
        }
        // ================================================================
        // Mode: 3d — not yet implemented
        // ================================================================
        else if (mode == "3d")
        {
            if (rank == 0)
            {
                std::cout << "\n============================================================\n";
                std::cout << "   Navier-Stokes 3D — Not yet implemented\n";
                std::cout << "============================================================\n";
                std::cout << "  3D NS requires uz_dof_handler + 4-block saddle-point system.\n";
                std::cout << "  This will be implemented in a future update.\n\n";
            }
            return 0;
        }
        // ================================================================
        // Mode: temporal — temporal convergence study
        // ================================================================
        else if (mode == "temporal")
        {
            bool pass = run_ns_temporal_convergence<dim>(params, mpi_comm);
            return pass ? 0 : 1;
        }
        else
        {
            if (rank == 0)
                std::cerr << "Unknown mode: " << mode
                          << " (use mms, 2d, 3d, or temporal)\n";
            return 1;
        }
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