// ============================================================================
// mms/magnetic/magnetic_mms_test.cc - Monolithic Magnetics MMS Test (PARALLEL)
//
// Self-contained parallel test using PRODUCTION:
//   - setup/magnetic_setup.h
//   - assembly/magnetic_assembler.h
//   - solvers/magnetic_solver.h
//
// Tests the combined M+phi block system from equations (42c)-(42d).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/magnetic/magnetic_mms_test.h"
#include "mms/magnetic/magnetic_mms.h"

// Production code
#include "setup/magnetic_setup.h"
#include "assembly/magnetic_assembler.h"
#include "solvers/magnetic_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

constexpr int dim = 2;

// ============================================================================
// MagneticMMSConvergenceResult implementation
// ============================================================================
void MagneticMMSConvergenceResult::compute_rates()
{
    auto compute_rate = [](double e_coarse, double e_fine,
                           double h_coarse, double h_fine) -> double
    {
        if (e_coarse > 1e-15 && e_fine > 1e-15)
            return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
        return 0.0;
    };

    M_L2_rates.clear();
    M_Linf_rates.clear();
    M_H1_rates.clear();
    phi_L2_rates.clear();
    phi_Linf_rates.clear();
    phi_H1_rates.clear();

    for (size_t i = 1; i < results.size(); ++i)
    {
        const double h_c = results[i-1].h;
        const double h_f = results[i].h;

        M_L2_rates.push_back(compute_rate(
            results[i-1].M_L2, results[i].M_L2, h_c, h_f));
        M_Linf_rates.push_back(compute_rate(
            results[i-1].M_Linf, results[i].M_Linf, h_c, h_f));
        M_H1_rates.push_back(compute_rate(
            results[i-1].M_H1, results[i].M_H1, h_c, h_f));
        phi_L2_rates.push_back(compute_rate(
            results[i-1].phi_L2, results[i].phi_L2, h_c, h_f));
        phi_Linf_rates.push_back(compute_rate(
            results[i-1].phi_Linf, results[i].phi_Linf, h_c, h_f));
        phi_H1_rates.push_back(compute_rate(
            results[i-1].phi_H1, results[i].phi_H1, h_c, h_f));
    }
}

bool MagneticMMSConvergenceResult::passes(double tolerance) const
{
    if (M_L2_rates.empty() || phi_L2_rates.empty() || phi_H1_rates.empty())
        return false;

    const double M_rate = M_L2_rates.back();
    const double phi_L2_rate = phi_L2_rates.back();
    const double phi_H1_rate = phi_H1_rates.back();

    return M_rate >= expected_M_L2_rate - tolerance &&
           phi_L2_rate >= expected_phi_L2_rate - tolerance &&
           phi_H1_rate >= expected_phi_H1_rate - tolerance;
}

void MagneticMMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "Monolithic Magnetics MMS Convergence\n";
    std::cout << "========================================\n";
    std::cout << "FE: M=DG" << degree_M << ", phi=CG" << degree_phi << "\n";
    std::cout << "Expected rates: M_L2=" << expected_M_L2_rate
              << ", phi_L2=" << expected_phi_L2_rate
              << ", phi_H1=" << expected_phi_H1_rate << "\n\n";

    // --- M table ---
    std::cout << "Magnetization (DG" << degree_M << "):\n";
    std::cout << std::left
              << std::setw(5) << "Ref"
              << std::setw(10) << "h"
              << std::setw(12) << "M_L2"
              << std::setw(7) << "rate"
              << std::setw(12) << "M_Linf"
              << std::setw(7) << "rate"
              << std::setw(12) << "M_H1"
              << std::setw(7) << "rate"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(5) << r.refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.h
                  << std::setw(12) << r.M_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? M_L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.M_Linf
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? M_Linf_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.M_H1
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? M_H1_rates[i-1] : 0.0)
                  << "\n";
    }

    // --- phi table ---
    std::cout << "\nPotential phi (CG" << degree_phi << "):\n";
    std::cout << std::left
              << std::setw(5) << "Ref"
              << std::setw(10) << "h"
              << std::setw(12) << "phi_L2"
              << std::setw(7) << "rate"
              << std::setw(12) << "phi_Linf"
              << std::setw(7) << "rate"
              << std::setw(12) << "phi_H1"
              << std::setw(7) << "rate"
              << std::setw(8) << "time"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(5) << r.refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.h
                  << std::setw(12) << r.phi_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? phi_L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.phi_Linf
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? phi_Linf_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.phi_H1
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << (i > 0 ? phi_H1_rates[i-1] : 0.0)
                  << std::fixed << std::setprecision(1)
                  << std::setw(8) << r.total_time
                  << "\n";
    }

    std::cout << "========================================\n";
    if (passes())
        std::cout << "[PASS] All convergence rates within tolerance!\n";
    else
    {
        std::cout << "[FAIL] Rates below expected!\n";
        if (!M_L2_rates.empty())
        {
            std::cout << "  M_L2 rate:   " << std::fixed << std::setprecision(2)
                      << M_L2_rates.back() << " (expected " << expected_M_L2_rate << ")\n";
            std::cout << "  phi_L2 rate: " << phi_L2_rates.back()
                      << " (expected " << expected_phi_L2_rate << ")\n";
            std::cout << "  phi_H1 rate: " << phi_H1_rates.back()
                      << " (expected " << expected_phi_H1_rate << ")\n";
        }
    }
}

void MagneticMMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[Magnetic MMS] Failed to open " << filename << "\n";
        return;
    }

    file << "refinement,h,n_dofs,"
         << "M_L2,M_L2_rate,M_Linf,M_Linf_rate,M_H1,M_H1_rate,Mx_L2,My_L2,"
         << "phi_L2,phi_L2_rate,phi_Linf,phi_Linf_rate,phi_H1,phi_H1_rate,"
         << "total_time\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        file << r.refinement << ","
             << std::scientific << std::setprecision(6) << r.h << ","
             << r.n_dofs << ","
             << r.M_L2 << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? M_L2_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6)
             << r.M_Linf << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? M_Linf_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6)
             << r.M_H1 << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? M_H1_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6)
             << r.Mx_L2 << ","
             << r.My_L2 << ","
             << r.phi_L2 << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? phi_L2_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6)
             << r.phi_Linf << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? phi_Linf_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6)
             << r.phi_H1 << ","
             << std::fixed << std::setprecision(2)
             << (i > 0 ? phi_H1_rates[i-1] : 0.0) << ","
             << std::fixed << std::setprecision(4) << r.total_time << "\n";
    }

    file.close();
    std::cout << "[Magnetic MMS] Results written to " << filename << "\n";
}

// ============================================================================
// run_magnetic_mms_single - Self-contained parallel test
// ============================================================================
MagneticMMSResult run_magnetic_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    MagneticMMSResult result;
    result.refinement = refinement;

    dealii::ConditionalOStream pcout(std::cout,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Create distributed mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params.domain.initial_cells_x;
    subdivisions[1] = params.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Setup magnetics DoFHandler: FESystem (DG^dim + CG)
    // ========================================================================
    dealii::FESystem<dim> fe_mag(
        dealii::FE_DGQ<dim>(params.fe.degree_magnetization), dim,
        dealii::FE_Q<dim>(params.fe.degree_potential), 1);

    dealii::DoFHandler<dim> mag_dof_handler(triangulation);
    mag_dof_handler.distribute_dofs(fe_mag);

    // CRITICAL: component_wise renumbering before setup_magnetic_system
    dealii::DoFRenumbering::component_wise(mag_dof_handler);

    dealii::IndexSet mag_locally_owned = mag_dof_handler.locally_owned_dofs();
    dealii::IndexSet mag_locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(mag_dof_handler);

    result.n_dofs = mag_dof_handler.n_dofs();

    // ========================================================================
    // Setup constraints and sparsity (PRODUCTION)
    // ========================================================================
    dealii::AffineConstraints<double> mag_constraints;
    dealii::TrilinosWrappers::SparseMatrix mag_matrix;

    setup_magnetic_system<dim>(
        mag_dof_handler, mag_locally_owned, mag_locally_relevant,
        mag_constraints, mag_matrix, mpi_communicator, pcout);

    // ========================================================================
    // Setup dummy U=0, theta=1 DoFHandlers + vectors
    // (Required by assembler's synchronized cell iteration)
    // ========================================================================
    dealii::FE_Q<dim> fe_CG(params.fe.degree_velocity);

    dealii::DoFHandler<dim> U_dof_handler(triangulation);
    dealii::DoFHandler<dim> theta_dof_handler(triangulation);
    U_dof_handler.distribute_dofs(fe_CG);
    theta_dof_handler.distribute_dofs(fe_CG);

    dealii::IndexSet U_locally_owned = U_dof_handler.locally_owned_dofs();
    dealii::IndexSet U_locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(U_dof_handler);
    dealii::IndexSet theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    dealii::IndexSet theta_locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);

    dealii::TrilinosWrappers::MPI::Vector Ux(U_locally_owned, U_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Uy(U_locally_owned, U_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_locally_owned, theta_locally_relevant, mpi_communicator);

    Ux = 0.0;
    Uy = 0.0;
    theta_vec = 1.0;  // Full ferrofluid phase

    // ========================================================================
    // Create vectors for magnetics system
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector mag_solution(mag_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector mag_old(mag_locally_owned, mag_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector system_rhs(mag_locally_owned, mpi_communicator);

    // ========================================================================
    // Initialize with exact solution at t_start
    //
    // MagneticExactSolution provides (Mx, My, phi) for the FESystem.
    // VectorTools::interpolate handles both DG (cell-local) and CG (shared)
    // components correctly on distributed meshes.
    // ========================================================================
    {
        MagneticExactSolution<dim> exact_init(t_start, L_y);
        dealii::VectorTools::interpolate(mag_dof_handler, exact_init, mag_solution);
        mag_constraints.distribute(mag_solution);
        mag_old = mag_solution;
    }

    // Compute min h
    {
        double local_min_h = std::numeric_limits<double>::max();
        for (const auto& cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                local_min_h = std::min(local_min_h, cell->diameter());
        MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);
    }

    // ========================================================================
    // PRODUCTION assembler and solver
    // ========================================================================
    MagneticAssembler<dim> assembler(
        mms_params, mag_dof_handler, U_dof_handler,
        theta_dof_handler, mag_constraints, mpi_communicator);

    MagneticSolver<dim> solver(mag_locally_owned, mpi_communicator);

    // ========================================================================
    // Time stepping
    // ========================================================================
    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Update old solution
        mag_old = mag_solution;

        // Assemble monolithic system
        assembler.assemble(
            mag_matrix, system_rhs,
            Ux, Uy, theta_vec, mag_old,
            dt, current_time);

        // Solve with MUMPS
        solver.solve(mag_matrix, mag_solution, system_rhs);

        // Enforce constraints (pinned phi DoF + hanging nodes)
        mag_constraints.distribute(mag_solution);
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector mag_ghosted(
        mag_locally_owned, mag_locally_relevant, mpi_communicator);
    mag_ghosted = mag_solution;

    MagneticMMSError errors = compute_magnetic_mms_errors_parallel<dim>(
        mag_dof_handler, mag_ghosted, current_time, L_y, mpi_communicator);

    result.Mx_L2 = errors.Mx_L2;
    result.My_L2 = errors.My_L2;
    result.M_L2 = errors.M_L2;
    result.M_H1 = errors.M_H1;
    result.M_Linf = errors.M_Linf;
    result.phi_L2 = errors.phi_L2;
    result.phi_H1 = errors.phi_H1;
    result.phi_Linf = errors.phi_Linf;

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// run_magnetic_mms_standalone - Full convergence study
// ============================================================================
MagneticMMSConvergenceResult run_magnetic_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    MagneticMMSConvergenceResult result;
    result.degree_M = params.fe.degree_magnetization;
    result.degree_phi = params.fe.degree_potential;
    result.expected_M_L2_rate = params.fe.degree_magnetization + 1;
    result.expected_phi_L2_rate = params.fe.degree_potential + 1;
    result.expected_phi_H1_rate = params.fe.degree_potential;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[MAGNETIC_MMS] Running monolithic magnetics convergence study...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  Solver: MUMPS (direct)\n";
        std::cout << "  FE: M=DG" << result.degree_M << ", phi=CG" << result.degree_phi << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Expected rates: M_L2=" << result.expected_M_L2_rate
                  << ", phi_L2=" << result.expected_phi_L2_rate
                  << ", phi_H1=" << result.expected_phi_H1_rate << "\n";
        std::cout << "  Using PRODUCTION: magnetic_setup + magnetic_assembler + magnetic_solver\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        MagneticMMSResult r = run_magnetic_mms_single(
            ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "M_L2=" << std::scientific << std::setprecision(2) << r.M_L2
                      << ", phi_L2=" << r.phi_L2
                      << ", phi_H1=" << r.phi_H1
                      << ", time=" << std::fixed << std::setprecision(1)
                      << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}
