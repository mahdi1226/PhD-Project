// ============================================================================
// mms/poisson/poisson_mms_test.cc - Poisson MMS Test Implementation (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses distributed triangulation
//   - Uses Trilinos matrix/vectors
//   - MPI reductions for global error norms
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/poisson/poisson_mms_test.h"
#include "mms/poisson/poisson_mms.h"

// Production code
#include "setup/poisson_setup.h"
#include "assembly/poisson_assembler.h"
#include "solvers/poisson_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

constexpr int dim = 2;

// ============================================================================
// Helper: Convert enum to string
// ============================================================================

std::string to_string(PoissonSolverType type)
{
    switch (type)
    {
        case PoissonSolverType::AMG:    return "AMG";
        case PoissonSolverType::Direct: return "Direct";
        default: return "Unknown";
    }
}

// ============================================================================
// PoissonMMSConvergenceResult Implementation
// ============================================================================

void PoissonMMSConvergenceResult::compute_rates()
{
    L2_rates.clear();
    H1_rates.clear();

    for (size_t i = 1; i < results.size(); ++i)
    {
        const double h_fine = results[i].h;
        const double h_coarse = results[i-1].h;

        if (results[i-1].L2_error > 1e-15 && results[i].L2_error > 1e-15)
        {
            L2_rates.push_back(
                std::log(results[i-1].L2_error / results[i].L2_error)
                / std::log(h_coarse / h_fine));
        }
        else
        {
            L2_rates.push_back(0.0);
        }

        if (results[i-1].H1_error > 1e-15 && results[i].H1_error > 1e-15)
        {
            H1_rates.push_back(
                std::log(results[i-1].H1_error / results[i].H1_error)
                / std::log(h_coarse / h_fine));
        }
        else
        {
            H1_rates.push_back(0.0);
        }
    }
}

bool PoissonMMSConvergenceResult::passes(double tolerance) const
{
    if (L2_rates.empty() || H1_rates.empty())
        return false;

    return (L2_rates.back() >= expected_L2_rate - tolerance) &&
           (H1_rates.back() >= expected_H1_rate - tolerance);
}

void PoissonMMSConvergenceResult::print() const
{
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  POISSON MMS CONVERGENCE RESULTS (PARALLEL)\n";
    std::cout << "================================================================\n";
    std::cout << "  Solver: " << to_string(solver_type) << "\n";
    std::cout << "  FE degree: Q" << fe_degree << "\n";
    std::cout << "  Expected rates: L2 = " << expected_L2_rate
              << ", H1 = " << expected_H1_rate << "\n";
    std::cout << "================================================================\n\n";

    std::cout << std::left
              << std::setw(5)  << "Ref"
              << std::setw(10) << "DoFs"
              << std::setw(12) << "h"
              << std::setw(12) << "L2 error"
              << std::setw(8)  << "rate"
              << std::setw(12) << "H1 error"
              << std::setw(8)  << "rate"
              << std::setw(8)  << "iters"
              << std::setw(10) << "time(s)"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];

        std::cout << std::left
                  << std::setw(5)  << r.refinement
                  << std::setw(10) << r.n_dofs
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.h
                  << std::setw(12) << r.L2_error
                  << std::fixed << std::setprecision(2)
                  << std::setw(8)  << (i > 0 ? L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.H1_error
                  << std::fixed << std::setprecision(2)
                  << std::setw(8)  << (i > 0 ? H1_rates[i-1] : 0.0)
                  << std::setw(8)  << r.solver_iterations
                  << std::setw(10) << r.total_time
                  << "\n";
    }

    std::cout << "\n";
    if (!L2_rates.empty())
    {
        std::cout << "Asymptotic rates: L2 = " << std::fixed << std::setprecision(2)
                  << L2_rates.back() << ", H1 = " << H1_rates.back() << "\n";
        std::cout << "STATUS: " << (passes(0.3) ? "PASS" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

void PoissonMMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    file << "refinement,n_dofs,h,L2_error,L2_rate,H1_error,H1_rate,"
         << "iterations,total_time,solver_type\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        file << r.refinement << ","
             << r.n_dofs << ","
             << std::scientific << std::setprecision(6) << r.h << ","
             << r.L2_error << ","
             << std::fixed << std::setprecision(3)
             << (i > 0 ? L2_rates[i-1] : 0.0) << ","
             << std::scientific << std::setprecision(6) << r.H1_error << ","
             << std::fixed << std::setprecision(3)
             << (i > 0 ? H1_rates[i-1] : 0.0) << ","
             << r.solver_iterations << ","
             << std::fixed << std::setprecision(4) << r.total_time << ","
             << to_string(r.solver_type) << "\n";
    }

    file.close();
    std::cout << "[MMS] Results written to " << filename << "\n";
}

// ============================================================================
// Run single Poisson MMS test (PARALLEL)
// ============================================================================

PoissonMMSResult run_poisson_mms_single(
    unsigned int refinement,
    const Parameters& params,
    PoissonSolverType solver_type,
    MPI_Comm mpi_communicator)
{
    PoissonMMSResult result;
    result.refinement = refinement;
    result.solver_type = solver_type;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    auto total_start = std::chrono::high_resolution_clock::now();

    const double time = 1.0;  // MMS solution amplitude
    Parameters mutable_params = params;
    mutable_params.enable_mms = true;

    const double L_y = params.domain.y_max - params.domain.y_min;

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
    // Setup DoFs
    // ========================================================================
    dealii::FE_Q<dim> fe(params.fe.degree_potential);
    dealii::DoFHandler<dim> phi_dof_handler(triangulation);
    phi_dof_handler.distribute_dofs(fe);

    result.n_dofs = phi_dof_handler.n_dofs();

    // Get min h (global)
    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // Build IndexSets
    dealii::IndexSet phi_locally_owned = phi_dof_handler.locally_owned_dofs();
    dealii::IndexSet phi_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler);

    // ========================================================================
    // Setup constraints and sparsity - PRODUCTION CODE
    // ========================================================================
    auto setup_start = std::chrono::high_resolution_clock::now();

    dealii::AffineConstraints<double> phi_constraints;
    dealii::TrilinosWrappers::SparseMatrix phi_matrix;

    setup_poisson_constraints_and_sparsity<dim>(
        phi_dof_handler,
        phi_locally_owned, phi_locally_relevant,
        phi_constraints, phi_matrix,
        mpi_communicator, pcout);

    auto setup_end = std::chrono::high_resolution_clock::now();
    result.setup_time = std::chrono::duration<double>(setup_end - setup_start).count();

    // ========================================================================
    // Initialize vectors
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector phi_rhs(phi_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_solution(phi_locally_owned, mpi_communicator);
    phi_solution = 0;

    // Empty M vectors for standalone test (need a dummy DoFHandler)
    dealii::FE_DGQ<dim> fe_dg(0);
    dealii::DoFHandler<dim> M_dof_handler(triangulation);
    M_dof_handler.distribute_dofs(fe_dg);
    dealii::TrilinosWrappers::MPI::Vector mx_empty, my_empty;

    // ========================================================================
    // PRODUCTION ASSEMBLY
    // ========================================================================
    auto asm_start = std::chrono::high_resolution_clock::now();

    assemble_poisson_system<dim>(
        phi_dof_handler, M_dof_handler,
        mx_empty, my_empty,
        mutable_params, time,
        phi_constraints,
        phi_matrix, phi_rhs);

    auto asm_end = std::chrono::high_resolution_clock::now();
    result.assembly_time = std::chrono::duration<double>(asm_end - asm_start).count();

    // ========================================================================
    // PRODUCTION SOLVE
    // ========================================================================
    auto solve_start = std::chrono::high_resolution_clock::now();

    LinearSolverParams solver_params;
    solver_params.type = LinearSolverParams::Type::CG;
    solver_params.preconditioner = LinearSolverParams::Preconditioner::AMG;
    solver_params.rel_tolerance = 1e-10;
    solver_params.abs_tolerance = 1e-12;
    solver_params.max_iterations = 500;
    solver_params.use_iterative = (solver_type == PoissonSolverType::AMG);
    solver_params.fallback_to_direct = true;

    SolverInfo solver_info = solve_poisson_system(
        phi_matrix, phi_rhs, phi_solution,
        phi_constraints, phi_locally_owned,
        solver_params, mpi_communicator, /*log_output=*/false);

    auto solve_end = std::chrono::high_resolution_clock::now();
    result.solve_time = std::chrono::duration<double>(solve_end - solve_start).count();

    result.solver_iterations = solver_info.iterations;
    result.solver_residual = solver_info.residual;
    result.used_direct_fallback = solver_info.used_direct;

    // ========================================================================
    // Compute errors (PARALLEL)
    // ========================================================================
    // Need ghosted vector for reading
    dealii::TrilinosWrappers::MPI::Vector phi_relevant(
        phi_locally_owned, phi_locally_relevant, mpi_communicator);
    phi_relevant = phi_solution;

    PoissonMMSError errors = compute_poisson_mms_errors_parallel<dim>(
        phi_dof_handler, phi_relevant, time, L_y, mpi_communicator);

    result.L2_error = errors.L2_error;
    result.H1_error = errors.H1_error;
    result.Linf_error = errors.Linf_error;

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// Public interface - convergence study
// ============================================================================

PoissonMMSConvergenceResult run_poisson_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    PoissonSolverType solver_type,
    MPI_Comm mpi_communicator)
{
    PoissonMMSConvergenceResult result;
    result.fe_degree = params.fe.degree_potential;
    result.expected_L2_rate = params.fe.degree_potential + 1;
    result.expected_H1_rate = params.fe.degree_potential;
    result.solver_type = solver_type;
    result.standalone = true;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[POISSON_MMS] Running parallel convergence study...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  Solver: " << to_string(solver_type) << "\n";
        std::cout << "  FE degree: Q" << params.fe.degree_potential << "\n";
        std::cout << "  Expected: L2 = " << result.expected_L2_rate
                  << ", H1 = " << result.expected_H1_rate << "\n";
        std::cout << "  Using PRODUCTION: poisson_setup + poisson_assembler + poisson_solver\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        PoissonMMSResult r = run_poisson_mms_single(ref, params, solver_type, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "L2=" << std::scientific << std::setprecision(2) << r.L2_error
                      << ", H1=" << r.H1_error
                      << ", iters=" << r.solver_iterations
                      << ", time=" << std::fixed << std::setprecision(2) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}