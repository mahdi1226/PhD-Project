// ============================================================================
// solvers/ch_solver.cc - Parallel Cahn-Hilliard Solver (Trilinos)
//
// Uses Trilinos GMRES + AMG for iterative, or Amesos direct solvers.
//
// IMPORTANT FIXES (Phase 1):
//  1) Removed unsafe "if ||rhs|| small then solution=0" shortcut.
//  2) MPI-correct extraction: use a ghosted coupled solution vector
//     constructed with (owned, relevant) index sets.
// ============================================================================

#include "solvers/ch_solver.h"
#include "solvers/solver_info.h"

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/utilities.h>

#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// Helper: Try a specific direct solver type
// Returns true if successful, false otherwise
// ============================================================================
static bool try_direct_solver(
    const std::string& solver_type,
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    int rank,
    bool verbose)
{
    if (verbose && rank == 0)
        std::cout << "[CH Direct] Trying " << solver_type << "...\n";

    try
    {
        dealii::SolverControl solver_control(1, 0);  // Direct = 1 iteration
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = solver_type;

        dealii::TrilinosWrappers::SolverDirect solver(solver_control, data);

        if (verbose && rank == 0)
            std::cout << "[CH Direct]   Initializing (symbolic + numeric factorization)...\n";

        solver.initialize(matrix);

        if (verbose && rank == 0)
            std::cout << "[CH Direct]   Solving...\n";

        solver.solve(solution, rhs);

        if (verbose && rank == 0)
            std::cout << "[CH Direct] SUCCESS with " << solver_type << "\n";

        return true;
    }
    catch (dealii::ExceptionBase& e)
    {
        if (verbose && rank == 0)
            std::cout << "[CH Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (std::exception& e)
    {
        if (verbose && rank == 0)
            std::cout << "[CH Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (...)
    {
        if (verbose && rank == 0)
            std::cout << "[CH Direct]   " << solver_type << " failed: unknown exception\n";
        return false;
    }
}

// ============================================================================
// solve_ch_system - Solve and extract in one call
// ============================================================================
SolverInfo solve_ch_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& ch_locally_relevant,  // <-- NEW (required for MPI-correct extraction)
    const dealii::IndexSet& theta_locally_owned,
    const dealii::IndexSet& psi_locally_owned,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm,
    bool verbose)
{
    SolverInfo info;
    info.solver_name = "CH";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    auto start = std::chrono::high_resolution_clock::now();

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Owned-only coupled solution (this is what solvers fill efficiently)
    dealii::TrilinosWrappers::MPI::Vector coupled_solution(ch_locally_owned, mpi_comm);
    coupled_solution = 0;

    const double rhs_norm = rhs.l2_norm();

    // Setup solver control (relative tolerance based on ||rhs||)
    dealii::SolverControl solver_control(
        params.max_iterations,
        params.rel_tolerance * rhs_norm,
        /*log_history=*/false,
        /*log_result=*/false);

    bool converged = false;

    // ========================================================================
    // Direct solver (if requested)
    // ========================================================================
    if (!params.use_iterative)
    {
        if (verbose && rank == 0)
        {
            std::cout << "[CH Direct] Matrix size: " << info.matrix_size
                      << " x " << matrix.n()
                      << ", nnz: " << info.nnz << "\n";
        }

        if (!converged)
            converged = try_direct_solver("Amesos_Mumps", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Umfpack", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Superludist", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Superlu", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Klu", matrix, rhs, coupled_solution, rank, verbose);

        if (converged)
        {
            info.iterations = 1;
            info.residual = 0.0;
            info.converged = true;
            info.used_direct = true;

            if (verbose && rank == 0)
                std::cout << "[CH] Direct solver completed\n";
        }
    }

    // ========================================================================
    // Iterative solver (GMRES + AMG)
    // ========================================================================
    if (!converged && params.use_iterative)
    {
        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = params.gmres_restart;
        gmres_data.right_preconditioning = true;

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);

        dealii::TrilinosWrappers::PreconditionAMG preconditioner;
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 1e-4;
        amg_data.elliptic = false;
        amg_data.higher_order_elements = true;

        try
        {
            preconditioner.initialize(matrix, amg_data);

            solver.solve(matrix, coupled_solution, rhs, preconditioner);

            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = true;
            converged = true;
        }
        catch (dealii::SolverControl::NoConvergence&)
        {
            info.iterations = solver_control.last_step();
            info.residual = solver_control.last_value();
            info.converged = false;

            if (verbose && rank == 0)
            {
                std::cerr << "[CH] GMRES failed after " << info.iterations
                          << " iterations, residual = " << info.residual << "\n";
            }
        }
        catch (std::exception& e)
        {
            if (verbose && rank == 0)
                std::cerr << "[CH] Exception: " << e.what() << "\n";
            info.converged = false;
        }
    }

    // ========================================================================
    // Fallback to direct if iterative failed
    // ========================================================================
    if (!converged && params.fallback_to_direct)
    {
        if (verbose && rank == 0)
            std::cerr << "[CH] Falling back to direct solver\n";

        if (!converged)
            converged = try_direct_solver("Amesos_Mumps", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Umfpack", matrix, rhs, coupled_solution, rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Klu", matrix, rhs, coupled_solution, rank, verbose);

        if (converged)
        {
            info.iterations = 1;
            info.residual = 0.0;
            info.converged = true;
            info.used_direct = true;
        }
    }

    // Always distribute constraints after solve (also correct for MMS/Dirichlet)
    constraints.distribute(coupled_solution);

    // ------------------------------------------------------------------------
    // MPI-correct extraction:
    // create ghosted view of coupled_solution so operator[] is valid
    // for indices that are locally relevant but not owned.
    // ------------------------------------------------------------------------
    dealii::TrilinosWrappers::MPI::Vector coupled_solution_relevant(
        ch_locally_owned, ch_locally_relevant, mpi_comm);
    coupled_solution_relevant = coupled_solution; // imports ghost values

    // Extract θ and ψ from ghosted coupled solution
    for (auto idx = theta_locally_owned.begin(); idx != theta_locally_owned.end(); ++idx)
        theta_solution[*idx] = coupled_solution_relevant[theta_to_ch_map[*idx]];
    for (auto idx = psi_locally_owned.begin(); idx != psi_locally_owned.end(); ++idx)
        psi_solution[*idx] = coupled_solution_relevant[psi_to_ch_map[*idx]];

    theta_solution.compress(dealii::VectorOperation::insert);
    psi_solution.compress(dealii::VectorOperation::insert);

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    if (verbose && rank == 0)
    {
        std::cout << "[CH] iterations=" << info.iterations
                  << ", residual=" << std::scientific << info.residual
                  << ", time=" << std::fixed << std::setprecision(2)
                  << info.solve_time << "s\n";
    }

    return info;
}