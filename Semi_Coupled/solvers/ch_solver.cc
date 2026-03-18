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
#include "solvers/direct_solver_utils.h"

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/utilities.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>

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

        converged = DirectSolverUtils::try_direct_solver_chain(
            matrix, rhs, coupled_solution, "CH", rank, verbose);

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
    // Iterative solver (GMRES + AMG or ILU)
    // ========================================================================
    if (!converged && params.use_iterative)
    {
        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = params.gmres_restart;
        gmres_data.right_preconditioning = true;

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);

        // Select preconditioner based on params (AMG default, ILU for HPC)
        std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> preconditioner;

        if (params.preconditioner == LinearSolverParams::Preconditioner::ILU)
        {
            auto ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
            dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
            ilu_data.ilu_fill = 2;
            ilu_data.ilu_atol = 0.0;
            ilu_data.ilu_rtol = 1.0;
            ilu->initialize(matrix, ilu_data);
            preconditioner = std::move(ilu);

            if (verbose && rank == 0)
                std::cout << "[CH] Using ILU preconditioner\n";
        }
        else
        {
            auto amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
            dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
            amg_data.smoother_sweeps = 2;
            amg_data.aggregation_threshold = 0.02;    // Was 1e-4 (too aggressive for indefinite θ-ψ)
            amg_data.elliptic = false;                 // CH is indefinite (not SPD)
            amg_data.higher_order_elements = true;     // Q2 elements
            amg_data.smoother_type = "Chebyshev";      // Better than Gauss-Seidel for indefinite
            amg->initialize(matrix, amg_data);
            preconditioner = std::move(amg);

            if (verbose && rank == 0)
                std::cout << "[CH] Using AMG preconditioner (Chebyshev smoother)\n";
        }

        try
        {
            solver.solve(matrix, coupled_solution, rhs, *preconditioner);

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

        converged = DirectSolverUtils::try_direct_solver_chain(
            matrix, rhs, coupled_solution, "CH", rank, verbose);

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