// ============================================================================
// cahn_hilliard/cahn_hilliard_solve.cc - Coupled System Solver
//
// Solves the monolithic [θ ψ]^T system using direct solvers, then extracts
// individual θ and ψ solutions via the index maps.
//
// Direct solver chain: Amesos_Mumps → Amesos_Superludist → Amesos_Klu
//
// Iterative fallback: GMRES + AMG (for large problems, not default).
// The coupled CH system is indefinite (saddle-point-like), so direct
// solvers are preferred and more reliable.
//
// Extraction requires a ghosted coupled vector to safely read off-process
// values via the index maps.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include <chrono>
#include <iomanip>

// For suppressing MUMPS stderr output
#include <unistd.h>
#include <fcntl.h>

// ============================================================================
// Helper: Try a specific Amesos direct solver
// ============================================================================
namespace
{
    bool try_direct_solver(
        const std::string& solver_type,
        const dealii::TrilinosWrappers::SparseMatrix& matrix,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        dealii::TrilinosWrappers::MPI::Vector& solution,
        int rank,
        bool verbose,
        bool suppress_output = false)
    {
        if (verbose && rank == 0)
            std::cout << "[CH Direct] Trying " << solver_type << "...\n";

        // Optionally suppress stderr (MUMPS prints errors there even on
        // recoverable failures like workspace underestimation)
        int saved_stderr = -1;
        if (suppress_output)
        {
            fflush(stderr);
            saved_stderr = dup(STDERR_FILENO);
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0)
            {
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
        }

        bool success = false;
        try
        {
            dealii::SolverControl solver_control(1, 0);
            dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
            data.solver_type = solver_type;

            dealii::TrilinosWrappers::SolverDirect solver(solver_control, data);
            solver.initialize(matrix);
            solver.solve(solution, rhs);

            success = true;
            if (verbose && rank == 0)
                std::cout << "[CH Direct] SUCCESS with " << solver_type << "\n";
        }
        catch (...)
        {
            if (verbose && rank == 0)
                std::cout << "[CH Direct]   " << solver_type << " failed\n";
        }

        // Restore stderr
        if (saved_stderr >= 0)
        {
            fflush(stderr);
            dup2(saved_stderr, STDERR_FILENO);
            close(saved_stderr);
        }

        return success;
    }
} // anonymous namespace

// ============================================================================
// Solve coupled system and extract θ, ψ
// ============================================================================
template <int dim>
SolverInfo CahnHilliardSubsystem<dim>::solve_coupled_system()
{
    SolverInfo info;
    info.solver_name = "CH";
    info.matrix_size = system_matrix_.m();
    info.nnz = system_matrix_.n_nonzero_elements();

    auto t_start = std::chrono::high_resolution_clock::now();

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);
    const bool verbose = params_.solvers.ch.verbose;

    // Owned-only coupled solution
    dealii::TrilinosWrappers::MPI::Vector coupled_solution(
        ch_locally_owned_, mpi_comm_);
    coupled_solution = 0;

    const double rhs_norm = system_rhs_.l2_norm();
    bool converged = false;

    // ========================================================================
    // Direct solver (default for CH — indefinite system)
    // ========================================================================
    if (!params_.solvers.ch.use_iterative)
    {
        if (verbose && rank == 0)
        {
            pcout_ << "[CH Direct] Matrix size: " << info.matrix_size
                   << " x " << system_matrix_.n()
                   << ", nnz: " << info.nnz
                   << ", ||rhs|| = " << std::scientific << rhs_norm << "\n";
        }

        if (!converged)
            converged = try_direct_solver("Amesos_Mumps", system_matrix_,
                                           system_rhs_, coupled_solution,
                                           rank, verbose,
                                           /*suppress_output=*/true);
        if (!converged)
            converged = try_direct_solver("Amesos_Superludist", system_matrix_,
                                           system_rhs_, coupled_solution,
                                           rank, verbose);
        if (!converged)
            converged = try_direct_solver("Amesos_Klu", system_matrix_,
                                           system_rhs_, coupled_solution,
                                           rank, verbose);

        if (converged)
        {
            info.iterations = 1;
            info.residual = 0.0;
            info.converged = true;
            info.used_direct = true;
        }
    }

    // ========================================================================
    // Iterative solver (GMRES + AMG — fallback for large problems)
    // ========================================================================
    if (!converged && (params_.solvers.ch.use_iterative || params_.solvers.ch.fallback_to_direct == false))
    {
        dealii::SolverControl solver_control(
            params_.solvers.ch.max_iterations,
            params_.solvers.ch.rel_tolerance * rhs_norm,
            /*log_history=*/false,
            /*log_result=*/false);

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = params_.solvers.ch.gmres_restart;
        gmres_data.right_preconditioning = true;

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(
            solver_control, gmres_data);

        dealii::TrilinosWrappers::PreconditionAMG preconditioner;
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 1e-4;
        amg_data.elliptic = false;  // CH is indefinite
        amg_data.higher_order_elements = (fe_.degree > 1);

        try
        {
            preconditioner.initialize(system_matrix_, amg_data);
            solver.solve(system_matrix_, coupled_solution,
                         system_rhs_, preconditioner);

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

            if (rank == 0)
                std::cerr << "[CH] GMRES failed: " << info.iterations
                          << " its, residual = " << info.residual << "\n";
        }
    }

    // ========================================================================
    // Fallback to direct if iterative failed
    // ========================================================================
    if (!converged && params_.solvers.ch.fallback_to_direct)
    {
        if (rank == 0)
            std::cerr << "[CH] Falling back to direct solver\n";

        if (!converged)
            converged = try_direct_solver("Amesos_Mumps", system_matrix_,
                                           system_rhs_, coupled_solution,
                                           rank, verbose,
                                           /*suppress_output=*/true);
        if (!converged)
            converged = try_direct_solver("Amesos_Klu", system_matrix_,
                                           system_rhs_, coupled_solution,
                                           rank, verbose);

        if (converged)
        {
            info.iterations = 1;
            info.residual = 0.0;
            info.converged = true;
            info.used_direct = true;
        }
    }

    if (!converged && rank == 0)
        std::cerr << "[CH] WARNING: All solvers failed!\n";

    // ========================================================================
    // Distribute constraints
    // ========================================================================
    ch_constraints_.distribute(coupled_solution);

    // ========================================================================
    // Extract θ and ψ from coupled solution
    //
    // MPI-correct: create ghosted coupled vector so operator[] is valid
    // for indices that are locally relevant but not owned.
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector coupled_relevant(
        ch_locally_owned_, ch_locally_relevant_, mpi_comm_);
    coupled_relevant = coupled_solution;

    // Extract θ from coupled solution
    for (auto idx = theta_locally_owned_.begin();
         idx != theta_locally_owned_.end(); ++idx)
    {
        theta_solution_[*idx] = coupled_relevant[theta_to_ch_map_[*idx]];
    }

    // Extract ψ from coupled solution
    for (auto idx = psi_locally_owned_.begin();
         idx != psi_locally_owned_.end(); ++idx)
    {
        psi_solution_[*idx] = coupled_relevant[psi_to_ch_map_[*idx]];
    }

    theta_solution_.compress(dealii::VectorOperation::insert);
    psi_solution_.compress(dealii::VectorOperation::insert);

    auto t_end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(t_end - t_start).count();

    if (verbose && rank == 0)
    {
        pcout_ << "[CH] " << (info.used_direct ? "Direct" : "Iterative")
               << " solve: its=" << info.iterations
               << ", residual=" << std::scientific << info.residual
               << ", time=" << std::fixed << std::setprecision(2)
               << info.solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;