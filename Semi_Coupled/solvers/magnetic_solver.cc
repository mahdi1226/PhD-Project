// ============================================================================
// solvers/magnetic_solver.cc - Monolithic Magnetics Solver (PARALLEL)
//
// Direct (default) and iterative (GMRES + cached ILU) paths.
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/magnetic_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>

#include <iostream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagneticSolver<dim>::MagneticSolver(
    const dealii::IndexSet& locally_owned,
    MPI_Comm mpi_communicator)
    : locally_owned_(locally_owned)
    , mpi_communicator_(mpi_communicator)
    , last_n_iterations_(0)
    , last_used_direct_(true)
{
}

// ============================================================================
// Direct solver cascade (MUMPS → SuperLU_DIST → KLU)
// ============================================================================
template <int dim>
bool MagneticSolver<dim>::solve_direct(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    const double tol = 1e-12;
    dealii::SolverControl solver_control(1, tol);

    try
    {
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = "Amesos_Mumps";
        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
        direct_solver.solve(system_matrix, solution, rhs);
        last_n_iterations_ = 1;
        last_used_direct_ = true;
        return true;
    }
    catch (std::exception&) {}

    try
    {
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = "Amesos_Superludist";
        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
        direct_solver.solve(system_matrix, solution, rhs);
        last_n_iterations_ = 1;
        last_used_direct_ = true;
        return true;
    }
    catch (std::exception&) {}

    try
    {
        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control);
        direct_solver.solve(system_matrix, solution, rhs);
        last_n_iterations_ = 1;
        last_used_direct_ = true;
        return true;
    }
    catch (std::exception&) {}

    return false;
}

// ============================================================================
// Iterative solver (GMRES + cached ILU)
// ============================================================================
template <int dim>
bool MagneticSolver<dim>::solve_iterative(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    bool rebuild_preconditioner)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    const double rhs_norm = rhs.l2_norm();
    const double tol = params.rel_tolerance * rhs_norm;

    dealii::SolverControl solver_control(
        params.max_iterations, tol, /*log_history=*/false, /*log_result=*/false);

    if (rebuild_preconditioner || !cached_ilu_)
    {
        cached_ilu_ = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = static_cast<unsigned int>(params.ilu_fill);
        ilu_data.ilu_atol = 1e-10;
        ilu_data.ilu_rtol = 1.0;
        try
        {
            cached_ilu_->initialize(system_matrix, ilu_data);
        }
        catch (const std::exception& e)
        {
            if (rank == 0)
                std::cerr << "[Magnetic] ILU initialize failed: " << e.what() << "\n";
            cached_ilu_.reset();
            return false;
        }
    }

    dealii::TrilinosWrappers::SolverGMRES solver(solver_control);

    try
    {
        solver.solve(system_matrix, solution, rhs, *cached_ilu_);
        last_n_iterations_ = solver_control.last_step();
        last_used_direct_ = false;
        if (params.verbose && rank == 0)
        {
            std::cout << "[Magnetic GMRES+ILU] " << last_n_iterations_
                      << " iters, res = " << solver_control.last_value() << "\n";
        }
        return true;
    }
    catch (const dealii::SolverControl::NoConvergence& e)
    {
        last_n_iterations_ = e.last_step;
        if (rank == 0)
        {
            std::cerr << "[Magnetic GMRES+ILU] did NOT converge: "
                      << "iters=" << e.last_step
                      << ", res=" << e.last_residual << "\n";
        }
        return false;
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "[Magnetic GMRES+ILU] exception: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// Solve (parameterized API)
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    bool rebuild_preconditioner)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_n_iterations_ = 0;
        last_used_direct_ = !params.use_iterative;
        return;
    }

    if (params.use_iterative)
    {
        if (solve_iterative(system_matrix, solution, rhs, params, rebuild_preconditioner))
            return;

        if (params.fallback_to_direct)
        {
            if (rank == 0)
                std::cerr << "[Magnetic] Iterative failed; falling back to direct\n";
            cached_ilu_.reset();
            if (solve_direct(system_matrix, solution, rhs))
                return;
        }
        if (rank == 0)
            std::cerr << "[Magnetic] All solvers failed!\n";
        return;
    }

    if (!solve_direct(system_matrix, solution, rhs))
    {
        if (rank == 0)
            std::cerr << "[Magnetic] All direct solvers failed; trying iterative fallback\n";
        LinearSolverParams fb = params;
        fb.use_iterative = true;
        fb.max_iterations = 2000;
        solve_iterative(system_matrix, solution, rhs, fb, /*rebuild=*/true);
    }
}

// ============================================================================
// Backward-compatible overload (always direct)
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    LinearSolverParams default_params;
    default_params.use_iterative = false;
    default_params.fallback_to_direct = false;
    solve(system_matrix, solution, rhs, default_params, /*rebuild=*/true);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticSolver<2>;
