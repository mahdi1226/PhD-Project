// ============================================================================
// solvers/magnetic_solver.cc - Monolithic Magnetics Solver (PARALLEL)
// ============================================================================

#include "solvers/magnetic_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
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
// Direct cascade (MUMPS → SuperLU_DIST → KLU)
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
// Iterative path: GMRES + cached MagneticBlockPreconditioner
// ============================================================================
template <int dim>
bool MagneticSolver<dim>::solve_iterative(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    dealii::types::global_dof_index n_M_dofs,
    bool rebuild_preconditioner)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    const double rhs_norm = rhs.l2_norm();
    const double tol = params.rel_tolerance * rhs_norm;

    dealii::SolverControl solver_control(
        params.max_iterations, tol, /*log_history=*/false, /*log_result=*/false);

    if (rebuild_preconditioner || !cached_block_prec_)
    {
        try
        {
            cached_block_prec_ = std::make_unique<MagneticBlockPreconditioner>(
                system_matrix, locally_owned_, n_M_dofs, mpi_communicator_);
        }
        catch (const std::exception& e)
        {
            if (rank == 0)
                std::cerr << "[Magnetic] Block preconditioner build failed: "
                          << e.what() << "\n";
            cached_block_prec_.reset();
            return false;
        }
    }

    // Use deal.II's templated SolverGMRES so we can pass our custom
    // preconditioner (only requires vmult()).
    dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData
        gmres_data;
    gmres_data.right_preconditioning = true;
    gmres_data.max_n_tmp_vectors = params.gmres_restart;
    dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>
        solver(solver_control, gmres_data);

    try
    {
        solver.solve(system_matrix, solution, rhs, *cached_block_prec_);
        last_n_iterations_ = solver_control.last_step();
        last_residual_     = solver_control.last_value();
        last_used_direct_  = false;
        if (params.verbose && rank == 0)
        {
            std::cout << "[Magnetic GMRES+Block] " << last_n_iterations_
                      << " iters, res = " << last_residual_ << "\n";
        }
        return true;
    }
    catch (const dealii::SolverControl::NoConvergence& e)
    {
        last_n_iterations_ = e.last_step;
        last_residual_     = e.last_residual;
        if (rank == 0)
        {
            std::cerr << "[Magnetic GMRES+Block] did NOT converge: "
                      << "iters=" << e.last_step
                      << ", res=" << e.last_residual << "\n";
        }
        return false;
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "[Magnetic GMRES+Block] exception: " << e.what() << "\n";
        return false;
    }
}

// ============================================================================
// Public solve (parameterized)
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    dealii::types::global_dof_index n_M_dofs,
    bool rebuild_preconditioner)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    // Reset bookkeeping for this solve
    last_converged_ = false;

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_n_iterations_ = 0;
        last_residual_     = 0.0;
        last_used_direct_  = !params.use_iterative;
        last_converged_    = true;  // trivially: 0 = 0
        return;
    }

    if (params.use_iterative)
    {
        if (solve_iterative(system_matrix, solution, rhs, params,
                            n_M_dofs, rebuild_preconditioner))
        {
            last_converged_ = true;
            return;
        }

        if (params.fallback_to_direct)
        {
            if (rank == 0)
                std::cerr << "[Magnetic] Iterative failed; falling back to direct\n";
            cached_block_prec_.reset();
            if (solve_direct(system_matrix, solution, rhs))
            {
                last_converged_ = true;
                return;
            }
        }
        if (rank == 0)
            std::cerr << "[Magnetic] All solvers failed!\n";
        // last_converged_ stays false
        return;
    }

    if (solve_direct(system_matrix, solution, rhs))
    {
        last_converged_ = true;
        return;
    }

    if (rank == 0)
        std::cerr << "[Magnetic] All direct solvers failed; trying iterative\n";
    LinearSolverParams fb = params;
    fb.use_iterative = true;
    fb.max_iterations = 2000;
    if (solve_iterative(system_matrix, solution, rhs, fb,
                        n_M_dofs, /*rebuild=*/true))
        last_converged_ = true;
    // else: last_converged_ stays false; caller should check.
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
    solve(system_matrix, solution, rhs, default_params,
          /*n_M_dofs=*/0, /*rebuild=*/true);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticSolver<2>;
