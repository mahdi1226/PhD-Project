// ============================================================================
// solvers/magnetic_solver.cc - Monolithic Magnetics Solver (PARALLEL)
//
// MUMPS direct solver with fallback cascade:
//   1. Amesos_Mumps (parallel direct)
//   2. Amesos_Superludist (parallel direct)
//   3. Default (KLU, serial)
//   4. GMRES + ILU (iterative fallback)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/magnetic_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

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
{
}

// ============================================================================
// Solve
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    const unsigned int this_rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    // Zero RHS guard
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_n_iterations_ = 0;
        return;
    }

    const double tol = 1e-12;
    dealii::SolverControl solver_control(1, tol);

    // Try MUMPS first
    try
    {
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = "Amesos_Mumps";

        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
        direct_solver.solve(system_matrix, solution, rhs);

        last_n_iterations_ = 1;
        return;
    }
    catch (std::exception&)
    {
    }

    // Try SuperLU_DIST
    try
    {
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = "Amesos_Superludist";

        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
        direct_solver.solve(system_matrix, solution, rhs);

        last_n_iterations_ = 1;
        return;
    }
    catch (std::exception&)
    {
    }

    // Try default (KLU)
    try
    {
        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control);
        direct_solver.solve(system_matrix, solution, rhs);

        last_n_iterations_ = 1;
        return;
    }
    catch (std::exception&)
    {
    }

    // Iterative fallback: GMRES + ILU
    if (this_rank == 0)
        std::cerr << "[Magnetic] All direct solvers failed, trying GMRES+ILU\n";

    dealii::SolverControl iterative_control(2000, 1e-8 * rhs_norm);

    try
    {
        dealii::TrilinosWrappers::SolverGMRES solver(iterative_control);
        dealii::TrilinosWrappers::PreconditionILU ilu;
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = 1;
        ilu.initialize(system_matrix, ilu_data);

        solver.solve(system_matrix, solution, rhs, ilu);
        last_n_iterations_ = iterative_control.last_step();
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        last_n_iterations_ = e.last_step;
        if (this_rank == 0)
        {
            std::cerr << "[Magnetic] GMRES did not converge after "
                      << last_n_iterations_ << " iterations. "
                      << "Residual = " << e.last_residual << "\n";
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticSolver<2>;
