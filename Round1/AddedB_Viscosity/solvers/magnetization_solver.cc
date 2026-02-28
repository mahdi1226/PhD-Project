// ============================================================================
// solvers/magnetization_solver.cc - DG Magnetization Solver (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses Trilinos Amesos_Mumps for parallel direct solving
//   - Falls back to iterative GMRES+ILU if needed
//
// OPTIMIZATION: Direct solver with MUMPS is much faster than KLU
// for DG systems at typical problem sizes (< 500K DoFs).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/magnetization_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>

#include <iostream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagnetizationSolver<dim>::MagnetizationSolver(
    const LinearSolverParams& params,
    const dealii::IndexSet& locally_owned,
    MPI_Comm mpi_communicator)
    : params_(params)
    , locally_owned_(locally_owned)
    , mpi_communicator_(mpi_communicator)
    , matrix_ptr_(nullptr)
    , last_n_iterations_(0)
    , use_direct_(!params.use_iterative)
    , initialized_(false)
    , preconditioner_initialized_(false)
{
}

// ============================================================================
// Initialize - also sets up preconditioner for reuse
// ============================================================================
template <int dim>
void MagnetizationSolver<dim>::initialize(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix)
{
    matrix_ptr_ = &system_matrix;
    initialized_ = true;

    // Pre-initialize preconditioner if using iterative solver
    // This is reused for both Mx and My solves
    if (!use_direct_)
    {
        try
        {
            // ILU(0) works well for DG matrices due to their block structure
            dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
            ilu_data.ilu_fill = 0;      // ILU(0) - no extra fill
            ilu_data.ilu_atol = 1e-12;  // Absolute drop tolerance
            ilu_data.ilu_rtol = 1.0;    // Relative drop tolerance
            ilu_data.overlap = 0;       // No overlap for block-diagonal

            ilu_preconditioner_.initialize(system_matrix, ilu_data);
            preconditioner_initialized_ = true;
        }
        catch (std::exception& e)
        {
            const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);
            if (rank == 0)
                std::cerr << "[Magnetization] ILU init failed, using Jacobi: " << e.what() << "\n";
            preconditioner_initialized_ = false;
        }
    }
}

// ============================================================================
// Solve using MUMPS (parallel direct) - much faster than default KLU
// ============================================================================
template <int dim>
void MagnetizationSolver<dim>::solve(
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    Assert(initialized_,
           dealii::ExcMessage("Solver not initialized. Call initialize() first."));
    Assert(matrix_ptr_ != nullptr,
           dealii::ExcMessage("Matrix pointer is null."));

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    // Check for zero RHS
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_n_iterations_ = 0;
        return;
    }

    // Ensure solution is properly sized
    if (solution.size() != rhs.size())
        solution.reinit(locally_owned_, mpi_communicator_);

    bool converged = false;

    // ========================================================================
    // Direct solver with MUMPS (parallel) - much faster than default KLU
    // ========================================================================
    if (use_direct_)
    {
        const double tol = std::max(params_.abs_tolerance, params_.rel_tolerance * rhs_norm);
        dealii::SolverControl solver_control(1, tol);

        // Try MUMPS first (parallel direct solver)
        try
        {
            dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
            data.solver_type = "Amesos_Mumps";  // Use MUMPS instead of default KLU

            dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
            direct_solver.solve(*matrix_ptr_, solution, rhs);

            last_n_iterations_ = 1;
            converged = true;
        }
        catch (std::exception& e)
        {
            // MUMPS not available or failed, try SuperLU_DIST
            try
            {
                dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
                data.solver_type = "Amesos_Superludist";

                dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);
                direct_solver.solve(*matrix_ptr_, solution, rhs);

                last_n_iterations_ = 1;
                converged = true;
            }
            catch (std::exception& e2)
            {
                // Fall back to default (KLU)
                try
                {
                    dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control);
                    direct_solver.solve(*matrix_ptr_, solution, rhs);

                    last_n_iterations_ = 1;
                    converged = true;
                }
                catch (std::exception& e3)
                {
                    if (this_rank == 0)
                        std::cerr << "[Magnetization] All direct solvers failed, trying iterative\n";
                    converged = false;
                }
            }
        }
    }

    // ========================================================================
    // Iterative solver (fallback or if configured)
    // ========================================================================
    if (!converged)
    {
        const double tol = std::max(params_.abs_tolerance, params_.rel_tolerance * rhs_norm);
        dealii::SolverControl solver_control(params_.max_iterations, tol);

        try
        {
            dealii::TrilinosWrappers::SolverGMRES solver(solver_control);

            if (preconditioner_initialized_)
            {
                solver.solve(*matrix_ptr_, solution, rhs, ilu_preconditioner_);
            }
            else
            {
                dealii::TrilinosWrappers::PreconditionJacobi point_jacobi;
                point_jacobi.initialize(*matrix_ptr_);
                solver.solve(*matrix_ptr_, solution, rhs, point_jacobi);
            }

            last_n_iterations_ = solver_control.last_step();
            converged = true;
        }
        catch (dealii::SolverControl::NoConvergence& e)
        {
            last_n_iterations_ = e.last_step;

            if (this_rank == 0)
            {
                std::cerr << "[Magnetization] GMRES did not converge after "
                          << last_n_iterations_ << " iterations. "
                          << "Residual = " << e.last_residual << "\n";
            }
        }
        catch (std::exception& e)
        {
            if (this_rank == 0)
                std::cerr << "[Magnetization] Iterative solver failed: " << e.what() << "\n";
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagnetizationSolver<2>;