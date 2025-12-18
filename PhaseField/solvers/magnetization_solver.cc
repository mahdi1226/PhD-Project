// ============================================================================
// solvers/magnetization_solver.cc - DG Magnetization System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Simple solver wrapper: direct (UMFPACK) or iterative (GMRES).
// The paper does not prescribe a specific linear solver.
// ============================================================================

#include "solvers/magnetization_solver.h"
#include "utilities/parameters.h"

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <iostream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagnetizationSolver<dim>::MagnetizationSolver(const LinearSolverParams& params)
    : params_(params)
    , matrix_ptr_(nullptr)
    , last_n_iterations_(0)
    , use_direct_(!params.use_iterative)  // use_direct = NOT use_iterative
    , initialized_(false)
{
}

// ============================================================================
// Initialize (factorize for direct, store pointer for iterative)
// ============================================================================
template <int dim>
void MagnetizationSolver<dim>::initialize(
    const dealii::SparseMatrix<double>& system_matrix)
{
    matrix_ptr_ = &system_matrix;

    if (use_direct_)
    {
        direct_solver_.initialize(system_matrix);
    }

    initialized_ = true;
}

// ============================================================================
// Solve
// ============================================================================
template <int dim>
void MagnetizationSolver<dim>::solve(
    dealii::Vector<double>& solution,
    const dealii::Vector<double>& rhs) const
{
    Assert(initialized_,
           dealii::ExcMessage("Solver not initialized. Call initialize() first."));
    Assert(matrix_ptr_ != nullptr,
           dealii::ExcMessage("Matrix pointer is null."));

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    if (use_direct_)
    {
        direct_solver_.vmult(solution, rhs);
        last_n_iterations_ = 1;
    }
    else
    {
        const double rhs_norm = rhs.l2_norm();
        const double tol = std::max(params_.abs_tolerance,
                                    params_.rel_tolerance * rhs_norm);

        dealii::SolverControl solver_control(params_.max_iterations, tol);
        dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);

        dealii::PreconditionJacobi<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(*matrix_ptr_);

        try
        {
            solver.solve(*matrix_ptr_, solution, rhs, preconditioner);
            last_n_iterations_ = solver_control.last_step();
        }
        catch (dealii::SolverControl::NoConvergence& e)
        {
            std::cerr << "[MagnetizationSolver] GMRES did not converge: "
                      << e.what() << std::endl;
            std::cerr << "  Last residual: " << e.last_residual << std::endl;
            last_n_iterations_ = solver_control.last_step();

            if (params_.fallback_to_direct)
            {
                std::cerr << "[MagnetizationSolver] Falling back to direct solver.\n";
                dealii::SparseDirectUMFPACK fallback;
                fallback.initialize(*matrix_ptr_);
                fallback.vmult(solution, rhs);
                last_n_iterations_ = 1;
            }
        }
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class MagnetizationSolver<2>;
template class MagnetizationSolver<3>;