// ============================================================================
// solvers/poisson_solver.cc - Parallel Magnetostatic Poisson Solver
//
// Uses CG + Trilinos AMG for distributed Poisson systems.
//
// OPTIMIZATION: AMG preconditioner is built ONCE and reused.
// ============================================================================

#include "solvers/poisson_solver.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/base/utilities.h>

#include <chrono>
#include <iostream>

// ============================================================================
// PoissonSolver - Class with cached preconditioner
// ============================================================================

PoissonSolver::PoissonSolver(
    const LinearSolverParams& params,
    const dealii::IndexSet& locally_owned,
    MPI_Comm mpi_comm)
    : params_(params)
    , locally_owned_(locally_owned)
    , mpi_comm_(mpi_comm)
    , matrix_ptr_(nullptr)
    , initialized_(false)
{
}

void PoissonSolver::initialize(const dealii::TrilinosWrappers::SparseMatrix& matrix)
{
    matrix_ptr_ = &matrix;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    // Build AMG preconditioner ONCE
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 1e-4;
    amg_data.elliptic = true;  // Poisson is elliptic
    amg_data.higher_order_elements = true;

    try
    {
        amg_preconditioner_.initialize(matrix, amg_data);
        initialized_ = true;

        if (rank == 0)
            std::cout << "[Poisson] AMG preconditioner initialized (cached)\n";
    }
    catch (std::exception& e)
    {
        if (rank == 0)
            std::cerr << "[Poisson] AMG init failed: " << e.what() << "\n";
        initialized_ = false;
    }
}

SolverInfo PoissonSolver::solve(
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    SolverInfo info;
    info.solver_name = "Poisson";
    info.matrix_size = matrix_ptr_ ? matrix_ptr_->m() : 0;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    auto start = std::chrono::high_resolution_clock::now();

    // Check for zero RHS
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        info.iterations = 0;
        info.residual = 0.0;
        info.converged = true;

        auto end = std::chrono::high_resolution_clock::now();
        info.solve_time = std::chrono::duration<double>(end - start).count();
        return info;
    }

    // Setup solver
    const double tol = std::max(params_.abs_tolerance, params_.rel_tolerance * rhs_norm);
    dealii::SolverControl solver_control(params_.max_iterations, tol);
    dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control);

    try
    {
        if (initialized_)
        {
            // Use cached AMG preconditioner
            solver.solve(*matrix_ptr_, solution, rhs, amg_preconditioner_);
        }
        else
        {
            // Fallback to Jacobi
            dealii::TrilinosWrappers::PreconditionJacobi jacobi;
            jacobi.initialize(*matrix_ptr_);
            solver.solve(*matrix_ptr_, solution, rhs, jacobi);
        }

        info.iterations = solver_control.last_step();
        info.residual = solver_control.last_value();
        info.converged = true;
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        info.iterations = solver_control.last_step();
        info.residual = solver_control.last_value();
        info.converged = false;

        if (verbose && rank == 0)
            std::cerr << "[Poisson] CG failed after " << info.iterations
                      << " iterations, residual = " << info.residual << "\n";
    }

    constraints.distribute(solution);

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    if (verbose && rank == 0)
    {
        std::cout << "[Poisson] iterations=" << info.iterations
                  << ", residual=" << std::scientific << info.residual
                  << ", time=" << std::fixed << info.solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Legacy free function (rebuilds preconditioner each call)
// ============================================================================
SolverInfo solve_poisson_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::IndexSet& locally_owned,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm,
    bool verbose)
{
    // Create temporary solver (preconditioner not cached)
    PoissonSolver solver(params, locally_owned, mpi_comm);
    solver.initialize(matrix);
    return solver.solve(rhs, solution, constraints, verbose);
}