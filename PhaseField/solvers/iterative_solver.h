// ============================================================================
// solvers/iterative_solver.h - Iterative Linear Solver Wrapper
//
// Unified interface for CG and GMRES solvers with various preconditioners.
// Replaces direct UMFPACK calls for large systems.
//
// Usage:
//   IterativeSolver<SparseMatrix<double>, Vector<double>> solver(params);
//   solver.initialize(matrix);  // Setup preconditioner
//   unsigned int iters = solver.solve(solution, rhs);
//
// Reference: Phase 1 of performance improvement plan
// ============================================================================
#ifndef ITERATIVE_SOLVER_H
#define ITERATIVE_SOLVER_H

#include "solvers/solver_parameters.h"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <memory>
#include <chrono>

/**
 * @brief Iterative linear solver with automatic preconditioner selection
 *
 * Wraps deal.II's CG and GMRES solvers with ILU, SSOR, or Jacobi preconditioners.
 * Falls back to UMFPACK if iterative solver fails.
 *
 * @tparam MatrixType  Matrix type (e.g., SparseMatrix<double>)
 * @tparam VectorType  Vector type (e.g., Vector<double>)
 */
template <typename MatrixType = dealii::SparseMatrix<double>,
          typename VectorType = dealii::Vector<double>>
class IterativeSolver
{
public:
    /**
     * @brief Constructor
     * @param params  Solver configuration
     * @param name    Identifier for logging (e.g., "CH", "NS", "Poisson")
     */
    explicit IterativeSolver(const LinearSolverParams& params,
                             const std::string& name = "Linear")
        : params_(params)
        , name_(name)
        , matrix_ptr_(nullptr)
        , last_iterations_(0)
        , last_residual_(0.0)
        , initialized_(false)
    {}

    /**
     * @brief Initialize preconditioner with system matrix
     *
     * Must be called before solve() and whenever matrix changes.
     * For ILU: performs incomplete factorization
     * For SSOR/Jacobi: stores matrix reference
     */
    void initialize(const MatrixType& matrix);

    /**
     * @brief Solve A*x = b
     *
     * @param solution  [IN/OUT] Initial guess, overwritten with solution
     * @param rhs       Right-hand side vector
     * @return          Number of iterations (0 if direct solver used)
     */
    unsigned int solve(VectorType& solution, const VectorType& rhs);

    /**
     * @brief Get iteration count from last solve
     */
    unsigned int last_iterations() const { return last_iterations_; }

    /**
     * @brief Get final residual from last solve
     */
    double last_residual() const { return last_residual_; }

private:
    // Solve using CG (for SPD systems)
    unsigned int solve_cg(VectorType& solution, const VectorType& rhs);

    // Solve using GMRES (for nonsymmetric systems)
    unsigned int solve_gmres(VectorType& solution, const VectorType& rhs);

    // Fallback direct solve
    void solve_direct(VectorType& solution, const VectorType& rhs);

    // Configuration
    LinearSolverParams params_;
    std::string name_;

    // Matrix reference (non-owning)
    const MatrixType* matrix_ptr_;

    // Preconditioners (only one is used based on params_)
    dealii::PreconditionJacobi<MatrixType> prec_jacobi_;
    dealii::PreconditionSSOR<MatrixType> prec_ssor_;
    dealii::SparseILU<double> prec_ilu_;

    // Direct solver fallback
    mutable dealii::SparseDirectUMFPACK direct_solver_;

    // Statistics
    mutable unsigned int last_iterations_;
    mutable double last_residual_;
    bool initialized_;
};

// ============================================================================
// Implementation
// ============================================================================

template <typename MatrixType, typename VectorType>
void IterativeSolver<MatrixType, VectorType>::initialize(const MatrixType& matrix)
{
    matrix_ptr_ = &matrix;

    switch (params_.preconditioner)
    {
        case LinearSolverParams::Preconditioner::Jacobi:
            prec_jacobi_.initialize(matrix);
            break;

        case LinearSolverParams::Preconditioner::SSOR:
            prec_ssor_.initialize(matrix, params_.ssor_omega);
            break;

        case LinearSolverParams::Preconditioner::ILU:
            prec_ilu_.initialize(matrix);
            break;

        case LinearSolverParams::Preconditioner::None:
            // No preconditioner setup needed
            break;
    }

    // Pre-factorize for direct solver fallback
    if (params_.fallback_to_direct)
    {
        try {
            direct_solver_.initialize(matrix);
        } catch (...) {
            // Direct solver init failed; will error if fallback needed
        }
    }

    initialized_ = true;
}

template <typename MatrixType, typename VectorType>
unsigned int IterativeSolver<MatrixType, VectorType>::solve(
    VectorType& solution,
    const VectorType& rhs)
{
    Assert(initialized_,
           dealii::ExcMessage("Solver not initialized. Call initialize() first."));
    Assert(matrix_ptr_ != nullptr,
           dealii::ExcMessage("Matrix pointer is null."));

    // Handle zero RHS
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_iterations_ = 0;
        last_residual_ = 0.0;
        if (params_.verbose)
            std::cout << "[" << name_ << "] Zero RHS, solution set to zero\n";
        return 0;
    }

    // Ensure solution is properly sized
    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();

    unsigned int iterations = 0;

    switch (params_.solver_type)
    {
        case LinearSolverParams::Type::CG:
            iterations = solve_cg(solution, rhs);
            break;

        case LinearSolverParams::Type::GMRES:
            iterations = solve_gmres(solution, rhs);
            break;

        case LinearSolverParams::Type::Direct:
            solve_direct(solution, rhs);
            iterations = 1;
            break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    if (params_.log_convergence && iterations > 0)
    {
        std::cout << "[" << name_ << "] "
                  << (params_.solver_type == LinearSolverParams::Type::CG ? "CG" : "GMRES")
                  << " converged in " << iterations << " iterations, "
                  << "residual = " << last_residual_
                  << ", time = " << solve_time << "s\n";
    }

    return iterations;
}

template <typename MatrixType, typename VectorType>
unsigned int IterativeSolver<MatrixType, VectorType>::solve_cg(
    VectorType& solution,
    const VectorType& rhs)
{
    const double rhs_norm = rhs.l2_norm();
    const double tol = std::max(params_.abs_tolerance,
                                params_.rel_tolerance * rhs_norm);

    dealii::SolverControl solver_control(params_.max_iterations, tol);
    dealii::SolverCG<VectorType> solver(solver_control);

    try
    {
        switch (params_.preconditioner)
        {
            case LinearSolverParams::Preconditioner::Jacobi:
                solver.solve(*matrix_ptr_, solution, rhs, prec_jacobi_);
                break;

            case LinearSolverParams::Preconditioner::SSOR:
                solver.solve(*matrix_ptr_, solution, rhs, prec_ssor_);
                break;

            case LinearSolverParams::Preconditioner::ILU:
                solver.solve(*matrix_ptr_, solution, rhs, prec_ilu_);
                break;

            case LinearSolverParams::Preconditioner::None:
                solver.solve(*matrix_ptr_, solution, rhs,
                             dealii::PreconditionIdentity());
                break;
        }

        last_iterations_ = solver_control.last_step();
        last_residual_ = solver_control.last_value();
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        std::cerr << "[" << name_ << "] WARNING: CG did not converge after "
                  << e.last_step << " iterations. "
                  << "Residual = " << e.last_residual << "\n";

        last_iterations_ = e.last_step;
        last_residual_ = e.last_residual;

        if (params_.fallback_to_direct)
        {
            std::cerr << "[" << name_ << "] Falling back to direct solver.\n";
            solve_direct(solution, rhs);
        }
    }

    return last_iterations_;
}

template <typename MatrixType, typename VectorType>
unsigned int IterativeSolver<MatrixType, VectorType>::solve_gmres(
    VectorType& solution,
    const VectorType& rhs)
{
    const double rhs_norm = rhs.l2_norm();
    const double tol = std::max(params_.abs_tolerance,
                                params_.rel_tolerance * rhs_norm);

    dealii::SolverControl solver_control(params_.max_iterations, tol);

    // GMRES with restart
    typename dealii::SolverGMRES<VectorType>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = params_.gmres_restart + 2;

    dealii::SolverGMRES<VectorType> solver(solver_control, gmres_data);

    try
    {
        switch (params_.preconditioner)
        {
            case LinearSolverParams::Preconditioner::Jacobi:
                solver.solve(*matrix_ptr_, solution, rhs, prec_jacobi_);
                break;

            case LinearSolverParams::Preconditioner::SSOR:
                solver.solve(*matrix_ptr_, solution, rhs, prec_ssor_);
                break;

            case LinearSolverParams::Preconditioner::ILU:
                solver.solve(*matrix_ptr_, solution, rhs, prec_ilu_);
                break;

            case LinearSolverParams::Preconditioner::None:
                solver.solve(*matrix_ptr_, solution, rhs,
                             dealii::PreconditionIdentity());
                break;
        }

        last_iterations_ = solver_control.last_step();
        last_residual_ = solver_control.last_value();
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        std::cerr << "[" << name_ << "] WARNING: GMRES did not converge after "
                  << e.last_step << " iterations. "
                  << "Residual = " << e.last_residual << "\n";

        last_iterations_ = e.last_step;
        last_residual_ = e.last_residual;

        if (params_.fallback_to_direct)
        {
            std::cerr << "[" << name_ << "] Falling back to direct solver.\n";
            solve_direct(solution, rhs);
        }
    }

    return last_iterations_;
}

template <typename MatrixType, typename VectorType>
void IterativeSolver<MatrixType, VectorType>::solve_direct(
    VectorType& solution,
    const VectorType& rhs)
{
    direct_solver_.vmult(solution, rhs);
    last_iterations_ = 1;
    last_residual_ = 0.0;  // Direct solver is "exact" to machine precision

    if (params_.verbose)
        std::cout << "[" << name_ << "] Direct solve (UMFPACK)\n";
}

#endif // ITERATIVE_SOLVER_H