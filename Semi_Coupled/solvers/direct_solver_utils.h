// ============================================================================
// solvers/direct_solver_utils.h - Shared Direct Solver Fallback Utilities
//
// Provides a common try_direct_solver() helper used by CH and NS solvers
// to attempt various Trilinos/Amesos direct solver backends.
// ============================================================================
#ifndef DIRECT_SOLVER_UTILS_H
#define DIRECT_SOLVER_UTILS_H

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <iostream>
#include <string>

namespace DirectSolverUtils
{

/**
 * @brief Try a specific Amesos direct solver type.
 *
 * Attempts to solve the system using the specified Trilinos/Amesos backend.
 * Returns true on success, false if the solver is unavailable or fails.
 *
 * @param solver_type  Amesos solver name (e.g., "Amesos_Mumps", "Amesos_Klu")
 * @param matrix       System matrix
 * @param rhs          Right-hand side vector
 * @param solution     [OUT] Solution vector
 * @param prefix       Log prefix (e.g., "CH", "NS")
 * @param rank         MPI rank (only rank 0 prints)
 * @param verbose      Whether to print progress messages
 * @return true if solve succeeded
 */
inline bool try_direct_solver(
    const std::string& solver_type,
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const std::string& prefix,
    int rank,
    bool verbose)
{
    if (verbose && rank == 0)
        std::cout << "[" << prefix << " Direct] Trying " << solver_type << "...\n";

    try
    {
        dealii::SolverControl solver_control(1, 0);  // Direct = 1 iteration
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = solver_type;

        dealii::TrilinosWrappers::SolverDirect solver(solver_control, data);

        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct]   Initializing (symbolic + numeric factorization)...\n";

        solver.initialize(matrix);

        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct]   Solving...\n";

        solver.solve(solution, rhs);

        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct] SUCCESS with " << solver_type << "\n";

        return true;
    }
    catch (const dealii::ExceptionBase& e)
    {
        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (const std::exception& e)
    {
        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (...)
    {
        if (verbose && rank == 0)
            std::cout << "[" << prefix << " Direct]   " << solver_type << " failed: unknown exception\n";
        return false;
    }
}

/**
 * @brief Try the standard direct solver fallback chain.
 *
 * Tries: Mumps → Umfpack → Superludist → Superlu → Klu
 *
 * @return true if any solver succeeded
 */
inline bool try_direct_solver_chain(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const std::string& prefix,
    int rank,
    bool verbose)
{
    if (try_direct_solver("Amesos_Mumps", matrix, rhs, solution, prefix, rank, verbose))
        return true;
    if (try_direct_solver("Amesos_Umfpack", matrix, rhs, solution, prefix, rank, verbose))
        return true;
    if (try_direct_solver("Amesos_Superludist", matrix, rhs, solution, prefix, rank, verbose))
        return true;
    if (try_direct_solver("Amesos_Superlu", matrix, rhs, solution, prefix, rank, verbose))
        return true;
    if (try_direct_solver("Amesos_Klu", matrix, rhs, solution, prefix, rank, verbose))
        return true;
    return false;
}

} // namespace DirectSolverUtils

#endif // DIRECT_SOLVER_UTILS_H
