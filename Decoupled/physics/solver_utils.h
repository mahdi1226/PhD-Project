// ============================================================================
// physics/solver_utils.h — Shared Direct Solver Fallback Utility
//
// Provides a reusable try-MUMPS → SuperLU_DIST → KLU fallback chain
// for direct solvers. Used by magnetization_solve.cc and poisson_solve.cc.
//
// This eliminates duplicate nested try-catch blocks across subsystems.
// ============================================================================
#ifndef SOLVER_UTILS_H
#define SOLVER_UTILS_H

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/base/conditional_ostream.h>

#include <string>
#include <vector>

namespace SolverUtils
{

/**
 * @brief Try a chain of direct solvers (MUMPS → SuperLU_DIST → KLU).
 *
 * Returns true if any solver succeeded, false if all failed.
 * On success, `solution` contains the result.
 *
 * @param matrix      System matrix
 * @param solution    Output solution vector
 * @param rhs         Right-hand side vector
 * @param tol         Solver tolerance
 * @param label       Label for error messages (e.g., "Magnetization Mx")
 * @param pcout       Output stream (only rank 0 should print)
 */
inline bool try_direct_solvers(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    double tol,
    const std::string& label,
    dealii::ConditionalOStream& pcout)
{
    dealii::SolverControl solver_control(1, tol);

    // Solver chain: MUMPS (parallel) → SuperLU_DIST → KLU (serial)
    static const std::vector<std::string> solver_types = {
        "Amesos_Mumps", "Amesos_Superludist", ""  // "" = KLU default
    };

    for (const auto& solver_type : solver_types)
    {
        try
        {
            if (solver_type.empty())
            {
                // KLU: default constructor (no AdditionalData)
                dealii::TrilinosWrappers::SolverDirect solver(solver_control);
                solver.solve(matrix, solution, rhs);
            }
            else
            {
                dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
                data.solver_type = solver_type;
                dealii::TrilinosWrappers::SolverDirect solver(
                    solver_control, data);
                solver.solve(matrix, solution, rhs);
            }
            return true;  // success
        }
        catch (std::exception&)
        {
            // Try next solver in chain
        }
    }

    pcout << "[" << label << "] All direct solvers failed, "
          << "falling back to iterative" << std::endl;
    return false;
}

}  // namespace SolverUtils

#endif // SOLVER_UTILS_H
