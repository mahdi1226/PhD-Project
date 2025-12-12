// ============================================================================
// solvers/ch_solver.cc - Cahn-Hilliard System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "ch_solver.h"
#include "output/logger.h"

#include <deal.II/lac/sparse_direct.h>

template <int dim>
CHSolver<dim>::CHSolver(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      CHSolver constructed");
}

template <int dim>
void CHSolver<dim>::solve()
{
    const unsigned int n_theta = problem_.theta_dof_handler_.n_dofs();
    const unsigned int n_psi   = problem_.psi_dof_handler_.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // Solution vector for coupled system
    dealii::Vector<double> solution(n_total);

    // Solve with UMFPACK (direct solver)
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(problem_.ch_matrix_);
    solver.vmult(solution, problem_.ch_rhs_);

    // Extract θ (first n_theta entries)
    for (unsigned int i = 0; i < n_theta; ++i)
        problem_.theta_solution_[i] = solution[i];

    // Extract ψ (next n_psi entries)
    for (unsigned int i = 0; i < n_psi; ++i)
        problem_.psi_solution_[i] = solution[n_theta + i];

    // Apply constraints
    problem_.theta_constraints_.distribute(problem_.theta_solution_);
    problem_.psi_constraints_.distribute(problem_.psi_solution_);
}

template class CHSolver<2>;
template class CHSolver<3>;