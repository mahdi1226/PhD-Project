// ============================================================================
// solvers/ch_solver.cc - Cahn-Hilliard Solver
//
// FIXED VERSION: Uses combined constraints properly
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
    const unsigned int n_psi = problem_.psi_dof_handler_.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // Solution vector for coupled system
    dealii::Vector<double> solution(n_total);

    // Condense the system with combined constraints BEFORE solving
    problem_.ch_combined_constraints_.condense(problem_.ch_matrix_, problem_.ch_rhs_);

    // Solve using UMFPACK
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(problem_.ch_matrix_);
    solver.vmult(solution, problem_.ch_rhs_);

    // Distribute constraints to solution
    problem_.ch_combined_constraints_.distribute(solution);

    // Extract θ and ψ from coupled solution using index maps
    for (unsigned int i = 0; i < n_theta; ++i)
        problem_.theta_solution_[i] = solution[problem_.theta_to_ch_map_[i]];

    for (unsigned int i = 0; i < n_psi; ++i)
        problem_.psi_solution_[i] = solution[problem_.psi_to_ch_map_[i]];

    // Apply individual constraints
    problem_.theta_constraints_.distribute(problem_.theta_solution_);
    problem_.psi_constraints_.distribute(problem_.psi_solution_);
}

template class CHSolver<2>;
// template class CHSolver<3>;