// ============================================================================
// solvers/ns_solver.cc - Navier-Stokes Solver
//
// FIXED VERSION: Uses combined constraints properly
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "ns_solver.h"
#include "output/logger.h"

#include <deal.II/lac/sparse_direct.h>

template <int dim>
NSSolver<dim>::NSSolver(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      NSSolver constructed");
}

template <int dim>
void NSSolver<dim>::solve()
{
    const unsigned int n_ux = problem_.ux_dof_handler_.n_dofs();
    const unsigned int n_uy = problem_.uy_dof_handler_.n_dofs();
    const unsigned int n_p = problem_.p_dof_handler_.n_dofs();
    const unsigned int n_total = n_ux + n_uy + n_p;

    // Solution vector for coupled system
    dealii::Vector<double> solution(n_total);

    // Condense the system with combined constraints BEFORE solving
    problem_.ns_combined_constraints_.condense(problem_.ns_matrix_, problem_.ns_rhs_);

    // Solve using UMFPACK
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(problem_.ns_matrix_);
    solver.vmult(solution, problem_.ns_rhs_);

    // Distribute constraints to solution
    problem_.ns_combined_constraints_.distribute(solution);

    // Extract u_x, u_y, p from coupled solution using index maps
    for (unsigned int i = 0; i < n_ux; ++i)
        problem_.ux_solution_[i] = solution[problem_.ux_to_ns_map_[i]];

    for (unsigned int i = 0; i < n_uy; ++i)
        problem_.uy_solution_[i] = solution[problem_.uy_to_ns_map_[i]];

    for (unsigned int i = 0; i < n_p; ++i)
        problem_.p_solution_[i] = solution[problem_.p_to_ns_map_[i]];

    // Apply individual constraints
    problem_.ux_constraints_.distribute(problem_.ux_solution_);
    problem_.uy_constraints_.distribute(problem_.uy_solution_);
    problem_.p_constraints_.distribute(problem_.p_solution_);
}

template class NSSolver<2>;
// template class NSSolver<3>;