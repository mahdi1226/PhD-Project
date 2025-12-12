// ============================================================================
// solvers/ns_solver.cc - Navier-Stokes Solver
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
    const unsigned int n_u = problem_.ux_dof_handler_.n_dofs();
    const unsigned int n_p = problem_.p_dof_handler_.n_dofs();
    const unsigned int n_total = 2 * n_u + n_p;

    dealii::Vector<double> solution(n_total);

    dealii::SparseDirectUMFPACK solver;
    solver.initialize(problem_.ns_matrix_);
    solver.vmult(solution, problem_.ns_rhs_);

    // Extract u_x [0, n_u)
    for (unsigned int i = 0; i < n_u; ++i)
        problem_.ux_solution_[i] = solution[i];

    // Extract u_y [n_u, 2*n_u)
    for (unsigned int i = 0; i < n_u; ++i)
        problem_.uy_solution_[i] = solution[n_u + i];

    // Extract p [2*n_u, 2*n_u + n_p)
    for (unsigned int i = 0; i < n_p; ++i)
        problem_.p_solution_[i] = solution[2 * n_u + i];

    problem_.ux_constraints_.distribute(problem_.ux_solution_);
    problem_.uy_constraints_.distribute(problem_.uy_solution_);
    problem_.p_constraints_.distribute(problem_.p_solution_);
}

template class NSSolver<2>;
// template class NSSolver<3>;