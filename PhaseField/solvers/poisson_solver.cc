// ============================================================================
// solvers/poisson_solver.cc - Poisson (Magnetostatics) Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "poisson_solver.h"
#include "output/logger.h"

#include <deal.II/lac/sparse_direct.h>

template <int dim>
PoissonSolver<dim>::PoissonSolver(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      PoissonSolver constructed");
}

template <int dim>
void PoissonSolver<dim>::solve()
{
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(problem_.poisson_matrix_);
    solver.vmult(problem_.phi_solution_, problem_.poisson_rhs_);

    problem_.phi_constraints_.distribute(problem_.phi_solution_);
}

template class PoissonSolver<2>;
//template class PoissonSolver<3>;