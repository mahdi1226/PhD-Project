// ============================================================================
// solvers/magnetization_solver.cc - Magnetization Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization_solver.h"
#include "output/logger.h"

template <int dim>
MagnetizationSolver<dim>::MagnetizationSolver(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      MagnetizationSolver constructed");
}

template <int dim>
void MagnetizationSolver<dim>::solve()
{
    // Equilibrium mode: m already computed in assembler via projection
    // No linear system to solve
}

template class MagnetizationSolver<2>;
// template class MagnetizationSolver<3>;