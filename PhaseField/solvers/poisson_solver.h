// ============================================================================
// solvers/poisson_solver.h - Poisson (Magnetostatics) Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include "core/phase_field.h"

/**
 * @brief Solves the Poisson equation for magnetic potential φ
 *
 * After solving: h = ∇φ
 */
template <int dim>
class PoissonSolver
{
public:
    explicit PoissonSolver(PhaseFieldProblem<dim>& problem);
    
    /// Solve the Poisson system
    void solve();

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // POISSON_SOLVER_H
