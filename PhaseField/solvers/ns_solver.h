// ============================================================================
// solvers/ns_solver.h - Navier-Stokes System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_SOLVER_H
#define NS_SOLVER_H

#include "core/phase_field.h"

/**
 * @brief Solves the coupled Navier-Stokes system for u and p
 *
 * The system is solved monolithically using a direct solver (UMFPACK).
 */
template <int dim>
class NSSolver
{
public:
    explicit NSSolver(PhaseFieldProblem<dim>& problem);
    
    /// Solve the assembled NS system
    void solve();

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // NS_SOLVER_H
