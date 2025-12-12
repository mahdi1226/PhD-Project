// ============================================================================
// solvers/ch_solver.h - Cahn-Hilliard System Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_SOLVER_H
#define CH_SOLVER_H

#include "core/phase_field.h"

/**
 * @brief Solves the coupled Cahn-Hilliard system for θ and ψ
 *
 * The system is solved monolithically using a direct solver (UMFPACK).
 */
template <int dim>
class CHSolver
{
public:
    explicit CHSolver(PhaseFieldProblem<dim>& problem);
    
    /// Solve the assembled CH system
    void solve();

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // CH_SOLVER_H
