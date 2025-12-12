// ============================================================================
// solvers/magnetization_solver.h - Magnetization Equation Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_SOLVER_H
#define MAGNETIZATION_SOLVER_H

#include "core/phase_field.h"

/**
 * @brief Solves the magnetization equation for m
 *
 * Two modes:
 *   1. Full PDE solve (Eq. 42c)
 *   2. Quasi-equilibrium: m = χ_θ h (when T → 0)
 */
template <int dim>
class MagnetizationSolver
{
public:
    explicit MagnetizationSolver(PhaseFieldProblem<dim>& problem);
    
    /// Solve the magnetization system (or apply equilibrium)
    void solve();

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // MAGNETIZATION_SOLVER_H
