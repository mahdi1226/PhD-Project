// ============================================================================
// ns_solver.h - Navier-Stokes linear system solver
// ============================================================================
#ifndef NSCH_NS_SOLVER_H
#define NSCH_NS_SOLVER_H

#include "utilities/nsch_linear_algebra.h"

/**
 * @brief Solve the Navier-Stokes linear system using UMFPACK
 */
void solve_navier_stokes_system(
    const NSMatrix&      matrix,
    const NSVector&      rhs,
    NSVector&            solution,
    const NSConstraints& constraints);

#endif // NSCH_NS_SOLVER_H