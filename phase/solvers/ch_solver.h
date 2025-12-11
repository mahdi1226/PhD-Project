// ============================================================================
// ch_solver.h - Cahn-Hilliard linear system solver
// ============================================================================
#ifndef NSCH_CH_SOLVER_H
#define NSCH_CH_SOLVER_H

#include "utilities/nsch_linear_algebra.h"

/**
 * @brief Solve the Cahn-Hilliard linear system using UMFPACK
 */
void solve_cahn_hilliard_system(
    const CHMatrix&      matrix,
    const CHVector&      rhs,
    CHVector&            solution,
    const CHConstraints& constraints);

#endif // NSCH_CH_SOLVER_H