// ============================================================================
// ch_solver.cc - Cahn-Hilliard linear system solver
// ============================================================================
#include "ch_solver.h"

#include <deal.II/lac/sparse_direct.h>

void solve_cahn_hilliard_system(
    const CHMatrix&      matrix,
    const CHVector&      rhs,
    CHVector&            solution,
    const CHConstraints& constraints)
{
    // Use UMFPACK direct solver
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(solution, rhs);

    // Apply constraints
    constraints.distribute(solution);
}