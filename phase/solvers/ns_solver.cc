// ============================================================================
// ns_solver.cc - Navier-Stokes linear system solver
// ============================================================================
#include "ns_solver.h"

#include <deal.II/lac/sparse_direct.h>

void solve_navier_stokes_system(
    const NSMatrix&      matrix,
    const NSVector&      rhs,
    NSVector&            solution,
    const NSConstraints& constraints)
{
    // Use UMFPACK direct solver
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(solution, rhs);

    // Apply constraints
    constraints.distribute(solution);
}