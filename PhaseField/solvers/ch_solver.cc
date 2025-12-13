// ============================================================================
// solvers/ch_solver.cc - Cahn-Hilliard System Solver Implementation
//
// Extracted from OLD nsch_problem_solver.cc lines 298-335
//
// Uses UMFPACK direct solver for the coupled θ-ψ system.
// ============================================================================

#include "ch_solver.h"

#include <deal.II/lac/sparse_direct.h>

void solve_ch_system(
    const dealii::SparseMatrix<double>&  matrix,
    const dealii::Vector<double>&        rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>&              theta_solution,
    dealii::Vector<double>&              psi_solution)
{
    // Solve coupled system
    dealii::Vector<double> coupled_solution(rhs.size());

    dealii::SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(coupled_solution, rhs);

    // Distribute constrained DoFs
    constraints.distribute(coupled_solution);

    // Extract θ solution using index map
    for (unsigned int i = 0; i < theta_solution.size(); ++i)
        theta_solution[i] = coupled_solution[theta_to_ch_map[i]];

    // Extract ψ solution using index map
    for (unsigned int i = 0; i < psi_solution.size(); ++i)
        psi_solution[i] = coupled_solution[psi_to_ch_map[i]];
}