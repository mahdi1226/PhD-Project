// ============================================================================
// solvers/ns_solver.cc - Navier-Stokes Linear Solver Implementation
//
// Solves the saddle-point NS system using UMFPACK direct solver.
//
// Workflow:
//   1. Assembler calls condense(matrix, rhs) to incorporate constraints
//   2. This solver solves the modified system
//   3. distribute(solution) fixes up constrained DoF values
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/ns_solver.h"

#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <chrono>

void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    using namespace dealii;

    // Ensure solution vector is properly sized
    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();

    // Use UMFPACK direct solver (robust for saddle point systems)
    // Note: Matrix should already have constraints condensed by assembler
    SparseDirectUMFPACK solver;
    solver.initialize(matrix);
    solver.vmult(solution, rhs);

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    // Apply constraints (distribute values to constrained DoFs)
    // This handles:
    //   - Dirichlet BCs: sets prescribed values
    //   - Hanging nodes: interpolates from parent DoFs
    constraints.distribute(solution);

    if (verbose)
    {
        // Compute residual for diagnostics
        Vector<double> residual(rhs.size());
        matrix.vmult(residual, solution);
        residual -= rhs;

        // Zero out constrained DoFs in residual (they're not meaningful)
        for (unsigned int i = 0; i < residual.size(); ++i)
            if (constraints.is_constrained(i))
                residual[i] = 0.0;

        std::cout << "[NS Solver] Size: " << matrix.m()
                  << ", nnz: " << matrix.n_nonzero_elements()
                  << ", time: " << solve_time << "s"
                  << ", |residual|: " << residual.l2_norm() << "\n";
    }
}

void extract_ns_solutions(
    const dealii::Vector<double>& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution)
{
    // Ensure output vectors are properly sized
    if (ux_solution.size() != ux_to_ns_map.size())
        ux_solution.reinit(ux_to_ns_map.size());
    if (uy_solution.size() != uy_to_ns_map.size())
        uy_solution.reinit(uy_to_ns_map.size());
    if (p_solution.size() != p_to_ns_map.size())
        p_solution.reinit(p_to_ns_map.size());

    // Extract ux
    for (unsigned int i = 0; i < ux_to_ns_map.size(); ++i)
        ux_solution[i] = ns_solution[ux_to_ns_map[i]];

    // Extract uy
    for (unsigned int i = 0; i < uy_to_ns_map.size(); ++i)
        uy_solution[i] = ns_solution[uy_to_ns_map[i]];

    // Extract p
    for (unsigned int i = 0; i < p_to_ns_map.size(); ++i)
        p_solution[i] = ns_solution[p_to_ns_map[i]];
}