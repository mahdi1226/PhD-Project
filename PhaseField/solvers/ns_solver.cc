// ============================================================================
// solvers/ns_solver.cc - Navier-Stokes Linear Solver Implementation
//
// Implements three solvers:
//   1. Simple GMRES + ILU (baseline)
//   2. FGMRES + Block Schur preconditioner (following step-56)
//   3. Direct UMFPACK (fallback)
//
// Reference: deal.II step-22, step-56
// ============================================================================

#include "solvers/ns_solver.h"
#include "solvers/ns_block_preconditioner.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <iostream>
#include <chrono>

// ============================================================================
// Simple GMRES + ILU solver (baseline)
// ============================================================================
void solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    const double rel_tolerance = 1e-3;
    const double abs_tolerance = 1e-6;
    const unsigned int max_iterations = 3000;
    const unsigned int gmres_restart = 150;

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        std::cout << "[NS Solver] Zero RHS\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = std::max(abs_tolerance, rel_tolerance * rhs_norm);
    dealii::SolverControl solver_control(max_iterations, tol);

    typename dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = gmres_restart + 2;
    dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);

    dealii::SparseILU<double> preconditioner;
    unsigned int iterations = 0;
    double final_residual = 0.0;

    try
    {
        preconditioner.initialize(matrix);
        solver.solve(matrix, solution, rhs, preconditioner);
        iterations = solver_control.last_step();
        final_residual = solver_control.last_value();
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        iterations = e.last_step;
        final_residual = e.last_residual;
        std::cerr << "[NS Solver] GMRES: " << iterations << " iters, "
                  << "res = " << final_residual << " (tol = " << tol << ")\n";

        std::cerr << "[NS Solver] Falling back to UMFPACK.\n";
        dealii::SparseDirectUMFPACK direct;
        direct.initialize(matrix);
        direct.vmult(solution, rhs);
        iterations = 1;
        final_residual = 0.0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    if (verbose)
    {
        std::cout << "[NS Solver] Size: " << matrix.m()
                  << ", iters: " << iterations
                  << ", res: " << final_residual
                  << ", time: " << solve_time << "s\n";
    }
}

// ============================================================================
// FGMRES + Block Schur preconditioner (following step-56)
// ============================================================================
void solve_ns_system_schur(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::SparseMatrix<double>& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    bool verbose)
{
    const double rel_tolerance = 1e-6;
    const unsigned int max_iterations = 500;

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        std::cout << "[NS Schur] Zero RHS\n";
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = rel_tolerance * rhs_norm;
    dealii::SolverControl solver_control(max_iterations, tol);

    dealii::SolverFGMRES<dealii::Vector<double>> solver(solver_control);

    const bool do_solve_A = true;
    BlockSchurPreconditioner preconditioner(
        matrix, pressure_mass,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        do_solve_A);

    unsigned int iterations = 0;
    double final_residual = 0.0;

    try
    {
        solver.solve(matrix, solution, rhs, preconditioner);
        iterations = solver_control.last_step();
        final_residual = solver_control.last_value();
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        iterations = e.last_step;
        final_residual = e.last_residual;
        std::cerr << "[NS Schur] FGMRES: " << iterations << " iters, "
                  << "res = " << final_residual << " (tol = " << tol << ")\n";

        std::cerr << "[NS Schur] Falling back to UMFPACK.\n";
        dealii::SparseDirectUMFPACK direct;
        direct.initialize(matrix);
        direct.vmult(solution, rhs);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    if (verbose)
    {
        std::cout << "[NS Schur] FGMRES iters: " << iterations
                  << ", A solves: " << preconditioner.n_iterations_A
                  << ", S solves: " << preconditioner.n_iterations_S
                  << ", time: " << solve_time << "s\n";
    }
}

// ============================================================================
// Direct solver (UMFPACK)
// ============================================================================
void solve_ns_system_direct(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();

    dealii::SparseDirectUMFPACK direct;
    direct.initialize(matrix);
    direct.vmult(solution, rhs);

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    if (verbose)
        std::cout << "[NS Direct] Size: " << matrix.m()
                  << ", time: " << solve_time << "s\n";
}

// ============================================================================
// Extract solutions
// ============================================================================
void extract_ns_solutions(
    const dealii::Vector<double>& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution)
{
    if (ux_solution.size() != ux_to_ns_map.size())
        ux_solution.reinit(ux_to_ns_map.size());
    if (uy_solution.size() != uy_to_ns_map.size())
        uy_solution.reinit(uy_to_ns_map.size());
    if (p_solution.size() != p_to_ns_map.size())
        p_solution.reinit(p_to_ns_map.size());

    for (unsigned int i = 0; i < ux_to_ns_map.size(); ++i)
        ux_solution[i] = ns_solution[ux_to_ns_map[i]];
    for (unsigned int i = 0; i < uy_to_ns_map.size(); ++i)
        uy_solution[i] = ns_solution[uy_to_ns_map[i]];
    for (unsigned int i = 0; i < p_to_ns_map.size(); ++i)
        p_solution[i] = ns_solution[p_to_ns_map[i]];
}

// ============================================================================
// Assemble pressure mass matrix
// ============================================================================
template <int dim>
void assemble_pressure_mass_matrix(
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::SparsityPattern& sparsity,
    dealii::SparseMatrix<double>& mass_matrix)
{
    const unsigned int n_p = p_dof_handler.n_dofs();
    const dealii::FiniteElement<dim>& fe = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    dealii::DynamicSparsityPattern dsp(n_p, n_p);
    dealii::DoFTools::make_sparsity_pattern(p_dof_handler, dsp);
    sparsity.copy_from(dsp);
    mass_matrix.reinit(sparsity);

    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : p_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell_matrix = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                         fe_values.shape_value(j, q) *
                                         fe_values.JxW(q);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                mass_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));
    }

    std::cout << "[Pressure Mass] Assembled: " << n_p << " DoFs\n";
}

// Explicit instantiations
template void assemble_pressure_mass_matrix<2>(
    const dealii::DoFHandler<2>&, dealii::SparsityPattern&, dealii::SparseMatrix<double>&);
template void assemble_pressure_mass_matrix<3>(
    const dealii::DoFHandler<3>&, dealii::SparsityPattern&, dealii::SparseMatrix<double>&);