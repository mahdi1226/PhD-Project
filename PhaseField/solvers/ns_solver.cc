// ============================================================================
// solvers/ns_solver.cc - Navier-Stokes Linear Solver Implementation
//
// UPDATED: Now returns SolverInfo with iterations/residual/time
// ============================================================================

#include "solvers/ns_solver.h"
#include "solvers/ns_block_preconditioner.h"
#include "utilities/parameters.h"

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
// Simple GMRES + ILU solver - RETURNS SolverInfo
// ============================================================================
SolverInfo solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const LinearSolverParams& params,
    bool log_output)
{
    SolverInfo info;
    info.solver_name = "NS";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        if (log_output)
            std::cout << "[NS Solver] Zero RHS\n";
        info.converged = true;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = std::max(params.abs_tolerance,
                                params.rel_tolerance * rhs_norm);
    dealii::SolverControl solver_control(params.max_iterations, tol);

    typename dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = params.gmres_restart + 2;
    dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control);

    dealii::SparseILU<double> preconditioner;
    unsigned int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;

    try
    {
        preconditioner.initialize(matrix);
        solver.solve(matrix, solution, rhs, preconditioner);
        iterations = solver_control.last_step();
        final_residual = solver_control.last_value();
        converged = true;
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        iterations = e.last_step;
        final_residual = e.last_residual;
        std::cerr << "[NS Solver] GMRES: " << iterations << " iters, "
                  << "res = " << final_residual << " (tol = " << tol << ")\n";

        if (params.fallback_to_direct)
        {
            std::cerr << "[NS Solver] Falling back to UMFPACK.\n";
            dealii::SparseDirectUMFPACK direct;
            direct.initialize(matrix);
            direct.vmult(solution, rhs);
            iterations = 1;
            final_residual = 0.0;
            converged = true;
            info.used_direct = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    // Fill SolverInfo
    info.iterations = iterations;
    info.residual = final_residual;
    info.solve_time = solve_time;
    info.converged = converged;

    if (log_output || params.verbose)
    {
        std::cout << "[NS Solver] Size: " << matrix.m()
                  << ", iters: " << iterations
                  << ", res: " << final_residual
                  << ", time: " << solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Legacy interface with default parameters
// ============================================================================
SolverInfo solve_ns_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    LinearSolverParams default_params;
    default_params.type = LinearSolverParams::Type::GMRES;
    default_params.preconditioner = LinearSolverParams::Preconditioner::ILU;
    default_params.rel_tolerance = 1e-3;
    default_params.abs_tolerance = 1e-6;
    default_params.max_iterations = 3000;
    default_params.gmres_restart = 150;
    default_params.use_iterative = true;
    default_params.fallback_to_direct = true;
    default_params.verbose = verbose;

    return solve_ns_system(matrix, rhs, solution, constraints,
                           default_params, /*log_output=*/verbose);
}

// ============================================================================
// FGMRES + Block Schur preconditioner - RETURNS SolverInfo
// ============================================================================
SolverInfo solve_ns_system_schur(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::SparseMatrix<double>& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const LinearSolverParams& params,
    bool log_output)
{
    SolverInfo info;
    info.solver_name = "NS-Schur";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        if (log_output)
            std::cout << "[NS Schur] Zero RHS\n";
        info.converged = true;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const double tol = std::max(params.abs_tolerance,
                                params.rel_tolerance * rhs_norm);
    dealii::SolverControl solver_control(params.max_iterations, tol);

    dealii::SolverFGMRES<dealii::Vector<double>> solver(solver_control);

    const bool do_solve_A = true;
    BlockSchurPreconditioner preconditioner(
        matrix, pressure_mass,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        do_solve_A);

    unsigned int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;

    try
    {
        solver.solve(matrix, solution, rhs, preconditioner);
        iterations = solver_control.last_step();
        final_residual = solver_control.last_value();
        converged = true;
    }
    catch (dealii::SolverControl::NoConvergence& e)
    {
        iterations = e.last_step;
        final_residual = e.last_residual;
        std::cerr << "[NS Schur] FGMRES: " << iterations << " iters, "
                  << "res = " << final_residual << " (tol = " << tol << ")\n";

        if (params.fallback_to_direct)
        {
            std::cerr << "[NS Schur] Falling back to UMFPACK.\n";
            dealii::SparseDirectUMFPACK direct;
            direct.initialize(matrix);
            direct.vmult(solution, rhs);
            converged = true;
            info.used_direct = true;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    // Fill SolverInfo
    info.iterations = iterations;
    info.residual = final_residual;
    info.solve_time = solve_time;
    info.converged = converged;

    if (log_output || params.verbose)
    {
        std::cout << "[NS Schur] FGMRES iters: " << iterations
                  << ", A solves: " << preconditioner.n_iterations_A
                  << ", S solves: " << preconditioner.n_iterations_S
                  << ", time: " << solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Legacy Schur interface with default parameters
// ============================================================================
SolverInfo solve_ns_system_schur(
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
    LinearSolverParams default_params;
    default_params.type = LinearSolverParams::Type::FGMRES;
    default_params.preconditioner = LinearSolverParams::Preconditioner::BlockSchur;
    default_params.rel_tolerance = 1e-6;
    default_params.abs_tolerance = 1e-10;
    default_params.max_iterations = 1500;
    default_params.gmres_restart = 100;
    default_params.use_iterative = true;
    default_params.fallback_to_direct = true;
    default_params.verbose = verbose;

    return solve_ns_system_schur(matrix, rhs, solution, constraints,
                                 pressure_mass, ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                                 default_params, /*log_output=*/true);
}

// ============================================================================
// Direct solver (UMFPACK) - RETURNS SolverInfo
// ============================================================================
SolverInfo solve_ns_system_direct(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    dealii::Vector<double>& solution,
    const dealii::AffineConstraints<double>& constraints,
    bool verbose)
{
    SolverInfo info;
    info.solver_name = "NS-Direct";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();
    info.used_direct = true;
    info.iterations = 1;

    if (solution.size() != rhs.size())
        solution.reinit(rhs.size());

    auto start = std::chrono::high_resolution_clock::now();

    dealii::SparseDirectUMFPACK direct;
    direct.initialize(matrix);
    direct.vmult(solution, rhs);

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    info.solve_time = solve_time;
    info.converged = true;
    info.residual = 0.0;

    if (verbose)
        std::cout << "[NS Direct] Size: " << matrix.m()
                  << ", time: " << solve_time << "s\n";

    return info;
}

// ============================================================================
// Extract solutions (unchanged)
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
// Assemble pressure mass matrix (unchanged)
// ============================================================================
template <int dim>
void assemble_pressure_mass_matrix(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    dealii::SparsityPattern& sparsity,
    dealii::SparseMatrix<double>& mass_matrix)
{
    const unsigned int n_p = p_dof_handler.n_dofs();
    const dealii::FiniteElement<dim>& fe = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    dealii::DynamicSparsityPattern dsp(n_p, n_p);
    dealii::DoFTools::make_sparsity_pattern(
        p_dof_handler,
        dsp,
        p_constraints,
        /*keep_constrained_dofs=*/ false);
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

        p_constraints.distribute_local_to_global(
            cell_matrix,
            local_dof_indices,
            mass_matrix);
    }

    std::cout << "[Pressure Mass] Assembled: " << n_p << " DoFs\n";
}

// Explicit instantiations
template void assemble_pressure_mass_matrix<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    dealii::SparseMatrix<double>&);
template void assemble_pressure_mass_matrix<3>(
    const dealii::DoFHandler<3>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    dealii::SparseMatrix<double>&);