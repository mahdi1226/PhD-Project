// ============================================================================
// solvers/ch_solver.cc - Cahn-Hilliard System Solver Implementation
//
// UPDATED: Now returns SolverInfo with iterations/residual/time
// ============================================================================

#include "solvers/ch_solver.h"
#include "utilities/parameters.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>

#include <iostream>
#include <chrono>

// ============================================================================
// Main solver with explicit parameters - RETURNS SolverInfo
// ============================================================================
SolverInfo solve_ch_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution,
    const LinearSolverParams& params,
    bool log_output)
{
    SolverInfo info;
    info.solver_name = "CH";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    dealii::Vector<double> coupled_solution(rhs.size());

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        coupled_solution = 0;
        constraints.distribute(coupled_solution);

        for (unsigned int i = 0; i < theta_solution.size(); ++i)
            theta_solution[i] = coupled_solution[theta_to_ch_map[i]];
        for (unsigned int i = 0; i < psi_solution.size(); ++i)
            psi_solution[i] = coupled_solution[psi_to_ch_map[i]];

        if (log_output)
            std::cout << "[CH Solver] Zero RHS, solution set to zero\n";

        info.converged = true;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();
    bool converged = false;
    unsigned int iterations = 0;
    double final_residual = 0.0;

    if (params.use_iterative)
    {
        const double tol = std::max(params.abs_tolerance,
                                    params.rel_tolerance * rhs_norm);

        dealii::SolverControl solver_control(params.max_iterations, tol);

        typename dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = params.gmres_restart + 2;
        dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control, gmres_data);

        dealii::SparseILU<double> preconditioner;
        preconditioner.initialize(matrix);

        try
        {
            solver.solve(matrix, coupled_solution, rhs, preconditioner);
            converged = true;
            iterations = solver_control.last_step();
            final_residual = solver_control.last_value();
        }
        catch (dealii::SolverControl::NoConvergence& e)
        {
            std::cerr << "[CH Solver] WARNING: GMRES did not converge after "
                      << e.last_step << " iterations. "
                      << "Residual = " << e.last_residual << "\n";
            iterations = e.last_step;
            final_residual = e.last_residual;

            if (params.fallback_to_direct)
                std::cerr << "[CH Solver] Falling back to direct solver.\n";
        }
    }

    if (!converged && params.fallback_to_direct)
    {
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(coupled_solution, rhs);
        converged = true;
        iterations = 1;
        final_residual = 0.0;
        info.used_direct = true;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(coupled_solution);

    // Extract individual solutions
    for (unsigned int i = 0; i < theta_solution.size(); ++i)
        theta_solution[i] = coupled_solution[theta_to_ch_map[i]];

    for (unsigned int i = 0; i < psi_solution.size(); ++i)
        psi_solution[i] = coupled_solution[psi_to_ch_map[i]];

    // Fill SolverInfo
    info.iterations = iterations;
    info.residual = final_residual;
    info.solve_time = solve_time;
    info.converged = converged;

    if (log_output || params.verbose)
    {
        std::cout << "[CH Solver] Size: " << matrix.m()
                  << ", nnz: " << matrix.n_nonzero_elements()
                  << ", iterations: " << iterations
                  << ", residual: " << final_residual
                  << ", time: " << solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Legacy interface with default parameters
// ============================================================================
SolverInfo solve_ch_system(
    const dealii::SparseMatrix<double>& matrix,
    const dealii::Vector<double>& rhs,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution)
{
    // Default CH parameters (nonsymmetric: GMRES + ILU)
    LinearSolverParams default_params;
    default_params.type = LinearSolverParams::Type::GMRES;
    default_params.preconditioner = LinearSolverParams::Preconditioner::ILU;
    default_params.rel_tolerance = 1e-8;
    default_params.abs_tolerance = 1e-12;
    default_params.max_iterations = 2000;
    default_params.gmres_restart = 50;
    default_params.use_iterative = true;
    default_params.fallback_to_direct = true;
    default_params.verbose = false;

    return solve_ch_system(matrix, rhs, constraints,
                           theta_to_ch_map, psi_to_ch_map,
                           theta_solution, psi_solution,
                           default_params, /*log_output=*/true);
}