// ============================================================================
// mms/ns/ns_variable_nu_mms_test.cc - NS Variable Viscosity MMS Test
//
// Tests variable viscosity ν(θ) using PRODUCTION code:
//   - setup_ns_coupled_system() from setup/ns_setup.h
//   - assemble_ns_system() from assembly/ns_assembler.h
//   - solve_ns_system_*() from solvers/ns_solver.h
//
// APPROACH:
//   - Prescribe θ analytically (not solved from CH)
//   - Use different ν_water ≠ ν_ferro to create variable viscosity
//   - Verify NS convergence rates maintained
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ns/ns_variable_nu_mms_test.h"
#include "mms/ns/ns_variable_nu_mms.h"
#include "mms/ns/ns_mms.h"
#include "mms/mms_verification.h"

// Production setup
#include "setup/ns_setup.h"

// Production assembly
#include "assembly/ns_assembler.h"

// Production solvers
#include "solvers/ns_solver.h"
#include "solvers/ns_block_preconditioner.h"

// For NSSolverType
#include "mms/ns/ns_mms_test.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>

// Global NS solver type (set by test_mms.cc command line)
extern NSSolverType g_ns_solver_type;

namespace
{
constexpr int dim = 2;

// ============================================================================
// Run single refinement level
// ============================================================================
NSMMSResult run_single_refinement_varnu(
    unsigned int refinement,
    Parameters params,
    unsigned int n_time_steps)
{
    NSMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Time parameters
    // ========================================================================
    const double t_init = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double L_y = params.domain.y_max - params.domain.y_min;

    // Variable viscosity parameters
    const double nu_water = 1.0;
    const double nu_ferro = 2.0;  // Different from nu_water!
    params.physics.nu_water = nu_water;
    params.physics.nu_ferro = nu_ferro;

    // ========================================================================
    // Setup mesh
    // ========================================================================
    dealii::Triangulation<dim> triangulation;
    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params.domain.initial_cells_x;
    subdivisions[1] = params.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    // Assign boundary IDs
    for (auto& face : triangulation.active_face_iterators())
    {
        if (!face->at_boundary())
            continue;
        const auto center = face->center();
        const double tol = 1e-10;
        if (std::abs(center[1] - params.domain.y_min) < tol)
            face->set_boundary_id(0);
        else if (std::abs(center[0] - params.domain.x_max) < tol)
            face->set_boundary_id(1);
        else if (std::abs(center[1] - params.domain.y_max) < tol)
            face->set_boundary_id(2);
        else if (std::abs(center[0] - params.domain.x_min) < tol)
            face->set_boundary_id(3);
    }

    triangulation.refine_global(refinement);
    result.h = dealii::GridTools::minimal_cell_diameter(triangulation);

    // ========================================================================
    // Setup NS (Taylor-Hood Q2-Q1)
    // ========================================================================
    dealii::FE_Q<dim> fe_velocity(params.fe.degree_velocity);
    dealii::FE_Q<dim> fe_pressure(params.fe.degree_pressure);

    dealii::DoFHandler<dim> ux_dof_handler(triangulation);
    dealii::DoFHandler<dim> uy_dof_handler(triangulation);
    dealii::DoFHandler<dim> p_dof_handler(triangulation);

    ux_dof_handler.distribute_dofs(fe_velocity);
    uy_dof_handler.distribute_dofs(fe_velocity);
    p_dof_handler.distribute_dofs(fe_pressure);

    const unsigned int n_ux = ux_dof_handler.n_dofs();
    const unsigned int n_uy = uy_dof_handler.n_dofs();
    const unsigned int n_p = p_dof_handler.n_dofs();
    result.n_dofs = n_ux + n_uy + n_p;

    // ========================================================================
    // Setup θ (for variable viscosity) - Q1 elements
    // ========================================================================
    dealii::FE_Q<dim> fe_theta(params.fe.degree_phase);
    dealii::DoFHandler<dim> theta_dof_handler(triangulation);
    theta_dof_handler.distribute_dofs(fe_theta);

    dealii::Vector<double> theta_solution(theta_dof_handler.n_dofs());
    dealii::Vector<double> psi_solution(theta_dof_handler.n_dofs());  // Not used, just for assembler signature

    // Project prescribed θ
    NSVarNuPrescribedTheta<dim> prescribed_theta(L_y);
    dealii::VectorTools::interpolate(theta_dof_handler, prescribed_theta, theta_solution);
    psi_solution = 0;  // ψ not used in variable viscosity

    // ========================================================================
    // Setup NS constraints (MMS Dirichlet BCs)
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;

    setup_ns_mms_velocity_constraints(ux_dof_handler, uy_dof_handler,
                                      ux_constraints, uy_constraints);
    setup_ns_mms_pressure_constraints(p_dof_handler, p_constraints);

    // ========================================================================
    // Setup coupled NS system using PRODUCTION code
    // ========================================================================
    std::vector<dealii::types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
    dealii::AffineConstraints<double> ns_constraints;
    dealii::SparsityPattern ns_sparsity;

    setup_ns_coupled_system<dim>(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_constraints, uy_constraints, p_constraints,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_constraints, ns_sparsity, false);

    dealii::SparseMatrix<double> ns_matrix(ns_sparsity);
    dealii::Vector<double> ns_rhs(result.n_dofs);
    dealii::Vector<double> ns_solution(result.n_dofs);

    // ========================================================================
    // Setup pressure mass matrix for Schur solver
    // ========================================================================
    dealii::SparsityPattern p_mass_sparsity;
    dealii::SparseMatrix<double> pressure_mass_matrix;
    std::unique_ptr<BlockSchurPreconditioner> schur_preconditioner;

    if (g_ns_solver_type == NSSolverType::Schur)
    {
        assemble_pressure_mass_matrix<dim>(
            p_dof_handler, p_constraints,
            p_mass_sparsity, pressure_mass_matrix);
    }

    // ========================================================================
    // Solution vectors
    // ========================================================================
    dealii::Vector<double> ux_solution(n_ux), ux_old(n_ux);
    dealii::Vector<double> uy_solution(n_uy), uy_old(n_uy);
    dealii::Vector<double> p_solution(n_p);

    // Initial conditions from exact solution
    NSExactVelocityX<dim> exact_ux_init(t_init, L_y);
    NSExactVelocityY<dim> exact_uy_init(t_init, L_y);
    NSExactPressure<dim> exact_p_init(t_init, L_y);

    dealii::VectorTools::interpolate(ux_dof_handler, exact_ux_init, ux_solution);
    dealii::VectorTools::interpolate(uy_dof_handler, exact_uy_init, uy_solution);
    dealii::VectorTools::interpolate(p_dof_handler, exact_p_init, p_solution);

    ux_old = ux_solution;
    uy_old = uy_solution;

    // ========================================================================
    // Time stepping
    // ========================================================================
    double current_time = t_init;
    unsigned int total_iterations = 0;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ==================================================================
        // Assemble NS system using PRODUCTION assembler
        // The assembler handles variable ν(θ) automatically when theta_solution
        // is provided with different nu_water/nu_ferro
        // ==================================================================
        ns_matrix = 0;
        ns_rhs = 0;

        assemble_ns_system<dim>(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            theta_dof_handler, theta_dof_handler,  // psi_dof = theta_dof (same mesh)
            nullptr, nullptr,  // No phi, M DoF handlers
            ux_old, uy_old,
            theta_solution, psi_solution,
            nullptr, nullptr, nullptr,  // No phi, mx, my solutions
            params,
            dt, current_time,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_constraints,
            ns_matrix, ns_rhs);

        // ==================================================================
        // Solve using PRODUCTION solvers based on global solver type
        // ==================================================================
        ns_solution = 0;
        SolverInfo ns_info;

        switch (g_ns_solver_type)
        {
            case NSSolverType::Direct:
                ns_info = solve_ns_system_direct(
                    ns_matrix, ns_rhs, ns_solution,
                    ns_constraints, false);
                break;

            case NSSolverType::GMRES_ILU:
                ns_info = solve_ns_system(
                    ns_matrix, ns_rhs, ns_solution,
                    ns_constraints,
                    params.solvers.ns.max_iterations,
                    params.solvers.ns.rel_tolerance,
                    false);
                break;

            case NSSolverType::Schur:
                if (!schur_preconditioner)
                {
                    schur_preconditioner = std::make_unique<BlockSchurPreconditioner>(
                        ns_matrix, pressure_mass_matrix,
                        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                        true);
                }
                else
                {
                    schur_preconditioner->update(ns_matrix, pressure_mass_matrix);
                }
                ns_info = solve_ns_system_schur(
                    ns_matrix, ns_rhs, ns_solution,
                    ns_constraints, *schur_preconditioner,
                    params.solvers.ns.max_iterations,
                    params.solvers.ns.rel_tolerance,
                    false);
                break;
        }

        total_iterations += ns_info.iterations;
        ns_constraints.distribute(ns_solution);

        // ==================================================================
        // Extract solutions
        // ==================================================================
        for (unsigned int i = 0; i < n_ux; ++i)
        {
            ux_solution[i] = ns_solution[ux_to_ns_map[i]];
            uy_solution[i] = ns_solution[uy_to_ns_map[i]];
        }
        for (unsigned int i = 0; i < n_p; ++i)
            p_solution[i] = ns_solution[p_to_ns_map[i]];

        ux_constraints.distribute(ux_solution);
        uy_constraints.distribute(uy_solution);
        p_constraints.distribute(p_solution);

        ux_old = ux_solution;
        uy_old = uy_solution;
    }

    // ========================================================================
    // Compute errors using PRODUCTION error computation
    // ========================================================================
    NSMMSError errors = compute_ns_mms_error(
        ux_dof_handler, uy_dof_handler, p_dof_handler,
        ux_solution, uy_solution, p_solution,
        current_time, L_y);

    result.ux_L2 = errors.ux_L2;
    result.ux_H1 = errors.ux_H1;
    result.uy_L2 = errors.uy_L2;
    result.uy_H1 = errors.uy_H1;
    result.p_L2 = errors.p_L2;
    result.div_U_L2 = errors.div_U_L2;
    result.solver_iterations = total_iterations;

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

}  // anonymous namespace

// ============================================================================
// Main entry point
// ============================================================================
MMSConvergenceResult run_ns_variable_nu_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::NS_VARIABLE_NU;
    result.fe_degree = params.fe.degree_velocity;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = params.fe.degree_velocity + 1;  // 3 for Q2
    result.expected_H1_rate = params.fe.degree_velocity;      // 2 for Q2

    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.enable_ns = true;

    const double t_init = 0.1;
    const double t_final = 0.2;
    result.dt = (t_final - t_init) / n_time_steps;

    std::cout << "\n[NS_VARIABLE_NU] Running convergence study...\n";
    std::cout << "  t in [" << t_init << ", " << t_final << "], dt = " << result.dt << "\n";
    std::cout << "  ν_water = " << 1.0 << ", ν_ferro = " << 2.0 << "\n";
    std::cout << "  θ = cos(πx)cos(πy/L_y) (prescribed)\n";
    std::cout << "  Testing variable viscosity: ν(θ) = ν_w(1-θ)/2 + ν_f(1+θ)/2\n";
    std::cout << "  NS Solver: " << to_string(g_ns_solver_type) << "\n";
    std::cout << "  Using PRODUCTION assembler and solver\n\n";

    for (unsigned int ref : refinements)
    {
        std::cout << "  Refinement " << ref << "... " << std::flush;

        NSMMSResult ns_result = run_single_refinement_varnu(ref, mms_params, n_time_steps);

        result.refinements.push_back(ns_result.refinement);
        result.h_values.push_back(ns_result.h);
        result.ux_L2.push_back(ns_result.ux_L2);
        result.ux_H1.push_back(ns_result.ux_H1);
        result.uy_L2.push_back(ns_result.uy_L2);
        result.uy_H1.push_back(ns_result.uy_H1);
        result.p_L2.push_back(ns_result.p_L2);
        result.div_u_L2.push_back(ns_result.div_U_L2);
        result.n_dofs.push_back(ns_result.n_dofs);
        result.wall_times.push_back(ns_result.total_time);

        std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << ns_result.ux_L2
                  << ", p_L2=" << ns_result.p_L2
                  << ", iters=" << ns_result.solver_iterations
                  << ", time=" << std::fixed << std::setprecision(1) << ns_result.total_time << "s\n";
    }

    result.compute_rates();
    return result;
}