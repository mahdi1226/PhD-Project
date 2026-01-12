// ============================================================================
// mms/ns/ns_magnetization_mms_test.cc - NS with Magnetic Force MMS Test
//
// Tests the magnetic body force term in production NS assembler:
//   F_mag = μ₀(M·∇)H
//
// APPROACH:
//   1. Use NS MMS exact solutions (u, p) from ns_mms.h
//   2. Prescribe M and φ with known analytical forms
//   3. Production assembler computes F_mag from M, H=∇φ
//   4. Verify NS convergence rates are maintained
//
// PRODUCTION CODE USED:
//   - setup_ns_coupled_system() from setup/ns_setup.h
//   - assemble_ns_system() from assembly/ns_assembler.h (WITH magnetic force)
//   - solve_ns_system_schur() from solvers/ns_solver.h
//   - BlockSchurPreconditioner from solvers/ns_block_preconditioner.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ns/ns_magnetization_mms_test.h"
#include "mms/ns/ns_mms.h"
#include "mms/ns/ns_magnetization_mms.h"
#include "../mms_core/mms_verification.h"

// PRODUCTION code
#include "setup/ns_setup.h"
#include "assembly/ns_assembler.h"
#include "solvers/ns_solver.h"
#include "solvers/ns_block_preconditioner.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <memory>


constexpr int dim = 2;

// ============================================================================
// NS_MAGNETIZATION Result Structure
// ============================================================================
struct NSMagMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;

    double total_time = 0.0;
    unsigned int solver_iterations = 0;
};

struct NSMagMMSConvergenceResult
{
    std::vector<NSMagMMSResult> results;
    std::vector<double> ux_L2_rate;
    std::vector<double> ux_H1_rate;
    std::vector<double> p_L2_rate;

    unsigned int fe_degree_velocity = 2;
    unsigned int fe_degree_pressure = 1;
    unsigned int n_time_steps = 0;
    double dt = 0.0;
    double nu = 1.0;
    double mu_0 = 1.0;

    double expected_vel_L2_rate = 3.0;
    double expected_vel_H1_rate = 2.0;
    double expected_p_L2_rate = 2.0;

    void compute_rates();
    void print() const;
    bool passes(double tol = 0.3) const;
};

void NSMagMMSConvergenceResult::compute_rates()
{
    const size_t n = results.size();
    ux_L2_rate.resize(n, 0.0);
    ux_H1_rate.resize(n, 0.0);
    p_L2_rate.resize(n, 0.0);

    for (size_t i = 1; i < n; ++i)
    {
        const double h_ratio = results[i-1].h / results[i].h;
        const double log_h = std::log(h_ratio);

        if (results[i-1].ux_L2 > 1e-15 && results[i].ux_L2 > 1e-15)
            ux_L2_rate[i] = std::log(results[i-1].ux_L2 / results[i].ux_L2) / log_h;
        if (results[i-1].ux_H1 > 1e-15 && results[i].ux_H1 > 1e-15)
            ux_H1_rate[i] = std::log(results[i-1].ux_H1 / results[i].ux_H1) / log_h;
        if (results[i-1].p_L2 > 1e-15 && results[i].p_L2 > 1e-15)
            p_L2_rate[i] = std::log(results[i-1].p_L2 / results[i].p_L2) / log_h;
    }
}

void NSMagMMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: NS_MAGNETIZATION\n";
    std::cout << "========================================\n";
    std::cout << "  μ₀ = " << mu_0 << ", ν = " << nu << "\n";
    std::cout << "  Testing magnetic body force: F_mag = μ₀(M·∇)H\n\n";

    std::cout << std::left
              << std::setw(6) << "Ref"
              << std::setw(12) << "h"
              << std::setw(12) << "ux_L2" << std::setw(8) << "rate"
              << std::setw(12) << "ux_H1" << std::setw(8) << "rate"
              << std::setw(12) << "p_L2" << std::setw(8) << "rate"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        std::cout << std::left << std::setw(6) << results[i].refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << results[i].h
                  << std::setw(12) << results[i].ux_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << ux_L2_rate[i]
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << results[i].ux_H1
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << ux_H1_rate[i]
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << results[i].p_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << p_L2_rate[i]
                  << "\n";
    }

    std::cout << "========================================\n";
    if (passes())
        std::cout << "[PASS] Convergence rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some rates below expected!\n";
}

bool NSMagMMSConvergenceResult::passes(double tol) const
{
    if (results.size() < 2)
        return false;

    const size_t last = results.size() - 1;
    const double min_vel_L2 = std::min(expected_vel_L2_rate, 2.0) - tol;
    const double min_vel_H1 = expected_vel_H1_rate - tol;
    const double min_p_L2 = expected_p_L2_rate - tol;

    bool pass = true;
    if (ux_L2_rate[last] < min_vel_L2) pass = false;
    if (ux_H1_rate[last] < min_vel_H1) pass = false;
    if (p_L2_rate[last] < min_p_L2) pass = false;

    return pass;
}

// ============================================================================
// Main test function - NS with magnetic force
// ============================================================================
NSMagMMSConvergenceResult run_ns_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps)
{
    NSMagMMSConvergenceResult result;
    result.fe_degree_velocity = params.fe.degree_velocity;
    result.fe_degree_pressure = params.fe.degree_pressure;
    result.n_time_steps = n_time_steps;
    result.nu = 1.0;  // Constant viscosity for MMS
    result.mu_0 = params.physics.mu_0;

    result.expected_vel_L2_rate = params.fe.degree_velocity + 1;
    result.expected_vel_H1_rate = params.fe.degree_velocity;
    result.expected_p_L2_rate = params.fe.degree_pressure + 1;

    const double t_init = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    result.dt = dt;

    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.enable_magnetic = true;
    mms_params.time.dt = dt;
    mms_params.physics.nu_water = 1.0;
    mms_params.physics.nu_ferro = 1.0;

    const double L_y = params.domain.y_max - params.domain.y_min;

    std::cout << "\n[NS_MAGNETIZATION] Running convergence study...\n";
    std::cout << "  t in [" << t_init << ", " << t_final << "], dt = " << dt << "\n";
    std::cout << "  ν = " << result.nu << ", μ₀ = " << result.mu_0 << "\n";
    std::cout << "  Testing magnetic body force: F_mag = μ₀(M·∇)H\n";
    std::cout << "  Using PRODUCTION assembler with magnetic force\n";
    std::cout << "  Using PRODUCTION Schur solver\n\n";

    for (unsigned int ref : refinements)
    {
        std::cout << "  Refinement " << ref << "... " << std::flush;
        auto total_start = std::chrono::high_resolution_clock::now();

        NSMagMMSResult res;
        res.refinement = ref;

        // ====================================================================
        // STEP 1: Create mesh
        // ====================================================================
        Triangulation<dim> triangulation;

        Point<dim> p1(mms_params.domain.x_min, mms_params.domain.y_min);
        Point<dim> p2(mms_params.domain.x_max, mms_params.domain.y_max);

        std::vector<unsigned int> subdivisions(dim);
        subdivisions[0] = mms_params.domain.initial_cells_x;
        subdivisions[1] = mms_params.domain.initial_cells_y;

        GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

        for (auto& face : triangulation.active_face_iterators())
        {
            if (!face->at_boundary()) continue;
            const auto center = face->center();
            const double tol = 1e-10;

            if (std::abs(center[1] - mms_params.domain.y_min) < tol)
                face->set_boundary_id(0);
            else if (std::abs(center[0] - mms_params.domain.x_max) < tol)
                face->set_boundary_id(1);
            else if (std::abs(center[1] - mms_params.domain.y_max) < tol)
                face->set_boundary_id(2);
            else if (std::abs(center[0] - mms_params.domain.x_min) < tol)
                face->set_boundary_id(3);
        }

        triangulation.refine_global(ref);
        res.h = GridTools::minimal_cell_diameter(triangulation);

        // ====================================================================
        // STEP 2: Create FE and DoF handlers
        // ====================================================================
        FE_Q<dim> fe_Q2(mms_params.fe.degree_velocity);
        FE_Q<dim> fe_Q1(mms_params.fe.degree_pressure);
        FE_DGQ<dim> fe_DG(mms_params.fe.degree_magnetization);

        // NS DoFs
        DoFHandler<dim> ux_dof_handler(triangulation);
        DoFHandler<dim> uy_dof_handler(triangulation);
        DoFHandler<dim> p_dof_handler(triangulation);

        ux_dof_handler.distribute_dofs(fe_Q2);
        uy_dof_handler.distribute_dofs(fe_Q2);
        p_dof_handler.distribute_dofs(fe_Q1);

        // CH DoFs (needed for assembler interface)
        DoFHandler<dim> theta_dof_handler(triangulation);
        DoFHandler<dim> psi_dof_handler(triangulation);
        theta_dof_handler.distribute_dofs(fe_Q2);
        psi_dof_handler.distribute_dofs(fe_Q2);

        // Magnetic DoFs
        DoFHandler<dim> phi_dof_handler(triangulation);
        DoFHandler<dim> mx_dof_handler(triangulation);

        phi_dof_handler.distribute_dofs(fe_Q2);
        mx_dof_handler.distribute_dofs(fe_DG);

        const unsigned int n_ux = ux_dof_handler.n_dofs();
        const unsigned int n_uy = uy_dof_handler.n_dofs();
        const unsigned int n_p = p_dof_handler.n_dofs();
        const unsigned int n_phi = phi_dof_handler.n_dofs();
        const unsigned int n_m = mx_dof_handler.n_dofs();

        res.n_dofs = n_ux + n_uy + n_p;

        // ====================================================================
        // STEP 3: Setup constraints
        // ====================================================================
        AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;

        DoFTools::make_hanging_node_constraints(ux_dof_handler, ux_constraints);
        DoFTools::make_hanging_node_constraints(uy_dof_handler, uy_constraints);
        DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);

        for (unsigned int bid = 0; bid <= 3; ++bid)
        {
            VectorTools::interpolate_boundary_values(
                ux_dof_handler, bid,
                Functions::ZeroFunction<dim>(), ux_constraints);
            VectorTools::interpolate_boundary_values(
                uy_dof_handler, bid,
                Functions::ZeroFunction<dim>(), uy_constraints);
        }

        if (!p_constraints.is_constrained(0))
        {
            p_constraints.add_line(0);
            p_constraints.set_inhomogeneity(0, 0.0);
        }

        ux_constraints.close();
        uy_constraints.close();
        p_constraints.close();

        // ====================================================================
        // STEP 4: PRODUCTION setup - setup_ns_coupled_system()
        // ====================================================================
        std::vector<types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
        AffineConstraints<double> ns_constraints;
        SparsityPattern ns_sparsity;

        setup_ns_coupled_system<dim>(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_constraints, uy_constraints, p_constraints,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_constraints, ns_sparsity, false);

        // Pressure mass matrix for Schur
        SparsityPattern p_mass_sparsity;
        SparseMatrix<double> pressure_mass_matrix;
        assemble_pressure_mass_matrix<dim>(
            p_dof_handler, p_constraints,
            p_mass_sparsity, pressure_mass_matrix);

        // ====================================================================
        // STEP 5: Allocate system
        // ====================================================================
        SparseMatrix<double> ns_matrix(ns_sparsity);
        Vector<double> ns_rhs(n_ux + n_uy + n_p);
        Vector<double> ns_solution(n_ux + n_uy + n_p);

        Vector<double> ux_solution(n_ux), uy_solution(n_uy), p_solution(n_p);
        Vector<double> ux_old(n_ux), uy_old(n_uy);

        // CH fields (dummy - set to constant)
        Vector<double> theta_solution(theta_dof_handler.n_dofs());
        Vector<double> psi_solution(psi_dof_handler.n_dofs());
        theta_solution = 1.0;  // Constant theta = 1 (ferrofluid)
        psi_solution = 0.0;

        // Magnetic fields - PRESCRIBED
        Vector<double> phi_solution(n_phi);
        Vector<double> mx_solution(n_m);
        Vector<double> my_solution(n_m);

        // Initialize magnetic fields with prescribed functions
        NSMagPrescribedPhi<dim> prescribed_phi(L_y);
        NSMagPrescribedMx<dim> prescribed_mx(L_y);
        NSMagPrescribedMy<dim> prescribed_my(L_y);

        VectorTools::interpolate(phi_dof_handler, prescribed_phi, phi_solution);
        VectorTools::interpolate(mx_dof_handler, prescribed_mx, mx_solution);
        VectorTools::interpolate(mx_dof_handler, prescribed_my, my_solution);

        // Schur preconditioner
        std::unique_ptr<BlockSchurPreconditioner> schur_preconditioner;

        // ====================================================================
        // STEP 6: Initialize NS with exact solution at t_init
        // ====================================================================
        NSExactVelocityX<dim> exact_ux_init(t_init, L_y);
        NSExactVelocityY<dim> exact_uy_init(t_init, L_y);

        VectorTools::interpolate(ux_dof_handler, exact_ux_init, ux_old);
        VectorTools::interpolate(uy_dof_handler, exact_uy_init, uy_old);

        // ====================================================================
        // STEP 7: Time stepping loop
        // ====================================================================
        double current_time = t_init;
        unsigned int total_iterations = 0;

        for (unsigned int step = 0; step < n_time_steps; ++step)
        {
            current_time += dt;

            ns_matrix = 0;
            ns_rhs = 0;

            // PRODUCTION ASSEMBLER WITH MAGNETIC FORCE
            // Pass phi, mx, my DoF handlers and solutions
            assemble_ns_system<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                theta_dof_handler, psi_dof_handler,
                &phi_dof_handler, &mx_dof_handler,  // Magnetic DoF handlers
                ux_old, uy_old,
                theta_solution, psi_solution,
                &phi_solution, &mx_solution, &my_solution,  // Magnetic fields
                mms_params,
                dt, current_time,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_constraints,
                ns_matrix, ns_rhs);

            // PRODUCTION SCHUR SOLVE
            ns_solution = 0;
            SolverInfo info;

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

            info = solve_ns_system_schur(
                ns_matrix, ns_rhs, ns_solution, ns_constraints,
                *schur_preconditioner,
                mms_params.solvers.ns.max_iterations,
                mms_params.solvers.ns.rel_tolerance,
                false);

            ns_constraints.distribute(ns_solution);
            total_iterations += info.iterations;

            // Extract solutions
            for (unsigned int i = 0; i < n_ux; ++i)
                ux_solution[i] = ns_solution[ux_to_ns_map[i]];
            for (unsigned int i = 0; i < n_uy; ++i)
                uy_solution[i] = ns_solution[uy_to_ns_map[i]];
            for (unsigned int i = 0; i < n_p; ++i)
                p_solution[i] = ns_solution[p_to_ns_map[i]];

            ux_old = ux_solution;
            uy_old = uy_solution;
        }

        res.solver_iterations = total_iterations;

        // ====================================================================
        // STEP 8: Compute errors
        // ====================================================================
        NSMMSError errors = compute_ns_mms_error(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_solution, uy_solution, p_solution,
            current_time, L_y);

        res.ux_L2 = errors.ux_L2;
        res.ux_H1 = errors.ux_H1;
        res.uy_L2 = errors.uy_L2;
        res.uy_H1 = errors.uy_H1;
        res.p_L2 = errors.p_L2;
        res.div_U_L2 = errors.div_U_L2;

        auto total_end = std::chrono::high_resolution_clock::now();
        res.total_time = std::chrono::duration<double>(total_end - total_start).count();

        result.results.push_back(res);

        std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << res.ux_L2
                  << ", p_L2=" << res.p_L2
                  << ", iters=" << res.solver_iterations
                  << ", time=" << std::fixed << std::setprecision(1) << res.total_time << "s\n";
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// Wrapper for mms_verification.cc integration
// ============================================================================
MMSConvergenceResult run_ns_magnetization_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps)
{
    NSMagMMSConvergenceResult ns_mag_result = run_ns_magnetization_mms(
        refinements, params, n_time_steps);

    // Convert to generic MMSConvergenceResult
    MMSConvergenceResult result;
    result.level = MMSLevel::NS_MAGNETIZATION;
    result.fe_degree = ns_mag_result.fe_degree_velocity;
    result.n_time_steps = ns_mag_result.n_time_steps;
    result.dt = ns_mag_result.dt;
    result.expected_L2_rate = ns_mag_result.expected_vel_L2_rate;
    result.expected_H1_rate = ns_mag_result.expected_vel_H1_rate;

    for (const auto& r : ns_mag_result.results)
    {
        result.refinements.push_back(r.refinement);
        result.h_values.push_back(r.h);
        result.ux_L2.push_back(r.ux_L2);
        result.ux_H1.push_back(r.ux_H1);
        result.uy_L2.push_back(r.uy_L2);
        result.uy_H1.push_back(r.uy_H1);
        result.p_L2.push_back(r.p_L2);
        result.div_u_L2.push_back(r.div_U_L2);
        result.n_dofs.push_back(r.n_dofs);
        result.wall_times.push_back(r.total_time);
    }

    result.compute_rates();
    return result;
}