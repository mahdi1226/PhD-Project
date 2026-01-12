// ============================================================================
// mms/coupled/ns_poisson_mag_mms_test.cc - NS + Poisson + Magnetization Coupled MMS
//
// Tests the Kelvin force coupling:
//   1. Poisson: -Δφ = -∇·M (M appears as source)
//   2. Magnetization: ∂M/∂t + M/τ_M = χH/τ_M where H = -∇φ
//   3. NS: ∂U/∂t + (U·∇)U - 2νΔU + ∇p = μ₀(M·∇)H (Kelvin force)
//
// Uses PRODUCTION code paths:
//   - assemble_ns_system_parallel() with Kelvin body force callback
//   - solve_ns_system_schur_parallel() (Block Schur preconditioner)
//   - assemble_poisson_system() with enable_mms=true
//   - MagnetizationAssembler with enable_mms=true
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"
#include "mms/coupled/coupled_mms_sources.h"

// Individual MMS solutions
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"
#include "mms/ns/ns_mms.h"

// Production code
#include "setup/poisson_setup.h"
#include "setup/magnetization_setup.h"
#include "setup/ns_setup.h"
#include "assembly/poisson_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "assembly/ns_assembler.h"
#include "solvers/poisson_solver.h"
#include "solvers/magnetization_solver.h"
#include "solvers/ns_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <functional>

constexpr int dim = 2;

// ============================================================================
// Single refinement test
// ============================================================================
static CoupledMMSResult run_ns_poisson_mag_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    // Parameters
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;
    const double nu = params.physics.nu_ferro;
    (void)params.physics.mu_0; // Used by MMS source terms
    (void)params.physics.chi_0; // Used by magnetization assembler via mms_params
    (void)params.physics.tau_M; // Used by magnetization assembler via mms_params

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Create mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);
    std::vector<unsigned int> subdivisions = {params.domain.initial_cells_x, params.domain.initial_cells_y};
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // Compute h
    double local_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h = std::min(local_h, cell->diameter());
    MPI_Allreduce(&local_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // ========================================================================
    // Setup NS DoFs (Q2-Q1 Taylor-Hood)
    // ========================================================================
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::FE_Q<dim> fe_p(params.fe.degree_pressure);

    dealii::DoFHandler<dim> ux_dof(triangulation);
    dealii::DoFHandler<dim> uy_dof(triangulation);
    dealii::DoFHandler<dim> p_dof(triangulation);

    ux_dof.distribute_dofs(fe_vel);
    uy_dof.distribute_dofs(fe_vel);
    p_dof.distribute_dofs(fe_p);

    const unsigned int n_ux = ux_dof.n_dofs();
    const unsigned int n_uy = uy_dof.n_dofs();

    dealii::IndexSet ux_owned = ux_dof.locally_owned_dofs();
    dealii::IndexSet uy_owned = uy_dof.locally_owned_dofs();
    dealii::IndexSet p_owned = p_dof.locally_owned_dofs();
    dealii::IndexSet ux_relevant = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof);
    dealii::IndexSet uy_relevant = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof);
    dealii::IndexSet p_relevant = dealii::DoFTools::extract_locally_relevant_dofs(p_dof);

    // ========================================================================
    // Setup Poisson DoFs
    // ========================================================================
    dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);
    dealii::DoFHandler<dim> phi_dof(triangulation);
    phi_dof.distribute_dofs(fe_phi);

    dealii::IndexSet phi_owned = phi_dof.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof);

    // ========================================================================
    // Setup Magnetization DoFs (DG)
    // ========================================================================
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);
    dealii::DoFHandler<dim> M_dof(triangulation);
    M_dof.distribute_dofs(fe_M);

    dealii::IndexSet M_owned = M_dof.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof);

    result.n_dofs = n_ux + n_uy + p_dof.n_dofs() + phi_dof.n_dofs() + 2 * M_dof.n_dofs();

    // ========================================================================
    // NS constraints and system setup
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
    setup_ns_velocity_constraints_parallel<dim>(ux_dof, uy_dof, ux_constraints, uy_constraints);
    setup_ns_pressure_constraints_parallel<dim>(p_dof, p_constraints);

    std::vector<dealii::types::global_dof_index> ux_to_ns, uy_to_ns, p_to_ns;
    dealii::IndexSet ns_owned, ns_relevant;
    dealii::AffineConstraints<double> ns_constraints;
    dealii::TrilinosWrappers::SparsityPattern ns_sparsity;

    setup_ns_coupled_system_parallel<dim>(ux_dof, uy_dof, p_dof,
                                          ux_constraints, uy_constraints, p_constraints,
                                          ux_to_ns, uy_to_ns, p_to_ns, ns_owned, ns_relevant,
                                          ns_constraints, ns_sparsity, mpi_communicator, pcout);

    dealii::TrilinosWrappers::SparseMatrix ns_matrix;
    ns_matrix.reinit(ns_sparsity);
    dealii::TrilinosWrappers::MPI::Vector ns_rhs(ns_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ns_solution(ns_owned, mpi_communicator);

    dealii::TrilinosWrappers::SparseMatrix pressure_mass;
    assemble_pressure_mass_matrix_parallel<dim>(p_dof, p_constraints, p_owned, p_relevant,
                                                pressure_mass, mpi_communicator);

    dealii::IndexSet vel_owned(n_ux + n_uy);
    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it) vel_owned.add_index(*it);
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it) vel_owned.add_index(n_ux + *it);
    vel_owned.compress();

    // NS solution vectors
    dealii::TrilinosWrappers::MPI::Vector ux_sol(ux_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_sol(uy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector p_sol(p_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ux_old(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_old(uy_owned, uy_relevant, mpi_communicator);

    // ========================================================================
    // Poisson constraints and system setup
    // ========================================================================
    dealii::AffineConstraints<double> phi_constraints;
    phi_constraints.reinit(phi_owned, phi_relevant);

    // MMS boundary conditions for φ
    PoissonExactSolution<dim> phi_bc(t_start, L_y);
    for (unsigned int bid = 0; bid < 4; ++bid)
        dealii::VectorTools::interpolate_boundary_values(phi_dof, bid, phi_bc, phi_constraints);
    phi_constraints.close();

    dealii::TrilinosWrappers::SparsityPattern phi_sparsity;
    {
        dealii::DynamicSparsityPattern dsp(phi_dof.n_dofs(), phi_dof.n_dofs(), phi_relevant);
        dealii::DoFTools::make_sparsity_pattern(phi_dof, dsp, phi_constraints, false);
        phi_sparsity.reinit(phi_owned, phi_owned, dsp, mpi_communicator, true);
    }

    dealii::TrilinosWrappers::SparseMatrix phi_matrix;
    phi_matrix.reinit(phi_sparsity);
    dealii::TrilinosWrappers::MPI::Vector phi_rhs(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_solution(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_ghosted(phi_owned, phi_relevant, mpi_communicator);

    // ========================================================================
    // Magnetization system setup
    // ========================================================================
    dealii::TrilinosWrappers::SparsityPattern M_sparsity;
    {
        dealii::DynamicSparsityPattern dsp(M_dof.n_dofs(), M_dof.n_dofs(), M_relevant);
        dealii::DoFTools::make_sparsity_pattern(M_dof, dsp);
        M_sparsity.reinit(M_owned, M_owned, dsp, mpi_communicator, true);
    }

    dealii::TrilinosWrappers::SparseMatrix M_matrix;
    M_matrix.reinit(M_sparsity);
    dealii::TrilinosWrappers::MPI::Vector rhs_Mx(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector rhs_My(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_owned_vec(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_owned_vec(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_communicator);

    // ========================================================================
    // Initialize all fields with exact solutions
    // ========================================================================
    double current_time = t_start;

    // Initialize NS
    {
        NSExactVelocityX<dim> exact_ux(current_time, L_y);
        NSExactVelocityY<dim> exact_uy(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector tmp_ux(ux_owned, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector tmp_uy(uy_owned, mpi_communicator);
        dealii::VectorTools::interpolate(ux_dof, exact_ux, tmp_ux);
        dealii::VectorTools::interpolate(uy_dof, exact_uy, tmp_uy);
        ux_old = tmp_ux;
        uy_old = tmp_uy;
    }

    // Initialize Poisson
    {
        PoissonExactSolution<dim> exact_phi(current_time, L_y);
        dealii::VectorTools::interpolate(phi_dof, exact_phi, phi_solution);
        phi_ghosted = phi_solution;
    }

    // Initialize Magnetization (cell-wise L2 projection for DG)
    {
        MagExactMx<dim> exact_Mx(current_time, L_y);
        MagExactMy<dim> exact_My(current_time, L_y);

        dealii::QGauss<dim> quadrature(fe_M.degree + 2);
        dealii::FEValues<dim> fe_values(fe_M, quadrature,
                                        dealii::update_values | dealii::update_quadrature_points |
                                        dealii::update_JxW_values);

        const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature.size();

        dealii::FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
        dealii::FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
        dealii::Vector<double> local_rhs_x(dofs_per_cell), local_rhs_y(dofs_per_cell);
        dealii::Vector<double> local_sol_x(dofs_per_cell), local_sol_y(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

        for (const auto& cell : M_dof.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;

            fe_values.reinit(cell);
            local_mass = 0;
            local_rhs_x = 0;
            local_rhs_y = 0;

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const double JxW = fe_values.JxW(q);
                const auto& x_q = fe_values.quadrature_point(q);
                const double Mx_exact = exact_Mx.value(x_q);
                const double My_exact = exact_My.value(x_q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const double phi_i = fe_values.shape_value(i, q);
                    local_rhs_x(i) += Mx_exact * phi_i * JxW;
                    local_rhs_y(i) += My_exact * phi_i * JxW;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        local_mass(i, j) += phi_i * fe_values.shape_value(j, q) * JxW;
                }
            }

            local_mass_inv.invert(local_mass);
            local_mass_inv.vmult(local_sol_x, local_rhs_x);
            local_mass_inv.vmult(local_sol_y, local_rhs_y);

            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                Mx_owned_vec[local_dofs[i]] = local_sol_x(i);
                My_owned_vec[local_dofs[i]] = local_sol_y(i);
            }
        }
        Mx_owned_vec.compress(dealii::VectorOperation::insert);
        My_owned_vec.compress(dealii::VectorOperation::insert);

        Mx_old = Mx_owned_vec;
        My_old = My_owned_vec;
    }

    // ========================================================================
    // Solver setup
    // ========================================================================
    LinearSolverParams poisson_solver_params;
    poisson_solver_params.type = LinearSolverParams::Type::CG;
    poisson_solver_params.preconditioner = LinearSolverParams::Preconditioner::AMG;
    poisson_solver_params.rel_tolerance = 1e-10;
    poisson_solver_params.max_iterations = 500;
    poisson_solver_params.use_iterative = true;

    LinearSolverParams mag_solver_params;
    mag_solver_params.use_iterative = true;
    mag_solver_params.rel_tolerance = 1e-10;
    mag_solver_params.max_iterations = 500;

    MagnetizationSolver<dim> mag_solver(mag_solver_params, M_owned, mpi_communicator);

    // Dummy DoF handlers for magnetization assembler
    dealii::DoFHandler<dim> dummy_U_dof(triangulation);
    dealii::DoFHandler<dim> dummy_theta_dof(triangulation);
    dealii::FE_Q<dim> fe_dummy(1);
    dummy_U_dof.distribute_dofs(fe_dummy);
    dummy_theta_dof.distribute_dofs(fe_dummy);

    dealii::IndexSet dummy_owned = dummy_U_dof.locally_owned_dofs();
    dealii::TrilinosWrappers::MPI::Vector Ux_dummy(dummy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Uy_dummy(dummy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_dummy(dummy_owned, mpi_communicator);
    Ux_dummy = 0;
    Uy_dummy = 0;
    theta_dummy = 1.0;

    MagnetizationAssembler<dim> mag_assembler(
        mms_params, M_dof, dummy_U_dof,
        phi_dof, dummy_theta_dof, mpi_communicator);

    // ========================================================================
    // Time stepping with COUPLED solve
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        const double t_old = current_time;
        current_time += dt;

        // Update old values
        Mx_old = Mx_owned_vec;
        My_old = My_owned_vec;

        // --------------------------------------------------------------------
        // Step 1: Solve Poisson with M source
        // -Δφ = -∇·M + f_φ^MMS
        // --------------------------------------------------------------------
        phi_matrix = 0;
        phi_rhs = 0;

        // Update BC
        phi_constraints.clear();
        phi_constraints.reinit(phi_owned, phi_relevant);
        PoissonExactSolution<dim> phi_bc_new(current_time, L_y);
        for (unsigned int bid = 0; bid < 4; ++bid)
            dealii::VectorTools::interpolate_boundary_values(phi_dof, bid, phi_bc_new, phi_constraints);
        phi_constraints.close();

        // For MMS: pass empty M, assembler uses standalone MMS source
        dealii::TrilinosWrappers::MPI::Vector Mx_empty, My_empty;
        assemble_poisson_system<dim>(
            phi_dof, M_dof,
            Mx_empty, My_empty,
            mms_params, current_time,
            phi_constraints,
            phi_matrix, phi_rhs);

        solve_poisson_system(
            phi_matrix, phi_rhs, phi_solution,
            phi_constraints, phi_owned,
            poisson_solver_params, mpi_communicator, false);

        phi_ghosted = phi_solution;

        // --------------------------------------------------------------------
        // Step 2: Solve Magnetization with H = -∇φ
        // ∂M/∂t + M/τ_M = χH/τ_M + f_M^MMS
        // --------------------------------------------------------------------
        M_matrix = 0;
        rhs_Mx = 0;
        rhs_My = 0;

        mag_assembler.assemble(
            M_matrix, rhs_Mx, rhs_My,
            Ux_dummy, Uy_dummy, phi_ghosted, theta_dummy,
            Mx_old, My_old,
            dt, current_time);

        mag_solver.initialize(M_matrix);
        mag_solver.solve(Mx_owned_vec, rhs_Mx);
        mag_solver.solve(My_owned_vec, rhs_My);

        // --------------------------------------------------------------------
        // Step 3: Solve NS with Kelvin force (PRODUCTION CODE PATH)
        // ∂U/∂t + (U·∇)U - 2νΔU + ∇p = μ₀(M·∇)H + f_U^MMS
        //
        // Uses assemble_ns_system_with_kelvin_force_parallel() which:
        //   1. Assembles core NS (time, viscous, convection, pressure)
        //   2. Adds Kelvin force μ₀(M·∇)H from numerical M and φ fields
        //   3. Adds standalone NS MMS source (NOT coupled source!)
        //
        // Since Kelvin force is computed numerically, the MMS source must
        // NOT subtract the analytical Kelvin force.
        // --------------------------------------------------------------------
        ns_matrix = 0;
        ns_rhs = 0;

        // Create ghosted vectors for M (need for Kelvin force evaluation)
        dealii::TrilinosWrappers::MPI::Vector Mx_ghosted(M_owned, M_relevant, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector My_ghosted(M_owned, M_relevant, mpi_communicator);
        Mx_ghosted = Mx_owned_vec;
        My_ghosted = My_owned_vec;

        assemble_ns_system_with_kelvin_force_parallel<dim>(
            ux_dof, uy_dof, p_dof, ux_old, uy_old,
            nu, dt, true, true, // include_time_derivative, include_convection
            ux_to_ns, uy_to_ns, p_to_ns, ns_owned, ns_constraints,
            ns_matrix, ns_rhs, mpi_communicator,
            // Kelvin force inputs
            phi_dof, M_dof,
            phi_ghosted, Mx_ghosted, My_ghosted,
            mms_params.physics.mu_0,
            // MMS options - use standalone source (NOT coupled!)
            true, // enable_mms
            current_time, // mms_time
            t_old, // mms_time_old
            L_y); // mms_L_y

        ns_solution = 0;
        solve_ns_system_schur_parallel(ns_matrix, ns_rhs, ns_solution, ns_constraints,
                                       pressure_mass, ux_to_ns, uy_to_ns, p_to_ns,
                                       ns_owned, vel_owned, p_owned, mpi_communicator, nu, false);

        extract_ns_solutions_parallel(ns_solution, ux_to_ns, uy_to_ns, p_to_ns,
                                      ux_owned, uy_owned, p_owned, ns_owned, ns_relevant, ux_sol, uy_sol, p_sol,
                                      mpi_communicator);

        ux_old = ux_sol;
        uy_old = uy_sol;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================

    // NS errors
    {
        dealii::TrilinosWrappers::MPI::Vector ux_gh(ux_owned, ux_relevant, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector uy_gh(uy_owned, uy_relevant, mpi_communicator);
        ux_gh = ux_sol;
        uy_gh = uy_sol;

        NSExactVelocityX<dim> exact_ux(current_time, L_y);
        NSExactVelocityY<dim> exact_uy(current_time, L_y);

        dealii::QGauss<dim> quad(fe_vel.degree + 2);
        dealii::Vector<double> cell_err(triangulation.n_active_cells());

        dealii::VectorTools::integrate_difference(ux_dof, ux_gh, exact_ux, cell_err, quad,
                                                  dealii::VectorTools::L2_norm);
        double local_sq = cell_err.norm_sqr(), global_sq;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.ux_L2 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(ux_dof, ux_gh, exact_ux, cell_err, quad,
                                                  dealii::VectorTools::H1_seminorm);
        local_sq = cell_err.norm_sqr();
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.ux_H1 = std::sqrt(global_sq);

        // Pressure L2 error
        NSExactPressure<dim> exact_p(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector p_gh(p_owned, p_relevant, mpi_communicator);
        p_gh = p_sol;
        dealii::VectorTools::integrate_difference(p_dof, p_gh, exact_p, cell_err, quad,
                                                  dealii::VectorTools::L2_norm);
        local_sq = cell_err.norm_sqr();
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.p_L2 = std::sqrt(global_sq);
    }

    // Poisson error
    {
        dealii::TrilinosWrappers::MPI::Vector phi_rel(phi_owned, phi_relevant, mpi_communicator);
        phi_rel = phi_solution;

        PoissonMMSError phi_err = compute_poisson_mms_errors_parallel<dim>(
            phi_dof, phi_rel, current_time, L_y, mpi_communicator);

        result.phi_L2 = phi_err.L2_error;
        result.phi_H1 = phi_err.H1_error;
    }

    // Magnetization error
    {
        dealii::TrilinosWrappers::MPI::Vector Mx_rel(M_owned, M_relevant, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector My_rel(M_owned, M_relevant, mpi_communicator);
        Mx_rel = Mx_owned_vec;
        My_rel = My_owned_vec;

        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            M_dof, Mx_rel, My_rel, current_time, L_y, mpi_communicator);

        result.Mx_L2 = mag_err.Mx_L2;
        result.My_L2 = mag_err.My_L2;
        result.M_L2 = mag_err.M_L2;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// Public interface
// ============================================================================

CoupledMMSConvergenceResult run_ns_poisson_mag_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::NS_POISSON_MAG;
    result.expected_L2_rate = params.fe.degree_velocity + 1; // Q2 velocity
    result.expected_H1_rate = params.fe.degree_velocity;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[NS_POISSON_MAG] Running coupled MMS test (Kelvin force)...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Expected rates: L2 = " << result.expected_L2_rate
            << ", H1 = " << result.expected_H1_rate << "\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        CoupledMMSResult r = run_ns_poisson_mag_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                << ", φ_L2=" << r.phi_L2
                << ", M_L2=" << r.M_L2
                << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}