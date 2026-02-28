// ============================================================================
// mms/coupled/mag_ch_mms_test.cc - Magnetization + CH Coupled MMS Test
//
// Tests χ(θ) coupling: CH → θ → χ(θ) → Magnetization
//
// Algorithm:
//   1. Solve CH standalone (with MMS source, U=0)
//   2. Use θ from CH solve in magnetization equation
//   3. Magnetization uses exact H = -∇φ* (from poisson_mms.h)
//   4. This isolates the χ(θ) interpolation across FE spaces
//
// Production code paths:
//   - assemble_ch_system() with enable_mms=true
//   - solve_ch_system()
//   - MagnetizationAssembler with enable_mms=true (uses θ for χ(θ))
//   - MagnetizationSolver (MUMPS direct)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"

// MMS exact solutions
#include "mms/ch/ch_mms.h"
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

// Production setup
#include "setup/magnetization_setup.h"

// Production assembly
#include "assembly/ch_assembler.h"
#include "assembly/magnetization_assembler.h"

// Production solvers
#include "solvers/ch_solver.h"
#include "solvers/magnetization_solver.h"

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
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>

constexpr int dim = 2;

// ============================================================================
// Single refinement test
// ============================================================================
static CoupledMMSResult run_mag_ch_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;

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
    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y
    };
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    // Set boundary IDs: 0=bottom, 1=right, 2=top, 3=left
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        for (const auto& face : cell->face_iterators())
        {
            if (!face->at_boundary()) continue;
            const auto center = face->center();
            if (std::abs(center[1] - params.domain.y_min) < 1e-10) face->set_boundary_id(0);
            else if (std::abs(center[0] - params.domain.x_max) < 1e-10) face->set_boundary_id(1);
            else if (std::abs(center[1] - params.domain.y_max) < 1e-10) face->set_boundary_id(2);
            else if (std::abs(center[0] - params.domain.x_min) < 1e-10) face->set_boundary_id(3);
        }
    }
    triangulation.refine_global(refinement);

    // Compute h
    double local_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h = std::min(local_h, cell->diameter());
    MPI_Allreduce(&local_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // ========================================================================
    // Setup CH DoFs (Q2)
    // ========================================================================
    dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);
    dealii::DoFHandler<dim> theta_dof(triangulation), psi_dof(triangulation);
    theta_dof.distribute_dofs(fe_phase);
    psi_dof.distribute_dofs(fe_phase);

    dealii::IndexSet theta_owned = theta_dof.locally_owned_dofs();
    dealii::IndexSet psi_owned = psi_dof.locally_owned_dofs();
    dealii::IndexSet theta_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof);
    dealii::IndexSet psi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof);

    const unsigned int n_theta = theta_dof.n_dofs();
    const unsigned int n_psi = psi_dof.n_dofs();
    const unsigned int n_ch = n_theta + n_psi;

    // ========================================================================
    // Setup Magnetization DoFs (DG-Q1)
    // ========================================================================
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);
    dealii::DoFHandler<dim> M_dof(triangulation);
    M_dof.distribute_dofs(fe_M);

    dealii::IndexSet M_owned = M_dof.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof);

    // Dummy velocity DoFs (U=0 for this test)
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::DoFHandler<dim> ux_dof(triangulation), uy_dof(triangulation);
    ux_dof.distribute_dofs(fe_vel);
    uy_dof.distribute_dofs(fe_vel);

    dealii::IndexSet ux_owned = ux_dof.locally_owned_dofs();
    dealii::IndexSet ux_relevant = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof);

    // Dummy Poisson DoFs - we use exact phi
    dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);
    dealii::DoFHandler<dim> phi_dof(triangulation);
    phi_dof.distribute_dofs(fe_phi);

    dealii::IndexSet phi_owned = phi_dof.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof);

    result.n_dofs = n_ch + 2 * M_dof.n_dofs();

    // ========================================================================
    // Setup CH coupled system
    // ========================================================================
    dealii::IndexSet ch_owned(n_ch), ch_relevant(n_ch);
    for (auto it = theta_owned.begin(); it != theta_owned.end(); ++it)
        ch_owned.add_index(*it);
    for (auto it = psi_owned.begin(); it != psi_owned.end(); ++it)
        ch_owned.add_index(n_theta + *it);
    for (auto it = theta_relevant.begin(); it != theta_relevant.end(); ++it)
        ch_relevant.add_index(*it);
    for (auto it = psi_relevant.begin(); it != psi_relevant.end(); ++it)
        ch_relevant.add_index(n_theta + *it);
    ch_owned.compress();
    ch_relevant.compress();

    std::vector<dealii::types::global_dof_index> theta_to_ch(n_theta), psi_to_ch(n_psi);
    for (unsigned int i = 0; i < n_theta; ++i) theta_to_ch[i] = i;
    for (unsigned int i = 0; i < n_psi; ++i) psi_to_ch[i] = n_theta + i;

    // CH constraints and BCs
    dealii::AffineConstraints<double> theta_constraints, psi_constraints, ch_constraints;
    theta_constraints.reinit(theta_owned, theta_relevant);
    psi_constraints.reinit(psi_owned, psi_relevant);

    CHMMSBoundaryTheta<dim> theta_bc(L_y);
    CHMMSBoundaryPsi<dim> psi_bc(L_y);
    theta_bc.set_time(t_start);
    psi_bc.set_time(t_start);

    for (unsigned int bid = 0; bid < 4; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(theta_dof, bid, theta_bc, theta_constraints);
        dealii::VectorTools::interpolate_boundary_values(psi_dof, bid, psi_bc, psi_constraints);
    }
    theta_constraints.close();
    psi_constraints.close();

    // CH sparsity pattern
    dealii::TrilinosWrappers::SparseMatrix ch_matrix;
    {
        dealii::TrilinosWrappers::SparsityPattern ch_sparsity;
        ch_sparsity.reinit(ch_owned, ch_owned, ch_relevant, mpi_communicator);

        std::vector<dealii::types::global_dof_index> theta_dofs(fe_phase.n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> psi_dofs(fe_phase.n_dofs_per_cell());

        for (const auto& cell : theta_dof.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
                &triangulation, cell->level(), cell->index(), &psi_dof);
            cell->get_dof_indices(theta_dofs);
            psi_cell->get_dof_indices(psi_dofs);

            for (unsigned int i = 0; i < fe_phase.n_dofs_per_cell(); ++i)
                for (unsigned int j = 0; j < fe_phase.n_dofs_per_cell(); ++j)
                {
                    ch_sparsity.add(theta_to_ch[theta_dofs[i]], theta_to_ch[theta_dofs[j]]);
                    ch_sparsity.add(theta_to_ch[theta_dofs[i]], psi_to_ch[psi_dofs[j]]);
                    ch_sparsity.add(psi_to_ch[psi_dofs[i]], theta_to_ch[theta_dofs[j]]);
                    ch_sparsity.add(psi_to_ch[psi_dofs[i]], psi_to_ch[psi_dofs[j]]);
                }
        }
        ch_sparsity.compress();
        ch_matrix.reinit(ch_sparsity);
    }
    dealii::TrilinosWrappers::MPI::Vector ch_rhs(ch_owned, mpi_communicator);

    // ========================================================================
    // Setup Magnetization system
    // ========================================================================
    dealii::AffineConstraints<double> M_constraints;
    M_constraints.close();

    dealii::TrilinosWrappers::SparseMatrix M_matrix;
    setup_magnetization_sparsity<dim>(M_dof, M_owned, M_relevant, M_matrix, mpi_communicator, pcout);

    dealii::TrilinosWrappers::MPI::Vector Mx_rhs(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_rhs(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_solution(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_solution(M_owned, mpi_communicator);

    // ========================================================================
    // Solution vectors
    // ========================================================================

    // CH
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_rel(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_old(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_vec(psi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_rel(psi_owned, psi_relevant, mpi_communicator);

    // Dummy velocity (U = 0)
    dealii::TrilinosWrappers::MPI::Vector ux_zero(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_zero(ux_owned, ux_relevant, mpi_communicator);
    ux_zero = 0;
    uy_zero = 0;

    // Exact phi (interpolated each time step)
    dealii::TrilinosWrappers::MPI::Vector phi_vec(phi_owned, phi_relevant, mpi_communicator);

    // Magnetization
    dealii::TrilinosWrappers::MPI::Vector Mx_vec(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_vec(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_communicator);

    // ========================================================================
    // Initialize from exact solutions at t_start
    // ========================================================================
    double current_time = t_start;

    // CH IC
    CHMMSInitialTheta<dim> theta_ic(t_start, L_y);
    CHMMSInitialPsi<dim> psi_ic(t_start, L_y);
    dealii::VectorTools::interpolate(theta_dof, theta_ic, theta_vec);
    dealii::VectorTools::interpolate(psi_dof, psi_ic, psi_vec);
    theta_rel = theta_vec;
    psi_rel = psi_vec;
    theta_old = theta_vec;

    // Exact phi at t_start
    PoissonExactSolution<dim> phi_exact_fn(t_start, L_y);
    {
        dealii::TrilinosWrappers::MPI::Vector phi_tmp(phi_owned, mpi_communicator);
        dealii::VectorTools::interpolate(phi_dof, phi_exact_fn, phi_tmp);
        phi_vec = phi_tmp;
    }

    // Magnetization IC (L2 projection for DG)
    MagExactMx<dim> Mx_ic(t_start, L_y);
    MagExactMy<dim> My_ic(t_start, L_y);
    dealii::VectorTools::interpolate(M_dof, Mx_ic, Mx_solution);
    dealii::VectorTools::interpolate(M_dof, My_ic, My_solution);
    Mx_vec = Mx_solution;
    My_vec = My_solution;
    Mx_old = Mx_solution;
    My_old = My_solution;

    // ========================================================================
    // Create assembler and solver
    // ========================================================================
    // Note: MagnetizationAssembler takes theta_dof for chi(theta) evaluation
    std::unique_ptr<MagnetizationAssembler<dim>> mag_assembler =
        std::make_unique<MagnetizationAssembler<dim>>(
            mms_params, M_dof, ux_dof, phi_dof, theta_dof, mpi_communicator);

    std::unique_ptr<MagnetizationSolver<dim>> mag_solver =
        std::make_unique<MagnetizationSolver<dim>>(
            mms_params.solvers.magnetization, M_owned, mpi_communicator);

    // ========================================================================
    // Time stepping
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ====================================================================
        // Step 1: Solve CH (standalone, U=0)
        // ====================================================================

        // Update CH boundary conditions
        theta_constraints.clear();
        theta_constraints.reinit(theta_owned, theta_relevant);
        psi_constraints.clear();
        psi_constraints.reinit(psi_owned, psi_relevant);

        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);

        for (unsigned int bid = 0; bid < 4; ++bid)
        {
            dealii::VectorTools::interpolate_boundary_values(
                theta_dof, bid, theta_bc, theta_constraints);
            dealii::VectorTools::interpolate_boundary_values(
                psi_dof, bid, psi_bc, psi_constraints);
        }
        theta_constraints.close();
        psi_constraints.close();

        // Rebuild combined constraints
        ch_constraints.clear();
        ch_constraints.reinit(ch_owned, ch_relevant);
        for (auto it = theta_relevant.begin(); it != theta_relevant.end(); ++it)
            if (theta_constraints.is_constrained(*it))
            {
                ch_constraints.add_line(theta_to_ch[*it]);
                ch_constraints.set_inhomogeneity(theta_to_ch[*it],
                    theta_constraints.get_inhomogeneity(*it));
            }
        for (auto it = psi_relevant.begin(); it != psi_relevant.end(); ++it)
            if (psi_constraints.is_constrained(*it))
            {
                ch_constraints.add_line(psi_to_ch[*it]);
                ch_constraints.set_inhomogeneity(psi_to_ch[*it],
                    psi_constraints.get_inhomogeneity(*it));
            }
        ch_constraints.close();

        theta_old = theta_vec;

        // Assemble and solve CH
        ch_matrix = 0;
        ch_rhs = 0;
        assemble_ch_system<dim>(
            theta_dof, psi_dof, theta_old,
            ux_dof, uy_dof, ux_zero, uy_zero,  // U=0
            mms_params, dt, current_time,
            theta_to_ch, psi_to_ch,
            ch_constraints, ch_matrix, ch_rhs);

        solve_ch_system(
            ch_matrix, ch_rhs, ch_constraints,
            ch_owned, ch_relevant,
            theta_owned, psi_owned,
            theta_to_ch, psi_to_ch,
            theta_vec, psi_vec,
            mms_params.solvers.ch, mpi_communicator, false);

        theta_rel = theta_vec;
        psi_rel = psi_vec;

        // ====================================================================
        // Step 2: Update exact phi for current time
        // ====================================================================
        phi_exact_fn.set_time(current_time);
        {
            dealii::TrilinosWrappers::MPI::Vector phi_tmp(phi_owned, mpi_communicator);
            dealii::VectorTools::interpolate(phi_dof, phi_exact_fn, phi_tmp);
            phi_vec = phi_tmp;
        }

        // ====================================================================
        // Step 3: Solve Magnetization with theta from CH and exact H
        // ====================================================================
        Mx_old = Mx_vec;
        My_old = My_vec;

        M_matrix = 0;
        Mx_rhs = 0;
        My_rhs = 0;

        // The assembler uses theta_rel for chi(theta) evaluation
        mag_assembler->assemble(
            M_matrix, Mx_rhs, My_rhs,
            ux_zero, uy_zero,   // U = 0 (no advection)
            phi_vec,            // Exact phi (for H = h_a - grad phi)
            theta_rel,          // theta from CH solve (THIS IS THE COUPLING!)
            Mx_old, My_old,     // M from previous time step
            dt, current_time);

        mag_solver->initialize(M_matrix);
        mag_solver->solve(Mx_solution, Mx_rhs);
        mag_solver->solve(My_solution, My_rhs);

        Mx_vec = Mx_solution;
        My_vec = My_solution;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================

    // CH errors
    {
        CHExactTheta<dim> exact_theta(L_y);
        exact_theta.set_time(current_time);
        dealii::QGauss<dim> quad(fe_phase.degree + 2);
        dealii::Vector<double> cell_err(triangulation.n_active_cells());

        dealii::VectorTools::integrate_difference(
            theta_dof, theta_rel, exact_theta, cell_err, quad, dealii::VectorTools::L2_norm);
        double local_sq = cell_err.norm_sqr(), global_sq;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.theta_L2 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(
            theta_dof, theta_rel, exact_theta, cell_err, quad, dealii::VectorTools::H1_seminorm);
        local_sq = cell_err.norm_sqr();
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.theta_H1 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(
            theta_dof, theta_rel, exact_theta, cell_err, quad, dealii::VectorTools::Linfty_norm);
        double local_linf = cell_err.linfty_norm(), global_linf;
        MPI_Allreduce(&local_linf, &global_linf, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
        result.theta_Linf = global_linf;
    }

    // Magnetization errors
    {
        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            M_dof, Mx_vec, My_vec, current_time, L_y, mpi_communicator);

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
CoupledMMSConvergenceResult run_mag_ch_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::MAG_CH;
    result.expected_L2_rate = params.fe.degree_phase + 1;  // Q2 -> 3
    result.expected_H1_rate = params.fe.degree_phase;      // Q2 -> 2
    result.expected_DG_rate = 2.0;                         // DG-Q1 -> 2

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n========================================\n";
        std::cout << "[MAG_CH] Magnetization + CH Coupled MMS Test\n";
        std::cout << "========================================\n";
        std::cout << "  Tests: chi(theta) coupling from CH to Magnetization\n";
        std::cout << "  Algorithm:\n";
        std::cout << "    1. Solve CH standalone (U=0)\n";
        std::cout << "    2. Feed theta into Magnetization with exact H\n";
        std::cout << "  MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Expected rates:\n";
        std::cout << "    theta: L2=" << result.expected_L2_rate << ", H1=" << result.expected_H1_rate << "\n";
        std::cout << "    M: L2=" << result.expected_DG_rate << " (DG-Q1)\n";
        std::cout << "========================================\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Ref " << ref << "... " << std::flush;

        CoupledMMSResult r = run_mag_ch_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "theta_L2=" << std::scientific << std::setprecision(2) << r.theta_L2
                      << ", M_L2=" << r.M_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}
