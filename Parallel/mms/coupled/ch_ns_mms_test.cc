// ============================================================================
// mms/coupled/ch_ns_mms_test.cc - CH + NS Coupled MMS Test
//
// Tests phase field advection by velocity:
//   CH equation: ∂θ/∂t + U·∇θ = γΔψ + f_θ
//
// The advection term U·∇θ couples NS velocity into the CH equation.
//
// Uses PRODUCTION code paths:
//   - assemble_ns_system_parallel() with enable_mms=true
//   - solve_ns_system_schur_parallel() (Block Schur preconditioner)
//   - assemble_ch_system()
//   - solve_ch_system()
//
// CRITICAL FIX: All CH MMS classes now receive L_y parameter for consistency
//               with the L_y-scaled NS velocity field.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"
#include "mms/coupled/coupled_mms_sources.h"
#include "mms/ch/ch_mms.h"
#include "mms/ns/ns_mms.h"

#include "assembly/ch_assembler.h"
#include "solvers/ch_solver.h"
#include "setup/ns_setup.h"
#include "assembly/ns_assembler.h"
#include "solvers/ns_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

constexpr int dim = 2;

// ============================================================================
// CH + NS coupled single refinement test
// ============================================================================
static CoupledMMSResult run_ch_ns_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;

    dealii::ConditionalOStream pcout(std::cout,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;
    const double nu = params.physics.nu_ferro;

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Create mesh
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);
    std::vector<unsigned int> subdivisions = {params.domain.initial_cells_x, params.domain.initial_cells_y};
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    // Boundary IDs
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

    // Setup NS DoFs
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::FE_Q<dim> fe_p(params.fe.degree_pressure);
    dealii::DoFHandler<dim> ux_dof(triangulation), uy_dof(triangulation), p_dof(triangulation);
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

    // Setup CH DoFs
    dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);
    dealii::DoFHandler<dim> theta_dof(triangulation), psi_dof(triangulation);
    theta_dof.distribute_dofs(fe_phase);
    psi_dof.distribute_dofs(fe_phase);

    const unsigned int n_theta = theta_dof.n_dofs();
    const unsigned int n_psi = psi_dof.n_dofs();
    const unsigned int n_ch = n_theta + n_psi;

    dealii::IndexSet theta_owned = theta_dof.locally_owned_dofs();
    dealii::IndexSet psi_owned = psi_dof.locally_owned_dofs();
    dealii::IndexSet theta_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof);
    dealii::IndexSet psi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof);

    // Combined CH IndexSets
    dealii::IndexSet ch_owned(n_ch), ch_relevant(n_ch);
    for (auto it = theta_owned.begin(); it != theta_owned.end(); ++it) ch_owned.add_index(*it);
    for (auto it = psi_owned.begin(); it != psi_owned.end(); ++it) ch_owned.add_index(n_theta + *it);
    for (auto it = theta_relevant.begin(); it != theta_relevant.end(); ++it) ch_relevant.add_index(*it);
    for (auto it = psi_relevant.begin(); it != psi_relevant.end(); ++it) ch_relevant.add_index(n_theta + *it);
    ch_owned.compress();
    ch_relevant.compress();

    std::vector<dealii::types::global_dof_index> theta_to_ch(n_theta), psi_to_ch(n_psi);
    for (unsigned int i = 0; i < n_theta; ++i) theta_to_ch[i] = i;
    for (unsigned int i = 0; i < n_psi; ++i) psi_to_ch[i] = n_theta + i;

    result.n_dofs = n_ch + n_ux + n_uy + p_dof.n_dofs();

    // Compute h
    double local_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned()) local_h = std::min(local_h, cell->diameter());
    MPI_Allreduce(&local_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // Setup NS system
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
    assemble_pressure_mass_matrix_parallel<dim>(p_dof, p_constraints, p_owned, p_relevant, pressure_mass, mpi_communicator);

    dealii::IndexSet vel_owned(n_ux + n_uy);
    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it) vel_owned.add_index(*it);
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it) vel_owned.add_index(n_ux + *it);
    vel_owned.compress();

    // Setup CH system
    dealii::AffineConstraints<double> theta_constraints, psi_constraints, ch_constraints;
    dealii::TrilinosWrappers::SparseMatrix ch_matrix;

    theta_constraints.reinit(theta_owned, theta_relevant);
    psi_constraints.reinit(psi_owned, psi_relevant);

    // CRITICAL: Pass L_y to boundary condition functions
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

    // Build CH sparsity
    {
        dealii::TrilinosWrappers::SparsityPattern ch_sparsity;
        ch_sparsity.reinit(ch_owned, ch_owned, ch_relevant, mpi_communicator);

        std::vector<dealii::types::global_dof_index> theta_dofs(fe_phase.n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> psi_dofs(fe_phase.n_dofs_per_cell());

        for (const auto& cell : theta_dof.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(&triangulation, cell->level(), cell->index(), &psi_dof);
            cell->get_dof_indices(theta_dofs);
            psi_cell->get_dof_indices(psi_dofs);

            for (unsigned int i = 0; i < fe_phase.n_dofs_per_cell(); ++i)
            {
                for (unsigned int j = 0; j < fe_phase.n_dofs_per_cell(); ++j)
                {
                    ch_sparsity.add(theta_to_ch[theta_dofs[i]], theta_to_ch[theta_dofs[j]]);
                    ch_sparsity.add(theta_to_ch[theta_dofs[i]], psi_to_ch[psi_dofs[j]]);
                    ch_sparsity.add(psi_to_ch[psi_dofs[i]], theta_to_ch[theta_dofs[j]]);
                    ch_sparsity.add(psi_to_ch[psi_dofs[i]], psi_to_ch[psi_dofs[j]]);
                }
            }
        }
        ch_sparsity.compress();
        ch_matrix.reinit(ch_sparsity);
    }

    dealii::TrilinosWrappers::MPI::Vector ch_rhs(ch_owned, mpi_communicator);

    // NS solution vectors
    dealii::TrilinosWrappers::MPI::Vector ux_sol(ux_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_sol(uy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector p_sol(p_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ux_old(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_old(uy_owned, uy_relevant, mpi_communicator);

    // CH solution vectors
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_rel(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_old(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_vec(psi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_rel(psi_owned, psi_relevant, mpi_communicator);

    // Initialize
    double current_time = t_start;
    {
        NSExactVelocityX<dim> exact_ux(current_time, L_y);
        NSExactVelocityY<dim> exact_uy(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector tmp_ux(ux_owned, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector tmp_uy(uy_owned, mpi_communicator);
        dealii::VectorTools::interpolate(ux_dof, exact_ux, tmp_ux);
        dealii::VectorTools::interpolate(uy_dof, exact_uy, tmp_uy);
        ux_old = tmp_ux;
        uy_old = tmp_uy;

        // CRITICAL: Pass L_y to initial condition functions
        CHMMSInitialTheta<dim> theta_ic(current_time, L_y);
        CHMMSInitialPsi<dim> psi_ic(current_time, L_y);
        dealii::VectorTools::interpolate(theta_dof, theta_ic, theta_vec);
        dealii::VectorTools::interpolate(psi_dof, psi_ic, psi_vec);
        theta_rel = theta_vec;
        psi_rel = psi_vec;
        theta_old = theta_vec;
    }

    // Time loop
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ====================================================================
        // Step 1: Solve NS using Block Schur preconditioner
        // ====================================================================
        ns_matrix = 0;
        ns_rhs = 0;
        assemble_ns_system_parallel<dim>(ux_dof, uy_dof, p_dof, ux_old, uy_old,
            nu, dt, true, true,  // include_time_derivative, include_convection
            ux_to_ns, uy_to_ns, p_to_ns, ns_owned, ns_constraints,
            ns_matrix, ns_rhs, mpi_communicator,
            true,                   // enable_mms
            current_time,           // mms_time
            current_time - dt,      // mms_time_old
            L_y);                   // mms_L_y

        ns_solution = 0;
        solve_ns_system_schur_parallel(ns_matrix, ns_rhs, ns_solution, ns_constraints,
            pressure_mass, ux_to_ns, uy_to_ns, p_to_ns,
            ns_owned, vel_owned, p_owned, mpi_communicator, nu, false);

        extract_ns_solutions_parallel(ns_solution, ux_to_ns, uy_to_ns, p_to_ns,
            ux_owned, uy_owned, p_owned, ns_owned, ns_relevant, ux_sol, uy_sol, p_sol, mpi_communicator);
        ux_old = ux_sol;
        uy_old = uy_sol;

        // ====================================================================
        // Step 2: Update CH boundary conditions
        // CRITICAL: Pass L_y to boundary condition functions
        // ====================================================================
        theta_constraints.clear();
        theta_constraints.reinit(theta_owned, theta_relevant);
        psi_constraints.clear();
        psi_constraints.reinit(psi_owned, psi_relevant);
        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);
        for (unsigned int bid = 0; bid < 4; ++bid)
        {
            dealii::VectorTools::interpolate_boundary_values(theta_dof, bid, theta_bc, theta_constraints);
            dealii::VectorTools::interpolate_boundary_values(psi_dof, bid, psi_bc, psi_constraints);
        }
        theta_constraints.close();
        psi_constraints.close();

        ch_constraints.clear();
        ch_constraints.reinit(ch_owned, ch_relevant);
        for (auto it = theta_relevant.begin(); it != theta_relevant.end(); ++it)
            if (theta_constraints.is_constrained(*it))
            {
                ch_constraints.add_line(theta_to_ch[*it]);
                ch_constraints.set_inhomogeneity(theta_to_ch[*it], theta_constraints.get_inhomogeneity(*it));
            }
        for (auto it = psi_relevant.begin(); it != psi_relevant.end(); ++it)
            if (psi_constraints.is_constrained(*it))
            {
                ch_constraints.add_line(psi_to_ch[*it]);
                ch_constraints.set_inhomogeneity(psi_to_ch[*it], psi_constraints.get_inhomogeneity(*it));
            }
        ch_constraints.close();

        theta_old = theta_vec;

        // ====================================================================
        // Step 3: Solve CH with advection using PRODUCTION assembler
        // (MMS sources with L_y are handled internally by assembler)
        // ====================================================================
        ch_matrix = 0;
        ch_rhs = 0;
        assemble_ch_system<dim>(theta_dof, psi_dof, theta_old,
            ux_dof, uy_dof,            // Velocity DoF handlers
            ux_old, uy_old,            // Velocity solutions (ghosted)
            mms_params, dt, current_time, theta_to_ch, psi_to_ch,
            ch_constraints, ch_matrix, ch_rhs);

        solve_ch_system(ch_matrix, ch_rhs, ch_constraints, ch_owned, theta_owned, psi_owned,
            theta_to_ch, psi_to_ch, theta_vec, psi_vec,
            mms_params.solvers.ch, mpi_communicator, false);

        theta_rel = theta_vec;
        psi_rel = psi_vec;
    }

    // Compute errors
    // CRITICAL: Pass L_y to exact solution functions
    CHExactTheta<dim> exact_theta(L_y);
    CHExactPsi<dim> exact_psi(L_y);
    exact_theta.set_time(current_time);
    exact_psi.set_time(current_time);
    dealii::QGauss<dim> quad(fe_phase.degree + 2);
    dealii::Vector<double> cell_err(triangulation.n_active_cells());

    dealii::VectorTools::integrate_difference(theta_dof, theta_rel, exact_theta, cell_err, quad, dealii::VectorTools::L2_norm);
    double local_sq = cell_err.norm_sqr(), global_sq;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.theta_L2 = std::sqrt(global_sq);

    dealii::VectorTools::integrate_difference(theta_dof, theta_rel, exact_theta, cell_err, quad, dealii::VectorTools::H1_seminorm);
    local_sq = cell_err.norm_sqr();
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.theta_H1 = std::sqrt(global_sq);

    NSExactVelocityX<dim> exact_ux(current_time, L_y);
    dealii::TrilinosWrappers::MPI::Vector ux_gh(ux_owned, ux_relevant, mpi_communicator);
    ux_gh = ux_sol;
    dealii::VectorTools::integrate_difference(ux_dof, ux_gh, exact_ux, cell_err, quad, dealii::VectorTools::L2_norm);
    local_sq = cell_err.norm_sqr();
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.ux_L2 = std::sqrt(global_sq);

    // ux_H1 error
    dealii::VectorTools::integrate_difference(ux_dof, ux_gh, exact_ux, cell_err, quad,
                                              dealii::VectorTools::H1_seminorm);
    local_sq = cell_err.norm_sqr();
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.ux_H1 = std::sqrt(global_sq);

    // Pressure error
    NSExactPressure<dim> exact_p(current_time, L_y);
    dealii::TrilinosWrappers::MPI::Vector p_gh(p_owned, p_relevant, mpi_communicator);
    p_gh = p_sol;
    dealii::VectorTools::integrate_difference(p_dof, p_gh, exact_p, cell_err, quad,
                                              dealii::VectorTools::L2_norm);
    local_sq = cell_err.norm_sqr();
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.p_L2 = std::sqrt(global_sq);

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// Public interface
CoupledMMSConvergenceResult run_ch_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::CH_NS;
    result.expected_L2_rate = params.fe.degree_phase + 1;
    result.expected_H1_rate = params.fe.degree_phase;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (rank == 0)
    {
        std::cout << "\n[CH_NS] Running coupled MMS test...\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
    }

    for (unsigned int ref : refinements)
    {
        if (rank == 0) std::cout << "  Ref " << ref << "... " << std::flush;
        CoupledMMSResult r = run_ch_ns_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);
        if (rank == 0)
            std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << r.theta_L2
                      << ", ux_L2=" << r.ux_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
    }

    result.compute_rates();
    return result;
}