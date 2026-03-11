// ============================================================================
// mms/coupled/magnetic_ns_mms_test.cc - Magnetic → NS Coupled MMS Test
//
// Tests Kelvin force coupling: Monolithic M+φ → Kelvin force → NS
//
// Algorithm:
//   1. Solve monolithic M+φ (θ=1, U=0)
//   2. Extract M and φ from monolithic solution to auxiliary DoFHandlers
//   3. Solve NS with Kelvin force μ₀[(M·∇)H + ½(∇·M)H]
//
// This is the CRITICAL coupling for Rosensweig instability!
//
// Production code paths:
//   - setup_magnetic_system(), MagneticAssembler, MagneticSolver (MUMPS)
//   - setup_ns_*_parallel(), assemble_ns_system_with_kelvin_force_parallel()
//   - solve_ns_system_direct_parallel(), extract_ns_solutions_parallel()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"

// MMS exact solutions
#include "mms/ns/ns_mms.h"
#include "mms/magnetic/magnetic_mms.h"

// Production setup
#include "setup/magnetic_setup.h"
#include "setup/ns_setup.h"

// Production assembly
#include "assembly/magnetic_assembler.h"
#include "assembly/ns_assembler.h"

// Production solvers
#include "solvers/magnetic_solver.h"
#include "solvers/ns_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>

constexpr int dim = 2;

// ============================================================================
// Helper: Extract M components and phi from monolithic vector
// Same logic as PhaseFieldProblem::extract_magnetic_components()
// ============================================================================
static void extract_magnetic_to_auxiliary(
    const dealii::DoFHandler<dim>& mag_dof,
    const dealii::TrilinosWrappers::MPI::Vector& mag_ghosted,
    const dealii::DoFHandler<dim>& M_dof,
    const dealii::DoFHandler<dim>& phi_dof,
    dealii::TrilinosWrappers::MPI::Vector& Mx_out,
    dealii::TrilinosWrappers::MPI::Vector& My_out,
    dealii::TrilinosWrappers::MPI::Vector& phi_out)
{
    const auto& fe_mag = mag_dof.get_fe();
    const unsigned int dofs_per_cell_mag = fe_mag.dofs_per_cell;
    const unsigned int dofs_per_cell_M = M_dof.get_fe().dofs_per_cell;
    const unsigned int dofs_per_cell_phi = phi_dof.get_fe().dofs_per_cell;

    std::vector<dealii::types::global_dof_index> mag_indices(dofs_per_cell_mag);
    std::vector<dealii::types::global_dof_index> M_indices(dofs_per_cell_M);
    std::vector<dealii::types::global_dof_index> phi_indices(dofs_per_cell_phi);

    for (const auto& cell : mag_dof.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        cell->get_dof_indices(mag_indices);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            M_cell(&cell->get_triangulation(), cell->level(), cell->index(), &M_dof);
        typename dealii::DoFHandler<dim>::active_cell_iterator
            phi_cell(&cell->get_triangulation(), cell->level(), cell->index(), &phi_dof);

        M_cell->get_dof_indices(M_indices);
        phi_cell->get_dof_indices(phi_indices);

        // Extract Mx (component 0) and My (component 1) from monolithic
        unsigned int M_local = 0;
        for (unsigned int i = 0; i < dofs_per_cell_mag; ++i)
        {
            const unsigned int comp = fe_mag.system_to_component_index(i).first;
            if (comp == 0) // Mx
            {
                if (Mx_out.locally_owned_elements().is_element(M_indices[M_local]))
                    Mx_out[M_indices[M_local]] = mag_ghosted[mag_indices[i]];
                M_local++;
            }
        }

        // Extract My
        unsigned int My_local = 0;
        for (unsigned int i = 0; i < dofs_per_cell_mag; ++i)
        {
            const unsigned int comp = fe_mag.system_to_component_index(i).first;
            if (comp == 1) // My
            {
                if (My_out.locally_owned_elements().is_element(M_indices[My_local]))
                    My_out[M_indices[My_local]] = mag_ghosted[mag_indices[i]];
                My_local++;
            }
        }

        // Extract phi (component dim = 2)
        unsigned int phi_local = 0;
        for (unsigned int i = 0; i < dofs_per_cell_mag; ++i)
        {
            const unsigned int comp = fe_mag.system_to_component_index(i).first;
            if (comp == dim) // phi
            {
                if (phi_out.locally_owned_elements().is_element(phi_indices[phi_local]))
                    phi_out[phi_indices[phi_local]] = mag_ghosted[mag_indices[i]];
                phi_local++;
            }
        }
    }

    Mx_out.compress(dealii::VectorOperation::insert);
    My_out.compress(dealii::VectorOperation::insert);
    phi_out.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Single refinement test
// ============================================================================
static CoupledMMSResult run_magnetic_ns_single(
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
    const double nu = params.physics.nu_ferro;
    const double mu_0 = params.physics.mu_0;

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

    double local_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h = std::min(local_h, cell->diameter());
    MPI_Allreduce(&local_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // ========================================================================
    // Setup Monolithic Magnetics DoFs: FESystem (DG^dim + CG)
    // ========================================================================
    dealii::FESystem<dim> fe_mag(
        dealii::FE_DGQ<dim>(params.fe.degree_magnetization), dim,
        dealii::FE_Q<dim>(params.fe.degree_potential), 1);

    dealii::DoFHandler<dim> mag_dof(triangulation);
    mag_dof.distribute_dofs(fe_mag);
    dealii::DoFRenumbering::component_wise(mag_dof);

    dealii::IndexSet mag_owned = mag_dof.locally_owned_dofs();
    dealii::IndexSet mag_relevant = dealii::DoFTools::extract_locally_relevant_dofs(mag_dof);

    // Auxiliary DoFHandlers for M and phi (for NS Kelvin force assembly)
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);
    dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);

    dealii::DoFHandler<dim> M_dof(triangulation), phi_dof(triangulation);
    M_dof.distribute_dofs(fe_M);
    phi_dof.distribute_dofs(fe_phi);

    dealii::IndexSet M_owned = M_dof.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof);
    dealii::IndexSet phi_owned = phi_dof.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof);

    // ========================================================================
    // Setup NS DoFs
    // ========================================================================
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::FE_Q<dim> fe_p(params.fe.degree_pressure);
    dealii::DoFHandler<dim> ux_dof(triangulation), uy_dof(triangulation), p_dof(triangulation);
    ux_dof.distribute_dofs(fe_vel);
    uy_dof.distribute_dofs(fe_vel);
    p_dof.distribute_dofs(fe_p);

    dealii::IndexSet ux_owned = ux_dof.locally_owned_dofs();
    dealii::IndexSet uy_owned = uy_dof.locally_owned_dofs();
    dealii::IndexSet p_owned = p_dof.locally_owned_dofs();
    dealii::IndexSet ux_relevant = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof);
    dealii::IndexSet uy_relevant = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof);
    dealii::IndexSet p_relevant = dealii::DoFTools::extract_locally_relevant_dofs(p_dof);

    // Dummy theta DoF (θ=1 for this test)
    dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);
    dealii::DoFHandler<dim> theta_dof(triangulation);
    theta_dof.distribute_dofs(fe_phase);
    dealii::IndexSet theta_owned = theta_dof.locally_owned_dofs();
    dealii::IndexSet theta_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof);

    result.n_dofs = mag_dof.n_dofs() + ux_dof.n_dofs() + uy_dof.n_dofs() + p_dof.n_dofs();

    // ========================================================================
    // Setup Monolithic Magnetics system (PRODUCTION)
    // ========================================================================
    dealii::AffineConstraints<double> mag_constraints;
    dealii::TrilinosWrappers::SparseMatrix mag_matrix;

    setup_magnetic_system<dim>(
        mag_dof, mag_owned, mag_relevant,
        mag_constraints, mag_matrix, mpi_communicator, pcout);

    dealii::TrilinosWrappers::MPI::Vector mag_rhs(mag_owned, mpi_communicator);

    // ========================================================================
    // Setup NS system
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
    setup_ns_velocity_constraints_parallel<dim>(ux_dof, uy_dof, ux_constraints, uy_constraints);
    setup_ns_pressure_constraints_parallel<dim>(p_dof, p_constraints);

    std::vector<dealii::types::global_dof_index> ux_to_ns, uy_to_ns, p_to_ns;
    dealii::IndexSet ns_owned, ns_relevant;
    dealii::AffineConstraints<double> ns_constraints;
    dealii::TrilinosWrappers::SparsityPattern ns_sparsity;

    setup_ns_coupled_system_parallel<dim>(
        ux_dof, uy_dof, p_dof,
        ux_constraints, uy_constraints, p_constraints,
        ux_to_ns, uy_to_ns, p_to_ns,
        ns_owned, ns_relevant, ns_constraints, ns_sparsity,
        mpi_communicator, pcout);

    dealii::TrilinosWrappers::SparseMatrix ns_matrix;
    ns_matrix.reinit(ns_sparsity);
    dealii::TrilinosWrappers::MPI::Vector ns_rhs(ns_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ns_solution(ns_owned, mpi_communicator);

    // ========================================================================
    // Solution vectors
    // ========================================================================

    // Dummy theta = 1 (full ferrofluid)
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_rel(theta_owned, theta_relevant, mpi_communicator);
    theta_vec = 1.0;
    theta_rel = theta_vec;

    // Dummy velocity (U = 0 for magnetic solve)
    dealii::TrilinosWrappers::MPI::Vector ux_zero(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_zero(ux_owned, ux_relevant, mpi_communicator);
    ux_zero = 0;
    uy_zero = 0;

    // Monolithic magnetics
    dealii::TrilinosWrappers::MPI::Vector mag_solution(mag_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector mag_old(mag_owned, mag_relevant, mpi_communicator);

    // Auxiliary extracted vectors (for NS Kelvin force)
    dealii::TrilinosWrappers::MPI::Vector Mx_vec(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_vec(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_vec(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_rel(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_rel(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_rel(phi_owned, phi_relevant, mpi_communicator);

    // NS
    dealii::TrilinosWrappers::MPI::Vector ux_sol(ux_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_sol(uy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector p_sol(p_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ux_old(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_old(uy_owned, uy_relevant, mpi_communicator);

    // ========================================================================
    // Initialize from exact solutions at t_start
    // ========================================================================
    double current_time = t_start;

    // Monolithic magnetics IC
    {
        MagneticExactSolution<dim> mag_ic(t_start, L_y);
        dealii::VectorTools::interpolate(mag_dof, mag_ic, mag_solution);
        mag_constraints.distribute(mag_solution);
        mag_old = mag_solution;
    }

    // NS IC
    {
        NSExactVelocityX<dim> ux_ic(t_start, L_y);
        NSExactVelocityY<dim> uy_ic(t_start, L_y);
        dealii::TrilinosWrappers::MPI::Vector tmp_ux(ux_owned, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector tmp_uy(uy_owned, mpi_communicator);
        dealii::VectorTools::interpolate(ux_dof, ux_ic, tmp_ux);
        dealii::VectorTools::interpolate(uy_dof, uy_ic, tmp_uy);
        ux_old = tmp_ux;
        uy_old = tmp_uy;
    }

    // ========================================================================
    // Create PRODUCTION assembler and solver
    // ========================================================================
    MagneticAssembler<dim> mag_assembler(
        mms_params, mag_dof, ux_dof, theta_dof,
        mag_constraints, mpi_communicator);

    MagneticSolver<dim> mag_solver(mag_owned, mpi_communicator);

    // ========================================================================
    // Time stepping
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;
        const double t_old = current_time - dt;

        // ====================================================================
        // Step 1: Solve Monolithic M+φ (θ=1, U=0)
        // ====================================================================
        mag_old = mag_solution;

        mag_assembler.assemble(
            mag_matrix, mag_rhs,
            ux_zero, uy_zero,   // U = 0 (no advection in this test)
            theta_rel,          // θ = 1 (constant)
            mag_old,
            dt, current_time);

        mag_solver.solve(mag_matrix, mag_solution, mag_rhs);
        mag_constraints.distribute(mag_solution);

        // ====================================================================
        // Step 2: Extract M and φ to auxiliary DoFHandlers
        // ====================================================================
        dealii::TrilinosWrappers::MPI::Vector mag_ghosted(
            mag_owned, mag_relevant, mpi_communicator);
        mag_ghosted = mag_solution;

        extract_magnetic_to_auxiliary(
            mag_dof, mag_ghosted, M_dof, phi_dof,
            Mx_vec, My_vec, phi_vec);

        Mx_rel = Mx_vec;
        My_rel = My_vec;
        phi_rel = phi_vec;

        // ====================================================================
        // Step 3: Solve NS with Kelvin force from M, H
        // ====================================================================
        ns_matrix = 0;
        ns_rhs = 0;

        assemble_ns_system_with_kelvin_force_parallel<dim>(
            ux_dof, uy_dof, p_dof,
            ux_old, uy_old,
            nu, dt, true, true,  // include_time, include_convection
            ux_to_ns, uy_to_ns, p_to_ns,
            ns_owned, ns_constraints,
            ns_matrix, ns_rhs, mpi_communicator,
            phi_dof, M_dof,
            phi_rel, Mx_rel, My_rel,
            mu_0,
            true, current_time, t_old, L_y);  // MMS params

        ns_solution = 0;
        solve_ns_system_direct_parallel(
            ns_matrix, ns_rhs, ns_solution, ns_constraints,
            p_to_ns, ns_owned, mpi_communicator, false);

        extract_ns_solutions_parallel(
            ns_solution, ux_to_ns, uy_to_ns, p_to_ns,
            ux_owned, uy_owned, p_owned,
            ns_owned, ns_relevant,
            ux_sol, uy_sol, p_sol,
            mpi_communicator);

        ux_old = ux_sol;
        uy_old = uy_sol;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================

    // Monolithic magnetics errors (M and φ)
    {
        dealii::TrilinosWrappers::MPI::Vector mag_ghosted(
            mag_owned, mag_relevant, mpi_communicator);
        mag_ghosted = mag_solution;

        MagneticMMSError mag_err = compute_magnetic_mms_errors_parallel<dim>(
            mag_dof, mag_ghosted, current_time, L_y, mpi_communicator);

        result.Mx_L2 = mag_err.Mx_L2;
        result.My_L2 = mag_err.My_L2;
        result.M_L2 = mag_err.M_L2;
        result.M_H1 = mag_err.M_H1;
        result.M_Linf = mag_err.M_Linf;
        result.phi_L2 = mag_err.phi_L2;
        result.phi_H1 = mag_err.phi_H1;
        result.phi_Linf = mag_err.phi_Linf;
    }

    // NS errors
    {
        NSExactVelocityX<dim> exact_ux(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector ux_gh(ux_owned, ux_relevant, mpi_communicator);
        ux_gh = ux_sol;

        dealii::QGauss<dim> quad(fe_vel.degree + 2);
        dealii::Vector<double> cell_err(triangulation.n_active_cells());

        dealii::VectorTools::integrate_difference(
            ux_dof, ux_gh, exact_ux, cell_err, quad, dealii::VectorTools::L2_norm);
        double local_sq = cell_err.norm_sqr(), global_sq;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.ux_L2 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(
            ux_dof, ux_gh, exact_ux, cell_err, quad, dealii::VectorTools::H1_seminorm);
        local_sq = cell_err.norm_sqr();
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.ux_H1 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(
            ux_dof, ux_gh, exact_ux, cell_err, quad, dealii::VectorTools::Linfty_norm);
        double local_max = cell_err.linfty_norm(), global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
        result.ux_Linf = global_max;

        // Pressure
        NSExactPressure<dim> exact_p(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector p_gh(p_owned, p_relevant, mpi_communicator);
        p_gh = p_sol;

        dealii::VectorTools::integrate_difference(
            p_dof, p_gh, exact_p, cell_err, quad, dealii::VectorTools::L2_norm);
        local_sq = cell_err.norm_sqr();
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.p_L2 = std::sqrt(global_sq);

        dealii::VectorTools::integrate_difference(
            p_dof, p_gh, exact_p, cell_err, quad, dealii::VectorTools::Linfty_norm);
        local_max = cell_err.linfty_norm();
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
        result.p_Linf = global_max;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// Public interface
// ============================================================================
CoupledMMSConvergenceResult run_magnetic_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::MAGNETIC_NS;
    result.expected_L2_rate = params.fe.degree_velocity + 1;  // Q2 -> 3
    result.expected_H1_rate = params.fe.degree_velocity;      // Q2 -> 2
    result.expected_DG_rate = 2.0;                            // DG-Q1 -> 2

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n========================================\n";
        std::cout << "[MAGNETIC_NS] Magnetic → NS Coupled MMS Test\n";
        std::cout << "========================================\n";
        std::cout << "  Tests: Monolithic M+φ → Kelvin force → NS\n";
        std::cout << "  Algorithm:\n";
        std::cout << "    1. Solve monolithic M+φ (θ=1, U=0)\n";
        std::cout << "    2. Extract M, φ to auxiliary DoFs\n";
        std::cout << "    3. Solve NS with Kelvin force\n";
        std::cout << "  MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Expected rates:\n";
        std::cout << "    U: L2=" << result.expected_L2_rate << ", H1=" << result.expected_H1_rate << "\n";
        std::cout << "    p: L2=2 (Q1)\n";
        std::cout << "    M: L2=" << result.expected_DG_rate << " (DG-Q1)\n";
        std::cout << "    φ: L2=3, H1=2 (CG-Q2)\n";
        std::cout << "========================================\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Ref " << ref << "... " << std::flush;

        CoupledMMSResult r = run_magnetic_ns_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                      << ", p_L2=" << r.p_L2
                      << ", M_L2=" << r.M_L2
                      << ", phi_L2=" << r.phi_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}
