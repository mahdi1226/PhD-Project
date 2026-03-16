// ============================================================================
// mms/coupled/full_system_mms_test.cc - Full System Coupled MMS Test
//
// Tests ALL subsystems coupled together with monolithic magnetics:
//   1. CH: ∂θ/∂t + U·∇θ = γΔψ + f_θ
//   2. Monolithic M+φ: combined magnetization + Poisson block system
//   3. NS: ∂U/∂t + (U·∇)U - νΔU + ∇p = μ₀[(M·∇)H + ½(∇·M)H] + f_NS
//
// Time stepping (matches production PhaseFieldProblem::run()):
//   1. Solve CH ONCE (uses U^{n-1})
//   2. Solve monolithic M+φ (uses θ^n, U^{n-1})
//   3. Extract M, φ to auxiliary DoFs
//   4. Solve NS ONCE (uses Kelvin force from M, H)
//
// Uses PRODUCTION code paths:
//   - assemble_ch_system() with enable_mms=true
//   - solve_ch_system()
//   - setup_magnetic_system(), MagneticAssembler, MagneticSolver (MUMPS)
//   - assemble_ns_system_with_kelvin_force_parallel() with enable_mms=true
//   - solve_ns_system_direct_parallel()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"

// MMS exact solutions
#include "mms/ch/ch_mms.h"
#include "mms/ns/ns_mms.h"
#include "mms/magnetic/magnetic_mms.h"

// Production setup
#include "setup/magnetic_setup.h"
#include "setup/ns_setup.h"

// Production assembly
#include "assembly/ch_assembler.h"
#include "assembly/magnetic_assembler.h"
#include "assembly/ns_assembler.h"

// Production solvers
#include "solvers/ch_solver.h"
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

        // Extract Mx (component 0)
        unsigned int M_local = 0;
        for (unsigned int i = 0; i < dofs_per_cell_mag; ++i)
        {
            const unsigned int comp = fe_mag.system_to_component_index(i).first;
            if (comp == 0)
            {
                if (Mx_out.locally_owned_elements().is_element(M_indices[M_local]))
                    Mx_out[M_indices[M_local]] = mag_ghosted[mag_indices[i]];
                M_local++;
            }
        }

        // Extract My (component 1)
        unsigned int My_local = 0;
        for (unsigned int i = 0; i < dofs_per_cell_mag; ++i)
        {
            const unsigned int comp = fe_mag.system_to_component_index(i).first;
            if (comp == 1)
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
            if (comp == dim)
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
// Full system single refinement test
// ============================================================================
static CoupledMMSResult run_full_system_single(
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
    // Setup DoF handlers
    // ========================================================================

    // CH DoFs (Q2)
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

    // NS DoFs (Q2 velocity, DG pressure — paper A1)
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::FE_DGQ<dim> fe_p(params.fe.degree_pressure);
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

    // Monolithic Magnetics DoFs: FESystem (DG^dim + CG)
    dealii::FESystem<dim> fe_mag(
        dealii::FE_DGQ<dim>(params.fe.degree_magnetization), dim,
        dealii::FE_Q<dim>(params.fe.degree_potential), 1);

    dealii::DoFHandler<dim> mag_dof(triangulation);
    mag_dof.distribute_dofs(fe_mag);
    dealii::DoFRenumbering::component_wise(mag_dof);

    dealii::IndexSet mag_owned = mag_dof.locally_owned_dofs();
    dealii::IndexSet mag_relevant = dealii::DoFTools::extract_locally_relevant_dofs(mag_dof);

    // Auxiliary DoFHandlers for NS Kelvin force assembly
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);
    dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);

    dealii::DoFHandler<dim> M_dof(triangulation), phi_dof(triangulation);
    M_dof.distribute_dofs(fe_M);
    phi_dof.distribute_dofs(fe_phi);

    dealii::IndexSet M_owned = M_dof.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof);
    dealii::IndexSet phi_owned = phi_dof.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof);

    result.n_dofs = n_ch + ux_dof.n_dofs() + uy_dof.n_dofs()
                  + p_dof.n_dofs() + mag_dof.n_dofs();

    // ========================================================================
    // Setup CH system
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

    // CH constraints and boundary conditions
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
    // Setup Monolithic Magnetics system (PRODUCTION)
    // ========================================================================
    dealii::AffineConstraints<double> mag_constraints;
    dealii::TrilinosWrappers::SparseMatrix mag_matrix;

    setup_magnetic_system<dim>(
        mag_dof, mag_owned, mag_relevant,
        mag_constraints, mag_matrix, mpi_communicator, pcout);

    dealii::TrilinosWrappers::MPI::Vector mag_rhs(mag_owned, mpi_communicator);

    // ========================================================================
    // Solution vectors
    // ========================================================================
    // CH
    dealii::TrilinosWrappers::MPI::Vector theta_vec(theta_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_rel(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_old(theta_owned, theta_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_vec(psi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_rel(psi_owned, psi_relevant, mpi_communicator);

    // NS
    dealii::TrilinosWrappers::MPI::Vector ux_sol(ux_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_sol(uy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector p_sol(p_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ux_old(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_old(uy_owned, uy_relevant, mpi_communicator);

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

    // ========================================================================
    // Initialize from exact solutions at t_start
    // ========================================================================
    double current_time = t_start;

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

    // CH IC
    CHMMSInitialTheta<dim> theta_ic(t_start, L_y);
    CHMMSInitialPsi<dim> psi_ic(t_start, L_y);
    dealii::VectorTools::interpolate(theta_dof, theta_ic, theta_vec);
    dealii::VectorTools::interpolate(psi_dof, psi_ic, psi_vec);
    theta_rel = theta_vec;
    psi_rel = psi_vec;
    theta_old = theta_vec;

    // Monolithic magnetics IC: combined (Mx, My, phi)
    {
        MagneticExactSolution<dim> mag_ic(t_start, L_y);
        dealii::VectorTools::interpolate(mag_dof, mag_ic, mag_solution);
        mag_constraints.distribute(mag_solution);
        mag_old = mag_solution;
    }

    // ========================================================================
    // Create PRODUCTION assembler and solver
    // ========================================================================
    MagneticAssembler<dim> mag_assembler(
        mms_params, mag_dof, ux_dof, theta_dof,
        mag_constraints, mpi_communicator);

    MagneticSolver<dim> mag_solver(mag_owned, mpi_communicator);

    // ========================================================================
    // Time stepping (matches production algorithm)
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;
        const double t_old_time = current_time - dt;

        // ====================================================================
        // Step 1: Solve CH (uses U^{n-1})
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

        theta_old = theta_rel;

        // Assemble and solve CH
        ch_matrix = 0;
        ch_rhs = 0;
        assemble_ch_system<dim>(
            theta_dof, psi_dof, theta_old,
            ux_dof, uy_dof, ux_old, uy_old,
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
        // Step 2: Solve monolithic M+φ (uses θ^n from CH, U^{n-1})
        // ====================================================================
        mag_old = mag_solution;

        mag_assembler.assemble(
            mag_matrix, mag_rhs,
            ux_old, uy_old,     // Velocity from previous time step
            theta_rel,          // θ from CH solve (χ(θ) coupling)
            mag_old,            // Previous combined (M, φ) solution
            dt, current_time);

        mag_solver.solve(mag_matrix, mag_solution, mag_rhs);
        mag_constraints.distribute(mag_solution);

        // ====================================================================
        // Step 3: Extract M, φ to auxiliary DoFs for NS Kelvin force
        // ====================================================================
        {
            dealii::TrilinosWrappers::MPI::Vector mag_ghosted(
                mag_owned, mag_relevant, mpi_communicator);
            mag_ghosted = mag_solution;

            extract_magnetic_to_auxiliary(
                mag_dof, mag_ghosted, M_dof, phi_dof,
                Mx_vec, My_vec, phi_vec);

            Mx_rel = Mx_vec;
            My_rel = My_vec;
            phi_rel = phi_vec;
        }

        // ====================================================================
        // Step 4: Solve NS with Kelvin force
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
            true, current_time, t_old_time, L_y);  // MMS params

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

        // Update old solutions
        ux_old = ux_sol;
        uy_old = uy_sol;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================

    // CH errors (L2, H1, Linf)
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
        double local_max = cell_err.linfty_norm(), global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
        result.theta_Linf = global_max;
    }

    // NS errors (L2, H1, Linf)
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

        // Pressure (with mean subtraction)
        NSExactPressure<dim> exact_p(current_time, L_y);
        dealii::TrilinosWrappers::MPI::Vector p_gh(p_owned, p_relevant, mpi_communicator);
        p_gh = p_sol;

        dealii::QGauss<dim> quad_p(fe_p.degree + 2);
        dealii::FEValues<dim> fe_values_p(p_dof.get_fe(), quad_p,
            dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

        double local_p_integral = 0.0, local_exact_p_integral = 0.0, local_volume = 0.0;
        std::vector<double> p_vals(quad_p.size());

        for (const auto& cell : p_dof.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            fe_values_p.reinit(cell);
            fe_values_p.get_function_values(p_gh, p_vals);
            for (unsigned int q = 0; q < quad_p.size(); ++q)
            {
                const double JxW = fe_values_p.JxW(q);
                local_p_integral += p_vals[q] * JxW;
                local_exact_p_integral += exact_p.value(fe_values_p.quadrature_point(q)) * JxW;
                local_volume += JxW;
            }
        }

        double global_p_int = 0, global_exact_p_int = 0, global_vol = 0;
        MPI_Allreduce(&local_p_integral, &global_p_int, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        MPI_Allreduce(&local_exact_p_integral, &global_exact_p_int, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        MPI_Allreduce(&local_volume, &global_vol, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

        const double p_mean = global_p_int / global_vol;
        const double exact_p_mean = global_exact_p_int / global_vol;

        double local_p_err_sq = 0.0;
        double local_p_Linf = 0.0;
        for (const auto& cell : p_dof.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            fe_values_p.reinit(cell);
            fe_values_p.get_function_values(p_gh, p_vals);
            for (unsigned int q = 0; q < quad_p.size(); ++q)
            {
                const double JxW = fe_values_p.JxW(q);
                const double p_err = (p_vals[q] - p_mean) -
                    (exact_p.value(fe_values_p.quadrature_point(q)) - exact_p_mean);
                local_p_err_sq += p_err * p_err * JxW;
                local_p_Linf = std::max(local_p_Linf, std::abs(p_err));
            }
        }
        MPI_Allreduce(&local_p_err_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
        result.p_L2 = std::sqrt(global_sq);

        double global_p_Linf = 0.0;
        MPI_Allreduce(&local_p_Linf, &global_p_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
        result.p_Linf = global_p_Linf;
    }

    // Monolithic magnetics errors (M and φ: L2, H1, Linf)
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

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// Public interface
// ============================================================================
CoupledMMSConvergenceResult run_full_system_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::FULL_SYSTEM;
    result.expected_L2_rate = params.fe.degree_phase + 1;  // Q2 -> 3
    result.expected_H1_rate = params.fe.degree_phase;      // Q2 -> 2
    result.expected_DG_rate = 2.0;                         // DG-Q1 -> 2

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n========================================\n";
        std::cout << "[FULL_SYSTEM] Full System Coupled MMS Test\n";
        std::cout << "========================================\n";
        std::cout << "  Tests: CH + Monolithic Magnetics + NS\n";
        std::cout << "  Couplings:\n";
        std::cout << "    - CH advected by NS (U·∇θ)\n";
        std::cout << "    - Monolithic M+φ (χ(θ) coupling)\n";
        std::cout << "    - NS with Kelvin force μ₀(M·∇)H\n";
        std::cout << "  MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Expected rates:\n";
        std::cout << "    θ, U: L2=" << result.expected_L2_rate << ", H1=" << result.expected_H1_rate << "\n";
        std::cout << "    φ:    L2=" << result.expected_L2_rate << ", H1=" << result.expected_H1_rate << "\n";
        std::cout << "    M:    L2=" << result.expected_DG_rate << " (DG-Q1)\n";
        std::cout << "========================================\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Ref " << ref << "... " << std::flush;

        CoupledMMSResult r = run_full_system_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << r.theta_L2
                      << ", U_L2=" << r.ux_L2
                      << ", φ_L2=" << r.phi_L2
                      << ", M_L2=" << r.M_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}
