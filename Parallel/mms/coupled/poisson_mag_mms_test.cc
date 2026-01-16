// ============================================================================
// mms/coupled/poisson_mag_mms_test.cc - Poisson + Magnetization Coupled MMS
//
// Tests the bidirectional coupling with PICARD ITERATION (matches production):
//   1. Poisson: -Δφ = -∇·M (M appears as source)
//   2. Magnetization: ∂M/∂t + M/τ_M = χH/τ_M where H = -∇φ
//
// Uses PRODUCTION code paths:
//   - setup_poisson_constraints_and_sparsity() for Poisson setup
//   - assemble_poisson_matrix() once (matrix is constant)
//   - assemble_poisson_rhs() each Picard iteration (depends on M)
//   - PoissonSolver with cached AMG preconditioner
//   - setup_magnetization_sparsity() for DG setup
//   - MagnetizationAssembler (cached, reused)
//   - MagnetizationSolver (MUMPS direct)
//   - Picard iteration with under-relaxation (matches production)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"

// Individual MMS solutions
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

// Production code
#include "setup/poisson_setup.h"
#include "setup/magnetization_setup.h"
#include "assembly/poisson_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "solvers/poisson_solver.h"
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
#include <deal.II/lac/full_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <memory>

constexpr int dim = 2;

// ============================================================================
// Helper: to_string for CoupledMMSLevel
// ============================================================================
std::string to_string(CoupledMMSLevel level)
{
    switch (level)
    {
        case CoupledMMSLevel::POISSON_MAGNETIZATION: return "POISSON_MAGNETIZATION";
        case CoupledMMSLevel::CH_NS:                 return "CH_NS";
        case CoupledMMSLevel::FULL_SYSTEM:           return "FULL_SYSTEM";
        default:                                      return "UNKNOWN";
    }
}

// ============================================================================
// CoupledMMSConvergenceResult implementation
// ============================================================================
void CoupledMMSConvergenceResult::compute_rates()
{
    const size_t n = results.size();
    if (n < 2) return;

    auto compute_rate = [](double e_fine, double e_coarse, double h_fine, double h_coarse) {
        if (e_fine < 1e-14 || e_coarse < 1e-14) return 0.0;
        return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
    };

    theta_L2_rate.resize(n, 0.0);
    theta_H1_rate.resize(n, 0.0);
    ux_L2_rate.resize(n, 0.0);
    ux_H1_rate.resize(n, 0.0);
    p_L2_rate.resize(n, 0.0);
    phi_L2_rate.resize(n, 0.0);
    phi_H1_rate.resize(n, 0.0);
    M_L2_rate.resize(n, 0.0);

    for (size_t i = 1; i < n; ++i)
    {
        const auto& fine = results[i];
        const auto& coarse = results[i-1];

        theta_L2_rate[i] = compute_rate(fine.theta_L2, coarse.theta_L2, fine.h, coarse.h);
        theta_H1_rate[i] = compute_rate(fine.theta_H1, coarse.theta_H1, fine.h, coarse.h);
        ux_L2_rate[i] = compute_rate(fine.ux_L2, coarse.ux_L2, fine.h, coarse.h);
        ux_H1_rate[i] = compute_rate(fine.ux_H1, coarse.ux_H1, fine.h, coarse.h);
        p_L2_rate[i] = compute_rate(fine.p_L2, coarse.p_L2, fine.h, coarse.h);
        phi_L2_rate[i] = compute_rate(fine.phi_L2, coarse.phi_L2, fine.h, coarse.h);
        phi_H1_rate[i] = compute_rate(fine.phi_H1, coarse.phi_H1, fine.h, coarse.h);
        M_L2_rate[i] = compute_rate(fine.M_L2, coarse.M_L2, fine.h, coarse.h);
    }
}

bool CoupledMMSConvergenceResult::passes(double tol) const
{
    if (results.size() < 2) return false;

    bool pass = true;
    const double L2_min = expected_L2_rate - tol;
    const double H1_min = expected_H1_rate - tol;
    const double DG_min = 2.0 - tol;  // DG-Q1 gets rate 2

    for (size_t i = 1; i < results.size(); ++i)
    {
        switch (level)
        {
            case CoupledMMSLevel::POISSON_MAGNETIZATION:
                if (phi_L2_rate[i] < L2_min) pass = false;
                if (phi_H1_rate[i] < H1_min) pass = false;
                if (M_L2_rate[i] < DG_min) pass = false;
                break;

            case CoupledMMSLevel::CH_NS:
                if (theta_L2_rate[i] < L2_min) pass = false;
                if (theta_H1_rate[i] < H1_min) pass = false;
                if (ux_L2_rate[i] < L2_min) pass = false;
                if (ux_H1_rate[i] < H1_min) pass = false;
                break;

            case CoupledMMSLevel::FULL_SYSTEM:
                if (theta_L2_rate[i] < L2_min) pass = false;
                if (phi_L2_rate[i] < L2_min) pass = false;
                if (M_L2_rate[i] < DG_min) pass = false;
                if (ux_L2_rate[i] < L2_min) pass = false;
                break;
        }
    }

    return pass;
}

void CoupledMMSConvergenceResult::print() const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: " << to_string(level) << "\n";
    std::cout << "========================================\n";

    if (level == CoupledMMSLevel::POISSON_MAGNETIZATION ||
        level == CoupledMMSLevel::FULL_SYSTEM)
    {
        std::cout << "--- Poisson Errors ---\n";
        std::cout << std::setw(5) << "Ref" << std::setw(10) << "h"
                  << std::setw(10) << "φ_L2" << std::setw(7) << "rate"
                  << std::setw(10) << "φ_H1" << std::setw(7) << "rate" << "\n";
        std::cout << std::string(49, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(10) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(10) << results[i].phi_L2
                      << std::setw(7) << std::fixed << std::setprecision(2) << phi_L2_rate[i]
                      << std::setw(10) << std::scientific << results[i].phi_H1
                      << std::setw(7) << std::fixed << phi_H1_rate[i] << "\n";
        }

        std::cout << "--- Magnetization Errors ---\n";
        std::cout << std::setw(5) << "Ref" << std::setw(10) << "h"
                  << std::setw(10) << "M_L2" << std::setw(7) << "rate" << "\n";
        std::cout << std::string(32, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(10) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(10) << results[i].M_L2
                      << std::setw(7) << std::fixed << std::setprecision(2) << M_L2_rate[i] << "\n";
        }
    }

    std::cout << "========================================\n";
    if (passes())
        std::cout << "[PASS] All rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some rates below expected!\n";
}

void CoupledMMSConvergenceResult::write_csv(const std::string& filename) const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::ofstream file(filename);
    if (!file.is_open()) return;

    file << "refinement,h,phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,M_L2,M_L2_rate,time\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        file << results[i].refinement << ","
             << results[i].h << ","
             << results[i].phi_L2 << "," << phi_L2_rate[i] << ","
             << results[i].phi_H1 << "," << phi_H1_rate[i] << ","
             << results[i].M_L2 << "," << M_L2_rate[i] << ","
             << results[i].total_time << "\n";
    }
}

// ============================================================================
// Single refinement test - matches production solve_poisson_magnetization_picard()
// ============================================================================
static CoupledMMSResult run_poisson_mag_single(
    unsigned int refinement,
    Parameters params,      // Changed from const Parameters& to Parameters (copy)
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    // Get parameters from params (no hardcoding!)
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;

    // Picard iteration parameters from params
    const double picard_tol = params.picard_tolerance;
    const double omega = 0.35;  // Under-relaxation factor
    const unsigned int max_picard = params.picard_iterations;

    if (this_rank == 0)
    {
        std::cout << "[COUPLED] ref=" << refinement
                  << ", tau_M=" << params.physics.tau_M
                  << ", chi_0=" << params.physics.chi_0
                  << ", dt=" << dt << "\n";
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Create mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        {params.domain.initial_cells_x,
         static_cast<unsigned int>(std::round(params.domain.initial_cells_x * L_y))},
        dealii::Point<dim>(params.domain.x_min, params.domain.y_min),
        dealii::Point<dim>(params.domain.x_max, params.domain.y_max));

    triangulation.refine_global(refinement);
    result.h = 1.0 / (params.domain.initial_cells_x * std::pow(2.0, refinement));

    // ========================================================================
    // Finite elements (from params)
    // ========================================================================
    dealii::FE_Q<dim> fe_phi(params.fe.degree_velocity);
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);

    // ========================================================================
    // DoF handlers
    // ========================================================================
    dealii::DoFHandler<dim> phi_dof_handler(triangulation);
    dealii::DoFHandler<dim> M_dof_handler(triangulation);

    phi_dof_handler.distribute_dofs(fe_phi);
    M_dof_handler.distribute_dofs(fe_M);

    // ========================================================================
    // IndexSets
    // ========================================================================
    dealii::IndexSet phi_owned = phi_dof_handler.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler);

    dealii::IndexSet M_owned = M_dof_handler.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof_handler);

    result.n_dofs = phi_dof_handler.n_dofs() + 2 * M_dof_handler.n_dofs();

    // ========================================================================
    // Poisson setup (PRODUCTION: matrix assembled ONCE, AMG cached)
    // ========================================================================
    dealii::AffineConstraints<double> phi_constraints;
    dealii::TrilinosWrappers::SparseMatrix phi_matrix;

    setup_poisson_constraints_and_sparsity<dim>(
        phi_dof_handler, phi_owned, phi_relevant,
        phi_constraints, phi_matrix,
        mpi_communicator, pcout);

    // Assemble Poisson matrix ONCE (it's constant: -Δ)
    assemble_poisson_matrix<dim>(phi_dof_handler, phi_constraints, phi_matrix);

    // Create cached Poisson solver with AMG preconditioner
    std::unique_ptr<PoissonSolver> poisson_solver = std::make_unique<PoissonSolver>(
        params.solvers.poisson, phi_owned, mpi_communicator);
    poisson_solver->initialize(phi_matrix);

    dealii::TrilinosWrappers::MPI::Vector phi_rhs(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_solution(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi_relevant_vec(phi_owned, phi_relevant, mpi_communicator);

    // ========================================================================
    // Magnetization setup (PRODUCTION: DG flux sparsity)
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix M_matrix;

    setup_magnetization_sparsity<dim>(
        M_dof_handler, M_owned, M_relevant,
        M_matrix, mpi_communicator, pcout);

    dealii::TrilinosWrappers::MPI::Vector Mx_solution(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_solution(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_old(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_old(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_relevant_vec(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_relevant_vec(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector rhs_Mx(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector rhs_My(M_owned, mpi_communicator);

    // Dummy velocity/theta for assembler (no advection or χ(θ) variation in this test)
    dealii::DoFHandler<dim> dummy_U_dof(triangulation);
    dealii::DoFHandler<dim> dummy_theta_dof(triangulation);
    dealii::FE_Q<dim> fe_dummy(1);
    dummy_U_dof.distribute_dofs(fe_dummy);
    dummy_theta_dof.distribute_dofs(fe_dummy);

    dealii::IndexSet dummy_owned = dummy_U_dof.locally_owned_dofs();
    dealii::IndexSet dummy_relevant = dealii::DoFTools::extract_locally_relevant_dofs(dummy_U_dof);
    dealii::TrilinosWrappers::MPI::Vector Ux_dummy(dummy_owned, dummy_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Uy_dummy(dummy_owned, dummy_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_dummy(dummy_owned, dummy_relevant, mpi_communicator);
    Ux_dummy = 0;
    Uy_dummy = 0;
    theta_dummy = 1.0;  // Constant θ=1 gives χ = χ_0

    // PRODUCTION: Cached magnetization assembler and solver
    std::unique_ptr<MagnetizationAssembler<dim>> mag_assembler =
        std::make_unique<MagnetizationAssembler<dim>>(
            params, M_dof_handler, dummy_U_dof,
            phi_dof_handler, dummy_theta_dof, mpi_communicator);

    LinearSolverParams mag_solver_params;
    mag_solver_params.use_iterative = false;  // MUMPS direct
    mag_solver_params.rel_tolerance = 1e-10;

    std::unique_ptr<MagnetizationSolver<dim>> mag_solver =
        std::make_unique<MagnetizationSolver<dim>>(mag_solver_params, M_owned, mpi_communicator);

    // ========================================================================
    // Initialize fields with exact solutions at t_start
    // ========================================================================
    double current_time = t_start;

    // Initialize φ
    {
        PoissonExactSolution<dim> exact_phi(current_time, L_y);
        dealii::VectorTools::interpolate(phi_dof_handler, exact_phi, phi_solution);
        phi_relevant_vec = phi_solution;
    }

    // Initialize M (cell-wise L² projection for DG)
    {
        MagExactMx<dim> exact_Mx(current_time, L_y);
        MagExactMy<dim> exact_My(current_time, L_y);

        dealii::QGauss<dim> quadrature(fe_M.degree + 2);
        dealii::FEValues<dim> fe_values(fe_M, quadrature,
            dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

        const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature.size();

        dealii::FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
        dealii::FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
        dealii::Vector<double> local_rhs_x(dofs_per_cell);
        dealii::Vector<double> local_rhs_y(dofs_per_cell);
        dealii::Vector<double> local_sol_x(dofs_per_cell);
        dealii::Vector<double> local_sol_y(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

        for (const auto& cell : M_dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

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
                    {
                        const double phi_j = fe_values.shape_value(j, q);
                        local_mass(i, j) += phi_i * phi_j * JxW;
                    }
                }
            }

            local_mass_inv.invert(local_mass);
            local_mass_inv.vmult(local_sol_x, local_rhs_x);
            local_mass_inv.vmult(local_sol_y, local_rhs_y);

            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                Mx_solution[local_dofs[i]] = local_sol_x(i);
                My_solution[local_dofs[i]] = local_sol_y(i);
            }
        }
        Mx_solution.compress(dealii::VectorOperation::insert);
        My_solution.compress(dealii::VectorOperation::insert);
    }

    Mx_old = Mx_solution;
    My_old = My_solution;
    Mx_relevant_vec = Mx_solution;
    My_relevant_vec = My_solution;

    // ========================================================================
    // Time stepping with Picard iteration
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_communicator);

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Save old solution for time derivative
        Mx_old = Mx_solution;
        My_old = My_solution;

        // Picard iteration for nonlinear coupling
        for (unsigned int picard_iter = 0; picard_iter < max_picard; ++picard_iter)
        {
            // Save for convergence check and under-relaxation
            Mx_prev = Mx_solution;
            My_prev = My_solution;

            // ----------------------------------------------------------------
            // Step 1: Solve Poisson with current M as source
            // -Δφ = -∇·M + f_φ^MMS
            // PRODUCTION: assemble_poisson_rhs (matrix already assembled)
            // ----------------------------------------------------------------
            Mx_relevant_vec = Mx_solution;
            My_relevant_vec = My_solution;

            phi_rhs = 0;
            assemble_poisson_rhs<dim>(
                phi_dof_handler, M_dof_handler,
                Mx_relevant_vec, My_relevant_vec,
                params, current_time,
                phi_constraints, phi_rhs);

            poisson_solver->solve(phi_rhs, phi_solution, phi_constraints, false);
            phi_relevant_vec = phi_solution;

            // ----------------------------------------------------------------
            // Step 2: Solve Magnetization with just-computed φ
            // ∂M/∂t + M/τ_M = χH/τ_M + f_M^MMS, where H = -∇φ
            // PRODUCTION: MagnetizationAssembler
            // ----------------------------------------------------------------
            dealii::TrilinosWrappers::MPI::Vector Mx_old_rel(M_owned, M_relevant, mpi_communicator);
            dealii::TrilinosWrappers::MPI::Vector My_old_rel(M_owned, M_relevant, mpi_communicator);
            Mx_old_rel = Mx_old;
            My_old_rel = My_old;

            M_matrix = 0;
            rhs_Mx = 0;
            rhs_My = 0;

            mag_assembler->assemble(
                M_matrix, rhs_Mx, rhs_My,
                Ux_dummy, Uy_dummy, phi_relevant_vec, theta_dummy,
                Mx_old_rel, My_old_rel,
                dt, current_time);

            // Initialize solver (factorization) - first Picard iteration only
            if (picard_iter == 0)
                mag_solver->initialize(M_matrix);

            mag_solver->solve(Mx_solution, rhs_Mx);
            mag_solver->solve(My_solution, rhs_My);

            // ----------------------------------------------------------------
            // Apply under-relaxation (matches production)
            // M^{k+1} = ω * M_new + (1-ω) * M_prev
            // ----------------------------------------------------------------
            if (omega < 1.0)
            {
                Mx_solution.sadd(omega, 1.0 - omega, Mx_prev);
                My_solution.sadd(omega, 1.0 - omega, My_prev);
            }

            // ----------------------------------------------------------------
            // Check convergence
            // ----------------------------------------------------------------
            dealii::TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_communicator);
            dealii::TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_communicator);
            Mx_diff = Mx_solution;
            My_diff = My_solution;
            Mx_diff -= Mx_prev;
            My_diff -= My_prev;

            double M_change = Mx_diff.l2_norm() + My_diff.l2_norm();
            double M_norm = Mx_solution.l2_norm() + My_solution.l2_norm() + 1e-14;
            double picard_residual = M_change / M_norm;

            if (picard_residual < picard_tol)
                break;
        }

        // Update ghosted vectors
        Mx_relevant_vec = Mx_solution;
        My_relevant_vec = My_solution;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================

    // Poisson error (with mean correction for Neumann BC)
    {
        dealii::TrilinosWrappers::MPI::Vector phi_rel(phi_owned, phi_relevant, mpi_communicator);
        phi_rel = phi_solution;

        PoissonMMSError phi_err = compute_poisson_mms_errors_parallel<dim>(
            phi_dof_handler, phi_rel, current_time, L_y, mpi_communicator);

        result.phi_L2 = phi_err.L2_error;
        result.phi_H1 = phi_err.H1_error;
    }

    // Magnetization error
    {
        dealii::TrilinosWrappers::MPI::Vector Mx_rel(M_owned, M_relevant, mpi_communicator);
        dealii::TrilinosWrappers::MPI::Vector My_rel(M_owned, M_relevant, mpi_communicator);
        Mx_rel = Mx_solution;
        My_rel = My_solution;

        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            M_dof_handler, Mx_rel, My_rel, current_time, L_y, mpi_communicator);

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
CoupledMMSConvergenceResult run_poisson_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::POISSON_MAGNETIZATION;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[POISSON_MAG] Running coupled MMS test with Picard iteration...\n";
        std::cout << "  MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Picard tol: " << params.picard_tolerance << ", max iter: " << params.picard_iterations << "\n";
        std::cout << "  tau_M: " << params.physics.tau_M << ", chi_0: " << params.physics.chi_0 << "\n";
        std::cout << "  Expected rates: φ L2 = 3, φ H1 = 2, M L2 = 2\n";
    }

    // Set MMS mode
    params.enable_mms = true;

    for (unsigned int ref : refinements)
    {
        CoupledMMSResult r = run_poisson_mag_single(ref, params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "  ref=" << ref
                      << " φ_L2=" << std::scientific << std::setprecision(2) << r.phi_L2
                      << ", M_L2=" << r.M_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}