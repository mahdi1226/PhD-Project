// ============================================================================
// mms/coupled/poisson_mag_mms_test.cc - Poisson + Magnetization Coupled MMS
//
// Tests the bidirectional coupling with PICARD ITERATION (matches production):
//   1. Poisson: -Δφ = -∇·M (M appears as source)
//   2. Magnetization: ∂M/∂t + M/τ_M = χH/τ_M where H = ∇φ
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
#include <fstream>
#include <chrono>
#include <cmath>
#include <memory>

#include "utilities/mpi_tools.h"

constexpr int dim = 2;


// ============================================================================
// Single refinement test - matches production solve_poisson_magnetization_picard()
// ============================================================================
static CoupledMMSResult run_poisson_mag_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;


    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    // Get parameters from params
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;

    // Picard iteration parameters
    const double picard_tol = 1e-10;
    const double omega = 1.0;  // Under-relaxation factor
    const unsigned int max_picard = 50;

    pcout << "[POISSON_MAG] ref=" << refinement
          << ", tau_M=" << params.physics.tau_M
          << ", chi_0=" << params.physics.chi_0
          << ", dt=" << dt << "\n";

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
        {
            params.domain.initial_cells_x,
            static_cast<unsigned int>(std::round(params.domain.initial_cells_x * L_y))
        },
        dealii::Point<dim>(params.domain.x_min, params.domain.y_min),
        dealii::Point<dim>(params.domain.x_max, params.domain.y_max));

    triangulation.refine_global(refinement);
    result.h = 1.0 / (params.domain.initial_cells_x * std::pow(2.0, refinement));

    // ========================================================================
    // Finite elements
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

    assemble_poisson_matrix<dim>(phi_dof_handler, phi_constraints, phi_matrix);

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

    // Picard iteration temporaries
    dealii::TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_communicator);

    // Dummy velocity/theta (no advection or χ(θ) variation in this test)
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
    Parameters local_params = params;
    local_params.enable_mms = true;

    std::unique_ptr<MagnetizationAssembler<dim>> mag_assembler =
        std::make_unique<MagnetizationAssembler<dim>>(
            local_params, M_dof_handler, dummy_U_dof,
            phi_dof_handler, dummy_theta_dof, mpi_communicator);

    std::unique_ptr<MagnetizationSolver<dim>> mag_solver =
        std::make_unique<MagnetizationSolver<dim>>(
            params.solvers.magnetization, M_owned, mpi_communicator);

    // ========================================================================
    // Initialize solutions from exact at t = t_start
    // ========================================================================
    double current_time = t_start;

    PoissonExactSolution<dim> phi_exact(current_time, L_y);
    MagExactMx<dim> Mx_exact(current_time, L_y);
    MagExactMy<dim> My_exact(current_time, L_y);

    dealii::VectorTools::interpolate(phi_dof_handler, phi_exact, phi_solution);
    phi_constraints.distribute(phi_solution);
    phi_relevant_vec = phi_solution;

    dealii::VectorTools::interpolate(M_dof_handler, Mx_exact, Mx_solution);
    dealii::VectorTools::interpolate(M_dof_handler, My_exact, My_solution);

    Mx_old = Mx_solution;
    My_old = My_solution;
    Mx_relevant_vec = Mx_solution;
    My_relevant_vec = My_solution;

    // ========================================================================
    // Time stepping
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Save old M for time derivative
        Mx_old = Mx_solution;
        My_old = My_solution;

        // ====================================================================
        // Picard iteration for Poisson ↔ Magnetization coupling
        // ====================================================================
        for (unsigned int picard_iter = 0; picard_iter < max_picard; ++picard_iter)
        {
            // Save for convergence check and under-relaxation
            Mx_prev = Mx_solution;
            My_prev = My_solution;

            // ----------------------------------------------------------------
            // Step 1: Solve Poisson with current M as source
            // ----------------------------------------------------------------
            Mx_relevant_vec = Mx_solution;
            My_relevant_vec = My_solution;

            phi_rhs = 0;
            assemble_poisson_rhs<dim>(
                phi_dof_handler, M_dof_handler,
                Mx_relevant_vec, My_relevant_vec,
                local_params, current_time,
                phi_constraints, phi_rhs);

            poisson_solver->solve(phi_rhs, phi_solution, phi_constraints, false);
            phi_relevant_vec = phi_solution;

            // ----------------------------------------------------------------
            // Step 2: Solve Magnetization with just-computed φ
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

            // Initialize solver on first Picard iteration
            if (picard_iter == 0)
                mag_solver->initialize(M_matrix);

            mag_solver->solve(Mx_solution, rhs_Mx);
            mag_solver->solve(My_solution, rhs_My);

            // ----------------------------------------------------------------
            // Apply under-relaxation: M^{k+1} = ω·M_new + (1-ω)·M_prev
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
            {
                if (this_rank == 0 && step == 0)
                    pcout << "  Picard converged in " << (picard_iter + 1)
                          << " iterations\n";
                break;
            }
        }

        // Update ghosted vectors
        Mx_relevant_vec = Mx_solution;
        My_relevant_vec = My_solution;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    // Poisson error
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
    Parameters params,  // BY VALUE - fresh copy each call
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
        std::cout << "  Picard tol: " << params.picard_tolerance
                  << ", max iter: " << params.picard_iterations << "\n";
        std::cout << "  tau_M: " << params.physics.tau_M
                  << ", chi_0: " << params.physics.chi_0 << "\n";
        std::cout << "  Expected rates: φ L2 = 3, φ H1 = 2, M L2 = 2\n\n";
    }

    for (unsigned int ref : refinements)
    {
        // Fresh copy each iteration to avoid any potential corruption
        Parameters iter_params = params;
        iter_params.enable_mms = true;

        CoupledMMSResult r = run_poisson_mag_single(ref, iter_params, n_time_steps, mpi_communicator);
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