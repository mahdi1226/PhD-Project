// ============================================================================
// mms/magnetization/magnetization_mms_test.cc - Magnetization MMS Test (PARALLEL)
//
// Self-contained parallel test using PRODUCTION:
//   - setup/magnetization_setup.h
//   - assembly/magnetization_assembler.h
//   - solvers/magnetization_solver.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/magnetization/magnetization_mms_test.h"
#include "mms/magnetization/magnetization_mms.h"

// Production code
#include "setup/magnetization_setup.h"
#include "assembly/magnetization_assembler.h"
#include "solvers/magnetization_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

constexpr int dim = 2;

// ============================================================================
// Helper
// ============================================================================
std::string to_string(MagSolverType type)
{
    switch (type)
    {
        case MagSolverType::Direct: return "Direct";
        case MagSolverType::GMRES:  return "GMRES";
        default: return "Unknown";
    }
}

// ============================================================================
// MagMMSConvergenceResult implementation
// ============================================================================
void MagMMSConvergenceResult::compute_rates()
{
    M_L2_rates.clear();
    for (size_t i = 1; i < results.size(); ++i)
    {
        const double e_fine = results[i].M_L2;
        const double e_coarse = results[i-1].M_L2;
        const double h_fine = results[i].h;
        const double h_coarse = results[i-1].h;

        if (e_coarse > 1e-15 && e_fine > 1e-15)
            M_L2_rates.push_back(std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine));
        else
            M_L2_rates.push_back(0.0);
    }
}

bool MagMMSConvergenceResult::passes(double tolerance) const
{
    if (M_L2_rates.empty())
        return false;
    return M_L2_rates.back() >= expected_L2_rate - tolerance;
}

void MagMMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "Magnetization MMS Convergence Results\n";
    std::cout << "========================================\n";
    std::cout << "Solver: " << to_string(solver_type) << "\n";
    std::cout << "FE degree: DG" << fe_degree << "\n";
    std::cout << "Expected L2 rate: " << expected_L2_rate << "\n\n";

    std::cout << std::left
              << std::setw(6) << "Ref"
              << std::setw(12) << "h"
              << std::setw(12) << "M_L2"
              << std::setw(8) << "rate"
              << std::setw(12) << "Mx_L2"
              << std::setw(12) << "My_L2"
              << std::setw(10) << "time(s)"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(6) << r.refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.h
                  << std::setw(12) << r.M_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << (i > 0 ? M_L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.Mx_L2
                  << std::setw(12) << r.My_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << r.total_time
                  << "\n";
    }

    std::cout << "========================================\n";
    if (passes())
        std::cout << "[PASS] Convergence rate within tolerance!\n";
    else
        std::cout << "[FAIL] Rate below expected!\n";
}

// ============================================================================
// run_magnetization_mms_single - Self-contained parallel test
// ============================================================================
MagMMSResult run_magnetization_mms_single(
    unsigned int refinement,
    const Parameters& params,
    MagSolverType solver_type,
    MPI_Comm mpi_communicator)
{
    MagMMSResult result;
    result.refinement = refinement;
    result.solver_type = solver_type;

    dealii::ConditionalOStream pcout(std::cout,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / 100;
    const unsigned int n_steps = 100;

    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.physics.tau_M = 1.0;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Create distributed mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params.domain.initial_cells_x;
    subdivisions[1] = params.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Setup DoF handlers - M (DG), U/phi/theta (CG)
    // ========================================================================
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);
    dealii::FE_Q<dim> fe_CG(params.fe.degree_velocity);  // For U, phi, theta

    dealii::DoFHandler<dim> M_dof_handler(triangulation);
    dealii::DoFHandler<dim> U_dof_handler(triangulation);
    dealii::DoFHandler<dim> phi_dof_handler(triangulation);
    dealii::DoFHandler<dim> theta_dof_handler(triangulation);

    M_dof_handler.distribute_dofs(fe_M);
    U_dof_handler.distribute_dofs(fe_CG);
    phi_dof_handler.distribute_dofs(fe_CG);
    theta_dof_handler.distribute_dofs(fe_CG);

    // IndexSets
    dealii::IndexSet M_locally_owned = M_dof_handler.locally_owned_dofs();
    dealii::IndexSet M_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof_handler);

    dealii::IndexSet U_locally_owned = U_dof_handler.locally_owned_dofs();
    dealii::IndexSet U_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(U_dof_handler);

    dealii::IndexSet phi_locally_owned = phi_dof_handler.locally_owned_dofs();
    dealii::IndexSet phi_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler);

    dealii::IndexSet theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    dealii::IndexSet theta_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);

    result.n_dofs = 2 * M_dof_handler.n_dofs();  // Mx + My

    // ========================================================================
    // Setup vectors
    // ========================================================================
    // M vectors (DG - no ghosts needed for assembly output, but need ghosted for reading)
    dealii::TrilinosWrappers::MPI::Vector Mx_owned(M_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_owned(M_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_old(M_locally_owned, M_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_old(M_locally_owned, M_locally_relevant, mpi_communicator);

    // Dummy vectors (U=0, phi=0, theta=1)
    dealii::TrilinosWrappers::MPI::Vector Ux(U_locally_owned, U_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Uy(U_locally_owned, U_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector phi(phi_locally_owned, phi_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta(theta_locally_owned, theta_locally_relevant, mpi_communicator);

    Ux = 0.0;
    Uy = 0.0;
    phi = 0.0;
    theta = 1.0;  // Constant phase field

    // RHS vectors
    dealii::TrilinosWrappers::MPI::Vector rhs_x(M_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector rhs_y(M_locally_owned, mpi_communicator);

    // ========================================================================
    // Setup matrix using PRODUCTION code
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix M_matrix;
    setup_magnetization_sparsity<dim>(
        M_dof_handler, M_locally_owned, M_locally_relevant,
        M_matrix, mpi_communicator, pcout);

    // ========================================================================
    // Initialize with exact solution at t_start
    // ========================================================================
    MagExactMx<dim> exact_Mx_init(t_start, L_y);
    MagExactMy<dim> exact_My_init(t_start, L_y);

    // For DG, use L² projection (cell-local)
    {
        const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();
        dealii::QGauss<dim> quadrature(fe_M.degree + 2);
        const unsigned int n_q_points = quadrature.size();

        dealii::FEValues<dim> fe_values(fe_M, quadrature,
            dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

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

                const double Mx_exact = exact_Mx_init.value(x_q);
                const double My_exact = exact_My_init.value(x_q);

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
                Mx_owned[local_dofs[i]] = local_sol_x(i);
                My_owned[local_dofs[i]] = local_sol_y(i);
            }
        }
        Mx_owned.compress(dealii::VectorOperation::insert);
        My_owned.compress(dealii::VectorOperation::insert);
    }

    // Copy to old (ghosted)
    Mx_old = Mx_owned;
    My_old = My_owned;

    // Compute min h
    {
        double local_min_h = std::numeric_limits<double>::max();
        for (const auto& cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                local_min_h = std::min(local_min_h, cell->diameter());
        MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);
    }

    // ========================================================================
    // PRODUCTION assembler and solver
    // ========================================================================
    MagnetizationAssembler<dim> assembler(
        mms_params, M_dof_handler, U_dof_handler,
        phi_dof_handler, theta_dof_handler, mpi_communicator);

    LinearSolverParams solver_params;
    solver_params.use_iterative = (solver_type == MagSolverType::GMRES);
    solver_params.max_iterations = 500;
    solver_params.rel_tolerance = 1e-10;

    MagnetizationSolver<dim> solver(solver_params, M_locally_owned, mpi_communicator);

    // ========================================================================
    // Time stepping
    // ========================================================================
    double current_time = t_start;

    for (unsigned int step = 0; step < n_steps; ++step)
    {
        current_time += dt;

        // Update old
        Mx_old = Mx_owned;
        My_old = My_owned;

        // Assemble
        assembler.assemble(
            M_matrix, rhs_x, rhs_y,
            Ux, Uy, phi, theta,
            Mx_old, My_old,
            dt, current_time);

        // Solve
        solver.initialize(M_matrix);
        solver.solve(Mx_owned, rhs_x);
        solver.solve(My_owned, rhs_y);
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    // Need ghosted vectors for error computation
    dealii::TrilinosWrappers::MPI::Vector Mx_ghosted(M_locally_owned, M_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_ghosted(M_locally_owned, M_locally_relevant, mpi_communicator);
    Mx_ghosted = Mx_owned;
    My_ghosted = My_owned;

    MagMMSError errors = compute_mag_mms_errors_parallel<dim>(
        M_dof_handler, Mx_ghosted, My_ghosted, current_time, L_y, mpi_communicator);

    result.Mx_L2 = errors.Mx_L2;
    result.My_L2 = errors.My_L2;
    result.M_L2 = errors.M_L2;

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// run_magnetization_mms_standalone - Full convergence study
// ============================================================================
MagMMSConvergenceResult run_magnetization_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    MagSolverType solver_type,
    MPI_Comm mpi_communicator)
{
    MagMMSConvergenceResult result;
    result.fe_degree = params.fe.degree_magnetization;
    result.solver_type = solver_type;
    result.expected_L2_rate = params.fe.degree_magnetization + 1;  // DG optimal

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[MAGNETIZATION_MMS] Running parallel convergence study...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  Solver: " << to_string(solver_type) << "\n";
        std::cout << "  FE degree: DG" << params.fe.degree_magnetization << "\n";
        std::cout << "  Expected L2 rate: " << result.expected_L2_rate << "\n";
        std::cout << "  Using PRODUCTION: setup + assembler + solver\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        MagMMSResult r = run_magnetization_mms_single(ref, params, solver_type, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "M_L2=" << std::scientific << std::setprecision(2) << r.M_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}