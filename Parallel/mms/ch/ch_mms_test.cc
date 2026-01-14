// ============================================================================
// mms/ch/ch_mms_test.cc - CH MMS Test Implementation (Parallel Version)
//
// Uses PRODUCTION code directly (no MMSContext):
//   - setup_ch_coupled_system() from setup/ch_setup.h
//   - assemble_ch_system() from assembly/ch_assembler.h
//   - solve_ch_system() from solvers/ch_solver.h
//
// NO PARAMETER OVERRIDES - uses Parameters defaults from parameters.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ch/ch_mms_test.h"
#include "mms/ch/ch_mms.h"

// PRODUCTION components
#include "setup/ch_setup.h"
#include "assembly/ch_assembler.h"
#include "solvers/ch_solver.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

constexpr int dim = 2;

// ============================================================================
// CHMMSConvergenceResult Implementation
// ============================================================================

void CHMMSConvergenceResult::compute_rates()
{
    theta_L2_rates.clear();
    theta_H1_rates.clear();
    psi_L2_rates.clear();

    for (size_t i = 1; i < results.size(); ++i)
    {
        const double h_ratio = results[i-1].h / results[i].h;
        const double log_h_ratio = std::log(h_ratio);

        if (results[i-1].theta_L2 > 1e-15 && results[i].theta_L2 > 1e-15)
        {
            theta_L2_rates.push_back(
                std::log(results[i-1].theta_L2 / results[i].theta_L2) / log_h_ratio);
        }
        else
        {
            theta_L2_rates.push_back(0.0);
        }

        if (results[i-1].theta_H1 > 1e-15 && results[i].theta_H1 > 1e-15)
        {
            theta_H1_rates.push_back(
                std::log(results[i-1].theta_H1 / results[i].theta_H1) / log_h_ratio);
        }
        else
        {
            theta_H1_rates.push_back(0.0);
        }

        if (results[i-1].psi_L2 > 1e-15 && results[i].psi_L2 > 1e-15)
        {
            psi_L2_rates.push_back(
                std::log(results[i-1].psi_L2 / results[i].psi_L2) / log_h_ratio);
        }
        else
        {
            psi_L2_rates.push_back(0.0);
        }
    }
}

void CHMMSConvergenceResult::print() const
{
    std::cout << "\n--- CH Errors ---\n";
    std::cout << std::left
              << std::setw(5) << "Ref"
              << std::setw(10) << "h"
              << std::setw(10) << "θ_L2"
              << std::setw(6) << "rate"
              << std::setw(10) << "θ_H1"
              << std::setw(6) << "rate"
              << std::setw(10) << "ψ_L2"
              << std::setw(6) << "rate"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        std::cout << std::left << std::setw(5) << r.refinement
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.h
                  << std::setw(10) << r.theta_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(6) << (i > 0 ? theta_L2_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.theta_H1
                  << std::fixed << std::setprecision(2)
                  << std::setw(6) << (i > 0 ? theta_H1_rates[i-1] : 0.0)
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.psi_L2
                  << std::fixed << std::setprecision(2)
                  << std::setw(6) << (i > 0 && !psi_L2_rates.empty() ? psi_L2_rates[i-1] : 0.0)
                  << "\n";
    }
}

void CHMMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[CH MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    file << "refinement,h,n_dofs,theta_L2,theta_L2_rate,theta_H1,theta_H1_rate,"
         << "psi_L2,psi_L2_rate,total_time\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        file << r.refinement << ","
             << std::scientific << std::setprecision(6) << r.h << ","
             << r.n_dofs << ","
             << r.theta_L2 << ","
             << (i > 0 ? theta_L2_rates[i-1] : 0.0) << ","
             << r.theta_H1 << ","
             << (i > 0 ? theta_H1_rates[i-1] : 0.0) << ","
             << r.psi_L2 << ","
             << (i > 0 ? psi_L2_rates[i-1] : 0.0) << ","
             << std::fixed << std::setprecision(4) << r.total_time << "\n";
    }

    file.close();
}

bool CHMMSConvergenceResult::passes(double tol) const
{
    if (theta_L2_rates.empty())
        return false;

    const double last_L2_rate = theta_L2_rates.back();
    const double last_H1_rate = theta_H1_rates.back();

    return (last_L2_rate >= expected_L2_rate - tol) &&
           (last_H1_rate >= expected_H1_rate - tol);
}

// ============================================================================
// Parallel error computation helper
// ============================================================================
static CHMMSErrors compute_ch_mms_errors_parallel(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double time,
    double L_y,
    MPI_Comm mpi_communicator)
{
    CHMMSErrors errors;

    // Exact solutions
    CHExactTheta<dim> exact_theta(L_y);
    CHExactPsi<dim> exact_psi(L_y);
    exact_theta.set_time(time);
    exact_psi.set_time(time);

    // Quadrature for error integration
    const unsigned int quad_degree = theta_dof_handler.get_fe().degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);

    dealii::FEValues<dim> fe_values(theta_dof_handler.get_fe(), quadrature,
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<double> psi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    double local_theta_L2_sq = 0.0;
    double local_theta_H1_sq = 0.0;
    double local_psi_L2_sq = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        // Get psi values on same cell
        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &theta_dof_handler.get_triangulation(),
            cell->level(), cell->index(), &psi_dof_handler);
        dealii::FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
                                            dealii::update_values);
        psi_fe_values.reinit(psi_cell);
        psi_fe_values.get_function_values(psi_solution, psi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            // θ errors
            const double theta_exact = exact_theta.value(x_q);
            const auto grad_theta_exact = exact_theta.gradient(x_q);
            const double theta_err = theta_values[q] - theta_exact;
            const auto grad_theta_err = theta_gradients[q] - grad_theta_exact;

            local_theta_L2_sq += theta_err * theta_err * JxW;
            local_theta_H1_sq += grad_theta_err * grad_theta_err * JxW;

            // ψ errors
            const double psi_exact = exact_psi.value(x_q);
            const double psi_err = psi_values[q] - psi_exact;
            local_psi_L2_sq += psi_err * psi_err * JxW;
        }
    }

    // MPI reduction
    double global_theta_L2_sq = 0.0;
    double global_theta_H1_sq = 0.0;
    double global_psi_L2_sq = 0.0;

    MPI_Allreduce(&local_theta_L2_sq, &global_theta_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_theta_H1_sq, &global_theta_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_psi_L2_sq, &global_psi_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    errors.theta_L2 = std::sqrt(global_theta_L2_sq);
    errors.theta_H1 = std::sqrt(global_theta_H1_sq);
    errors.psi_L2 = std::sqrt(global_psi_L2_sq);

    return errors;
}

// ============================================================================
// run_ch_mms_single - Single refinement test (ALL SETUP INLINED)
// ============================================================================
CHMMSResult run_ch_mms_single(
    unsigned int refinement,
    const Parameters& params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CHMMSResult result;
    result.refinement = refinement;

    dealii::ConditionalOStream pcout(std::cout,
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

    auto total_start = std::chrono::high_resolution_clock::now();

    // Time stepping parameters
    const double t_init = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    const double L_y = params.domain.y_max - params.domain.y_min;

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    // ========================================================================
    // Create mesh (INLINED from MMSContext::setup_mesh)
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

    // Assign boundary IDs: 0=bottom, 1=right, 2=top, 3=left
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (const auto& face : cell->face_iterators())
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
    }

    triangulation.refine_global(refinement);

    // ========================================================================
    // Setup CH system (INLINED from MMSContext::setup_ch)
    // ========================================================================
    dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);

    dealii::DoFHandler<dim> theta_dof_handler(triangulation);
    dealii::DoFHandler<dim> psi_dof_handler(triangulation);

    theta_dof_handler.distribute_dofs(fe_phase);
    psi_dof_handler.distribute_dofs(fe_phase);

    const unsigned int n_theta = theta_dof_handler.n_dofs();
    const unsigned int n_psi = psi_dof_handler.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // IndexSets
    dealii::IndexSet theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    dealii::IndexSet theta_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    dealii::IndexSet psi_locally_owned = psi_dof_handler.locally_owned_dofs();
    dealii::IndexSet psi_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler);

    // Combined IndexSets for coupled system
    dealii::IndexSet ch_locally_owned, ch_locally_relevant;
    ch_locally_owned.set_size(n_total);
    ch_locally_relevant.set_size(n_total);

    for (auto idx = theta_locally_owned.begin(); idx != theta_locally_owned.end(); ++idx)
        ch_locally_owned.add_index(*idx);
    for (auto idx = psi_locally_owned.begin(); idx != psi_locally_owned.end(); ++idx)
        ch_locally_owned.add_index(n_theta + *idx);

    for (auto idx = theta_locally_relevant.begin(); idx != theta_locally_relevant.end(); ++idx)
        ch_locally_relevant.add_index(*idx);
    for (auto idx = psi_locally_relevant.begin(); idx != psi_locally_relevant.end(); ++idx)
        ch_locally_relevant.add_index(n_theta + *idx);

    ch_locally_owned.compress();
    ch_locally_relevant.compress();

    // Index maps
    std::vector<dealii::types::global_dof_index> theta_to_ch_map(n_theta);
    std::vector<dealii::types::global_dof_index> psi_to_ch_map(n_psi);
    for (unsigned int i = 0; i < n_theta; ++i)
        theta_to_ch_map[i] = i;
    for (unsigned int i = 0; i < n_psi; ++i)
        psi_to_ch_map[i] = n_theta + i;

    // Individual constraints with MMS BCs
    dealii::AffineConstraints<double> theta_constraints;
    dealii::AffineConstraints<double> psi_constraints;
    theta_constraints.reinit(theta_locally_owned, theta_locally_relevant);
    psi_constraints.reinit(psi_locally_owned, psi_locally_relevant);

    CHMMSBoundaryTheta<dim> theta_bc(L_y);
    CHMMSBoundaryPsi<dim> psi_bc(L_y);
    theta_bc.set_time(t_init);
    psi_bc.set_time(t_init);

    for (unsigned int bid = 0; bid < 4; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(theta_dof_handler, bid, theta_bc, theta_constraints);
        dealii::VectorTools::interpolate_boundary_values(psi_dof_handler, bid, psi_bc, psi_constraints);
    }
    theta_constraints.close();
    psi_constraints.close();

    // PRODUCTION: Setup coupled system
    dealii::AffineConstraints<double> ch_constraints;
    dealii::TrilinosWrappers::SparseMatrix ch_matrix;

    setup_ch_coupled_system<dim>(
        theta_dof_handler, psi_dof_handler,
        theta_constraints, psi_constraints,
        ch_locally_owned, ch_locally_relevant,
        theta_to_ch_map, psi_to_ch_map,
        ch_constraints, ch_matrix,
        mpi_communicator, pcout);

    // Solution vectors
    dealii::TrilinosWrappers::MPI::Vector theta_owned(theta_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector theta_relevant(theta_locally_owned, theta_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_owned(psi_locally_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector psi_relevant(psi_locally_owned, psi_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ch_rhs(ch_locally_owned, mpi_communicator);

    // ========================================================================
    // Apply MMS initial conditions (INLINED from MMSContext)
    // ========================================================================
    CHMMSInitialTheta<dim> theta_ic(t_init, L_y);
    CHMMSInitialPsi<dim> psi_ic(t_init, L_y);

    dealii::VectorTools::interpolate(theta_dof_handler, theta_ic, theta_owned);
    dealii::VectorTools::interpolate(psi_dof_handler, psi_ic, psi_owned);

    theta_owned.compress(dealii::VectorOperation::insert);
    psi_owned.compress(dealii::VectorOperation::insert);
    theta_relevant = theta_owned;
    psi_relevant = psi_owned;

    // ========================================================================
    // Create dummy velocity DoFHandlers (CH standalone has no velocity)
    // ========================================================================
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::DoFHandler<dim> ux_dof_handler(triangulation);
    dealii::DoFHandler<dim> uy_dof_handler(triangulation);
    ux_dof_handler.distribute_dofs(fe_vel);
    uy_dof_handler.distribute_dofs(fe_vel);

    dealii::IndexSet ux_locally_owned = ux_dof_handler.locally_owned_dofs();
    dealii::IndexSet ux_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
    dealii::TrilinosWrappers::MPI::Vector ux_dummy(ux_locally_owned, ux_locally_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_dummy(ux_locally_owned, ux_locally_relevant, mpi_communicator);
    ux_dummy = 0;
    uy_dummy = 0;

    // ========================================================================
    // Time stepping loop
    // ========================================================================
    double total_solve_time = 0.0;
    unsigned int total_iterations = 0;
    double last_residual = 0.0;
    double current_time = t_init;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ====================================================================
        // Update constraints for current time (INLINED from MMSContext)
        // ====================================================================
        theta_constraints.clear();
        theta_constraints.reinit(theta_locally_owned, theta_locally_relevant);
        psi_constraints.clear();
        psi_constraints.reinit(psi_locally_owned, psi_locally_relevant);

        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);

        for (unsigned int bid = 0; bid < 4; ++bid)
        {
            dealii::VectorTools::interpolate_boundary_values(theta_dof_handler, bid, theta_bc, theta_constraints);
            dealii::VectorTools::interpolate_boundary_values(psi_dof_handler, bid, psi_bc, psi_constraints);
        }
        theta_constraints.close();
        psi_constraints.close();

        // Rebuild combined constraints
        ch_constraints.clear();
        ch_constraints.reinit(ch_locally_owned, ch_locally_relevant);

        for (auto idx = theta_locally_relevant.begin(); idx != theta_locally_relevant.end(); ++idx)
        {
            const unsigned int i = *idx;
            if (theta_constraints.is_constrained(i))
            {
                const auto* entries = theta_constraints.get_constraint_entries(i);
                if (entries == nullptr || entries->empty())
                {
                    ch_constraints.add_line(theta_to_ch_map[i]);
                    ch_constraints.set_inhomogeneity(theta_to_ch_map[i],
                                                     theta_constraints.get_inhomogeneity(i));
                }
            }
        }

        for (auto idx = psi_locally_relevant.begin(); idx != psi_locally_relevant.end(); ++idx)
        {
            const unsigned int i = *idx;
            if (psi_constraints.is_constrained(i))
            {
                const auto* entries = psi_constraints.get_constraint_entries(i);
                if (entries == nullptr || entries->empty())
                {
                    ch_constraints.add_line(psi_to_ch_map[i]);
                    ch_constraints.set_inhomogeneity(psi_to_ch_map[i],
                                                     psi_constraints.get_inhomogeneity(i));
                }
            }
        }
        ch_constraints.close();

        // ====================================================================
        // PRODUCTION assembly
        // ====================================================================
        assemble_ch_system<dim>(
            theta_dof_handler, psi_dof_handler,
            theta_relevant,
            ux_dof_handler, uy_dof_handler,
            ux_dummy, uy_dummy,
            mms_params, dt, current_time,
            theta_to_ch_map, psi_to_ch_map,
            ch_constraints,
            ch_matrix, ch_rhs);

        // ====================================================================
        // PRODUCTION solver
        // ====================================================================
        auto solve_start = std::chrono::high_resolution_clock::now();

        LinearSolverParams solver_params = params.solvers.ch;
        if (solver_type == CHSolverType::Direct)
        {
            solver_params.use_iterative = false;
            solver_params.fallback_to_direct = true;
        }

        SolverInfo info = solve_ch_system(
            ch_matrix, ch_rhs, ch_constraints,
            ch_locally_owned, theta_locally_owned, psi_locally_owned,
            theta_to_ch_map, psi_to_ch_map,
            theta_owned, psi_owned,
            solver_params, mpi_communicator, false);

        total_iterations += info.iterations;
        last_residual = info.residual;

        // Update ghost values
        theta_owned.compress(dealii::VectorOperation::insert);
        psi_owned.compress(dealii::VectorOperation::insert);
        theta_relevant = theta_owned;
        psi_relevant = psi_owned;

        auto solve_end = std::chrono::high_resolution_clock::now();
        total_solve_time += std::chrono::duration<double>(solve_end - solve_start).count();
    }

    // ========================================================================
    // Compute errors (parallel reduction)
    // ========================================================================
    CHMMSErrors errors = compute_ch_mms_errors_parallel(
        theta_dof_handler, psi_dof_handler,
        theta_relevant, psi_relevant,
        current_time, L_y, mpi_communicator);

    auto total_end = std::chrono::high_resolution_clock::now();

    // Compute h
    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());

    double global_min_h = 0.0;
    MPI_Allreduce(&local_min_h, &global_min_h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // Fill result (set both direct and alias names for compatibility)
    result.h = global_min_h;
    result.n_dofs = n_total;
    result.theta_L2 = errors.theta_L2;
    result.theta_H1 = errors.theta_H1;
    result.psi_L2 = errors.psi_L2;
    result.theta_L2_error = errors.theta_L2;  // Alias for mms_verification.cc
    result.theta_H1_error = errors.theta_H1;
    result.psi_L2_error = errors.psi_L2;
    result.solve_time = total_solve_time;
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();
    result.solver_iterations = total_iterations;
    result.solver_residual = last_residual;

    return result;
}

// ============================================================================
// run_ch_mms_standalone - Full convergence study
// ============================================================================

CHMMSConvergenceResult run_ch_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    CHSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    CHMMSConvergenceResult result;
    result.fe_degree = params.fe.degree_phase;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = params.fe.degree_phase + 1;
    result.expected_H1_rate = params.fe.degree_phase;

    const double t_init = 0.1;
    const double t_final = 0.2;
    result.dt = (t_final - t_init) / n_time_steps;

    if (this_rank == 0)
    {
        std::cout << "\n[CH_STANDALONE] Running parallel convergence study...\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  t ∈ [" << t_init << ", " << t_final << "], dt = " << result.dt << "\n";
        std::cout << "  ε = " << params.physics.epsilon
                  << ", γ = " << params.physics.mobility << "\n";
        std::cout << "  FE degree = Q" << params.fe.degree_phase << "\n";
        std::cout << "  Solver = " << (solver_type == CHSolverType::Direct ? "Direct" : "GMRES+AMG") << "\n";
        std::cout << "  Using PRODUCTION: setup_ch_coupled_system + assemble_ch_system + solve_ch_system\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        CHMMSResult single = run_ch_mms_single(ref, params, solver_type, n_time_steps, mpi_communicator);
        result.results.push_back(single);

        if (this_rank == 0)
        {
            std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << single.theta_L2
                      << ", θ_H1=" << single.theta_H1
                      << ", time=" << std::fixed << std::setprecision(1) << single.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// Wrapper for mms_verification.cc compatibility
// ============================================================================
CHMMSConvergenceResult run_ch_mms_parallel(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    // Convert to new result format
    CHMMSConvergenceResult conv_result = run_ch_mms_standalone(
        refinements, params, CHSolverType::GMRES_AMG, n_time_steps, mpi_communicator);

    // Map to CHMMSConvergenceResult format expected by mms_verification.cc
    CHMMSConvergenceResult result;
    result.fe_degree = conv_result.fe_degree;
    result.n_time_steps = conv_result.n_time_steps;
    result.dt = conv_result.dt;
    result.expected_L2_rate = conv_result.expected_L2_rate;
    result.expected_H1_rate = conv_result.expected_H1_rate;

    for (const auto& r : conv_result.results)
    {
        CHMMSResult mapped;
        mapped.refinement = r.refinement;
        mapped.h = r.h;
        mapped.n_dofs = r.n_dofs;
        mapped.theta_L2 = r.theta_L2;
        mapped.theta_H1 = r.theta_H1;
        mapped.psi_L2 = r.psi_L2;
        mapped.total_time = r.total_time;
        result.results.push_back(mapped);
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// compare_ch_solvers - Compare direct vs iterative
// ============================================================================

void compare_ch_solvers(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n[CH Solver Comparison] Refinement " << refinement << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    // Direct solver
    if (this_rank == 0)
        std::cout << "Direct (Amesos):\n";
    CHMMSResult direct = run_ch_mms_single(refinement, params, CHSolverType::Direct, n_time_steps, mpi_communicator);
    if (this_rank == 0)
    {
        std::cout << "  θ_L2 = " << std::scientific << direct.theta_L2
                  << ", time = " << std::fixed << std::setprecision(3) << direct.total_time << "s\n";
    }

    // Iterative solver
    if (this_rank == 0)
        std::cout << "Iterative (GMRES+AMG):\n";
    CHMMSResult iterative = run_ch_mms_single(refinement, params, CHSolverType::GMRES_AMG, n_time_steps, mpi_communicator);
    if (this_rank == 0)
    {
        std::cout << "  θ_L2 = " << std::scientific << iterative.theta_L2
                  << ", iters = " << iterative.solver_iterations
                  << ", time = " << std::fixed << std::setprecision(3) << iterative.total_time << "s\n";

        std::cout << "\nSpeedup (direct/iterative): "
                  << std::setprecision(2) << direct.total_time / iterative.total_time << "x\n";
    }
}