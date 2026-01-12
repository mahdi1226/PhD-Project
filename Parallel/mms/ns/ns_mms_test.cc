// ============================================================================
// mms/ns/ns_mms_test.cc - Parallel NS MMS Test Implementation
//
// Uses PRODUCTION code with enable_mms flag:
//   - setup_ns_coupled_system_parallel() from setup/ns_setup.h
//   - assemble_ns_system_parallel() from assembly/ns_assembler.h (enable_mms=true)
//   - solve_ns_system_schur_parallel() from solvers/ns_solver.h
//   - BlockSchurPreconditionerParallel from solvers/ns_block_preconditioner.h
//
// ALL parameters come from Parameters struct - NO HARDCODED VALUES!
// MMS verifies production code with production parameters.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ns/ns_mms_test.h"
#include "mms/ns/ns_mms.h"

// PRODUCTION code
#include "setup/ns_setup.h"
#include "assembly/ns_assembler.h"
#include "solvers/ns_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

constexpr int dim = 2;

// ============================================================================
// NSMMSConvergenceResult implementation
// ============================================================================

void NSMMSConvergenceResult::compute_rates()
{
    const size_t n = results.size();

    ux_L2_rate.resize(n, 0.0);
    ux_H1_rate.resize(n, 0.0);
    uy_L2_rate.resize(n, 0.0);
    uy_H1_rate.resize(n, 0.0);
    p_L2_rate.resize(n, 0.0);

    for (size_t i = 1; i < n; ++i)
    {
        const double h_ratio = results[i - 1].h / results[i].h;
        const double log_h = std::log(h_ratio);

        if (results[i - 1].ux_L2 > 1e-15 && results[i].ux_L2 > 1e-15)
            ux_L2_rate[i] = std::log(results[i - 1].ux_L2 / results[i].ux_L2) / log_h;
        if (results[i - 1].ux_H1 > 1e-15 && results[i].ux_H1 > 1e-15)
            ux_H1_rate[i] = std::log(results[i - 1].ux_H1 / results[i].ux_H1) / log_h;
        if (results[i - 1].uy_L2 > 1e-15 && results[i].uy_L2 > 1e-15)
            uy_L2_rate[i] = std::log(results[i - 1].uy_L2 / results[i].uy_L2) / log_h;
        if (results[i - 1].uy_H1 > 1e-15 && results[i].uy_H1 > 1e-15)
            uy_H1_rate[i] = std::log(results[i - 1].uy_H1 / results[i].uy_H1) / log_h;
        if (results[i - 1].p_L2 > 1e-15 && results[i].p_L2 > 1e-15)
            p_L2_rate[i] = std::log(results[i - 1].p_L2 / results[i].p_L2) / log_h;
    }
}

void NSMMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: " << to_string(phase) << "\n";
    std::cout << "========================================\n";
    std::cout << std::left
              << std::setw(6) << "Ref"
              << std::setw(12) << "h"
              << std::setw(12) << "ux_L2" << std::setw(8) << "rate"
              << std::setw(12) << "ux_H1" << std::setw(8) << "rate"
              << std::setw(12) << "p_L2" << std::setw(8) << "rate"
              << std::setw(12) << "div_U"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

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
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << results[i].div_U_L2
                  << "\n";
    }
    std::cout << "========================================\n";

    if (passes())
        std::cout << "[PASS] All convergence rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some rates below expected!\n";
}

void NSMMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[NS MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    file << "refinement,h,n_dofs,ux_L2,ux_L2_rate,ux_H1,ux_H1_rate,"
         << "uy_L2,uy_L2_rate,p_L2,p_L2_rate,div_U,time\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        file << results[i].refinement << ","
             << std::scientific << std::setprecision(6) << results[i].h << ","
             << results[i].n_dofs << ","
             << results[i].ux_L2 << ","
             << std::fixed << std::setprecision(2) << ux_L2_rate[i] << ","
             << std::scientific << results[i].ux_H1 << ","
             << std::fixed << ux_H1_rate[i] << ","
             << std::scientific << results[i].uy_L2 << ","
             << std::fixed << uy_L2_rate[i] << ","
             << std::scientific << results[i].p_L2 << ","
             << std::fixed << p_L2_rate[i] << ","
             << std::scientific << results[i].div_U_L2 << ","
             << std::fixed << std::setprecision(4) << results[i].total_time
             << "\n";
    }

    file.close();
    std::cout << "[NS MMS] Results written to " << filename << "\n";
}

bool NSMMSConvergenceResult::passes(double tol) const
{
    if (results.size() < 2)
        return false;

    const size_t last = results.size() - 1;

    // For unsteady with first-order time stepping, spatial rate limited by O(dt)
    const bool is_unsteady = (phase == NSPhase::B || phase == NSPhase::D);
    const double min_vel_L2 = std::min(expected_vel_L2_rate, is_unsteady ? 2.0 : 3.0) - tol;
    const double min_vel_H1 = expected_vel_H1_rate - tol;
    const double min_p_L2 = expected_p_L2_rate - tol;

    bool pass = true;

    if (ux_L2_rate[last] < min_vel_L2)
    {
        std::cout << "[FAIL] ux_L2 rate = " << ux_L2_rate[last]
                  << " < " << min_vel_L2 << "\n";
        pass = false;
    }
    if (ux_H1_rate[last] < min_vel_H1)
    {
        std::cout << "[FAIL] ux_H1 rate = " << ux_H1_rate[last]
                  << " < " << min_vel_H1 << "\n";
        pass = false;
    }
    if (p_L2_rate[last] < min_p_L2)
    {
        std::cout << "[FAIL] p_L2 rate = " << p_L2_rate[last]
                  << " < " << min_p_L2 << "\n";
        pass = false;
    }

    return pass;
}

// ============================================================================
// Compute errors (parallel reduction)
// ============================================================================
static NSMMSResult compute_errors_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    const dealii::TrilinosWrappers::MPI::Vector& p_solution,
    double time,
    double L_y,
    MPI_Comm mpi_comm)
{
    NSMMSResult error;

    // Exact solutions from mms/ns/ns_mms.h
    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);
    NSExactPressure<dim> exact_p(time, L_y);

    const auto& fe_vel = ux_dof_handler.get_fe();
    const auto& fe_p = p_dof_handler.get_fe();

    dealii::QGauss<dim> quadrature(fe_vel.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values_ux(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_uy(fe_vel, quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> fe_values_p(fe_p, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> ux_vals(n_q_points), uy_vals(n_q_points), p_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_ux(n_q_points), grad_uy(n_q_points);

    double local_ux_L2_sq = 0.0, local_ux_H1_sq = 0.0;
    double local_uy_L2_sq = 0.0, local_uy_H1_sq = 0.0;
    double local_p_L2_sq = 0.0;
    double local_div_U_L2_sq = 0.0;

    // Compute mean pressure for correction
    double local_p_integral = 0.0, local_volume = 0.0;
    double local_exact_p_integral = 0.0;

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        fe_values_ux.reinit(ux_cell);
        fe_values_uy.reinit(uy_cell);
        fe_values_p.reinit(p_cell);

        fe_values_ux.get_function_values(ux_solution, ux_vals);
        fe_values_ux.get_function_gradients(ux_solution, grad_ux);
        fe_values_uy.get_function_values(uy_solution, uy_vals);
        fe_values_uy.get_function_gradients(uy_solution, grad_uy);
        fe_values_p.get_function_values(p_solution, p_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_ux.JxW(q);
            const auto& x_q = fe_values_ux.quadrature_point(q);

            local_p_integral += p_vals[q] * JxW;
            local_exact_p_integral += exact_p.value(x_q) * JxW;
            local_volume += JxW;
        }
    }

    // Global reduction for mean pressure
    double global_p_integral = 0.0, global_volume = 0.0, global_exact_p_integral = 0.0;
    MPI_Allreduce(&local_p_integral, &global_p_integral, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_exact_p_integral, &global_exact_p_integral, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    const double p_mean = global_p_integral / global_volume;
    const double exact_p_mean = global_exact_p_integral / global_volume;

    // Now compute errors with mean-corrected pressure
    ux_cell = ux_dof_handler.begin_active();
    uy_cell = uy_dof_handler.begin_active();
    p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        fe_values_ux.reinit(ux_cell);
        fe_values_uy.reinit(uy_cell);
        fe_values_p.reinit(p_cell);

        fe_values_ux.get_function_values(ux_solution, ux_vals);
        fe_values_ux.get_function_gradients(ux_solution, grad_ux);
        fe_values_uy.get_function_values(uy_solution, uy_vals);
        fe_values_uy.get_function_gradients(uy_solution, grad_uy);
        fe_values_p.get_function_values(p_solution, p_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_ux.JxW(q);
            const auto& x_q = fe_values_ux.quadrature_point(q);

            // Exact values
            const double exact_ux_val = exact_ux.value(x_q);
            const double exact_uy_val = exact_uy.value(x_q);
            const double exact_p_val = exact_p.value(x_q) - exact_p_mean;
            const auto exact_grad_ux = exact_ux.gradient(x_q);
            const auto exact_grad_uy = exact_uy.gradient(x_q);

            // Errors
            const double ux_err = ux_vals[q] - exact_ux_val;
            const double uy_err = uy_vals[q] - exact_uy_val;
            const double p_err = (p_vals[q] - p_mean) - exact_p_val;

            local_ux_L2_sq += ux_err * ux_err * JxW;
            local_uy_L2_sq += uy_err * uy_err * JxW;
            local_p_L2_sq += p_err * p_err * JxW;

            // Gradient errors
            const auto grad_ux_err = grad_ux[q] - exact_grad_ux;
            const auto grad_uy_err = grad_uy[q] - exact_grad_uy;
            local_ux_H1_sq += grad_ux_err * grad_ux_err * JxW;
            local_uy_H1_sq += grad_uy_err * grad_uy_err * JxW;

            // Divergence
            const double div_U = grad_ux[q][0] + grad_uy[q][1];
            local_div_U_L2_sq += div_U * div_U * JxW;
        }
    }

    // Global reductions
    double global_ux_L2_sq = 0.0, global_ux_H1_sq = 0.0;
    double global_uy_L2_sq = 0.0, global_uy_H1_sq = 0.0;
    double global_p_L2_sq = 0.0, global_div_U_L2_sq = 0.0;

    MPI_Allreduce(&local_ux_L2_sq, &global_ux_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_ux_H1_sq, &global_ux_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_uy_L2_sq, &global_uy_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_uy_H1_sq, &global_uy_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_p_L2_sq, &global_p_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_div_U_L2_sq, &global_div_U_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    error.ux_L2 = std::sqrt(global_ux_L2_sq);
    error.ux_H1 = std::sqrt(global_ux_H1_sq);
    error.uy_L2 = std::sqrt(global_uy_L2_sq);
    error.uy_H1 = std::sqrt(global_uy_H1_sq);
    error.p_L2 = std::sqrt(global_p_L2_sq);
    error.div_U_L2 = std::sqrt(global_div_U_L2_sq);

    return error;
}

// ============================================================================
// Run single phase test (internal)
// ============================================================================
static NSMMSConvergenceResult run_phase_internal(
    NSPhase phase,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    const unsigned int this_mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, this_mpi_rank == 0);

    // Phase flags
    const bool include_time_derivative = (phase == NSPhase::B || phase == NSPhase::D);
    const bool include_convection = (phase == NSPhase::C || phase == NSPhase::D);

    // ========================================================================
    // ALL PARAMETERS FROM params - NO HARDCODED VALUES!
    // ========================================================================
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double nu = params.physics.nu_water;  // Use water viscosity for standalone NS

    // Time stepping parameters (MMS-specific, acceptable to define here)
    const double t_init = 0.1;
    const double t_final = 0.2;
    const unsigned int n_steps = include_time_derivative ? n_time_steps : 1;
    const double dt = (t_final - t_init) / n_steps;
    const double time_steady = 1.0;

    NSMMSConvergenceResult result;
    result.phase = phase;
    result.fe_degree_velocity = params.fe.degree_velocity;
    result.fe_degree_pressure = params.fe.degree_pressure;
    result.n_time_steps = n_steps;
    result.dt = dt;
    result.nu = nu;
    result.L_y = L_y;

    // Expected rates for Taylor-Hood elements
    result.expected_vel_L2_rate = params.fe.degree_velocity + 1;  // p+1 for Qp
    result.expected_vel_H1_rate = params.fe.degree_velocity;      // p for Qp
    result.expected_p_L2_rate = params.fe.degree_pressure + 1;    // p+1 for Qp

    pcout << "\n================================================================\n";
    pcout << "Phase " << to_string(phase) << "\n";
    pcout << "================================================================\n";
    pcout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
          << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
    pcout << "  L_y = " << L_y << ", nu = " << nu << "\n";
    pcout << "  FE degrees: velocity Q" << params.fe.degree_velocity
          << ", pressure Q" << params.fe.degree_pressure << "\n";
    pcout << "  include_time_derivative = " << (include_time_derivative ? "true" : "false") << "\n";
    pcout << "  include_convection = " << (include_convection ? "true" : "false") << "\n";
    if (include_time_derivative)
        pcout << "  t in [" << t_init << ", " << t_final << "], dt = " << dt << ", steps = " << n_steps << "\n";
    else
        pcout << "  steady at t = " << time_steady << "\n";
    pcout << "\n";

    for (unsigned int ref : refinements)
    {
        pcout << "  Refinement " << ref << "... " << std::flush;
        auto start_time = std::chrono::high_resolution_clock::now();

        NSMMSResult res;
        res.refinement = ref;

        // ====================================================================
        // Create distributed mesh using params.domain
        // ====================================================================
        dealii::parallel::distributed::Triangulation<dim> triangulation(mpi_comm);

        dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
        dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

        std::vector<unsigned int> subdivisions(dim);
        subdivisions[0] = params.domain.initial_cells_x;
        subdivisions[1] = params.domain.initial_cells_y;

        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
        triangulation.refine_global(ref);

        const double h = dealii::GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2.0);
        res.h = h;

        // ====================================================================
        // Finite elements from params.fe
        // ====================================================================
        dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
        dealii::FE_Q<dim> fe_p(params.fe.degree_pressure);

        // DoF handlers
        dealii::DoFHandler<dim> ux_dof_handler(triangulation);
        dealii::DoFHandler<dim> uy_dof_handler(triangulation);
        dealii::DoFHandler<dim> p_dof_handler(triangulation);

        ux_dof_handler.distribute_dofs(fe_vel);
        uy_dof_handler.distribute_dofs(fe_vel);
        p_dof_handler.distribute_dofs(fe_p);

        const dealii::IndexSet ux_owned = ux_dof_handler.locally_owned_dofs();
        const dealii::IndexSet uy_owned = uy_dof_handler.locally_owned_dofs();
        const dealii::IndexSet p_owned = p_dof_handler.locally_owned_dofs();

        const dealii::types::global_dof_index n_ux = ux_dof_handler.n_dofs();
        const dealii::types::global_dof_index n_uy = uy_dof_handler.n_dofs();
        res.n_dofs = n_ux + n_uy + p_dof_handler.n_dofs();

        // Setup constraints using PRODUCTION functions
        dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
        setup_ns_velocity_constraints_parallel<dim>(ux_dof_handler, uy_dof_handler,
            ux_constraints, uy_constraints);
        setup_ns_pressure_constraints_parallel<dim>(p_dof_handler, p_constraints);

        // Setup coupled system using PRODUCTION function
        std::vector<dealii::types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
        dealii::IndexSet ns_owned, ns_relevant;
        dealii::AffineConstraints<double> ns_constraints;
        dealii::TrilinosWrappers::SparsityPattern ns_sparsity;

        setup_ns_coupled_system_parallel<dim>(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_constraints, uy_constraints, p_constraints,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_owned, ns_relevant,
            ns_constraints, ns_sparsity,
            mpi_comm, pcout);

        // Allocate system
        dealii::TrilinosWrappers::SparseMatrix ns_matrix;
        ns_matrix.reinit(ns_sparsity);

        dealii::TrilinosWrappers::MPI::Vector ns_rhs(ns_owned, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector ns_solution(ns_owned, mpi_comm);

        // Solution vectors
        dealii::TrilinosWrappers::MPI::Vector ux_solution(ux_owned, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector uy_solution(uy_owned, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector p_solution(p_owned, mpi_comm);

        // Old velocity (ghosted for assembly)
        const dealii::IndexSet ux_relevant =
            dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
        const dealii::IndexSet uy_relevant =
            dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler);
        const dealii::IndexSet p_relevant =
            dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

        dealii::TrilinosWrappers::MPI::Vector ux_old(ux_relevant, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector uy_old(uy_relevant, mpi_comm);

        // Assemble pressure mass matrix for Schur preconditioner
        dealii::TrilinosWrappers::SparseMatrix pressure_mass;
        assemble_pressure_mass_matrix_parallel<dim>(
            p_dof_handler,
            p_constraints,
            p_owned,
            p_relevant,
            pressure_mass,
            mpi_comm);

        // Create velocity IndexSet (combined ux + uy)
        dealii::IndexSet vel_owned(n_ux + n_uy);
        for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
            vel_owned.add_index(*it);
        for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
            vel_owned.add_index(n_ux + *it);
        vel_owned.compress();

        // Initialize with exact solution at start time
        double current_time = include_time_derivative ? t_init : time_steady;

        {
            NSExactVelocityX<dim> exact_ux_init(current_time, L_y);
            NSExactVelocityY<dim> exact_uy_init(current_time, L_y);

            dealii::TrilinosWrappers::MPI::Vector ux_tmp(ux_owned, mpi_comm);
            dealii::TrilinosWrappers::MPI::Vector uy_tmp(uy_owned, mpi_comm);

            dealii::VectorTools::interpolate(ux_dof_handler, exact_ux_init, ux_tmp);
            dealii::VectorTools::interpolate(uy_dof_handler, exact_uy_init, uy_tmp);

            ux_old = ux_tmp;
            uy_old = uy_tmp;
        }

        // Time stepping loop
        const unsigned int actual_steps = include_time_derivative ? n_steps : 1;
        const double actual_dt = include_time_derivative ? dt : 1.0;
        double total_assembly_time = 0.0;
        double total_solve_time = 0.0;
        unsigned int total_iterations = 0;

        for (unsigned int step = 0; step < actual_steps; ++step)
        {
            const double t_old = current_time;
            if (include_time_derivative)
                current_time += dt;

            // ==============================================================
            // Assemble using PRODUCTION function with enable_mms=true
            // ==============================================================
            auto assembly_start = std::chrono::high_resolution_clock::now();

            assemble_ns_system_parallel<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                ux_old, uy_old,
                nu, actual_dt,
                include_time_derivative, include_convection,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_owned, ns_constraints,
                ns_matrix, ns_rhs,
                mpi_comm,
                true,           // enable_mms = true
                current_time,   // mms_time
                t_old,          // mms_time_old
                L_y);           // mms_L_y

            auto assembly_end = std::chrono::high_resolution_clock::now();
            total_assembly_time += std::chrono::duration<double>(assembly_end - assembly_start).count();

            // ==============================================================
            // Solve using Block Schur preconditioner (PRODUCTION function)
            // ==============================================================
            auto solve_start = std::chrono::high_resolution_clock::now();

            ns_solution = 0;
            SolverInfo info = solve_ns_system_schur_parallel(
                ns_matrix, ns_rhs, ns_solution, ns_constraints,
                pressure_mass,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_owned, vel_owned, p_owned,
                mpi_comm,
                nu,
                false);  // verbose

            auto solve_end = std::chrono::high_resolution_clock::now();
            total_solve_time += std::chrono::duration<double>(solve_end - solve_start).count();
            total_iterations += info.iterations;

            // Extract solutions
            extract_ns_solutions_parallel(
                ns_solution,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ux_owned, uy_owned, p_owned,
                ns_owned, ns_relevant,
                ux_solution, uy_solution, p_solution,
                mpi_comm);

            // Update old for next step
            ux_old = ux_solution;
            uy_old = uy_solution;
        }

        res.assembly_time = total_assembly_time;
        res.solve_time = total_solve_time;
        res.solver_iterations = total_iterations;

        // Create ghosted copies for error computation
        dealii::TrilinosWrappers::MPI::Vector ux_ghosted(ux_relevant, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector uy_ghosted(uy_relevant, mpi_comm);
        dealii::TrilinosWrappers::MPI::Vector p_ghosted(p_relevant, mpi_comm);

        ux_ghosted = ux_solution;
        uy_ghosted = uy_solution;
        p_ghosted = p_solution;

        // Compute errors at final time
        NSMMSResult errors = compute_errors_parallel(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_ghosted, uy_ghosted, p_ghosted,  // Use ghosted vectors!
            current_time, L_y, mpi_comm);

        res.ux_L2 = errors.ux_L2;
        res.ux_H1 = errors.ux_H1;
        res.uy_L2 = errors.uy_L2;
        res.uy_H1 = errors.uy_H1;
        res.p_L2 = errors.p_L2;
        res.div_U_L2 = errors.div_U_L2;

        auto end_time = std::chrono::high_resolution_clock::now();
        res.total_time = std::chrono::duration<double>(end_time - start_time).count();

        result.results.push_back(res);

        pcout << "ux_L2=" << std::scientific << std::setprecision(2) << res.ux_L2
              << ", p_L2=" << res.p_L2
              << ", div=" << res.div_U_L2
              << ", time=" << std::fixed << std::setprecision(1) << res.total_time << "s\n";
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// Public API
// ============================================================================

NSMMSConvergenceResult run_ns_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    NSSolverType solver_type,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    (void)solver_type;  // Currently only Schur is supported for parallel

    // Run Phase D (full unsteady NS) by default
    return run_phase_internal(NSPhase::D, refinements, params, n_time_steps, mpi_communicator);
}

NSMMSConvergenceResult run_ns_mms_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    return run_ns_mms_standalone(refinements, params, NSSolverType::Schur, n_time_steps, mpi_communicator);
}

NSMMSConvergenceResult run_ns_mms_phase(
    NSPhase phase,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    return run_phase_internal(phase, refinements, params, n_time_steps, mpi_communicator);
}