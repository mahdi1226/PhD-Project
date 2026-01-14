// ============================================================================
// test_ns_phases.cc - Parallel NS MMS Phase-by-Phase Verification (A-D)
//
// Tests NS discretization in 4 phases:
//   Phase A: Steady Stokes - ν(T(U), T(V)) - (p, ∇·V) + (∇·U, q) = (f, V)
//   Phase B: Unsteady Stokes - adds (U/τ, V) time derivative
//   Phase C: Steady NS - adds B_h(U_old, U, V) convection
//   Phase D: Unsteady NS - full equation
//
// Uses Block Schur preconditioner for proper pressure convergence.
//
// Expected convergence rates for Q2-Q1 Taylor-Hood:
//   - Velocity L2: ~3.0 (p+1 for Q2)
//   - Velocity H1: ~2.0 (p for Q2)
//   - Pressure L2: ~2.0 (p+1 for Q1)
//
// Build: cmake --build . --target parallel_test_ns_phases
// Run:   mpirun -np 4 ./parallel_test_ns_phases [phase] [refs...]
//        phase = A, B, C, D, or ALL
//        refs = refinement levels (default: 3 4 5)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "ns_mms.h"
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
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

constexpr int dim = 2;

// ============================================================================
// Phase enumeration
// ============================================================================
enum class NSPhase
{
    A,  // Steady Stokes
    B,  // Unsteady Stokes
    C,  // Steady NS
    D   // Unsteady NS
};

std::string phase_name(NSPhase phase)
{
    switch (phase)
    {
        case NSPhase::A: return "A: Steady Stokes";
        case NSPhase::B: return "B: Unsteady Stokes";
        case NSPhase::C: return "C: Steady NS";
        case NSPhase::D: return "D: Unsteady NS";
    }
    return "Unknown";
}

// ============================================================================
// Error result structure
// ============================================================================
struct ErrorResult
{
    double h = 0.0;
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;
    double time = 0.0;
};

// ============================================================================
// Compute errors (parallel reduction)
// ============================================================================
ErrorResult compute_errors_parallel(
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
    ErrorResult error;

    // Create ghosted vectors for evaluation
    const dealii::IndexSet ux_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
    const dealii::IndexSet uy_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler);
    const dealii::IndexSet p_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

    dealii::TrilinosWrappers::MPI::Vector ux_ghosted(ux_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector uy_ghosted(uy_relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector p_ghosted(p_relevant, mpi_comm);

    ux_ghosted = ux_solution;
    uy_ghosted = uy_solution;
    p_ghosted = p_solution;

    // Exact solutions
    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);
    NSExactPressure<dim> exact_p(time, L_y);

    // Quadrature
    dealii::QGauss<dim> quadrature(ux_dof_handler.get_fe().degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(ux_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(uy_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> p_fe_values(p_dof_handler.get_fe(), quadrature,
        dealii::update_values);

    std::vector<double> ux_values(n_q_points), uy_values(n_q_points), p_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points), uy_gradients(n_q_points);

    double local_ux_L2_sq = 0.0, local_ux_H1_sq = 0.0;
    double local_uy_L2_sq = 0.0, local_uy_H1_sq = 0.0;
    double local_p_L2_sq = 0.0, local_div_U_L2_sq = 0.0;
    double local_p_mean_num = 0.0, local_p_mean_exact = 0.0, local_area = 0.0;

    // First pass: compute mean pressures
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        p_fe_values.reinit(p_cell);

        p_fe_values.get_function_values(p_ghosted, p_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            local_p_mean_num += p_values[q] * JxW;
            local_p_mean_exact += exact_p.value(x_q) * JxW;
            local_area += JxW;
        }
    }

    // Global reduction for means
    double global_p_mean_num = 0.0, global_p_mean_exact = 0.0, global_area = 0.0;
    MPI_Allreduce(&local_p_mean_num, &global_p_mean_num, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_p_mean_exact, &global_p_mean_exact, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_area, &global_area, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    const double p_mean_num = global_p_mean_num / global_area;
    const double p_mean_exact = global_p_mean_exact / global_area;

    // Second pass: compute errors
    ux_cell = ux_dof_handler.begin_active();
    uy_cell = uy_dof_handler.begin_active();
    p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        ux_fe_values.get_function_values(ux_ghosted, ux_values);
        uy_fe_values.get_function_values(uy_ghosted, uy_values);
        p_fe_values.get_function_values(p_ghosted, p_values);
        ux_fe_values.get_function_gradients(ux_ghosted, ux_gradients);
        uy_fe_values.get_function_gradients(uy_ghosted, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Velocity errors
            const double ux_exact = exact_ux.value(x_q);
            const double uy_exact = exact_uy.value(x_q);
            const dealii::Tensor<1, dim> grad_ux_exact = exact_ux.gradient(x_q);
            const dealii::Tensor<1, dim> grad_uy_exact = exact_uy.gradient(x_q);

            const double ux_err = ux_values[q] - ux_exact;
            const double uy_err = uy_values[q] - uy_exact;
            const dealii::Tensor<1, dim> grad_ux_err = ux_gradients[q] - grad_ux_exact;
            const dealii::Tensor<1, dim> grad_uy_err = uy_gradients[q] - grad_uy_exact;

            local_ux_L2_sq += ux_err * ux_err * JxW;
            local_uy_L2_sq += uy_err * uy_err * JxW;
            local_ux_H1_sq += (grad_ux_err * grad_ux_err) * JxW;
            local_uy_H1_sq += (grad_uy_err * grad_uy_err) * JxW;

            // Pressure error (zero-mean adjusted)
            const double p_exact = exact_p.value(x_q);
            const double p_err = (p_values[q] - p_mean_num) - (p_exact - p_mean_exact);
            local_p_L2_sq += p_err * p_err * JxW;

            // Divergence
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];
            local_div_U_L2_sq += div_U * div_U * JxW;
        }
    }

    // Global reduction for errors
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
// Run single phase test
// ============================================================================
int run_phase_test(
    NSPhase phase,
    const std::vector<unsigned int>& refinements,
    MPI_Comm mpi_comm)
{
    const unsigned int this_mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, this_mpi_rank == 0);

    // Phase flags
    const bool include_time_derivative = (phase == NSPhase::B || phase == NSPhase::D);
    const bool include_convection = (phase == NSPhase::C || phase == NSPhase::D);

    // Parameters
    const double L_x = 1.0, L_y = 1.0;
    const double nu = 1.0;
    const double t_init = 0.1;
    const double t_final = 0.2;
    const unsigned int n_steps = include_time_derivative ? 10 : 1;
    const double dt = (t_final - t_init) / n_steps;
    const double time_steady = 1.0;  // For steady phases

    pcout << "\n================================================================\n";
    pcout << "Phase " << phase_name(phase) << "\n";
    pcout << "================================================================\n";
    pcout << "  include_time_derivative = " << (include_time_derivative ? "true" : "false") << "\n";
    pcout << "  include_convection = " << (include_convection ? "true" : "false") << "\n";
    if (include_time_derivative)
        pcout << "  t in [" << t_init << ", " << t_final << "], dt = " << dt << ", steps = " << n_steps << "\n";
    else
        pcout << "  steady at t = " << time_steady << "\n";
    pcout << "\n";

    std::vector<ErrorResult> errors;
    std::vector<double> h_values;

    for (unsigned int ref : refinements)
    {
        pcout << "  Refinement " << ref << "... " << std::flush;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Create distributed mesh
        dealii::parallel::distributed::Triangulation<dim> triangulation(mpi_comm);
        dealii::GridGenerator::hyper_rectangle(triangulation,
            dealii::Point<dim>(0.0, 0.0), dealii::Point<dim>(L_x, L_y));
        triangulation.refine_global(ref);

        const double h = dealii::GridTools::minimal_cell_diameter(triangulation) / std::sqrt(2.0);
        h_values.push_back(h);

        // Finite elements: Q2-Q1 Taylor-Hood
        dealii::FE_Q<dim> fe_Q2(2);
        dealii::FE_Q<dim> fe_Q1(1);

        // DoF handlers
        dealii::DoFHandler<dim> ux_dof_handler(triangulation);
        dealii::DoFHandler<dim> uy_dof_handler(triangulation);
        dealii::DoFHandler<dim> p_dof_handler(triangulation);

        ux_dof_handler.distribute_dofs(fe_Q2);
        uy_dof_handler.distribute_dofs(fe_Q2);
        p_dof_handler.distribute_dofs(fe_Q1);

        const dealii::IndexSet ux_owned = ux_dof_handler.locally_owned_dofs();
        const dealii::IndexSet uy_owned = uy_dof_handler.locally_owned_dofs();
        const dealii::IndexSet p_owned = p_dof_handler.locally_owned_dofs();

        const dealii::types::global_dof_index n_ux = ux_dof_handler.n_dofs();
        const dealii::types::global_dof_index n_uy = uy_dof_handler.n_dofs();

        // Setup constraints
        dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
        setup_ns_velocity_constraints_parallel(ux_dof_handler, uy_dof_handler,
            ux_constraints, uy_constraints);
        setup_ns_pressure_constraints_parallel(p_dof_handler, p_constraints);

        // Setup coupled system
        std::vector<dealii::types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
        dealii::IndexSet ns_owned, ns_relevant;
        dealii::AffineConstraints<double> ns_constraints;
        dealii::TrilinosWrappers::SparsityPattern ns_sparsity;

        setup_ns_coupled_system_parallel(
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

        // ====================================================================
        // Assemble pressure mass matrix for Schur preconditioner
        // ====================================================================
        dealii::TrilinosWrappers::SparseMatrix pressure_mass;
        assemble_pressure_mass_matrix_parallel<dim>(
            p_dof_handler,
            p_constraints,
            p_owned,
            p_relevant,
            pressure_mass,
            mpi_comm);

        // ====================================================================
        // Create velocity IndexSet (combined ux + uy)
        // ====================================================================
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

        for (unsigned int step = 0; step < actual_steps; ++step)
        {
            const double t_old = current_time;
            if (include_time_derivative)
                current_time += dt;

            // Assemble using production assembler with enable_mms=true
            assemble_ns_system_parallel<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                ux_old, uy_old,
                nu, actual_dt,
                include_time_derivative, include_convection,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_owned, ns_constraints,
                ns_matrix, ns_rhs,
                mpi_comm,
                true,           // enable_mms
                current_time,   // mms_time
                t_old,          // mms_time_old
                L_y);           // mms_L_y

            // Debug: check matrix and RHS before solve (ALL ranks must call these)
            {
                int rank;
                MPI_Comm_rank(mpi_comm, &rank);
                double mat_norm = ns_matrix.frobenius_norm();  // collective!
                double rhs_norm = ns_rhs.l2_norm();            // collective!
                if (rank == 0)
                    std::cout << "[DEBUG] Before solve: matrix.frobenius_norm() = " << mat_norm
                              << ", rhs.l2_norm() = " << rhs_norm << std::endl;
            }


            // Solve using Block Schur preconditioner
            ns_solution = 0;
            solve_ns_system_schur_parallel(
                ns_matrix, ns_rhs, ns_solution, ns_constraints,
                pressure_mass,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_owned, vel_owned, p_owned,
                mpi_comm,
                nu,      // viscosity for Schur scaling
                true);  // verbose

            // Debug: check ns_solution directly
            {
                int rank;
                MPI_Comm_rank(mpi_comm, &rank);
                double local_sum = 0;
                for (auto it = ns_owned.begin(); it != ns_owned.end(); ++it)
                    local_sum += std::abs(ns_solution[*it]);
                double global_sum;
                MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
                if (rank == 0)
                    std::cout << "[DEBUG] ns_solution sum of |values|: " << global_sum << std::endl;
            }

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

        // Compute errors at final time
        ErrorResult err = compute_errors_parallel(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_solution, uy_solution, p_solution,
            current_time, L_y, mpi_comm);
        err.h = h;

        auto end_time = std::chrono::high_resolution_clock::now();
        err.time = std::chrono::duration<double>(end_time - start_time).count();

        errors.push_back(err);

        pcout << "ux_L2=" << std::scientific << std::setprecision(2) << err.ux_L2
              << ", p_L2=" << err.p_L2
              << ", div=" << err.div_U_L2
              << ", time=" << std::fixed << std::setprecision(1) << err.time << "s\n";
    }

    // Print convergence table
    pcout << "\n========================================\n";
    pcout << "Convergence Results: " << phase_name(phase) << "\n";
    pcout << "========================================\n";
    pcout << std::left << std::setw(6) << "Ref"
          << std::setw(12) << "h"
          << std::setw(12) << "ux_L2" << std::setw(8) << "rate"
          << std::setw(12) << "ux_H1" << std::setw(8) << "rate"
          << std::setw(12) << "p_L2" << std::setw(8) << "rate"
          << "\n";
    pcout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        double ux_L2_rate = 0, ux_H1_rate = 0, p_L2_rate = 0;
        if (i > 0)
        {
            ux_L2_rate = std::log(errors[i-1].ux_L2 / errors[i].ux_L2) /
                         std::log(h_values[i-1] / h_values[i]);
            ux_H1_rate = std::log(errors[i-1].ux_H1 / errors[i].ux_H1) /
                         std::log(h_values[i-1] / h_values[i]);
            p_L2_rate = std::log(errors[i-1].p_L2 / errors[i].p_L2) /
                        std::log(h_values[i-1] / h_values[i]);
        }

        pcout << std::left << std::setw(6) << refinements[i]
              << std::scientific << std::setprecision(2)
              << std::setw(12) << h_values[i]
              << std::setw(12) << errors[i].ux_L2
              << std::fixed << std::setprecision(2) << std::setw(8) << ux_L2_rate
              << std::scientific << std::setprecision(2)
              << std::setw(12) << errors[i].ux_H1
              << std::fixed << std::setprecision(2) << std::setw(8) << ux_H1_rate
              << std::scientific << std::setprecision(2)
              << std::setw(12) << errors[i].p_L2
              << std::fixed << std::setprecision(2) << std::setw(8) << p_L2_rate
              << "\n";
    }

    pcout << "========================================\n";

    // Check expected rates
    bool pass = true;
    const double tol = 0.3;

    if (errors.size() >= 2)
    {
        const size_t last = errors.size() - 1;
        double ux_L2_rate = std::log(errors[last-1].ux_L2 / errors[last].ux_L2) /
                            std::log(h_values[last-1] / h_values[last]);
        double ux_H1_rate = std::log(errors[last-1].ux_H1 / errors[last].ux_H1) /
                            std::log(h_values[last-1] / h_values[last]);
        double p_L2_rate = std::log(errors[last-1].p_L2 / errors[last].p_L2) /
                           std::log(h_values[last-1] / h_values[last]);

        // Expected rates (first-order time can limit spatial for unsteady)
        double expected_L2 = include_time_derivative ? 2.0 : 3.0;
        double expected_H1 = 2.0;
        double expected_p = 2.0;

        if (ux_L2_rate < expected_L2 - tol)
        {
            pcout << "[FAIL] ux_L2 rate = " << ux_L2_rate << " < " << expected_L2 - tol << "\n";
            pass = false;
        }
        if (ux_H1_rate < expected_H1 - tol)
        {
            pcout << "[FAIL] ux_H1 rate = " << ux_H1_rate << " < " << expected_H1 - tol << "\n";
            pass = false;
        }
        if (p_L2_rate < expected_p - tol)
        {
            pcout << "[FAIL] p_L2 rate = " << p_L2_rate << " < " << expected_p - tol << "\n";
            pass = false;
        }
    }

    if (pass)
        pcout << "\n[PASS] All convergence rates within expected bounds!\n";
    else
        pcout << "\n[FAIL] Some convergence rates below expected.\n";

    return pass ? 0 : 1;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int this_mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, this_mpi_rank == 0);

    // Parse arguments
    std::string phase_str = "A";
    std::vector<unsigned int> refinements = {3, 4, 5};

    if (argc >= 2)
        phase_str = argv[1];

    if (argc >= 3)
    {
        refinements.clear();
        for (int i = 2; i < argc; ++i)
            refinements.push_back(std::stoi(argv[i]));
    }

    pcout << "================================================================\n";
    pcout << "Parallel NS MMS Phase Testing\n";
    pcout << "================================================================\n";
    pcout << "MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
    pcout << "Phase: " << phase_str << "\n";
    pcout << "Refinements: ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n";

    int result = 0;

    if (phase_str == "A" || phase_str == "ALL")
        result |= run_phase_test(NSPhase::A, refinements, mpi_comm);

    if (phase_str == "B" || phase_str == "ALL")
        result |= run_phase_test(NSPhase::B, refinements, mpi_comm);

    if (phase_str == "C" || phase_str == "ALL")
        result |= run_phase_test(NSPhase::C, refinements, mpi_comm);

    if (phase_str == "D" || phase_str == "ALL")
        result |= run_phase_test(NSPhase::D, refinements, mpi_comm);

    if (phase_str != "A" && phase_str != "B" && phase_str != "C" &&
        phase_str != "D" && phase_str != "ALL")
    {
        pcout << "Usage: " << argv[0] << " [phase] [refs...]\n";
        pcout << "  phase = A (steady Stokes), B (unsteady Stokes),\n";
        pcout << "          C (steady NS), D (unsteady NS), ALL\n";
        pcout << "  refs = refinement levels (default: 3 4 5)\n";
        return 1;
    }

    return result;
}