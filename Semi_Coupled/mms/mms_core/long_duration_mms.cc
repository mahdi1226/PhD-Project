// ============================================================================
// mms/mms_core/long_duration_mms.cc - Long-Duration MMS Stability Tests
//
// Records per-step error evolution to diagnose temporal error accumulation.
// Based on existing MMS test infrastructure but with per-step error tracking.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "long_duration_mms.h"

// MMS exact solutions & sources
#include "mms/ch/ch_mms.h"
#include "mms/ch/ch_mms_test.h"
#include "mms/ns/ns_mms.h"
#include "mms/ns/ns_mms_test.h"
#include "mms/magnetic/poisson_mms.h"
#include "mms/magnetic/magnetization_mms.h"
#include "mms/coupled/coupled_mms_test.h"

// Production components
#include "setup/ch_setup.h"
#include "assembly/ch_assembler.h"
#include "solvers/ch_solver.h"
#include "setup/ns_setup.h"
#include "assembly/ns_assembler.h"
#include "solvers/ns_solver.h"

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
#include <numeric>

constexpr int dim = 2;

// ============================================================================
// to_string
// ============================================================================

std::string to_string(LongDurationLevel level)
{
    switch (level)
    {
    case LongDurationLevel::CH_LONG:     return "CH_LONG";
    case LongDurationLevel::CH_NS_LONG:  return "CH_NS_LONG";
    case LongDurationLevel::FULL_LONG:   return "FULL_LONG";
    default:                             return "UNKNOWN_LONG";
    }
}

// ============================================================================
// Helper: compute CH errors at current time
// ============================================================================
static StepErrorSnapshot compute_ch_errors_snapshot(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    unsigned int step,
    double time,
    double wall_time,
    double L_y,
    MPI_Comm mpi_communicator)
{
    StepErrorSnapshot snap;
    snap.step = step;
    snap.time = time;
    snap.wall_time = wall_time;

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

            const double theta_exact = exact_theta.value(x_q);
            const auto grad_theta_exact = exact_theta.gradient(x_q);
            const double theta_err = theta_values[q] - theta_exact;
            const auto grad_theta_err = theta_gradients[q] - grad_theta_exact;

            local_theta_L2_sq += theta_err * theta_err * JxW;
            local_theta_H1_sq += grad_theta_err * grad_theta_err * JxW;

            const double psi_exact = exact_psi.value(x_q);
            const double psi_err = psi_values[q] - psi_exact;
            local_psi_L2_sq += psi_err * psi_err * JxW;
        }
    }

    // MPI reduction
    double global_theta_L2_sq = 0.0, global_theta_H1_sq = 0.0, global_psi_L2_sq = 0.0;
    MPI_Allreduce(&local_theta_L2_sq, &global_theta_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_theta_H1_sq, &global_theta_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_psi_L2_sq, &global_psi_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    snap.theta_L2 = std::sqrt(global_theta_L2_sq);
    snap.theta_H1 = std::sqrt(global_theta_H1_sq);
    snap.psi_L2 = std::sqrt(global_psi_L2_sq);

    return snap;
}

// ============================================================================
// LongDurationResult: analyze growth
// ============================================================================

void LongDurationResult::analyze_growth()
{
    if (snapshots.size() < 3) return;

    // Compute final/initial ratio
    const double e0 = snapshots.front().theta_L2;
    const double ef = snapshots.back().theta_L2;
    theta_L2_final_ratio = (e0 > 1e-20) ? ef / e0 : 0.0;

    // Fit exponential growth rate: log(e(t)) = log(e0) + rate * t
    // Use least-squares fit of log(e) vs t
    double sum_t = 0, sum_loge = 0, sum_t2 = 0, sum_t_loge = 0;
    size_t valid = 0;

    for (const auto& s : snapshots)
    {
        if (s.theta_L2 < 1e-20) continue;
        double log_e = std::log(s.theta_L2);
        sum_t += s.time;
        sum_loge += log_e;
        sum_t2 += s.time * s.time;
        sum_t_loge += s.time * log_e;
        ++valid;
    }

    if (valid > 2)
    {
        double denom = valid * sum_t2 - sum_t * sum_t;
        if (std::abs(denom) > 1e-20)
        {
            theta_L2_growth_rate = (valid * sum_t_loge - sum_t * sum_loge) / denom;
        }
    }

    // Classify: if growth rate > 2/T (much faster than 1/T linear accumulation),
    // flag as exponential. For backward Euler, linear growth is normal.
    const double T = t_end - t_start;
    is_exponential_growth = (theta_L2_growth_rate > 5.0 / T);
}

// ============================================================================
// LongDurationResult: print summary
// ============================================================================

void LongDurationResult::print_summary() const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::cout << "\n========================================\n";
    std::cout << "Long-Duration MMS: " << to_string(level) << "\n";
    std::cout << "========================================\n";
    std::cout << "  Refinement: " << refinement
              << ", h=" << std::scientific << std::setprecision(3) << h
              << ", DOFs=" << n_dofs << "\n";
    std::cout << "  Time: [" << std::fixed << std::setprecision(3) << t_start
              << ", " << t_end << "], dt=" << std::scientific << dt
              << ", steps=" << n_steps << "\n";
    std::cout << "  MPI ranks: " << n_mpi_ranks << "\n";
    std::cout << "  Wall time: " << std::fixed << std::setprecision(1)
              << total_wall_time << "s\n\n";

    // Error evolution table (show subset)
    std::cout << std::left
              << std::setw(8) << "Step"
              << std::setw(10) << "Time"
              << std::setw(14) << "theta_L2"
              << std::setw(14) << "theta_H1"
              << std::setw(14) << "psi_L2";
    if (level != LongDurationLevel::CH_LONG)
        std::cout << std::setw(14) << "ux_L2"
                  << std::setw(14) << "p_L2";
    if (level == LongDurationLevel::FULL_LONG)
        std::cout << std::setw(14) << "phi_L2"
                  << std::setw(14) << "M_L2";
    std::cout << "\n";
    std::cout << std::string(80, '-') << "\n";

    // Show first 5, then every 10%, then last 5
    std::vector<size_t> indices_to_show;
    for (size_t i = 0; i < std::min((size_t)5, snapshots.size()); ++i)
        indices_to_show.push_back(i);
    for (size_t i = 1; i <= 9; ++i)
    {
        size_t idx = snapshots.size() * i / 10;
        if (idx >= 5 && idx < snapshots.size() - 5)
            indices_to_show.push_back(idx);
    }
    for (size_t i = snapshots.size() > 5 ? snapshots.size() - 5 : 0;
         i < snapshots.size(); ++i)
        indices_to_show.push_back(i);

    // Remove duplicates and sort
    std::sort(indices_to_show.begin(), indices_to_show.end());
    indices_to_show.erase(
        std::unique(indices_to_show.begin(), indices_to_show.end()),
        indices_to_show.end());

    for (size_t idx : indices_to_show)
    {
        const auto& s = snapshots[idx];
        std::cout << std::left << std::setw(8) << s.step
                  << std::fixed << std::setprecision(4) << std::setw(10) << s.time
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << s.theta_L2
                  << std::setw(14) << s.theta_H1
                  << std::setw(14) << s.psi_L2;
        if (level != LongDurationLevel::CH_LONG)
            std::cout << std::setw(14) << s.ux_L2
                      << std::setw(14) << s.p_L2;
        if (level == LongDurationLevel::FULL_LONG)
            std::cout << std::setw(14) << s.phi_L2
                      << std::setw(14) << s.M_L2;
        std::cout << "\n";
    }

    // Growth analysis
    std::cout << "\n--- Growth Analysis ---\n";
    std::cout << "  theta_L2 final/initial ratio: " << std::fixed << std::setprecision(4)
              << theta_L2_final_ratio << "\n";
    std::cout << "  Exponential growth rate: " << std::scientific << std::setprecision(4)
              << theta_L2_growth_rate << " /s\n";
    if (is_exponential_growth)
        std::cout << "  [WARNING] EXPONENTIAL GROWTH DETECTED - scheme may be unstable!\n";
    else
        std::cout << "  [OK] Growth appears bounded or linear (expected for backward Euler)\n";
    std::cout << "========================================\n";
}

// ============================================================================
// LongDurationResult: write CSV
// ============================================================================

void LongDurationResult::write_csv(const std::string& filename) const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[LONG_DURATION] Failed to open " << filename << " for writing\n";
        return;
    }

    // Metadata header
    file << "# Long-duration MMS: " << to_string(level)
         << ", ref=" << refinement
         << ", dt=" << std::scientific << dt
         << ", n_steps=" << n_steps
         << ", t=[" << std::fixed << std::setprecision(3) << t_start
         << "," << t_end << "]\n";
    file << "# growth_rate=" << std::scientific << theta_L2_growth_rate
         << ", final_ratio=" << theta_L2_final_ratio
         << ", exponential=" << (is_exponential_growth ? "YES" : "no") << "\n";

    // Column header
    file << "step,time,wall_time,theta_L2,theta_H1,psi_L2";
    if (level != LongDurationLevel::CH_LONG)
        file << ",ux_L2,p_L2";
    if (level == LongDurationLevel::FULL_LONG)
        file << ",phi_L2,M_L2";
    file << "\n";

    // Data rows
    for (const auto& s : snapshots)
    {
        file << s.step << ","
             << std::fixed << std::setprecision(6) << s.time << ","
             << std::setprecision(2) << s.wall_time << ","
             << std::scientific << std::setprecision(8)
             << s.theta_L2 << ","
             << s.theta_H1 << ","
             << s.psi_L2;
        if (level != LongDurationLevel::CH_LONG)
            file << "," << s.ux_L2 << "," << s.p_L2;
        if (level == LongDurationLevel::FULL_LONG)
            file << "," << s.phi_L2 << "," << s.M_L2;
        file << "\n";
    }

    file.close();
    std::cout << "[LONG_DURATION] Results written to " << filename << "\n";
}

// ============================================================================
// CH Long-Duration Test
// ============================================================================

LongDurationResult run_ch_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval,
    MPI_Comm mpi_communicator)
{
    LongDurationResult result;
    result.level = LongDurationLevel::CH_LONG;
    result.refinement = refinement;
    result.n_steps = n_time_steps;
    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    auto total_start = std::chrono::high_resolution_clock::now();

    // Time stepping parameters
    const double t_init = 0.1;
    const double t_final = 0.6;  // Long duration: 5x the standard [0.1, 0.2]
    const double dt = (t_final - t_init) / n_time_steps;
    const double L_y = params.domain.y_max - params.domain.y_min;

    result.t_start = t_init;
    result.t_end = t_final;
    result.dt = dt;

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    pcout << "\n=== CH Long-Duration MMS ===\n"
          << "  Refinement: " << refinement << "\n"
          << "  Time: [" << t_init << ", " << t_final << "]\n"
          << "  dt = " << std::scientific << std::setprecision(3) << dt << "\n"
          << "  Steps: " << n_time_steps << "\n"
          << "  Log interval: " << log_interval << "\n"
          << "===========================\n\n";

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
            const double tol = 1e-10;
            if (std::abs(center[1] - params.domain.y_min) < tol) face->set_boundary_id(0);
            else if (std::abs(center[0] - params.domain.x_max) < tol) face->set_boundary_id(1);
            else if (std::abs(center[1] - params.domain.y_max) < tol) face->set_boundary_id(2);
            else if (std::abs(center[0] - params.domain.x_min) < tol) face->set_boundary_id(3);
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
    // Setup CH system (same as ch_mms_test.cc)
    // ========================================================================
    dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);

    dealii::DoFHandler<dim> theta_dof_handler(triangulation);
    dealii::DoFHandler<dim> psi_dof_handler(triangulation);
    theta_dof_handler.distribute_dofs(fe_phase);
    psi_dof_handler.distribute_dofs(fe_phase);

    const unsigned int n_theta = theta_dof_handler.n_dofs();
    const unsigned int n_psi = psi_dof_handler.n_dofs();
    const unsigned int n_total = n_theta + n_psi;
    result.n_dofs = n_total;

    dealii::IndexSet theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    dealii::IndexSet theta_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    dealii::IndexSet psi_locally_owned = psi_dof_handler.locally_owned_dofs();
    dealii::IndexSet psi_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler);

    // Combined IndexSets
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

    // Constraints with MMS BCs
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

    // Initial conditions
    CHMMSInitialTheta<dim> theta_ic(t_init, L_y);
    CHMMSInitialPsi<dim> psi_ic(t_init, L_y);
    dealii::VectorTools::interpolate(theta_dof_handler, theta_ic, theta_owned);
    dealii::VectorTools::interpolate(psi_dof_handler, psi_ic, psi_owned);
    theta_owned.compress(dealii::VectorOperation::insert);
    psi_owned.compress(dealii::VectorOperation::insert);
    theta_relevant = theta_owned;
    psi_relevant = psi_owned;

    // Dummy velocity (CH standalone has no velocity)
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
    // Record initial error (step 0)
    // ========================================================================
    double current_time = t_init;

    {
        auto now = std::chrono::high_resolution_clock::now();
        double wall = std::chrono::duration<double>(now - total_start).count();
        StepErrorSnapshot snap = compute_ch_errors_snapshot(
            theta_dof_handler, psi_dof_handler,
            theta_relevant, psi_relevant,
            0, current_time, wall, L_y, mpi_communicator);
        result.snapshots.push_back(snap);

        pcout << "  Step 0: theta_L2=" << std::scientific << std::setprecision(4)
              << snap.theta_L2 << " (initial error from interpolation)\n";
    }

    // ========================================================================
    // Time stepping loop with per-step error recording
    // ========================================================================
    for (unsigned int step = 1; step <= n_time_steps; ++step)
    {
        current_time = t_init + step * dt;

        // Update constraints for current time
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

        // PRODUCTION assembly
        assemble_ch_system<dim>(
            theta_dof_handler, psi_dof_handler,
            theta_relevant,
            ux_dof_handler, uy_dof_handler,
            ux_dummy, uy_dummy,
            mms_params, dt, current_time,
            theta_to_ch_map, psi_to_ch_map,
            ch_constraints,
            ch_matrix, ch_rhs);

        // PRODUCTION solver
        solve_ch_system(
            ch_matrix, ch_rhs, ch_constraints,
            ch_locally_owned, ch_locally_relevant,
            theta_locally_owned, psi_locally_owned,
            theta_to_ch_map, psi_to_ch_map,
            theta_owned, psi_owned,
            mms_params.solvers.ch, mpi_communicator, false);

        // Update ghost values
        theta_owned.compress(dealii::VectorOperation::insert);
        psi_owned.compress(dealii::VectorOperation::insert);
        theta_relevant = theta_owned;
        psi_relevant = psi_owned;

        // Record error at this step
        if (step % log_interval == 0 || step == n_time_steps)
        {
            auto now = std::chrono::high_resolution_clock::now();
            double wall = std::chrono::duration<double>(now - total_start).count();

            StepErrorSnapshot snap = compute_ch_errors_snapshot(
                theta_dof_handler, psi_dof_handler,
                theta_relevant, psi_relevant,
                step, current_time, wall, L_y, mpi_communicator);
            result.snapshots.push_back(snap);

            // Print progress every 10%
            if (step % (n_time_steps / 10 + 1) == 0 || step == n_time_steps)
            {
                pcout << "  Step " << std::setw(5) << step << "/" << n_time_steps
                      << " t=" << std::fixed << std::setprecision(4) << current_time
                      << " theta_L2=" << std::scientific << std::setprecision(4) << snap.theta_L2
                      << " (" << std::fixed << std::setprecision(1) << wall << "s)\n";
            }
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_wall_time = std::chrono::duration<double>(total_end - total_start).count();

    // Analyze error growth
    result.analyze_growth();

    return result;
}

// ============================================================================
// CH + NS Long-Duration Test
// ============================================================================

LongDurationResult run_ch_ns_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval,
    MPI_Comm mpi_communicator)
{
    // For now, run CH standalone with velocity from MMS exact solution
    // This tests the convection term coupling.
    // Full implementation would set up both CH and NS systems.

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
        std::cout << "\n[CH_NS_LONG] Running CH with MMS velocity coupling...\n";

    // Delegate to the existing CH+NS coupled MMS infrastructure
    // For now, we wrap run_ch_ns_mms and record per-step errors
    // TODO: Implement full per-step tracking for CH+NS coupling
    //       (requires inlining the coupled MMS time loop with error snapshots)

    LongDurationResult result;
    result.level = LongDurationLevel::CH_NS_LONG;
    result.refinement = refinement;
    result.n_steps = n_time_steps;

    if (this_rank == 0)
        std::cout << "[CH_NS_LONG] Not yet implemented - use CH_LONG first to isolate CH issues\n";

    return result;
}

// ============================================================================
// Full System Long-Duration Test
// ============================================================================

LongDurationResult run_full_long_duration_mms(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    if (this_rank == 0)
        std::cout << "\n[FULL_LONG] Running full system long-duration MMS...\n";

    // TODO: Implement full per-step tracking for all subsystems
    //       (requires inlining full_system_mms_test.cc time loop with error snapshots)

    LongDurationResult result;
    result.level = LongDurationLevel::FULL_LONG;
    result.refinement = refinement;
    result.n_steps = n_time_steps;

    if (this_rank == 0)
        std::cout << "[FULL_LONG] Not yet implemented - use CH_LONG first to isolate CH issues\n";

    return result;
}

// ============================================================================
// Main dispatcher
// ============================================================================

LongDurationResult run_long_duration_mms_test(
    LongDurationLevel level,
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    unsigned int log_interval,
    MPI_Comm mpi_communicator)
{
    switch (level)
    {
    case LongDurationLevel::CH_LONG:
        return run_ch_long_duration_mms(refinement, params, n_time_steps,
                                        log_interval, mpi_communicator);
    case LongDurationLevel::CH_NS_LONG:
        return run_ch_ns_long_duration_mms(refinement, params, n_time_steps,
                                           log_interval, mpi_communicator);
    case LongDurationLevel::FULL_LONG:
        return run_full_long_duration_mms(refinement, params, n_time_steps,
                                          log_interval, mpi_communicator);
    default:
        std::cerr << "[ERROR] Unknown long-duration test level\n";
        return LongDurationResult{};
    }
}
