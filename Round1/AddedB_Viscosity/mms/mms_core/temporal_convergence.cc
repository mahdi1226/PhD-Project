// ============================================================================
// mms/mms_core/temporal_convergence.cc - Temporal Convergence Implementation
//
// Verifies O(τ) convergence for backward Euler time integration.
//
// Each test fixes a fine spatial mesh and varies the number of time steps.
// Rate = log(e₁/e₂) / log(τ₁/τ₂), expected ≈ 1.0 for backward Euler.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "temporal_convergence.h"

// Subsystem test interfaces
#include "mms/ch/ch_mms_test.h"
#include "mms/ns/ns_mms_test.h"
#include "mms/magnetization/magnetization_mms_test.h"
#include "mms/coupled/coupled_mms_test.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// ============================================================================
// to_string
// ============================================================================

std::string to_string(TemporalTestLevel level)
{
    switch (level)
    {
    case TemporalTestLevel::CH_TEMPORAL:   return "CH_TEMPORAL";
    case TemporalTestLevel::NS_TEMPORAL:   return "NS_TEMPORAL";
    case TemporalTestLevel::MAG_TEMPORAL:  return "MAG_TEMPORAL";
    case TemporalTestLevel::FULL_TEMPORAL: return "FULL_TEMPORAL";
    default:                               return "UNKNOWN_TEMPORAL";
    }
}

// ============================================================================
// Helper: compute single temporal rate
// ============================================================================

static double compute_temporal_rate(double e_fine, double e_coarse,
                                    double dt_fine, double dt_coarse)
{
    if (e_coarse < 1e-15 || e_fine < 1e-15) return 0.0;
    return std::log(e_coarse / e_fine) / std::log(dt_coarse / dt_fine);
}

static void fill_temporal_rates(const std::vector<double>& errors,
                                const std::vector<double>& dt_values,
                                std::vector<double>& rates)
{
    rates.clear();
    for (size_t i = 1; i < errors.size(); ++i)
        rates.push_back(compute_temporal_rate(errors[i], errors[i - 1],
                                              dt_values[i], dt_values[i - 1]));
}

// ============================================================================
// TemporalConvergenceResult Implementation
// ============================================================================

void TemporalConvergenceResult::compute_rates()
{
    fill_temporal_rates(theta_L2, dt_values, theta_L2_rate);
    fill_temporal_rates(theta_H1, dt_values, theta_H1_rate);
    fill_temporal_rates(ux_L2, dt_values, ux_L2_rate);
    fill_temporal_rates(p_L2, dt_values, p_L2_rate);
    fill_temporal_rates(M_L2, dt_values, M_L2_rate);
    fill_temporal_rates(phi_L2, dt_values, phi_L2_rate);
}

bool TemporalConvergenceResult::passes(double tolerance) const
{
    const double min_rate = expected_rate - tolerance;
    bool pass = true;

    switch (level)
    {
    case TemporalTestLevel::CH_TEMPORAL:
        for (size_t i = 0; i < theta_L2_rate.size(); ++i)
        {
            if (theta_L2_rate[i] < min_rate) pass = false;
        }
        break;

    case TemporalTestLevel::NS_TEMPORAL:
        for (size_t i = 0; i < ux_L2_rate.size(); ++i)
        {
            if (ux_L2_rate[i] < min_rate) pass = false;
        }
        break;

    case TemporalTestLevel::MAG_TEMPORAL:
        for (size_t i = 0; i < M_L2_rate.size(); ++i)
        {
            if (M_L2_rate[i] < min_rate) pass = false;
        }
        break;

    case TemporalTestLevel::FULL_TEMPORAL:
        for (size_t i = 0; i < theta_L2_rate.size(); ++i)
        {
            if (theta_L2_rate[i] < min_rate) pass = false;
        }
        for (size_t i = 0; i < ux_L2_rate.size(); ++i)
        {
            if (ux_L2_rate[i] < min_rate) pass = false;
        }
        for (size_t i = 0; i < M_L2_rate.size(); ++i)
        {
            if (M_L2_rate[i] < min_rate) pass = false;
        }
        break;

    default:
        break;
    }

    return pass;
}

void TemporalConvergenceResult::print() const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::cout << "\n========================================\n";
    std::cout << "Temporal Convergence: " << to_string(level) << "\n";
    std::cout << "Fixed mesh: ref=" << refinement
              << ", h=" << std::scientific << std::setprecision(3) << h
              << ", DOFs=" << n_dofs << "\n";
    std::cout << "MPI ranks: " << n_mpi_ranks << "\n";
    std::cout << "Expected rate: " << std::fixed << std::setprecision(1) << expected_rate
              << " (backward Euler)\n";
    std::cout << "========================================\n";

    // CH temporal table
    if (!theta_L2.empty())
    {
        std::cout << "\n--- CH Temporal Errors ---\n";
        std::cout << std::left
            << std::setw(10) << "n_steps"
            << std::setw(12) << "dt"
            << std::setw(9) << "wall(s)"
            << std::setw(12) << "theta_L2"
            << std::setw(8) << "rate"
            << std::setw(12) << "theta_H1"
            << std::setw(8) << "rate"
            << "\n";
        std::cout << std::string(71, '-') << "\n";

        for (size_t i = 0; i < time_step_counts.size(); ++i)
        {
            std::cout << std::left << std::setw(10) << time_step_counts[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << dt_values[i]
                << std::fixed << std::setprecision(1)
                << std::setw(9) << wall_times[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << theta_L2[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? theta_L2_rate[i - 1] : 0.0)
                << std::scientific << std::setprecision(3)
                << std::setw(12) << theta_H1[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? theta_H1_rate[i - 1] : 0.0)
                << "\n";
        }
    }

    // NS temporal table
    if (!ux_L2.empty())
    {
        std::cout << "\n--- NS Temporal Errors ---\n";
        std::cout << std::left
            << std::setw(10) << "n_steps"
            << std::setw(12) << "dt"
            << std::setw(9) << "wall(s)"
            << std::setw(12) << "ux_L2"
            << std::setw(8) << "rate"
            << std::setw(12) << "p_L2"
            << std::setw(8) << "rate"
            << "\n";
        std::cout << std::string(71, '-') << "\n";

        for (size_t i = 0; i < time_step_counts.size(); ++i)
        {
            std::cout << std::left << std::setw(10) << time_step_counts[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << dt_values[i]
                << std::fixed << std::setprecision(1)
                << std::setw(9) << wall_times[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << ux_L2[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? ux_L2_rate[i - 1] : 0.0)
                << std::scientific << std::setprecision(3)
                << std::setw(12) << p_L2[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? p_L2_rate[i - 1] : 0.0)
                << "\n";
        }
    }

    // Magnetization temporal table
    if (!M_L2.empty())
    {
        std::cout << "\n--- Magnetization Temporal Errors ---\n";
        std::cout << std::left
            << std::setw(10) << "n_steps"
            << std::setw(12) << "dt"
            << std::setw(9) << "wall(s)"
            << std::setw(12) << "M_L2"
            << std::setw(8) << "rate"
            << "\n";
        std::cout << std::string(51, '-') << "\n";

        for (size_t i = 0; i < time_step_counts.size(); ++i)
        {
            std::cout << std::left << std::setw(10) << time_step_counts[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << dt_values[i]
                << std::fixed << std::setprecision(1)
                << std::setw(9) << wall_times[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << M_L2[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? M_L2_rate[i - 1] : 0.0)
                << "\n";
        }
    }

    // Poisson temporal table (only for full system)
    if (!phi_L2.empty())
    {
        std::cout << "\n--- Poisson Temporal Errors ---\n";
        std::cout << std::left
            << std::setw(10) << "n_steps"
            << std::setw(12) << "dt"
            << std::setw(12) << "phi_L2"
            << std::setw(8) << "rate"
            << "\n";
        std::cout << std::string(42, '-') << "\n";

        for (size_t i = 0; i < time_step_counts.size(); ++i)
        {
            std::cout << std::left << std::setw(10) << time_step_counts[i]
                << std::scientific << std::setprecision(3)
                << std::setw(12) << dt_values[i]
                << std::setw(12) << phi_L2[i]
                << std::fixed << std::setprecision(2)
                << std::setw(8) << (i > 0 ? phi_L2_rate[i - 1] : 0.0)
                << "\n";
        }
    }

    // Final verdict
    std::cout << "\n========================================\n";
    if (passes())
        std::cout << "[PASS] Temporal rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some temporal rates below expected!\n";
    std::cout << "========================================\n";
}

void TemporalConvergenceResult::write_csv(const std::string& filename) const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[TEMPORAL] Failed to open " << filename << " for writing\n";
        return;
    }

    // Header
    file << "n_time_steps,dt,wall_time";

    switch (level)
    {
    case TemporalTestLevel::CH_TEMPORAL:
        file << ",theta_L2,theta_L2_rate,theta_H1,theta_H1_rate";
        break;
    case TemporalTestLevel::NS_TEMPORAL:
        file << ",ux_L2,ux_L2_rate,p_L2,p_L2_rate";
        break;
    case TemporalTestLevel::MAG_TEMPORAL:
        file << ",M_L2,M_L2_rate";
        break;
    case TemporalTestLevel::FULL_TEMPORAL:
        file << ",theta_L2,theta_L2_rate"
             << ",ux_L2,ux_L2_rate,p_L2,p_L2_rate"
             << ",phi_L2,phi_L2_rate"
             << ",M_L2,M_L2_rate";
        break;
    default:
        break;
    }
    file << "\n";

    // Data rows
    for (size_t i = 0; i < time_step_counts.size(); ++i)
    {
        file << time_step_counts[i] << ","
             << std::scientific << std::setprecision(6) << dt_values[i] << ","
             << std::fixed << std::setprecision(2) << wall_times[i];

        auto rate_val = [](const std::vector<double>& rates, size_t idx) -> double
        {
            return idx > 0 && idx - 1 < rates.size() ? rates[idx - 1] : 0.0;
        };

        switch (level)
        {
        case TemporalTestLevel::CH_TEMPORAL:
            file << "," << std::scientific << theta_L2[i]
                 << "," << std::fixed << std::setprecision(2) << rate_val(theta_L2_rate, i)
                 << "," << std::scientific << theta_H1[i]
                 << "," << std::fixed << rate_val(theta_H1_rate, i);
            break;
        case TemporalTestLevel::NS_TEMPORAL:
            file << "," << std::scientific << ux_L2[i]
                 << "," << std::fixed << std::setprecision(2) << rate_val(ux_L2_rate, i)
                 << "," << std::scientific << p_L2[i]
                 << "," << std::fixed << rate_val(p_L2_rate, i);
            break;
        case TemporalTestLevel::MAG_TEMPORAL:
            file << "," << std::scientific << M_L2[i]
                 << "," << std::fixed << std::setprecision(2) << rate_val(M_L2_rate, i);
            break;
        case TemporalTestLevel::FULL_TEMPORAL:
            file << "," << std::scientific << theta_L2[i]
                 << "," << std::fixed << std::setprecision(2) << rate_val(theta_L2_rate, i)
                 << "," << std::scientific << ux_L2[i]
                 << "," << std::fixed << rate_val(ux_L2_rate, i)
                 << "," << std::scientific << p_L2[i]
                 << "," << std::fixed << rate_val(p_L2_rate, i)
                 << "," << std::scientific << phi_L2[i]
                 << "," << std::fixed << rate_val(phi_L2_rate, i)
                 << "," << std::scientific << M_L2[i]
                 << "," << std::fixed << rate_val(M_L2_rate, i);
            break;
        default:
            break;
        }
        file << "\n";
    }

    file.close();
    std::cout << "[TEMPORAL] Results written to " << filename << "\n";
}

// ============================================================================
// CH Temporal Test
// ============================================================================

TemporalConvergenceResult run_ch_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    Parameters mutable_params = params;
    mutable_params.enable_mms = true;

    TemporalConvergenceResult result;
    result.level = TemporalTestLevel::CH_TEMPORAL;
    result.refinement = refinement;
    result.expected_rate = 1.0;
    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    const double T = 0.1;  // Time interval [0.1, 0.2]

    if (this_rank == 0)
    {
        std::cout << "\n=== CH Temporal Convergence ===\n"
                  << "Fixed refinement: " << refinement << "\n"
                  << "Time step counts:";
        for (auto n : time_step_counts) std::cout << " " << n;
        std::cout << "\n===============================\n\n";
    }

    for (const auto n_steps : time_step_counts)
    {
        const double dt = T / n_steps;

        if (this_rank == 0)
            std::cout << "[CH_TEMPORAL] n_steps=" << n_steps
                      << ", dt=" << std::scientific << std::setprecision(3) << dt << "...\n";

        CHMMSResult r = run_ch_mms_single(
            refinement, mutable_params, CHSolverType::GMRES_AMG,
            n_steps, mpi_communicator);

        result.time_step_counts.push_back(n_steps);
        result.dt_values.push_back(dt);
        result.wall_times.push_back(r.total_time);
        result.theta_L2.push_back(r.theta_L2);
        result.theta_H1.push_back(r.theta_H1);

        // Record mesh info from first run
        if (result.h == 0.0)
        {
            result.h = r.h;
            result.n_dofs = r.n_dofs;
        }

        if (this_rank == 0)
            std::cout << "  theta_L2=" << std::scientific << std::setprecision(3)
                      << r.theta_L2 << ", time=" << std::fixed << std::setprecision(1)
                      << r.total_time << "s\n";
    }

    result.total_wall_time = 0.0;
    for (auto t : result.wall_times) result.total_wall_time += t;

    result.compute_rates();
    return result;
}

// ============================================================================
// NS Temporal Test
// ============================================================================

TemporalConvergenceResult run_ns_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    Parameters mutable_params = params;
    mutable_params.enable_mms = true;

    TemporalConvergenceResult result;
    result.level = TemporalTestLevel::NS_TEMPORAL;
    result.refinement = refinement;
    result.expected_rate = 1.0;
    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n=== NS Temporal Convergence ===\n"
                  << "Fixed refinement: " << refinement << "\n"
                  << "Time step counts:";
        for (auto n : time_step_counts) std::cout << " " << n;
        std::cout << "\n===============================\n\n";
    }

    const double T = 0.1;  // Time interval [0.1, 0.2]

    for (const auto n_steps : time_step_counts)
    {
        const double dt = T / n_steps;

        if (this_rank == 0)
            std::cout << "[NS_TEMPORAL] n_steps=" << n_steps
                      << ", dt=" << std::scientific << std::setprecision(3) << dt << "...\n";

        // Run NS with single refinement level
        std::vector<unsigned int> single_ref = {refinement};
        NSMMSConvergenceResult ns_result = run_ns_mms_standalone(
            single_ref, mutable_params, n_steps, mpi_communicator);

        if (!ns_result.results.empty())
        {
            const auto& r = ns_result.results[0];

            result.time_step_counts.push_back(n_steps);
            result.dt_values.push_back(dt);
            result.wall_times.push_back(r.total_time);
            result.ux_L2.push_back(r.ux_L2);
            result.p_L2.push_back(r.p_L2);

            if (result.h == 0.0)
            {
                result.h = r.h;
                result.n_dofs = r.n_dofs;
            }

            if (this_rank == 0)
                std::cout << "  ux_L2=" << std::scientific << std::setprecision(3)
                          << r.ux_L2 << ", p_L2=" << r.p_L2
                          << ", time=" << std::fixed << std::setprecision(1)
                          << r.total_time << "s\n";
        }
    }

    result.total_wall_time = 0.0;
    for (auto t : result.wall_times) result.total_wall_time += t;

    result.compute_rates();
    return result;
}

// ============================================================================
// Magnetization Temporal Test
// ============================================================================

TemporalConvergenceResult run_magnetization_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    Parameters mutable_params = params;
    mutable_params.enable_mms = true;

    TemporalConvergenceResult result;
    result.level = TemporalTestLevel::MAG_TEMPORAL;
    result.refinement = refinement;
    result.expected_rate = 1.0;
    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n=== Magnetization Temporal Convergence ===\n"
                  << "Fixed refinement: " << refinement << "\n"
                  << "Time step counts:";
        for (auto n : time_step_counts) std::cout << " " << n;
        std::cout << "\n==========================================\n\n";
    }

    const double T = 0.1;  // Time interval [0.1, 0.2]

    for (const auto n_steps : time_step_counts)
    {
        const double dt = T / n_steps;

        if (this_rank == 0)
            std::cout << "[MAG_TEMPORAL] n_steps=" << n_steps
                      << ", dt=" << std::scientific << std::setprecision(3) << dt << "...\n";

        // Run Magnetization with single refinement level
        // Note: run_magnetization_mms_standalone uses its own hardcoded n_time_steps.
        // For temporal convergence we need an overload that accepts n_time_steps.
        // Use run_magnetization_mms_single_temporal() which we add as an overload.
        std::vector<unsigned int> single_ref = {refinement};
        MagMMSConvergenceResult mag_result = run_magnetization_mms_standalone(
            single_ref, mutable_params, MagSolverType::Direct,
            n_steps, mpi_communicator);

        if (!mag_result.results.empty())
        {
            const auto& r = mag_result.results[0];

            result.time_step_counts.push_back(n_steps);
            result.dt_values.push_back(dt);
            result.wall_times.push_back(r.total_time);
            result.M_L2.push_back(r.M_L2);

            if (result.h == 0.0)
            {
                result.h = r.h;
                result.n_dofs = r.n_dofs;
            }

            if (this_rank == 0)
                std::cout << "  M_L2=" << std::scientific << std::setprecision(3)
                          << r.M_L2 << ", time=" << std::fixed << std::setprecision(1)
                          << r.total_time << "s\n";
        }
    }

    result.total_wall_time = 0.0;
    for (auto t : result.wall_times) result.total_wall_time += t;

    result.compute_rates();
    return result;
}

// ============================================================================
// Full System Temporal Test
// ============================================================================

TemporalConvergenceResult run_full_system_temporal_mms(
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator)
{
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    TemporalConvergenceResult result;
    result.level = TemporalTestLevel::FULL_TEMPORAL;
    result.refinement = refinement;
    result.expected_rate = 1.0;
    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

    if (this_rank == 0)
    {
        std::cout << "\n=== Full System Temporal Convergence ===\n"
                  << "Fixed refinement: " << refinement << "\n"
                  << "Time step counts:";
        for (auto n : time_step_counts) std::cout << " " << n;
        std::cout << "\n========================================\n\n";
    }

    for (const auto n_steps : time_step_counts)
    {
        const double T = 0.1;
        const double dt = T / n_steps;

        if (this_rank == 0)
            std::cout << "[FULL_TEMPORAL] n_steps=" << n_steps
                      << ", dt=" << std::scientific << std::setprecision(3) << dt << "...\n";

        // Run full system with single refinement level
        std::vector<unsigned int> single_ref = {refinement};
        CoupledMMSConvergenceResult coupled_result = run_full_system_mms(
            single_ref, params, n_steps, mpi_communicator);

        if (!coupled_result.results.empty())
        {
            const auto& r = coupled_result.results[0];

            result.time_step_counts.push_back(n_steps);
            result.dt_values.push_back(dt);
            result.wall_times.push_back(r.total_time);
            result.theta_L2.push_back(r.theta_L2);
            result.theta_H1.push_back(r.theta_H1);
            result.ux_L2.push_back(r.ux_L2);
            result.p_L2.push_back(r.p_L2);
            result.phi_L2.push_back(r.phi_L2);
            result.M_L2.push_back(r.M_L2);

            if (result.h == 0.0)
            {
                result.h = r.h;
                result.n_dofs = r.n_dofs;
            }

            if (this_rank == 0)
                std::cout << "  theta_L2=" << std::scientific << std::setprecision(3)
                          << r.theta_L2
                          << ", ux_L2=" << r.ux_L2
                          << ", M_L2=" << r.M_L2
                          << ", time=" << std::fixed << std::setprecision(1)
                          << r.total_time << "s\n";
        }
    }

    result.total_wall_time = 0.0;
    for (auto t : result.wall_times) result.total_wall_time += t;

    result.compute_rates();
    return result;
}

// ============================================================================
// Main dispatcher
// ============================================================================

TemporalConvergenceResult run_temporal_mms_test(
    TemporalTestLevel level,
    unsigned int refinement,
    const Parameters& params,
    const std::vector<unsigned int>& time_step_counts,
    MPI_Comm mpi_communicator)
{
    switch (level)
    {
    case TemporalTestLevel::CH_TEMPORAL:
        return run_ch_temporal_mms(refinement, params, time_step_counts, mpi_communicator);
    case TemporalTestLevel::NS_TEMPORAL:
        return run_ns_temporal_mms(refinement, params, time_step_counts, mpi_communicator);
    case TemporalTestLevel::MAG_TEMPORAL:
        return run_magnetization_temporal_mms(refinement, params, time_step_counts, mpi_communicator);
    case TemporalTestLevel::FULL_TEMPORAL:
        return run_full_system_temporal_mms(refinement, params, time_step_counts, mpi_communicator);
    default:
        std::cerr << "[ERROR] Unknown temporal test level\n";
        return TemporalConvergenceResult{};
    }
}
