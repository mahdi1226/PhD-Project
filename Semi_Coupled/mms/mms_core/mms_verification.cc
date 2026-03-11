// ============================================================================
// mms/mms_core/mms_verification.cc - MMS Verification Implementation (PARALLEL)
//
// STANDALONE TESTS ONLY - Coupled tests are in coupled_mms_test.h/cc
//
// Standalone: CH_STANDALONE, NS_STANDALONE
// Monolithic magnetics: see mms/magnetic/magnetic_mms_test.h
//
// For coupled tests, see mms/coupled/coupled_mms_test.h:
//   - CH_MAGNETIC, MAGNETIC_NS, NS_CH, FULL_SYSTEM
// ============================================================================

#include "mms_verification.h"

// MMS test modules (each calls production code internally)
#include "mms/ch/ch_mms_test.h"
#include "mms/ns/ns_mms_test.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// ============================================================================
// Helper: compute convergence rate
// ============================================================================

static double compute_single_rate(double e_fine, double e_coarse,
                                  double h_fine, double h_coarse)
{
    if (e_coarse < 1e-15 || e_fine < 1e-15) return 0.0;
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

static void fill_rates(const std::vector<double>& errors,
                       const std::vector<double>& h_values,
                       std::vector<double>& rates)
{
    rates.clear();
    for (size_t i = 1; i < errors.size(); ++i)
        rates.push_back(compute_single_rate(errors[i], errors[i - 1],
                                            h_values[i], h_values[i - 1]));
}

// ============================================================================
// MMSConvergenceResult Implementation
// ============================================================================

void MMSConvergenceResult::compute_rates()
{
    fill_rates(theta_L2, h_values, theta_L2_rate);
    fill_rates(theta_H1, h_values, theta_H1_rate);
    fill_rates(psi_L2, h_values, psi_L2_rate);
    fill_rates(theta_Linf, h_values, theta_Linf_rate);
    fill_rates(psi_Linf, h_values, psi_Linf_rate);
    fill_rates(ux_L2, h_values, ux_L2_rate);
    fill_rates(ux_H1, h_values, ux_H1_rate);
    fill_rates(uy_L2, h_values, uy_L2_rate);
    fill_rates(uy_H1, h_values, uy_H1_rate);
    fill_rates(p_L2, h_values, p_L2_rate);
    fill_rates(ux_Linf, h_values, ux_Linf_rate);
    fill_rates(uy_Linf, h_values, uy_Linf_rate);
    fill_rates(p_Linf, h_values, p_Linf_rate);
    fill_rates(div_u_L2, h_values, div_u_L2_rate);
}

void MMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: " << to_string(level) << "\n";
    std::cout << "MPI ranks: " << n_mpi_ranks << "\n";
    std::cout << "Total wall time: " << std::fixed << std::setprecision(1)
              << total_wall_time << " s\n";
    std::cout << "========================================\n";

    switch (level)
    {
    case MMSLevel::CH_STANDALONE:
        print_ch_table();
        break;
    case MMSLevel::NS_STANDALONE:
        print_ns_table();
        break;
    default:
        std::cout << "Unknown test level\n";
    }

    std::cout << "========================================\n";
    if (passes())
        std::cout << "[PASS] Convergence rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some rates below expected!\n";
}

void MMSConvergenceResult::print_ch_table() const
{
    std::cout << "\n--- CH Errors ---\n";
    if (theta_L2.empty()) {
        std::cout << "[WARNING] No CH data to display\n";
        return;
    }
    std::cout << std::left
        << std::setw(5) << "Ref"
        << std::setw(10) << "h"
        << std::setw(9) << "wall(s)"
        << std::setw(10) << "theta_L2"
        << std::setw(6) << "rate"
        << std::setw(10) << "theta_Linf"
        << std::setw(6) << "rate"
        << std::setw(10) << "theta_H1"
        << std::setw(6) << "rate"
        << std::setw(10) << "psi_L2"
        << std::setw(6) << "rate"
        << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        std::cout << std::left << std::setw(5) << refinements[i]
            << std::scientific << std::setprecision(2)
            << std::setw(10) << h_values[i]
            << std::fixed << std::setprecision(1)
            << std::setw(9) << wall_times[i]
            << std::scientific << std::setprecision(2)
            << std::setw(10) << theta_L2[i]
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 ? theta_L2_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << (i < theta_Linf.size() ? theta_Linf[i] : 0.0)
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 && !theta_Linf_rate.empty() ? theta_Linf_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << theta_H1[i]
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 ? theta_H1_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << (i < psi_L2.size() ? psi_L2[i] : 0.0)
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 && !psi_L2_rate.empty() ? psi_L2_rate[i - 1] : 0.0)
            << "\n";
    }
}

void MMSConvergenceResult::print_ns_table() const
{
    std::cout << "\n--- NS Errors ---\n";
    if (ux_L2.empty()) {
        std::cout << "[WARNING] No NS data to display\n";
        return;
    }
    std::cout << std::left
        << std::setw(5) << "Ref"
        << std::setw(10) << "h"
        << std::setw(9) << "wall(s)"
        << std::setw(10) << "ux_L2"
        << std::setw(6) << "rate"
        << std::setw(10) << "ux_Linf"
        << std::setw(6) << "rate"
        << std::setw(10) << "ux_H1"
        << std::setw(6) << "rate"
        << std::setw(10) << "p_L2"
        << std::setw(6) << "rate"
        << std::setw(10) << "p_Linf"
        << std::setw(6) << "rate"
        << "\n";
    std::cout << std::string(104, '-') << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        std::cout << std::left << std::setw(5) << refinements[i]
            << std::scientific << std::setprecision(2)
            << std::setw(10) << h_values[i]
            << std::fixed << std::setprecision(1)
            << std::setw(9) << wall_times[i]
            << std::scientific << std::setprecision(2)
            << std::setw(10) << ux_L2[i]
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 ? ux_L2_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << (i < ux_Linf.size() ? ux_Linf[i] : 0.0)
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 && !ux_Linf_rate.empty() ? ux_Linf_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << ux_H1[i]
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 ? ux_H1_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << p_L2[i]
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 ? p_L2_rate[i - 1] : 0.0)
            << std::scientific << std::setprecision(2)
            << std::setw(10) << (i < p_Linf.size() ? p_Linf[i] : 0.0)
            << std::fixed << std::setprecision(2)
            << std::setw(6) << (i > 0 && !p_Linf_rate.empty() ? p_Linf_rate[i - 1] : 0.0)
            << "\n";
    }
}

bool MMSConvergenceResult::passes(double tolerance) const
{
    if (refinements.size() < 2) return true;

    const double L2_min = expected_L2_rate - tolerance;
    const double H1_min = expected_H1_rate - tolerance;

    bool pass = true;

    switch (level)
    {
    case MMSLevel::CH_STANDALONE:
        for (size_t i = 0; i < theta_L2_rate.size(); ++i)
        {
            if (theta_L2_rate[i] < L2_min) pass = false;
            if (theta_H1_rate[i] < H1_min) pass = false;
        }
        break;

    case MMSLevel::NS_STANDALONE:
        for (size_t i = 0; i < ux_L2_rate.size(); ++i)
        {
            if (ux_L2_rate[i] < L2_min) pass = false;
            if (ux_H1_rate[i] < H1_min) pass = false;
        }
        break;

    default:
        break;
    }

    return pass;
}

// ============================================================================
// Standalone Test Runners
// ============================================================================

static MMSConvergenceResult run_ch_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    params.enable_mms = true;

    CHMMSConvergenceResult ch_result = run_ch_mms_standalone(
        refinements, params, CHSolverType::GMRES_AMG, n_time_steps, mpi_communicator);

    MMSConvergenceResult result;
    result.level = MMSLevel::CH_STANDALONE;
    result.fe_degree = params.fe.degree_phase;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = params.fe.degree_phase + 1;
    result.expected_H1_rate = params.fe.degree_phase;

    for (const auto& r : ch_result.results)
    {
        result.refinements.push_back(r.refinement);
        result.h_values.push_back(r.h);
        result.n_dofs.push_back(r.n_dofs);
        result.wall_times.push_back(r.total_time);
        result.theta_L2.push_back(r.theta_L2);
        result.theta_H1.push_back(r.theta_H1);
        result.psi_L2.push_back(r.psi_L2);
        result.theta_Linf.push_back(r.theta_Linf);
        result.psi_Linf.push_back(r.psi_Linf);
    }

    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    result.total_wall_time = 0.0;
    for (const auto& t : result.wall_times)
        result.total_wall_time += t;

    result.compute_rates();
    return result;
}

static MMSConvergenceResult run_ns_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    params.enable_mms = true;

    NSMMSConvergenceResult ns_result = run_ns_mms_standalone(
        refinements, params, n_time_steps, mpi_communicator);

    MMSConvergenceResult result;
    result.level = MMSLevel::NS_STANDALONE;
    result.fe_degree = params.fe.degree_velocity;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = params.fe.degree_velocity + 1;
    result.expected_H1_rate = params.fe.degree_velocity;

    for (const auto& r : ns_result.results)
    {
        result.refinements.push_back(r.refinement);
        result.h_values.push_back(r.h);
        result.n_dofs.push_back(r.n_dofs);
        result.wall_times.push_back(r.total_time);
        result.ux_L2.push_back(r.ux_L2);
        result.ux_H1.push_back(r.ux_H1);
        result.uy_L2.push_back(r.uy_L2);
        result.uy_H1.push_back(r.uy_H1);
        result.p_L2.push_back(r.p_L2);
        result.div_u_L2.push_back(r.div_U_L2);
        result.ux_Linf.push_back(r.ux_Linf);
        result.uy_Linf.push_back(r.uy_Linf);
        result.p_Linf.push_back(r.p_Linf);
    }

    result.n_mpi_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    result.total_wall_time = 0.0;
    for (const auto& t : result.wall_times)
        result.total_wall_time += t;

    result.compute_rates();
    return result;
}

// ============================================================================
// Main dispatcher
// ============================================================================

MMSConvergenceResult run_mms_test(
    MMSLevel level,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    Parameters mutable_params = params;

    switch (level)
    {
    case MMSLevel::CH_STANDALONE:
        return run_ch_standalone(refinements, mutable_params, n_time_steps, mpi_communicator);

    case MMSLevel::NS_STANDALONE:
        return run_ns_standalone(refinements, mutable_params, n_time_steps, mpi_communicator);

    default:
        std::cerr << "[ERROR] Unknown MMS level: " << static_cast<int>(level) << "\n";
        return MMSConvergenceResult{};
    }
}

// ============================================================================
// Write results to CSV
// ============================================================================

void MMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    file << "refinement,h,n_dofs,wall_time";

    switch (level)
    {
    case MMSLevel::CH_STANDALONE:
        file << ",theta_L2,theta_L2_rate,theta_Linf,theta_Linf_rate"
             << ",theta_H1,theta_H1_rate,psi_L2,psi_L2_rate";
        break;
    case MMSLevel::NS_STANDALONE:
        file << ",ux_L2,ux_L2_rate,ux_Linf,ux_Linf_rate"
             << ",ux_H1,ux_H1_rate,p_L2,p_L2_rate,p_Linf,p_Linf_rate,div_u_L2";
        break;
    default:
        break;
    }
    file << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        file << refinements[i] << ","
            << std::scientific << std::setprecision(6) << h_values[i] << ","
            << n_dofs[i] << ","
            << std::fixed << std::setprecision(4) << wall_times[i];

        auto safe_rate = [](const std::vector<double>& rates, size_t idx) -> double
        {
            return idx > 0 && idx - 1 < rates.size() ? rates[idx - 1] : 0.0;
        };

        switch (level)
        {
        case MMSLevel::CH_STANDALONE:
            file << "," << std::scientific << theta_L2[i]
                << "," << std::fixed << std::setprecision(2) << safe_rate(theta_L2_rate, i)
                << "," << std::scientific << (i < theta_Linf.size() ? theta_Linf[i] : 0.0)
                << "," << std::fixed << safe_rate(theta_Linf_rate, i)
                << "," << std::scientific << theta_H1[i]
                << "," << std::fixed << safe_rate(theta_H1_rate, i)
                << "," << std::scientific << psi_L2[i]
                << "," << std::fixed << safe_rate(psi_L2_rate, i);
            break;
        case MMSLevel::NS_STANDALONE:
            file << "," << std::scientific << ux_L2[i]
                << "," << std::fixed << safe_rate(ux_L2_rate, i)
                << "," << std::scientific << (i < ux_Linf.size() ? ux_Linf[i] : 0.0)
                << "," << std::fixed << safe_rate(ux_Linf_rate, i)
                << "," << std::scientific << ux_H1[i]
                << "," << std::fixed << safe_rate(ux_H1_rate, i)
                << "," << std::scientific << p_L2[i]
                << "," << std::fixed << safe_rate(p_L2_rate, i)
                << "," << std::scientific << (i < p_Linf.size() ? p_Linf[i] : 0.0)
                << "," << std::fixed << safe_rate(p_Linf_rate, i)
                << "," << std::scientific << div_u_L2[i];
            break;
        default:
            break;
        }
        file << "\n";
    }

    file.close();
    std::cout << "[MMS] Results written to " << filename << "\n";
}

// ============================================================================
// to_string helper
// ============================================================================

std::string to_string(MMSLevel level)
{
    switch (level)
    {
    case MMSLevel::CH_STANDALONE:           return "CH_STANDALONE";
    case MMSLevel::NS_STANDALONE:           return "NS_STANDALONE";
    default:                                return "UNKNOWN";
    }
}
