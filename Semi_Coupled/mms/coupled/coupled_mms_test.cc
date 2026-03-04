// ============================================================================
// mms/coupled/coupled_mms_test.cc - CoupledMMSConvergenceResult Implementation
//
// Unified implementation for ALL coupled MMS tests:
//   - CH_NS: Phase advection by velocity
//   - POISSON_MAGNETIZATION: φ ↔ M Picard iteration
//   - NS_MAGNETIZATION: NS with Kelvin force (M, H given)
//   - NS_POISSON_MAG: 3-way coupling (φ ↔ M Picard, then NS with Kelvin)
//   - FULL_SYSTEM: All four subsystems coupled
//
// This file should REPLACE the implementations currently scattered in
// poisson_mag_mms_test.cc. Include this file in your build instead.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// ============================================================================
// to_string for CoupledMMSLevel - ALL LEVELS
// ============================================================================
std::string to_string(CoupledMMSLevel level)
{
    switch (level)
    {
    case CoupledMMSLevel::CH_NS:                 return "CH_NS";
    case CoupledMMSLevel::POISSON_MAGNETIZATION: return "POISSON_MAGNETIZATION";
    case CoupledMMSLevel::NS_MAGNETIZATION:      return "NS_MAGNETIZATION";
    case CoupledMMSLevel::NS_POISSON_MAG:        return "NS_POISSON_MAG";
    case CoupledMMSLevel::MAG_CH:                return "MAG_CH";
    case CoupledMMSLevel::FULL_SYSTEM:           return "FULL_SYSTEM";
    default:                                     return "UNKNOWN";
    }
}

// ============================================================================
// compute_rates - works for all levels
// ============================================================================
void CoupledMMSConvergenceResult::compute_rates()
{
    const size_t n = results.size();
    if (n < 2) return;

    auto compute_rate = [](double e_fine, double e_coarse, double h_fine, double h_coarse)
    {
        if (e_fine < 1e-14 || e_coarse < 1e-14) return 0.0;
        return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
    };

    // Resize all rate vectors
    theta_L2_rate.resize(n, 0.0);
    theta_H1_rate.resize(n, 0.0);
    ux_L2_rate.resize(n, 0.0);
    ux_H1_rate.resize(n, 0.0);
    p_L2_rate.resize(n, 0.0);
    phi_L2_rate.resize(n, 0.0);
    phi_H1_rate.resize(n, 0.0);
    M_L2_rate.resize(n, 0.0);
    theta_Linf_rate.resize(n, 0.0);
    ux_Linf_rate.resize(n, 0.0);
    p_Linf_rate.resize(n, 0.0);
    phi_Linf_rate.resize(n, 0.0);
    M_Linf_rate.resize(n, 0.0);

    for (size_t i = 1; i < n; ++i)
    {
        const auto& fine = results[i];
        const auto& coarse = results[i - 1];

        theta_L2_rate[i] = compute_rate(fine.theta_L2, coarse.theta_L2, fine.h, coarse.h);
        theta_H1_rate[i] = compute_rate(fine.theta_H1, coarse.theta_H1, fine.h, coarse.h);
        ux_L2_rate[i] = compute_rate(fine.ux_L2, coarse.ux_L2, fine.h, coarse.h);
        ux_H1_rate[i] = compute_rate(fine.ux_H1, coarse.ux_H1, fine.h, coarse.h);
        p_L2_rate[i] = compute_rate(fine.p_L2, coarse.p_L2, fine.h, coarse.h);
        phi_L2_rate[i] = compute_rate(fine.phi_L2, coarse.phi_L2, fine.h, coarse.h);
        phi_H1_rate[i] = compute_rate(fine.phi_H1, coarse.phi_H1, fine.h, coarse.h);
        M_L2_rate[i] = compute_rate(fine.M_L2, coarse.M_L2, fine.h, coarse.h);
        theta_Linf_rate[i] = compute_rate(fine.theta_Linf, coarse.theta_Linf, fine.h, coarse.h);
        ux_Linf_rate[i] = compute_rate(fine.ux_Linf, coarse.ux_Linf, fine.h, coarse.h);
        p_Linf_rate[i] = compute_rate(fine.p_Linf, coarse.p_Linf, fine.h, coarse.h);
        phi_Linf_rate[i] = compute_rate(fine.phi_Linf, coarse.phi_Linf, fine.h, coarse.h);
        M_Linf_rate[i] = compute_rate(fine.M_Linf, coarse.M_Linf, fine.h, coarse.h);
    }
}

// ============================================================================
// passes - ALL LEVELS
// ============================================================================
bool CoupledMMSConvergenceResult::passes(double tol) const
{
    if (results.size() < 2) return false;

    bool pass = true;
    const double L2_min = expected_L2_rate - tol;
    const double H1_min = expected_H1_rate - tol;
    const double DG_min = expected_DG_rate - tol;
    const double P_min = 2.0 - tol;  // Q1 pressure gets rate 2

    for (size_t i = 1; i < results.size(); ++i)
    {
        switch (level)
        {
        case CoupledMMSLevel::CH_NS:
            if (theta_L2_rate[i] < L2_min) pass = false;
            if (theta_H1_rate[i] < H1_min) pass = false;
            if (ux_L2_rate[i] < L2_min) pass = false;
            if (ux_H1_rate[i] < H1_min) pass = false;
            break;

        case CoupledMMSLevel::POISSON_MAGNETIZATION:
            if (phi_L2_rate[i] < L2_min) pass = false;
            if (phi_H1_rate[i] < H1_min) pass = false;
            if (M_L2_rate[i] < DG_min) pass = false;
            break;

        case CoupledMMSLevel::NS_MAGNETIZATION:
            // NS solved, M/φ are exact (interpolated)
            if (ux_L2_rate[i] < L2_min) pass = false;
            if (ux_H1_rate[i] < H1_min) pass = false;
            if (p_L2_rate[i] < P_min) pass = false;
            break;

        case CoupledMMSLevel::NS_POISSON_MAG:
            // All three solved: NS + Poisson + Magnetization
            if (ux_L2_rate[i] < L2_min) pass = false;
            if (ux_H1_rate[i] < H1_min) pass = false;
            if (p_L2_rate[i] < P_min) pass = false;
            if (phi_L2_rate[i] < L2_min) pass = false;
            if (phi_H1_rate[i] < H1_min) pass = false;
            if (M_L2_rate[i] < DG_min) pass = false;
            break;

        case CoupledMMSLevel::MAG_CH:
            if (M_L2_rate[i] < DG_min) pass = false;
            break;

        case CoupledMMSLevel::FULL_SYSTEM:
            if (theta_L2_rate[i] < L2_min) pass = false;
            if (ux_L2_rate[i] < L2_min) pass = false;
            if (phi_L2_rate[i] < L2_min) pass = false;
            if (M_L2_rate[i] < DG_min) pass = false;
            break;

        default:
            pass = false;
            break;
        }
    }

    return pass;
}

// ============================================================================
// print - ALL LEVELS with proper tables
// ============================================================================
void CoupledMMSConvergenceResult::print() const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: " << to_string(level) << "\n";
    std::cout << "========================================\n";

    // -------------------------------------------------------------------------
    // Poisson table (POISSON_MAGNETIZATION, NS_POISSON_MAG, FULL_SYSTEM)
    // -------------------------------------------------------------------------
    if (level == CoupledMMSLevel::POISSON_MAGNETIZATION ||
        level == CoupledMMSLevel::NS_POISSON_MAG ||
        level == CoupledMMSLevel::FULL_SYSTEM)
    {
        std::cout << "\n--- Poisson Errors ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "φ_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "φ_L∞"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "φ_H1"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(77, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(12) << results[i].phi_L2
                      << std::setw(8) << std::fixed << std::setprecision(2) << phi_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].phi_Linf
                      << std::setw(8) << std::fixed << std::setprecision(2) << phi_Linf_rate[i]
                      << std::setw(12) << std::scientific << results[i].phi_H1
                      << std::setw(8) << std::fixed << phi_H1_rate[i] << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Magnetization table (POISSON_MAGNETIZATION, NS_POISSON_MAG, MAG_CH, FULL_SYSTEM)
    // -------------------------------------------------------------------------
    if (level == CoupledMMSLevel::POISSON_MAGNETIZATION ||
        level == CoupledMMSLevel::NS_POISSON_MAG ||
        level == CoupledMMSLevel::MAG_CH ||
        level == CoupledMMSLevel::FULL_SYSTEM)
    {
        std::cout << "\n--- Magnetization Errors ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "M_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "M_L∞"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(57, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(12) << results[i].M_L2
                      << std::setw(8) << std::fixed << std::setprecision(2) << M_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].M_Linf
                      << std::setw(8) << std::fixed << std::setprecision(2) << M_Linf_rate[i] << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // CH table (CH_NS, MAG_CH, FULL_SYSTEM)
    // -------------------------------------------------------------------------
    if (level == CoupledMMSLevel::CH_NS ||
        level == CoupledMMSLevel::MAG_CH ||
        level == CoupledMMSLevel::FULL_SYSTEM)
    {
        std::cout << "\n--- Cahn-Hilliard Errors ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "θ_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "θ_L∞"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "θ_H1"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(77, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(12) << results[i].theta_L2
                      << std::setw(8) << std::fixed << std::setprecision(2) << theta_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].theta_Linf
                      << std::setw(8) << std::fixed << std::setprecision(2) << theta_Linf_rate[i]
                      << std::setw(12) << std::scientific << results[i].theta_H1
                      << std::setw(8) << std::fixed << theta_H1_rate[i] << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // NS table (CH_NS, NS_MAGNETIZATION, NS_POISSON_MAG, FULL_SYSTEM)
    // -------------------------------------------------------------------------
    if (level == CoupledMMSLevel::CH_NS ||
        level == CoupledMMSLevel::NS_MAGNETIZATION ||
        level == CoupledMMSLevel::NS_POISSON_MAG ||
        level == CoupledMMSLevel::FULL_SYSTEM)
    {
        std::cout << "\n--- Navier-Stokes Errors ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(10) << "wall(s)"
                  << std::setw(12) << "U_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "U_L∞"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "U_H1"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "p_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "p_L∞"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(127, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2) << results[i].h
                      << std::setw(10) << std::fixed << std::setprecision(1) << results[i].total_time
                      << std::setw(12) << std::scientific << std::setprecision(2) << results[i].ux_L2
                      << std::setw(8) << std::fixed << std::setprecision(2) << ux_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].ux_Linf
                      << std::setw(8) << std::fixed << std::setprecision(2) << ux_Linf_rate[i]
                      << std::setw(12) << std::scientific << results[i].ux_H1
                      << std::setw(8) << std::fixed << ux_H1_rate[i]
                      << std::setw(12) << std::scientific << results[i].p_L2
                      << std::setw(8) << std::fixed << p_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].p_Linf
                      << std::setw(8) << std::fixed << std::setprecision(2) << p_Linf_rate[i] << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Final verdict
    // -------------------------------------------------------------------------
    std::cout << "\n========================================\n";
    if (passes())
        std::cout << "[PASS] All convergence rates within tolerance!\n";
    else
        std::cout << "[FAIL] Some rates below expected!\n";
    std::cout << "========================================\n";
}

// ============================================================================
// write_csv - writes appropriate columns based on test level
// ============================================================================
void CoupledMMSConvergenceResult::write_csv(const std::string& filename) const
{
    const int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (this_rank != 0) return;

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    // Header based on test level
    file << "refinement,h,n_dofs,wall_time";

    switch (level)
    {
    case CoupledMMSLevel::CH_NS:
        file << ",theta_L2,theta_L2_rate,theta_Linf,theta_Linf_rate,theta_H1,theta_H1_rate"
             << ",ux_L2,ux_L2_rate,ux_Linf,ux_Linf_rate,ux_H1,ux_H1_rate"
             << ",p_L2,p_L2_rate,p_Linf,p_Linf_rate";
        break;

    case CoupledMMSLevel::POISSON_MAGNETIZATION:
        file << ",phi_L2,phi_L2_rate,phi_Linf,phi_Linf_rate,phi_H1,phi_H1_rate"
             << ",M_L2,M_L2_rate,M_Linf,M_Linf_rate";
        break;

    case CoupledMMSLevel::NS_MAGNETIZATION:
        file << ",ux_L2,ux_L2_rate,ux_Linf,ux_Linf_rate,ux_H1,ux_H1_rate"
             << ",p_L2,p_L2_rate,p_Linf,p_Linf_rate";
        break;

    case CoupledMMSLevel::NS_POISSON_MAG:
        file << ",ux_L2,ux_L2_rate,ux_Linf,ux_Linf_rate,ux_H1,ux_H1_rate"
             << ",p_L2,p_L2_rate,p_Linf,p_Linf_rate"
             << ",phi_L2,phi_L2_rate,phi_Linf,phi_Linf_rate,phi_H1,phi_H1_rate"
             << ",M_L2,M_L2_rate,M_Linf,M_Linf_rate";
        break;

    case CoupledMMSLevel::MAG_CH:
        file << ",theta_L2,theta_L2_rate,theta_Linf,theta_Linf_rate,theta_H1,theta_H1_rate"
             << ",M_L2,M_L2_rate,M_Linf,M_Linf_rate";
        break;

    case CoupledMMSLevel::FULL_SYSTEM:
        file << ",theta_L2,theta_L2_rate,theta_Linf,theta_Linf_rate,theta_H1,theta_H1_rate"
             << ",ux_L2,ux_L2_rate,ux_Linf,ux_Linf_rate,ux_H1,ux_H1_rate"
             << ",p_L2,p_L2_rate,p_Linf,p_Linf_rate"
             << ",phi_L2,phi_L2_rate,phi_Linf,phi_Linf_rate,phi_H1,phi_H1_rate"
             << ",M_L2,M_L2_rate,M_Linf,M_Linf_rate";
        break;

    default:
        break;
    }
    file << "\n";

    // Data rows
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];

        file << r.refinement << ","
             << std::scientific << std::setprecision(6) << r.h << ","
             << r.n_dofs << ","
             << std::fixed << std::setprecision(2) << r.total_time;

        switch (level)
        {
        case CoupledMMSLevel::CH_NS:
            file << "," << std::scientific << r.theta_L2
                 << "," << std::fixed << std::setprecision(2) << theta_L2_rate[i]
                 << "," << std::scientific << r.theta_Linf
                 << "," << std::fixed << theta_Linf_rate[i]
                 << "," << std::scientific << r.theta_H1
                 << "," << std::fixed << theta_H1_rate[i]
                 << "," << std::scientific << r.ux_L2
                 << "," << std::fixed << ux_L2_rate[i]
                 << "," << std::scientific << r.ux_Linf
                 << "," << std::fixed << ux_Linf_rate[i]
                 << "," << std::scientific << r.ux_H1
                 << "," << std::fixed << ux_H1_rate[i]
                 << "," << std::scientific << r.p_L2
                 << "," << std::fixed << p_L2_rate[i]
                 << "," << std::scientific << r.p_Linf
                 << "," << std::fixed << p_Linf_rate[i];
            break;

        case CoupledMMSLevel::POISSON_MAGNETIZATION:
            file << "," << std::scientific << r.phi_L2
                 << "," << std::fixed << std::setprecision(2) << phi_L2_rate[i]
                 << "," << std::scientific << r.phi_Linf
                 << "," << std::fixed << phi_Linf_rate[i]
                 << "," << std::scientific << r.phi_H1
                 << "," << std::fixed << phi_H1_rate[i]
                 << "," << std::scientific << r.M_L2
                 << "," << std::fixed << M_L2_rate[i]
                 << "," << std::scientific << r.M_Linf
                 << "," << std::fixed << M_Linf_rate[i];
            break;

        case CoupledMMSLevel::NS_MAGNETIZATION:
            file << "," << std::scientific << r.ux_L2
                 << "," << std::fixed << std::setprecision(2) << ux_L2_rate[i]
                 << "," << std::scientific << r.ux_Linf
                 << "," << std::fixed << ux_Linf_rate[i]
                 << "," << std::scientific << r.ux_H1
                 << "," << std::fixed << ux_H1_rate[i]
                 << "," << std::scientific << r.p_L2
                 << "," << std::fixed << p_L2_rate[i]
                 << "," << std::scientific << r.p_Linf
                 << "," << std::fixed << p_Linf_rate[i];
            break;

        case CoupledMMSLevel::NS_POISSON_MAG:
            file << "," << std::scientific << r.ux_L2
                 << "," << std::fixed << std::setprecision(2) << ux_L2_rate[i]
                 << "," << std::scientific << r.ux_Linf
                 << "," << std::fixed << ux_Linf_rate[i]
                 << "," << std::scientific << r.ux_H1
                 << "," << std::fixed << ux_H1_rate[i]
                 << "," << std::scientific << r.p_L2
                 << "," << std::fixed << p_L2_rate[i]
                 << "," << std::scientific << r.p_Linf
                 << "," << std::fixed << p_Linf_rate[i]
                 << "," << std::scientific << r.phi_L2
                 << "," << std::fixed << phi_L2_rate[i]
                 << "," << std::scientific << r.phi_Linf
                 << "," << std::fixed << phi_Linf_rate[i]
                 << "," << std::scientific << r.phi_H1
                 << "," << std::fixed << phi_H1_rate[i]
                 << "," << std::scientific << r.M_L2
                 << "," << std::fixed << M_L2_rate[i]
                 << "," << std::scientific << r.M_Linf
                 << "," << std::fixed << M_Linf_rate[i];
            break;

        case CoupledMMSLevel::MAG_CH:
            file << "," << std::scientific << r.theta_L2
                 << "," << std::fixed << std::setprecision(2) << theta_L2_rate[i]
                 << "," << std::scientific << r.theta_Linf
                 << "," << std::fixed << theta_Linf_rate[i]
                 << "," << std::scientific << r.theta_H1
                 << "," << std::fixed << theta_H1_rate[i]
                 << "," << std::scientific << r.M_L2
                 << "," << std::fixed << M_L2_rate[i]
                 << "," << std::scientific << r.M_Linf
                 << "," << std::fixed << M_Linf_rate[i];
            break;

        case CoupledMMSLevel::FULL_SYSTEM:
            file << "," << std::scientific << r.theta_L2
                 << "," << std::fixed << std::setprecision(2) << theta_L2_rate[i]
                 << "," << std::scientific << r.theta_Linf
                 << "," << std::fixed << theta_Linf_rate[i]
                 << "," << std::scientific << r.theta_H1
                 << "," << std::fixed << theta_H1_rate[i]
                 << "," << std::scientific << r.ux_L2
                 << "," << std::fixed << ux_L2_rate[i]
                 << "," << std::scientific << r.ux_Linf
                 << "," << std::fixed << ux_Linf_rate[i]
                 << "," << std::scientific << r.ux_H1
                 << "," << std::fixed << ux_H1_rate[i]
                 << "," << std::scientific << r.p_L2
                 << "," << std::fixed << p_L2_rate[i]
                 << "," << std::scientific << r.p_Linf
                 << "," << std::fixed << p_Linf_rate[i]
                 << "," << std::scientific << r.phi_L2
                 << "," << std::fixed << phi_L2_rate[i]
                 << "," << std::scientific << r.phi_Linf
                 << "," << std::fixed << phi_Linf_rate[i]
                 << "," << std::scientific << r.phi_H1
                 << "," << std::fixed << phi_H1_rate[i]
                 << "," << std::scientific << r.M_L2
                 << "," << std::fixed << M_L2_rate[i]
                 << "," << std::scientific << r.M_Linf
                 << "," << std::fixed << M_Linf_rate[i];
            break;

        default:
            break;
        }
        file << "\n";
    }

    file.close();
    std::cout << "[MMS] Results written to " << filename << "\n";
}