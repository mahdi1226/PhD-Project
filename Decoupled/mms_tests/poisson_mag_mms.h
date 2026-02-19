// ============================================================================
// mms_tests/poisson_mag_mms.h — Coupled Poisson↔Magnetization MMS
//
// COUPLED SOURCE TERM:
//   The standalone Poisson source is f = −Δφ*.
//   When M is present, the Poisson PDE becomes:
//     (∇φ, ∇X) = (h_a − M, ∇X)
//   so the MMS source must also account for ∇·M*:
//     f_coupled = −Δφ* − ∇·M*
//
// EXACT SOLUTIONS (from subsystem MMS headers):
//   φ* = t · cos(πx) · cos(πy/L_y)
//   M* = t · [sin(πx)·sin(πy/L_y), cos(πx)·sin(πy/L_y)]
//
// EXPECTED RATES:
//   φ: L2 = p+1, H1 = p   (CG Q_p)
//   M: L2 = 2              (DG Q1)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_MAG_MMS_H
#define POISSON_MAG_MMS_H

#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"
#include "utilities/parameters.h"

#include <mpi.h>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Coupled Poisson MMS source: f = −Δφ* − ∇·M*
//
// −Δφ* = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
// ∇·M* = ∂Mx*/∂x + ∂My*/∂y
//       = t·π·cos(πx)·sin(πy/L_y) + t·(π/L_y)·cos(πx)·cos(πy/L_y)
// ============================================================================
template <int dim>
double compute_poisson_mms_source_coupled(
    const dealii::Point<dim>& pt,
    double time,
    double L_y)
{
    const double x = pt[0];
    const double y = pt[1];

    // −Δφ*
    const double neg_lap_phi = time * M_PI * M_PI
                               * (1.0 + 1.0 / (L_y * L_y))
                               * std::cos(M_PI * x)
                               * std::cos(M_PI * y / L_y);

    // ∇·M*
    const double dMx_dx = time * M_PI * std::cos(M_PI * x)
                          * std::sin(M_PI * y / L_y);
    const double dMy_dy = time * (M_PI / L_y) * std::cos(M_PI * x)
                          * std::cos(M_PI * y / L_y);

    return neg_lap_phi - (dMx_dx + dMy_dy);
}

// ============================================================================
// Single-refinement result
// ============================================================================
struct PoissonMagMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs     = 0;
    double h = 0.0;

    // Poisson errors
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;

    // Magnetization errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2  = 0.0;

    // Picard iterations (last time step)
    unsigned int picard_iters = 0;

    // Wall time
    double time_s = 0.0;
};

// ============================================================================
// Convergence result across refinement levels
// ============================================================================
struct PoissonMagMMSConvergenceResult
{
    std::vector<PoissonMagMMSResult> results;

    // Computed rates (index 0 = 0, meaningful from index 1)
    std::vector<double> phi_L2_rate;
    std::vector<double> phi_H1_rate;
    std::vector<double> M_L2_rate;

    // Expected rates
    double expected_phi_L2_rate = 3.0;   // Q2 → O(h³)
    double expected_phi_H1_rate = 2.0;   // Q2 → O(h²)
    double expected_M_L2_rate   = 2.0;   // DG-Q1 → O(h²)

    void compute_rates()
    {
        const size_t n = results.size();
        phi_L2_rate.assign(n, 0.0);
        phi_H1_rate.assign(n, 0.0);
        M_L2_rate.assign(n, 0.0);

        for (size_t i = 1; i < n; ++i)
        {
            const auto& f = results[i];
            const auto& c = results[i - 1];

            auto rate = [](double e_f, double e_c, double h_f, double h_c)
            {
                if (e_f < 1e-14 || e_c < 1e-14) return 0.0;
                return std::log(e_c / e_f) / std::log(h_c / h_f);
            };

            phi_L2_rate[i] = rate(f.phi_L2, c.phi_L2, f.h, c.h);
            phi_H1_rate[i] = rate(f.phi_H1, c.phi_H1, f.h, c.h);
            M_L2_rate[i]   = rate(f.M_L2, c.M_L2, f.h, c.h);
        }
    }

    bool passes(double tol = 0.3) const
    {
        if (results.size() < 2) return false;

        for (size_t i = 1; i < results.size(); ++i)
        {
            if (phi_L2_rate[i] < expected_phi_L2_rate - tol) return false;
            if (phi_H1_rate[i] < expected_phi_H1_rate - tol) return false;
            if (M_L2_rate[i]   < expected_M_L2_rate - tol)   return false;
        }
        return true;
    }

    void print() const
    {
        std::cout << "\n--- Poisson (φ) ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "φ_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "φ_H1"
                  << std::setw(8) << "rate" << "\n";
        std::cout << std::string(57, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2)
                      << results[i].h
                      << std::setw(12) << results[i].phi_L2
                      << std::setw(8) << std::fixed << std::setprecision(2)
                      << phi_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].phi_H1
                      << std::setw(8) << std::fixed << phi_H1_rate[i] << "\n";
        }

        std::cout << "\n--- Magnetization (M) ---\n";
        std::cout << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "M_L2"
                  << std::setw(8) << "rate"
                  << std::setw(12) << "Mx_L2"
                  << std::setw(12) << "My_L2"
                  << std::setw(10) << "time(s)" << "\n";
        std::cout << std::string(71, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            std::cout << std::setw(5) << results[i].refinement
                      << std::setw(12) << std::scientific << std::setprecision(2)
                      << results[i].h
                      << std::setw(12) << results[i].M_L2
                      << std::setw(8) << std::fixed << std::setprecision(2)
                      << M_L2_rate[i]
                      << std::setw(12) << std::scientific << results[i].Mx_L2
                      << std::setw(12) << results[i].My_L2
                      << std::setw(10) << std::fixed << std::setprecision(1)
                      << results[i].time_s << "\n";
        }
    }

    void write_csv(const std::string& filename) const
    {
        // Ensure mms_results/ directory exists
        std::system("mkdir -p mms_results");

        std::ofstream file("mms_results/" + filename);
        if (!file.is_open()) return;

        file << "refinement,h,n_dofs,"
             << "phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,"
             << "M_L2,M_L2_rate,Mx_L2,My_L2,"
             << "picard_iters,time_s\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            file << results[i].refinement << ","
                 << results[i].h << ","
                 << results[i].n_dofs << ","
                 << results[i].phi_L2 << "," << phi_L2_rate[i] << ","
                 << results[i].phi_H1 << "," << phi_H1_rate[i] << ","
                 << results[i].M_L2 << "," << M_L2_rate[i] << ","
                 << results[i].Mx_L2 << "," << results[i].My_L2 << ","
                 << results[i].picard_iters << ","
                 << results[i].time_s << "\n";
        }
    }
};

// ============================================================================
// Test runner declaration
// ============================================================================
PoissonMagMMSConvergenceResult run_poisson_mag_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator);

#endif // POISSON_MAG_MMS_H