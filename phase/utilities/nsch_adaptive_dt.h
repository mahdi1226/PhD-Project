// ============================================================================
// nsch_adaptive_dt.h - Adaptive time-stepping for NS-CH solver
// ============================================================================
#ifndef NSCH_ADAPTIVE_DT_H
#define NSCH_ADAPTIVE_DT_H

#include "diagnostics/nsch_verification.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ============================================================================
// Adaptive time-step controller
// ============================================================================
class AdaptiveTimeStep
{
public:
    // Parameters
    double dt_min;
    double dt_max;
    double dt_initial;

    double cfl_target;       // Target CFL number
    double cfl_max;          // Maximum allowed CFL

    double energy_tol;       // Tolerance for energy increase
    double growth_factor;    // Factor to increase dt when stable
    double shrink_factor;    // Factor to decrease dt when unstable

    // State
    double dt_current;
    double energy_prev;
    bool   first_step;
    int    stable_steps;     // Count of consecutive stable steps
    int    steps_since_change;

    // Statistics
    int    n_reductions;
    int    n_increases;
    double dt_min_used;
    double dt_max_used;

    // Constructor with default parameters
    AdaptiveTimeStep(double dt_init = 1e-4,
                     double dt_min_val = 1e-8,
                     double dt_max_val = 1e-3)
        : dt_min(dt_min_val)
        , dt_max(dt_max_val)
        , dt_initial(dt_init)
        , cfl_target(0.5)
        , cfl_max(1.0)
        , energy_tol(1e-6)
        , growth_factor(1.1)
        , shrink_factor(0.5)
        , dt_current(dt_init)
        , energy_prev(0.0)
        , first_step(true)
        , stable_steps(0)
        , steps_since_change(0)
        , n_reductions(0)
        , n_increases(0)
        , dt_min_used(dt_init)
        , dt_max_used(dt_init)
    {}

    // Main function: compute new dt based on metrics
    double compute_new_dt(const NSCHVerificationMetrics& m, double h_min)
    {
        double dt_new = dt_current;
        bool reduced = false;
        std::string reason;

        // ====================================================================
        // Check 1: Energy increase (most important for stability)
        // ====================================================================
        if (!first_step)
        {
            double energy_change = m.total_energy - energy_prev;
            double relative_change = std::abs(energy_change) / (std::abs(energy_prev) + 1e-14);

            // If energy increased significantly, reduce dt
            if (energy_change > energy_tol * std::abs(energy_prev) && energy_change > 1e-10)
            {
                dt_new = shrink_factor * dt_current;
                reduced = true;
                reason = "energy increasing";
                stable_steps = 0;
            }
            // If energy increased by a lot (>1%), be more aggressive
            if (relative_change > 0.01 && energy_change > 0)
            {
                dt_new = 0.25 * dt_current;  // More aggressive reduction
                reduced = true;
                reason = "large energy increase";
                stable_steps = 0;
            }
        }

        // ====================================================================
        // Check 2: CFL condition
        // ====================================================================
        if (m.cfl_number > cfl_max)
        {
            double dt_cfl = cfl_target / m.cfl_number * dt_current;
            if (dt_cfl < dt_new)
            {
                dt_new = dt_cfl;
                reduced = true;
                reason = "CFL violation";
                stable_steps = 0;
            }
        }

        // ====================================================================
        // Check 3: Velocity-based CFL estimate for next step
        // ====================================================================
        if (m.u_max > 1e-10)
        {
            double dt_cfl_est = cfl_target * h_min / m.u_max;
            if (dt_cfl_est < dt_new)
            {
                dt_new = dt_cfl_est;
                reduced = true;
                reason = "CFL estimate";
                stable_steps = 0;
            }
        }

        // ====================================================================
        // Check 4: Divergence growth (indicates NS instability)
        // ====================================================================
        if (m.divergence_L2 > 1.0)  // Significant divergence
        {
            dt_new = std::min(dt_new, 0.5 * dt_current);
            reduced = true;
            reason = "high divergence";
            stable_steps = 0;
        }

        // ====================================================================
        // Check 5: Bound violation approaching
        // ====================================================================
        if (m.c_min < -1.05 || m.c_max > 1.05)
        {
            dt_new = std::min(dt_new, 0.5 * dt_current);
            reduced = true;
            reason = "bound violation";
            stable_steps = 0;
        }

        // ====================================================================
        // Increase dt if stable for several steps
        // ====================================================================
        if (!reduced)
        {
            stable_steps++;
            steps_since_change++;

            // Only increase if stable for at least 20 steps and not changed recently
            if (stable_steps > 20 && steps_since_change > 10)
            {
                // Check that we have good margin on all metrics
                bool can_increase = (m.cfl_number < 0.3 * cfl_max) &&
                                   (m.divergence_L2 < 0.5) &&
                                   (m.c_min > -1.02 && m.c_max < 1.02);

                if (can_increase && dt_current < dt_max)
                {
                    dt_new = growth_factor * dt_current;
                    n_increases++;
                    steps_since_change = 0;
                }
            }
        }
        else
        {
            n_reductions++;
            steps_since_change = 0;
        }

        // ====================================================================
        // Clamp to bounds
        // ====================================================================
        dt_new = std::clamp(dt_new, dt_min, dt_max);

        // ====================================================================
        // Update state
        // ====================================================================
        if (reduced && dt_new < dt_current)
        {
            std::cout << "  [ADAPTIVE] dt: " << dt_current << " -> " << dt_new
                      << " (" << reason << ")" << std::endl;
        }

        energy_prev = m.total_energy;
        first_step = false;
        dt_current = dt_new;

        // Update statistics
        dt_min_used = std::min(dt_min_used, dt_new);
        dt_max_used = std::max(dt_max_used, dt_new);

        return dt_new;
    }

    // Initialize with first energy value
    void initialize(double initial_energy)
    {
        energy_prev = initial_energy;
        first_step = false;
    }

    // Get current dt
    double get_dt() const { return dt_current; }

    // Print summary
    void print_summary() const
    {
        std::cout << "\n--- Adaptive Time-Stepping Summary ---\n"
                  << "  dt range used: [" << dt_min_used << ", " << dt_max_used << "]\n"
                  << "  Reductions:    " << n_reductions << "\n"
                  << "  Increases:     " << n_increases << "\n"
                  << "  Final dt:      " << dt_current << "\n";
    }
};

#endif // NSCH_ADAPTIVE_DT_H