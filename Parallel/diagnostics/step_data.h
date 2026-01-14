// ============================================================================
// diagnostics/step_data.h - Per-Step Diagnostic Data Structures
//
// Centralized structs for all diagnostic quantities computed each time step.
// Used by MetricsLogger, ConsoleLogger, and TimingLogger.
//
// UPDATED: Added Rosensweig validation fields (wavelength tracking)
// ============================================================================
#ifndef STEP_DATA_H
#define STEP_DATA_H

#include <string>
#include <algorithm>
#include <cmath>

// ============================================================================
// Per-Step Timing Data
// ============================================================================

/**
 * @brief Timing data for a single time step
 */
struct StepTiming
{
    double ch_time = 0.0;           // Cahn-Hilliard solve time (s)
    double poisson_time = 0.0;      // Poisson solve time (s)
    double mag_time = 0.0;          // Magnetization transport time (s)
    double ns_time = 0.0;           // Navier-Stokes solve time (s)
    double output_time = 0.0;       // VTK/diagnostics output time (s)
    double step_total = 0.0;        // Total step time (s)

    // Cumulative totals (updated externally)
    double cumul_ch = 0.0;
    double cumul_poisson = 0.0;
    double cumul_mag = 0.0;
    double cumul_ns = 0.0;
    double cumul_total = 0.0;

    // Memory usage (MB)
    double memory_mb = 0.0;

    /**
     * @brief Compute step_total from subsystem times
     */
    void compute_step_total()
    {
        step_total = ch_time + poisson_time + mag_time + ns_time + output_time;
    }
};

// ============================================================================
// Per-Step Diagnostic Data (Full)
// ============================================================================

/**
 * @brief All diagnostic data for a single time step
 */
struct StepData
{
    // ========================================================================
    // Time stepping
    // ========================================================================
    unsigned int step = 0;
    double time = 0.0;
    double dt = 0.0;

    // ========================================================================
    // Cahn-Hilliard diagnostics
    // ========================================================================
    double theta_min = 0.0;
    double theta_max = 0.0;
    double mass = 0.0;
    double E_CH = 0.0;              // Cahn-Hilliard energy
    double dE_CH_dt = 0.0;          // Energy rate of change

    unsigned int ch_iterations = 0;
    double ch_residual = 0.0;
    double ch_time = 0.0;           // seconds

    // ========================================================================
    // Poisson/Magnetic diagnostics
    // ========================================================================
    double phi_min = 0.0;
    double phi_max = 0.0;
    double H_max = 0.0;             // max|∇φ|
    double M_max = 0.0;             // max|M|
    double E_mag = 0.0;             // Magnetic energy (μ₀/2)∫μ_θ|H|²
    double mu_min = 1.0;
    double mu_max = 1.0;

    unsigned int poisson_iterations = 0;
    double poisson_residual = 0.0;
    double poisson_time = 0.0;

    // ========================================================================
    // Magnetization transport diagnostics
    // ========================================================================
    unsigned int mag_iterations = 0;
    double mag_residual = 0.0;
    double mag_time = 0.0;

    // ========================================================================
    // Navier-Stokes diagnostics
    // ========================================================================
    double ux_min = 0.0;
    double ux_max = 0.0;
    double uy_min = 0.0;
    double uy_max = 0.0;
    double U_max = 0.0;             // max|U|
    double E_kin = 0.0;             // Kinetic energy (1/2)||U||²

    double divU_L2 = 0.0;           // ||∇·U||_L2
    double divU_Linf = 0.0;         // max|∇·U|
    double CFL = 0.0;               // CFL number

    double p_min = 0.0;
    double p_max = 0.0;

    // Forces
    double F_cap_max = 0.0;         // max|F_capillary|
    double F_mag_max = 0.0;         // max|F_magnetic|
    double F_grav_max = 0.0;        // max|F_gravity|

    unsigned int ns_outer_iterations = 0;
    unsigned int ns_inner_iterations = 0;
    double ns_residual = 0.0;
    double ns_time = 0.0;

    // ========================================================================
    // System-level diagnostics (for console)
    // ========================================================================
    double E_internal = 0.0;        // E_CH + E_kin (should dissipate)
    double E_total = 0.0;           // E_CH + E_kin + E_mag
    double dE_internal_dt = 0.0;    // Should be ≤ 0 for stability
    double dE_total_dt = 0.0;       // Can be > 0 due to magnetic ramping

    double system_residual = 0.0;   // max(ch, poisson, ns residuals)

    // ========================================================================
    // Interface tracking (for Rosensweig)
    // ========================================================================
    double interface_y_min = 0.0;
    double interface_y_max = 0.0;
    double interface_y_mean = 0.0;
    double interface_y_initial = 0.0;  // Store initial for delta computation
    unsigned int spike_count = 0;

    // ========================================================================
    // Rosensweig validation (NEW)
    // Populated by update_validation_diagnostics() from validation_diagnostics.h
    // ========================================================================
    double interface_amplitude = 0.0;   // (y_max - y_min) / 2 (spike height)
    double dominant_wavelength = 0.0;   // Measured wavelength from profile
    double wavelength_theory = 0.0;     // λ_c = 2π√(σ/ρg)
    double wavelength_error = 0.0;      // |measured - theory| / theory

    // ========================================================================
    // Timing
    // ========================================================================
    double wall_time_step = 0.0;    // This step's wall time (s)
    double wall_time_total = 0.0;   // Cumulative wall time (s)

    // ========================================================================
    // Mesh info (for AMR)
    // ========================================================================
    unsigned int n_active_cells = 0;
    unsigned int n_dofs_total = 0;

    // ========================================================================
    // Warning flags
    // ========================================================================
    bool theta_bounds_violated = false;
    bool divU_large = false;
    bool energy_increasing = false;
    bool solver_fallback_used = false;

    // ========================================================================
    // Helper methods
    // ========================================================================

    /**
     * @brief Compute derived quantities from subsystem data
     */
    void compute_derived()
    {
        E_internal = E_CH + E_kin;
        E_total = E_CH + E_kin + E_mag;

        // System residual = max of all subsystem residuals
        system_residual = std::max({ch_residual, poisson_residual, ns_residual});

        // Warning flags
        theta_bounds_violated = (theta_min < -1.01 || theta_max > 1.01);
        divU_large = (divU_L2 > 0.1);
        energy_increasing = (dE_internal_dt > 1e-6);

        // Also update interface_amplitude from y_min/y_max if not set by validation
        if (interface_amplitude == 0.0 && interface_y_max > interface_y_min)
        {
            interface_amplitude = (interface_y_max - interface_y_min) / 2.0;
        }
    }

    /**
     * @brief Compute energy rates from previous step
     */
    void compute_energy_rates(const StepData& prev)
    {
        if (dt > 0.0 && step > 0)
        {
            dE_CH_dt = (E_CH - prev.E_CH) / dt;
            dE_internal_dt = (E_internal - prev.E_internal) / dt;
            dE_total_dt = (E_total - prev.E_total) / dt;
        }
    }

    /**
     * @brief Check if any warnings are active
     */
    bool has_warnings() const
    {
        return theta_bounds_violated || divU_large || energy_increasing || solver_fallback_used;
    }

    /**
     * @brief Check if interface has changed significantly
     * @param threshold Minimum change to report (default 0.005)
     */
    bool interface_changed(double threshold = 0.005) const
    {
        return std::abs(interface_y_max - interface_y_initial) > threshold;
    }

    /**
     * @brief Check if wavelength validation is passing (NEW)
     * @param tolerance Acceptable relative error (default 20%)
     */
    bool wavelength_valid(double tolerance = 0.20) const
    {
        return (wavelength_theory > 0 &&
                dominant_wavelength > 0 &&
                wavelength_error < tolerance);
    }
};

// ============================================================================
// MMS Convergence Data
// ============================================================================

/**
 * @brief MMS convergence data for a single refinement level
 */
struct ConvergenceData
{
    unsigned int refinement = 0;
    double h = 0.0;

    // CH errors
    double theta_L2 = 0.0;
    double theta_H1 = 0.0;
    double psi_L2 = 0.0;

    // Poisson errors
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;

    // NS errors
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double divU_L2 = 0.0;
};

#endif // STEP_DATA_H