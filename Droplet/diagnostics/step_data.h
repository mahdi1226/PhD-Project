// ============================================================================
// diagnostics/step_data.h - Per-Step Diagnostic Data Structure
//
// Centralized struct for all diagnostic quantities computed each time step.
// Used by CSVLogger and console output.
// ============================================================================
#ifndef STEP_DATA_H
#define STEP_DATA_H

#include <string>
#include <limits>

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

    // CH solver (filled from solver, not CHDiagnosticData)
    unsigned int ch_iterations = 0;
    double ch_residual = 0.0;
    double ch_time = 0.0;           // seconds

    // CH flags
    bool ch_bounds_violated = false;
    bool ch_energy_increasing = false;

    // ========================================================================
    // Poisson/Magnetic diagnostics
    // ========================================================================
    double phi_min = 0.0;
    double phi_max = 0.0;
    double H_max = 0.0;             // max|∇φ|
    double M_max = 0.0;             // max|M|
    double E_mag = 0.0;             // Magnetic energy (μ₀/2)∫μ_θ|H|²
    double mu_min = 1.0;            // Permeability range
    double mu_max = 1.0;

    // Poisson solver
    unsigned int poisson_iterations = 0;
    double poisson_residual = 0.0;
    double poisson_time = 0.0;      // seconds

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

    // NS solver
    double ns_time = 0.0;           // seconds

    // ========================================================================
    // Total energy (Eq. 44)
    // ========================================================================
    double E_internal = 0.0;        // E_CH + E_kin (internal, should dissipate)
    double E_total = 0.0;           // E_CH + E_kin + E_mag (includes external input)
    double dE_internal_dt = 0.0;    // Should be ≤ 0 for stability (no external input)
    double dE_total_dt = 0.0;       // Can be > 0 due to magnetic field ramping

    // ========================================================================
    // Interface tracking (for Rosensweig)
    // ========================================================================
    double interface_y_min = 0.0;   // min y where θ ≈ 0
    double interface_y_max = 0.0;   // max y where θ ≈ 0
    double interface_y_mean = 0.0;  // mean interface position

    // ========================================================================
    // Warnings/flags
    // ========================================================================
    bool theta_bounds_violated = false;
    bool divU_large = false;
    bool energy_increasing = false;  // Internal energy (E_CH + E_kin) increasing

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Compute derived quantities
    void compute_derived()
    {
        E_internal = E_CH + E_kin;
        E_total = E_CH + E_kin + E_mag;
        theta_bounds_violated = (theta_min < -1.001 || theta_max > 1.001);
        divU_large = (divU_L2 > 0.1);
        // Only warn if INTERNAL energy increases (magnetic energy is external input)
        energy_increasing = (dE_internal_dt > 1e-6);
    }

    /// Check if any warnings
    bool has_warnings() const
    {
        return theta_bounds_violated || divU_large || energy_increasing;
    }
};

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