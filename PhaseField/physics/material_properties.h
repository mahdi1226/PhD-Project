// ============================================================================
// physics/material_properties.h - Physical Constants and Material Functions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// RUNTIME PARAMETERS: Physical constants are now in Parameters::Physics struct.
// Use params_.physics.epsilon, params_.physics.chi_0, etc.
//
// Paper values:
//   Rosensweig (6.2): epsilon=0.01,  chi_0=0.5, lambda=0.05
//   Hedgehog (6.3):   epsilon=0.005, chi_0=0.9, lambda=0.025
//
// CRITICAL FOR ENERGY STABILITY (Eq. 43, Theorem 4.1):
// All material coefficients MUST be evaluated at θ^{k-1} (the OLD/FROZEN value)
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>

// ============================================================================
// DEFAULT Physical Constants (Section 6.2 - Rosensweig defaults)
// These are FALLBACKS. Prefer using params_.physics.* for runtime values.
// ============================================================================

// Cahn-Hilliard (Eq. 14a-14b) - DEFAULTS
constexpr double epsilon_default = 0.01;        // interface thickness
constexpr double mobility_default = 0.0002;     // mobility (γ)
constexpr double lambda_default = 0.05;         // capillary coefficient

// For backward compatibility (deprecated - use params_.physics.*)
constexpr double epsilon = 0.01;
constexpr double mobility = 0.0002;
constexpr double lambda = 0.05;

// Viscosity (Eq. 17, p.501)
constexpr double nu_water = 1.0;            // non-magnetic phase
constexpr double nu_ferro = 2.0;            // ferrofluid phase

// Magnetic (Section 6.2, p.520) - DEFAULTS
constexpr double chi_0_default = 0.5;       // susceptibility
constexpr double chi_0 = 0.5;               // backward compatibility
constexpr double mu_0 = 1.0;                // permeability of free space
constexpr double tau_M = 0.0;               // magnetization relaxation time

// Density / Gravity (Eq. 19, p.501)
constexpr double rho = 1.0;                 // reference density
constexpr double r = 0.1;                   // density ratio
constexpr double gravity = 9.81;            // physical (m/s²)
constexpr double gravity_dimensionless = 30000.0;  // non-dimensionalized

// Numerical stabilization
constexpr double grad_div = 0.0;            // grad-div parameter

// ============================================================================
// Material Functions (Eq. 17-18, p.501)
//
// CRITICAL: Always pass theta_OLD for energy stability (Eq. 43)
//
// Parameter-aware versions take epsilon/chi_0 as arguments.
// Legacy versions use global constants (for backward compatibility).
// ============================================================================

/** Smoothed Heaviside H(x) = 1/(1+exp(-x)) */
inline double heaviside(double x)
{
    if (x > 30.0) return 1.0;
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

/** H'(x) = H(x)(1-H(x)) */
inline double heaviside_derivative(double x)
{
    const double H = heaviside(x);
    return H * (1.0 - H);
}

/** ν_θ = ν_w + (ν_f - ν_w) H(θ/ε) - PARAMETER VERSION */
inline double viscosity(double theta, double eps)
{
    return nu_water + (nu_ferro - nu_water) * heaviside(theta / eps);
}

/** ν_θ = ν_w + (ν_f - ν_w) H(θ/ε) - LEGACY (uses global epsilon) */
inline double viscosity(double theta)
{
    return viscosity(theta, epsilon);
}

/** χ_θ = χ₀ H(θ/ε) - PARAMETER VERSION */
inline double susceptibility(double theta, double eps, double chi0)
{
    return chi0 * heaviside(theta / eps);
}

/** χ_θ = χ₀ H(θ/ε) - LEGACY (uses global constants) */
inline double susceptibility(double theta)
{
    return susceptibility(theta, epsilon, chi_0);
}

/** μ_θ = 1 + χ_θ - PARAMETER VERSION */
inline double permeability(double theta, double eps, double chi0)
{
    return 1.0 + susceptibility(theta, eps, chi0);
}

/** μ_θ = 1 + χ_θ - LEGACY */
inline double permeability(double theta)
{
    return 1.0 + susceptibility(theta);
}

/** ρ_θ = 1 + r H(θ/ε) - PARAMETER VERSION */
inline double density_ratio(double theta, double eps)
{
    return 1.0 + r * heaviside(theta / eps);
}

/** ρ_θ = 1 + r H(θ/ε) - LEGACY */
inline double density_ratio(double theta)
{
    return density_ratio(theta, epsilon);
}

// ============================================================================
// Double-well potential (Eq. 2-3, p.499)
// ============================================================================

/** F(θ) = (1/4)(θ²-1)² truncated */
inline double double_well_potential(double theta)
{
    if (theta <= -1.0)
    {
        const double t = theta + 1.0;
        return t * t;
    }
    else if (theta >= 1.0)
    {
        const double t = theta - 1.0;
        return t * t;
    }
    else
    {
        const double t = theta * theta - 1.0;
        return 0.25 * t * t;
    }
}

/** f(θ) = F'(θ) = θ³ - θ truncated */
inline double double_well_derivative(double theta)
{
    if (theta <= -1.0)
        return 2.0 * (theta + 1.0);
    else if (theta >= 1.0)
        return 2.0 * (theta - 1.0);
    else
        return theta * theta * theta - theta;
}

/** f'(θ) = F''(θ) = 3θ² - 1 truncated */
inline double double_well_second_derivative(double theta)
{
    if (theta <= -1.0 || theta >= 1.0)
        return 2.0;
    else
        return 3.0 * theta * theta - 1.0;
}

// ============================================================================
// NOTE: compute_susceptibility is defined in poisson_assembler.h
// Do not duplicate it here to avoid linker errors.
// ============================================================================

#endif // MATERIAL_PROPERTIES_H