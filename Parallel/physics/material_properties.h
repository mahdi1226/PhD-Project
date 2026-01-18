// ============================================================================
// physics/material_properties.h - Material Functions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// All material coefficients are now runtime parameters in Parameters::Physics.
// Functions in this file require explicit parameter values - NO GLOBALS.
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
// Smoothed Heaviside and Derivatives (Eq. 17-18, p.501)
// ============================================================================

/**
 * @brief Smoothed Heaviside function H(x) = 1/(1+exp(-x))
 *
 * Used for smooth interpolation between phases.
 * H(x) → 0 as x → -∞
 * H(x) → 1 as x → +∞
 * H(0) = 0.5
 */
inline double heaviside(double x)
{
    if (x > 30.0) return 1.0;   // Prevent overflow
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Derivative of smoothed Heaviside: H'(x) = H(x)(1-H(x))
 */
inline double heaviside_derivative(double x)
{
    const double H = heaviside(x);
    return H * (1.0 - H);
}

// ============================================================================
// Viscosity (Eq. 17, p.501)
//
//   ν(θ) = ν_water + (ν_ferro - ν_water) H(θ/ε)
//
// Interpolates between non-magnetic (θ=-1) and ferrofluid (θ=+1) phases.
// ============================================================================

/**
 * @brief Viscosity ν(θ) = ν_w + (ν_f - ν_w) H(θ/ε)
 *
 * @param theta     Phase field value (use θ^{k-1} for energy stability!)
 * @param epsilon   Interface thickness ε
 * @param nu_water  Viscosity of non-magnetic phase
 * @param nu_ferro  Viscosity of ferrofluid phase
 * @return Interpolated viscosity
 */
inline double viscosity(double theta, double epsilon, double nu_water, double nu_ferro)
{
    return nu_water + (nu_ferro - nu_water) * heaviside(theta / epsilon);
}

// ============================================================================
// Magnetic Susceptibility (Eq. 18, p.501)
//
//   χ(θ) = χ₀ H(θ/ε)
//
// χ ≈ 0 in non-magnetic phase (θ=-1)
// χ ≈ χ₀ in ferrofluid phase (θ=+1)
// ============================================================================

/**
 * @brief Susceptibility χ(θ) = χ₀ H(θ/ε)
 *
 * @param theta    Phase field value (use θ^{k-1} for energy stability!)
 * @param epsilon  Interface thickness ε
 * @param chi_0    Maximum susceptibility in ferrofluid
 * @return Interpolated susceptibility
 */
inline double susceptibility(double theta, double epsilon, double chi_0)
{
    // θ = +1 : ferrofluid  → χ = χ₀
    // θ = -1 : non-magnetic → χ = 0
    if (theta / epsilon > 20.0) return chi_0;  // avoid overflow
    if (theta / epsilon < -20.0) return 0.0;   // avoid underflow
    return chi_0 / (1.0 + std::exp(-theta / epsilon));
}

// ============================================================================
// Magnetic Permeability
//
//   μ(θ) = 1 + χ(θ) = 1 + χ₀ H(θ/ε)
// ============================================================================

/**
 * @brief Permeability μ(θ) = 1 + χ(θ)
 *
 * @param theta    Phase field value
 * @param epsilon  Interface thickness ε
 * @param chi_0    Maximum susceptibility
 * @return Interpolated permeability
 */
inline double permeability(double theta, double epsilon, double chi_0)
{
    return 1.0 + susceptibility(theta, epsilon, chi_0);
}

// ============================================================================
// Density Ratio (Eq. 19, p.501)
//
//   ρ(θ) = 1 + r H(θ/ε)
//
// where r = (ρ_ferro - ρ_water) / ρ_water is the density ratio parameter.
// ============================================================================

/**
 * @brief Density ratio ρ(θ) = 1 + r H(θ/ε)
 *
 * @param theta    Phase field value
 * @param epsilon  Interface thickness ε
 * @param r        Density ratio parameter
 * @return Interpolated density ratio
 */
inline double density_ratio(double theta, double epsilon, double r)
{
    return 1.0 + r * heaviside(theta / epsilon);
}

// ============================================================================
// Double-Well Potential (Eq. 2-3, p.499)
//
//   F(θ) = (1/4)(θ² - 1)²
//   f(θ) = F'(θ) = θ³ - θ
//   f'(θ) = F''(θ) = 3θ² - 1
//
// Truncated outside [-1, 1] for stability with overshoots.
// ============================================================================

/**
 * @brief Double-well potential F(θ) = (1/4)(θ² - 1)²
 *
 * Truncated to quadratic outside [-1, 1] to handle overshoots.
 */
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

/**
 * @brief Double-well derivative f(θ) = F'(θ) = θ³ - θ
 *
 * Truncated to linear outside [-1, 1].
 */
inline double double_well_derivative(double theta)
{
    if (theta <= -1.0)
        return 2.0 * (theta + 1.0);
    else if (theta >= 1.0)
        return 2.0 * (theta - 1.0);
    else
        return theta * theta * theta - theta;
}

/**
 * @brief Double-well second derivative f'(θ) = F''(θ) = 3θ² - 1
 *
 * Truncated to constant outside [-1, 1].
 */
inline double double_well_second_derivative(double theta)
{
    if (theta <= -1.0 || theta >= 1.0)
        return 2.0;
    else
        return 3.0 * theta * theta - 1.0;
}

#endif // MATERIAL_PROPERTIES_H