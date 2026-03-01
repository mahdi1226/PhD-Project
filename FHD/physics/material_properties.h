// ============================================================================
// physics/material_properties.h — Material Property Functions
//
// Phase A (Nochetto 2015): Single-phase ferrofluid
//   All material constants (ν, ν_r, μ₀, ȷ, c₁, c₂, σ, 𝒯, κ₀) are spatially
//   uniform and stored in Parameters. No phase-field dependence.
//
// Phase B (future): Two-phase ferrofluid with Cahn-Hilliard
//   Material properties become functions of the phase field c (or θ):
//     χ(θ) = χ₀·(θ+1)/2        magnetic susceptibility
//     ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2  viscosity
//     ρ(θ) = 1 + r·H(θ/ε)      density ratio
//
//   Convention: θ ∈ {-1, +1}. Mapping from Φ ∈ {0, 1}: Φ = (θ+1)/2.
//
// This file provides both:
//   1. Phase A trivial accessors (constants from params)
//   2. Phase B interpolation functions (for future use)
//
// CRITICAL FOR ENERGY STABILITY (Phase B):
//   All material coefficients MUST be evaluated at θ^{k-1} (the OLD value).
// ============================================================================
#ifndef FHD_MATERIAL_PROPERTIES_H
#define FHD_MATERIAL_PROPERTIES_H

#include <cmath>

// ============================================================================
// Smoothed Heaviside: H(x) = 1/(1+exp(-x))
//
// Smooth interpolation between phases:
//   H(x) → 0 as x → -∞   (non-magnetic phase, θ = -1)
//   H(x) → 1 as x → +∞   (ferrofluid phase,   θ = +1)
// ============================================================================
inline double heaviside(double x)
{
    if (x > 30.0) return 1.0;
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

inline double heaviside_derivative(double x)
{
    const double H = heaviside(x);
    return H * (1.0 - H);
}

// ============================================================================
// Phase B: Magnetic Susceptibility χ(θ) = χ₀·(θ+1)/2
//
// θ = +1 (ferrofluid)    → χ = χ₀
// θ = -1 (non-magnetic)  → χ = 0
// ============================================================================
inline double susceptibility(double theta, double chi_0)
{
    const double phi = 0.5 * (theta + 1.0);
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return chi_0 * phi_clamped;
}

inline double susceptibility_derivative(double theta, double chi_0)
{
    const double phi = 0.5 * (theta + 1.0);
    if (phi < 0.0 || phi > 1.0)
        return 0.0;
    return 0.5 * chi_0;
}

// ============================================================================
// Phase B: Magnetic Permeability μ(θ) = 1 + χ(θ)
// ============================================================================
inline double permeability(double theta, double chi_0)
{
    return 1.0 + susceptibility(theta, chi_0);
}

// ============================================================================
// Phase B: Viscosity ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2
//
// Linear interpolation between phases.
// ============================================================================
inline double viscosity(double theta, double nu_water, double nu_ferro)
{
    const double phi = 0.5 * (theta + 1.0);
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return nu_water * (1.0 - phi_clamped) + nu_ferro * phi_clamped;
}

// ============================================================================
// Phase B: Density ratio ρ(θ) = 1 + r·H(θ/ε)
//
// Sigmoid interpolation (unlike linear chi and nu).
// ============================================================================
inline double density_ratio(double theta, double epsilon, double r)
{
    return 1.0 + r * heaviside(theta / epsilon);
}

// ============================================================================
// Phase B: Double-Well Potential (θ ∈ {-1,+1} convention)
//
// F(θ)  = (1/16)(θ² − 1)²
// f(θ)  = F'(θ)  = (θ³ − θ)/4
// f'(θ) = F''(θ) = (3θ² − 1)/4
//
// Truncated to quadratic/linear outside [-1, 1] for overshoot safety.
// ============================================================================
inline double double_well_potential(double theta)
{
    if (theta <= -1.0)
    {
        const double t = theta + 1.0;
        return 0.25 * t * t;
    }
    else if (theta >= 1.0)
    {
        const double t = theta - 1.0;
        return 0.25 * t * t;
    }
    else
    {
        const double t = theta * theta - 1.0;
        return 0.0625 * t * t;
    }
}

inline double double_well_derivative(double theta)
{
    if (theta <= -1.0)
        return 0.5 * (theta + 1.0);
    else if (theta >= 1.0)
        return 0.5 * (theta - 1.0);
    else
        return 0.25 * (theta * theta * theta - theta);
}

inline double double_well_second_derivative(double theta)
{
    if (theta <= -1.0 || theta >= 1.0)
        return 0.5;
    else
        return 0.25 * (3.0 * theta * theta - 1.0);
}

#endif // FHD_MATERIAL_PROPERTIES_H
