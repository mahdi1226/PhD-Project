// ============================================================================
// physics/material_properties.h — Material Property Functions
//
// Phase A (Nochetto 2015): Single-phase ferrofluid
//   All material constants (ν, ν_r, μ₀, ȷ, c₁, c₂, σ, 𝒯, κ₀) are spatially
//   uniform and stored in Parameters. No phase-field dependence.
//
// Phase B: Two-phase ferrofluid with Cahn-Hilliard
//   Material properties become functions of the phase field θ:
//     χ(θ) = χ₀·H(θ/ε)               magnetic susceptibility (sigmoid)
//     ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)  viscosity (sigmoid)
//     ρ(θ) = 1 + r·H(θ/ε)            density ratio (sigmoid)
//
//   Zhang, He & Yang (2021) use sigmoid 1/(1+e^{-(2Φ-1)/ε}) for ALL
//   material properties. In θ convention: H(θ/ε) = 1/(1+e^{-θ/ε}).
//
//   Convention: θ ∈ {-1, +1}. Mapping from Φ ∈ {0, 1}: Φ = (θ+1)/2.
//
//   NOTE (Issue B6): Linear interpolation (chi_ferro*(θ+1)/2) was tested
//   for Rosensweig instability stability, but the current code uses sigmoid
//   for all material properties. The deformation benchmark results were
//   obtained with sigmoid interpolation. For small ε/R (e.g., ε=0.02,
//   R=0.2), sigmoid acts as a sharp step — effectively identical to
//   phase-wise constant properties.
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
// Phase B: Magnetic Susceptibility χ(θ) = χ₀·H(θ/ε)
//
// Zhang (2021): χ(Φ) = χ₀/(1+e^{-(2Φ-1)/ε})
// In θ convention: χ(θ) = χ₀·H(θ/ε) = χ₀/(1+e^{-θ/ε})
//
// θ = +1 (ferrofluid)    → χ ≈ χ₀  (sigmoid → 1)
// θ = -1 (non-magnetic)  → χ ≈ 0   (sigmoid → 0)
// Transition width ~ ε (sharp step for small ε)
// ============================================================================
inline double susceptibility(double theta, double chi_0, double epsilon)
{
    return chi_0 * heaviside(theta / epsilon);
}

inline double susceptibility_derivative(double theta, double chi_0, double epsilon)
{
    return chi_0 * heaviside_derivative(theta / epsilon) / epsilon;
}

// ============================================================================
// Phase B: Magnetic Permeability μ(θ) = 1 + χ(θ)
// ============================================================================
inline double permeability(double theta, double chi_0, double epsilon)
{
    return 1.0 + susceptibility(theta, chi_0, epsilon);
}

// ============================================================================
// Phase B: Viscosity ν(θ) = ν_w + (ν_f − ν_w)·H(θ/ε)
//
// Zhang (2021): ν(Φ) = ν_w + (ν_f − ν_w)/(1+e^{-(2Φ-1)/ε})
// In θ convention: ν(θ) = ν_w + (ν_f − ν_w)·H(θ/ε)
//
// Sigmoid interpolation: sharp transition at interface.
// ============================================================================
inline double viscosity(double theta, double nu_water, double nu_ferro,
                        double epsilon)
{
    return nu_water + (nu_ferro - nu_water) * heaviside(theta / epsilon);
}

// ============================================================================
// Phase B: Density ratio ρ(θ) = 1 + r·H(θ/ε)
//
// Sigmoid interpolation (same as chi and nu in current implementation).
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
