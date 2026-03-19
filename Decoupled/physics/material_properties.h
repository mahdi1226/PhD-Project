// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Functions
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021, B167-B193
//
// Convention: Φ ∈ {0, 1} — DIRECTLY matching Zhang's paper.
//   Φ = 1: ferrofluid phase
//   Φ = 0: non-magnetic phase (air/water)
//
// Material property interpolation (Zhang):
//   χ(Φ) = χ₀·Φ                        (linear)
//   ν(Φ) = ν_f·Φ + ν_w·(1-Φ)           (linear)
//   ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))     (sigmoid, Zhang Eq 4.2)
//
// All functions take explicit parameter values — NO GLOBALS.
//
// CRITICAL FOR ENERGY STABILITY:
//   All material coefficients MUST be evaluated at Φ^{n-1} (the OLD value).
//
// Includes:
//   Poisson + Magnetization:
//     - Susceptibility χ(Φ) = χ₀·Φ         (linear)
//     - Permeability   μ(Φ) = 1 + χ(Φ)
//
//   Cahn-Hilliard (Zhang Eq 2.2):
//     - Double-well potential  G(Φ) = (1/4)Φ²(1-Φ)²
//     - Double-well derivative g(Φ) = G'(Φ) = Φ³ - (3/2)Φ² + (1/2)Φ
//
//     Zhang's energy: E₁ = (λ/ε)∫G(Φ)dΩ
//     Chemical potential: W = -λε∇²Φ + (λ/ε)g(Φ)
//     Stabilization: S = λ/(4ε) [from L = max|G''| = 1/2, S = (λ/ε)L/2]
//
//     NOTE: G = (1/4)F where F(Φ) = Φ²(1-Φ)². The (1/4) comes from
//     Zhang Eq 2.2: E₁ = (λ/(4ε))∫Φ²(1-Φ)² = (λ/ε)∫G(Φ).
//     All assembly code uses (λ/ε)·g(Φ), so the (1/4) is absorbed here.
//
//   Navier-Stokes:
//     - Viscosity  ν(Φ) = ν_f·Φ + ν_w·(1-Φ)  (linear)
//     - Density    ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))  (sigmoid, Zhang Eq 4.2)
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>

// ============================================================================
// Smoothed Heaviside
//
//   H(x) = 1/(1+exp(-x))
//
// Used for density interpolation (Zhang Eq 4.2).
// ============================================================================
inline double heaviside(double x)
{
    if (x > 30.0) return 1.0;
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Derivative of smoothed Heaviside: H'(x) = H(x)(1 - H(x))
 */
inline double heaviside_derivative(double x)
{
    const double H = heaviside(x);
    return H * (1.0 - H);
}

// ============================================================================
// Magnetic Susceptibility (Zhang: LINEAR in Φ)
//
//   χ(Φ) = χ₀ · Φ          (Φ ∈ {0,1})
//
//   Φ = 1 (ferrofluid)    → χ = χ₀
//   Φ = 0 (non-magnetic)  → χ = 0
//   Φ = 0.5 (interface)   → χ = χ₀/2
//
// NOTE: epsilon parameter kept in signature for API compatibility but
//       is NOT used. Zhang's chi is linear, not sigmoid.
// ============================================================================
inline double susceptibility(double phi, double /*epsilon*/, double chi_0)
{
    // Clamp to [0,1] for safety (overshoots possible in CH)
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return chi_0 * phi_clamped;
}

// ============================================================================
// Susceptibility derivative dχ/dΦ
//
//   χ(Φ) = χ₀ · Φ  →  dχ/dΦ = χ₀   (when Φ ∈ [0,1], else 0)
// ============================================================================
inline double susceptibility_derivative(double phi, double /*epsilon*/, double chi_0)
{
    if (phi < 0.0 || phi > 1.0)
        return 0.0;  // clamped region
    return chi_0;
}

// ============================================================================
// Magnetic Permeability
//
//   μ(Φ) = 1 + χ(Φ) = 1 + χ₀·Φ
//
// Used in Poisson diagnostics: E_mag = ½∫μ(Φ)|H|² dΩ
// ============================================================================
inline double permeability(double phi, double epsilon, double chi_0)
{
    return 1.0 + susceptibility(phi, epsilon, chi_0);
}

// ============================================================================
// Viscosity (Zhang: LINEAR in Φ)
//
//   ν(Φ) = ν_f·Φ + ν_w·(1-Φ)       (Φ ∈ {0,1})
//
//   Φ = 1 (ferrofluid)    → ν = ν_f   (higher viscosity)
//   Φ = 0 (non-magnetic)  → ν = ν_w   (lower viscosity)
//
// CRITICAL: Must use Φ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (Zhang Eq 4.4): ν_w = 1.0, ν_f = 2.0  →  ν ∈ [1, 2]
// ============================================================================
inline double viscosity(double phi, double /*epsilon*/,
                        double nu_water, double nu_ferro)
{
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return nu_water * (1.0 - phi_clamped) + nu_ferro * phi_clamped;
}

// ============================================================================
// Density Ratio (Zhang Eq 4.2: SIGMOID)
//
//   ρ(Φ) = 1 + r / (1 + exp((1-2Φ)/ε))
//
//   Φ = 1 (ferrofluid)    → ρ ≈ 1 + r  (heavier)
//   Φ = 0 (non-magnetic)  → ρ ≈ 1      (reference density)
//
// CRITICAL: Must use Φ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (Zhang Eq 4.4): r = 0.1  →  ρ ∈ [1.0, 1.1]
// ============================================================================
inline double density_ratio(double phi, double epsilon, double r)
{
    // Zhang Eq 4.2: ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))
    return 1.0 + r * heaviside((2.0 * phi - 1.0) / epsilon);
}

// ============================================================================
// Double-Well Potential — Zhang Eq 2.2
//
//   G(Φ) = (1/4) Φ²(1-Φ)²
//
// Zhang's CH energy: E₁ = (λ/ε)∫G(Φ)dΩ = (λ/(4ε))∫Φ²(1-Φ)²dΩ
//
// The (1/4) factor is included here so that all assembly code uses the
// simple coefficient (λ/ε) consistently:
//   E₁ = (λ/ε) ∫ G(Φ) dΩ
//   δE₁/δΦ = (λ/ε) g(Φ)  where g = G'
//
// Stabilization: S = λ/(4ε), from L = max|G''| = 1/2, S = (λ/ε)·L/2.
//
// Truncated outside [0, 1] for overshoot safety using quadratic extension.
// G(0) = G(1) = 0, G'(0) = G'(1) = 0, G''(0) = G''(1) = 1/2.
// ============================================================================
inline double double_well_potential(double phi)
{
    if (phi <= 0.0)
    {
        return 0.25 * phi * phi;  // quadratic extension: G ≈ (1/4)Φ²
    }
    else if (phi >= 1.0)
    {
        const double t = phi - 1.0;
        return 0.25 * t * t;  // quadratic extension: G ≈ (1/4)(Φ-1)²
    }
    else
    {
        // G(Φ) = (1/4) Φ²(1-Φ)²
        return 0.25 * phi * phi * (1.0 - phi) * (1.0 - phi);
    }
}

/**
 * @brief Double-well derivative g(Φ) = G'(Φ) = Φ³ - (3/2)Φ² + (1/2)Φ
 *
 * Equivalently: g(Φ) = (1/2)Φ(1-Φ)(1-2Φ) = (1/4) d/dΦ[Φ²(1-Φ)²]
 * Used in CH assembly: (λ/ε)·g(Φ) is the nonlinear reaction term.
 *
 * Truncated to linear outside [0, 1]:
 *   g(Φ) ≈ (1/2)Φ for Φ < 0     [matching g(0)=0, g'(0)=1/2]
 *   g(Φ) ≈ (1/2)(Φ-1) for Φ > 1 [matching g(1)=0, g'(1)=1/2]
 */
inline double double_well_derivative(double phi)
{
    if (phi <= 0.0)
        return 0.5 * phi;  // linear extension
    else if (phi >= 1.0)
        return 0.5 * (phi - 1.0);  // linear extension
    else
        return phi * phi * phi - 1.5 * phi * phi + 0.5 * phi;
}

#endif // MATERIAL_PROPERTIES_H
