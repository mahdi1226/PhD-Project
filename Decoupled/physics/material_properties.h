// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Functions
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
//            Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Convention: Code uses θ ∈ {-1, +1}. Zhang uses Φ ∈ {0, 1}.
//             Mapping: Φ = (θ+1)/2.
//
// Zhang's material property interpolation is LINEAR in Φ:
//   χ(Φ) = χ₀·Φ           → χ(θ) = χ₀·(θ+1)/2
//   ν(Φ) = ν_f·Φ + ν_w·(1-Φ) → ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2
//   ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))  → ρ(θ) = 1 + r·H(θ/ε)
//
// Note: Density uses sigmoid (Zhang Eq 4.2), chi and nu use LINEAR.
//
// All functions take explicit parameter values — NO GLOBALS.
//
// CRITICAL FOR ENERGY STABILITY:
//   All material coefficients MUST be evaluated at θ^{k-1} (the OLD value).
//
// Includes:
//   Poisson + Magnetization:
//     - Susceptibility χ(θ) = χ₀·(θ+1)/2  (linear, Zhang convention)
//     - Permeability   μ(θ) = 1 + χ(θ)
//
//   Cahn-Hilliard:
//     - Double-well potential  F(θ)  = (1/4)(θ² - 1)²
//     - Double-well derivative f(θ)  = θ³ - θ
//     - Double-well curvature  f'(θ) = 3θ² - 1
//
//   Navier-Stokes:
//     - Viscosity  ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2  (linear, Zhang convention)
//     - Density    ρ(θ) = 1 + r·H(θ/ε)                (sigmoid, Zhang Eq 4.2)
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>

// ============================================================================
// Smoothed Heaviside (Eq. 17-18, p.501)
//
//   H(x) = 1/(1+exp(-x))
//
// Smooth interpolation between phases:
//   H(x) → 0 as x → -∞   (non-magnetic phase, θ = -1)
//   H(x) → 1 as x → +∞   (ferrofluid phase,   θ = +1)
//   H(0) = 0.5
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
// Magnetic Susceptibility (Zhang convention: LINEAR in Φ)
//
//   Zhang:  χ(Φ) = χ₀ · Φ          (Φ ∈ {0,1})
//   Code:   χ(θ) = χ₀ · (θ+1)/2    (θ ∈ {-1,+1})
//
//   θ = +1 (ferrofluid)    → χ = χ₀
//   θ = -1 (non-magnetic)  → χ = 0
//   θ =  0 (interface)     → χ = χ₀/2
//
// NOTE: epsilon parameter kept in signature for API compatibility but
//       is NOT used. Zhang's chi is linear, not sigmoid.
// ============================================================================
inline double susceptibility(double theta, double /*epsilon*/, double chi_0)
{
    // Linear interpolation: Φ = (θ+1)/2, clamp to [0,1] for safety
    const double phi = 0.5 * (theta + 1.0);
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return chi_0 * phi_clamped;
}

// ============================================================================
// Magnetic Permeability
//
//   μ(θ) = 1 + χ(θ) = 1 + χ₀ H(θ/ε)
//
// Used in Poisson diagnostics: E_mag = ½∫μ(θ)|H|² dΩ
//
// NOTE: μ(θ) does NOT appear in the Poisson assembly (Eq. 42d).
// The paper's formulation eliminates μ from the weak form.
// ============================================================================
inline double permeability(double theta, double epsilon, double chi_0)
{
    return 1.0 + susceptibility(theta, epsilon, chi_0);
}

// ============================================================================
// Viscosity (Zhang convention: LINEAR in Φ)
//
//   Zhang:  ν(Φ) = ν_f·Φ + ν_w·(1-Φ)       (Φ ∈ {0,1})
//   Code:   ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2  (θ ∈ {-1,+1})
//
// Interpolates between non-magnetic and ferrofluid phases:
//   θ = +1 (ferrofluid)    → ν = ν_f   (higher viscosity)
//   θ = -1 (non-magnetic)  → ν = ν_w   (lower viscosity)
//
// Used in NS assembly: (ν(θ) D(U), D(V))
//
// CRITICAL: Must use θ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (Zhang Eq 4.4): ν_w = 1.0, ν_f = 2.0  →  ν ∈ [1, 2]
//
// NOTE: epsilon parameter kept in signature for API compatibility but
//       is NOT used. Zhang's nu is linear, not sigmoid.
// ============================================================================

/**
 * @brief Viscosity ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2  (linear interpolation)
 *
 * @param theta     Phase field value (use θ^{n-1} for energy stability!)
 * @param epsilon   Interface thickness ε (unused — linear interpolation)
 * @param nu_water  Viscosity of non-magnetic phase ν_w
 * @param nu_ferro  Viscosity of ferrofluid phase ν_f
 * @return Interpolated viscosity
 */
inline double viscosity(double theta, double /*epsilon*/,
                        double nu_water, double nu_ferro)
{
    // Linear interpolation: Φ = (θ+1)/2, clamp to [0,1] for safety
    const double phi = 0.5 * (theta + 1.0);
    const double phi_clamped = (phi < 0.0) ? 0.0 : (phi > 1.0 ? 1.0 : phi);
    return nu_water * (1.0 - phi_clamped) + nu_ferro * phi_clamped;
}

// ============================================================================
// Density Ratio (Zhang Eq 4.2: SIGMOID, NOT linear)
//
//   Zhang:  ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))
//   Code:   ρ(θ) = 1 + r·H(θ/ε)
//
// Proof of equivalence: With Φ=(θ+1)/2, (1-2Φ)/ε = -θ/ε,
// so 1/(1+exp((1-2Φ)/ε)) = 1/(1+exp(-θ/ε)) = H(θ/ε).  ✓
//
// Interpolates between phases:
//   θ = +1 (ferrofluid)    → ρ ≈ 1 + r  (heavier)
//   θ = -1 (non-magnetic)  → ρ ≈ 1      (reference density)
//
// NOTE: Unlike chi and nu, Zhang's density uses a sigmoid (not linear).
//       This is already correctly implemented.
//
// CRITICAL: Must use θ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (Zhang Eq 4.4): r = 0.1  →  ρ ∈ [1.0, 1.1]
// ============================================================================

/**
 * @brief Density ratio ρ(θ) = 1 + r·H(θ/ε)  (Zhang Eq 4.2)
 *
 * @param theta    Phase field value (use θ^{n-1} for energy stability!)
 * @param epsilon  Interface thickness ε
 * @param r        Density ratio parameter r = (ρ_f - ρ_w)/ρ_w
 * @return Interpolated density ratio (dimensionless)
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
// NOTE: These are the ε-FREE forms. The assembler applies the 1/ε scaling
// from the paper:  (1/ε)f(θ) in Eq. 42b.
//
// Truncated to quadratic/linear outside [-1, 1] for overshoot safety.
// Inside the physical range |θ| ≤ 1, the functions are exact.
// ============================================================================

/**
 * @brief Double-well potential F(θ) = (1/4)(θ² - 1)²
 *
 * Used in CH diagnostics: E_CH = λ ∫ [ε/2 |∇θ|² + (1/ε)F(θ)] dΩ
 *
 * Truncated to quadratic outside [-1, 1].
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
 * Used in CH assembly (Eq. 42b): the nonlinear term in the ψ equation.
 * Applied with 1/ε scaling by the assembler.
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
 * Used for stability analysis and Newton linearization (if needed).
 *
 * Truncated to constant (= 2) outside [-1, 1].
 */
inline double double_well_second_derivative(double theta)
{
    if (theta <= -1.0 || theta >= 1.0)
        return 2.0;
    else
        return 3.0 * theta * theta - 1.0;
}

#endif // MATERIAL_PROPERTIES_H