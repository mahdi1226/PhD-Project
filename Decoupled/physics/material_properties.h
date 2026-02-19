// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Functions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Eq. 2-3 (double-well), Eq. 17-19 (Heaviside interpolation)
//
// All functions take explicit parameter values — NO GLOBALS.
//
// CRITICAL FOR ENERGY STABILITY (Theorem 4.1):
//   All material coefficients MUST be evaluated at θ^{k-1} (the OLD value).
//
// Includes:
//   Poisson + Magnetization:
//     - Smoothed Heaviside H(x)
//     - Susceptibility χ(θ) = χ₀ H(θ/ε)          (Eq. 18)
//     - Permeability   μ(θ) = 1 + χ(θ)
//
//   Cahn-Hilliard:
//     - Double-well potential  F(θ)  = (1/4)(θ² - 1)²     (Eq. 2)
//     - Double-well derivative f(θ)  = θ³ - θ              (Eq. 3)
//     - Double-well curvature  f'(θ) = 3θ² - 1
//
//   Navier-Stokes:
//     - Viscosity  ν(θ) = ν_w + (ν_f - ν_w) H(θ/ε)       (Eq. 17)
//     - Density    ρ(θ) = 1 + r H(θ/ε)                    (Eq. 19)
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
// Magnetic Susceptibility (Eq. 18, p.501)
//
//   χ(θ) = χ₀ H(θ/ε)
//
//   θ = +1 (ferrofluid)    → χ ≈ χ₀
//   θ = -1 (non-magnetic)  → χ ≈ 0
// ============================================================================
inline double susceptibility(double theta, double epsilon, double chi_0)
{
    return chi_0 * heaviside(theta / epsilon);
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
// Viscosity (Eq. 17, p.501)
//
//   ν(θ) = ν_w + (ν_f - ν_w) H(θ/ε)
//
// Interpolates between non-magnetic and ferrofluid phases:
//   θ = +1 (ferrofluid)    → ν ≈ ν_f   (higher viscosity)
//   θ = -1 (non-magnetic)  → ν ≈ ν_w   (lower viscosity)
//
// Used in NS assembly (Eq. 42e): (ν(θ) T(U), T(V))
//
// CRITICAL: Must use θ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (§6.2): ν_w = 1.0, ν_f = 2.0  →  ν ∈ [1, 2]
// ============================================================================

/**
 * @brief Viscosity ν(θ) = ν_w + (ν_f - ν_w) H(θ/ε)
 *
 * @param theta     Phase field value (use θ^{n-1} for energy stability!)
 * @param epsilon   Interface thickness ε
 * @param nu_water  Viscosity of non-magnetic phase ν_w
 * @param nu_ferro  Viscosity of ferrofluid phase ν_f
 * @return Interpolated viscosity
 */
inline double viscosity(double theta, double epsilon,
                        double nu_water, double nu_ferro)
{
    return nu_water + (nu_ferro - nu_water) * heaviside(theta / epsilon);
}

// ============================================================================
// Density Ratio (Eq. 19, p.502)
//
//   ρ(θ) = 1 + r H(θ/ε)
//
// where r = (ρ_ferro - ρ_water) / ρ_water is the density ratio parameter.
//
// Interpolates between phases:
//   θ = +1 (ferrofluid)    → ρ ≈ 1 + r  (heavier)
//   θ = -1 (non-magnetic)  → ρ ≈ 1      (reference density)
//
// Used in NS assembly (Eq. 42e):
//   - Mass term:    ρ(θ)(U^n - U^{n-1})/τ
//   - Convection:   ρ(θ) B_h(U; U, V)
//   - Gravity:      ρ(θ) g
//
// CRITICAL: Must use θ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (§6.2): r = 0.1  →  ρ ∈ [1.0, 1.1]
// ============================================================================

/**
 * @brief Density ratio ρ(θ) = 1 + r H(θ/ε)
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