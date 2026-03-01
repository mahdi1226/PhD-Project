// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Functions
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
//            Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Convention: Code uses θ ∈ {-1, +1}. Zhang uses Φ ∈ {0, 1}.
//             Mapping: Φ = (θ+1)/2.
//
// Zhang's material property interpolation (p. B170) uses SIGMOID:
//   χ(Φ) = χ₀/(1+exp(-(2Φ-1)/ε))  → χ(θ) = χ₀·H(θ/ε)
//   ν(Φ) = ν_w + (ν_f-ν_w)/(1+exp(-(2Φ-1)/ε))  → ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)
//   ρ(Φ) = 1 + r/(1+exp((1-2Φ)/ε))  → ρ(θ) = 1 + r·H(θ/ε)
//
// ALL three (chi, nu, rho) use sigmoid H(θ/ε).
//
// All functions take explicit parameter values — NO GLOBALS.
//
// CRITICAL FOR ENERGY STABILITY:
//   All material coefficients MUST be evaluated at θ^{k-1} (the OLD value).
//
// Includes:
//   Poisson + Magnetization:
//     - Susceptibility χ(θ) = χ₀·H(θ/ε)  (sigmoid, Zhang p. B170)
//     - Permeability   μ(θ) = 1 + χ(θ)
//
//   Cahn-Hilliard:
//     - Double-well potential  F(θ)  = (1/16)(θ² - 1)²
//     - Double-well derivative f(θ)  = (θ³ - θ)/4
//     - Double-well curvature  f'(θ) = (3θ² - 1)/4
//
//   DERIVATION: Zhang uses Φ∈{0,1} with F(Φ) = ¼Φ²(1-Φ)².
//   Substituting Φ = (θ+1)/2:
//     F_Φ = ¼·((θ+1)/2)²·((1-θ)/2)² = (θ²-1)²/64
//   Energy: E = λ_Φ ∫[ε/2|∇Φ|² + (1/ε)F_Φ] = λ_Φ ∫[ε/8|∇θ|² + (θ²-1)²/(64ε)]
//   In θ-convention: E = λ_θ ∫[ε/2|∇θ|² + (1/ε)F_θ]
//   Matching:  λ_θ = λ_Φ/4,  F_θ = (θ²-1)²/16
//
//   Navier-Stokes:
//     - Viscosity  ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)     (sigmoid, Zhang p. B170)
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
// Magnetic Susceptibility (Zhang p. B170: SIGMOID)
//
//   Zhang:  χ(Φ) = χ₀ / (1 + exp(-(2Φ-1)/ε))   (Φ ∈ {0,1})
//   Code:   χ(θ) = χ₀ · H(θ/ε)                  (θ ∈ {-1,+1})
//
//   Proof:  Φ = (θ+1)/2  →  (2Φ-1)/ε = θ/ε
//           1/(1+exp(-(2Φ-1)/ε)) = 1/(1+exp(-θ/ε)) = H(θ/ε)  ✓
//
//   θ = +1 (ferrofluid)    → χ ≈ χ₀  (H(1/ε) ≈ 1)
//   θ = -1 (non-magnetic)  → χ ≈ 0   (H(-1/ε) ≈ 0)
//   θ =  0 (interface)     → χ = χ₀/2
// ============================================================================
inline double susceptibility(double theta, double epsilon, double chi_0)
{
    return chi_0 * heaviside(theta / epsilon);
}

// ============================================================================
// Susceptibility derivative dχ/dθ
//
//   χ(θ) = χ₀ · H(θ/ε)
//   dχ/dθ = (χ₀/ε) · H'(θ/ε) = (χ₀/ε) · H(θ/ε)(1 - H(θ/ε))
// ============================================================================
inline double susceptibility_derivative(double theta, double epsilon, double chi_0)
{
    return (chi_0 / epsilon) * heaviside_derivative(theta / epsilon);
}

// ============================================================================
// Magnetic Permeability
//
//   μ(θ) = 1 + χ(θ) = 1 + χ₀·H(θ/ε)
//
// Used in Poisson diagnostics: E_mag = ½∫μ(θ)|H|² dΩ
//
// NOTE: μ(θ) does NOT appear in the Poisson assembly (Eq 3.15).
// The paper's formulation eliminates μ from the weak form.
// ============================================================================
inline double permeability(double theta, double epsilon, double chi_0)
{
    return 1.0 + susceptibility(theta, epsilon, chi_0);
}

// ============================================================================
// Viscosity (Zhang p. B170: SIGMOID)
//
//   Zhang:  ν(Φ) = ν_w + (ν_f-ν_w)/(1+exp(-(2Φ-1)/ε))   (Φ ∈ {0,1})
//   Code:   ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)                (θ ∈ {-1,+1})
//
//   Proof:  Same H(θ/ε) equivalence as χ(θ) above.
//
// Interpolates between non-magnetic and ferrofluid phases:
//   θ = +1 (ferrofluid)    → ν ≈ ν_f   (higher viscosity)
//   θ = -1 (non-magnetic)  → ν ≈ ν_w   (lower viscosity)
//
// Used in NS assembly: (ν(θ) D(U), D(V))
//
// CRITICAL: Must use θ^{n-1} (LAGGED) for energy stability.
//
// Rosensweig (Zhang Eq 4.4): ν_w = 1.0, ν_f = 2.0  →  ν ∈ [1, 2]
// ============================================================================

/**
 * @brief Viscosity ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)  (sigmoid, Zhang p. B170)
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
// NOTE: All three material functions (chi, nu, rho) use sigmoid H(θ/ε).
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
// Double-Well Potential — θ∈{-1,+1} convention
//
// Zhang uses Φ∈{0,1}: F_Φ(Φ) = ¼Φ²(1-Φ)², f_Φ = ½Φ(1-Φ)(1-2Φ).
// Converting via Φ = (θ+1)/2 and matching λ_θ·E_θ = λ_Φ·E_Φ with λ_θ = λ_Φ/4:
//
//   F(θ) = (1/16)(θ² - 1)²
//   f(θ) = F'(θ) = (θ³ - θ)/4
//   f'(θ) = F''(θ) = (3θ² - 1)/4
//
// NOTE: These are the ε-FREE forms. The assembler applies the λ/ε scaling:
//   δE₁/δθ = (λ/ε)·f(θ)
//
// Truncated to quadratic/linear outside [-1, 1] for overshoot safety.
// Inside the physical range |θ| ≤ 1, the functions are exact.
// ============================================================================

/**
 * @brief Double-well potential F(θ) = (1/16)(θ² - 1)²
 *
 * Derived from Zhang's F(Φ) = ¼Φ²(1-Φ)² via Φ=(θ+1)/2 and λ_θ=λ_Φ/4.
 * Used in CH diagnostics: E_CH = λ ∫ [ε/2 |∇θ|² + (1/ε)F(θ)] dΩ
 *
 * Truncated to quadratic outside [-1, 1]: F ≈ (1/4)(θ∓1)²
 * (matching F(±1)=0, F'(±1)=0, F''(±1)=1/2)
 */
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
        return 0.0625 * t * t;  // (θ²-1)²/16
    }
}

/**
 * @brief Double-well derivative f(θ) = F'(θ) = (θ³ - θ)/4
 *
 * Derived from Zhang's f(Φ) = ½Φ(1-Φ)(1-2Φ) via Φ=(θ+1)/2.
 * Used in CH assembly: the nonlinear term in the ψ equation.
 * Applied with λ/ε scaling by the assembler.
 *
 * Truncated to linear outside [-1, 1]: f ≈ (1/2)(θ∓1)
 * (matching f(±1)=0, f'(±1)=1/2)
 */
inline double double_well_derivative(double theta)
{
    if (theta <= -1.0)
        return 0.5 * (theta + 1.0);
    else if (theta >= 1.0)
        return 0.5 * (theta - 1.0);
    else
        return 0.25 * (theta * theta * theta - theta);
}

/**
 * @brief Double-well second derivative f'(θ) = F''(θ) = (3θ² - 1)/4
 *
 * Used for stability analysis and Newton linearization (if needed).
 * min f'(θ) = -1/4 at θ=0  →  S1 ≥ λ/(4ε) for SAV convexity.
 *
 * Truncated to constant (= 1/2) outside [-1, 1].
 */
inline double double_well_second_derivative(double theta)
{
    if (theta <= -1.0 || theta >= 1.0)
        return 0.5;
    else
        return 0.25 * (3.0 * theta * theta - 1.0);
}

#endif // MATERIAL_PROPERTIES_H