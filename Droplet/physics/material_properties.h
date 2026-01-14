// ============================================================================
// physics/material_properties.h - Physical Constants and Material Functions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// ALL physical constants are here. ONE place, ONE value.
// If compiler complains about redefinition, GOOD - remove the duplicate.
//
// For Hedgehog (Section 6.3), change: epsilon=0.005, chi_0=0.9, lambda=0.025
//
// CRITICAL FOR ENERGY STABILITY (Eq. 43, Theorem 4.1):
// All material coefficients MUST be evaluated at θ^{k-1} (the OLD/FROZEN value)
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>

// ============================================================================
// Physical Constants (Section 6.2, p.520-522)
// ============================================================================

// Cahn-Hilliard (Eq. 14a-14b)
constexpr double epsilon = 0.01;            // interface thickness
constexpr double mobility = 0.0002;            // mobility it is gamma : c++/clang naming collision

// Capillary (Eq. 14e)
constexpr double lambda = 0.05;             // capillary coefficient

// Viscosity (Eq. 17, p.501)
constexpr double nu_water = 1.0;            // non-magnetic phase
constexpr double nu_ferro = 2.0;            // ferrofluid phase

// Magnetic (Section 6.2, p.520)
constexpr double chi_0 = 0.5;               // susceptibility (must be <= 4)
constexpr double mu_0 = 1.0;                // permeability of free space
constexpr double tau_M = 0.0;               // magnetization relaxation time

// Langevin nonlinear magnetization (Afkhami 2010, Rosensweig 1985)
// M = M_s * L(ξ) where L(ξ) = coth(ξ) - 1/ξ, ξ = μ₀mH/(k_B T)
// For χ(H): χ = 3χ₀ L(ξ)/ξ where ξ = H/H_c, H_c = M_s/(3χ₀)
constexpr double M_s = 15000.0;             // saturation magnetization [A/m]
constexpr double H_c = 10000.0;             // characteristic field [A/m] = M_s/(3χ₀)

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

/** ν_θ = ν_w + (ν_f - ν_w) H(θ/ε) */
inline double viscosity(double theta)
{
    return nu_water + (nu_ferro - nu_water) * heaviside(theta / epsilon);
}

/** χ_θ = χ₀ H(θ/ε) */
inline double susceptibility(double theta)
{
    return chi_0 * heaviside(theta / epsilon);
}

/** Langevin function L(ξ) = coth(ξ) - 1/ξ
 *  Taylor expansion for small ξ: L(ξ) ≈ ξ/3 - ξ³/45 + ...
 */
inline double langevin_function(double xi)
{
    if (std::abs(xi) < 1e-4)
        return xi / 3.0 - xi * xi * xi / 45.0;  // Taylor for numerical stability
    return 1.0 / std::tanh(xi) - 1.0 / xi;
}

/** Field-dependent susceptibility χ(θ, H) with Langevin saturation
 *  Linear: χ = χ₀ H(θ/ε)
 *  Langevin: χ = 3χ₀ L(ξ)/ξ where ξ = H/H_c
 *
 *  @param theta        Phase field value
 *  @param H_mag        Magnetic field magnitude |H|
 *  @param use_langevin If true, apply Langevin saturation
 */
inline double susceptibility_langevin(double theta, double H_mag, bool use_langevin)
{
    const double chi_linear = chi_0 * heaviside(theta / epsilon);

    if (!use_langevin || H_mag < 1e-10)
        return chi_linear;

    // Langevin model: χ(H) = 3χ₀ L(ξ)/ξ where ξ = H/H_c
    const double xi = H_mag / H_c;
    const double L_xi = langevin_function(xi);

    // Scale by phase field (ferrofluid vs non-magnetic)
    const double phase_factor = heaviside(theta / epsilon);
    return 3.0 * chi_0 * phase_factor * L_xi / xi;
}

/** μ_θ = 1 + χ_θ */
inline double permeability(double theta)
{
    return 1.0 + susceptibility(theta);
}

/** ρ_θ = 1 + r H(θ/ε) */
inline double density_ratio(double theta)
{
    return 1.0 + r * heaviside(theta / epsilon);
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

#endif // MATERIAL_PROPERTIES_H