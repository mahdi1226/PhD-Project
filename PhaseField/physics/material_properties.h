// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Properties
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 16-18, p.501; Equation 43 for coefficient freezing
//
// CRITICAL FOR ENERGY STABILITY (Eq. 43, Theorem 4.1):
// =====================================================
// All material coefficients MUST be evaluated at θ^{k-1} (the OLD/FROZEN value),
// NOT at the current iterate θ^k. This is required for the discrete energy
// estimate to hold.
//
// CORRECT usage in assemblers:
//   const double nu = mat_props.viscosity(theta_OLD);      // ✓
//   const double chi = mat_props.susceptibility(theta_OLD); // ✓
//
// WRONG usage (breaks energy stability):
//   const double nu = mat_props.viscosity(theta_CURRENT);   // ✗
//
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @brief Phase-dependent material properties
 *
 * Sigmoid function (Eq. 18, p.501):
 *   H(x) = 1 / (1 + exp(-x))
 *
 * Viscosity (Eq. 17, p.501):
 *   ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
 *
 * Susceptibility (Eq. 17, p.501):
 *   χ_θ = χ₀ H(θ/ε)
 *
 * Permeability (Eq. 17, for Poisson):
 *   μ_θ = 1 + χ₀ H(θ/ε) = 1 + χ_θ
 *
 * Bounds (Eq. 16, p.501):
 *   min(ν_w, ν_f) ≤ ν_θ ≤ max(ν_w, ν_f)
 *   0 ≤ χ_θ ≤ χ₀
 *
 * Parameters (Section 6.2, p.520):
 *   ν_w = 1.0  (water/non-magnetic viscosity)
 *   ν_f = 2.0  (ferrofluid viscosity)
 *   χ₀ = 0.5   (ferrofluid susceptibility)
 *   ε = 0.01   (interface thickness)
 *
 * Constraint (Proposition 3.1, p.502):
 *   χ₀ ≤ 4 for energy stability
 *
 * =========================================================================
 * IMPORTANT: For energy stability (Eq. 43), always pass θ^{k-1} (old value)
 * to these functions, NOT the current iterate θ^k!
 * =========================================================================
 */
class MaterialProperties
{
public:
    /**
     * @brief Constructor with material parameters
     * @param nu_water Viscosity of non-magnetic phase ν_w
     * @param nu_ferro Viscosity of ferrofluid phase ν_f
     * @param chi_0 Susceptibility of ferrofluid χ₀
     * @param epsilon Interface thickness ε
     */
    MaterialProperties(double nu_water, double nu_ferro,
                       double chi_0, double epsilon);

    // ========================================================================
    // Smoothed Heaviside function
    // ========================================================================

    /**
     * @brief Smoothed Heaviside (sigmoid) H(x) = 1/(1+exp(-x))
     *
     * Eq. 18, p.501
     */
    static double heaviside(double x);

    /**
     * @brief Derivative of smoothed Heaviside H'(x) = H(x)(1-H(x))
     *
     * For linearization purposes
     */
    static double heaviside_derivative(double x);

    // ========================================================================
    // Material property evaluation
    // NOTE: For energy stability, always pass theta_old (θ^{k-1})!
    // ========================================================================

    /**
     * @brief Phase-dependent viscosity ν_θ
     *
     * ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
     * Eq. 17, p.501
     *
     * @param theta Phase field value θ (USE θ^{k-1} for energy stability!)
     */
    double viscosity(double theta) const;

    /**
     * @brief Phase-dependent susceptibility χ_θ
     *
     * χ_θ = χ₀ H(θ/ε)
     * Eq. 17, p.501
     *
     * @param theta Phase field value θ (USE θ^{k-1} for energy stability!)
     */
    double susceptibility(double theta) const;

    /**
     * @brief Phase-dependent permeability μ_θ
     *
     * μ_θ = 1 + χ₀ H(θ/ε) = 1 + χ_θ
     * Eq. 17, p.501 (for Poisson equation)
     *
     * @param theta Phase field value θ (USE θ^{k-1} for energy stability!)
     */
    double permeability(double theta) const;

    /**
     * @brief Phase-dependent density ratio for gravity
     *
     * ρ_θ = 1 + r H(θ/ε)
     * Eq. 19, p.501
     *
     * @param theta Phase field value θ (USE θ^{k-1} for energy stability!)
     * @param r Density contrast ratio
     */
    double density_ratio(double theta, double r) const;

    // ========================================================================
    // Double-well potential and derivatives
    // ========================================================================

    /**
     * @brief Double-well potential F(θ) = (1/4)(θ²-1)² (truncated)
     *
     * Truncated potential (Eq. 2, p.499):
     *   F(θ) = (θ+1)²         if θ ≤ -1
     *        = (1/4)(θ²-1)²   if -1 ≤ θ ≤ 1
     *        = (θ-1)²         if θ ≥ 1
     */
    static double double_well_potential(double theta);

    /**
     * @brief Derivative of double-well potential f(θ) = F'(θ)
     *
     * Truncated potential (Eq. 2-3, p.499):
     *   f(θ) = 2(θ+1)     if θ ≤ -1
     *        = θ³ - θ     if -1 ≤ θ ≤ 1
     *        = 2(θ-1)     if θ ≥ 1
     *
     * Bounds: |f(θ)| ≤ 2|θ| + 1
     */
    static double double_well_derivative(double theta);

    /**
     * @brief Second derivative of double-well potential f'(θ) = F''(θ)
     *
     *   f'(θ) = 2           if θ ≤ -1
     *         = 3θ² - 1     if -1 ≤ θ ≤ 1
     *         = 2           if θ ≥ 1
     *
     * Bounds: |f'(θ)| ≤ 2 (Lipschitz constant)
     *
     * This is needed for Newton linearization (if used instead of lagged scheme)
     */
    static double double_well_second_derivative(double theta);

    // ========================================================================
    // Accessors
    // ========================================================================

    double get_nu_water() const { return nu_water_; }
    double get_nu_ferro() const { return nu_ferro_; }
    double get_chi_0() const { return chi_0_; }
    double get_epsilon() const { return epsilon_; }

private:
    double nu_water_;   // ν_w = 1.0 (p.520)
    double nu_ferro_;   // ν_f = 2.0 (p.520)
    double chi_0_;      // χ₀ = 0.5 (p.520)
    double epsilon_;    // ε = 0.01 (p.522)
};

// ============================================================================
// Inline implementations for performance-critical functions
// ============================================================================

inline MaterialProperties::MaterialProperties(double nu_water, double nu_ferro,
                                               double chi_0, double epsilon)
    : nu_water_(nu_water)
    , nu_ferro_(nu_ferro)
    , chi_0_(chi_0)
    , epsilon_(epsilon)
{
    // Paper Proposition 3.1, p.502: χ₀ ≤ 4 required for energy stability
    if (chi_0 > 4.0)
    {
        throw std::invalid_argument(
            "MaterialProperties: chi_0 must be <= 4 for energy stability "
            "(Proposition 3.1, p.502). Got chi_0 = " + std::to_string(chi_0));
    }
}

inline double MaterialProperties::heaviside(double x)
{
    // Clamp to avoid overflow in exp
    if (x > 30.0) return 1.0;
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

inline double MaterialProperties::heaviside_derivative(double x)
{
    const double H = heaviside(x);
    return H * (1.0 - H);
}

inline double MaterialProperties::viscosity(double theta) const
{
    const double H = heaviside(theta / epsilon_);
    return nu_water_ + (nu_ferro_ - nu_water_) * H;
}

inline double MaterialProperties::susceptibility(double theta) const
{
    const double H = heaviside(theta / epsilon_);
    return chi_0_ * H;
}

inline double MaterialProperties::permeability(double theta) const
{
    return 1.0 + susceptibility(theta);
}

inline double MaterialProperties::density_ratio(double theta, double r) const
{
    const double H = heaviside(theta / epsilon_);
    return 1.0 + r * H;
}

inline double MaterialProperties::double_well_potential(double theta)
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

inline double MaterialProperties::double_well_derivative(double theta)
{
    if (theta <= -1.0)
        return 2.0 * (theta + 1.0);
    else if (theta >= 1.0)
        return 2.0 * (theta - 1.0);
    else
        return theta * theta * theta - theta;
}

inline double MaterialProperties::double_well_second_derivative(double theta)
{
    if (theta <= -1.0)
        return 2.0;
    else if (theta >= 1.0)
        return 2.0;
    else
        return 3.0 * theta * theta - 1.0;
}

#endif // MATERIAL_PROPERTIES_H