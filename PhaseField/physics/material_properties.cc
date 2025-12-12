// ============================================================================
// physics/material_properties.cc - Material Properties Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 16-18, p.501; Section 6.2, p.520
// ============================================================================

#include "material_properties.h"
#include "output/logger.h"
#include <cmath>
#include <algorithm>

MaterialProperties::MaterialProperties(double nu_water, double nu_ferro,
                                       double chi_0, double epsilon)
    : nu_water_(nu_water)
    , nu_ferro_(nu_ferro)
    , chi_0_(chi_0)
    , epsilon_(epsilon)
{
    Logger::info("MaterialProperties constructor");
    Logger::info("  ν_w = " + std::to_string(nu_water_) + " (p.520)");
    Logger::info("  ν_f = " + std::to_string(nu_ferro_) + " (p.520)");
    Logger::info("  χ₀ = " + std::to_string(chi_0_) + " (p.520)");
    Logger::info("  ε = " + std::to_string(epsilon_) + " (p.522)");
    
    // Check energy stability constraint (Proposition 3.1, p.502)
    if (chi_0_ > 4.0)
    {
        Logger::info("  WARNING: χ₀ > 4 violates energy stability (Prop. 3.1)");
    }
}

// ============================================================================
// heaviside()
//
// Smoothed Heaviside function (Eq. 18, p.501):
//   H(x) = 1 / (1 + exp(-x))
//
// Properties:
//   H(x) → 0 as x → -∞
//   H(x) → 1 as x → +∞
//   H(0) = 0.5
// ============================================================================
double MaterialProperties::heaviside(double x)
{
    // Clamp to avoid overflow in exp
    if (x > 30.0) return 1.0;
    if (x < -30.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// ============================================================================
// viscosity()
//
// Phase-dependent viscosity (Eq. 17, p.501):
//   ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
//
// For θ = +1 (ferrofluid):  H(1/ε) ≈ 1, so ν_θ ≈ ν_f
// For θ = -1 (non-magnetic): H(-1/ε) ≈ 0, so ν_θ ≈ ν_w
// ============================================================================
double MaterialProperties::viscosity(double theta) const
{
    const double H = heaviside(theta / epsilon_);
    return nu_water_ + (nu_ferro_ - nu_water_) * H;
}

// ============================================================================
// susceptibility()
//
// Phase-dependent susceptibility (Eq. 17, p.501):
//   χ_θ = χ₀ H(θ/ε)
//
// For θ = +1 (ferrofluid):  χ_θ ≈ χ₀
// For θ = -1 (non-magnetic): χ_θ ≈ 0
// ============================================================================
double MaterialProperties::susceptibility(double theta) const
{
    const double H = heaviside(theta / epsilon_);
    return chi_0_ * H;
}

// ============================================================================
// double_well_derivative()
//
// Truncated double-well potential derivative (Eq. 2-3, p.499):
//
// F(θ) = (θ+1)²           if θ ≤ -1
//      = (1/4)(θ² - 1)²   if -1 ≤ θ ≤ 1
//      = (θ-1)²           if θ ≥ 1
//
// f(θ) = F'(θ):
//      = 2(θ+1)           if θ ≤ -1
//      = θ³ - θ           if -1 ≤ θ ≤ 1
//      = 2(θ-1)           if θ ≥ 1
//
// Bounds (Eq. 3, p.499):
//   |f(θ)| ≤ 2|θ| + 1
//   |f'(θ)| ≤ 2
// ============================================================================
double MaterialProperties::double_well_derivative(double theta)
{
    if (theta <= -1.0)
    {
        // f(θ) = 2(θ + 1) for θ ≤ -1
        return 2.0 * (theta + 1.0);
    }
    else if (theta >= 1.0)
    {
        // f(θ) = 2(θ - 1) for θ ≥ 1
        return 2.0 * (theta - 1.0);
    }
    else
    {
        // f(θ) = θ³ - θ for -1 ≤ θ ≤ 1
        return theta * theta * theta - theta;
    }
}
