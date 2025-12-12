// ============================================================================
// physics/material_properties.h - Phase-Dependent Material Properties
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 16-18, p.501
// ============================================================================
#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <cmath>

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
    
    /**
     * @brief Smoothed Heaviside function H(x) = 1/(1+exp(-x))
     * 
     * Eq. 18, p.501
     */
    static double heaviside(double x);
    
    /**
     * @brief Phase-dependent viscosity ν_θ
     * 
     * ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
     * Eq. 17, p.501
     * 
     * @param theta Phase field value θ ∈ [-1, 1]
     */
    double viscosity(double theta) const;
    
    /**
     * @brief Phase-dependent susceptibility χ_θ
     * 
     * χ_θ = χ₀ H(θ/ε)
     * Eq. 17, p.501
     * 
     * @param theta Phase field value θ ∈ [-1, 1]
     */
    double susceptibility(double theta) const;
    
    /**
     * @brief Derivative of double-well potential f(θ) = F'(θ)
     * 
     * Truncated potential (Eq. 2-3, p.499):
     *   f(θ) = 2(θ+1)     if θ ≤ -1
     *        = θ³ - θ     if -1 ≤ θ ≤ 1
     *        = 2(θ-1)     if θ ≥ 1
     * 
     * Bounds: |f(θ)| ≤ 2|θ| + 1, |f'(θ)| ≤ 2
     */
    static double double_well_derivative(double theta);

private:
    double nu_water_;   // ν_w = 1.0 (p.520)
    double nu_ferro_;   // ν_f = 2.0 (p.520)
    double chi_0_;      // χ₀ = 0.5 (p.520)
    double epsilon_;    // ε = 0.01 (p.522)
};

#endif // MATERIAL_PROPERTIES_H
