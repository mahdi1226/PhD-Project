// ============================================================================
// physics/initial_conditions.h - Initial Conditions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 41, p.505; Section 6.2, p.522
// ============================================================================
#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

/**
 * @brief Initial condition for phase field θ
 *
 * Configuration (Section 6.2, p.522):
 *   "ferrofluid pool of 0.2 units of depth"
 *
 * Our assumption (not explicitly stated in paper):
 *   θ₀(x,y) = tanh((y - pool_depth) / (ε√2))
 *
 * This gives:
 *   θ ≈ -1 (non-magnetic) for y > pool_depth
 *   θ ≈ +1 (ferrofluid) for y < pool_depth
 *
 * Parameters:
 *   pool_depth = 0.2 (p.522)
 *   ε = 0.01 (p.522)
 */
template <int dim>
class InitialTheta : public dealii::Function<dim>
{
public:
    InitialTheta(double pool_depth, double epsilon);
    
    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override;

private:
    double pool_depth_;  // 0.2 (p.522)
    double epsilon_;     // ε = 0.01 (p.522)
};

/**
 * @brief Initial condition for chemical potential ψ
 *
 * From equilibrium (Eq. 14b with ∂θ/∂t = 0):
 *   ψ = εΔθ - (1/ε)f(θ)
 *
 * For smooth tanh profile, |Δθ| is small, so approximately:
 *   ψ₀ ≈ -(1/ε)(θ³ - θ)
 */
template <int dim>
class InitialPsi : public dealii::Function<dim>
{
public:
    InitialPsi(double pool_depth, double epsilon);
    
    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override;

private:
    double pool_depth_;
    double epsilon_;
};

/**
 * @brief Initial condition for velocity u
 *
 * Assumption (not stated in paper):
 *   u₀ = 0 (fluid at rest)
 *
 * Paper states (p.522): "ferrofluid pool ... at rest at t = 0"
 */
template <int dim>
class InitialVelocity : public dealii::Function<dim>
{
public:
    InitialVelocity();
    
    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override;
};

/**
 * @brief Initial condition for magnetization m
 *
 * Assumption (quasi-equilibrium):
 *   m₀ = χ_θ h_a(0)
 *
 * At t = 0, dipole intensity α_s(0) = 0, so h_a(0) = 0.
 * Therefore: m₀ = 0
 */
template <int dim>
class InitialMagnetization : public dealii::Function<dim>
{
public:
    InitialMagnetization();
    
    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override;
};

#endif // INITIAL_CONDITIONS_H
