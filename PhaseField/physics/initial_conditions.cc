// ============================================================================
// physics/initial_conditions.cc - Initial Conditions Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.522
// ============================================================================

#include "initial_conditions.h"
#include "output/logger.h"
#include <cmath>

// ============================================================================
// InitialTheta
//
// θ₀(x,y) = tanh((y - pool_depth) / (ε√2))
//
// This is the equilibrium profile for Cahn-Hilliard.
// Pool depth = 0.2 (p.522)
// ε = 0.01 (p.522)
// ============================================================================
template <int dim>
InitialTheta<dim>::InitialTheta(double pool_depth, double epsilon)
    : dealii::Function<dim>(1)
    , pool_depth_(pool_depth)
    , epsilon_(epsilon)
{
    Logger::info("InitialTheta: pool_depth = " + std::to_string(pool_depth_) +
                 ", ε = " + std::to_string(epsilon_));
}

template <int dim>
double InitialTheta<dim>::value(const dealii::Point<dim>& p,
                                 const unsigned int /*component*/) const
{
    // θ₀ = tanh((y - pool_depth) / (ε√2))
    // Note: tanh profile is equilibrium solution for CH
    const double y = p[1];
    const double interface_width = epsilon_ * std::sqrt(2.0);
    return std::tanh((y - pool_depth_) / interface_width);
}

// ============================================================================
// InitialPsi
//
// ψ₀ ≈ -(1/ε)(θ₀³ - θ₀) from equilibrium
// ============================================================================
template <int dim>
InitialPsi<dim>::InitialPsi(double pool_depth, double epsilon)
    : dealii::Function<dim>(1)
    , pool_depth_(pool_depth)
    , epsilon_(epsilon)
{
    Logger::info("InitialPsi: pool_depth = " + std::to_string(pool_depth_) +
                 ", ε = " + std::to_string(epsilon_));
}

template <int dim>
double InitialPsi<dim>::value(const dealii::Point<dim>& p,
                               const unsigned int /*component*/) const
{
    // First compute θ₀
    const double y = p[1];
    const double interface_width = epsilon_ * std::sqrt(2.0);
    const double theta = std::tanh((y - pool_depth_) / interface_width);
    
    // ψ₀ = -(1/ε)(θ³ - θ) from equilibrium f(θ) = θ³ - θ
    const double f_theta = theta * theta * theta - theta;
    return -f_theta / epsilon_;
}

// ============================================================================
// InitialVelocity
//
// u₀ = 0 (fluid at rest)
// ============================================================================
template <int dim>
InitialVelocity<dim>::InitialVelocity()
    : dealii::Function<dim>(1)
{
    Logger::info("InitialVelocity: u₀ = 0");
}

template <int dim>
double InitialVelocity<dim>::value(const dealii::Point<dim>& /*p*/,
                                    const unsigned int /*component*/) const
{
    return 0.0;
}

// ============================================================================
// InitialMagnetization
//
// m₀ = 0 (since h_a(0) = 0 when α_s(0) = 0)
// ============================================================================
template <int dim>
InitialMagnetization<dim>::InitialMagnetization()
    : dealii::Function<dim>(1)
{
    Logger::info("InitialMagnetization: m₀ = 0");
}

template <int dim>
double InitialMagnetization<dim>::value(const dealii::Point<dim>& /*p*/,
                                         const unsigned int /*component*/) const
{
    return 0.0;
}

// Explicit instantiations
template class InitialTheta<2>;
template class InitialTheta<3>;
template class InitialPsi<2>;
template class InitialPsi<3>;
template class InitialVelocity<2>;
template class InitialVelocity<3>;
template class InitialMagnetization<2>;
template class InitialMagnetization<3>;
