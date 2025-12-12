// ============================================================================
// assembly/ns_assembler.h - Navier-Stokes System Assembler
//
// Assembles the coupled Navier-Stokes system for velocity u and pressure p.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42e-f, p.505
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include "core/phase_field.h"

/**
 * @brief Assembles the Navier-Stokes system
 *
 * Discrete scheme (Eq. 42e-f, p.505):
 *
 *   (δU^k/τ, V) + B_h(U^{k-1}, U^k, V) + (ν_Θ T(U^k), T(V)) - (P^k, div V)
 *       = μ₀ B_h^m(V, H^k, M^k) + (λ/ε)(Θ^{k-1} ∇Ψ^k, V)
 *
 *   (Q, div U^k) = 0
 *
 * where:
 *   - T(u) = ½(∇u + ∇u^T) is the symmetric gradient
 *   - B_h is the skew-symmetric convection form (Eq. 37)
 *   - B_h^m is the magnetic trilinear form (Eq. 38)
 *   - ν_Θ = ν(Θ^{k-1}) is phase-dependent viscosity
 *
 * Forces on RHS:
 *   - Kelvin force: μ₀ B_h^m(V, H^k, M^k) = μ₀(m·∇)h
 *   - Capillary force: (λ/ε)(θ∇ψ, v)
 *   - Optional gravity: ((1 + rH(θ/ε))g, v) [Eq. 19]
 */
template <int dim>
class NSAssembler
{
public:
    explicit NSAssembler(PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Assemble the NS system matrix and RHS
     * @param dt Time step size τ
     * @param current_time Current simulation time
     */
    void assemble(double dt, double current_time);

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // NS_ASSEMBLER_H
