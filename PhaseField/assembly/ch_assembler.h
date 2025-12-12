// ============================================================================
// assembly/ch_assembler.h - Cahn-Hilliard System Assembler
//
// Assembles the coupled Cahn-Hilliard system for phase field θ and 
// chemical potential ψ.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-b, p.505
// ============================================================================
#ifndef CH_ASSEMBLER_H
#define CH_ASSEMBLER_H

#include "core/phase_field.h"

/**
 * @brief Assembles the Cahn-Hilliard system
 *
 * Discrete scheme (Eq. 42a-b, p.505):
 *
 *   (δΘ^k/τ, Λ) - (U^k Θ^{k-1}, ∇Λ) - γ(∇Ψ^k, ∇Λ) = 0
 *
 *   (Ψ^k, Υ) + ε(∇Θ^k, ∇Υ) + (1/ε)(f(Θ^{k-1}), Υ) + (1/η)(δΘ^k, Υ) = 0
 *
 * where:
 *   - δΘ^k = Θ^k - Θ^{k-1}
 *   - f(θ) = θ³ - θ (derivative of double-well potential)
 *   - η ≤ ε is the stabilization parameter (Proposition 4.1)
 */
template <int dim>
class CHAssembler
{
public:
    explicit CHAssembler(PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Assemble the CH system matrix and RHS
     * @param dt Time step size τ
     * @param current_time Current simulation time
     */
    void assemble(double dt, double current_time);

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // CH_ASSEMBLER_H
