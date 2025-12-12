// ============================================================================
// assembly/poisson_assembler.h - Poisson (Magnetostatics) Assembler
//
// Assembles the Poisson equation for magnetic potential φ.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 13, 14d, 42d, p.500-505
// ============================================================================
#ifndef POISSON_ASSEMBLER_H
#define POISSON_ASSEMBLER_H

#include "core/phase_field.h"

/**
 * @brief Assembles the Poisson (magnetostatics) system
 *
 * Strong form (Eq. 13, 14d, p.500-501):
 *   -Δφ = ∇·(m - h_a)     in Ω
 *   ∂_n φ = (h_a - m)·n   on Γ
 *
 * Discrete scheme (Eq. 42d, p.505):
 *   (∇Φ^k, ∇X) = (h_a^k - M^k, ∇X)
 *
 * After solving: H^k = ∇Φ^k
 *
 * Applied field h_a (Eq. 97-98, p.519):
 *   2D dipole potential: φ_s(x) = d·(x_s - x) / |x_s - x|²
 *   h_a = Σ_s α_s ∇φ_s
 *
 * Simplification (Section 5, p.510):
 *   When χ₀ ≪ 1, can skip Poisson and set h := h_a directly
 */
template <int dim>
class PoissonAssembler
{
public:
    explicit PoissonAssembler(PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Assemble the Poisson system matrix and RHS
     * @param current_time Current simulation time (for time-dependent h_a)
     */
    void assemble(double current_time);

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // POISSON_ASSEMBLER_H
