// ============================================================================
// assembly/magnetization_assembler.h - Magnetization Equation Assembler
//
// Assembles the magnetization equation for m.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42c, p.505
// ============================================================================
#ifndef MAGNETIZATION_ASSEMBLER_H
#define MAGNETIZATION_ASSEMBLER_H

#include "core/phase_field.h"

/**
 * @brief Assembles the magnetization system
 *
 * Discrete scheme (Eq. 42c, p.505):
 *
 *   (δM^k/τ, Z) - B_h^m(U^k, Z, M^k) + (1/T)(M^k, Z) = (1/T)(χ_Θ H^k, Z)
 *
 * where:
 *   - δM^k = M^k - M^{k-1}
 *   - T is the relaxation time (range 10⁻⁵–10⁻⁹ s, p.500)
 *   - χ_Θ = χ(Θ^{k-1}) = χ₀ H(θ/ε) is phase-dependent susceptibility
 *   - H^k = ∇Φ^k is the magnetic field
 *   - B_h^m is the skew-symmetric trilinear form (Eq. 38)
 *
 * Quasi-equilibrium approximation (T → 0):
 *   When T is very small, magnetization relaxes almost instantly:
 *   m ≈ χ_θ h
 *   This bypasses solving the PDE and directly computes m from h.
 */
template <int dim>
class MagnetizationAssembler
{
public:
    explicit MagnetizationAssembler(PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Assemble the magnetization system matrix and RHS
     * @param dt Time step size τ
     * @param current_time Current simulation time
     */
    void assemble(double dt, double current_time);

private:
    PhaseFieldProblem<dim>& problem_;
};

#endif // MAGNETIZATION_ASSEMBLER_H
