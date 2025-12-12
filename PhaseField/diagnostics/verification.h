// ============================================================================
// diagnostics/verification.h - Solution Verification and Diagnostics
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef VERIFICATION_H
#define VERIFICATION_H

#include "core/phase_field.h"

/**
 * @brief Verification and diagnostic tools
 *
 * Provides:
 *   - Mass conservation check for θ (Eq. 4, p.499)
 *   - Energy computation (Eq. 22-24, p.502)
 *   - L2/H1 norm computation
 *   - MMS verification (Method of Manufactured Solutions)
 */
template <int dim>
class Verification
{
public:
    explicit Verification(const PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Check mass conservation ∫_Ω θ dx = const
     * 
     * Eq. 4, p.499: d/dt ∫_Ω θ dx = 0
     */
    double compute_mass() const;
    
    /**
     * @brief Compute total energy E(u, m, h, θ; t)
     * 
     * Eq. 22, p.502:
     *   E = ½‖u‖² + (μ₀/2)‖m‖² + (μ₀/2)‖h‖² 
     *     + (λ/2)‖∇θ‖² + (λ/2ε)∫F(θ)dx
     */
    double compute_energy() const;
    
    /**
     * @brief Compute interface area (proxy for |∇θ|)
     */
    double compute_interface_area() const;
    
    /**
     * @brief Print diagnostic summary
     */
    void print_diagnostics() const;

private:
    const PhaseFieldProblem<dim>& problem_;
};

#endif // VERIFICATION_H
