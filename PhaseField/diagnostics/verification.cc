// ============================================================================
// diagnostics/verification.cc - Verification Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "verification.h"
#include "output/logger.h"

template <int dim>
Verification<dim>::Verification(const PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("Verification constructor");
}

template <int dim>
double Verification<dim>::compute_mass() const
{
    Logger::info("  Verification::compute_mass() [skeleton]");
    
    // TODO: Compute ∫_Ω θ dx
    // Mass conservation (Eq. 4, p.499): d/dt ∫_Ω θ dx = 0
    
    return 0.0;
}

template <int dim>
double Verification<dim>::compute_energy() const
{
    Logger::info("  Verification::compute_energy() [skeleton]");
    
    // TODO: Compute total energy (Eq. 22, p.502)
    // E = ½‖u‖² + (μ₀/2)‖m‖² + (μ₀/2)‖h‖² + (λ/2)‖∇θ‖² + (λ/2ε)∫F(θ)dx
    
    return 0.0;
}

template <int dim>
double Verification<dim>::compute_interface_area() const
{
    Logger::info("  Verification::compute_interface_area() [skeleton]");
    
    // TODO: Compute ∫_Ω |∇θ| dx as proxy for interface length/area
    
    return 0.0;
}

template <int dim>
void Verification<dim>::print_diagnostics() const
{
    Logger::info("  Verification::print_diagnostics()");
    Logger::info("    Mass: " + std::to_string(compute_mass()));
    Logger::info("    Energy: " + std::to_string(compute_energy()));
    Logger::info("    Interface area: " + std::to_string(compute_interface_area()));
}

template class Verification<2>;
template class Verification<3>;
