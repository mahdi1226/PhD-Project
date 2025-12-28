// ============================================================================
// diagnostics/ch_diagnostics.h - Cahn-Hilliard Diagnostics (PAPER_MATCH v2)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// FIXES:
//   - Uses params.physics.epsilon instead of global constant
//   - Energy formula: E = ε/2|∇θ|² + (1/ε)W(θ) (NO mobility factor)
//
// Diagnostics:
//   - θ bounds: should stay in [-1, 1]
//   - Mass: ∫θ dΩ (conserved for Neumann BCs)
//   - Energy: E_CH = ∫[ε/2|∇θ|² + (1/ε)F(θ)] dΩ
//   - Energy decay: dE/dt ≤ 0 (discrete energy law)
// ============================================================================
#ifndef CH_DIAGNOSTICS_H
#define CH_DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>

#include "utilities/parameters.h"

#include <string>
#include <fstream>

// ============================================================================
// CH Diagnostic Data
// ============================================================================
struct CHDiagnosticData
{
    unsigned int step = 0;
    double time = 0.0;

    // Bounds
    double theta_min = 0.0;
    double theta_max = 0.0;
    bool bounds_violated = false;

    // Mass
    double mass = 0.0;           // ∫θ dΩ

    // Energy (Eq. 42 in paper - NO mobility factor!)
    // E_CH = ∫[ε/2|∇θ|² + (1/ε)W(θ)] dΩ
    double energy = 0.0;         // E_CH
    double energy_prev = 0.0;    // E_CH from previous step
    double energy_rate = 0.0;    // (E - E_prev) / dt
    bool energy_increasing = false;
};

// ============================================================================
// Compute θ bounds
// ============================================================================
void compute_theta_bounds(
    const dealii::Vector<double>& theta_solution,
    double& theta_min,
    double& theta_max);

// ============================================================================
// Check if θ bounds are violated (outside [-1-tol, 1+tol])
// ============================================================================
bool check_theta_bounds(
    double theta_min,
    double theta_max,
    double tolerance = 0.01);

// ============================================================================
// Compute mass ∫θ dΩ
// ============================================================================
template <int dim>
double compute_mass(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution);

// ============================================================================
// Compute CH energy (Eq. 42 in paper)
//
// E_CH = ∫[ε/2 |∇θ|² + (1/ε) W(θ)] dΩ
//
// where W(θ) = (1/4)(θ² - 1)² is the double-well potential
//
// NOTE: Mobility γ does NOT appear in energy formula!
//       It only appears in the evolution equation: ∂θ/∂t = γ Δψ
// ============================================================================
template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params);

// ============================================================================
// Compute all CH diagnostics (UPDATED to use Parameters)
// ============================================================================
template <int dim>
CHDiagnosticData compute_ch_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    unsigned int step,
    double time,
    double dt,
    double energy_prev);

// ============================================================================
// Print CH diagnostics (console output)
// ============================================================================
void print_ch_diagnostics(const CHDiagnosticData& data, bool verbose = false);

// ============================================================================
// CH Diagnostics Logger (CSV)
// ============================================================================
class CHDiagnosticsLogger
{
public:
    CHDiagnosticsLogger() = default;

    void open(const std::string& filename);
    void write(const CHDiagnosticData& data);
    void close();
    bool is_open() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

#endif // CH_DIAGNOSTICS_H