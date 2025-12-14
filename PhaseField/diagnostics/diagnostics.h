// ============================================================================
// diagnostics/diagnostics.h - Simulation Diagnostics
//
// Computes and outputs diagnostic quantities:
//   - Mass conservation (∫θ dΩ)
//   - Cahn-Hilliard energy
//   - Phase field bounds (θ_min, θ_max)
//   - Velocity diagnostics (u_max, ∇·u)
//   - CFL number
//   - Force magnitudes
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>

#include "utilities/parameters.h"

#include <string>
#include <fstream>

/**
 * @brief Container for all diagnostic quantities
 */
struct DiagnosticData
{
    unsigned int step = 0;
    double time = 0.0;

    // Cahn-Hilliard diagnostics
    double mass = 0.0;           // ∫(θ+1)/2 dΩ (ferrofluid volume fraction)
    double mass_raw = 0.0;       // ∫θ dΩ (raw integral)
    double energy_ch = 0.0;      // ∫[ε/2|∇θ|² + (1/ε)F(θ)] dΩ
    double theta_min = 0.0;
    double theta_max = 0.0;

    // Navier-Stokes diagnostics
    double u_max = 0.0;          // max|u|
    double div_u_L2 = 0.0;       // ||∇·u||_L2
    double cfl = 0.0;            // u_max * dt / h_min

    // Force magnitudes (L2 norms)
    double F_cap_L2 = 0.0;       // ||F_cap||_L2
    double F_mag_L2 = 0.0;       // ||F_mag||_L2
    double F_grav_L2 = 0.0;      // ||F_grav||_L2

    // Poisson diagnostics
    double phi_min = 0.0;
    double phi_max = 0.0;

    // Solver diagnostics
    double ch_residual = 0.0;
    double poisson_residual = 0.0;
    double ns_residual = 0.0;
};

/**
 * @brief CSV logger for diagnostics
 */
class DiagnosticsLogger
{
public:
    DiagnosticsLogger() = default;

    /// Open CSV file and write header
    void open(const std::string& filename, bool ns_enabled, bool magnetic_enabled);

    /// Write one row of diagnostic data
    void write(const DiagnosticData& data);

    /// Close file
    void close();

    /// Check if file is open
    bool is_open() const { return file_.is_open(); }

private:
    std::ofstream file_;
    bool ns_enabled_ = false;
    bool magnetic_enabled_ = false;
};

// ============================================================================
// Diagnostic computation functions
// ============================================================================

/**
 * @brief Compute Cahn-Hilliard energy
 *
 * E_CH = ∫[ε/2 |∇θ|² + (1/ε) F(θ)] dΩ
 *
 * where F(θ) = (1/4)(θ² - 1)² is the double-well potential
 */
template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::FE_Q<dim>& fe,
    double epsilon);

/**
 * @brief Compute theta bounds (min and max)
 */
void compute_theta_bounds(
    const dealii::Vector<double>& theta_solution,
    double& theta_min,
    double& theta_max);

/**
 * @brief Compute maximum velocity magnitude
 */
double compute_u_max(
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution);

/**
 * @brief Compute L2 norm of velocity divergence
 *
 * ||∇·u||_L2 = sqrt(∫|∂ux/∂x + ∂uy/∂y|² dΩ)
 */
template <int dim>
double compute_div_u_L2(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::FE_Q<dim>& fe);

/**
 * @brief Compute force magnitudes (L2 norms)
 *
 * Computes ||F_cap||, ||F_mag||, ||F_grav|| over the domain.
 * Forces are computed at quadrature points and integrated.
 */
template <int dim>
void compute_force_magnitudes(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,  // nullptr if not magnetic
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,      // nullptr if not magnetic
    const Parameters& params,
    double time,
    double& F_cap_L2,
    double& F_mag_L2,
    double& F_grav_L2);

/**
 * @brief Compute all diagnostics in one pass
 *
 * More efficient than calling individual functions - shares FEValues setup.
 */
template <int dim>
DiagnosticData compute_all_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const dealii::Vector<double>* ux_solution,
    const dealii::Vector<double>* uy_solution,
    const dealii::FE_Q<dim>& fe_Q2,
    const Parameters& params,
    unsigned int step,
    double time,
    double dt,
    double h_min);

#endif // DIAGNOSTICS_H