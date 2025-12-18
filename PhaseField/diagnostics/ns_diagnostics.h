// ============================================================================
// diagnostics/ns_diagnostics.h - Navier-Stokes Diagnostics
//
// Computes diagnostic quantities for the NS subsystem:
//   - Velocity bounds and norms
//   - Pressure bounds
//   - Kinetic energy: ½∫|U|² dx
//   - Divergence (incompressibility): ||div U||
//   - CFL number
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_DIAGNOSTICS_H
#define NS_DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <string>

/**
 * @brief Container for NS diagnostic quantities
 */
struct NSDiagnostics
{
    // Velocity bounds
    double ux_min = 0.0;
    double ux_max = 0.0;
    double uy_min = 0.0;
    double uy_max = 0.0;

    // Velocity norms
    double U_L2_norm = 0.0;     // ||U||_{L²}
    double U_max = 0.0;         // max |U|

    // Pressure bounds
    double p_min = 0.0;
    double p_max = 0.0;

    // Kinetic energy: ½∫|U|² dx
    double kinetic_energy = 0.0;

    // Incompressibility: div(U)
    double div_U_L2 = 0.0;      // ||div U||_{L²}
    double div_U_max = 0.0;     // max |div U|

    // CFL number: max|U| * dt / h_min
    double cfl = 0.0;

    /**
     * @brief Print diagnostics to console (single line)
     */
    void print(unsigned int step, double time) const;

    /**
     * @brief Get header string for CSV output
     */
    static std::string header();

    /**
     * @brief Get CSV-formatted data string
     */
    std::string to_csv(unsigned int step, double time) const;
};

/**
 * @brief Compute NS diagnostics
 *
 * @param ux_dof_handler  DoFHandler for velocity x
 * @param uy_dof_handler  DoFHandler for velocity y
 * @param p_dof_handler   DoFHandler for pressure
 * @param ux_solution     Velocity x solution
 * @param uy_solution     Velocity y solution
 * @param p_solution      Pressure solution
 * @param dt              Time step (for CFL)
 * @param h_min           Minimum mesh size (for CFL)
 * @return NSDiagnostics struct with computed values
 */
template <int dim>
NSDiagnostics compute_ns_diagnostics(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::Vector<double>& p_solution,
    double dt,
    double h_min);

#endif // NS_DIAGNOSTICS_H