// ============================================================================
// diagnostics/validation_diagnostics.h - Rosensweig Validation Diagnostics
//
// Extends interface tracking to measure wavelength and compare against
// linear stability theory: lambda_c = 2*pi*sqrt(sigma/(rho*g))
//
// Integrates with existing StepData and logging infrastructure.
//
// Usage:
//   // After computing StepData:
//   update_validation_diagnostics<dim>(data,
//       theta_dof_handler, theta_solution, params, comm);
//
//   // At end of simulation:
//   RosensweigValidation validation = validate_rosensweig(data, params);
//   validation.print_summary(std::cout);
//
// Reference: Cowley & Rosensweig, J. Fluid Mech. 30 (1967) 671-688
// ============================================================================
#ifndef VALIDATION_DIAGNOSTICS_H
#define VALIDATION_DIAGNOSTICS_H

#include "diagnostics/step_data.h"
#include "utilities/parameters.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// ============================================================================
// Interface Profile - y(x) along the interface
// ============================================================================
struct InterfaceProfile
{
    std::vector<double> x;              // x-coordinates
    std::vector<double> y;              // y-coordinates (interface height)

    double amplitude = 0.0;             // (y_max - y_min) / 2
    double y_mean = 0.0;                // Mean interface height
    unsigned int n_peaks = 0;           // Number of peaks detected
    double dominant_wavelength = 0.0;   // Estimated wavelength

    bool valid = false;
};

// ============================================================================
// Rosensweig Validation Result
// ============================================================================
struct RosensweigValidation
{
    // Physical parameters used
    double surface_tension = 0.0;
    double density = 0.0;
    double gravity = 0.0;

    // Theoretical predictions
    double lambda_c_theory = 0.0;
    double capillary_length = 0.0;

    // Measured values
    double lambda_measured = 0.0;
    unsigned int n_spikes = 0;
    double amplitude = 0.0;

    // Validation
    double wavelength_error = 0.0;
    bool instability_observed = false;
    bool wavelength_ok = false;
    bool passed = false;

    void print_summary(std::ostream& out = std::cout) const;
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Rosensweig Theory Functions
// ============================================================================
namespace RosensweigTheory
{
    double capillary_length(double surface_tension, double density, double gravity);
    double critical_wavelength(double surface_tension, double density, double gravity);
    double critical_wavenumber(double surface_tension, double density, double gravity);
}

// ============================================================================
// Interface analysis functions
// ============================================================================

template <int dim>
InterfaceProfile extract_interface_profile(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double x_min,
    double x_max,
    unsigned int n_samples,
    MPI_Comm comm = MPI_COMM_WORLD);

unsigned int count_peaks(const InterfaceProfile& profile, double threshold = 0.01);

double estimate_wavelength(const InterfaceProfile& profile, double domain_width);

double estimate_wavelength_zero_crossing(const InterfaceProfile& profile);

template <int dim>
InterfaceProfile analyze_interface(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD);

RosensweigValidation validate_rosensweig(
    const InterfaceProfile& profile,
    const Parameters& params,
    double tolerance = 0.20);

template <int dim>
void update_validation_diagnostics(
    StepData& data,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD);

void write_interface_profile_csv(
    const InterfaceProfile& profile,
    const std::string& filename,
    MPI_Comm comm = MPI_COMM_WORLD);

// ============================================================================
// ValidationLogger - Records validation data over time
// ============================================================================
class ValidationLogger
{
public:
    void open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD);
    void log(const StepData& data);
    void close();

private:
    MPI_Comm comm_;
    bool is_root_ = false;
    std::ofstream file_;
};

#endif // VALIDATION_DIAGNOSTICS_H
