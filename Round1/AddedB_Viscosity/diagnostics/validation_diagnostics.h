// ============================================================================
// diagnostics/validation_diagnostics.h - Rosensweig Validation Diagnostics
//
// Extends interface tracking to measure wavelength and compare against
// linear stability theory: λ_c = 2π√(σ/(ρg))
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
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/geometry_info.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    double surface_tension = 0.0;       // λ (sigma/capillary coefficient)
    double density = 0.0;               // 1 + r
    double gravity = 0.0;               // g (non-dim)

    // Theoretical predictions
    double lambda_c_theory = 0.0;       // Critical wavelength = 2π√(σ/ρg)
    double capillary_length = 0.0;      // l_c = √(σ/ρg)

    // Measured values
    double lambda_measured = 0.0;       // Measured wavelength
    unsigned int n_spikes = 0;          // Number of spikes
    double amplitude = 0.0;             // Spike amplitude

    // Validation
    double wavelength_error = 0.0;      // |λ_meas - λ_theory| / λ_theory
    bool instability_observed = false;
    bool wavelength_ok = false;         // Error < tolerance
    bool passed = false;

    /**
     * @brief Print validation summary
     */
    void print_summary(std::ostream& out = std::cout) const
    {
        out << "\n"
            << "============================================================\n"
            << "ROSENSWEIG INSTABILITY VALIDATION\n"
            << "============================================================\n"
            << "\n"
            << "Physical Parameters:\n"
            << "  Surface tension (λ): " << surface_tension << "\n"
            << "  Density (1+r):       " << density << "\n"
            << "  Gravity (g):         " << gravity << "\n"
            << "\n"
            << "Theoretical Predictions:\n"
            << "  Capillary length:    " << capillary_length << "\n"
            << "  Critical wavelength: " << lambda_c_theory << "\n"
            << "\n"
            << "Measurements:\n"
            << "  Number of spikes:    " << n_spikes << "\n"
            << "  Spike amplitude:     " << amplitude << "\n"
            << "  Measured wavelength: " << lambda_measured << "\n"
            << "\n"
            << "Validation:\n"
            << "  Instability seen:    " << (instability_observed ? "YES" : "NO") << "\n"
            << "  Wavelength error:    " << wavelength_error * 100.0 << "%\n"
            << "  Wavelength OK:       " << (wavelength_ok ? "YES" : "NO") << "\n"
            << "  OVERALL:             " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n"
            << "============================================================\n";
    }

    /**
     * @brief Write validation to CSV
     */
    void write_csv(const std::string& filename) const
    {
        std::ofstream file(filename);
        file << "# Rosensweig Validation Results\n";
        file << "parameter,value\n";
        file << "surface_tension," << surface_tension << "\n";
        file << "density," << density << "\n";
        file << "gravity," << gravity << "\n";
        file << "lambda_theory," << lambda_c_theory << "\n";
        file << "lambda_measured," << lambda_measured << "\n";
        file << "wavelength_error," << wavelength_error << "\n";
        file << "n_spikes," << n_spikes << "\n";
        file << "amplitude," << amplitude << "\n";
        file << "instability_observed," << (instability_observed ? 1 : 0) << "\n";
        file << "passed," << (passed ? 1 : 0) << "\n";
        file.close();
    }
};

// ============================================================================
// Rosensweig Theory Functions
// ============================================================================
namespace RosensweigTheory
{
    /**
     * @brief Capillary length l_c = √(σ/ρg)
     */
    inline double capillary_length(double surface_tension, double density, double gravity)
    {
        if (density <= 0 || gravity <= 0) return 0.0;
        return std::sqrt(surface_tension / (density * gravity));
    }

    /**
     * @brief Critical wavelength λ_c = 2π√(σ/ρg)
     */
    inline double critical_wavelength(double surface_tension, double density, double gravity)
    {
        return 2.0 * M_PI * capillary_length(surface_tension, density, gravity);
    }

    /**
     * @brief Critical wavenumber k_c = √(ρg/σ)
     */
    inline double critical_wavenumber(double surface_tension, double density, double gravity)
    {
        if (surface_tension <= 0) return 0.0;
        return std::sqrt(density * gravity / surface_tension);
    }
}

// ============================================================================
// Extract interface profile y(x) - Parallel version
// ============================================================================
template <int dim>
InterfaceProfile extract_interface_profile(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double x_min,
    double x_max,
    unsigned int n_samples,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    InterfaceProfile profile;
    profile.x.resize(n_samples, 0.0);
    profile.y.resize(n_samples, 0.0);

    const double dx = (x_max - x_min) / (n_samples - 1);

    // Initialize x coordinates
    for (unsigned int i = 0; i < n_samples; ++i)
        profile.x[i] = x_min + i * dx;

    // Local arrays: for each x-bin, collect all crossing y values
    std::vector<double> local_y_sum(n_samples, 0.0);
    std::vector<unsigned int> local_y_count(n_samples, 0);

    // Scan all cells for interface crossings
    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        // Get θ at all vertices
        std::vector<double> vertex_theta(dealii::GeometryInfo<dim>::vertices_per_cell);
        std::vector<dealii::Point<dim>> vertex_points(dealii::GeometryInfo<dim>::vertices_per_cell);

        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            vertex_points[v] = cell->vertex(v);
            const unsigned int vertex_dof = cell->vertex_dof_index(v, 0);
            vertex_theta[v] = theta_solution[vertex_dof];
        }

        // Check each edge for sign change
        for (unsigned int edge = 0; edge < dealii::GeometryInfo<dim>::lines_per_cell; ++edge)
        {
            const unsigned int v1 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 0);
            const unsigned int v2 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 1);

            const double t1 = vertex_theta[v1];
            const double t2 = vertex_theta[v2];

            if (t1 * t2 < 0)
            {
                // Linear interpolation
                const double s = -t1 / (t2 - t1);
                const dealii::Point<dim> crossing =
                    vertex_points[v1] + s * (vertex_points[v2] - vertex_points[v1]);

                const double x_cross = crossing[0];
                const double y_cross = crossing[1];

                // Find which bin this crossing belongs to
                if (x_cross >= x_min && x_cross <= x_max)
                {
                    unsigned int bin = static_cast<unsigned int>((x_cross - x_min) / dx);
                    bin = std::min(bin, n_samples - 1);

                    local_y_sum[bin] += y_cross;
                    local_y_count[bin] += 1;
                }
            }
        }
    }

    // MPI reduction
    std::vector<double> global_y_sum(n_samples);
    std::vector<unsigned int> global_y_count(n_samples);

    MPI_Allreduce(local_y_sum.data(), global_y_sum.data(), n_samples,
                  MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_y_count.data(), global_y_count.data(), n_samples,
                  MPI_UNSIGNED, MPI_SUM, comm);

    // Compute average y for each bin
    double y_total = 0.0;
    unsigned int valid_count = 0;
    double y_min_val = std::numeric_limits<double>::max();
    double y_max_val = std::numeric_limits<double>::lowest();

    for (unsigned int i = 0; i < n_samples; ++i)
    {
        if (global_y_count[i] > 0)
        {
            profile.y[i] = global_y_sum[i] / global_y_count[i];
            y_total += profile.y[i];
            ++valid_count;
            y_min_val = std::min(y_min_val, profile.y[i]);
            y_max_val = std::max(y_max_val, profile.y[i]);
            profile.valid = true;
        }
    }

    if (valid_count > 0)
    {
        profile.y_mean = y_total / valid_count;
        profile.amplitude = (y_max_val - y_min_val) / 2.0;
    }

    // Interpolate missing values (simple linear interpolation)
    if (profile.valid)
    {
        // Forward pass
        double last_valid = profile.y_mean;
        for (unsigned int i = 0; i < n_samples; ++i)
        {
            if (global_y_count[i] > 0)
                last_valid = profile.y[i];
            else
                profile.y[i] = last_valid;
        }

        // Backward pass to smooth
        last_valid = profile.y_mean;
        for (int i = n_samples - 1; i >= 0; --i)
        {
            if (global_y_count[i] > 0)
                last_valid = profile.y[i];
            else
                profile.y[i] = (profile.y[i] + last_valid) / 2.0;
        }
    }

    return profile;
}

// ============================================================================
// Count peaks in interface profile
// ============================================================================
inline unsigned int count_peaks(const InterfaceProfile& profile, double threshold = 0.01)
{
    if (!profile.valid || profile.y.size() < 3)
        return 0;

    unsigned int n_peaks = 0;
    const double mean = profile.y_mean;

    // Find local maxima above mean + threshold
    for (size_t i = 1; i < profile.y.size() - 1; ++i)
    {
        if (profile.y[i] > profile.y[i-1] &&
            profile.y[i] > profile.y[i+1] &&
            profile.y[i] > mean + threshold)
        {
            ++n_peaks;
        }
    }

    return n_peaks;
}

// ============================================================================
// Estimate dominant wavelength from interface profile
// ============================================================================
inline double estimate_wavelength(const InterfaceProfile& profile, double domain_width)
{
    if (!profile.valid || profile.n_peaks == 0)
        return 0.0;

    // Simple estimate: wavelength ≈ domain_width / n_peaks
    return domain_width / profile.n_peaks;
}

// ============================================================================
// Estimate wavelength using zero-crossing method (more robust)
// ============================================================================
inline double estimate_wavelength_zero_crossing(const InterfaceProfile& profile)
{
    if (!profile.valid || profile.y.size() < 4)
        return 0.0;

    const double mean = profile.y_mean;

    // Find zero crossings (crossings of y = mean)
    std::vector<double> crossings;
    for (size_t i = 0; i < profile.y.size() - 1; ++i)
    {
        double y1 = profile.y[i] - mean;
        double y2 = profile.y[i+1] - mean;

        if (y1 * y2 < 0)
        {
            // Linear interpolation for crossing x
            double s = -y1 / (y2 - y1);
            double x_cross = profile.x[i] + s * (profile.x[i+1] - profile.x[i]);
            crossings.push_back(x_cross);
        }
    }

    if (crossings.size() < 2)
        return 0.0;

    // Wavelength = 2 × average distance between consecutive crossings
    double total_dist = 0.0;
    for (size_t i = 1; i < crossings.size(); ++i)
        total_dist += crossings[i] - crossings[i-1];

    double avg_half_wavelength = total_dist / (crossings.size() - 1);
    return 2.0 * avg_half_wavelength;
}

// ============================================================================
// Complete interface analysis (parallel)
// ============================================================================
template <int dim>
InterfaceProfile analyze_interface(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const double x_min = params.domain.x_min;
    const double x_max = params.domain.x_max;
    const double domain_width = x_max - x_min;
    const unsigned int n_samples = 256;  // Sufficient for wavelength detection

    InterfaceProfile profile = extract_interface_profile<dim>(
        theta_dof_handler, theta_solution, x_min, x_max, n_samples, comm);

    if (profile.valid)
    {
        // Count peaks
        double threshold = 0.005;  // Minimum amplitude to count as peak
        profile.n_peaks = count_peaks(profile, threshold);

        // Estimate wavelength using two methods and pick better one
        double lambda_peaks = estimate_wavelength(profile, domain_width);
        double lambda_crossings = estimate_wavelength_zero_crossing(profile);

        // Prefer zero-crossing method if both give reasonable results
        if (lambda_crossings > 0 && profile.n_peaks > 0)
            profile.dominant_wavelength = lambda_crossings;
        else if (lambda_peaks > 0)
            profile.dominant_wavelength = lambda_peaks;
        else
            profile.dominant_wavelength = 0.0;
    }

    return profile;
}

// ============================================================================
// Validate Rosensweig instability
// ============================================================================
inline RosensweigValidation validate_rosensweig(
    const InterfaceProfile& profile,
    const Parameters& params,
    double tolerance = 0.20)  // 20% tolerance
{
    RosensweigValidation result;

    // Extract physical parameters
    // λ in Nochetto = surface tension coefficient
    result.surface_tension = params.physics.lambda;
    result.density = 1.0 + params.physics.r;
    result.gravity = params.physics.gravity;

    // Compute theory predictions
    result.capillary_length = RosensweigTheory::capillary_length(
        result.surface_tension, result.density, result.gravity);
    result.lambda_c_theory = RosensweigTheory::critical_wavelength(
        result.surface_tension, result.density, result.gravity);

    // Fill in measurements
    result.n_spikes = profile.n_peaks;
    result.amplitude = profile.amplitude;
    result.lambda_measured = profile.dominant_wavelength;

    // Check if instability was observed
    result.instability_observed = (profile.amplitude > 0.01 && profile.n_peaks > 0);

    // Compute wavelength error
    if (result.lambda_c_theory > 0 && result.lambda_measured > 0)
    {
        result.wavelength_error = std::abs(result.lambda_measured - result.lambda_c_theory)
                                  / result.lambda_c_theory;
        result.wavelength_ok = (result.wavelength_error < tolerance);
    }

    // Overall pass/fail
    result.passed = result.instability_observed && result.wavelength_ok;

    return result;
}

// ============================================================================
// Update StepData with validation info (call in time loop)
// Requires new fields in StepData - see step_data_additions.h
// ============================================================================
template <int dim>
void update_validation_diagnostics(
    StepData& data,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    InterfaceProfile profile = analyze_interface<dim>(
        theta_dof_handler, theta_solution, params, comm);

    // Update StepData fields (these are the NEW fields to add to StepData)
    data.spike_count = profile.n_peaks;
    data.interface_amplitude = profile.amplitude;
    data.dominant_wavelength = profile.dominant_wavelength;

    // Compute theoretical wavelength for reference
    double lambda_theory = RosensweigTheory::critical_wavelength(
        params.physics.lambda,
        1.0 + params.physics.r,
        params.physics.gravity);
    data.wavelength_theory = lambda_theory;

    // Compute error if both are valid
    if (lambda_theory > 0 && profile.dominant_wavelength > 0)
    {
        data.wavelength_error = std::abs(profile.dominant_wavelength - lambda_theory)
                               / lambda_theory;
    }
}

// ============================================================================
// Write interface profile to CSV (for debugging/visualization)
// ============================================================================
inline void write_interface_profile_csv(
    const InterfaceProfile& profile,
    const std::string& filename,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    if (!MPIUtils::is_root(comm))
        return;

    std::ofstream file(filename);
    file << "# Interface profile y(x)\n";
    file << "x,y\n";
    file << std::setprecision(8);

    for (size_t i = 0; i < profile.x.size(); ++i)
        file << profile.x[i] << "," << profile.y[i] << "\n";

    file.close();
}

// ============================================================================
// ValidationLogger - Records validation data over time
// ============================================================================
class ValidationLogger
{
public:
    void open(const std::string& filename, MPI_Comm comm = MPI_COMM_WORLD)
    {
        comm_ = comm;
        is_root_ = MPIUtils::is_root(comm);

        if (is_root_)
        {
            file_.open(filename);
            file_ << "step,time,amplitude,n_spikes,wavelength_measured,wavelength_theory,wavelength_error\n";
        }
    }

    void log(const StepData& data)
    {
        if (!is_root_)
            return;

        file_ << data.step << ","
              << data.time << ","
              << data.interface_amplitude << ","
              << data.spike_count << ","
              << data.dominant_wavelength << ","
              << data.wavelength_theory << ","
              << data.wavelength_error << "\n";
    }

    void close()
    {
        if (is_root_ && file_.is_open())
            file_.close();
    }

private:
    MPI_Comm comm_;
    bool is_root_ = false;
    std::ofstream file_;
};

#endif // VALIDATION_DIAGNOSTICS_H