// ============================================================================
// diagnostics/validation_diagnostics.cc - Rosensweig Validation Diagnostics
//
// Implementation of interface tracking and wavelength validation.
// ============================================================================

#include "diagnostics/validation_diagnostics.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <deal.II/base/geometry_info.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

// ============================================================================
// RosensweigValidation methods
// ============================================================================

void RosensweigValidation::print_summary(std::ostream& out) const
{
    // Reset stream formatting — the caller may have left precision/flags
    // polluted from earlier output (the prior bug was a stray setprecision(1)
    // in an upstream printer that made lambda=0.025 round to "0.0").
    auto saved_flags = out.flags();
    auto saved_prec  = out.precision();
    out.copyfmt(std::ios(nullptr));
    out << std::defaultfloat << std::setprecision(6);

    out << "\n"
        << "============================================================\n"
        << "ROSENSWEIG INSTABILITY VALIDATION\n"
        << "============================================================\n"
        << "\n"
        << "Physical Parameters:\n"
        << "  Surface tension (lambda):  " << surface_tension << "\n"
        << "  Density contrast (Delta-rho): " << density << "\n"
        << "  Gravity (g):               " << gravity << "\n"
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
        << "  OVERALL:             " << (passed ? "PASSED" : "FAILED") << "\n"
        << "============================================================\n";

    // Restore caller's stream formatting.
    out.flags(saved_flags);
    out.precision(saved_prec);
}

void RosensweigValidation::write_csv(const std::string& filename) const
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

// ============================================================================
// RosensweigTheory namespace
// ============================================================================

double RosensweigTheory::capillary_length(double surface_tension, double density, double gravity)
{
    if (density <= 0 || gravity <= 0) return 0.0;
    return std::sqrt(surface_tension / (density * gravity));
}

double RosensweigTheory::critical_wavelength(double surface_tension, double density, double gravity)
{
    return 2.0 * M_PI * capillary_length(surface_tension, density, gravity);
}

double RosensweigTheory::critical_wavenumber(double surface_tension, double density, double gravity)
{
    if (surface_tension <= 0) return 0.0;
    return std::sqrt(density * gravity / surface_tension);
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
    MPI_Comm comm)
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

        // Get theta at all vertices
        std::vector<double> vertex_theta(dealii::GeometryInfo<dim>::vertices_per_cell);
        std::vector<dealii::Point<dim>> vertex_points(dealii::GeometryInfo<dim>::vertices_per_cell);

        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            vertex_points[v] = cell->vertex(v);
            const unsigned int vertex_dof = cell->vertex_dof_index(v, 0);
            // theta_solution must be ghosted (locally_relevant). Vertices on
            // partition boundaries are owned by neighbouring ranks; reading
            // them from a non-ghosted vector throws inside Trilinos.
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
unsigned int count_peaks(const InterfaceProfile& profile, double threshold)
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
double estimate_wavelength(const InterfaceProfile& profile, double domain_width)
{
    if (!profile.valid || profile.n_peaks == 0)
        return 0.0;

    // Simple estimate: wavelength ~ domain_width / n_peaks
    return domain_width / profile.n_peaks;
}

// ============================================================================
// Estimate wavelength using zero-crossing method (more robust)
// ============================================================================
double estimate_wavelength_zero_crossing(const InterfaceProfile& profile)
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

    // Wavelength = 2 * average distance between consecutive crossings
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
    MPI_Comm comm)
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
RosensweigValidation validate_rosensweig(
    const InterfaceProfile& profile,
    const Parameters& params,
    double tolerance)
{
    RosensweigValidation result;

    // Extract physical parameters
    result.surface_tension = params.physics.lambda;
    // Cowley-Rosensweig theory needs the density CONTRAST Δρ across the
    // interface, not the ferrofluid's absolute density ρ_ferro = 1+r.
    // Project nondim convention: ρ_water = 1, ρ_ferro = 1+r → Δρ = r.
    // Previously this used (1+r), which biased λ_c by ~factor √(1+1/r)
    // for r≈0.1 — substantial. Same fix applied to analyze_hedgehog.py.
    result.density = params.physics.r;
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
// ============================================================================
template <int dim>
void update_validation_diagnostics(
    StepData& data,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm)
{
    InterfaceProfile profile = analyze_interface<dim>(
        theta_dof_handler, theta_solution, params, comm);

    // Update StepData fields
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
void write_interface_profile_csv(
    const InterfaceProfile& profile,
    const std::string& filename,
    MPI_Comm comm)
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
// ValidationLogger
// ============================================================================

void ValidationLogger::open(const std::string& filename, MPI_Comm comm)
{
    comm_ = comm;
    is_root_ = MPIUtils::is_root(comm);

    if (is_root_)
    {
        file_.open(filename);
        file_ << "step,time,amplitude,n_spikes,wavelength_measured,wavelength_theory,wavelength_error\n";
    }
}

void ValidationLogger::log(const StepData& data)
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

void ValidationLogger::close()
{
    if (is_root_ && file_.is_open())
        file_.close();
}

// ============================================================================
// Explicit template instantiations for dim=2
// ============================================================================
template InterfaceProfile extract_interface_profile<2>(
    const dealii::DoFHandler<2>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double x_min,
    double x_max,
    unsigned int n_samples,
    MPI_Comm comm);

template InterfaceProfile analyze_interface<2>(
    const dealii::DoFHandler<2>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm);

template void update_validation_diagnostics<2>(
    StepData& data,
    const dealii::DoFHandler<2>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm);
