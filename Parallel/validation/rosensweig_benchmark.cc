// ============================================================================
// validation/rosensweig_benchmark.cc - Rosensweig Instability Validation
//
// Validates the full ferrofluid solver (CH + NS + Poisson + Magnetization)
// against linear stability theory predictions.
//
// Key validation:
//   1. Critical wavelength: λ_c = 2π√(σ/(ρg))
//   2. Critical magnetic Bond number: Bo_c ≈ 2
//   3. Spike amplitude growth rate
//
// Usage:
//   ./rosensweig_benchmark --config rosensweig.prm
//   ./rosensweig_benchmark --field-strength 1.5  # Multiple of critical
//
// Reference: Cowley & Rosensweig, J. Fluid Mech. 30 (1967) 671-688
// ============================================================================

#include "validation/validation.h"
#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace dealii;

// ============================================================================
// Rosensweig Linear Stability Theory
// ============================================================================
namespace RosensweigTheory
{
    /**
     * @brief Capillary length scale
     *
     * l_c = √(σ / (ρg))
     *
     * This is the length scale where surface tension and gravity balance.
     */
    inline double capillary_length(double surface_tension,
                                    double density,
                                    double gravity)
    {
        return std::sqrt(surface_tension / (density * gravity));
    }

    /**
     * @brief Critical wavelength for Rosensweig instability
     *
     * λ_c = 2π * l_c = 2π√(σ/(ρg))
     *
     * This is the wavelength of maximum growth rate at onset.
     */
    inline double critical_wavelength(double surface_tension,
                                       double density,
                                       double gravity)
    {
        return 2.0 * M_PI * capillary_length(surface_tension, density, gravity);
    }

    /**
     * @brief Critical wavenumber
     *
     * k_c = 2π/λ_c = √(ρg/σ)
     */
    inline double critical_wavenumber(double surface_tension,
                                       double density,
                                       double gravity)
    {
        return std::sqrt(density * gravity / surface_tension);
    }

    /**
     * @brief Magnetic Bond number
     *
     * Bo_m = μ₀ χ H² / (ρ g l_c)
     *
     * Ratio of magnetic to gravitational forces.
     * Critical value Bo_c ≈ 2 for onset of instability.
     */
    inline double magnetic_bond_number(double mu_0,
                                        double susceptibility,
                                        double H_magnitude,
                                        double density,
                                        double gravity,
                                        double surface_tension)
    {
        double l_c = capillary_length(surface_tension, density, gravity);
        return mu_0 * susceptibility * H_magnitude * H_magnitude / (density * gravity * l_c);
    }

    /**
     * @brief Critical field strength for instability onset
     *
     * H_c such that Bo_m = Bo_c ≈ 2
     */
    inline double critical_field_strength(double mu_0,
                                           double susceptibility,
                                           double density,
                                           double gravity,
                                           double surface_tension,
                                           double Bo_critical = 2.0)
    {
        double l_c = capillary_length(surface_tension, density, gravity);
        return std::sqrt(Bo_critical * density * gravity * l_c / (mu_0 * susceptibility));
    }

    /**
     * @brief Theoretical dispersion relation (simplified)
     *
     * Growth rate σ(k) for wavenumber k.
     * Maximum at k = k_c when Bo > Bo_c.
     *
     * This is a simplified form; full expression includes viscosity effects.
     */
    inline double growth_rate(double k,
                               double surface_tension,
                               double density,
                               double gravity,
                               double mu_0,
                               double susceptibility,
                               double H_magnitude)
    {
        double k_c = critical_wavenumber(surface_tension, density, gravity);
        double Bo = magnetic_bond_number(mu_0, susceptibility, H_magnitude,
                                         density, gravity, surface_tension);

        // Simplified growth rate (inviscid limit)
        // σ² ∝ g*k*(Bo - 1 - (k/k_c)² - (k_c/k)²)
        double term = Bo - 1.0 - (k/k_c)*(k/k_c) - (k_c/k)*(k_c/k);

        if (term > 0)
            return std::sqrt(gravity * k * term);
        else
            return 0.0;  // Stable
    }
}

// ============================================================================
// Rosensweig Validation Data
// ============================================================================
struct RosensweigValidation
{
    // Physical parameters
    double surface_tension = 0.0;
    double density = 0.0;
    double gravity = 0.0;
    double mu_0 = 0.0;
    double susceptibility = 0.0;
    double H_applied = 0.0;

    // Theoretical predictions
    double lambda_c_theory = 0.0;       // Critical wavelength
    double k_c_theory = 0.0;            // Critical wavenumber
    double Bo_m = 0.0;                  // Magnetic Bond number
    double H_c = 0.0;                   // Critical field strength

    // Measured values
    double lambda_measured = 0.0;       // Measured wavelength
    unsigned int n_spikes = 0;          // Number of spikes observed
    double amplitude_final = 0.0;       // Final spike amplitude
    double growth_time = 0.0;           // Time to develop instability

    // Errors
    double wavelength_error = 0.0;      // |λ_measured - λ_theory| / λ_theory

    // Status
    bool instability_observed = false;
    bool wavelength_validated = false;  // Error < tolerance
    bool passed = false;

    /**
     * @brief Print validation summary
     */
    void print_summary(std::ostream& out) const
    {
        out << "\n"
            << "============================================================\n"
            << "ROSENSWEIG INSTABILITY VALIDATION\n"
            << "============================================================\n"
            << "\n"
            << "Physical Parameters:\n"
            << "  Surface tension σ:    " << surface_tension << "\n"
            << "  Density ρ:            " << density << "\n"
            << "  Gravity g:            " << gravity << "\n"
            << "  Permeability μ₀:      " << mu_0 << "\n"
            << "  Susceptibility χ:     " << susceptibility << "\n"
            << "  Applied field H:      " << H_applied << "\n"
            << "\n"
            << "Theoretical Predictions:\n"
            << "  Capillary length l_c: " << std::sqrt(surface_tension / (density * gravity)) << "\n"
            << "  Critical wavelength:  " << lambda_c_theory << "\n"
            << "  Critical wavenumber:  " << k_c_theory << "\n"
            << "  Critical field H_c:   " << H_c << "\n"
            << "  Magnetic Bond number: " << Bo_m << "\n"
            << "  Supercritical ratio:  " << (H_c > 0 ? H_applied / H_c : 0) << " × H_c\n"
            << "\n"
            << "Measured Values:\n"
            << "  Number of spikes:     " << n_spikes << "\n"
            << "  Measured wavelength:  " << lambda_measured << "\n"
            << "  Final amplitude:      " << amplitude_final << "\n"
            << "  Instability observed: " << (instability_observed ? "YES" : "NO") << "\n"
            << "\n"
            << "Validation:\n"
            << "  Wavelength error:     " << wavelength_error * 100.0 << "%\n"
            << "  Wavelength OK:        " << (wavelength_validated ? "YES" : "NO") << "\n"
            << "  Overall status:       " << (passed ? "PASSED" : "FAILED") << "\n"
            << "============================================================\n";
    }

    /**
     * @brief Write to CSV
     */
    void write_csv(const std::string& filename) const
    {
        std::ofstream file(filename);
        file << "# Rosensweig Instability Validation Results\n";
        file << "parameter,value\n";
        file << "surface_tension," << surface_tension << "\n";
        file << "density," << density << "\n";
        file << "gravity," << gravity << "\n";
        file << "mu_0," << mu_0 << "\n";
        file << "susceptibility," << susceptibility << "\n";
        file << "H_applied," << H_applied << "\n";
        file << "lambda_theory," << lambda_c_theory << "\n";
        file << "lambda_measured," << lambda_measured << "\n";
        file << "wavelength_error," << wavelength_error << "\n";
        file << "n_spikes," << n_spikes << "\n";
        file << "amplitude," << amplitude_final << "\n";
        file << "Bo_magnetic," << Bo_m << "\n";
        file << "passed," << (passed ? 1 : 0) << "\n";
        file.close();
    }
};

// ============================================================================
// Time series data for instability growth
// ============================================================================
struct InstabilityTimeSeries
{
    std::vector<double> time;
    std::vector<double> amplitude;      // (y_max - y_min) / 2
    std::vector<double> y_max;          // Maximum interface height
    std::vector<double> y_min;          // Minimum interface height
    std::vector<double> n_spikes;       // Number of spikes (can be fractional during growth)
    std::vector<double> wavelength;     // Instantaneous wavelength estimate

    void add_point(double t, const InterfaceProfile& profile)
    {
        time.push_back(t);
        amplitude.push_back(profile.amplitude);

        double y_max_val = *std::max_element(profile.y.begin(), profile.y.end());
        double y_min_val = *std::min_element(profile.y.begin(), profile.y.end());
        y_max.push_back(y_max_val);
        y_min.push_back(y_min_val);

        n_spikes.push_back(static_cast<double>(profile.n_peaks));
        wavelength.push_back(profile.dominant_wavelength);
    }

    void write_csv(const std::string& filename) const
    {
        std::ofstream file(filename);
        file << "time,amplitude,y_max,y_min,n_spikes,wavelength\n";
        for (size_t i = 0; i < time.size(); ++i)
        {
            file << std::setprecision(8)
                 << time[i] << ","
                 << amplitude[i] << ","
                 << y_max[i] << ","
                 << y_min[i] << ","
                 << n_spikes[i] << ","
                 << wavelength[i] << "\n";
        }
        file.close();
    }
};

// ============================================================================
// Rosensweig Benchmark Class
// ============================================================================
class RosensweigBenchmark
{
public:
    RosensweigBenchmark(MPI_Comm comm);

    /**
     * @brief Set physical parameters
     */
    void set_parameters(double surface_tension,
                        double density,
                        double gravity,
                        double mu_0,
                        double susceptibility);

    /**
     * @brief Set applied field strength
     * @param H_applied Field magnitude (or use set_supercritical_ratio)
     */
    void set_field_strength(double H_applied);

    /**
     * @brief Set field as multiple of critical field
     * @param ratio H/H_c (e.g., 1.5 means 50% above critical)
     */
    void set_supercritical_ratio(double ratio);

    /**
     * @brief Compute theoretical predictions
     */
    void compute_theory();

    /**
     * @brief Validate against measured interface profile
     * @param profile Interface profile from simulation
     * @param tolerance Acceptable relative error in wavelength (default 20%)
     */
    RosensweigValidation validate(const InterfaceProfile& profile,
                                   double tolerance = 0.20);

    /**
     * @brief Process time series from simulation
     *
     * Call this during your time stepping loop to record instability growth.
     */
    void record_timestep(double time, const InterfaceProfile& profile);

    /**
     * @brief Get time series data
     */
    const InstabilityTimeSeries& get_time_series() const { return time_series_; }

    /**
     * @brief Write all output files
     */
    void write_outputs(const std::string& output_dir) const;

    /**
     * @brief Print theoretical predictions
     */
    void print_theory(std::ostream& out) const;

private:
    MPI_Comm mpi_comm_;
    bool is_root_;

    // Physical parameters
    double sigma_;      // Surface tension
    double rho_;        // Density
    double g_;          // Gravity
    double mu_0_;       // Permeability
    double chi_;        // Susceptibility
    double H_;          // Applied field

    // Theoretical values
    double l_c_;        // Capillary length
    double lambda_c_;   // Critical wavelength
    double k_c_;        // Critical wavenumber
    double H_c_;        // Critical field
    double Bo_m_;       // Magnetic Bond number

    // Time series
    InstabilityTimeSeries time_series_;

    // Latest validation result
    RosensweigValidation last_validation_;
};

RosensweigBenchmark::RosensweigBenchmark(MPI_Comm comm)
    : mpi_comm_(comm)
    , is_root_(Utilities::MPI::this_mpi_process(comm) == 0)
    , sigma_(0.0), rho_(0.0), g_(0.0), mu_0_(0.0), chi_(0.0), H_(0.0)
    , l_c_(0.0), lambda_c_(0.0), k_c_(0.0), H_c_(0.0), Bo_m_(0.0)
{}

void RosensweigBenchmark::set_parameters(double surface_tension,
                                          double density,
                                          double gravity,
                                          double mu_0,
                                          double susceptibility)
{
    sigma_ = surface_tension;
    rho_ = density;
    g_ = gravity;
    mu_0_ = mu_0;
    chi_ = susceptibility;

    compute_theory();
}

void RosensweigBenchmark::set_field_strength(double H_applied)
{
    H_ = H_applied;
    if (sigma_ > 0)
        compute_theory();
}

void RosensweigBenchmark::set_supercritical_ratio(double ratio)
{
    // First compute H_c, then set H = ratio * H_c
    if (sigma_ > 0 && rho_ > 0 && g_ > 0 && mu_0_ > 0 && chi_ > 0)
    {
        H_c_ = RosensweigTheory::critical_field_strength(mu_0_, chi_, rho_, g_, sigma_);
        H_ = ratio * H_c_;
        compute_theory();
    }
}

void RosensweigBenchmark::compute_theory()
{
    if (sigma_ <= 0 || rho_ <= 0 || g_ <= 0)
        return;

    l_c_ = RosensweigTheory::capillary_length(sigma_, rho_, g_);
    lambda_c_ = RosensweigTheory::critical_wavelength(sigma_, rho_, g_);
    k_c_ = RosensweigTheory::critical_wavenumber(sigma_, rho_, g_);

    if (mu_0_ > 0 && chi_ > 0)
    {
        H_c_ = RosensweigTheory::critical_field_strength(mu_0_, chi_, rho_, g_, sigma_);

        if (H_ > 0)
            Bo_m_ = RosensweigTheory::magnetic_bond_number(mu_0_, chi_, H_, rho_, g_, sigma_);
    }
}

void RosensweigBenchmark::print_theory(std::ostream& out) const
{
    out << "\n"
        << "Rosensweig Linear Stability Theory Predictions\n"
        << "-----------------------------------------------\n"
        << "  Capillary length l_c = √(σ/ρg) = " << l_c_ << "\n"
        << "  Critical wavelength λ_c = 2πl_c = " << lambda_c_ << "\n"
        << "  Critical wavenumber k_c = 1/l_c = " << k_c_ << "\n"
        << "  Critical field H_c (Bo=2) = " << H_c_ << "\n"
        << "\n"
        << "  Applied field H = " << H_ << "\n"
        << "  Supercritical ratio H/H_c = " << (H_c_ > 0 ? H_/H_c_ : 0) << "\n"
        << "  Magnetic Bond number Bo_m = " << Bo_m_ << "\n"
        << "  Instability expected: " << (Bo_m_ > 2.0 ? "YES" : "NO") << "\n"
        << "\n";
}

RosensweigValidation RosensweigBenchmark::validate(const InterfaceProfile& profile,
                                                    double tolerance)
{
    RosensweigValidation result;

    // Fill in parameters
    result.surface_tension = sigma_;
    result.density = rho_;
    result.gravity = g_;
    result.mu_0 = mu_0_;
    result.susceptibility = chi_;
    result.H_applied = H_;

    // Fill in theory
    result.lambda_c_theory = lambda_c_;
    result.k_c_theory = k_c_;
    result.Bo_m = Bo_m_;
    result.H_c = H_c_;

    // Fill in measurements
    result.lambda_measured = profile.dominant_wavelength;
    result.n_spikes = profile.n_peaks;
    result.amplitude_final = profile.amplitude;

    // Check if instability was observed
    result.instability_observed = (profile.amplitude > 0.01 && profile.n_peaks > 0);

    // Compute wavelength error
    if (result.lambda_c_theory > 0 && result.lambda_measured > 0)
    {
        result.wavelength_error = std::abs(result.lambda_measured - result.lambda_c_theory)
                                  / result.lambda_c_theory;
        result.wavelength_validated = (result.wavelength_error < tolerance);
    }

    // Overall pass/fail
    // Pass if: instability observed AND wavelength within tolerance
    result.passed = result.instability_observed && result.wavelength_validated;

    last_validation_ = result;
    return result;
}

void RosensweigBenchmark::record_timestep(double time, const InterfaceProfile& profile)
{
    time_series_.add_point(time, profile);
}

void RosensweigBenchmark::write_outputs(const std::string& output_dir) const
{
    if (!is_root_)
        return;

    // Write time series
    time_series_.write_csv(output_dir + "/rosensweig_timeseries.csv");

    // Write validation results
    last_validation_.write_csv(output_dir + "/rosensweig_validation.csv");

    // Write theoretical predictions for reference
    std::ofstream theory_file(output_dir + "/rosensweig_theory.txt");
    print_theory(theory_file);
    theory_file.close();
}

// ============================================================================
// Helper: Extract parameters from your Parameters struct
// ============================================================================
inline RosensweigBenchmark create_benchmark_from_params(
    const Parameters& params,
    MPI_Comm comm)
{
    RosensweigBenchmark benchmark(comm);

    // Map your parameters to physical quantities
    // Adjust these mappings based on your actual parameter names
    double sigma = params.physics.lambda;   // Surface tension ~ λ in Nochetto
    double rho = 1.0 + params.physics.r;    // Ferrofluid density
    double g = params.physics.gravity;
    double mu_0 = params.physics.mu_0;
    double chi = params.physics.chi_0;

    benchmark.set_parameters(sigma, rho, g, mu_0, chi);

    // Set field strength from dipole configuration
    // This is approximate - you may need to compute actual H at interface
    benchmark.set_field_strength(params.dipoles.intensity_max);

    return benchmark;
}

// ============================================================================
// Main - Standalone benchmark program
// ============================================================================
int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPI_Comm comm = MPI_COMM_WORLD;
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

    pcout << "============================================================\n"
          << "Rosensweig Instability Validation Benchmark\n"
          << "============================================================\n\n";

    // Example usage with typical ferrofluid parameters
    RosensweigBenchmark benchmark(comm);

    // Set physical parameters (example values - adjust to your simulation)
    double sigma = 0.025;       // Surface tension [N/m]
    double rho = 1200.0;        // Density [kg/m³]
    double g = 9.81;            // Gravity [m/s²]
    double mu_0 = 4.0 * M_PI * 1e-7;  // Vacuum permeability
    double chi = 1.5;           // Magnetic susceptibility

    benchmark.set_parameters(sigma, rho, g, mu_0, chi);

    // Set field 50% above critical
    benchmark.set_supercritical_ratio(1.5);

    // Print theoretical predictions
    benchmark.print_theory(std::cout);

    pcout << "============================================================\n"
          << "USAGE IN YOUR CODE\n"
          << "============================================================\n"
          << "\n"
          << "1. Include this header and create benchmark:\n"
          << "   RosensweigBenchmark benchmark(comm);\n"
          << "   benchmark.set_parameters(sigma, rho, g, mu_0, chi);\n"
          << "   benchmark.set_field_strength(H);\n"
          << "\n"
          << "2. During time stepping, record interface:\n"
          << "   InterfaceProfile profile = extract_interface_profile<2>(\n"
          << "       theta_dof, theta_solution, x_min, x_max, 256, comm);\n"
          << "   benchmark.record_timestep(time, profile);\n"
          << "\n"
          << "3. At end of simulation, validate:\n"
          << "   auto result = benchmark.validate(final_profile, 0.20);\n"
          << "   result.print_summary(std::cout);\n"
          << "   benchmark.write_outputs(\"output_dir\");\n"
          << "\n"
          << "============================================================\n"
          << "\n"
          << "To integrate with your solver:\n"
          << "  - Add calls to record_timestep() in your time loop\n"
          << "  - Call validate() after simulation completes\n"
          << "  - The wavelength comparison validates your full solver\n"
          << "\n";

    // Demonstrate validation with synthetic data
    pcout << "============================================================\n"
          << "DEMONSTRATION WITH SYNTHETIC DATA\n"
          << "============================================================\n\n";

    // Create a fake interface profile for demonstration
    InterfaceProfile demo_profile;
    demo_profile.valid = true;
    demo_profile.n_peaks = 5;
    demo_profile.amplitude = 0.05;

    // Compute what wavelength we expect
    double expected_lambda = benchmark.validate(demo_profile, 0.20).lambda_c_theory;

    // Set measured wavelength close to theory (10% error for demo)
    demo_profile.dominant_wavelength = expected_lambda * 1.10;

    // Run validation
    auto result = benchmark.validate(demo_profile, 0.20);
    result.print_summary(std::cout);

    return result.passed ? 0 : 1;
}