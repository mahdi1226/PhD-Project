// ============================================================================
// diagnostics/field_diagnostics.h - Applied Field Diagnostic (ENHANCED)
//
// Provides both console output and CSV logging for h_a field distribution.
// Key for diagnosing non-uniform field causing dome shapes instead of spikes.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef FIELD_DIAGNOSTIC_H
#define FIELD_DIAGNOSTIC_H

#include "assembly/poisson_assembler.h"
#include "utilities/parameters.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

/**
 * @brief Container for field distribution diagnostics at interface level
 */
struct FieldDistributionData
{
    double time = 0.0;
    unsigned int step = 0;

    // Field at key x-positions (at interface y-level)
    double h_a_x_left = 0.0;      // |h_a| at x = x_min + 0.1*L
    double h_a_x_center = 0.0;   // |h_a| at x = 0.5*(x_min + x_max)
    double h_a_x_right = 0.0;    // |h_a| at x = x_max - 0.1*L

    // Field components at center
    double h_a_center_x = 0.0;   // h_a_x component at center
    double h_a_center_y = 0.0;   // h_a_y component at center

    // Uniformity metrics
    double h_a_min = 0.0;        // min|h_a| across interface
    double h_a_max = 0.0;        // max|h_a| across interface
    double center_to_edge_ratio = 0.0;  // |h_a|_center / |h_a|_edge
    double uniformity = 0.0;     // min/max ratio (1.0 = perfectly uniform)

    // Ramp factor
    double ramp_factor = 0.0;

    /**
     * @brief CSV header string
     */
    static std::string header()
    {
        return "step,time,ramp_factor,h_a_left,h_a_center,h_a_right,"
               "h_a_center_x,h_a_center_y,h_a_min,h_a_max,"
               "center_edge_ratio,uniformity";
    }

    /**
     * @brief Convert to CSV row
     */
    std::string to_csv() const
    {
        std::ostringstream oss;
        oss << step << ","
            << std::scientific << std::setprecision(6)
            << time << ","
            << ramp_factor << ","
            << h_a_x_left << ","
            << h_a_x_center << ","
            << h_a_x_right << ","
            << h_a_center_x << ","
            << h_a_center_y << ","
            << h_a_min << ","
            << h_a_max << ","
            << center_to_edge_ratio << ","
            << uniformity;
        return oss.str();
    }
};

/**
 * @brief Compute field distribution at interface level
 *
 * Samples h_a at multiple x-positions along y = pool_depth.
 * Returns metrics useful for diagnosing field uniformity issues.
 */
inline FieldDistributionData compute_field_distribution(
    const Parameters& params,
    double time,
    unsigned int step)
{
    FieldDistributionData data;
    data.time = time;
    data.step = step;

    // Ramp factor
    data.ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(time / params.dipoles.ramp_time, 1.0) : 1.0;

    const double y_interface = params.ic.pool_depth;
    const double x_min = params.domain.x_min;
    const double x_max = params.domain.x_max;
    const double Lx = x_max - x_min;

    // Sample at 11 points across the interface
    const int n_samples = 11;
    std::vector<double> h_a_magnitudes(n_samples);

    double min_mag = 1e20;
    double max_mag = 0.0;

    for (int i = 0; i < n_samples; ++i)
    {
        double x = x_min + i * Lx / (n_samples - 1);
        dealii::Point<2> p(x, y_interface);
        auto h_a = compute_applied_field<2>(p, params, time);
        double mag = h_a.norm();

        h_a_magnitudes[i] = mag;
        min_mag = std::min(min_mag, mag);
        max_mag = std::max(max_mag, mag);

        // Store specific positions
        if (i == 1)  // x ~ 0.1*L from left
            data.h_a_x_left = mag;
        else if (i == n_samples / 2)  // center
        {
            data.h_a_x_center = mag;
            data.h_a_center_x = h_a[0];
            data.h_a_center_y = h_a[1];
        }
        else if (i == n_samples - 2)  // x ~ 0.1*L from right
            data.h_a_x_right = mag;
    }

    data.h_a_min = min_mag;
    data.h_a_max = max_mag;

    // Compute ratios
    double edge_avg = 0.5 * (data.h_a_x_left + data.h_a_x_right);
    data.center_to_edge_ratio = (edge_avg > 1e-12)
        ? data.h_a_x_center / edge_avg : 0.0;

    data.uniformity = (max_mag > 1e-12) ? min_mag / max_mag : 0.0;

    return data;
}

/**
 * @brief Logger for field distribution data
 */
class FieldDistributionLogger
{
public:
    FieldDistributionLogger() = default;

    void open(const std::string& filename)
    {
        file_.open(filename);
        if (file_.is_open())
        {
            file_ << FieldDistributionData::header() << "\n";
        }
    }

    void write(const FieldDistributionData& data)
    {
        if (file_.is_open())
        {
            file_ << data.to_csv() << "\n";
            file_.flush();
        }
    }

    void close()
    {
        if (file_.is_open())
            file_.close();
    }

    bool is_open() const { return file_.is_open(); }

private:
    std::ofstream file_;
};

/**
 * @brief Print h_a field at a grid of points (console output)
 *
 * Call this at the beginning of simulation to verify field distribution.
 */
inline void diagnose_applied_field(const Parameters& params, double time)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "APPLIED FIELD DIAGNOSTIC at t = " << time << "\n";
    std::cout << "============================================================\n";
    std::cout << "Dipole configuration:\n";
    std::cout << "  Number of dipoles: " << params.dipoles.positions.size() << "\n";
    std::cout << "  Direction: (" << params.dipoles.direction[0]
              << ", " << params.dipoles.direction[1] << ")\n";
    std::cout << "  Intensity max: " << params.dipoles.intensity_max << "\n";
    std::cout << "  Ramp time: " << params.dipoles.ramp_time << "\n";

    // Ramp factor
    const double ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(time / params.dipoles.ramp_time, 1.0) : 1.0;
    std::cout << "  Current ramp factor: " << ramp_factor << "\n";

    // Print dipole positions
    std::cout << "\nDipole positions:\n";
    double x_min_dip = 1e10, x_max_dip = -1e10;
    double y_min_dip = 1e10, y_max_dip = -1e10;
    for (size_t i = 0; i < params.dipoles.positions.size(); ++i)
    {
        const auto& pos = params.dipoles.positions[i];
        x_min_dip = std::min(x_min_dip, pos[0]);
        x_max_dip = std::max(x_max_dip, pos[0]);
        y_min_dip = std::min(y_min_dip, pos[1]);
        y_max_dip = std::max(y_max_dip, pos[1]);
        if (i < 5 || i >= params.dipoles.positions.size() - 2)
            std::cout << "  [" << i << "] (" << pos[0] << ", " << pos[1] << ")\n";
        else if (i == 5)
            std::cout << "  ... (" << params.dipoles.positions.size() - 7 << " more) ...\n";
    }
    std::cout << "  X range: [" << x_min_dip << ", " << x_max_dip << "]\n";
    std::cout << "  Y range: [" << y_min_dip << ", " << y_max_dip << "]\n";

    // Sample field at key points
    std::cout << "\nField h_a at interface level (y = " << params.ic.pool_depth << "):\n";
    std::cout << std::setw(10) << "x"
              << std::setw(15) << "h_a_x"
              << std::setw(15) << "h_a_y"
              << std::setw(15) << "|h_a|" << "\n";
    std::cout << "----------------------------------------------------\n";

    const double y_test = params.ic.pool_depth;
    double max_magnitude = 0.0;
    double min_magnitude = 1e10;

    for (double x = 0.0; x <= 1.0; x += 0.1)
    {
        dealii::Point<2> p(x, y_test);
        auto h_a = compute_applied_field<2>(p, params, time);
        double mag = h_a.norm();

        max_magnitude = std::max(max_magnitude, mag);
        min_magnitude = std::min(min_magnitude, mag);

        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                  << std::setw(15) << std::scientific << std::setprecision(4) << h_a[0]
                  << std::setw(15) << h_a[1]
                  << std::setw(15) << mag << "\n";
    }

    std::cout << "\nField uniformity:\n";
    std::cout << "  |h_a| range: [" << min_magnitude << ", " << max_magnitude << "]\n";
    std::cout << "  Ratio max/min: " << (min_magnitude > 0 ? max_magnitude/min_magnitude : 0) << "\n";

    // Check center vs edge ratio
    dealii::Point<2> p_center(0.5, y_test);
    dealii::Point<2> p_edge(0.1, y_test);
    auto h_center = compute_applied_field<2>(p_center, params, time);
    auto h_edge = compute_applied_field<2>(p_edge, params, time);

    std::cout << "  |h_a| at center (x=0.5): " << h_center.norm() << "\n";
    std::cout << "  |h_a| at edge (x=0.1):   " << h_edge.norm() << "\n";
    std::cout << "  Center/Edge ratio: " << h_center.norm() / (h_edge.norm() + 1e-12) << "\n";

    if (h_center.norm() > 3.0 * h_edge.norm())
    {
        std::cout << "\n*** WARNING: Field is highly concentrated at center! ***\n";
        std::cout << "    This will cause central spike instead of uniform deformation.\n";
        std::cout << "    Consider widening dipole x-range or using wider magnet.\n";
    }

    // Check for dome-causing field pattern
    if (h_center.norm() > 1.5 * h_edge.norm() && h_center.norm() < 3.0 * h_edge.norm())
    {
        std::cout << "\n*** NOTE: Field has moderate center/edge variation ***\n";
        std::cout << "    Center/Edge ratio of " << h_center.norm() / (h_edge.norm() + 1e-12)
                  << " may cause dome shape.\n";
        std::cout << "    For uniform deformation, aim for ratio close to 1.0.\n";
    }

    std::cout << "============================================================\n\n";
}

/**
 * @brief Write h_a field to CSV for visualization
 */
inline void write_field_csv(const Parameters& params, double time,
                            const std::string& filename)
{
    std::ofstream file(filename);
    file << "x,y,h_a_x,h_a_y,magnitude\n";

    const int nx = 50;
    const int ny = 30;

    for (int j = 0; j <= ny; ++j)
    {
        double y = params.domain.y_min +
                   j * (params.domain.y_max - params.domain.y_min) / ny;
        for (int i = 0; i <= nx; ++i)
        {
            double x = params.domain.x_min +
                       i * (params.domain.x_max - params.domain.x_min) / nx;

            dealii::Point<2> p(x, y);
            auto h_a = compute_applied_field<2>(p, params, time);

            file << x << "," << y << ","
                 << h_a[0] << "," << h_a[1] << ","
                 << h_a.norm() << "\n";
        }
    }

    file.close();
    std::cout << "Wrote field data to: " << filename << "\n";
}

/**
 * @brief Write full spatial field profile at a specific time
 *
 * Creates a 2D grid CSV for heatmap visualization.
 */
inline void write_field_profile(const Parameters& params, double time,
                                unsigned int step, const std::string& output_dir)
{
    std::string filename = output_dir + "/h_a_profile_" + std::to_string(step) + ".csv";
    write_field_csv(params, time, filename);
}

#endif // FIELD_DIAGNOSTIC_H