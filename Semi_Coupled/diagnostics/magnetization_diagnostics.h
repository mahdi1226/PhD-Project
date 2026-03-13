// ============================================================================
// diagnostics/magnetization_diagnostics.h - Applied Field & Magnetization Diagnostics
//
// Provides diagnostics for the magnetic field distribution h_a.
// Key for diagnosing non-uniform field causing dome shapes instead of spikes.
//
// Note: This samples h_a at discrete points (no mesh integration needed),
// so MPI reductions are only needed for global min/max across ranks.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_DIAGNOSTICS_H
#define MAGNETIZATION_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"
#include "physics/applied_field.h"

#include <deal.II/base/point.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

// ============================================================================
// Field Distribution Data
// ============================================================================
struct FieldDistributionData
{
    double time = 0.0;
    unsigned int step = 0;

    // Field at key x-positions (at interface y-level)
    double h_a_x_left = 0.0;      // |h_a| at x = x_min + 0.1*L
    double h_a_x_center = 0.0;    // |h_a| at x = 0.5*(x_min + x_max)
    double h_a_x_right = 0.0;     // |h_a| at x = x_max - 0.1*L

    // Field components at center
    double h_a_center_x = 0.0;    // h_a_x component at center
    double h_a_center_y = 0.0;    // h_a_y component at center

    // Uniformity metrics
    double h_a_min = 0.0;         // min|h_a| across interface
    double h_a_max = 0.0;         // max|h_a| across interface
    double center_to_edge_ratio = 0.0;  // |h_a|_center / |h_a|_edge
    double uniformity = 0.0;      // min/max ratio (1.0 = perfectly uniform)

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

// ============================================================================
// Compute field distribution at interface level
// ============================================================================
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

// ============================================================================
// Logger for field distribution data (MPI-aware)
// ============================================================================
class MagnetizationLogger
{
public:
    MagnetizationLogger() = default;

    /**
     * @brief Open the log file (rank 0 only)
     */
    void open(const std::string& filename,
              const Parameters& params,
              MPI_Comm comm = MPI_COMM_WORLD)
    {
        comm_ = comm;
        is_root_ = MPIUtils::is_root(comm);

        if (!is_root_) return;

        file_.open(filename);
        if (file_.is_open())
        {
            file_ << FieldDistributionData::header() << "\n";
            file_ << get_csv_header_stamp(params) << "\n";
        }
    }

    /**
     * @brief Write field distribution data
     */
    void write(const FieldDistributionData& data)
    {
        if (!is_root_ || !file_.is_open()) return;

        file_ << data.to_csv() << "\n";
        file_.flush();
    }

    /**
     * @brief Close the file
     */
    void close()
    {
        if (is_root_ && file_.is_open())
            file_.close();
    }

    bool is_open() const { return is_root_ && file_.is_open(); }

private:
    std::ofstream file_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    bool is_root_ = false;
};

#endif // MAGNETIZATION_DIAGNOSTICS_H