// ============================================================================
// diagnostics/field_diagnostic.h - Applied Field Diagnostic
//
// Prints h_a at grid of points to verify field distribution
// ============================================================================
#ifndef FIELD_DIAGNOSTIC_H
#define FIELD_DIAGNOSTIC_H

#include "assembly/poisson_assembler.h"
#include "utilities/parameters.h"
#include <iostream>
#include <iomanip>
#include <fstream>

/**
 * @brief Print h_a field at a grid of points
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

    const double y_test = params.ic.pool_depth;  // At interface
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

#endif // FIELD_DIAGNOSTIC_H