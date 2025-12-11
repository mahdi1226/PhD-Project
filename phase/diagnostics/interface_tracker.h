// ============================================================================
// diagnostics/interface_tracker.h - Track interface shape and spike locations
//
// Finds the c=0 contour and identifies local maxima (peaks/spikes)
// Reports (x,y) location of each spike for pattern analysis
// ============================================================================
#ifndef INTERFACE_TRACKER_H
#define INTERFACE_TRACKER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_vector.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

struct SpikeInfo
{
    double x;           // x-location of spike
    double y;           // y-location (height) of spike
    double prominence;  // How much higher than neighbors
};

struct InterfaceProfile
{
    // Global metrics
    double y_min;
    double y_max;
    double amplitude;
    double mean_height;

    // Spike information
    std::vector<SpikeInfo> spikes;
    std::vector<SpikeInfo> valleys;

    // Full interface trace (sorted by x)
    std::vector<std::pair<double, double>> interface_points; // (x, y) pairs
};

/**
 * @brief Extract interface profile from phase field
 *
 * Traces the c=0 contour and finds peaks/valleys
 */
template <int dim>
InterfaceProfile compute_interface_profile(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const dealii::BlockVector<double>& ch_solution,
    unsigned int fe_degree = 2,
    double c_threshold = 0.3)  // Points with |c| < threshold are "on interface"
{
    InterfaceProfile result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();
    result.mean_height = 0.0;

    const auto& fe = ch_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe_degree + 2);

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_quadrature_points);

    const dealii::FEValuesExtractors::Scalar c_extract(0);

    // Convert block vector to full vector
    const unsigned int n0 = ch_solution.block(0).size();
    const unsigned int n1 = ch_solution.block(1).size();
    dealii::Vector<double> ch_full(n0 + n1);
    for (unsigned int i = 0; i < n0; ++i)
        ch_full[i] = ch_solution.block(0)[i];
    for (unsigned int i = 0; i < n1; ++i)
        ch_full[n0 + i] = ch_solution.block(1)[i];

    std::vector<double> c_values(quadrature.size());

    // Collect all interface points
    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_extract].get_function_values(ch_full, c_values);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            if (std::abs(c_values[q]) < c_threshold)
            {
                const dealii::Point<dim>& p = fe_values.quadrature_point(q);
                double x = p[0];
                double y = p[1];

                result.interface_points.push_back({x, y});
                result.y_min = std::min(result.y_min, y);
                result.y_max = std::max(result.y_max, y);
                result.mean_height += y;
            }
        }
    }

    if (result.interface_points.empty())
    {
        result.y_min = result.y_max = result.amplitude = result.mean_height = 0.0;
        return result;
    }

    result.mean_height /= result.interface_points.size();
    result.amplitude = result.y_max - result.y_min;

    // Sort interface points by x-coordinate
    std::sort(result.interface_points.begin(), result.interface_points.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Bin the interface into x-slices to get a clean profile
    // This handles the diffuse interface (multiple y values per x)
    const double x_min = result.interface_points.front().first;
    const double x_max = result.interface_points.back().first;
    const int n_bins = 100;
    const double dx = (x_max - x_min) / n_bins;

    std::vector<double> y_profile(n_bins, 0.0);
    std::vector<int> y_count(n_bins, 0);

    for (const auto& pt : result.interface_points)
    {
        int bin = static_cast<int>((pt.first - x_min) / dx);
        bin = std::min(bin, n_bins - 1);
        bin = std::max(bin, 0);

        // Take the maximum y in each bin (top of interface)
        if (y_count[bin] == 0 || pt.second > y_profile[bin])
            y_profile[bin] = pt.second;
        y_count[bin]++;
    }

    // Find local maxima (spikes) and minima (valleys)
    for (int i = 2; i < n_bins - 2; ++i)
    {
        if (y_count[i] == 0) continue;

        double y_curr = y_profile[i];

        // Check for local maximum (spike)
        bool is_max = true;
        bool is_min = true;
        double neighbor_avg = 0.0;
        int neighbor_count = 0;

        for (int j = -2; j <= 2; ++j)
        {
            if (j == 0 || y_count[i+j] == 0) continue;
            double y_neighbor = y_profile[i+j];
            if (y_neighbor >= y_curr) is_max = false;
            if (y_neighbor <= y_curr) is_min = false;
            neighbor_avg += y_neighbor;
            neighbor_count++;
        }

        if (neighbor_count > 0)
            neighbor_avg /= neighbor_count;

        double x_pos = x_min + (i + 0.5) * dx;

        if (is_max && y_curr > result.mean_height)
        {
            SpikeInfo spike;
            spike.x = x_pos;
            spike.y = y_curr;
            spike.prominence = y_curr - neighbor_avg;

            // Only report significant spikes
            if (spike.prominence > 0.005)
                result.spikes.push_back(spike);
        }

        if (is_min && y_curr < result.mean_height)
        {
            SpikeInfo valley;
            valley.x = x_pos;
            valley.y = y_curr;
            valley.prominence = neighbor_avg - y_curr;

            if (valley.prominence > 0.005)
                result.valleys.push_back(valley);
        }
    }

    // Sort spikes by height (tallest first)
    std::sort(result.spikes.begin(), result.spikes.end(),
        [](const auto& a, const auto& b) { return a.y > b.y; });

    return result;
}

/**
 * @brief Print interface profile summary
 */
inline void print_interface_profile(const InterfaceProfile& profile)
{
    std::cout << "  [INTERFACE] amp=" << std::fixed << std::setprecision(3) << profile.amplitude
              << " y=[" << profile.y_min << "," << profile.y_max << "]"
              << " spikes=" << profile.spikes.size()
              << " valleys=" << profile.valleys.size() << "\n";

    // Print spike locations
    if (!profile.spikes.empty())
    {
        std::cout << "    Spikes: ";
        for (size_t i = 0; i < std::min(profile.spikes.size(), size_t(5)); ++i)
        {
            const auto& s = profile.spikes[i];
            std::cout << "(x=" << s.x << ",y=" << s.y << ")";
            if (i < profile.spikes.size() - 1 && i < 4) std::cout << " ";
        }
        if (profile.spikes.size() > 5)
            std::cout << " ... (" << profile.spikes.size() << " total)";
        std::cout << "\n";
    }

    if (!profile.valleys.empty())
    {
        std::cout << "    Valleys: ";
        for (size_t i = 0; i < std::min(profile.valleys.size(), size_t(5)); ++i)
        {
            const auto& v = profile.valleys[i];
            std::cout << "(x=" << v.x << ",y=" << v.y << ")";
            if (i < profile.valleys.size() - 1 && i < 4) std::cout << " ";
        }
        if (profile.valleys.size() > 5)
            std::cout << " ... (" << profile.valleys.size() << " total)";
        std::cout << "\n";
    }
}

/**
 * @brief Compact one-line output for time loop
 */
inline void print_interface_compact(const InterfaceProfile& profile)
{
    std::cout << "  [SPIKE] amp=" << std::fixed << std::setprecision(3) << profile.amplitude
              << " y=[" << profile.y_min << "," << profile.y_max << "]"
              << " n_spikes=" << profile.spikes.size();

    // Show x-positions of up to 3 spikes
    if (!profile.spikes.empty())
    {
        std::cout << " at x={";
        for (size_t i = 0; i < std::min(profile.spikes.size(), size_t(3)); ++i)
        {
            std::cout << profile.spikes[i].x;
            if (i < profile.spikes.size() - 1 && i < 2) std::cout << ",";
        }
        if (profile.spikes.size() > 3) std::cout << ",...";
        std::cout << "}";
    }
    std::cout << "\n";
}

#endif // INTERFACE_TRACKER_H