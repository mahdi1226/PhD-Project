// ============================================================================
// diagnostics/interface_tracking.h - Interface Position Tracking
//
// Computes the y-coordinates where θ ≈ 0 (the interface).
// Used for Rosensweig instability monitoring and Figure 7 comparison.
//
// Two methods provided:
//   1. Threshold-based: Find quadrature points where |θ| < threshold
//   2. Robust (recommended): Find edges where θ changes sign and interpolate
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef INTERFACE_TRACKING_H
#define INTERFACE_TRACKING_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/geometry_info.h>

#include <limits>
#include <cmath>
#include <iostream>

/**
 * @brief Interface position data
 */
struct InterfacePosition
{
    double y_min = 0.0;                  // Minimum y where θ = 0
    double y_max = 0.0;                  // Maximum y where θ = 0
    double y_mean = 0.0;                 // Mean interface position
    double amplitude = 0.0;              // (y_max - y_min) / 2 (spike height)
    unsigned int n_interface_points = 0; // Number of interface points found
    bool valid = false;                  // Whether interface was found
};

/**
 * @brief Compute interface position by finding where θ ≈ 0 (threshold method)
 *
 * The interface is defined as the region where |θ| < threshold.
 * We find the min/max y-coordinates in this region.
 *
 * @param theta_dof_handler  DoFHandler for θ
 * @param theta_solution     Phase field solution
 * @param threshold          Threshold for |θ| to be considered interface (default 0.5)
 * @param debug              Print debug information
 * @return InterfacePosition struct
 */
template <int dim>
InterfacePosition compute_interface_position(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    double threshold = 0.5,
    bool debug = false)
{
    InterfacePosition result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();

    double y_sum = 0.0;
    double weight_sum = 0.0;
    unsigned int interface_point_count = 0;

    // Debug: track θ statistics
    double theta_min_seen = std::numeric_limits<double>::max();
    double theta_max_seen = std::numeric_limits<double>::lowest();
    unsigned int total_q_points = 0;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    std::vector<double> theta_values(n_q_points);

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            ++total_q_points;

            // Track θ range
            theta_min_seen = std::min(theta_min_seen, theta);
            theta_max_seen = std::max(theta_max_seen, theta);

            // Check if we're near the interface
            if (std::abs(theta) < threshold)
            {
                const dealii::Point<dim>& point = fe_values.quadrature_point(q);
                const double y = point[1];
                const double JxW = fe_values.JxW(q);

                const double weight = (threshold - std::abs(theta)) * JxW;

                result.y_min = std::min(result.y_min, y);
                result.y_max = std::max(result.y_max, y);

                y_sum += y * weight;
                weight_sum += weight;
                ++interface_point_count;

                result.valid = true;
            }
        }
    }

    result.n_interface_points = interface_point_count;

    if (debug)
    {
        std::cout << "[InterfaceTracking THRESHOLD DEBUG]\n";
        std::cout << "  Total quadrature points: " << total_q_points << "\n";
        std::cout << "  θ range seen: [" << theta_min_seen << ", " << theta_max_seen << "]\n";
        std::cout << "  Threshold: |θ| < " << threshold << "\n";
        std::cout << "  Interface points found: " << interface_point_count << "\n";
        if (result.valid)
        {
            std::cout << "  y_min = " << result.y_min << "\n";
            std::cout << "  y_max = " << result.y_max << "\n";
        }
        else
        {
            std::cout << "  *** NO INTERFACE FOUND! ***\n";
        }
    }

    if (result.valid && weight_sum > 0)
    {
        result.y_mean = y_sum / weight_sum;
        result.amplitude = (result.y_max - result.y_min) / 2.0;
    }
    else
    {
        result.y_min = 0.0;
        result.y_max = 0.0;
        result.y_mean = 0.0;
        result.amplitude = 0.0;
    }

    return result;
}

/**
 * @brief Robust interface tracking by finding θ = 0 crossings (RECOMMENDED)
 *
 * Instead of looking for |θ| < threshold, find cell vertex pairs where θ
 * changes sign and interpolate the exact crossing location.
 *
 * This method works even for very sharp interfaces (small ε) where the
 * transition region may be narrower than mesh cells.
 *
 * @param theta_dof_handler  DoFHandler for θ
 * @param theta_solution     Phase field solution
 * @param debug              Print debug information
 * @return InterfacePosition struct
 */
template <int dim>
InterfacePosition compute_interface_position_robust(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    bool debug = false)
{
    InterfacePosition result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();

    double y_sum = 0.0;
    unsigned int crossing_count = 0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        // Get θ at all vertices
        std::vector<double> vertex_theta(dealii::GeometryInfo<dim>::vertices_per_cell);
        std::vector<dealii::Point<dim>> vertex_points(dealii::GeometryInfo<dim>::vertices_per_cell);

        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            vertex_points[v] = cell->vertex(v);
            const unsigned int vertex_dof = cell->vertex_dof_index(v, 0);
            vertex_theta[v] = theta_solution[vertex_dof];
        }

        // Check each vertex pair for sign change
        for (unsigned int v1 = 0; v1 < dealii::GeometryInfo<dim>::vertices_per_cell; ++v1)
        {
            for (unsigned int v2 = v1 + 1; v2 < dealii::GeometryInfo<dim>::vertices_per_cell; ++v2)
            {
                const double t1 = vertex_theta[v1];
                const double t2 = vertex_theta[v2];

                // Sign change means interface crosses this edge
                if (t1 * t2 < 0)  // Different signs
                {
                    // Linear interpolation: find where θ = 0
                    // t1 + s*(t2 - t1) = 0  =>  s = -t1 / (t2 - t1)
                    const double s = -t1 / (t2 - t1);

                    // Interpolate position
                    const dealii::Point<dim> crossing =
                        vertex_points[v1] + s * (vertex_points[v2] - vertex_points[v1]);

                    const double y = crossing[1];

                    result.y_min = std::min(result.y_min, y);
                    result.y_max = std::max(result.y_max, y);
                    y_sum += y;
                    ++crossing_count;
                    result.valid = true;
                }
            }
        }
    }

    result.n_interface_points = crossing_count;

    if (debug)
    {
        std::cout << "[InterfaceTracking ROBUST DEBUG]\n";
        std::cout << "  Zero-crossings found: " << crossing_count << "\n";
        if (result.valid)
        {
            std::cout << "  y_min = " << std::scientific << result.y_min << "\n";
            std::cout << "  y_max = " << result.y_max << "\n";
            std::cout << "  Spike amplitude = " << (result.y_max - result.y_min) << "\n";
        }
        else
        {
            std::cout << "  *** NO INTERFACE CROSSINGS FOUND! ***\n";
        }
    }

    if (result.valid && crossing_count > 0)
    {
        result.y_mean = y_sum / crossing_count;
        result.amplitude = (result.y_max - result.y_min) / 2.0;
    }
    else
    {
        result.y_min = 0.0;
        result.y_max = 0.0;
        result.y_mean = 0.0;
        result.amplitude = 0.0;
    }

    return result;
}

#endif // INTERFACE_TRACKING_H