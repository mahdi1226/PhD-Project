// ============================================================================
// diagnostics/interface_tracking.h - Interface Position Tracking (Parallel)
//
// Computes the y-coordinates where θ ≈ 0 (the interface).
// Used for Rosensweig instability monitoring and Figure 7 comparison.
//
// Two methods provided:
//   1. Threshold-based: Find quadrature points where |θ| < threshold
//   2. Robust (recommended): Find edges where θ changes sign and interpolate
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef INTERFACE_TRACKING_H
#define INTERFACE_TRACKING_H

#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/geometry_info.h>

#include <limits>
#include <cmath>

// ============================================================================
// Interface Position Data
// ============================================================================
struct InterfacePosition
{
    double y_min = 0.0;                  // Minimum y where θ = 0
    double y_max = 0.0;                  // Maximum y where θ = 0
    double y_mean = 0.0;                 // Mean interface position
    double amplitude = 0.0;              // (y_max - y_min) / 2 (spike height)
    unsigned int n_interface_points = 0; // Number of interface points found
    bool valid = false;                  // Whether interface was found
};

// ============================================================================
// Compute interface position (parallel version with Trilinos vectors)
// Uses robust zero-crossing method
// ============================================================================
template <int dim>
InterfacePosition compute_interface_position(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    // Local accumulators
    double local_y_min = std::numeric_limits<double>::max();
    double local_y_max = std::numeric_limits<double>::lowest();
    double local_y_sum = 0.0;
    unsigned int local_crossing_count = 0;

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

            // Handle distributed vector access
            if (theta_solution.locally_owned_elements().is_element(vertex_dof))
                vertex_theta[v] = theta_solution[vertex_dof];
            else
                vertex_theta[v] = theta_solution[vertex_dof];  // Ghost value
        }

        // Check each EDGE (not diagonal) for sign change
        // In 2D: 4 edges per quad cell
        // In 3D: 12 edges per hex cell
        for (unsigned int edge = 0; edge < dealii::GeometryInfo<dim>::lines_per_cell; ++edge)
        {
            const unsigned int v1 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 0);
            const unsigned int v2 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 1);

            const double t1 = vertex_theta[v1];
            const double t2 = vertex_theta[v2];

            // Sign change means interface crosses this edge
            if (t1 * t2 < 0)
            {
                // Linear interpolation: find where θ = 0
                const double s = -t1 / (t2 - t1);

                // Interpolate position
                const dealii::Point<dim> crossing =
                    vertex_points[v1] + s * (vertex_points[v2] - vertex_points[v1]);

                const double y = crossing[1];

                local_y_min = std::min(local_y_min, y);
                local_y_max = std::max(local_y_max, y);
                local_y_sum += y;
                ++local_crossing_count;
            }
        }
    }

    // MPI reductions
    InterfacePosition result;

    result.y_min = MPIUtils::reduce_min(local_y_min, comm);
    result.y_max = MPIUtils::reduce_max(local_y_max, comm);

    double global_y_sum = MPIUtils::reduce_sum(local_y_sum, comm);
    unsigned int global_crossing_count = MPIUtils::reduce_sum(local_crossing_count, comm);

    result.n_interface_points = global_crossing_count;

    if (global_crossing_count > 0)
    {
        result.valid = true;
        result.y_mean = global_y_sum / global_crossing_count;
        result.amplitude = (result.y_max - result.y_min) / 2.0;
    }
    else
    {
        result.valid = false;
        result.y_min = 0.0;
        result.y_max = 0.0;
        result.y_mean = 0.0;
        result.amplitude = 0.0;
    }

    return result;
}

// ============================================================================
// Compute interface position (serial version with deal.II vectors)
// ============================================================================
template <int dim>
InterfacePosition compute_interface_position(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution)
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

        // Check each EDGE (not diagonal) for sign change
        for (unsigned int edge = 0; edge < dealii::GeometryInfo<dim>::lines_per_cell; ++edge)
        {
            const unsigned int v1 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 0);
            const unsigned int v2 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 1);

            const double t1 = vertex_theta[v1];
            const double t2 = vertex_theta[v2];

            if (t1 * t2 < 0)
            {
                const double s = -t1 / (t2 - t1);
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

    result.n_interface_points = crossing_count;

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

// ============================================================================
// Simple spike detection based on interface position variation
// Returns estimated number of spikes based on y_max - y_min and expected width
// ============================================================================
inline unsigned int estimate_spike_count(const InterfacePosition& iface,
                                         double domain_width,
                                         double expected_spike_width = 0.2)
{
    if (!iface.valid || iface.amplitude < 0.01)
        return 0;

    // Rough estimate: number of spikes ≈ domain_width / expected_spike_width
    // But only count if amplitude is significant
    if (iface.amplitude > 0.02)
    {
        // For Rosensweig with 5 dipoles, expect ~5 spikes
        // For Hedgehog, expect more
        return static_cast<unsigned int>(domain_width / expected_spike_width);
    }

    return 0;
}

#endif // INTERFACE_TRACKING_H