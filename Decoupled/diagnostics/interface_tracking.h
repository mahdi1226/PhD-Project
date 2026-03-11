// ============================================================================
// diagnostics/interface_tracking.h — Interface Position Tracking (Parallel)
//
// Computes the y-coordinates where θ ≈ 0 (the diffuse interface).
// Used for Rosensweig instability monitoring: spike heights, amplitude,
// and comparison with Zhang, He & Yang (SIAM J. Sci. Comput. 43, 2021).
//
// Method: Zero-crossing detection on cell edges.
//   For each cell edge, checks if θ changes sign between vertices.
//   If so, linearly interpolates to find the crossing y-coordinate.
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef INTERFACE_TRACKING_H
#define INTERFACE_TRACKING_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/geometry_info.h>

#include <mpi.h>
#include <limits>
#include <cmath>

// ============================================================================
// Interface Position Data
// ============================================================================
struct InterfacePosition
{
    double y_min = 0.0;                  // Minimum y where θ = 0 (valley bottom)
    double y_max = 0.0;                  // Maximum y where θ = 0 (spike tip)
    double y_mean = 0.0;                 // Mean interface position
    double amplitude = 0.0;              // (y_max - y_min) / 2 (spike height)
    unsigned int n_interface_points = 0; // Number of zero crossings found
    bool valid = false;                  // Whether interface was found
};

// ============================================================================
// Compute interface position (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
InterfacePosition compute_interface_position(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    MPI_Comm comm)
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
            vertex_theta[v] = theta_solution[vertex_dof];
        }

        // Check each edge for sign change
        for (unsigned int edge = 0; edge < dealii::GeometryInfo<dim>::lines_per_cell; ++edge)
        {
            const unsigned int v1 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 0);
            const unsigned int v2 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 1);

            const double t1 = vertex_theta[v1];
            const double t2 = vertex_theta[v2];

            // Sign change → interface crosses this edge
            if (t1 * t2 < 0)
            {
                // Linear interpolation: find where θ = 0
                const double s = -t1 / (t2 - t1);
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

    double global_y_min = local_y_min;
    double global_y_max = local_y_max;
    double global_y_sum = local_y_sum;
    unsigned int global_crossing_count = local_crossing_count;

    MPI_Allreduce(MPI_IN_PLACE, &global_y_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_y_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_y_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_crossing_count, 1, MPI_UNSIGNED, MPI_SUM, comm);

    result.n_interface_points = global_crossing_count;

    if (global_crossing_count > 0)
    {
        result.valid = true;
        result.y_min = global_y_min;
        result.y_max = global_y_max;
        result.y_mean = global_y_sum / global_crossing_count;
        result.amplitude = (result.y_max - result.y_min) / 2.0;
    }

    return result;
}

#endif // INTERFACE_TRACKING_H
