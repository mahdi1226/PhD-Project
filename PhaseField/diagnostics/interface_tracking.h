// ============================================================================
// diagnostics/interface_tracking.h - Interface Position Tracking
//
// Computes the y-coordinates where θ ≈ 0 (the interface).
// Used for Rosensweig instability monitoring.
// ============================================================================
#ifndef INTERFACE_TRACKING_H
#define INTERFACE_TRACKING_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <limits>
#include <cmath>

/**
 * @brief Interface position data
 */
struct InterfacePosition
{
    double y_min = 0.0;     // Minimum y where θ ≈ 0
    double y_max = 0.0;     // Maximum y where θ ≈ 0
    double y_mean = 0.0;    // Mean interface position
    double amplitude = 0.0; // (y_max - y_min) / 2
    bool valid = false;     // Whether interface was found
};

/**
 * @brief Compute interface position by finding where θ ≈ 0
 *
 * The interface is defined as the region where |θ| < threshold.
 * We find the min/max y-coordinates in this region.
 *
 * @param theta_dof_handler  DoFHandler for θ
 * @param theta_solution     Phase field solution
 * @param threshold          Threshold for |θ| to be considered interface (default 0.5)
 * @return InterfacePosition struct
 */
template <int dim>
InterfacePosition compute_interface_position(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    double threshold = 0.5)
{
    InterfacePosition result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();

    double y_sum = 0.0;
    double weight_sum = 0.0;

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

            // Check if we're near the interface
            if (std::abs(theta) < threshold)
            {
                const dealii::Point<dim>& point = fe_values.quadrature_point(q);
                const double y = point[1];  // y-coordinate
                const double JxW = fe_values.JxW(q);

                // Weight by how close to θ = 0
                const double weight = (threshold - std::abs(theta)) * JxW;

                result.y_min = std::min(result.y_min, y);
                result.y_max = std::max(result.y_max, y);

                y_sum += y * weight;
                weight_sum += weight;

                result.valid = true;
            }
        }
    }

    if (result.valid && weight_sum > 0)
    {
        result.y_mean = y_sum / weight_sum;
        result.amplitude = (result.y_max - result.y_min) / 2.0;
    }
    else
    {
        // No interface found - reset to zero
        result.y_min = 0.0;
        result.y_max = 0.0;
        result.y_mean = 0.0;
        result.amplitude = 0.0;
    }

    return result;
}

/**
 * @brief Alternative: Find interface by interpolating θ = 0 crossings
 *
 * More accurate but more expensive. Finds actual zero crossings
 * along vertical lines.
 */
template <int dim>
InterfacePosition compute_interface_position_interpolated(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution)
{
    InterfacePosition result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();

    double y_sum = 0.0;
    unsigned int count = 0;

    const auto& fe = theta_dof_handler.get_fe();

    // Use face quadrature to find crossings on vertical edges
    dealii::QGauss<dim-1> face_quadrature(fe.degree + 1);

    dealii::FEFaceValues<dim> fe_face_values(fe, face_quadrature,
        dealii::update_values | dealii::update_quadrature_points);

    const unsigned int n_face_q_points = face_quadrature.size();
    std::vector<double> theta_face_values(n_face_q_points);

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
        {
            // Only check horizontal faces (top/bottom)
            // In 2D: face 2 = bottom, face 3 = top
            if (face != 2 && face != 3)
                continue;

            fe_face_values.reinit(cell, face);
            fe_face_values.get_function_values(theta_solution, theta_face_values);

            // Check for sign change
            bool has_positive = false;
            bool has_negative = false;

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
                if (theta_face_values[q] > 0) has_positive = true;
                if (theta_face_values[q] < 0) has_negative = true;
            }

            // If sign change, interface crosses this face
            if (has_positive && has_negative)
            {
                // Approximate y-coordinate of crossing
                const dealii::Point<dim>& center = cell->face(face)->center();
                const double y = center[1];

                result.y_min = std::min(result.y_min, y);
                result.y_max = std::max(result.y_max, y);
                y_sum += y;
                ++count;
                result.valid = true;
            }
        }
    }

    if (result.valid && count > 0)
    {
        result.y_mean = y_sum / count;
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