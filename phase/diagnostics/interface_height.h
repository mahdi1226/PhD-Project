// ============================================================================
// diagnostics/interface_height.h - Measure Rosensweig spike heights
//
// Computes interface position (where c ≈ 0) and tracks:
//   - y_max: highest point (spike peak)
//   - y_min: lowest point (valley)
//   - amplitude: y_max - y_min
//   - mean_height: average interface position
// ============================================================================
#ifndef INTERFACE_HEIGHT_H
#define INTERFACE_HEIGHT_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_vector.h>

#include <limits>
#include <cmath>

struct InterfaceMetrics
{
    double y_min;        // Lowest interface point
    double y_max;        // Highest interface point
    double amplitude;    // y_max - y_min
    double mean_height;  // Average interface position
    int    num_crossings; // Number of interface crossings found
};

/**
 * @brief Compute interface height metrics from phase field solution
 *
 * Finds cells where c crosses zero and tracks the y-coordinates
 * of the interface to measure spike heights.
 *
 * @param ch_dof_handler DoFHandler for CH system
 * @param ch_solution Current CH solution [c, mu]
 * @param fe_degree Polynomial degree of phase field FE
 * @return InterfaceMetrics containing height information
 */
template <int dim>
InterfaceMetrics compute_interface_height(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const dealii::BlockVector<double>& ch_solution,
    unsigned int fe_degree = 2)
{
    InterfaceMetrics result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();
    result.mean_height = 0.0;
    result.num_crossings = 0;

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
    double total_interface_y = 0.0;
    int total_crossings = 0;

    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_extract].get_function_values(ch_full, c_values);

        // Check if this cell contains the interface (c crosses 0)
        double c_min_cell = *std::min_element(c_values.begin(), c_values.end());
        double c_max_cell = *std::max_element(c_values.begin(), c_values.end());

        // Interface is where c ≈ 0 (between -1 and +1)
        // Cell contains interface if c changes sign or is near zero
        if (c_min_cell <= 0.1 && c_max_cell >= -0.1)
        {
            // Find approximate interface position within cell
            for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
                const double c = c_values[q];

                // Point is on/near interface if |c| < threshold
                if (std::abs(c) < 0.5)
                {
                    const dealii::Point<dim>& p = fe_values.quadrature_point(q);
                    double y = p[1];  // y-coordinate

                    result.y_min = std::min(result.y_min, y);
                    result.y_max = std::max(result.y_max, y);
                    total_interface_y += y;
                    total_crossings++;
                }
            }
        }
    }

    if (total_crossings > 0)
    {
        result.mean_height = total_interface_y / total_crossings;
        result.amplitude = result.y_max - result.y_min;
        result.num_crossings = total_crossings;
    }
    else
    {
        // No interface found - return zeros
        result.y_min = 0.0;
        result.y_max = 0.0;
        result.amplitude = 0.0;
        result.mean_height = 0.0;
    }

    return result;
}

/**
 * @brief Print interface height to console
 */
inline void print_interface_height(const InterfaceMetrics& m)
{
    std::cout << "  Interface: y_min=" << m.y_min
              << ", y_max=" << m.y_max
              << ", amplitude=" << m.amplitude
              << ", mean=" << m.mean_height << "\n";
}

#endif // INTERFACE_HEIGHT_H