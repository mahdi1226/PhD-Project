// ============================================================================
// diagnostics/nsch_scalar_diagnostics.h - Diagnostics for scalar-based architecture
//
// Helper functions that compute metrics from separate scalar solutions
// instead of BlockVectors. This bridges the gap to the existing verification code.
// ============================================================================
#ifndef NSCH_DIAGNOSTICS_H
#define NSCH_DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include "diagnostics/nsch_verification.h"
#include "diagnostics/interface_tracker.h"

#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================================
// Compute NSCH metrics from separate scalar solutions
// ============================================================================
template <int dim>
NSCHVerificationMetrics compute_nsch_metrics_from_scalars(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::Vector<double>& c_solution,
    const dealii::Vector<double>& mu_solution,
    double lambda,
    double dt,
    double h_min,
    double old_total_energy)
{
    NSCHVerificationMetrics m;
    m.c_min = std::numeric_limits<double>::max();
    m.c_max = std::numeric_limits<double>::lowest();
    m.u_max = 0.0;
    m.mass = 0.0;
    m.kinetic_energy = 0.0;
    m.ch_energy = 0.0;
    m.divergence_L2 = 0.0;

    // Quadrature
    dealii::QGauss<dim> quadrature(3);
    const unsigned int n_q = quadrature.size();

    // FEValues for concentration
    dealii::FEValues<dim> c_fe_values(c_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    // FEValues for velocity
    dealii::FEValues<dim> ux_fe_values(ux_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(ux_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients);

    std::vector<double> c_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> c_grads(n_q);
    std::vector<double> ux_vals(n_q);
    std::vector<double> uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> ux_grads(n_q);
    std::vector<dealii::Tensor<1, dim>> uy_grads(n_q);

    auto c_cell = c_dof_handler.begin_active();
    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = ux_dof_handler.begin_active();  // Same as ux

    for (; c_cell != c_dof_handler.end(); ++c_cell, ++ux_cell, ++uy_cell)
    {
        c_fe_values.reinit(c_cell);
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        c_fe_values.get_function_values(c_solution, c_vals);
        c_fe_values.get_function_gradients(c_solution, c_grads);
        ux_fe_values.get_function_values(ux_solution, ux_vals);
        uy_fe_values.get_function_values(uy_solution, uy_vals);
        ux_fe_values.get_function_gradients(ux_solution, ux_grads);
        uy_fe_values.get_function_gradients(uy_solution, uy_grads);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = c_fe_values.JxW(q);
            const double c = c_vals[q];
            const double ux = ux_vals[q];
            const double uy = uy_vals[q];
            const double u_mag = std::sqrt(ux * ux + uy * uy);
            const double div_u = ux_grads[q][0] + uy_grads[q][1];
            const double grad_c_sq = c_grads[q].norm_square();

            m.c_min = std::min(m.c_min, c);
            m.c_max = std::max(m.c_max, c);
            m.u_max = std::max(m.u_max, u_mag);

            m.mass += c * JxW;
            m.kinetic_energy += 0.5 * (ux * ux + uy * uy) * JxW;

            // CH energy: (lambda/2)|grad c|^2 + (1/4)(c^2-1)^2
            double dw = (c * c - 1.0);
            m.ch_energy += (0.5 * lambda * grad_c_sq + 0.25 * dw * dw) * JxW;

            m.divergence_L2 += div_u * div_u * JxW;
        }
    }

    m.divergence_L2 = std::sqrt(m.divergence_L2);
    m.total_energy = m.kinetic_energy + m.ch_energy;

    // CFL number
    if (dt > 0 && h_min > 0 && m.u_max > 1e-14)
        m.cfl_number = m.u_max * dt / h_min;
    else
        m.cfl_number = 0.0;

    // Note: energy_change_rate not in struct, tracked externally if needed
    (void)old_total_energy;
    (void)mu_solution;

    return m;
}

// ============================================================================
// Compute interface profile from scalar concentration solution
// Includes spike detection via binning and local maxima finding
// ============================================================================
template <int dim>
InterfaceProfile compute_interface_profile_from_scalar(
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::Vector<double>& c_solution,
    unsigned int fe_degree,
    double c_threshold = 0.3)  // Points with |c| < threshold are "on interface"
{
    InterfaceProfile result;
    result.y_min = std::numeric_limits<double>::max();
    result.y_max = std::numeric_limits<double>::lowest();
    result.mean_height = 0.0;

    dealii::QGauss<dim> quadrature(fe_degree + 2);
    dealii::FEValues<dim> fe_values(c_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_quadrature_points);

    std::vector<double> c_values(quadrature.size());

    // Collect all interface points (where |c| < threshold)
    for (const auto& cell : c_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(c_solution, c_values);

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

    if (x_max - x_min < 1e-10)
    {
        // Degenerate case - no x spread
        return result;
    }

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

            // Only report significant spikes (prominence > 0.5% of domain)
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

#endif