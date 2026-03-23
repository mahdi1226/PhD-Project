// ============================================================================
// diagnostics/magnetic_diagnostics.h - Magnetic Subsystem Diagnostics (Parallel)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Computes:
//   - φ bounds (min, max)
//   - H field magnitude |∇φ| (L², max)
//   - Magnetic energy: ½∫ μ(θ)|∇φ|² dx
//   - M magnitude (quasi-equilibrium): |χ(θ)∇φ|
//
// Uses the auxiliary phi_dof_handler (CG scalar) extracted from the
// monolithic M+φ system. All quantities are MPI-reduced.
// ============================================================================
#ifndef MAGNETIC_DIAGNOSTICS_H
#define MAGNETIC_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"
#include "physics/applied_field.h"
#include "physics/material_properties.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

// ============================================================================
// Magnetic Diagnostic Data
// ============================================================================
struct MagneticDiagnostics
{
    double phi_min = 0.0;
    double phi_max = 0.0;

    double H_L2_norm = 0.0;      // ||∇φ||_{L²}
    double H_max = 0.0;          // max |∇φ|

    double M_L2_norm = 0.0;      // ||M||_{L²} (quasi-equilibrium)
    double M_max = 0.0;          // max |M|

    double magnetic_energy = 0.0; // ½∫ μ(θ)|∇φ|² dx

    double mu_min = 1.0;
    double mu_max = 1.0;
};

// ============================================================================
// Compute magnetic diagnostics (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
MagneticDiagnostics compute_magnetic_diagnostics(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<double> theta_values(n_q_points);

    double local_H_L2_sq = 0.0;
    double local_M_L2_sq = 0.0;
    double local_magnetic_energy = 0.0;
    double local_H_max = 0.0;
    double local_M_max = 0.0;
    double local_phi_min = std::numeric_limits<double>::max();
    double local_phi_max = std::numeric_limits<double>::lowest();
    double local_mu_min = std::numeric_limits<double>::max();
    double local_mu_max = std::numeric_limits<double>::lowest();

    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        if (!phi_cell->is_locally_owned())
            continue;

        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        phi_fe_values.get_function_gradients(phi_solution, grad_phi_values);
        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Tensor<1, dim>& H = grad_phi_values[q];
            const double H_norm = H.norm();

            const double theta_q = theta_values[q];
            const double chi_theta = susceptibility(theta_q,
                params.physics.epsilon, params.physics.chi_0);
            const double mu_theta = 1.0 + chi_theta;
            const double M_norm = chi_theta * H_norm;

            local_H_L2_sq += H_norm * H_norm * JxW;
            local_M_L2_sq += M_norm * M_norm * JxW;
            local_magnetic_energy += 0.5 * mu_theta * H_norm * H_norm * JxW;

            local_H_max = std::max(local_H_max, H_norm);
            local_M_max = std::max(local_M_max, M_norm);
            local_mu_min = std::min(local_mu_min, mu_theta);
            local_mu_max = std::max(local_mu_max, mu_theta);
        }
    }

    // φ bounds from solution vector
    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        std::vector<dealii::types::global_dof_index> local_dof_indices(
            phi_dof_handler.get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto& idx : local_dof_indices)
        {
            if (phi_solution.locally_owned_elements().is_element(idx))
            {
                const double phi_val = phi_solution[idx];
                local_phi_min = std::min(local_phi_min, phi_val);
                local_phi_max = std::max(local_phi_max, phi_val);
            }
        }
    }

    MagneticDiagnostics result;

    double global_H_L2_sq = MPIUtils::reduce_sum(local_H_L2_sq, comm);
    double global_M_L2_sq = MPIUtils::reduce_sum(local_M_L2_sq, comm);

    result.H_L2_norm = std::sqrt(global_H_L2_sq);
    result.M_L2_norm = std::sqrt(global_M_L2_sq);
    result.magnetic_energy = MPIUtils::reduce_sum(local_magnetic_energy, comm);

    result.H_max = MPIUtils::reduce_max(local_H_max, comm);
    result.M_max = MPIUtils::reduce_max(local_M_max, comm);

    result.phi_min = MPIUtils::reduce_min(local_phi_min, comm);
    result.phi_max = MPIUtils::reduce_max(local_phi_max, comm);

    result.mu_min = MPIUtils::reduce_min(local_mu_min, comm);
    result.mu_max = MPIUtils::reduce_max(local_mu_max, comm);

    return result;
}

// ============================================================================
// Field Distribution Data (applied field h_a diagnostics)
// ============================================================================
struct FieldDistributionData
{
    double time = 0.0;
    unsigned int step = 0;

    double h_a_x_left = 0.0;
    double h_a_x_center = 0.0;
    double h_a_x_right = 0.0;

    double h_a_center_x = 0.0;
    double h_a_center_y = 0.0;

    double h_a_min = 0.0;
    double h_a_max = 0.0;
    double center_to_edge_ratio = 0.0;
    double uniformity = 0.0;

    double ramp_factor = 0.0;

    static std::string header()
    {
        return "step,time,ramp_factor,h_a_left,h_a_center,h_a_right,"
               "h_a_center_x,h_a_center_y,h_a_min,h_a_max,"
               "center_edge_ratio,uniformity";
    }

    std::string to_csv() const
    {
        std::ostringstream oss;
        oss << step << ","
            << std::scientific << std::setprecision(6)
            << time << "," << ramp_factor << ","
            << h_a_x_left << "," << h_a_x_center << "," << h_a_x_right << ","
            << h_a_center_x << "," << h_a_center_y << ","
            << h_a_min << "," << h_a_max << ","
            << center_to_edge_ratio << "," << uniformity;
        return oss.str();
    }
};

// ============================================================================
// Compute field distribution at interface level
// ============================================================================
inline FieldDistributionData compute_field_distribution(
    const Parameters& params, double time, unsigned int step)
{
    FieldDistributionData data;
    data.time = time;
    data.step = step;

    data.ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(time / params.dipoles.ramp_time, 1.0) : 1.0;

    const double y_interface = params.ic.pool_depth;
    const double x_min = params.domain.x_min;
    const double x_max = params.domain.x_max;
    const double Lx = x_max - x_min;

    const int n_samples = 11;
    double min_mag = 1e20, max_mag = 0.0;

    for (int i = 0; i < n_samples; ++i)
    {
        double x = x_min + i * Lx / (n_samples - 1);
        dealii::Point<2> p(x, y_interface);
        auto h_a = compute_applied_field<2>(p, params, time);
        double mag = h_a.norm();

        min_mag = std::min(min_mag, mag);
        max_mag = std::max(max_mag, mag);

        if (i == 1) data.h_a_x_left = mag;
        else if (i == n_samples / 2)
        {
            data.h_a_x_center = mag;
            data.h_a_center_x = h_a[0];
            data.h_a_center_y = h_a[1];
        }
        else if (i == n_samples - 2) data.h_a_x_right = mag;
    }

    data.h_a_min = min_mag;
    data.h_a_max = max_mag;

    double edge_avg = 0.5 * (data.h_a_x_left + data.h_a_x_right);
    data.center_to_edge_ratio = (edge_avg > 1e-12) ? data.h_a_x_center / edge_avg : 0.0;
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

    void open(const std::string& filename, const Parameters& params,
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

    void write(const FieldDistributionData& data)
    {
        if (!is_root_ || !file_.is_open()) return;
        file_ << data.to_csv() << "\n";
        file_.flush();
    }

    void close()
    {
        if (is_root_ && file_.is_open()) file_.close();
    }

    bool is_open() const { return is_root_ && file_.is_open(); }

private:
    std::ofstream file_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    bool is_root_ = false;
};

// ============================================================================
// Print applied field diagnostic (console, rank 0 only)
// ============================================================================
inline void diagnose_applied_field(const Parameters& params, double time,
                                    MPI_Comm comm = MPI_COMM_WORLD)
{
    if (!MPIUtils::is_root(comm)) return;

    std::cout << "\n============================================================\n"
              << "APPLIED FIELD DIAGNOSTIC at t = " << time << "\n"
              << "============================================================\n"
              << "Dipole configuration:\n"
              << "  Number of dipoles: " << params.dipoles.positions.size() << "\n"
              << "  Direction: (" << params.dipoles.direction[0]
              << ", " << params.dipoles.direction[1] << ")\n"
              << "  Intensity max: " << params.dipoles.intensity_max << "\n"
              << "  Ramp time: " << params.dipoles.ramp_time << "\n";

    const double ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(time / params.dipoles.ramp_time, 1.0) : 1.0;
    std::cout << "  Current ramp factor: " << ramp_factor << "\n";

    const double y_test = params.ic.pool_depth;
    std::cout << "\nField h_a at interface level (y = " << y_test << "):\n"
              << std::setw(10) << "x" << std::setw(15) << "h_a_x"
              << std::setw(15) << "h_a_y" << std::setw(15) << "|h_a|" << "\n"
              << "----------------------------------------------------\n";

    double max_mag = 0.0, min_mag = 1e10;
    for (double x = 0.0; x <= 1.0; x += 0.1)
    {
        dealii::Point<2> p(x, y_test);
        auto h_a = compute_applied_field<2>(p, params, time);
        double mag = h_a.norm();
        max_mag = std::max(max_mag, mag);
        min_mag = std::min(min_mag, mag);
        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << x
                  << std::setw(15) << std::scientific << std::setprecision(4) << h_a[0]
                  << std::setw(15) << h_a[1] << std::setw(15) << mag << "\n";
    }

    std::cout << "\nField uniformity:\n"
              << "  |h_a| range: [" << min_mag << ", " << max_mag << "]\n"
              << "  Ratio max/min: " << (min_mag > 0 ? max_mag/min_mag : 0) << "\n";

    dealii::Point<2> p_center(0.5, y_test), p_edge(0.1, y_test);
    auto h_center = compute_applied_field<2>(p_center, params, time);
    auto h_edge = compute_applied_field<2>(p_edge, params, time);
    std::cout << "  Center/Edge ratio: "
              << h_center.norm() / (h_edge.norm() + 1e-12) << "\n"
              << "============================================================\n\n";
}

// ============================================================================
// Write h_a field to CSV (rank 0 only)
// ============================================================================
inline void write_field_csv(const Parameters& params, double time,
                            const std::string& filename,
                            MPI_Comm comm = MPI_COMM_WORLD)
{
    if (!MPIUtils::is_root(comm)) return;
    std::ofstream file(filename);
    file << "x,y,h_a_x,h_a_y,magnitude\n";
    const int nx = 50, ny = 30;
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
                 << h_a[0] << "," << h_a[1] << "," << h_a.norm() << "\n";
        }
    }
    file.close();
}

inline void write_field_profile(const Parameters& params, double time,
                                unsigned int step, const std::string& output_dir,
                                MPI_Comm comm = MPI_COMM_WORLD)
{
    std::string filename = output_dir + "/h_a_profile_" + std::to_string(step) + ".csv";
    write_field_csv(params, time, filename, comm);
}

#endif // MAGNETIC_DIAGNOSTICS_H
