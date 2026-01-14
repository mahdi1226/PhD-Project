// ============================================================================
// validation/validation.h - Validation Metrics and Reference Data
//
// Provides:
//   1. Lid-driven cavity: Ghia reference data, centerline extraction
//   2. Rising bubble: Centroid, circularity, rise velocity (Hysing benchmark)
//   3. Rosensweig: Wavelength extraction, linear stability comparison
//   4. General: Mass conservation, energy stability, symmetry checks
//
// Reference papers:
//   - Ghia, Ghia & Shin, J. Comp. Physics 48 (1982) 387-411
//   - Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009) 1259-1288
//   - Cowley & Rosensweig, J. Fluid Mech. 30 (1967) 671-688
//   - Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef VALIDATION_H
#define VALIDATION_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/geometry_info.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>

// ============================================================================
// PART 1: GHIA REFERENCE DATA FOR LID-DRIVEN CAVITY
// ============================================================================

namespace GhiaData
{
    // Number of reference points
    constexpr unsigned int n_points = 17;

    // y-positions for vertical centerline data: u_x(0.5, y)
    constexpr double y_positions[n_points] = {
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
        0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
        0.9531, 0.9609, 0.9688, 0.9766, 1.0000
    };

    // x-positions for horizontal centerline data: u_y(x, 0.5)
    constexpr double x_positions[n_points] = {
        0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
        0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
        0.9453, 0.9531, 0.9609, 0.9688, 1.0000
    };

    // u_x along vertical centerline (x=0.5) at Re=100
    constexpr double ux_vertical_Re100[n_points] = {
        0.0000, -0.0372, -0.0419, -0.0477, -0.0643, -0.1015,
        -0.1566, -0.2109, -0.2058, -0.1364, 0.0033, 0.2315,
        0.6872, 0.7372, 0.7887, 0.8412, 1.0000
    };

    // u_y along horizontal centerline (y=0.5) at Re=100
    constexpr double uy_horizontal_Re100[n_points] = {
        0.0000, 0.0923, 0.1009, 0.1089, 0.1232, 0.1608,
        0.1751, 0.1753, 0.0545, -0.2453, -0.2245, -0.1691,
        -0.1031, -0.0886, -0.0739, -0.0591, 0.0000
    };

    // u_x along vertical centerline (x=0.5) at Re=400
    constexpr double ux_vertical_Re400[n_points] = {
        0.0000, -0.0811, -0.0918, -0.1021, -0.1465, -0.2433,
        -0.3273, -0.1713, -0.1117, 0.0262, 0.1626, 0.2909,
        0.5550, 0.6179, 0.6863, 0.7577, 1.0000
    };

    // u_y along horizontal centerline (y=0.5) at Re=400
    constexpr double uy_horizontal_Re400[n_points] = {
        0.0000, 0.1836, 0.1971, 0.2099, 0.2292, 0.2800,
        0.3004, 0.2956, 0.0549, -0.2183, -0.2370, -0.2308,
        -0.1755, -0.1557, -0.1346, -0.1119, 0.0000
    };

    // u_x along vertical centerline (x=0.5) at Re=1000
    constexpr double ux_vertical_Re1000[n_points] = {
        0.0000, -0.1812, -0.2023, -0.2228, -0.3004, -0.3885,
        -0.2735, -0.0620, -0.0608, 0.0570, 0.1886, 0.3372,
        0.4723, 0.5169, 0.5808, 0.6644, 1.0000
    };

    // u_y along horizontal centerline (y=0.5) at Re=1000
    constexpr double uy_horizontal_Re1000[n_points] = {
        0.0000, 0.2780, 0.2973, 0.3163, 0.3430, 0.3709,
        0.3330, 0.3262, 0.0257, -0.3195, -0.4271, -0.5155,
        -0.3921, -0.3366, -0.2771, -0.2178, 0.0000
    };

    /**
     * @brief Get reference data for a given Reynolds number
     * @param Re Reynolds number (100, 400, or 1000)
     * @param ux_ref Output: pointer to u_x reference data
     * @param uy_ref Output: pointer to u_y reference data
     * @return true if Re is supported, false otherwise
     */
    inline bool get_reference_data(int Re,
                                   const double*& ux_ref,
                                   const double*& uy_ref)
    {
        switch (Re)
        {
            case 100:
                ux_ref = ux_vertical_Re100;
                uy_ref = uy_horizontal_Re100;
                return true;
            case 400:
                ux_ref = ux_vertical_Re400;
                uy_ref = uy_horizontal_Re400;
                return true;
            case 1000:
                ux_ref = ux_vertical_Re1000;
                uy_ref = uy_horizontal_Re1000;
                return true;
            default:
                ux_ref = nullptr;
                uy_ref = nullptr;
                return false;
        }
    }
}

// ============================================================================
// PART 2: CENTERLINE EXTRACTION
// ============================================================================

/**
 * @brief Centerline velocity data
 */
struct CenterlineData
{
    std::vector<double> positions;      // Coordinate along centerline
    std::vector<double> velocities;     // Velocity component values
    unsigned int n_points = 0;
    bool valid = false;
};

/**
 * @brief Extract u_x along vertical centerline x = x_coord
 *
 * For lid-driven cavity: extract u_x(x_coord, y) for y ∈ [y_min, y_max]
 */
template <int dim>
CenterlineData extract_ux_vertical_centerline(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    double x_coord,
    double y_min,
    double y_max,
    unsigned int n_samples = 129,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    CenterlineData result;
    result.positions.resize(n_samples);
    result.velocities.resize(n_samples, 0.0);
    result.n_points = n_samples;

    const double dy = (y_max - y_min) / (n_samples - 1);

    // Local contribution arrays
    std::vector<double> local_vel(n_samples, 0.0);
    std::vector<int> local_count(n_samples, 0);

    const auto& fe = ux_dof_handler.get_fe();
    const double tol = dy * 0.5;  // Tolerance for point location

    for (const auto& cell : ux_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        // Check if cell contains x_coord
        double cell_x_min = cell->vertex(0)[0];
        double cell_x_max = cell->vertex(0)[0];
        for (unsigned int v = 1; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            cell_x_min = std::min(cell_x_min, cell->vertex(v)[0]);
            cell_x_max = std::max(cell_x_max, cell->vertex(v)[0]);
        }

        if (x_coord < cell_x_min - tol || x_coord > cell_x_max + tol)
            continue;

        // Sample at vertices near x_coord
        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const dealii::Point<dim>& vertex = cell->vertex(v);
            if (std::abs(vertex[0] - x_coord) > tol)
                continue;

            double y = vertex[1];
            int bin = static_cast<int>((y - y_min) / dy + 0.5);

            if (bin >= 0 && bin < static_cast<int>(n_samples))
            {
                unsigned int dof = cell->vertex_dof_index(v, 0);
                local_vel[bin] += ux_solution[dof];
                local_count[bin] += 1;
            }
        }
    }

    // MPI reduction
    std::vector<double> global_vel(n_samples);
    std::vector<int> global_count(n_samples);

    MPI_Allreduce(local_vel.data(), global_vel.data(), n_samples,
                  MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_count.data(), global_count.data(), n_samples,
                  MPI_INT, MPI_SUM, comm);

    // Compute averages and fill result
    for (unsigned int i = 0; i < n_samples; ++i)
    {
        result.positions[i] = y_min + i * dy;
        if (global_count[i] > 0)
            result.velocities[i] = global_vel[i] / global_count[i];
    }

    result.valid = true;
    return result;
}

/**
 * @brief Extract u_y along horizontal centerline y = y_coord
 *
 * For lid-driven cavity: extract u_y(x, y_coord) for x ∈ [x_min, x_max]
 */
template <int dim>
CenterlineData extract_uy_horizontal_centerline(
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    double y_coord,
    double x_min,
    double x_max,
    unsigned int n_samples = 129,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    CenterlineData result;
    result.positions.resize(n_samples);
    result.velocities.resize(n_samples, 0.0);
    result.n_points = n_samples;

    const double dx = (x_max - x_min) / (n_samples - 1);

    std::vector<double> local_vel(n_samples, 0.0);
    std::vector<int> local_count(n_samples, 0);

    const auto& fe = uy_dof_handler.get_fe();
    const double tol = dx * 0.5;

    for (const auto& cell : uy_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        double cell_y_min = cell->vertex(0)[1];
        double cell_y_max = cell->vertex(0)[1];
        for (unsigned int v = 1; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            cell_y_min = std::min(cell_y_min, cell->vertex(v)[1]);
            cell_y_max = std::max(cell_y_max, cell->vertex(v)[1]);
        }

        if (y_coord < cell_y_min - tol || y_coord > cell_y_max + tol)
            continue;

        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            const dealii::Point<dim>& vertex = cell->vertex(v);
            if (std::abs(vertex[1] - y_coord) > tol)
                continue;

            double x = vertex[0];
            int bin = static_cast<int>((x - x_min) / dx + 0.5);

            if (bin >= 0 && bin < static_cast<int>(n_samples))
            {
                unsigned int dof = cell->vertex_dof_index(v, 0);
                local_vel[bin] += uy_solution[dof];
                local_count[bin] += 1;
            }
        }
    }

    std::vector<double> global_vel(n_samples);
    std::vector<int> global_count(n_samples);

    MPI_Allreduce(local_vel.data(), global_vel.data(), n_samples,
                  MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_count.data(), global_count.data(), n_samples,
                  MPI_INT, MPI_SUM, comm);

    for (unsigned int i = 0; i < n_samples; ++i)
    {
        result.positions[i] = x_min + i * dx;
        if (global_count[i] > 0)
            result.velocities[i] = global_vel[i] / global_count[i];
    }

    result.valid = true;
    return result;
}

// ============================================================================
// PART 3: CAVITY VALIDATION METRICS
// ============================================================================

/**
 * @brief Cavity validation results
 */
struct CavityValidation
{
    int reynolds = 0;
    double ux_rms_error = 0.0;      // RMS error in u_x along vertical centerline
    double uy_rms_error = 0.0;      // RMS error in u_y along horizontal centerline
    double ux_max_error = 0.0;      // Max error in u_x
    double uy_max_error = 0.0;      // Max error in u_y
    double combined_rms = 0.0;      // sqrt(ux_rms² + uy_rms²)
    bool passed = false;            // Combined RMS < tolerance
};

/**
 * @brief Interpolate computed centerline data at reference positions
 */
inline double interpolate_centerline(const CenterlineData& data,
                                     double position)
{
    if (!data.valid || data.positions.empty())
        return 0.0;

    // Find bracketing indices
    for (unsigned int i = 0; i < data.positions.size() - 1; ++i)
    {
        if (position >= data.positions[i] && position <= data.positions[i + 1])
        {
            double t = (position - data.positions[i]) /
                      (data.positions[i + 1] - data.positions[i]);
            return (1.0 - t) * data.velocities[i] + t * data.velocities[i + 1];
        }
    }

    // Extrapolation (shouldn't happen if positions cover [0,1])
    if (position <= data.positions.front())
        return data.velocities.front();
    return data.velocities.back();
}

/**
 * @brief Compare computed centerlines with Ghia reference data
 */
inline CavityValidation validate_cavity(
    const CenterlineData& ux_vertical,
    const CenterlineData& uy_horizontal,
    int reynolds,
    double tolerance = 0.05)
{
    CavityValidation result;
    result.reynolds = reynolds;

    const double* ux_ref = nullptr;
    const double* uy_ref = nullptr;

    if (!GhiaData::get_reference_data(reynolds, ux_ref, uy_ref))
    {
        result.passed = false;
        return result;
    }

    // Compute u_x errors along vertical centerline
    double ux_sum_sq = 0.0;
    for (unsigned int i = 0; i < GhiaData::n_points; ++i)
    {
        double computed = interpolate_centerline(ux_vertical, GhiaData::y_positions[i]);
        double error = std::abs(computed - ux_ref[i]);
        ux_sum_sq += error * error;
        result.ux_max_error = std::max(result.ux_max_error, error);
    }
    result.ux_rms_error = std::sqrt(ux_sum_sq / GhiaData::n_points);

    // Compute u_y errors along horizontal centerline
    double uy_sum_sq = 0.0;
    for (unsigned int i = 0; i < GhiaData::n_points; ++i)
    {
        double computed = interpolate_centerline(uy_horizontal, GhiaData::x_positions[i]);
        double error = std::abs(computed - uy_ref[i]);
        uy_sum_sq += error * error;
        result.uy_max_error = std::max(result.uy_max_error, error);
    }
    result.uy_rms_error = std::sqrt(uy_sum_sq / GhiaData::n_points);

    // Combined metric
    result.combined_rms = std::sqrt(result.ux_rms_error * result.ux_rms_error +
                                    result.uy_rms_error * result.uy_rms_error);

    result.passed = (result.combined_rms < tolerance);

    return result;
}

/**
 * @brief Write centerline comparison to CSV
 */
inline void write_cavity_comparison_csv(
    const CenterlineData& ux_vertical,
    const CenterlineData& uy_horizontal,
    int reynolds,
    const std::string& filename)
{
    const double* ux_ref = nullptr;
    const double* uy_ref = nullptr;
    GhiaData::get_reference_data(reynolds, ux_ref, uy_ref);

    std::ofstream file(filename);
    file << "# Lid-driven cavity validation at Re = " << reynolds << "\n";
    file << "# Comparison with Ghia, Ghia & Shin (1982)\n\n";

    // Vertical centerline: u_x(0.5, y)
    file << "# Vertical centerline: u_x(0.5, y)\n";
    file << "y,ux_computed,ux_ghia,error\n";
    for (unsigned int i = 0; i < GhiaData::n_points; ++i)
    {
        double y = GhiaData::y_positions[i];
        double computed = interpolate_centerline(ux_vertical, y);
        double ref = ux_ref ? ux_ref[i] : 0.0;
        file << std::setprecision(6)
             << y << "," << computed << "," << ref << "," << (computed - ref) << "\n";
    }

    file << "\n# Horizontal centerline: u_y(x, 0.5)\n";
    file << "x,uy_computed,uy_ghia,error\n";
    for (unsigned int i = 0; i < GhiaData::n_points; ++i)
    {
        double x = GhiaData::x_positions[i];
        double computed = interpolate_centerline(uy_horizontal, x);
        double ref = uy_ref ? uy_ref[i] : 0.0;
        file << std::setprecision(6)
             << x << "," << computed << "," << ref << "," << (computed - ref) << "\n";
    }

    file.close();
}

// ============================================================================
// PART 4: RISING BUBBLE METRICS (Hysing benchmark)
// ============================================================================

/**
 * @brief Bubble metrics for Hysing benchmark
 */
struct BubbleMetrics
{
    double time = 0.0;
    double centroid_x = 0.0;        // x-coordinate of bubble center
    double centroid_y = 0.0;        // y-coordinate of bubble center (rise height)
    double area = 0.0;              // Bubble area
    double perimeter = 0.0;         // Interface length (approximate)
    double circularity = 0.0;       // 4π·Area / Perimeter² (1.0 for circle)
    double rise_velocity = 0.0;     // dy_centroid/dt
    double equivalent_diameter = 0.0;
    bool valid = false;
};

/**
 * @brief Compute bubble metrics
 *
 * Convention: θ > 0 inside bubble, θ < 0 outside
 */
template <int dim>
BubbleMetrics compute_bubble_metrics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double time,
    double prev_centroid_y,
    double dt,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    static_assert(dim == 2, "Bubble metrics only implemented for 2D");

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    double local_area = 0.0;
    double local_x_moment = 0.0;
    double local_y_moment = 0.0;
    double local_perimeter = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const dealii::Point<dim>& p = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            // Smooth Heaviside: H(θ) = 1 inside bubble, 0 outside
            double H_theta;
            if (theta > 1.0)
                H_theta = 1.0;
            else if (theta < -1.0)
                H_theta = 0.0;
            else
                H_theta = 0.5 * (1.0 + theta);

            local_area += H_theta * JxW;
            local_x_moment += p[0] * H_theta * JxW;
            local_y_moment += p[1] * H_theta * JxW;

            // Perimeter approximation: |∇θ| weighted by interface indicator
            double delta_theta = std::exp(-theta * theta / 0.1);
            local_perimeter += theta_gradients[q].norm() * delta_theta * JxW;
        }
    }

    // MPI reductions
    BubbleMetrics result;
    result.time = time;

    double global_area = MPIUtils::reduce_sum(local_area, comm);
    double global_x_moment = MPIUtils::reduce_sum(local_x_moment, comm);
    double global_y_moment = MPIUtils::reduce_sum(local_y_moment, comm);
    double global_perimeter = MPIUtils::reduce_sum(local_perimeter, comm);

    if (global_area > 1e-10)
    {
        result.valid = true;
        result.area = global_area;
        result.centroid_x = global_x_moment / global_area;
        result.centroid_y = global_y_moment / global_area;
        result.perimeter = global_perimeter;
        result.equivalent_diameter = 2.0 * std::sqrt(global_area / M_PI);

        if (global_perimeter > 1e-10)
            result.circularity = 4.0 * M_PI * global_area /
                                (global_perimeter * global_perimeter);

        if (dt > 0 && prev_centroid_y > -1e10)
            result.rise_velocity = (result.centroid_y - prev_centroid_y) / dt;
    }

    return result;
}

/**
 * @brief Write bubble metrics time series to CSV
 */
inline void write_bubble_metrics_csv(
    const std::vector<BubbleMetrics>& history,
    const std::string& filename)
{
    std::ofstream file(filename);
    file << "time,centroid_x,centroid_y,area,perimeter,circularity,rise_velocity\n";

    for (const auto& m : history)
    {
        if (!m.valid) continue;
        file << std::setprecision(8)
             << m.time << ","
             << m.centroid_x << ","
             << m.centroid_y << ","
             << m.area << ","
             << m.perimeter << ","
             << m.circularity << ","
             << m.rise_velocity << "\n";
    }
    file.close();
}

// ============================================================================
// PART 5: ROSENSWEIG WAVELENGTH ANALYSIS
// ============================================================================

/**
 * @brief Interface profile for wavelength analysis
 */
struct InterfaceProfile
{
    std::vector<double> x;          // x positions
    std::vector<double> y;          // Interface height y(x)
    double y_mean = 0.0;
    double amplitude = 0.0;         // (y_max - y_min) / 2
    double dominant_wavelength = 0.0;
    unsigned int n_peaks = 0;
    bool valid = false;
};

/**
 * @brief Extract interface profile y(x) for wavelength analysis
 */
template <int dim>
InterfaceProfile extract_interface_profile(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double x_min,
    double x_max,
    unsigned int n_samples = 256,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    static_assert(dim == 2, "Interface profile only for 2D");

    InterfaceProfile profile;
    profile.x.resize(n_samples);
    profile.y.resize(n_samples, 0.0);

    const double dx = (x_max - x_min) / (n_samples - 1);

    std::vector<double> local_y(n_samples, -1e10);  // Use -1e10 as "not found"

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        // Get vertex data
        std::vector<double> vertex_theta(dealii::GeometryInfo<dim>::vertices_per_cell);
        std::vector<dealii::Point<dim>> vertex_pos(dealii::GeometryInfo<dim>::vertices_per_cell);

        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v)
        {
            vertex_pos[v] = cell->vertex(v);
            unsigned int dof = cell->vertex_dof_index(v, 0);
            vertex_theta[v] = theta_solution[dof];
        }

        // Check edges for sign changes
        for (unsigned int edge = 0; edge < dealii::GeometryInfo<dim>::lines_per_cell; ++edge)
        {
            unsigned int v1 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 0);
            unsigned int v2 = dealii::GeometryInfo<dim>::line_to_cell_vertices(edge, 1);

            if (vertex_theta[v1] * vertex_theta[v2] < 0)
            {
                double s = -vertex_theta[v1] / (vertex_theta[v2] - vertex_theta[v1]);
                dealii::Point<dim> crossing = vertex_pos[v1] +
                    s * (vertex_pos[v2] - vertex_pos[v1]);

                int bin = static_cast<int>((crossing[0] - x_min) / dx + 0.5);
                if (bin >= 0 && bin < static_cast<int>(n_samples))
                {
                    // Take maximum y (top of interface)
                    local_y[bin] = std::max(local_y[bin], crossing[1]);
                }
            }
        }
    }

    // MPI reduction: take max y at each x
    std::vector<double> global_y(n_samples);
    MPI_Allreduce(local_y.data(), global_y.data(), n_samples,
                  MPI_DOUBLE, MPI_MAX, comm);

    // Fill profile
    double y_sum = 0.0;
    double y_min = 1e10, y_max = -1e10;
    unsigned int valid_count = 0;

    for (unsigned int i = 0; i < n_samples; ++i)
    {
        profile.x[i] = x_min + i * dx;
        if (global_y[i] > -1e9)
        {
            profile.y[i] = global_y[i];
            y_sum += global_y[i];
            y_min = std::min(y_min, global_y[i]);
            y_max = std::max(y_max, global_y[i]);
            ++valid_count;
        }
    }

    if (valid_count > n_samples / 4)
    {
        profile.valid = true;
        profile.y_mean = y_sum / valid_count;
        profile.amplitude = (y_max - y_min) / 2.0;

        // Fill gaps with mean
        for (unsigned int i = 0; i < n_samples; ++i)
        {
            if (global_y[i] <= -1e9)
                profile.y[i] = profile.y_mean;
        }

        // Count peaks
        for (unsigned int i = 1; i < n_samples - 1; ++i)
        {
            if (profile.y[i] > profile.y[i-1] &&
                profile.y[i] > profile.y[i+1] &&
                profile.y[i] > profile.y_mean + 0.1 * profile.amplitude)
            {
                ++profile.n_peaks;
            }
        }

        // Estimate wavelength from peak count
        if (profile.n_peaks > 0)
        {
            profile.dominant_wavelength = (x_max - x_min) / profile.n_peaks;
        }
    }

    return profile;
}

/**
 * @brief Rosensweig linear stability prediction
 *
 * Critical wavelength: λ_c = 2π * sqrt(σ / (ρ * g))
 * where σ = surface tension, ρ = density, g = gravity
 */
inline double rosensweig_critical_wavelength(double surface_tension,
                                              double density,
                                              double gravity)
{
    return 2.0 * M_PI * std::sqrt(surface_tension / (density * gravity));
}

/**
 * @brief Compare measured wavelength with theory
 */
struct RosensweigValidation
{
    double measured_wavelength = 0.0;
    double predicted_wavelength = 0.0;
    double relative_error = 0.0;
    unsigned int n_spikes = 0;
    bool passed = false;
};

inline RosensweigValidation validate_rosensweig(
    const InterfaceProfile& profile,
    double surface_tension,
    double density,
    double gravity,
    double tolerance = 0.2)  // 20% tolerance
{
    RosensweigValidation result;

    if (!profile.valid || profile.n_peaks == 0)
        return result;

    result.measured_wavelength = profile.dominant_wavelength;
    result.predicted_wavelength = rosensweig_critical_wavelength(
        surface_tension, density, gravity);
    result.n_spikes = profile.n_peaks;

    if (result.predicted_wavelength > 1e-10)
    {
        result.relative_error = std::abs(result.measured_wavelength -
                                         result.predicted_wavelength) /
                                result.predicted_wavelength;
        result.passed = (result.relative_error < tolerance);
    }

    return result;
}

// ============================================================================
// PART 6: GENERAL VALIDATION UTILITIES
// ============================================================================

/**
 * @brief Mass conservation check
 */
struct MassConservation
{
    double initial_mass = 0.0;
    double current_mass = 0.0;
    double absolute_error = 0.0;
    double relative_error = 0.0;
    bool conserved = true;
};

inline MassConservation check_mass_conservation(double initial_mass,
                                                 double current_mass,
                                                 double tolerance = 1e-10)
{
    MassConservation result;
    result.initial_mass = initial_mass;
    result.current_mass = current_mass;
    result.absolute_error = current_mass - initial_mass;

    if (std::abs(initial_mass) > 1e-14)
        result.relative_error = result.absolute_error / std::abs(initial_mass);

    result.conserved = (std::abs(result.relative_error) < tolerance);
    return result;
}

/**
 * @brief Energy stability check
 */
struct EnergyStability
{
    double E_prev = 0.0;
    double E_curr = 0.0;
    double dE = 0.0;
    double relative_increase = 0.0;
    bool stable = true;
};

inline EnergyStability check_energy_stability(double E_prev,
                                               double E_curr,
                                               double tolerance = 1e-4)
{
    EnergyStability result;
    result.E_prev = E_prev;
    result.E_curr = E_curr;
    result.dE = E_curr - E_prev;

    if (std::abs(E_prev) > 1e-14)
        result.relative_increase = result.dE / std::abs(E_prev);

    result.stable = (result.relative_increase < tolerance);
    return result;
}

/**
 * @brief Symmetry check (for parallel debugging)
 */
template <int dim>
double check_left_right_symmetry(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    double x_mid,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);

    double local_left = 0.0;
    double local_right = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const dealii::Point<dim>& p = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            if (p[0] < x_mid)
                local_left += theta_values[q] * JxW;
            else
                local_right += theta_values[q] * JxW;
        }
    }

    double global_left = MPIUtils::reduce_sum(local_left, comm);
    double global_right = MPIUtils::reduce_sum(local_right, comm);

    double sum = std::abs(global_left) + std::abs(global_right);
    if (sum > 1e-14)
        return std::abs(global_left - global_right) / sum;

    return 0.0;
}

#endif // VALIDATION_H