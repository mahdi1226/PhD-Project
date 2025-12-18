// ============================================================================
// diagnostics/ch_diagnostics.cc - Cahn-Hilliard Diagnostics Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "diagnostics/ch_diagnostics.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
// Compute θ bounds
// ============================================================================
void compute_theta_bounds(
    const dealii::Vector<double>& theta_solution,
    double& theta_min,
    double& theta_max)
{
    theta_min = *std::min_element(theta_solution.begin(), theta_solution.end());
    theta_max = *std::max_element(theta_solution.begin(), theta_solution.end());
}

// ============================================================================
// Check θ bounds violation
// ============================================================================
bool check_theta_bounds(double theta_min, double theta_max, double tolerance)
{
    return (theta_min < -1.0 - tolerance) || (theta_max > 1.0 + tolerance);
}

// ============================================================================
// Compute mass ∫θ dΩ
// ============================================================================
template <int dim>
double compute_mass(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution)
{
    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);

    double mass = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
            mass += theta_values[q] * fe_values.JxW(q);
    }

    return mass;
}

// ============================================================================
// Compute CH energy
// ============================================================================
template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    double epsilon)
{
    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    double energy = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double grad_theta_sq = theta_gradients[q].norm_square();

            // F(θ) = (1/4)(θ² - 1)²
            const double theta_sq = theta * theta;
            const double F_theta = 0.25 * (theta_sq - 1.0) * (theta_sq - 1.0);

            // E = ε/2 |∇θ|² + (1/ε) F(θ)
            energy += (0.5 * epsilon * grad_theta_sq +
                       (1.0 / epsilon) * F_theta) * fe_values.JxW(q);
        }
    }

    return energy;
}

// ============================================================================
// Compute all CH diagnostics
// ============================================================================
template <int dim>
CHDiagnosticData compute_ch_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    double epsilon,
    unsigned int step,
    double time,
    double dt,
    double energy_prev)
{
    CHDiagnosticData data;
    data.step = step;
    data.time = time;
    data.energy_prev = energy_prev;

    // Bounds
    compute_theta_bounds(theta_solution, data.theta_min, data.theta_max);
    data.bounds_violated = check_theta_bounds(data.theta_min, data.theta_max);

    // Mass
    data.mass = compute_mass<dim>(theta_dof_handler, theta_solution);

    // Energy
    data.energy = compute_ch_energy<dim>(theta_dof_handler, theta_solution, epsilon);

    // Energy rate
    if (step > 0 && dt > 0.0)
    {
        data.energy_rate = (data.energy - energy_prev) / dt;
        data.energy_increasing = (data.energy > energy_prev + 1e-12);
    }

    return data;
}

// ============================================================================
// Print CH diagnostics
// ============================================================================
void print_ch_diagnostics(const CHDiagnosticData& data, bool verbose)
{
    if (data.bounds_violated)
    {
        std::cout << "[CH WARNING] θ out of bounds: ["
                  << std::setprecision(4) << data.theta_min << ", "
                  << data.theta_max << "]\n";
    }

    if (data.energy_increasing && data.step > 0)
    {
        std::cout << "[CH WARNING] Energy increasing: dE/dt = "
                  << std::scientific << std::setprecision(2)
                  << data.energy_rate << "\n";
    }

    if (verbose)
    {
        std::cout << "[CH] step=" << data.step
                  << " θ∈[" << std::setprecision(4) << data.theta_min
                  << "," << data.theta_max << "]"
                  << " mass=" << std::setprecision(6) << data.mass
                  << " E=" << std::scientific << data.energy
                  << "\n";
    }
}

// ============================================================================
// CH Diagnostics Logger
// ============================================================================
void CHDiagnosticsLogger::open(const std::string& filename)
{
    file_.open(filename);
    if (!file_.is_open())
        throw std::runtime_error("Failed to open CH diagnostics file: " + filename);

    file_ << "step,time,theta_min,theta_max,bounds_violated,mass,energy,energy_rate,energy_increasing\n";
    file_.flush();
}

void CHDiagnosticsLogger::write(const CHDiagnosticData& data)
{
    if (!file_.is_open())
        return;

    file_ << data.step << ","
          << std::setprecision(10) << data.time << ","
          << std::setprecision(8) << data.theta_min << ","
          << data.theta_max << ","
          << (data.bounds_violated ? 1 : 0) << ","
          << data.mass << ","
          << data.energy << ","
          << data.energy_rate << ","
          << (data.energy_increasing ? 1 : 0) << "\n";
    file_.flush();
}

void CHDiagnosticsLogger::close()
{
    if (file_.is_open())
        file_.close();
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template double compute_mass<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&);

template double compute_ch_energy<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    double);

template CHDiagnosticData compute_ch_diagnostics<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    double,
    unsigned int,
    double,
    double,
    double);