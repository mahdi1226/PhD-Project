// ============================================================================
// nsch_verification.cc - Verification metrics implementation
// ============================================================================
#include "nsch_verification.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

// ============================================================================
// Helper: convert CH block vector to full vector
// ============================================================================
inline dealii::Vector<double> ch_to_full(const CHVector& block_vec)
{
    const unsigned int n0 = block_vec.block(0).size();
    const unsigned int n1 = block_vec.block(1).size();
    dealii::Vector<double> full(n0 + n1);

    for (unsigned int i = 0; i < n0; ++i)
        full[i] = block_vec.block(0)[i];
    for (unsigned int i = 0; i < n1; ++i)
        full[n0 + i] = block_vec.block(1)[i];

    return full;
}

// ============================================================================
// Helper: convert NS block vector to full vector
// ============================================================================
inline dealii::Vector<double> ns_to_full(const NSVector& block_vec)
{
    const unsigned int n0 = block_vec.block(0).size();
    const unsigned int n1 = block_vec.block(1).size();
    dealii::Vector<double> full(n0 + n1);

    for (unsigned int i = 0; i < n0; ++i)
        full[i] = block_vec.block(0)[i];
    for (unsigned int i = 0; i < n1; ++i)
        full[n0 + i] = block_vec.block(1)[i];

    return full;
}

// ============================================================================
// Compute mass: ∫c dΩ
// ============================================================================
template <int dim>
double compute_mass(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution)
{
    const auto& fe = ch_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> c_vals(n_q);
    const dealii::FEValuesExtractors::Scalar c_ex(0);
    const auto ch_full = ch_to_full(ch_solution);

    double mass = 0.0;
    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_ex].get_function_values(ch_full, c_vals);
        for (unsigned int q = 0; q < n_q; ++q)
            mass += c_vals[q] * fe_values.JxW(q);
    }
    return mass;
}

// ============================================================================
// Compute CH energy: E_CH = ∫[½λ|∇c|² + W(c)] dΩ
// ============================================================================
template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution,
    double                         lambda)
{
    const auto& fe = ch_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> c_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> c_grads(n_q);
    const dealii::FEValuesExtractors::Scalar c_ex(0);
    const auto ch_full = ch_to_full(ch_solution);

    double energy = 0.0;
    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_ex].get_function_values(ch_full, c_vals);
        fe_values[c_ex].get_function_gradients(ch_full, c_grads);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double c = c_vals[q];
            const double grad_c_sq = c_grads[q] * c_grads[q];
            const double W = 0.25 * (c * c - 1.0) * (c * c - 1.0);
            energy += (0.5 * lambda * grad_c_sq + W) * fe_values.JxW(q);
        }
    }
    return energy;
}

// ============================================================================
// Compute min/max of concentration
// ============================================================================
template <int dim>
std::pair<double, double> compute_c_bounds(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution)
{
    const auto& fe = ch_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature, dealii::update_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> c_vals(n_q);
    const dealii::FEValuesExtractors::Scalar c_ex(0);
    const auto ch_full = ch_to_full(ch_solution);

    double c_min = std::numeric_limits<double>::max();
    double c_max = std::numeric_limits<double>::lowest();

    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_ex].get_function_values(ch_full, c_vals);
        for (unsigned int q = 0; q < n_q; ++q)
        {
            c_min = std::min(c_min, c_vals[q]);
            c_max = std::max(c_max, c_vals[q]);
        }
    }
    return {c_min, c_max};
}

// ============================================================================
// Compute divergence: ‖∇·u‖_L2
// ============================================================================
template <int dim>
double compute_divergence_L2(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution)
{
    const auto& fe = ns_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<dealii::Tensor<2, dim>> u_grads(n_q);
    const dealii::FEValuesExtractors::Vector u_ex(0);
    const auto ns_full = ns_to_full(ns_solution);

    double div_sq = 0.0;
    for (const auto& cell : ns_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[u_ex].get_function_gradients(ns_full, u_grads);
        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double div_u = dealii::trace(u_grads[q]);
            div_sq += div_u * div_u * fe_values.JxW(q);
        }
    }
    return std::sqrt(div_sq);
}

// ============================================================================
// Compute kinetic energy: E_K = ½∫|u|² dΩ
// ============================================================================
template <int dim>
double compute_kinetic_energy(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution)
{
    const auto& fe = ns_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<dealii::Tensor<1, dim>> u_vals(n_q);
    const dealii::FEValuesExtractors::Vector u_ex(0);
    const auto ns_full = ns_to_full(ns_solution);

    double ke = 0.0;
    for (const auto& cell : ns_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[u_ex].get_function_values(ns_full, u_vals);
        for (unsigned int q = 0; q < n_q; ++q)
            ke += 0.5 * (u_vals[q] * u_vals[q]) * fe_values.JxW(q);
    }
    return ke;
}

// ============================================================================
// Compute max velocity magnitude
// ============================================================================
template <int dim>
double compute_max_velocity(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution)
{
    const auto& fe = ns_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature, dealii::update_values);

    const unsigned int n_q = quadrature.size();
    std::vector<dealii::Tensor<1, dim>> u_vals(n_q);
    const dealii::FEValuesExtractors::Vector u_ex(0);
    const auto ns_full = ns_to_full(ns_solution);

    double u_max = 0.0;
    for (const auto& cell : ns_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[u_ex].get_function_values(ns_full, u_vals);
        for (unsigned int q = 0; q < n_q; ++q)
            u_max = std::max(u_max, u_vals[q].norm());
    }
    return u_max;
}

// ============================================================================
// Compute interface area: ∫|∇c| dΩ
// ============================================================================
template <int dim>
double compute_interface_area(
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution)
{
    const auto& fe = ch_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<dealii::Tensor<1, dim>> c_grads(n_q);
    const dealii::FEValuesExtractors::Scalar c_ex(0);
    const auto ch_full = ch_to_full(ch_solution);

    double area = 0.0;
    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[c_ex].get_function_gradients(ch_full, c_grads);
        for (unsigned int q = 0; q < n_q; ++q)
            area += c_grads[q].norm() * fe_values.JxW(q);
    }
    return area;
}

// ============================================================================
// Compute all metrics
// ============================================================================
template <int dim>
NSCHVerificationMetrics compute_nsch_metrics(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const NSVector&                ns_solution,
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const CHVector&                ch_solution,
    double                         lambda,
    double                         dt,
    double                         h,
    double                         old_total_energy)
{
    NSCHVerificationMetrics m;

    m.mass = compute_mass(ch_dof_handler, ch_solution);
    m.ch_energy = compute_ch_energy(ch_dof_handler, ch_solution, lambda);
    auto [c_min, c_max] = compute_c_bounds(ch_dof_handler, ch_solution);
    m.c_min = c_min;
    m.c_max = c_max;

    m.divergence_L2 = compute_divergence_L2(ns_dof_handler, ns_solution);
    m.kinetic_energy = compute_kinetic_energy(ns_dof_handler, ns_solution);
    m.u_max = compute_max_velocity(ns_dof_handler, ns_solution);

    m.total_energy = m.kinetic_energy + m.ch_energy;
    m.energy_rate = (m.total_energy - old_total_energy) / dt;
    m.interface_area = compute_interface_area(ch_dof_handler, ch_solution);
    m.cfl_number = m.u_max * dt / h;

    return m;
}

// ============================================================================
// Print verification header
// ============================================================================
void print_nsch_verification_header()
{
    std::cout << "\n"
              << std::setw(6)  << "step"
              << std::setw(11) << "time"
              << std::setw(12) << "Mass"
              << std::setw(12) << "E_total"
              << std::setw(12) << "E_kinetic"
              << std::setw(12) << "div_u"
              << std::setw(10) << "c_min"
              << std::setw(10) << "c_max"
              << std::setw(10) << "|u|_max"
              << std::setw(8)  << "CFL"
              << "\n" << std::string(111, '-') << "\n";
}

// ============================================================================
// Print single verification line
// ============================================================================
void print_nsch_verification_line(unsigned int step, double time,
                                   const NSCHVerificationMetrics& m)
{
    std::cout << std::scientific << std::setprecision(2)
              << std::setw(6)  << step
              << std::setw(11) << time
              << std::setw(12) << m.mass
              << std::setw(12) << m.total_energy
              << std::setw(12) << m.kinetic_energy
              << std::setw(12) << m.divergence_L2
              << std::fixed << std::setprecision(4)
              << std::setw(10) << m.c_min
              << std::setw(10) << m.c_max
              << std::scientific << std::setprecision(2)
              << std::setw(10) << m.u_max
              << std::fixed << std::setprecision(3)
              << std::setw(8) << m.cfl_number;

    if (m.c_min < -1.1 || m.c_max > 1.1) std::cout << " !BOUNDS";
    if (m.energy_rate > 1e-8) std::cout << " !E_up";
    if (m.cfl_number > 1.0) std::cout << " !CFL";
    if (std::isnan(m.total_energy) || std::isinf(m.total_energy)) std::cout << " !NaN";
    std::cout << "\n";
}

// ============================================================================
// Print detailed summary
// ============================================================================
void print_nsch_verification_summary(const NSCHVerificationMetrics& m)
{
    std::cout << "\n"
              << "+--------------------------------------------------------------+\n"
              << "|           NS-CH VERIFICATION SUMMARY                         |\n"
              << "+--------------------------------------------------------------+\n"
              << std::scientific << std::setprecision(6)
              << "| Cahn-Hilliard:                                               |\n"
              << "|   Mass (integral c)  = " << std::setw(14) << m.mass << "                    |\n"
              << "|   CH Energy          = " << std::setw(14) << m.ch_energy << "                    |\n"
              << std::fixed << std::setprecision(4)
              << "|   c in [" << std::setw(8) << m.c_min << ", " << std::setw(8) << m.c_max << "]"
              << (m.c_min >= -1.05 && m.c_max <= 1.05 ? "  OK bounds" : "  BOUNDS VIOLATED") << "        |\n"
              << "+--------------------------------------------------------------+\n"
              << std::scientific << std::setprecision(6)
              << "| Navier-Stokes:                                               |\n"
              << "|   Kinetic Energy     = " << std::setw(14) << m.kinetic_energy << "                    |\n"
              << "|   ||div u||_L2       = " << std::setw(14) << m.divergence_L2 << "                    |\n"
              << "|   |u|_max            = " << std::setw(14) << m.u_max << "                    |\n"
              << "+--------------------------------------------------------------+\n"
              << "| Coupled:                                                     |\n"
              << "|   Total Energy       = " << std::setw(14) << m.total_energy << "                    |\n"
              << "|   dE/dt              = " << std::setw(14) << m.energy_rate
              << (m.energy_rate <= 1e-10 ? "  dissipating" : "  INCREASING") << "      |\n"
              << "|   Interface area     = " << std::setw(14) << m.interface_area << "                    |\n"
              << "+--------------------------------------------------------------+\n\n";
}

// ============================================================================
// Health check
// ============================================================================
bool check_nsch_health(const NSCHVerificationMetrics& m)
{
    if (std::isnan(m.mass) || std::isinf(m.mass)) return false;
    if (std::isnan(m.total_energy) || std::isinf(m.total_energy)) return false;
    if (std::isnan(m.divergence_L2) || std::isinf(m.divergence_L2)) return false;
    if (m.c_min < -2.0 || m.c_max > 2.0) return false;
    if (m.total_energy > 1e10) return false;
    if (m.cfl_number > 10.0) return false;
    return true;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template double compute_mass<2>(const dealii::DoFHandler<2>&, const CHVector&);
template double compute_ch_energy<2>(const dealii::DoFHandler<2>&, const CHVector&, double);
template std::pair<double, double> compute_c_bounds<2>(const dealii::DoFHandler<2>&, const CHVector&);
template double compute_divergence_L2<2>(const dealii::DoFHandler<2>&, const NSVector&);
template double compute_kinetic_energy<2>(const dealii::DoFHandler<2>&, const NSVector&);
template double compute_max_velocity<2>(const dealii::DoFHandler<2>&, const NSVector&);
template double compute_interface_area<2>(const dealii::DoFHandler<2>&, const CHVector&);
template NSCHVerificationMetrics compute_nsch_metrics<2>(
    const dealii::DoFHandler<2>&, const NSVector&,
    const dealii::DoFHandler<2>&, const CHVector&,
    double, double, double, double);