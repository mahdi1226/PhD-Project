// ============================================================================
// utilities/poisson_mms.h - MMS solutions for magnetostatic Poisson verification
//
// Equation: -∇·(μ(c)∇φ) = f
//
// For MMS verification, we choose:
//   φ = t^4 sin(πx) sin(πy)   (zero on boundary)
//   c = t^4 cos(πx) cos(πy)   (from NS-CH MMS)
//   μ(c) = 1 + χ_m * (1+c)/2
//
// Then compute f = -∇·(μ(c)∇φ) analytically
// ============================================================================
#ifndef POISSON_MMS_H
#define POISSON_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>

// Get MMS params from nsch_mms.h
#include "nsch_mms.h"

// ============================================================================
// Exact magnetic potential: φ = t^4 sin(πx) sin(πy)
// ============================================================================
template <int dim>
class MMSExactPotential : public dealii::Function<dim>
{
public:
    MMSExactPotential() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;
        return t4 * std::sin(pi * x) * std::sin(pi * y);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                     const unsigned int = 0) const
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;

        dealii::Tensor<1, dim> grad;
        grad[0] = t4 * pi * std::cos(pi * x) * std::sin(pi * y);
        grad[1] = t4 * pi * std::sin(pi * x) * std::cos(pi * y);
        return grad;
    }
};

// ============================================================================
// Source term for Poisson: f = -∇·(μ(c)∇φ)
//
// Let μ = 1 + χ_m * (1+c)/2
// Then ∇μ = (χ_m/2) ∇c
//
// -∇·(μ∇φ) = -∇μ·∇φ - μ Δφ
//          = -(χ_m/2)(∇c·∇φ) - μ Δφ
// ============================================================================
template <int dim>
class MMSSourcePoisson : public dealii::Function<dim>
{
public:
    MMSSourcePoisson(double chi_m_val = 1.0)
        : dealii::Function<dim>(1), chi_m(chi_m_val) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;

        const double sin_px = std::sin(pi * x), cos_px = std::cos(pi * x);
        const double sin_py = std::sin(pi * y), cos_py = std::cos(pi * y);

        // Phase field: c = t^4 cos(πx) cos(πy)
        const double c = t4 * cos_px * cos_py;

        // ∇c = t^4 π [-sin(πx)cos(πy), -cos(πx)sin(πy)]
        const double c_x = -t4 * pi * sin_px * cos_py;
        const double c_y = -t4 * pi * cos_px * sin_py;

        // Potential: φ = t^4 sin(πx) sin(πy)
        // ∇φ = t^4 π [cos(πx)sin(πy), sin(πx)cos(πy)]
        const double phi_x = t4 * pi * cos_px * sin_py;
        const double phi_y = t4 * pi * sin_px * cos_py;

        // Δφ = -2π² t^4 sin(πx) sin(πy)
        const double lap_phi = -2.0 * pi * pi * t4 * sin_px * sin_py;

        // Permeability: μ = 1 + χ_m * (1+c)/2
        const double mu = 1.0 + chi_m * (1.0 + c) / 2.0;

        // ∇c · ∇φ
        const double grad_c_dot_grad_phi = c_x * phi_x + c_y * phi_y;

        // f = -∇·(μ∇φ) = -(χ_m/2)(∇c·∇φ) - μ Δφ
        return -(chi_m / 2.0) * grad_c_dot_grad_phi - mu * lap_phi;
    }

private:
    double chi_m;
};

// ============================================================================
// Compute Poisson MMS error
// ============================================================================
template <int dim>
double compute_poisson_L2_error(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>&  solution,
    double current_time)
{
    MMSExactPotential<dim> exact_phi;
    exact_phi.set_time(current_time);

    dealii::Vector<float> difference_per_cell(
        dof_handler.get_triangulation().n_active_cells());

    dealii::VectorTools::integrate_difference(
        dof_handler,
        solution,
        exact_phi,
        difference_per_cell,
        dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
        dealii::VectorTools::L2_norm);

    return dealii::VectorTools::compute_global_error(
        dof_handler.get_triangulation(),
        difference_per_cell,
        dealii::VectorTools::L2_norm);
}

template <int dim>
double compute_poisson_H1_error(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>&  solution,
    double current_time)
{
    MMSExactPotential<dim> exact_phi;
    exact_phi.set_time(current_time);

    dealii::Vector<float> difference_per_cell(
        dof_handler.get_triangulation().n_active_cells());

    dealii::VectorTools::integrate_difference(
        dof_handler,
        solution,
        exact_phi,
        difference_per_cell,
        dealii::QGauss<dim>(dof_handler.get_fe().degree + 2),
        dealii::VectorTools::H1_seminorm);

    return dealii::VectorTools::compute_global_error(
        dof_handler.get_triangulation(),
        difference_per_cell,
        dealii::VectorTools::H1_seminorm);
}

#endif // POISSON_MMS_H