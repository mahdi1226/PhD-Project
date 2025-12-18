// ============================================================================
// mms/ns_mms.cc - Navier-Stokes Method of Manufactured Solutions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/ns_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// ============================================================================
// Print MMS errors
// ============================================================================
void NSMMSError::print(unsigned int refinement, double h) const
{
    std::cout << "[NS MMS] ref=" << refinement
              << " h=" << std::scientific << std::setprecision(2) << h
              << " ux_L2=" << std::setprecision(4) << ux_L2
              << " ux_H1=" << ux_H1
              << " uy_L2=" << uy_L2
              << " p_L2=" << p_L2
              << " divU=" << div_U_L2
              << std::defaultfloat << "\n";
}

// ============================================================================
// Print for convergence table (tab-separated)
// ============================================================================
void NSMMSError::print_for_convergence(double h) const
{
    std::cout << std::scientific << std::setprecision(6)
              << h << "\t"
              << ux_L2 << "\t" << ux_H1 << "\t"
              << uy_L2 << "\t" << uy_H1 << "\t"
              << p_L2 << "\n";
}

// ============================================================================
// Get exact velocity at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_mms_exact_velocity(
    const dealii::Point<dim>& p,
    double time,
    double L_y)
{
    const double x = p[0];
    const double y = p[1];

    dealii::Tensor<1, dim> U;

    // ux = t · (π/L_y) · sin²(πx) · sin(2πy/L_y)
    const double sin_px = std::sin(M_PI * x);
    U[0] = time * (M_PI / L_y) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y);

    // uy = -t · π · sin(2πx) · sin²(πy/L_y)
    const double sin_py = std::sin(M_PI * y / L_y);
    U[1] = -time * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;

    return U;
}

// ============================================================================
// Get exact pressure at a point
// ============================================================================
template <int dim>
double ns_mms_exact_pressure(
    const dealii::Point<dim>& p,
    double time,
    double L_y)
{
    const double x = p[0];
    const double y = p[1];
    return time * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// Compute MMS source term
//
// For the exact solution:
//   ux = t · (π/L_y) · sin²(πx) · sin(2πy/L_y)
//   uy = -t · π · sin(2πx) · sin²(πy/L_y)
//   p  = t · cos(πx) · cos(πy/L_y)
//
// We need f such that:
//   ∂U/∂t + (U·∇)U - ν∇·T(U) + ∇p = f
//
// where T(U) = ∇U + (∇U)^T is the symmetric gradient.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    // Precompute trig functions
    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);

    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;

    // Exact solution values
    const double ux = t * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy = -t * pi * sin_2px * sin_py * sin_py;

    // ========================================================================
    // Term 1: ∂U/∂t
    // ========================================================================
    const double dux_dt = (pi / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -pi * sin_2px * sin_py * sin_py;

    // ========================================================================
    // Term 2: (U·∇)U = ux·∂U/∂x + uy·∂U/∂y
    // ========================================================================
    // Gradients of ux
    const double dux_dx = t * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy = t * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;

    // Gradients of uy
    const double duy_dx = -t * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy = -t * (pi2 / L_y) * sin_2px * sin_2py;

    const double convect_x = ux * dux_dx + uy * dux_dy;
    const double convect_y = ux * duy_dx + uy * duy_dy;

    // ========================================================================
    // Term 3: -ν∇·T(U) where T = ∇U + (∇U)^T
    //
    // ∇·T(U) has components:
    //   (∇·T)_x = 2·∂²ux/∂x² + ∂²ux/∂y² + ∂²uy/∂x∂y
    //   (∇·T)_y = ∂²ux/∂x∂y + ∂²uy/∂x² + 2·∂²uy/∂y²
    // ========================================================================
    // Second derivatives of ux
    const double d2ux_dx2 = t * (2.0 * pi2 * pi / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t * (4.0 * pi2 * pi / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double d2ux_dxdy = t * (2.0 * pi2 * pi / (L_y * L_y)) * sin_2px * cos_2py;

    // Second derivatives of uy
    const double d2uy_dx2 = t * 4.0 * pi2 * pi * sin_2px * sin_py * sin_py;
    const double d2uy_dy2 = -t * (pi2 * pi / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);
    const double d2uy_dxdy = -t * (2.0 * pi2 * pi / L_y) * cos_2px * sin_2py;

    // Actually for symmetric gradient formulation:
    // -ν∇·T(U) = -ν(2∇·D(U)) where D = (∇U + ∇U^T)/2
    // This gives: -ν(∇²U + ∇(∇·U))
    // Since ∇·U = 0 for our exact solution, this simplifies to -ν∇²U

    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    const double viscous_x = -nu * laplacian_ux;
    const double viscous_y = -nu * laplacian_uy;

    // ========================================================================
    // Term 4: ∇p
    // ========================================================================
    const double dp_dx = -t * pi * sin_px * cos_py;
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // ========================================================================
    // Total source: f = ∂U/∂t + (U·∇)U - ν∇²U + ∇p
    // ========================================================================
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + convect_x + viscous_x + dp_dx;
    f[1] = duy_dt + convect_y + viscous_y + dp_dy;

    return f;
}

// ============================================================================
// Compute NS MMS errors
// ============================================================================
template <int dim>
NSMMSError compute_ns_mms_error(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::Vector<double>& p_solution,
    double time,
    double L_y)
{
    NSMMSError error;

    // Setup FE evaluation for velocity
    const auto& ux_fe = ux_dof_handler.get_fe();
    const auto& p_fe = p_dof_handler.get_fe();

    dealii::QGauss<dim> quadrature(ux_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> p_fe_values(p_fe, quadrature,
        dealii::update_values);

    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);
    std::vector<double> p_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);
    NSExactPressure<dim> exact_p(time, L_y);

    double ux_L2_sq = 0.0, ux_H1_sq = 0.0;
    double uy_L2_sq = 0.0, uy_H1_sq = 0.0;
    double p_L2_sq = 0.0;
    double div_U_L2_sq = 0.0;

    // Compute mean pressure for both numerical and exact (pressure is unique up to constant)
    double p_mean_num = 0.0, p_mean_exact = 0.0, domain_area = 0.0;

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    // First pass: compute mean pressures
    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        p_fe_values.reinit(p_cell);

        p_fe_values.get_function_values(p_solution, p_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            p_mean_num += p_values[q] * JxW;
            p_mean_exact += exact_p.value(x_q) * JxW;
            domain_area += JxW;
        }
    }

    p_mean_num /= domain_area;
    p_mean_exact /= domain_area;

    // Second pass: compute errors
    ux_cell = ux_dof_handler.begin_active();
    uy_cell = uy_dof_handler.begin_active();
    p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        ux_fe_values.get_function_values(ux_solution, ux_values);
        uy_fe_values.get_function_values(uy_solution, uy_values);
        p_fe_values.get_function_values(p_solution, p_values);
        ux_fe_values.get_function_gradients(ux_solution, ux_gradients);
        uy_fe_values.get_function_gradients(uy_solution, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Velocity errors
            const double ux_exact = exact_ux.value(x_q);
            const double uy_exact = exact_uy.value(x_q);
            const dealii::Tensor<1, dim> grad_ux_exact = exact_ux.gradient(x_q);
            const dealii::Tensor<1, dim> grad_uy_exact = exact_uy.gradient(x_q);

            const double ux_err = ux_values[q] - ux_exact;
            const double uy_err = uy_values[q] - uy_exact;
            const dealii::Tensor<1, dim> grad_ux_err = ux_gradients[q] - grad_ux_exact;
            const dealii::Tensor<1, dim> grad_uy_err = uy_gradients[q] - grad_uy_exact;

            ux_L2_sq += ux_err * ux_err * JxW;
            uy_L2_sq += uy_err * uy_err * JxW;
            ux_H1_sq += (grad_ux_err * grad_ux_err) * JxW;
            uy_H1_sq += (grad_uy_err * grad_uy_err) * JxW;

            // Pressure error (zero-mean adjusted)
            const double p_exact = exact_p.value(x_q);
            const double p_err = (p_values[q] - p_mean_num) - (p_exact - p_mean_exact);
            p_L2_sq += p_err * p_err * JxW;

            // Divergence (should be zero for exact, check numerical)
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];
            div_U_L2_sq += div_U * div_U * JxW;
        }
    }

    error.ux_L2 = std::sqrt(ux_L2_sq);
    error.ux_H1 = std::sqrt(ux_H1_sq);
    error.uy_L2 = std::sqrt(uy_L2_sq);
    error.uy_H1 = std::sqrt(uy_H1_sq);
    error.p_L2 = std::sqrt(p_L2_sq);
    error.div_U_L2 = std::sqrt(div_U_L2_sq);

    return error;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template dealii::Tensor<1, 2> ns_mms_exact_velocity<2>(
    const dealii::Point<2>&, double, double);

template double ns_mms_exact_pressure<2>(
    const dealii::Point<2>&, double, double);

template dealii::Tensor<1, 2> compute_ns_mms_source<2>(
    const dealii::Point<2>&, double, double, double);

template NSMMSError compute_ns_mms_error<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double,
    double);