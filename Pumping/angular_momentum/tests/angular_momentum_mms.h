// ============================================================================
// angular_momentum/tests/angular_momentum_mms.h - MMS for Angular Momentum
//
// PAPER EQUATION 42f (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   j(w^k/τ, z) + c₁(∇w^k, ∇z) + 4ν_r(w^k, z)
//     = j(w^{k-1}/τ, z) + (f_mms, z)
//
// STANDALONE MMS (U = 0, M = 0, h = 0):
//   LHS: (j/τ + 4ν_r)(w, z) + c₁(∇w, ∇z)
//   RHS: (j/τ)(w_old, z) + (f_mms, z)
//
// EXACT SOLUTION (homogeneous Dirichlet on [0,1]²):
//   w*(x,y,t) = t · sin(πx) · sin(πy)
//
// MMS SOURCE (backward Euler, standalone):
//   f = j(w*(t_new) - w*(t_old))/τ - c₁·Δw*(t_new) + 4ν_r·w*(t_new)
//
// Since Δw* = -2π²t·sin(πx)sin(πy):
//   f = j(t_new - t_old)/τ · sin(πx)sin(πy)
//     + (2π²c₁ + 4ν_r) · t_new · sin(πx)sin(πy)
//
// EXPECTED CONVERGENCE (CG Q_ℓ, spatial, τ = h²):
//   L2: O(h^{ℓ+1})    (ℓ=2 → rate ≈ 3.0)
//   H1: O(h^ℓ)        (ℓ=2 → rate ≈ 2.0)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================
#ifndef FHD_ANGULAR_MOMENTUM_MMS_H
#define FHD_ANGULAR_MOMENTUM_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact solution w* at a point
// ============================================================================
template <int dim>
double angular_momentum_exact(
    const dealii::Point<dim>& p, double time)
{
    return time * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
}

// ============================================================================
// Gradient of exact solution (for H1 error)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> angular_momentum_exact_gradient(
    const dealii::Point<dim>& p, double time)
{
    dealii::Tensor<1, dim> g;
    g[0] = time * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]);
    g[1] = time * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]);
    return g;
}

// ============================================================================
// MMS source for standalone test (U = 0, M = 0, h = 0)
//
// Source from substituting w* into the semi-discrete equation:
//   f = j·(w*(t_new) - w*(t_old))/τ - c₁·Δw*(t_new) + 4ν_r·w*(t_new)
//
// With Δw* = -2π²·t·sin(πx)·sin(πy), so -c₁·Δw* = +2π²c₁·t·sin(πx)sin(πy)
//
// Callback signature: (point, t_new, t_old, j_micro, c1, nu_r) → double
// ============================================================================
template <int dim>
double compute_angular_mms_source_standalone(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double j_micro, double c1, double nu_r,
    double /*w_old_disc*/,
    const dealii::Tensor<1, dim>& /*U_old_disc*/,
    double /*div_U_old_disc*/,
    bool /*include_convection*/)
{
    const double tau = t_new - t_old;
    const double w_new = angular_momentum_exact<dim>(p, t_new);
    const double w_old = angular_momentum_exact<dim>(p, t_old);

    // Time derivative: j(w*(t_new) - w*(t_old))/τ
    double f = j_micro * (w_new - w_old) / tau;

    // Diffusion: -c₁ Δw*(t_new) = +2π²c₁ · t_new · sin(πx)sin(πy)
    f += 2.0 * M_PI * M_PI * c1 * w_new;

    // Reaction: 4ν_r · w*(t_new)
    f += 4.0 * nu_r * w_new;

    return f;
}

// ============================================================================
// Error computation for angular momentum (parallel)
// ============================================================================
struct AngularMomentumMMSErrors
{
    double w_L2 = 0.0;
    double w_H1 = 0.0;
    double w_Linf = 0.0;
};

template <int dim>
AngularMomentumMMSErrors compute_angular_mms_errors(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& w_relevant,
    double time,
    MPI_Comm mpi_comm)
{
    AngularMomentumMMSErrors errors;

    const auto& fe = dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> w_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_w_vals(n_q);

    double local_L2_sq = 0.0, local_H1_sq = 0.0;
    double local_Linf = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(w_relevant, w_vals);
        fe_values.get_function_gradients(w_relevant, grad_w_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            const double w_ex = angular_momentum_exact<dim>(x_q, time);
            const auto grad_w_ex = angular_momentum_exact_gradient<dim>(x_q, time);

            const double err = w_vals[q] - w_ex;
            const auto grad_err = grad_w_vals[q] - grad_w_ex;

            local_L2_sq += err * err * JxW;
            local_H1_sq += (grad_err * grad_err) * JxW;
            local_Linf = std::max(local_Linf, std::abs(err));
        }
    }

    double global_L2_sq = 0.0, global_H1_sq = 0.0;
    double global_Linf = 0.0;
    MPI_Allreduce(&local_L2_sq, &global_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_H1_sq, &global_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_Linf, &global_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    errors.w_L2 = std::sqrt(global_L2_sq);
    errors.w_H1 = std::sqrt(global_H1_sq);
    errors.w_Linf = global_Linf;

    return errors;
}

#endif // FHD_ANGULAR_MOMENTUM_MMS_H
