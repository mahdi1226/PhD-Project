// ============================================================================
// navier_stokes/tests/navier_stokes_mms.h - MMS for Navier-Stokes
//
// PAPER EQUATION 42e (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
// EXACT SOLUTION (divergence-free, homogeneous Dirichlet on [0,1]²):
//   ux*(x,y,t) = t · π · sin²(πx) · sin(2πy)
//   uy*(x,y,t) = -t · π · sin(2πx) · sin²(πy)
//   p*(x,y,t)  = t · cos(πx) · cos(πy)
//
// VERIFICATION:
//   ∇·U* = t·π·2π·sin(πx)cos(πx)·sin(2πy)
//         - t·π·2π·sin(2πx)·sin(πy)cos(πy)
//         = t·π²·sin(2πx)·sin(2πy) - t·π²·sin(2πx)·sin(2πy) = 0  ✓
//   U*(x,y,t)|_{∂Ω} = 0  ✓
//
// STANDALONE MMS (no coupling, no convection — steady Stokes):
//   -ν_eff Δu + ∇p = f_mms,   ∇·u = 0
//
// Source: f = -ν_eff Δu* + ∇p*
//   (For div-free u*: Δu = div(2D(u)) = div(∇u + ∇uᵀ) = Δu since div(∇uᵀ)=∇(div u)=0)
//
// EXPECTED CONVERGENCE (CG Q2 / DG P1):
//   U_L2:  O(h³) — rate ≈ 3.0
//   U_H1:  O(h²) — rate ≈ 2.0
//   p_L2:  O(h²) — rate ≈ 2.0
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================
#ifndef FHD_NS_MMS_H
#define FHD_NS_MMS_H

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
// Exact solutions
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_exact_velocity(
    const dealii::Point<dim>& p, double time)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> U;
    U[0] = time * M_PI * std::sin(M_PI * x) * std::sin(M_PI * x) * std::sin(2.0 * M_PI * y);
    U[1] = -time * M_PI * std::sin(2.0 * M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * y);
    return U;
}

template <int dim>
double ns_exact_pressure(
    const dealii::Point<dim>& p, double time)
{
    return time * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1]);
}

// ============================================================================
// Gradient of exact velocity (for H1 errors and convection)
//
// grad_ux[d] = ∂ux/∂x_d,  grad_uy[d] = ∂uy/∂x_d
// ============================================================================
template <int dim>
void ns_exact_velocity_gradient(
    const dealii::Point<dim>& p, double time,
    dealii::Tensor<1, dim>& grad_ux,
    dealii::Tensor<1, dim>& grad_uy)
{
    const double x = p[0], y = p[1];
    const double sx = std::sin(M_PI * x), cx = std::cos(M_PI * x);
    const double sy = std::sin(M_PI * y), cy = std::cos(M_PI * y);
    const double s2x = std::sin(2.0 * M_PI * x), c2x = std::cos(2.0 * M_PI * x);
    const double s2y = std::sin(2.0 * M_PI * y), c2y = std::cos(2.0 * M_PI * y);

    // ux = t·π·sin²(πx)·sin(2πy)
    grad_ux[0] = time * M_PI * 2.0 * M_PI * sx * cx * s2y;    // = t·π·π·sin(2πx)·sin(2πy)
    grad_ux[1] = time * M_PI * sx * sx * 2.0 * M_PI * c2y;    // = t·2π²·sin²(πx)·cos(2πy)

    // uy = -t·π·sin(2πx)·sin²(πy)
    grad_uy[0] = -time * M_PI * 2.0 * M_PI * c2x * sy * sy;   // = -t·2π²·cos(2πx)·sin²(πy)
    grad_uy[1] = -time * M_PI * s2x * 2.0 * M_PI * sy * cy;   // = -t·π·π·sin(2πx)·sin(2πy)... wait
    // More carefully: ∂/∂y[-t·π·sin(2πx)·sin²(πy)] = -t·π·sin(2πx)·2πsin(πy)cos(πy)
    //                                                 = -t·π²·sin(2πx)·sin(2πy)
    grad_uy[1] = -time * M_PI * s2x * 2.0 * M_PI * sy * cy;
}

// ============================================================================
// Laplacian of exact velocity
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_exact_laplacian(
    const dealii::Point<dim>& p, double time)
{
    const double x = p[0], y = p[1];
    const double sx = std::sin(M_PI * x);
    const double sy = std::sin(M_PI * y);
    const double s2x = std::sin(2.0 * M_PI * x), c2x = std::cos(2.0 * M_PI * x);
    const double s2y = std::sin(2.0 * M_PI * y), c2y = std::cos(2.0 * M_PI * y);

    dealii::Tensor<1, dim> lap;

    // ux = t·π·sin²(πx)·sin(2πy)
    // Δux = ∂²ux/∂x² + ∂²ux/∂y²
    //     = t·2π³·cos(2πx)·sin(2πy) - t·4π³·sin²(πx)·sin(2πy)
    lap[0] = time * 2.0 * M_PI * M_PI * M_PI * c2x * s2y
             - time * 4.0 * M_PI * M_PI * M_PI * sx * sx * s2y;

    // uy = -t·π·sin(2πx)·sin²(πy)
    // Δuy = t·4π³·sin(2πx)·sin²(πy) - t·2π³·sin(2πx)·cos(2πy)
    lap[1] = time * 4.0 * M_PI * M_PI * M_PI * s2x * sy * sy
             - time * 2.0 * M_PI * M_PI * M_PI * s2x * c2y;

    return lap;
}

// ============================================================================
// Gradient of exact pressure
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_exact_pressure_gradient(
    const dealii::Point<dim>& p, double time)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> grad_p;
    grad_p[0] = -time * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    grad_p[1] = -time * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    return grad_p;
}

// ============================================================================
// MMS source for steady Stokes (no time derivative, no convection)
//
// Weak form: ν_eff(D(u), D(v)) - (p, ∇·v) = (f, v)
// Strong form (div-free u): -div(ν_eff D(u)) + ∇p = f
//   where div(D(u)) = (1/2)Δu for div-free u
//   so: f = -(ν_eff/2) Δu* + ∇p*
//
// Callback signature: (point, t_new, t_old, nu_eff) → Tensor<1,dim>
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_stokes(
    const dealii::Point<dim>& p,
    double t_new, double /*t_old*/, double nu_eff,
    const dealii::Tensor<1, dim>& /*U_old_disc*/,
    double /*div_U_old_disc*/ = 0.0,
    bool /*include_convection*/ = false)
{
    const auto lap = ns_exact_laplacian<dim>(p, t_new);
    const auto grad_p = ns_exact_pressure_gradient<dim>(p, t_new);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
        f[d] = -(nu_eff / 2.0) * lap[d] + grad_p[d];

    return f;
}

// ============================================================================
// MMS source for unsteady Stokes (backward Euler, no convection)
//
// f = (u*(t_new) - u*(t_old))/τ - (ν_eff/2) Δu*(t_new) + ∇p*(t_new)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_unsteady(
    const dealii::Point<dim>& p,
    double t_new, double t_old, double nu_eff,
    const dealii::Tensor<1, dim>& /*U_old_disc*/)
{
    const double tau = t_new - t_old;
    const auto U_new = ns_exact_velocity<dim>(p, t_new);
    const auto U_old = ns_exact_velocity<dim>(p, t_old);
    const auto lap = ns_exact_laplacian<dim>(p, t_new);
    const auto grad_p = ns_exact_pressure_gradient<dim>(p, t_new);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
        f[d] = (U_new[d] - U_old[d]) / tau - (nu_eff / 2.0) * lap[d] + grad_p[d];

    return f;
}

// ============================================================================
// Error computation for NS (parallel)
// ============================================================================
struct NSMMSErrors
{
    double ux_L2 = 0.0, uy_L2 = 0.0;
    double ux_H1 = 0.0, uy_H1 = 0.0;
    double U_L2 = 0.0;     // sqrt(ux_L2² + uy_L2²)
    double U_H1 = 0.0;     // sqrt(ux_H1² + uy_H1²)
    double U_Linf = 0.0;
    double p_L2 = 0.0;
    double p_Linf = 0.0;
};

template <int dim>
NSMMSErrors compute_ns_mms_errors(
    const dealii::DoFHandler<dim>& vel_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& p_relevant,
    double time,
    MPI_Comm mpi_comm)
{
    NSMMSErrors errors;

    // Velocity errors
    const auto& vel_fe = vel_dof_handler.get_fe();
    const unsigned int vel_quad_degree = vel_fe.degree + 2;
    dealii::QGauss<dim> vel_quad(vel_quad_degree);
    const unsigned int n_q = vel_quad.size();

    dealii::FEValues<dim> fe_values_vel(vel_fe, vel_quad,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> ux_vals(n_q), uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q), grad_uy_vals(n_q);

    double loc_ux_L2_sq = 0.0, loc_uy_L2_sq = 0.0;
    double loc_ux_H1_sq = 0.0, loc_uy_H1_sq = 0.0;
    double loc_U_Linf = 0.0;

    for (const auto& cell : vel_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values_vel.reinit(cell);
        fe_values_vel.get_function_values(ux_relevant, ux_vals);
        fe_values_vel.get_function_values(uy_relevant, uy_vals);
        fe_values_vel.get_function_gradients(ux_relevant, grad_ux_vals);
        fe_values_vel.get_function_gradients(uy_relevant, grad_uy_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values_vel.quadrature_point(q);
            const double JxW = fe_values_vel.JxW(q);

            const auto U_ex = ns_exact_velocity<dim>(x_q, time);
            dealii::Tensor<1, dim> grad_ux_ex, grad_uy_ex;
            ns_exact_velocity_gradient<dim>(x_q, time, grad_ux_ex, grad_uy_ex);

            const double ex = ux_vals[q] - U_ex[0];
            const double ey = uy_vals[q] - U_ex[1];

            loc_ux_L2_sq += ex * ex * JxW;
            loc_uy_L2_sq += ey * ey * JxW;

            const auto grad_ex = grad_ux_vals[q] - grad_ux_ex;
            const auto grad_ey = grad_uy_vals[q] - grad_uy_ex;
            loc_ux_H1_sq += (grad_ex * grad_ex) * JxW;
            loc_uy_H1_sq += (grad_ey * grad_ey) * JxW;

            loc_U_Linf = std::max(loc_U_Linf,
                std::sqrt(ex * ex + ey * ey));
        }
    }

    // Pressure errors
    const auto& p_fe = p_dof_handler.get_fe();
    const unsigned int p_quad_degree = p_fe.degree + 2;
    dealii::QGauss<dim> p_quad(p_quad_degree);
    const unsigned int n_qp = p_quad.size();

    dealii::FEValues<dim> fe_values_p(p_fe, p_quad,
        dealii::update_values | dealii::update_quadrature_points |
        dealii::update_JxW_values);

    std::vector<double> p_vals(n_qp);
    double loc_p_L2_sq = 0.0, loc_p_Linf = 0.0;

    // First pass: compute mean of numerical and exact pressure (for DG p)
    double loc_p_num_int = 0.0, loc_p_ex_int = 0.0, loc_vol = 0.0;
    for (const auto& cell : p_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values_p.reinit(cell);
        fe_values_p.get_function_values(p_relevant, p_vals);

        for (unsigned int q = 0; q < n_qp; ++q)
        {
            const auto& x_q = fe_values_p.quadrature_point(q);
            const double JxW = fe_values_p.JxW(q);
            loc_p_num_int += p_vals[q] * JxW;
            loc_p_ex_int += ns_exact_pressure<dim>(x_q, time) * JxW;
            loc_vol += JxW;
        }
    }

    double glob_p_num_int = dealii::Utilities::MPI::sum(loc_p_num_int, mpi_comm);
    double glob_p_ex_int  = dealii::Utilities::MPI::sum(loc_p_ex_int, mpi_comm);
    double glob_vol       = dealii::Utilities::MPI::sum(loc_vol, mpi_comm);
    const double p_shift = (glob_p_num_int - glob_p_ex_int) / glob_vol;

    // Second pass: compute pressure error with mean shift
    for (const auto& cell : p_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values_p.reinit(cell);
        fe_values_p.get_function_values(p_relevant, p_vals);

        for (unsigned int q = 0; q < n_qp; ++q)
        {
            const auto& x_q = fe_values_p.quadrature_point(q);
            const double JxW = fe_values_p.JxW(q);
            const double ep = (p_vals[q] - p_shift) - ns_exact_pressure<dim>(x_q, time);
            loc_p_L2_sq += ep * ep * JxW;
            loc_p_Linf = std::max(loc_p_Linf, std::abs(ep));
        }
    }

    // Global reductions
    double glob_ux_L2_sq = dealii::Utilities::MPI::sum(loc_ux_L2_sq, mpi_comm);
    double glob_uy_L2_sq = dealii::Utilities::MPI::sum(loc_uy_L2_sq, mpi_comm);
    double glob_ux_H1_sq = dealii::Utilities::MPI::sum(loc_ux_H1_sq, mpi_comm);
    double glob_uy_H1_sq = dealii::Utilities::MPI::sum(loc_uy_H1_sq, mpi_comm);
    double glob_U_Linf   = dealii::Utilities::MPI::max(loc_U_Linf, mpi_comm);
    double glob_p_L2_sq  = dealii::Utilities::MPI::sum(loc_p_L2_sq, mpi_comm);
    double glob_p_Linf   = dealii::Utilities::MPI::max(loc_p_Linf, mpi_comm);

    errors.ux_L2 = std::sqrt(glob_ux_L2_sq);
    errors.uy_L2 = std::sqrt(glob_uy_L2_sq);
    errors.ux_H1 = std::sqrt(glob_ux_H1_sq);
    errors.uy_H1 = std::sqrt(glob_uy_H1_sq);
    errors.U_L2 = std::sqrt(glob_ux_L2_sq + glob_uy_L2_sq);
    errors.U_H1 = std::sqrt(glob_ux_H1_sq + glob_uy_H1_sq);
    errors.U_Linf = glob_U_Linf;
    errors.p_L2 = std::sqrt(glob_p_L2_sq);
    errors.p_Linf = glob_p_Linf;

    return errors;
}

#endif // FHD_NS_MMS_H
