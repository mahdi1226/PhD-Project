// ============================================================================
// magnetization/tests/magnetization_mms.h - MMS for Magnetization
//
// PAPER EQUATION 42c (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (m^k/τ, z) + σ·a_h^m(m^k, z) + B_h^m(u^{k-1}; m^k, z) + (1/𝒯)(m^k, z)
//     = (1/𝒯)(κ₀·h^k, z) + (m^{k-1}/τ, z) + f_mms
//
// STANDALONE MMS (U = 0, σ = 0, h_a = 0, ∇φ = 0):
//   Reduces to: (1/τ + 1/𝒯)(m, z) = (1/τ)(m^{n-1}, z) + (κ₀/𝒯)(h, z) + (f_mms, z)
//   With h = 0: (1/τ + 1/𝒯)(m, z) = (1/τ)(m^{n-1}, z) + (f_mms, z)
//
// EXACT SOLUTION (designed with M*·n = 0 on ∂[0,1]²):
//   Mx*(x,y,t) = t · sin(πx) · sin(πy)
//   My*(x,y,t) = t · cos(πx) · sin(πy)
//
// MMS SOURCE (for backward Euler with mass + relaxation only):
//   f_Mx = (Mx*(t_new) - Mx*(t_old))/τ + (1/𝒯)·Mx*(t_new)
//   f_My = (My*(t_new) - My*(t_old))/τ + (1/𝒯)·My*(t_new)
//
// EXPECTED CONVERGENCE (DG Q_ℓ, spatial, τ = h²):
//   L2: O(h^{ℓ+1})    (ℓ=2 → rate ≈ 3.0)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================
#ifndef FHD_MAGNETIZATION_MMS_H
#define FHD_MAGNETIZATION_MMS_H

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
// Exact solution M* at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> magnetization_exact(
    const dealii::Point<dim>& p, double time)
{
    dealii::Tensor<1, dim> M;
    M[0] = time * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
    M[1] = time * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]);
    return M;
}

// ============================================================================
// MMS source for standalone test (U = 0, σ = 0, h = 0)
//
// Source from substituting M* into the semi-discrete equation:
//   f = (M*(t_new) - M*(t_old))/τ + (1/𝒯)·M*(t_new)
//
// Callback signature matches MagnetizationSubsystem::MmsSourceFunction:
//   (point, t_new, t_old, tau_M, kappa_0, H, U, div_U) → Tensor<1,dim>
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_standalone(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double tau_M, double /*kappa_0*/,
    const dealii::Tensor<1, dim>& /*H*/,
    const dealii::Tensor<1, dim>& /*U*/,
    double /*div_U*/,
    const dealii::Tensor<1, dim>& /*M_old_disc*/)
{
    const double tau = t_new - t_old;

    const dealii::Tensor<1, dim> M_new = magnetization_exact<dim>(p, t_new);
    const dealii::Tensor<1, dim> M_old = magnetization_exact<dim>(p, t_old);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
    {
        // Time derivative: (M*(t_new) - M*(t_old))/τ
        f[d] = (M_new[d] - M_old[d]) / tau;

        // Relaxation: (1/𝒯)·M*(t_new) — this is the part of
        // (1/𝒯)(m - κ₀h) that comes from m, with h = 0
        f[d] += M_new[d] / tau_M;
    }

    return f;
}

// ============================================================================
// Error computation for magnetization (parallel)
// ============================================================================
struct MagnetizationMMSErrors
{
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;       // sqrt(Mx_L2² + My_L2²)
    double M_Linf = 0.0;
};

template <int dim>
MagnetizationMMSErrors compute_mag_mms_errors(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    double time,
    MPI_Comm mpi_comm)
{
    MagnetizationMMSErrors errors;

    const auto& fe = dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_quadrature_points |
        dealii::update_JxW_values);

    std::vector<double> mx_vals(n_q), my_vals(n_q);

    double local_Mx_L2_sq = 0.0, local_My_L2_sq = 0.0;
    double local_Linf = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(Mx_relevant, mx_vals);
        fe_values.get_function_values(My_relevant, my_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            const auto M_ex = magnetization_exact<dim>(x_q, time);

            const double err_x = mx_vals[q] - M_ex[0];
            const double err_y = my_vals[q] - M_ex[1];

            local_Mx_L2_sq += err_x * err_x * JxW;
            local_My_L2_sq += err_y * err_y * JxW;

            local_Linf = std::max(local_Linf,
                std::sqrt(err_x * err_x + err_y * err_y));
        }
    }

    double global_Mx_L2_sq = 0.0, global_My_L2_sq = 0.0;
    double global_Linf = 0.0;
    MPI_Allreduce(&local_Mx_L2_sq, &global_Mx_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_My_L2_sq, &global_My_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_Linf, &global_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    errors.Mx_L2 = std::sqrt(global_Mx_L2_sq);
    errors.My_L2 = std::sqrt(global_My_L2_sq);
    errors.M_L2 = std::sqrt(global_Mx_L2_sq + global_My_L2_sq);
    errors.M_Linf = global_Linf;

    return errors;
}

// ============================================================================
// L² projection of exact M onto DG space (cell-local)
// ============================================================================
template <int dim>
void project_magnetization_exact(
    const dealii::DoFHandler<dim>& dof_handler,
    dealii::TrilinosWrappers::MPI::Vector& Mx_owned,
    dealii::TrilinosWrappers::MPI::Vector& My_owned,
    double time)
{
    const auto& fe = dof_handler.get_fe();
    const unsigned int dpc = fe.n_dofs_per_cell();
    const dealii::QGauss<dim> quadrature(fe.degree + 1);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_quadrature_points |
        dealii::update_JxW_values);

    dealii::FullMatrix<double> cell_mass(dpc, dpc);
    dealii::Vector<double> cell_rhs_x(dpc), cell_rhs_y(dpc);
    dealii::Vector<double> cell_sol_x(dpc), cell_sol_y(dpc);
    std::vector<dealii::types::global_dof_index> local_dofs(dpc);

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_mass = 0.0;
        cell_rhs_x = 0.0;
        cell_rhs_y = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);
            const auto M_ex = magnetization_exact<dim>(x_q, time);

            for (unsigned int i = 0; i < dpc; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                cell_rhs_x(i) += M_ex[0] * phi_i * JxW;
                cell_rhs_y(i) += M_ex[1] * phi_i * JxW;

                for (unsigned int j = 0; j < dpc; ++j)
                    cell_mass(i, j) += phi_i * fe_values.shape_value(j, q) * JxW;
            }
        }

        // Solve local mass system
        cell_mass.gauss_jordan();
        cell_mass.vmult(cell_sol_x, cell_rhs_x);
        cell_mass.vmult(cell_sol_y, cell_rhs_y);

        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dpc; ++i)
        {
            Mx_owned[local_dofs[i]] = cell_sol_x(i);
            My_owned[local_dofs[i]] = cell_sol_y(i);
        }
    }

    Mx_owned.compress(dealii::VectorOperation::insert);
    My_owned.compress(dealii::VectorOperation::insert);
}

#endif // FHD_MAGNETIZATION_MMS_H
