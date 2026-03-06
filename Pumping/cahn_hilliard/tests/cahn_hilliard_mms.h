// ============================================================================
// cahn_hilliard/tests/cahn_hilliard_mms.h - MMS Exact Solutions and Sources
//
// STABILIZED CH (backward Euler):
//
//   (phi^k - phi^{k-1})/dt = gamma * Delta(mu^k)                         (1)
//   mu^k = (1/ε)[Psi'(phi^{k-1}) + S(phi^k - phi^{k-1})] - ε*Delta(phi^k) (2)
//
//   Standard formulation: μ = -εΔθ + (1/ε)F'(θ), S = max|F''| = 1/2
//
// STANDALONE MMS (no convection, u = 0):
//
// EXACT SOLUTIONS (Neumann-compatible on [0,1]^2):
//   phi*(x, y, t) = A * t * cos(pi*x) * cos(pi*y),   A = 0.1
//   mu*(x, y, t)  = B * t * cos(2*pi*x) * cos(2*pi*y), B = 1.0
//
// Both satisfy homogeneous Neumann BCs on [0,1]^2.
//
// MMS SOURCES (use DISCRETE phi_old to avoid 1/tau amplification):
//
//   f_phi(x) = (1/dt)(phi*_new - phi_old_disc) - gamma * Delta_mu*_new
//   f_mu(x)  = mu*_new - (S/ε)*(phi*_new - phi_old_disc)
//              + ε * Delta_phi*_new - (1/ε)*Psi'(phi_old_disc)
//
// EXPECTED CONVERGENCE (CG Q_l with l=2):
//   phi_L2: O(h^3) — rate ~= 3.0
//   phi_H1: O(h^2) — rate ~= 2.0
//   mu_L2:  O(h^3) — rate ~= 3.0
//   mu_H1:  O(h^2) — rate ~= 2.0
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
//            Zhang, He, Yang (2021), SIAM J. Sci. Comput.
// ============================================================================
#ifndef FHD_CAHN_HILLIARD_MMS_H
#define FHD_CAHN_HILLIARD_MMS_H

#include "physics/material_properties.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>
#include <cmath>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// MMS constants
// ============================================================================
namespace CHmms
{
    constexpr double A = 0.1;  // phi amplitude (small: inside well)
    constexpr double B = 1.0;  // mu amplitude
}

// ============================================================================
// Exact solution: 2-component function for FESystem(FE_Q, 2)
//
//   Component 0: phi*(x,y,t) = A * t * cos(pi*x) * cos(pi*y)
//   Component 1: mu*(x,y,t)  = B * t * cos(2*pi*x) * cos(2*pi*y)
// ============================================================================
template <int dim>
class CHExactSolution : public dealii::Function<dim>
{
public:
    CHExactSolution(double time = 1.0)
        : dealii::Function<dim>(2), time_(time) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        if (component == 0)
        {
            // phi*
            double val = CHmms::A * time_ * std::cos(M_PI * p[0]);
            if constexpr (dim >= 2) val *= std::cos(M_PI * p[1]);
            return val;
        }
        else
        {
            // mu*
            double val = CHmms::B * time_ * std::cos(2.0 * M_PI * p[0]);
            if constexpr (dim >= 2) val *= std::cos(2.0 * M_PI * p[1]);
            return val;
        }
    }

    virtual dealii::Tensor<1, dim> gradient(
        const dealii::Point<dim>& p,
        const unsigned int component = 0) const override
    {
        dealii::Tensor<1, dim> grad;
        if (component == 0)
        {
            // grad phi*
            const double cx = std::cos(M_PI * p[0]);
            const double sx = std::sin(M_PI * p[0]);
            const double cy = (dim >= 2) ? std::cos(M_PI * p[1]) : 1.0;
            const double sy = (dim >= 2) ? std::sin(M_PI * p[1]) : 0.0;

            grad[0] = -CHmms::A * time_ * M_PI * sx * cy;
            if constexpr (dim >= 2)
                grad[1] = -CHmms::A * time_ * M_PI * cx * sy;
        }
        else
        {
            // grad mu*
            const double cx = std::cos(2.0 * M_PI * p[0]);
            const double sx = std::sin(2.0 * M_PI * p[0]);
            const double cy = (dim >= 2) ? std::cos(2.0 * M_PI * p[1]) : 1.0;
            const double sy = (dim >= 2) ? std::sin(2.0 * M_PI * p[1]) : 0.0;

            grad[0] = -CHmms::B * time_ * 2.0 * M_PI * sx * cy;
            if constexpr (dim >= 2)
                grad[1] = -CHmms::B * time_ * 2.0 * M_PI * cx * sy;
        }
        return grad;
    }

    void set_time(double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
};

// ============================================================================
// MMS source computation (per quadrature point)
//
// Uses discrete phi_old to avoid 1/tau amplification.
//
// New formulation: μ = -ε·Δθ + (1/ε)[F'(θ_old) + S(θ − θ_old)]
//   with S = 1/2 (max|F''|), S_eff = S/ε = 1/(2ε)
//
// f_phi = (1/dt)(phi*_new - phi_old_disc) - gamma * Delta_mu*_new
// f_mu  = mu*_new - (S/ε)*(phi*_new - phi_old_disc) + ε*Delta_phi*_new
//         - (1/ε)*Psi'(phi_old_disc)
// ============================================================================
template <int dim>
std::pair<double, double> compute_ch_mms_source(
    const dealii::Point<dim>& p,
    double t_new,
    double dt,
    double phi_old_disc,
    double epsilon,
    double gamma)
{
    // Standard CH: μ = -εΔθ + (1/ε)F'(θ), stabilized with S=1/2
    const double S_stab     = 0.5;                    // max|F''(θ)|
    const double S_eff      = S_stab / epsilon;       // = 1/(2ε)
    const double grad_coeff = epsilon;                 // coeff of Δθ
    const double pot_coeff  = 1.0 / epsilon;           // coeff of F'(θ)

    // Exact values at t_new
    double cos_phi = std::cos(M_PI * p[0]);
    double cos_mu  = std::cos(2.0 * M_PI * p[0]);
    for (unsigned int d = 1; d < dim; ++d)
    {
        cos_phi *= std::cos(M_PI * p[d]);
        cos_mu  *= std::cos(2.0 * M_PI * p[d]);
    }

    const double phi_star_new = CHmms::A * t_new * cos_phi;
    const double mu_star_new  = CHmms::B * t_new * cos_mu;

    // Laplacians at t_new
    // Delta phi* = -A * t_new * dim * pi^2 * cos_phi
    // Delta mu*  = -B * t_new * dim * (2pi)^2 * cos_mu
    const double lap_phi = -CHmms::A * t_new * dim * M_PI * M_PI * cos_phi;
    const double lap_mu  = -CHmms::B * t_new * dim * 4.0 * M_PI * M_PI * cos_mu;

    // f_phi = (1/dt)(phi*_new - phi_old_disc) - gamma * Delta_mu*_new
    const double f_phi = (1.0 / dt) * (phi_star_new - phi_old_disc)
                         - gamma * lap_mu;

    // f_mu = mu*_new - S_eff*(phi*_new - phi_old_disc) + ε*Delta_phi*_new
    //        - (1/ε)*Psi'(phi_old_disc)
    const double psi_prime = double_well_derivative(phi_old_disc);
    const double f_mu = mu_star_new
                        - S_eff * (phi_star_new - phi_old_disc)
                        + grad_coeff * lap_phi
                        - pot_coeff * psi_prime;

    return {f_phi, f_mu};
}

// ============================================================================
// Error computation (parallel)
// ============================================================================
struct CHMMSErrors
{
    double phi_L2 = 0.0, phi_H1 = 0.0;
    double mu_L2 = 0.0, mu_H1 = 0.0;
};

template <int dim>
CHMMSErrors compute_ch_mms_errors(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& solution_relevant,
    double time,
    MPI_Comm mpi_comm)
{
    CHMMSErrors errors;

    const auto& fe = dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    const dealii::FEValuesExtractors::Scalar phi_extract(0);
    const dealii::FEValuesExtractors::Scalar mu_extract(1);

    std::vector<double> phi_vals(n_q), mu_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_phi_vals(n_q), grad_mu_vals(n_q);

    CHExactSolution<dim> exact(time);

    double local_phi_L2_sq = 0.0, local_phi_H1_sq = 0.0;
    double local_mu_L2_sq  = 0.0, local_mu_H1_sq  = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values[phi_extract].get_function_values(solution_relevant, phi_vals);
        fe_values[phi_extract].get_function_gradients(solution_relevant, grad_phi_vals);
        fe_values[mu_extract].get_function_values(solution_relevant, mu_vals);
        fe_values[mu_extract].get_function_gradients(solution_relevant, grad_mu_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // phi errors
            const double phi_ex = exact.value(x_q, 0);
            const double phi_err = phi_vals[q] - phi_ex;
            local_phi_L2_sq += phi_err * phi_err * JxW;

            const auto grad_phi_ex = exact.gradient(x_q, 0);
            const auto grad_phi_err = grad_phi_vals[q] - grad_phi_ex;
            local_phi_H1_sq += (grad_phi_err * grad_phi_err) * JxW;

            // mu errors
            const double mu_ex = exact.value(x_q, 1);
            const double mu_err = mu_vals[q] - mu_ex;
            local_mu_L2_sq += mu_err * mu_err * JxW;

            const auto grad_mu_ex = exact.gradient(x_q, 1);
            const auto grad_mu_err = grad_mu_vals[q] - grad_mu_ex;
            local_mu_H1_sq += (grad_mu_err * grad_mu_err) * JxW;
        }
    }

    double global_phi_L2_sq = 0.0, global_phi_H1_sq = 0.0;
    double global_mu_L2_sq = 0.0, global_mu_H1_sq = 0.0;

    MPI_Allreduce(&local_phi_L2_sq, &global_phi_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_phi_H1_sq, &global_phi_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_mu_L2_sq,  &global_mu_L2_sq,  1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_mu_H1_sq,  &global_mu_H1_sq,  1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    errors.phi_L2 = std::sqrt(global_phi_L2_sq);
    errors.phi_H1 = std::sqrt(global_phi_H1_sq);
    errors.mu_L2  = std::sqrt(global_mu_L2_sq);
    errors.mu_H1  = std::sqrt(global_mu_H1_sq);

    return errors;
}

#endif // FHD_CAHN_HILLIARD_MMS_H
