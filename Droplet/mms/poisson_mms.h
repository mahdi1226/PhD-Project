// ============================================================================
// mms/poisson_mms.h - Poisson Method of Manufactured Solutions
//
// Provides exact solutions and source terms for Poisson verification.
//
// SIMPLIFIED MODEL (μ = 1):
//   (∇φ, ∇χ) = (h_a, ∇χ)
//   φ_exact = t · cos(πx) · cos(πy/L_y)
//   h_a_MMS = ∇φ_exact
//
// QUASI-EQUILIBRIUM MODEL (μ(θ) = 1 + χ(θ)):
//   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)
//   φ_exact = t · cos(πx) · cos(πy/L_y)
//   θ_exact = same as CH MMS (or prescribed)
//   h_a_MMS = μ(θ)∇φ_exact
//
// The exact solution satisfies homogeneous Neumann BC: ∂φ/∂n = 0
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_MMS_H
#define POISSON_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/parameters.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact solution: φ_exact = t · cos(πx) · cos(πy/L_y)
// ============================================================================
template <int dim>
class PoissonExactSolution : public dealii::Function<dim>
{
public:
    PoissonExactSolution(double time, double L_y = 0.6)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];

        dealii::Tensor<1, dim> grad;
        // ∂φ/∂x = -t·π·sin(πx)·cos(πy/L_y)
        grad[0] = -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        // ∂φ/∂y = -t·(π/L_y)·cos(πx)·sin(πy/L_y)
        grad[1] = -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);

        return grad;
    }

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// MMS source: h_a_MMS = ∇φ_exact (for simplified model, μ = 1)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> poisson_mms_source_simplified(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 0.6)
{
    const double x = p[0];
    const double y = p[1];

    dealii::Tensor<1, dim> h_a;
    // h_a = ∇φ_exact
    h_a[0] = -time * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
    h_a[1] = -time * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);

    return h_a;
}

// ============================================================================
// MMS source: h_a_MMS = μ(θ)∇φ_exact (for quasi-equilibrium model)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> poisson_mms_source_quasi_equilibrium(
    const dealii::Point<dim>& p,
    double theta_value,
    double time,
    double epsilon,
    double chi_0,
    double L_y = 0.6)
{
    // Compute μ(θ) = 1 + χ(θ) = 1 + χ₀·H(θ/ε)
    const double sigmoid_val = 1.0 / (1.0 + std::exp(-theta_value / epsilon));
    const double chi_theta = chi_0 * sigmoid_val;
    const double mu_theta = 1.0 + chi_theta;

    // h_a = μ(θ)∇φ_exact
    dealii::Tensor<1, dim> grad_phi = poisson_mms_source_simplified<dim>(p, time, L_y);

    return mu_theta * grad_phi;
}

// ============================================================================
// Poisson MMS Error Results
// ============================================================================
struct PoissonMMSError
{
    double L2_error = 0.0;
    double H1_error = 0.0;      // |∇(φ - φ_exact)|_{L²}
    double Linf_error = 0.0;

    void print(unsigned int refinement, double h) const;
};

// ============================================================================
// Compute Poisson MMS errors
// ============================================================================
template <int dim>
PoissonMMSError compute_poisson_mms_error(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    double time,
    double L_y = 0.6);

// ============================================================================
// Assemble Poisson system with MMS source (SIMPLIFIED, μ = 1)
// ============================================================================
template <int dim>
void assemble_poisson_system_mms_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    double current_time,
    double L_y,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

// ============================================================================
// Assemble Poisson system with MMS source (QUASI-EQUILIBRIUM)
// ============================================================================
template <int dim>
void assemble_poisson_system_mms_quasi_equilibrium(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    double L_y,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_MMS_H