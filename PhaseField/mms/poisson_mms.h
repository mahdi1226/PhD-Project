// ============================================================================
// mms/poisson_mms.h - Poisson Method of Manufactured Solutions (Header-Only)
//
// PAPER EQUATION 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// Two MMS modes:
//   1. STANDALONE (M=0): h_a_MMS = ∇φ_exact
//   2. COUPLED: Uses M_exact from magnetization_mms.h, adds f_MMS source
//
// EXACT SOLUTION (for both modes):
//   φ_exact = t · cos(πx) · cos(πy/L_y)
//
// This exact solution:
//   - Satisfies homogeneous Neumann BC (∂φ/∂n = 0 on all boundaries)
//   - Has zero mean (compatible with pinning any DoF)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_MMS_H
#define POISSON_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>
#include <algorithm>

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
    PoissonExactSolution(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;

        dealii::Tensor<1, dim> grad;
        grad[0] = -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        if constexpr (dim >= 2)
            grad[1] = -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        return grad;
    }

    void set_time(double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// MMS source for STANDALONE test (M=0): h_a_MMS = ∇φ_exact
// ============================================================================
template <int dim>
class PoissonMMSSource : public dealii::Function<dim>
{
public:
    PoissonMMSSource(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(dim), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component) const override
    {
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;

        if (component == 0)
            return -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        else if (component == 1 && dim >= 2)
            return -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        return 0.0;
    }

    dealii::Tensor<1, dim> tensor_value(const dealii::Point<dim>& p) const
    {
        dealii::Tensor<1, dim> h_a;
        h_a[0] = value(p, 0);
        if constexpr (dim >= 2)
            h_a[1] = value(p, 1);
        return h_a;
    }

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// MMS source for STANDALONE test (M=0, h_a=0)
//
// Strong form: -Δφ = ∇·(-M) = 0 (with M=0)
// We want φ_exact where -Δφ_exact ≠ 0
// So: f_MMS = -Δφ_exact
// ============================================================================
template <int dim>
double compute_poisson_mms_source_standalone(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    // φ_exact = t·cos(πx)·cos(πy/L_y)
    // -Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
    return time * M_PI * M_PI * (1.0 + 1.0/(L_y*L_y))
           * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// MMS source for COUPLED Poisson-Magnetization test
//
// Equation 42d weak form: (∇φ, ∇χ) = (h_a - M, ∇χ)
//
// For MMS, we add source term f_MMS to RHS:
//   (∇φ, ∇χ) = (h_a - M_numerical, ∇χ) + (f_MMS, χ)
//
// where f_MMS is computed so φ_exact satisfies the equation with M_exact:
//   (∇φ_exact, ∇χ) = (h_a - M_exact, ∇χ) + (f_MMS, χ)
//
// Converting to strong form (integration by parts):
//   -Δφ_exact = -∇·(h_a - M_exact) + f_MMS
//   f_MMS = -Δφ_exact + ∇·(h_a - M_exact)
//         = -Δφ_exact + ∇·h_a - ∇·M_exact
//         = -Δφ_exact - ∇·M_exact   (if h_a is constant, e.g., h_a = (0, 1))
//
// EXACT SOLUTIONS (from magnetization_mms.h):
//   Mx_exact = t·sin(πx)·sin(πy/L_y)
//   My_exact = t·cos(πx)·cos(πy/L_y)
//   φ_exact  = t·cos(πx)·cos(πy/L_y)
//
// DERIVATION:
//   -Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
//
//   ∇·M_exact = ∂Mx/∂x + ∂My/∂y
//             = t·π·cos(πx)·sin(πy/L_y) - t·(π/L_y)·cos(πx)·sin(πy/L_y)
//             = t·π·(1 - 1/L_y)·cos(πx)·sin(πy/L_y)
//
//   f_MMS = -Δφ_exact - ∇·M_exact
// ============================================================================
template <int dim>
double compute_poisson_mms_source_coupled(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    // -Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
    const double neg_laplacian_phi = time * M_PI * M_PI * (1.0 + 1.0/(L_y*L_y))
                                     * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);

    // ∇·M_exact = ∂Mx/∂x + ∂My/∂y
    // Mx = t·sin(πx)·sin(πy/L_y) → ∂Mx/∂x = t·π·cos(πx)·sin(πy/L_y)
    // My = t·cos(πx)·cos(πy/L_y) → ∂My/∂y = -t·(π/L_y)·cos(πx)·sin(πy/L_y)
    const double div_M = time * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y)
                       - time * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
    // Simplified: t·π·(1 - 1/L_y)·cos(πx)·sin(πy/L_y)

    // f_MMS = -Δφ_exact - ∇·M_exact
    return neg_laplacian_phi - div_M;
}

// ============================================================================
// Poisson MMS Source class for coupled test (Function interface)
// ============================================================================
template <int dim>
class PoissonMMSSourceCoupled : public dealii::Function<dim>
{
public:
    PoissonMMSSourceCoupled(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int = 0) const override
    {
        return compute_poisson_mms_source_coupled<dim>(p, time_, L_y_);
    }

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Poisson MMS Error Results
// ============================================================================
struct PoissonMMSError
{
    double L2_error = 0.0;     // L² error (after mean shift)
    double H1_error = 0.0;     // H¹ seminorm (gradient error)
    double Linf_error = 0.0;   // L∞ error

    void print(unsigned int refinement, double h) const
    {
        std::cout << "[Poisson MMS] ref=" << refinement
                  << " h=" << std::scientific << std::setprecision(2) << h
                  << " L2=" << std::setprecision(4) << L2_error
                  << " H1=" << H1_error
                  << " Linf=" << Linf_error
                  << std::defaultfloat << "\n";
    }

    void print_for_convergence() const
    {
        std::cout << std::scientific << std::setprecision(4)
                  << L2_error << "  " << H1_error << "\n";
    }
};

// ============================================================================
// Compute Poisson MMS errors
//
// For pure Neumann, the solution is unique up to a constant. We:
//   1. Compute H1 seminorm (gradient error) - unaffected by constant
//   2. For L2, compute the mean difference and subtract it
// ============================================================================
template <int dim>
PoissonMMSError compute_poisson_mms_error(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    double time,
    double L_y = 1.0)
{
    PoissonMMSError errors;

    const auto& fe = phi_dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    PoissonExactSolution<dim> exact_solution(time, L_y);

    // First pass: compute mean difference, H1 seminorm, and Linf
    double total_volume = 0.0;
    double mean_diff = 0.0;
    double H1_sq = 0.0;
    double Linf = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(phi_solution, phi_values);
        fe_values.get_function_gradients(phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // Accumulate for mean difference
            const double phi_exact = exact_solution.value(x_q);
            const double diff = phi_values[q] - phi_exact;
            mean_diff += diff * JxW;
            total_volume += JxW;
            Linf = std::max(Linf, std::abs(diff));

            // H1 seminorm (gradient error) - unaffected by constant shift
            const auto grad_exact = exact_solution.gradient(x_q);
            const auto grad_error = phi_gradients[q] - grad_exact;
            H1_sq += (grad_error * grad_error) * JxW;
        }
    }

    // Compute mean shift
    const double c_shift = mean_diff / total_volume;

    // Second pass: compute L2 error with mean-shifted solution
    double L2_sq = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(phi_solution, phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_exact = exact_solution.value(x_q);
            // Shift numerical solution by mean difference
            const double value_error = (phi_values[q] - c_shift) - phi_exact;
            L2_sq += value_error * value_error * JxW;
        }
    }

    errors.L2_error = std::sqrt(L2_sq);
    errors.H1_error = std::sqrt(H1_sq);
    errors.Linf_error = Linf;  // Note: not mean-shifted

    return errors;
}

// ============================================================================
// Assemble Poisson system with MMS source (STANDALONE, M=0)
//
// Assembles: (∇φ, ∇χ) = (h_a_MMS, ∇χ)
// where h_a_MMS = ∇φ_exact
// ============================================================================
template <int dim>
void assemble_poisson_mms_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints,
    double L_y = 1.0)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int quad_degree = fe.degree + 2;

    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    PoissonMMSSource<dim> mms_source(current_time, L_y);

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // MMS source: h_a = ∇φ_exact
            const dealii::Tensor<1, dim> h_a = mms_source.tensor_value(x_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ)
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }
}

// ============================================================================
// Apply MMS initial condition (project exact solution)
// ============================================================================
template <int dim>
void apply_poisson_mms_initial_condition(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::Vector<double>& phi_solution,
    double time,
    double L_y = 1.0)
{
    PoissonExactSolution<dim> exact_solution(time, L_y);
    dealii::VectorTools::interpolate(phi_dof_handler, exact_solution, phi_solution);
}

// ============================================================================
// Setup Poisson MMS constraints
//
// For MMS verification, we pin DoF 0 to the exact solution value
// at that point, ensuring the numerical solution can match exactly.
// ============================================================================
template <int dim>
void setup_poisson_mms_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints,
    double time,
    double L_y = 1.0)
{
    phi_constraints.clear();

    // Hanging node constraints (for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Find the location of DoF 0 and pin to exact value
    if (phi_dof_handler.n_dofs() > 0)
    {
        // Get support point for DoF 0
        std::vector<dealii::Point<dim>> support_points(phi_dof_handler.n_dofs());
        dealii::DoFTools::map_dofs_to_support_points(
            dealii::MappingQ1<dim>(),
            phi_dof_handler,
            support_points);

        // Compute exact solution at DoF 0 location
        PoissonExactSolution<dim> exact_solution(time, L_y);
        const double phi_exact_at_0 = exact_solution.value(support_points[0]);

        // Pin DoF 0 to exact value
        if (!phi_constraints.is_constrained(0))
        {
            phi_constraints.add_line(0);
            phi_constraints.set_inhomogeneity(0, phi_exact_at_0);
        }
    }

    phi_constraints.close();
}

#endif // POISSON_MMS_H