// ============================================================================
// mms/ch_mms.h - MMS (Method of Manufactured Solutions) for CH
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (DISCRETE scheme), p.505
//
// CRITICAL: Source terms derived from the ACTUAL WEAK FORM in ch_assembler.cc!
//
// Exact solutions:
//   θ_exact = t⁴ cos(πx) cos(πy)
//   ψ_exact = t⁴ sin(πx) sin(πy)
// ============================================================================
#ifndef CH_MMS_H
#define CH_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact phase field: θ = t⁴ cos(πx) cos(πy)
// ============================================================================
template <int dim>
class CHExactTheta : public dealii::Function<dim>
{
public:
    CHExactTheta() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;
        return t4 * std::cos(M_PI * x) * std::cos(M_PI * y);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                     const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;

        dealii::Tensor<1, dim> grad;
        grad[0] = -t4 * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
        grad[1] = -t4 * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
        return grad;
    }
};

// ============================================================================
// Exact chemical potential: ψ = t⁴ sin(πx) sin(πy)
// ============================================================================
template <int dim>
class CHExactPsi : public dealii::Function<dim>
{
public:
    CHExactPsi() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;
        return t4 * std::sin(M_PI * x) * std::sin(M_PI * y);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                     const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;

        dealii::Tensor<1, dim> grad;
        grad[0] = t4 * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
        grad[1] = t4 * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
        return grad;
    }
};

// ============================================================================
// Source term for θ equation
//
// The ASSEMBLER computes (weak form):
//   (1/dt)(θ, Λ) - γ(∇ψ, ∇Λ) = (1/dt)(θ_old, Λ) + (S_θ, Λ)
//
// Note: -γ(∇ψ, ∇Λ) in weak form = +γΔψ in strong form (IBP sign flip)
//
// Strong form for exact solution:
//   S_θ = (θ - θ_old)/dt + γΔψ
//
// With ψ = t⁴ sin(πx)sin(πy):
//   Δψ = -2π² t⁴ sin(πx)sin(πy)
//
// Therefore:
//   S_θ = (t⁴ - t_old⁴)/dt · cos·cos + γ·(-2π² t⁴)·sin·sin
//       = (t⁴ - t_old⁴)/dt · cos·cos - 2γπ² t⁴ sin·sin
// ============================================================================
template <int dim>
class CHSourceTheta : public dealii::Function<dim>
{
public:
    CHSourceTheta(double gamma, double dt)
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;
        const double x = p[0], y = p[1];

        const double t4 = t * t * t * t;
        const double t4_old = t_old * t_old * t_old * t_old;

        const double cos_px = std::cos(M_PI * x);
        const double cos_py = std::cos(M_PI * y);
        const double sin_px = std::sin(M_PI * x);
        const double sin_py = std::sin(M_PI * y);

        // Discrete time derivative: (θⁿ - θⁿ⁻¹)/dt
        const double dtheta_dt = (t4 - t4_old) / dt_ * cos_px * cos_py;

        // Δψ = -2π² t⁴ sin(πx) sin(πy)
        const double lap_psi = -2.0 * M_PI * M_PI * t4 * sin_px * sin_py;

        // S_θ = (θ - θ_old)/dt + γΔψ
        //     = dtheta_dt + gamma * lap_psi
        return dtheta_dt + gamma_ * lap_psi;
    }

private:
    double gamma_;
    double dt_;
};

// ============================================================================
// Source term for ψ equation
//
// The ASSEMBLER computes (weak form):
//   (ψ, Υ) + ε(∇θ, ∇Υ) + (1/η)(θ, Υ) = -(1/ε)(f_old, Υ) + (1/η)(θ_old, Υ) + (S_ψ, Υ)
//
// Note: +ε(∇θ, ∇Υ) in weak form = -εΔθ in strong form (IBP sign flip)
//
// Strong form for exact solution:
//   S_ψ = ψ - εΔθ + (1/ε)f(θ_old) + (1/η)(θ - θ_old)
//
// With θ = t⁴ cos(πx)cos(πy):
//   Δθ = -2π² t⁴ cos(πx)cos(πy)
//   -εΔθ = +2επ² t⁴ cos(πx)cos(πy)
//
// Therefore:
//   S_ψ = t⁴ sin·sin + 2επ² t⁴ cos·cos + (1/ε)f(θ_old) + (1/η)(θ - θ_old)
// ============================================================================
template <int dim>
class CHSourcePsi : public dealii::Function<dim>
{
public:
    CHSourcePsi(double epsilon, double dt)
        : dealii::Function<dim>(1), epsilon_(epsilon), dt_(dt) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;
        const double x = p[0], y = p[1];

        const double t4 = t * t * t * t;
        const double t4_old = t_old * t_old * t_old * t_old;

        const double cos_px = std::cos(M_PI * x);
        const double cos_py = std::cos(M_PI * y);
        const double sin_px = std::sin(M_PI * x);
        const double sin_py = std::sin(M_PI * y);

        // Values at current and old time
        const double theta_n = t4 * cos_px * cos_py;
        const double theta_old = t4_old * cos_px * cos_py;
        const double psi_n = t4 * sin_px * sin_py;

        // Δθ = -2π² t⁴ cos(πx) cos(πy)
        const double lap_theta = -2.0 * M_PI * M_PI * t4 * cos_px * cos_py;

        // f(θ_old) = θ_old³ - θ_old (LAGGED nonlinearity)
        const double f_old = theta_old * theta_old * theta_old - theta_old;

        // η = ε (stabilization parameter from paper)
        const double eta = epsilon_;

        // S_ψ = ψ - εΔθ + (1/ε)f(θ_old) + (1/η)(θ - θ_old)
        //     = psi_n - epsilon * lap_theta + ...
        return psi_n
               - epsilon_ * lap_theta   // Note: MINUS sign here!
               + (1.0 / epsilon_) * f_old
               + (1.0 / eta) * (theta_n - theta_old);
    }

private:
    double epsilon_;
    double dt_;
};

// ============================================================================
// INITIAL CONDITIONS for CH MMS
// ============================================================================
template <int dim>
class CHMMSInitialTheta : public dealii::Function<dim>
{
public:
    CHMMSInitialTheta(double t_init) : dealii::Function<dim>(1), t_init_(t_init) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1];
        const double t4 = t_init_ * t_init_ * t_init_ * t_init_;
        return t4 * std::cos(M_PI * x) * std::cos(M_PI * y);
    }

private:
    double t_init_;
};

template <int dim>
class CHMMSInitialPsi : public dealii::Function<dim>
{
public:
    CHMMSInitialPsi(double t_init) : dealii::Function<dim>(1), t_init_(t_init) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1];
        const double t4 = t_init_ * t_init_ * t_init_ * t_init_;
        return t4 * std::sin(M_PI * x) * std::sin(M_PI * y);
    }

private:
    double t_init_;
};

// ============================================================================
// BOUNDARY CONDITIONS for CH MMS (Dirichlet)
// ============================================================================
template <int dim>
class CHMMSBoundaryTheta : public dealii::Function<dim>
{
public:
    CHMMSBoundaryTheta() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;
        return t4 * std::cos(M_PI * x) * std::cos(M_PI * y);
    }
};

template <int dim>
class CHMMSBoundaryPsi : public dealii::Function<dim>
{
public:
    CHMMSBoundaryPsi() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;
        return t4 * std::sin(M_PI * x) * std::sin(M_PI * y);
    }
};

// ============================================================================
// MMS Error structure
// ============================================================================
struct CHMMSErrors
{
    double theta_L2 = 0.0;
    double theta_H1 = 0.0;
    double psi_L2 = 0.0;
    double h = 0.0;

    void print() const
    {
        std::cout << "CH MMS Errors:\n";
        std::cout << "  theta L2 = " << std::scientific << std::setprecision(4) << theta_L2 << "\n";
        std::cout << "  theta H1 = " << theta_H1 << "\n";
        std::cout << "  psi L2   = " << psi_L2 << "\n";
        std::cout << "  h        = " << h << "\n";
    }

    void print_for_convergence() const
    {
        std::cout << std::scientific << std::setprecision(4)
                  << h << "  " << theta_L2 << "  " << theta_H1 << "  " << psi_L2 << "\n";
    }
};

// ============================================================================
// Compute CH MMS errors
// ============================================================================
template <int dim>
CHMMSErrors compute_ch_mms_errors(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    double current_time)
{
    CHMMSErrors errors;

    CHExactTheta<dim> theta_exact;
    CHExactPsi<dim> psi_exact;
    theta_exact.set_time(current_time);
    psi_exact.set_time(current_time);

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 2);

    dealii::FEValues<dim> theta_fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> psi_fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    double theta_L2_sq = 0.0, theta_H1_sq = 0.0, psi_L2_sq = 0.0;
    double h_min = std::numeric_limits<double>::max();

    auto theta_cell = theta_dof_handler.begin_active();
    auto psi_cell = psi_dof_handler.begin_active();

    for (; theta_cell != theta_dof_handler.end(); ++theta_cell, ++psi_cell)
    {
        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);
        theta_cell->get_dof_indices(local_dof_indices);
        h_min = std::min(h_min, theta_cell->diameter());

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const auto& x_q = theta_fe_values.quadrature_point(q);

            double theta_h = 0.0, psi_h = 0.0;
            dealii::Tensor<1, dim> grad_theta_h;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                theta_h += theta_solution[local_dof_indices[i]] * theta_fe_values.shape_value(i, q);
                grad_theta_h += theta_solution[local_dof_indices[i]] * theta_fe_values.shape_grad(i, q);
                psi_h += psi_solution[local_dof_indices[i]] * psi_fe_values.shape_value(i, q);
            }

            const double theta_ex = theta_exact.value(x_q);
            const double psi_ex = psi_exact.value(x_q);
            const auto grad_theta_ex = theta_exact.gradient(x_q);

            theta_L2_sq += (theta_h - theta_ex) * (theta_h - theta_ex) * JxW;
            theta_H1_sq += (grad_theta_h - grad_theta_ex) * (grad_theta_h - grad_theta_ex) * JxW;
            psi_L2_sq += (psi_h - psi_ex) * (psi_h - psi_ex) * JxW;
        }
    }

    errors.theta_L2 = std::sqrt(theta_L2_sq);
    errors.theta_H1 = std::sqrt(theta_H1_sq);
    errors.psi_L2 = std::sqrt(psi_L2_sq);
    errors.h = h_min;

    return errors;
}

// ============================================================================
// Apply MMS Dirichlet boundary constraints
// ============================================================================
template <int dim>
void apply_ch_mms_boundary_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double current_time)
{
    theta_constraints.clear();
    psi_constraints.clear();

    CHMMSBoundaryTheta<dim> theta_bc;
    CHMMSBoundaryPsi<dim> psi_bc;
    theta_bc.set_time(current_time);
    psi_bc.set_time(current_time);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> theta_bc_map;
    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> psi_bc_map;

    for (unsigned int i = 0; i < 2 * dim; ++i)
    {
        theta_bc_map[i] = &theta_bc;
        psi_bc_map[i] = &psi_bc;
    }

    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler, theta_bc_map, theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler, psi_bc_map, psi_constraints);

    theta_constraints.close();
    psi_constraints.close();
}

// ============================================================================
// Apply MMS initial conditions
// ============================================================================
template <int dim>
void apply_ch_mms_initial_conditions(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution,
    double t_init)
{
    CHMMSInitialTheta<dim> theta_ic(t_init);
    CHMMSInitialPsi<dim> psi_ic(t_init);

    dealii::VectorTools::interpolate(theta_dof_handler, theta_ic, theta_solution);
    dealii::VectorTools::interpolate(psi_dof_handler, psi_ic, psi_solution);
}

// ============================================================================
// Source term for θ equation WITH CONVECTION (for coupled CH-NS test)
//
// Strong form: S_θ = ∂θ/∂t + U·∇θ + γΔψ
// ============================================================================
template <int dim>
class CHSourceThetaWithConvection : public dealii::Function<dim>
{
public:
    CHSourceThetaWithConvection(double gamma, double dt, double L_y = 1.0)
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt), L_y_(L_y) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;
        const double x = p[0], y = p[1];

        const double t4 = t * t * t * t;
        const double t4_old = t_old * t_old * t_old * t_old;

        const double cos_px = std::cos(M_PI * x);
        const double cos_py = std::cos(M_PI * y);
        const double sin_px = std::sin(M_PI * x);
        const double sin_py = std::sin(M_PI * y);

        // Time derivative: (θⁿ - θⁿ⁻¹)/dt
        const double dtheta_dt = (t4 - t4_old) / dt_ * cos_px * cos_py;

        // Δψ = -2π² t⁴ sin(πx) sin(πy)
        const double lap_psi = -2.0 * M_PI * M_PI * t4 * sin_px * sin_py;

        // Exact velocity from ns_mms.h at time t_old (lagged)
        // ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
        // uy = -t·π·sin(2πx)·sin²(πy/L_y)
        const double ux_old = t_old * (M_PI / L_y_) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y_);
        const double uy_old = -t_old * M_PI * std::sin(2.0 * M_PI * x) * std::sin(M_PI * y / L_y_) * std::sin(M_PI * y / L_y_);

        // ∇θ_old at t_old
        const double dtheta_dx = -t4_old * M_PI * sin_px * cos_py;
        const double dtheta_dy = -t4_old * M_PI * cos_px * sin_py;

        // Convection: U_old · ∇θ_old
        const double convection = ux_old * dtheta_dx + uy_old * dtheta_dy;

        // S_θ = (θ - θ_old)/dt + U·∇θ + γΔψ
        return dtheta_dt + convection + gamma_ * lap_psi;
    }

private:
    double gamma_;
    double dt_;
    double L_y_;
};

#endif // CH_MMS_H