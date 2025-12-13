// ============================================================================
// diagnostics/ch_mms.h - MMS (Method of Manufactured Solutions) for CH
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 14a-14b, p.499
//
// NOTE ON BOUNDARY CONDITIONS:
//   - MMS verification uses DIRICHLET BCs (θ = θ_exact on ∂Ω)
//   - Physical runs use NEUMANN BCs (∇θ·n = 0) as per paper
//   This is standard practice: Dirichlet simplifies MMS error analysis.
//
// Domain: [0,1] × [0,0.6] (paper domain, Section 6.2, p.520)
//
// Exact solutions:
//   θ_exact = t⁴ cos(πx) cos(πy)
//   ψ_exact = t⁴ sin(πx) sin(πy)
//
// Source term derivation (with u=0 for standalone CH):
//
//   Eq 14a: θ_t - γΔψ = S_θ
//   Eq 14b: ψ - εΔθ + (1/ε)f(θ) = S_ψ
//
// where f(θ) = θ³ - θ
// ============================================================================
#ifndef CH_MMS_H
#define CH_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <cmath>

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
// Source term for θ equation (Eq. 14a with u=0)
//
// Strong form: θ_t - γΔψ = S_θ
//
// Given:
//   θ = t⁴ cos(πx) cos(πy)
//   ψ = t⁴ sin(πx) sin(πy)
//
// Compute:
//   θ_t = 4t³ cos(πx) cos(πy)
//   Δψ = -2π² t⁴ sin(πx) sin(πy)
//
// Therefore:
//   S_θ = 4t³ cos(πx) cos(πy) + 2γπ² t⁴ sin(πx) sin(πy)
// ============================================================================
template <int dim>
class CHSourceTheta : public dealii::Function<dim>
{
public:
    CHSourceTheta(double gamma) : dealii::Function<dim>(1), gamma_(gamma) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t3 = t * t * t;
        const double t4 = t * t * t * t;
        const double sin_px = std::sin(M_PI * x);
        const double cos_px = std::cos(M_PI * x);
        const double sin_py = std::sin(M_PI * y);
        const double cos_py = std::cos(M_PI * y);

        // θ_t = 4t³ cos(πx) cos(πy)
        const double theta_t = 4.0 * t3 * cos_px * cos_py;

        // Δψ = -2π² t⁴ sin(πx) sin(πy)
        const double lap_psi = -2.0 * M_PI * M_PI * t4 * sin_px * sin_py;

        // S_θ = θ_t - γΔψ
        return theta_t + gamma_ * lap_psi;
    }

private:
    double gamma_;
};

// ============================================================================
// Source term for ψ equation (Eq. 14b)
//
// Strong form: ψ - εΔθ + (1/ε)f(θ) = S_ψ
//
// Given:
//   θ = t⁴ cos(πx) cos(πy)
//   ψ = t⁴ sin(πx) sin(πy)
//   f(θ) = θ³ - θ
//
// Compute:
//   Δθ = -2π² t⁴ cos(πx) cos(πy)
//   f(θ) = θ³ - θ = t¹² cos³(πx) cos³(πy) - t⁴ cos(πx) cos(πy)
//
// Therefore:
//   S_ψ = ψ - ε(-2π² t⁴ cos(πx) cos(πy)) + (1/ε)(t¹² cos³ - t⁴ cos)
//       = t⁴ sin sin + 2επ² t⁴ cos cos + (1/ε)(t¹² cos³ cos³ - t⁴ cos cos)
// ============================================================================
template <int dim>
class CHSourcePsi : public dealii::Function<dim>
{
public:
    CHSourcePsi(double epsilon) : dealii::Function<dim>(1), epsilon_(epsilon) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double t4 = t * t * t * t;
        //const double t12 = t4 * t4 * t4;
        const double sin_px = std::sin(M_PI * x);
        const double cos_px = std::cos(M_PI * x);
        const double sin_py = std::sin(M_PI * y);
        const double cos_py = std::cos(M_PI * y);

        // θ = t⁴ cos(πx) cos(πy)
        const double theta = t4 * cos_px * cos_py;

        // ψ = t⁴ sin(πx) sin(πy)
        const double psi = t4 * sin_px * sin_py;

        // Δθ = -2π² t⁴ cos(πx) cos(πy)
        const double lap_theta = -2.0 * M_PI * M_PI * t4 * cos_px * cos_py;

        // f(θ) = θ³ - θ
        const double f_theta = theta * theta * theta - theta;

        // S_ψ = ψ - εΔθ + (1/ε)f(θ)
        return psi - epsilon_ * lap_theta + (1.0 / epsilon_) * f_theta;
    }

private:
    double epsilon_;
};

// ============================================================================
// INITIAL CONDITIONS for CH MMS
//
// IC = exact solution at t = t_init (typically small, e.g., 0.1 to avoid t=0)
// These are wrappers that set time and delegate to exact solution classes.
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
//
// For MMS verification: θ = θ_exact, ψ = ψ_exact on ∂Ω
// These are the same as exact solutions but provided as separate classes
// for clarity and to match the pattern used in coupled MMS later.
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
// MMS Error structure for CH system
// ============================================================================
struct CHMMSErrors
{
    double theta_L2 = 0.0;    // ‖θ - θ_exact‖_L2
    double theta_H1 = 0.0;    // |θ - θ_exact|_H1 (seminorm)
    double psi_L2 = 0.0;      // ‖ψ - ψ_exact‖_L2
    double h = 0.0;           // Mesh size for convergence study

    void print() const;
    void print_for_convergence() const;
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
    double current_time);

// ============================================================================
// Apply MMS Dirichlet BCs to constraints
// Call this in setup_constraints() when MMS mode is enabled
// ============================================================================
template <int dim>
void apply_ch_mms_boundary_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double current_time);

// ============================================================================
// Initialize solution vectors with MMS IC
// Call this in initialize_solutions() when MMS mode is enabled
// ============================================================================
template <int dim>
void apply_ch_mms_initial_conditions(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution,
    double t_init);

#endif // CH_MMS_H