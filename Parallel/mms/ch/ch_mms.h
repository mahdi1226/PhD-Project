// ============================================================================
// mms/ch/ch_mms.h - MMS (Method of Manufactured Solutions) for CH
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Discrete scheme Eq. (42a)-(42b) (as implemented in your ch_assembler.cc)
//
// Exact solutions (dim=2, domain [0,1]×[0,L_y]):
//   θ_exact = t^4 cos(πx) cos(πy/L_y)
//   ψ_exact = t^4 sin(πx) sin(πy/L_y)
//
// IMPORTANT: L_y scaling ensures compatibility with NS MMS solutions which
// also use L_y-scaled y-coordinates. This is critical for coupled CH-NS tests.
//
// IMPORTANT ASSUMPTIONS (must match your driver):
//  1) You call set_time(current_time) on source term objects each step.
//  2) The dt passed into source objects equals the solver dt.
//  3) L_y parameter must match domain height (y_max - y_min).
// ============================================================================

#ifndef CH_MMS_H
#define CH_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace CHMMS
{
    // ---- small math helpers to avoid duplication ----
    inline double t4(const double t) { return t * t * t * t; }

    // ============================================================================
    // Exact solutions with L_y scaling (consistent with NS MMS)
    //
    // θ = t^4 cos(πx) cos(πy/L_y)
    // ψ = t^4 sin(πx) sin(πy/L_y)
    // ============================================================================

    template <int dim>
    inline double theta_exact_value(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        return t4(t) * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
    }

    template <int dim>
    inline double psi_exact_value(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        return t4(t) * std::sin(M_PI * x) * std::sin(M_PI * y / L_y);
    }

    template <int dim>
    inline dealii::Tensor<1, dim> theta_exact_grad(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        const double t4v = t4(t);

        // θ = t^4 cos(πx) cos(πy/L_y)
        // ∂θ/∂x = -t^4 π sin(πx) cos(πy/L_y)
        // ∂θ/∂y = -t^4 (π/L_y) cos(πx) sin(πy/L_y)

        dealii::Tensor<1, dim> grad;
        grad[0] = -t4v * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
        if constexpr (dim >= 2)
            grad[1] = -t4v * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
        return grad;
    }

    template <int dim>
    inline dealii::Tensor<1, dim> psi_exact_grad(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        const double t4v = t4(t);

        // ψ = t^4 sin(πx) sin(πy/L_y)
        // ∂ψ/∂x = t^4 π cos(πx) sin(πy/L_y)
        // ∂ψ/∂y = t^4 (π/L_y) sin(πx) cos(πy/L_y)

        dealii::Tensor<1, dim> grad;
        grad[0] = t4v * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
        if constexpr (dim >= 2)
            grad[1] = t4v * (M_PI / L_y) * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
        return grad;
    }

    template <int dim>
    inline double lap_theta_exact(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        // θ = t^4 cos(πx) cos(πy/L_y)
        // ∂²θ/∂x² = -t^4 π² cos(πx) cos(πy/L_y)
        // ∂²θ/∂y² = -t^4 (π/L_y)² cos(πx) cos(πy/L_y)
        // Δθ = -t^4 π² (1 + 1/L_y²) cos(πx) cos(πy/L_y)
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        return -M_PI * M_PI * (1.0 + 1.0 / (L_y * L_y)) * t4(t) * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
    }

    template <int dim>
    inline double lap_psi_exact(const dealii::Point<dim>& p, const double t, const double L_y = 1.0)
    {
        // ψ = t^4 sin(πx) sin(πy/L_y)
        // ∂²ψ/∂x² = -t^4 π² sin(πx) sin(πy/L_y)
        // ∂²ψ/∂y² = -t^4 (π/L_y)² sin(πx) sin(πy/L_y)
        // Δψ = -t^4 π² (1 + 1/L_y²) sin(πx) sin(πy/L_y)
        const double x = p[0];
        const double y = (dim >= 2 ? p[1] : 0.0);
        return -M_PI * M_PI * (1.0 + 1.0 / (L_y * L_y)) * t4(t) * std::sin(M_PI * x) * std::sin(M_PI * y / L_y);
    }
} // namespace CHMMS


// ============================================================================
// Exact phase field: θ = t^4 cos(πx) cos(πy/L_y)
// ============================================================================
template <int dim>
class CHExactTheta : public dealii::Function<dim>
{
public:
    explicit CHExactTheta(const double L_y = 1.0) : dealii::Function<dim>(1), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, this->get_time(), L_y_);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                    const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::theta_exact_grad<dim>(p, this->get_time(), L_y_);
    }

private:
    const double L_y_;
};

// ============================================================================
// Exact chemical potential: ψ = t^4 sin(πx) sin(πy/L_y)
// ============================================================================
template <int dim>
class CHExactPsi : public dealii::Function<dim>
{
public:
    explicit CHExactPsi(const double L_y = 1.0) : dealii::Function<dim>(1), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, this->get_time(), L_y_);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                    const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::psi_exact_grad<dim>(p, this->get_time(), L_y_);
    }

private:
    const double L_y_;
};


// ============================================================================
// Source term for θ equation (standalone CH, i.e., no convection)
//
// Matches your assembler weak form:
//
//   (1/dt)(θ, Λ) - γ(∇ψ,∇Λ) = (1/dt)(θ_old,Λ) + (S_θ,Λ)
//
// Strong form consistent with that discretization:
//
//   S_θ = (θ^n - θ^{n-1})/dt - γ Δψ^n
//
// But because your weak form has -γ(∇ψ,∇Λ), after IBP it corresponds to
// -γ Δψ on the left. Moving to RHS gives +γ Δψ.
// So the manufactured forcing you ADD to RHS is:
//
//   S_θ = (θ^n - θ^{n-1})/dt + γ Δψ^n
// ============================================================================
template <int dim>
class CHSourceTheta : public dealii::Function<dim>
{
public:
    CHSourceTheta(const double gamma, const double dt, const double L_y = 1.0)
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;

        const double theta_n = CHMMS::theta_exact_value<dim>(p, t, L_y_);
        const double theta_old = CHMMS::theta_exact_value<dim>(p, t_old, L_y_);
        const double lap_psi_n = CHMMS::lap_psi_exact<dim>(p, t, L_y_);

        const double dtheta_dt = (theta_n - theta_old) / dt_;

        return dtheta_dt + gamma_ * lap_psi_n;
    }

private:
    const double gamma_;
    const double dt_;
    const double L_y_;
};


// ============================================================================
// Source term for ψ equation (standalone CH)
//
// Matches your assembler weak form:
//
//   (ψ,Υ) + ε(∇θ,∇Υ) + (1/η)(θ,Υ)
//     = -(1/ε)(f_old,Υ) + (1/η)(θ_old,Υ) + (S_ψ,Υ)
//
// Strong form forcing to ADD on RHS:
//
//   S_ψ = ψ^n - ε Δθ^n + (1/ε) f(θ^{n-1}) + (1/η)(θ^n - θ^{n-1})
//
// NOTE: you use lagged nonlinearity f_old = θ_old^3 - θ_old.
// ============================================================================
template <int dim>
class CHSourcePsi : public dealii::Function<dim>
{
public:
    CHSourcePsi(const double epsilon, const double dt, const double L_y = 1.0)
        : dealii::Function<dim>(1), epsilon_(epsilon), dt_(dt), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;

        const double theta_n = CHMMS::theta_exact_value<dim>(p, t, L_y_);
        const double theta_old = CHMMS::theta_exact_value<dim>(p, t_old, L_y_);
        const double psi_n = CHMMS::psi_exact_value<dim>(p, t, L_y_);

        const double lap_theta_n = CHMMS::lap_theta_exact<dim>(p, t, L_y_);

        const double f_old = theta_old * theta_old * theta_old - theta_old;

        // Your code/comments assume η = ε for stabilization in this MMS.
        const double eta = epsilon_;

        return psi_n
            - epsilon_ * lap_theta_n
            + (1.0 / epsilon_) * f_old
            + (1.0 / eta) * (theta_n - theta_old);
    }

private:
    const double epsilon_;
    const double dt_;
    const double L_y_;
};


// ============================================================================
// Initial conditions for MMS
// ============================================================================
template <int dim>
class CHMMSInitialTheta : public dealii::Function<dim>
{
public:
    explicit CHMMSInitialTheta(const double t_init, const double L_y = 1.0)
        : dealii::Function<dim>(1), t_init_(t_init), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, t_init_, L_y_);
    }

private:
    const double t_init_;
    const double L_y_;
};

template <int dim>
class CHMMSInitialPsi : public dealii::Function<dim>
{
public:
    explicit CHMMSInitialPsi(const double t_init, const double L_y = 1.0)
        : dealii::Function<dim>(1), t_init_(t_init), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, t_init_, L_y_);
    }

private:
    const double t_init_;
    const double L_y_;
};


// ============================================================================
// Dirichlet boundary conditions for MMS
// ============================================================================
template <int dim>
class CHMMSBoundaryTheta : public dealii::Function<dim>
{
public:
    explicit CHMMSBoundaryTheta(const double L_y = 1.0) : dealii::Function<dim>(1), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, this->get_time(), L_y_);
    }

private:
    const double L_y_;
};

template <int dim>
class CHMMSBoundaryPsi : public dealii::Function<dim>
{
public:
    explicit CHMMSBoundaryPsi(const double L_y = 1.0) : dealii::Function<dim>(1), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, this->get_time(), L_y_);
    }

private:
    const double L_y_;
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
        std::cout << "CH MMS Errors:\n"
            << "  theta L2 = " << std::scientific << std::setprecision(4) << theta_L2 << "\n"
            << "  theta H1 = " << theta_H1 << "\n"
            << "  psi L2   = " << psi_L2 << "\n"
            << "  h        = " << h << "\n";
    }

    void print_for_convergence() const
    {
        std::cout << std::scientific << std::setprecision(4)
            << h << "  " << theta_L2 << "  " << theta_H1 << "  " << psi_L2 << "\n";
    }
};


// ============================================================================
// Apply MMS Dirichlet boundary constraints (MPI-safe)
//
// NOTE:
//  - For parallel, the caller MUST:
//      theta_constraints.clear(); theta_constraints.reinit(owned,relevant);
//      psi_constraints.clear();   psi_constraints.reinit(owned,relevant);
//    BEFORE calling this function.
//  - This function also does NOT call close() (caller decides when to close).
// ============================================================================
template <int dim>
void apply_ch_mms_boundary_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    const double current_time,
    const double L_y = 1.0)
{
    CHMMSBoundaryTheta<dim> theta_bc(L_y);
    CHMMSBoundaryPsi<dim> psi_bc(L_y);
    theta_bc.set_time(current_time);
    psi_bc.set_time(current_time);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> theta_bc_map;
    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> psi_bc_map;

    for (unsigned int bid = 0; bid < 2 * dim; ++bid)
    {
        theta_bc_map[bid] = &theta_bc;
        psi_bc_map[bid] = &psi_bc;
    }

    dealii::VectorTools::interpolate_boundary_values(theta_dof_handler,
                                                     theta_bc_map,
                                                     theta_constraints);

    dealii::VectorTools::interpolate_boundary_values(psi_dof_handler,
                                                     psi_bc_map,
                                                     psi_constraints);
}


// ============================================================================
// Source term for θ equation WITH CONVECTION (for coupled CH-NS MMS)
//
// Matches Nochetto Eq. 42a:
//   (δθ^k/τ, Λ) - (U^k θ^{k-1}, ∇Λ) - γ(∇ψ^k, ∇Λ) = 0
//
// The scheme uses U^k (velocity at current time) with θ^{k-1} (phase at old time).
//
// Strong form forcing to ADD on RHS:
//
//   S_θ = (θ^n - θ^{n-1})/dt + U^n · ∇θ^{n-1} + γ Δψ^n
//
// CRITICAL: Velocity U is evaluated at current time t (not t_old), matching
// the assembler which uses the NS solution from the current time step.
// The gradient ∇θ is evaluated at t_old (lagged phase field).
//
// Uses L_y-scaled coordinates for BOTH velocity AND θ to ensure consistency.
// ============================================================================
template <int dim>
class CHSourceThetaWithConvection : public dealii::Function<dim>
{
public:
    CHSourceThetaWithConvection(const double gamma,
                                const double dt,
                                const double L_y = 1.0)
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt), L_y_(L_y)
    {
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/  = 0) const override
    {
        static_assert(dim >= 2, "CHSourceThetaWithConvection expects dim >= 2.");

        const double t = this->get_time();
        const double t_old = t - dt_;

        const double x = p[0];
        const double y = p[1];

        // Discrete time derivative term (using L_y-scaled θ)
        const double theta_n = CHMMS::theta_exact_value<dim>(p, t, L_y_);
        const double theta_old = CHMMS::theta_exact_value<dim>(p, t_old, L_y_);
        const double dtheta_dt = (theta_n - theta_old) / dt_;

        // Convection uses U^n · ∇θ^{n-1} per Nochetto Eq. 42a
        // Velocity at CURRENT time t, gradient at OLD time t_old
        const auto grad_theta_old = CHMMS::theta_exact_grad<dim>(p, t_old, L_y_);

        // NS MMS velocity model (L_y-scaled) at CURRENT time t:
        //   ux = t*(π/L_y)*sin²(πx)*sin(2πy/L_y)
        //   uy = -t*π*sin(2πx)*sin²(πy/L_y)
        const double sin_px = std::sin(M_PI * x);
        const double sin_2px = std::sin(2.0 * M_PI * x);
        const double sin_pyl = std::sin(M_PI * y / L_y_);
        const double sin_2pyl = std::sin(2.0 * M_PI * y / L_y_);

        // CRITICAL FIX: Use t (current time), not t_old, for velocity
        // This matches the assembler which uses NS solution at current_time
        const double ux = t * (M_PI / L_y_) * (sin_px * sin_px) * sin_2pyl;
        const double uy = -t * M_PI * sin_2px * (sin_pyl * sin_pyl);

        const double convection = ux * grad_theta_old[0] + uy * grad_theta_old[1];

        // Laplacian term (using L_y-scaled ψ at current time)
        const double lap_psi_n = CHMMS::lap_psi_exact<dim>(p, t, L_y_);

        return dtheta_dt + convection + gamma_ * lap_psi_n;
    }

private:
    const double gamma_;
    const double dt_;
    const double L_y_;
};

#endif // CH_MMS_H