// ============================================================================
// cahn_hilliard/tests/cahn_hilliard_mms.h - MMS Definitions for CH Facade
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42b
//
// Exact solutions (dim=2/3, domain [0,x_max]×[0,L_y]×[0,L_z]):
//   θ_exact = t⁴ cos(πx) cos(πy/L_y) [cos(πz/L_z) in 3D]
//   ψ_exact = t⁴ sin(πx) sin(πy/L_y) [sin(πz/L_z) in 3D]
//
// Provides:
//   - Exact solution Function<dim> classes (for IC, BCs, error computation)
//   - MMS source term wrappers matching CahnHilliardSubsystem::MmsSourceFunction
//   - Parallel error computation using ghosted vectors + MPI reduction
//
// Usage with facade:
//   CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);
//   ch.setup();
//
//   // Wrap source terms into facade callbacks
//   CHSourceTheta<dim> s_theta(gamma, dt, L_y);
//   CHSourcePsi<dim>   s_psi(epsilon, dt, L_y);
//   ch.set_mms_source(
//       [&](const Point<dim>& p, double t) { s_theta.set_time(t); return s_theta.value(p); },
//       [&](const Point<dim>& p, double t) { s_psi.set_time(t);   return s_psi.value(p); });
// ============================================================================
#ifndef CAHN_HILLIARD_MMS_H
#define CAHN_HILLIARD_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/trilinos_vector.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Namespace: dim-generic exact solutions and derivatives
// ============================================================================
namespace CHMMS
{
    inline double t4(double t) { return t * t * t * t; }

    // ---- θ_exact = t⁴ ∏_d cos(π x_d / L_d) ----
    template <int dim>
    inline double theta_exact_value(const dealii::Point<dim>& p, double t,
                                     const double L[dim])
    {
        double val = t4(t);
        for (unsigned int d = 0; d < dim; ++d)
            val *= std::cos(M_PI * p[d] / L[d]);
        return val;
    }

    // ---- ψ_exact = t⁴ ∏_d sin(π x_d / L_d) ----
    template <int dim>
    inline double psi_exact_value(const dealii::Point<dim>& p, double t,
                                   const double L[dim])
    {
        double val = t4(t);
        for (unsigned int d = 0; d < dim; ++d)
            val *= std::sin(M_PI * p[d] / L[d]);
        return val;
    }

    // ---- ∇θ ----
    template <int dim>
    inline dealii::Tensor<1, dim> theta_exact_grad(const dealii::Point<dim>& p,
                                                    double t, const double L[dim])
    {
        // θ = t⁴ ∏_d cos(π x_d / L_d)
        // ∂θ/∂x_k = -t⁴ (π/L_k) sin(πx_k/L_k) ∏_{d≠k} cos(πx_d/L_d)
        const double t4v = t4(t);
        dealii::Tensor<1, dim> grad;

        for (unsigned int k = 0; k < dim; ++k)
        {
            double prod = t4v * (-M_PI / L[k]) * std::sin(M_PI * p[k] / L[k]);
            for (unsigned int d = 0; d < dim; ++d)
                if (d != k)
                    prod *= std::cos(M_PI * p[d] / L[d]);
            grad[k] = prod;
        }
        return grad;
    }

    // ---- ∇ψ ----
    template <int dim>
    inline dealii::Tensor<1, dim> psi_exact_grad(const dealii::Point<dim>& p,
                                                   double t, const double L[dim])
    {
        // ψ = t⁴ ∏_d sin(π x_d / L_d)
        // ∂ψ/∂x_k = t⁴ (π/L_k) cos(πx_k/L_k) ∏_{d≠k} sin(πx_d/L_d)
        const double t4v = t4(t);
        dealii::Tensor<1, dim> grad;

        for (unsigned int k = 0; k < dim; ++k)
        {
            double prod = t4v * (M_PI / L[k]) * std::cos(M_PI * p[k] / L[k]);
            for (unsigned int d = 0; d < dim; ++d)
                if (d != k)
                    prod *= std::sin(M_PI * p[d] / L[d]);
            grad[k] = prod;
        }
        return grad;
    }

    // ---- Δθ ----
    template <int dim>
    inline double lap_theta_exact(const dealii::Point<dim>& p, double t,
                                   const double L[dim])
    {
        // Δθ = -t⁴ π² (∑_d 1/L_d²) ∏_d cos(πx_d/L_d)
        double sum_inv_L2 = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
            sum_inv_L2 += 1.0 / (L[d] * L[d]);

        return -M_PI * M_PI * sum_inv_L2 * theta_exact_value<dim>(p, t, L);
    }

    // ---- Δψ ----
    template <int dim>
    inline double lap_psi_exact(const dealii::Point<dim>& p, double t,
                                 const double L[dim])
    {
        // Δψ = -t⁴ π² (∑_d 1/L_d²) ∏_d sin(πx_d/L_d)
        double sum_inv_L2 = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
            sum_inv_L2 += 1.0 / (L[d] * L[d]);

        return -M_PI * M_PI * sum_inv_L2 * psi_exact_value<dim>(p, t, L);
    }
} // namespace CHMMS


// ============================================================================
// Function<dim> wrappers for the exact solutions
// ============================================================================

template <int dim>
class CHExactTheta : public dealii::Function<dim>
{
public:
    explicit CHExactTheta(const double L[dim])
        : dealii::Function<dim>(1)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    // 2D convenience constructor (backward compatible)
    explicit CHExactTheta(double L_y)
        : dealii::Function<dim>(1)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, this->get_time(), L_);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                    const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::theta_exact_grad<dim>(p, this->get_time(), L_);
    }

private:
    double L_[dim];
};


template <int dim>
class CHExactPsi : public dealii::Function<dim>
{
public:
    explicit CHExactPsi(const double L[dim])
        : dealii::Function<dim>(1)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    explicit CHExactPsi(double L_y)
        : dealii::Function<dim>(1)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, this->get_time(), L_);
    }

    dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                    const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::psi_exact_grad<dim>(p, this->get_time(), L_);
    }

private:
    double L_[dim];
};


// ============================================================================
// Source terms for standalone CH (no convection)
//
// Eq 42a forcing: S_θ = (θⁿ - θⁿ⁻¹)/dt + γ Δψⁿ
// Eq 42b forcing: S_ψ = ψⁿ - ε Δθⁿ + (1/ε) f(θⁿ⁻¹) + (1/η)(θⁿ - θⁿ⁻¹)
//
// where f(θ) = θ³ - θ,  η = ε
// ============================================================================

template <int dim>
class CHSourceTheta : public dealii::Function<dim>
{
public:
    CHSourceTheta(double gamma, double dt, const double L[dim])
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    CHSourceTheta(double gamma, double dt, double L_y)
        : dealii::Function<dim>(1), gamma_(gamma), dt_(dt)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;

        const double theta_n   = CHMMS::theta_exact_value<dim>(p, t, L_);
        const double theta_old = CHMMS::theta_exact_value<dim>(p, t_old, L_);
        const double lap_psi_n = CHMMS::lap_psi_exact<dim>(p, t, L_);

        return (theta_n - theta_old) / dt_ + gamma_ * lap_psi_n;
    }

private:
    double gamma_, dt_;
    double L_[dim];
};


template <int dim>
class CHSourcePsi : public dealii::Function<dim>
{
public:
    CHSourcePsi(double epsilon, double dt, const double L[dim])
        : dealii::Function<dim>(1), epsilon_(epsilon), dt_(dt)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    CHSourcePsi(double epsilon, double dt, double L_y)
        : dealii::Function<dim>(1), epsilon_(epsilon), dt_(dt)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        const double t = this->get_time();
        const double t_old = t - dt_;

        const double theta_n   = CHMMS::theta_exact_value<dim>(p, t, L_);
        const double theta_old = CHMMS::theta_exact_value<dim>(p, t_old, L_);
        const double psi_n     = CHMMS::psi_exact_value<dim>(p, t, L_);
        const double lap_theta_n = CHMMS::lap_theta_exact<dim>(p, t, L_);

        // f(θ^{n-1}) = θ³ - θ
        const double f_old = theta_old * theta_old * theta_old - theta_old;

        // η = ε  (stabilization)
        const double eta = epsilon_;

        return psi_n
             - epsilon_ * lap_theta_n
             + (1.0 / epsilon_) * f_old
             + (1.0 / eta) * (theta_n - theta_old);
    }

private:
    double epsilon_, dt_;
    double L_[dim];
};


// ============================================================================
// Initial conditions for MMS
// ============================================================================

template <int dim>
class CHMMSInitialTheta : public dealii::Function<dim>
{
public:
    CHMMSInitialTheta(double t_init, const double L[dim])
        : dealii::Function<dim>(1), t_init_(t_init)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    CHMMSInitialTheta(double t_init, double L_y)
        : dealii::Function<dim>(1), t_init_(t_init)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, t_init_, L_);
    }

private:
    double t_init_;
    double L_[dim];
};


template <int dim>
class CHMMSInitialPsi : public dealii::Function<dim>
{
public:
    CHMMSInitialPsi(double t_init, const double L[dim])
        : dealii::Function<dim>(1), t_init_(t_init)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    CHMMSInitialPsi(double t_init, double L_y)
        : dealii::Function<dim>(1), t_init_(t_init)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, t_init_, L_);
    }

private:
    double t_init_;
    double L_[dim];
};


// ============================================================================
// Dirichlet boundary conditions for MMS
// ============================================================================

template <int dim>
class CHMMSBoundaryTheta : public dealii::Function<dim>
{
public:
    explicit CHMMSBoundaryTheta(const double L[dim])
        : dealii::Function<dim>(1)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    explicit CHMMSBoundaryTheta(double L_y)
        : dealii::Function<dim>(1)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::theta_exact_value<dim>(p, this->get_time(), L_);
    }

private:
    double L_[dim];
};


template <int dim>
class CHMMSBoundaryPsi : public dealii::Function<dim>
{
public:
    explicit CHMMSBoundaryPsi(const double L[dim])
        : dealii::Function<dim>(1)
    {
        for (unsigned int d = 0; d < dim; ++d) L_[d] = L[d];
    }

    explicit CHMMSBoundaryPsi(double L_y)
        : dealii::Function<dim>(1)
    {
        L_[0] = 1.0;
        if constexpr (dim >= 2) L_[1] = L_y;
        if constexpr (dim >= 3) L_[2] = 1.0;
    }

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        return CHMMS::psi_exact_value<dim>(p, this->get_time(), L_);
    }

private:
    double L_[dim];
};


// ============================================================================
// MMS error structure
// ============================================================================
struct CHMMSErrors
{
    double theta_L2   = 0.0;
    double theta_H1   = 0.0;
    double theta_Linf = 0.0;
    double psi_L2     = 0.0;
    double psi_Linf   = 0.0;
    double h          = 0.0;
};


// ============================================================================
// Parallel MMS error computation
//
// Takes ghosted solution vectors, iterates locally owned cells,
// reduces via MPI.
//
// IMPORTANT: theta_solution and psi_solution must be ghosted (locally_relevant).
// ============================================================================
template <int dim>
CHMMSErrors compute_ch_mms_errors(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double time,
    const double L[dim],
    MPI_Comm mpi_communicator)
{
    CHMMSErrors errors;

    CHExactTheta<dim> exact_theta(L);
    CHExactPsi<dim> exact_psi(L);
    exact_theta.set_time(time);
    exact_psi.set_time(time);

    const unsigned int quad_degree = theta_dof_handler.get_fe().degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);

    dealii::FEValues<dim> theta_fe_values(theta_dof_handler.get_fe(), quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> psi_fe_values(psi_dof_handler.get_fe(), quadrature,
        dealii::update_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> theta_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> theta_grads(n_q);
    std::vector<double> psi_vals(n_q);

    double local_theta_L2_sq = 0.0;
    double local_theta_H1_sq = 0.0;
    double local_theta_Linf  = 0.0;
    double local_psi_L2_sq   = 0.0;
    double local_psi_Linf    = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        theta_fe_values.reinit(cell);
        theta_fe_values.get_function_values(theta_solution, theta_vals);
        theta_fe_values.get_function_gradients(theta_solution, theta_grads);

        // Matching psi cell
        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &theta_dof_handler.get_triangulation(),
            cell->level(), cell->index(), &psi_dof_handler);
        psi_fe_values.reinit(psi_cell);
        psi_fe_values.get_function_values(psi_solution, psi_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = theta_fe_values.quadrature_point(q);
            const double JxW = theta_fe_values.JxW(q);

            // θ errors
            const double th_exact = exact_theta.value(x_q);
            const auto   grad_th_exact = exact_theta.gradient(x_q);
            const double th_err = theta_vals[q] - th_exact;
            const auto   grad_th_err = theta_grads[q] - grad_th_exact;

            local_theta_L2_sq += th_err * th_err * JxW;
            local_theta_H1_sq += grad_th_err * grad_th_err * JxW;
            local_theta_Linf = std::max(local_theta_Linf, std::abs(th_err));

            // ψ errors
            const double ps_exact = exact_psi.value(x_q);
            const double ps_err = psi_vals[q] - ps_exact;
            local_psi_L2_sq += ps_err * ps_err * JxW;
            local_psi_Linf = std::max(local_psi_Linf, std::abs(ps_err));
        }
    }

    // MPI reduction
    double global_theta_L2_sq = 0.0, global_theta_H1_sq = 0.0, global_psi_L2_sq = 0.0;
    double global_theta_Linf = 0.0, global_psi_Linf = 0.0;
    MPI_Allreduce(&local_theta_L2_sq, &global_theta_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_theta_H1_sq, &global_theta_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_psi_L2_sq,   &global_psi_L2_sq,   1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_theta_Linf,  &global_theta_Linf,  1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
    MPI_Allreduce(&local_psi_Linf,    &global_psi_Linf,    1, MPI_DOUBLE, MPI_MAX, mpi_communicator);

    errors.theta_L2   = std::sqrt(global_theta_L2_sq);
    errors.theta_H1   = std::sqrt(global_theta_H1_sq);
    errors.theta_Linf = global_theta_Linf;
    errors.psi_L2     = std::sqrt(global_psi_L2_sq);
    errors.psi_Linf   = global_psi_Linf;

    return errors;
}

#endif // CAHN_HILLIARD_MMS_H