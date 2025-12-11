// ============================================================================
// utilities/nsch_mms.h - Manufactured solutions for coupled NS-CH verification
// ============================================================================
//
// MMS Solutions with consistent t^4 time dependence:
//   u = t^4 [sin(πx)cos(πy), -cos(πx)sin(πy)]   (divergence-free)
//   p = t^4 cos(πx)cos(πy)                       (zero mean)
//   c = t^4 cos(πx)cos(πy)
//   μ = t^4 sin(πx)sin(πy)
// ============================================================================
#ifndef NSCH_MMS_H
#define NSCH_MMS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <cmath>

struct MMSParams
{
    double nu     = 1.0;
    double M      = 1.0;
    double lambda = 1.0;
};

inline MMSParams& get_mms_params()
{
    static MMSParams p;
    return p;
}

// ============================================================================
// Exact velocity: u = t^4 [sin(πx)cos(πy), -cos(πx)sin(πy)]
// ============================================================================
template <int dim>
class MMSExactVelocity : public dealii::Function<dim>
{
public:
    MMSExactVelocity() : dealii::Function<dim>(dim) {}

    double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;
        if (component == 0)
            return t4 * std::sin(pi * x) * std::cos(pi * y);
        else
            return -t4 * std::cos(pi * x) * std::sin(pi * y);
    }

    void vector_value(const dealii::Point<dim>& p, dealii::Vector<double>& values) const override
    {
        for (unsigned int c = 0; c < dim; ++c)
            values(c) = value(p, c);
    }
};

// ============================================================================
// Exact pressure: p = t^4 cos(πx)cos(πy)
// ============================================================================
template <int dim>
class MMSExactPressure : public dealii::Function<dim>
{
public:
    MMSExactPressure() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;
        return t4 * std::cos(pi * x) * std::cos(pi * y);
    }
};

// ============================================================================
// Exact phase field: c = t^4 cos(πx)cos(πy)
// ============================================================================
template <int dim>
class MMSExactPhaseField : public dealii::Function<dim>
{
public:
    MMSExactPhaseField() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;
        return t4 * std::cos(pi * x) * std::cos(pi * y);
    }
};

// ============================================================================
// Exact chemical potential: μ = t^4 sin(πx)sin(πy)
// ============================================================================
template <int dim>
class MMSExactChemPotential : public dealii::Function<dim>
{
public:
    MMSExactChemPotential() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0], y = p[1], t = this->get_time();
        const double pi = dealii::numbers::PI;
        const double t4 = t * t * t * t;
        return t4 * std::sin(pi * x) * std::sin(pi * y);
    }
};

// ============================================================================
// Combined NS solution [u, p]
// ============================================================================
template <int dim>
class MMSExactNS : public dealii::Function<dim>
{
public:
    MMSExactNS() : dealii::Function<dim>(dim + 1) {}

    double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        if (component < dim) {
            MMSExactVelocity<dim> u;
            u.set_time(this->get_time());
            return u.value(p, component);
        } else {
            MMSExactPressure<dim> pr;
            pr.set_time(this->get_time());
            return pr.value(p);
        }
    }

    void vector_value(const dealii::Point<dim>& p, dealii::Vector<double>& values) const override
    {
        for (unsigned int c = 0; c < dim + 1; ++c)
            values(c) = value(p, c);
    }
};

// ============================================================================
// Combined CH solution [c, μ]
// ============================================================================
template <int dim>
class MMSExactCH : public dealii::Function<dim>
{
public:
    MMSExactCH() : dealii::Function<dim>(2) {}

    double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        if (component == 0) {
            MMSExactPhaseField<dim> c;
            c.set_time(this->get_time());
            return c.value(p);
        } else {
            MMSExactChemPotential<dim> mu;
            mu.set_time(this->get_time());
            return mu.value(p);
        }
    }

    void vector_value(const dealii::Point<dim>& p, dealii::Vector<double>& values) const override
    {
        for (unsigned int c = 0; c < 2; ++c)
            values(c) = value(p, c);
    }
};

// ============================================================================
// Source term for c equation: ∂c/∂t + u·∇c - M Δμ = S_c
// ============================================================================
template <int dim>
class MMSSourceC : public dealii::Function<dim>
{
public:
    MMSSourceC() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double pi = dealii::numbers::PI;
        const double M = get_mms_params().M;
        const double t3 = t * t * t;
        const double t4 = t * t * t * t;
        const double sin_px = std::sin(pi * x), cos_px = std::cos(pi * x);
        const double sin_py = std::sin(pi * y), cos_py = std::cos(pi * y);

        // ∂c/∂t = 4t³ cos(πx)cos(πy)
        const double c_t = 4.0 * t3 * cos_px * cos_py;

        // u·∇c
        const double u_x = t4 * sin_px * cos_py;
        const double u_y = -t4 * cos_px * sin_py;
        const double grad_c_x = -t4 * pi * sin_px * cos_py;
        const double grad_c_y = -t4 * pi * cos_px * sin_py;
        const double u_dot_grad_c = u_x * grad_c_x + u_y * grad_c_y;

        // Δμ = -2π² t^4 sin(πx)sin(πy)
        const double lap_mu = -2.0 * pi * pi * t4 * sin_px * sin_py;

        return c_t + u_dot_grad_c - M * lap_mu;
    }
};

// ============================================================================
// Source term for μ equation: μ - f'(c) + λΔc = S_μ
// ============================================================================
template <int dim>
class MMSSourceMu : public dealii::Function<dim>
{
public:
    MMSSourceMu() : dealii::Function<dim>(1) {}

    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double pi = dealii::numbers::PI;
        const double lambda = get_mms_params().lambda;
        const double t4 = t * t * t * t;
        const double sin_px = std::sin(pi * x), cos_px = std::cos(pi * x);
        const double sin_py = std::sin(pi * y), cos_py = std::cos(pi * y);

        const double c = t4 * cos_px * cos_py;
        const double mu = t4 * sin_px * sin_py;
        const double f_prime_c = c * c * c - c;
        const double lap_c = -2.0 * pi * pi * t4 * cos_px * cos_py;

        return mu - f_prime_c + lambda * lap_c;
    }
};

// ============================================================================
// Source term for momentum equation:
//
// Strong form: ∂u/∂t + (u·∇)u - νΔu + ∇p = 0
//
// We solve: (u^{n+1}/dt, v) + θν(∇u^{n+1}, ∇v) - (p^{n+1}, div v) = RHS
//           -(div u^{n+1}, q) = 0
//
// The source term S_u must make the exact solution satisfy this discrete system.
// Since we want the exact (u,p) to satisfy the equations, S_u should contain
// all terms from the strong form: ∂u/∂t + (u·∇)u - νΔu + ∇p
// ============================================================================
template <int dim>
class MMSSourceMomentum : public dealii::Function<dim>
{
public:
    MMSSourceMomentum() : dealii::Function<dim>(dim) {}

    double value(const dealii::Point<dim>& p, const unsigned int component = 0) const override
    {
        const double t = this->get_time();
        const double x = p[0], y = p[1];
        const double pi = dealii::numbers::PI;
        const double nu = get_mms_params().nu;
        const double t3 = t * t * t;
        const double t4 = t * t * t * t;
        const double sin_px = std::sin(pi * x), cos_px = std::cos(pi * x);
        const double sin_py = std::sin(pi * y), cos_py = std::cos(pi * y);

        // u = t^4 [sin(πx)cos(πy), -cos(πx)sin(πy)]
        const double u_x = t4 * sin_px * cos_py;
        const double u_y = -t4 * cos_px * sin_py;

        // ∂u/∂t = 4t³ [sin(πx)cos(πy), -cos(πx)sin(πy)]
        const double u_x_t = 4.0 * t3 * sin_px * cos_py;
        const double u_y_t = -4.0 * t3 * cos_px * sin_py;

        // Velocity gradients
        const double u_x_dx = t4 * pi * cos_px * cos_py;
        const double u_x_dy = -t4 * pi * sin_px * sin_py;
        const double u_y_dx = t4 * pi * sin_px * sin_py;
        const double u_y_dy = -t4 * pi * cos_px * cos_py;

        // (u·∇)u
        const double conv_x = u_x * u_x_dx + u_y * u_x_dy;
        const double conv_y = u_x * u_y_dx + u_y * u_y_dy;

        // Δu
        const double lap_u_x = -2.0 * pi * pi * t4 * sin_px * cos_py;
        const double lap_u_y = 2.0 * pi * pi * t4 * cos_px * sin_py;

        // ∇p where p = t^4 cos(πx)cos(πy)
        const double p_x = -t4 * pi * sin_px * cos_py;
        const double p_y = -t4 * pi * cos_px * sin_py;

        // S_u = ∂u/∂t + (u·∇)u - νΔu + ∇p
        if (component == 0)
            return u_x_t + conv_x - nu * lap_u_x + p_x;
        else
            return u_y_t + conv_y - nu * lap_u_y + p_y;
    }

    void vector_value(const dealii::Point<dim>& p, dealii::Vector<double>& values) const override
    {
        for (unsigned int c = 0; c < dim; ++c)
            values(c) = value(p, c);
    }
};

#endif // NSCH_MMS_H