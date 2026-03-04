// ============================================================================
// physics/benchmark_initial_conditions.h — ICs for Phase B benchmarks
//
// Circular droplet and square relaxation initial conditions for the
// Cahn-Hilliard FESystem(FE_Q, 2): component 0 = phi, component 1 = mu.
//
// Equilibrium interface profile for F(θ) = (1/16)(θ²−1)²:
//   φ(d) = tanh(d / (2√2 ε))
// where d = signed distance (positive inside region).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================
#ifndef FHD_BENCHMARK_INITIAL_CONDITIONS_H
#define FHD_BENCHMARK_INITIAL_CONDITIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Circular droplet IC
//
//   phi(x,y) = tanh((R − |x − center|) / (2√2 ε))
//   mu(x,y)  = 0
//
// phi = +1 inside droplet, −1 outside
// ============================================================================
template <int dim>
class CircularDropletIC : public dealii::Function<dim>
{
public:
    CircularDropletIC(const dealii::Point<dim>& center,
                      double radius,
                      double epsilon)
        : dealii::Function<dim>(2)
        , center_(center)
        , radius_(radius)
        , width_(2.0 * std::sqrt(2.0) * epsilon)
    {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int component = 0) const override
    {
        if (component == 0)
        {
            const double r = center_.distance(p);
            return std::tanh((radius_ - r) / width_);
        }
        return 0.0;   // mu = 0
    }

private:
    dealii::Point<dim> center_;
    double radius_;
    double width_;
};

// ============================================================================
// Square region IC
//
//   phi(x,y) = tanh(d_square / (2√2 ε))
//   mu(x,y)  = 0
//
// d_square = signed distance to square boundary (positive inside)
// ============================================================================
template <int dim>
class SquareRegionIC : public dealii::Function<dim>
{
public:
    SquareRegionIC(const dealii::Point<dim>& lower_left,
                   const dealii::Point<dim>& upper_right,
                   double epsilon)
        : dealii::Function<dim>(2)
        , lower_(lower_left)
        , upper_(upper_right)
        , width_(2.0 * std::sqrt(2.0) * epsilon)
    {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int component = 0) const override
    {
        if (component == 0)
        {
            const double dx = std::min(p[0] - lower_[0], upper_[0] - p[0]);
            const double dy = std::min(p[1] - lower_[1], upper_[1] - p[1]);

            double d;
            if (dx >= 0 && dy >= 0)
                d = std::min(dx, dy);          // inside: nearest edge
            else if (dx < 0 && dy < 0)
                d = -std::sqrt(dx*dx + dy*dy); // outside corner
            else
                d = std::min(dx, dy);          // outside edge

            return std::tanh(d / width_);
        }
        return 0.0;   // mu = 0
    }

private:
    dealii::Point<dim> lower_;
    dealii::Point<dim> upper_;
    double width_;
};

#endif // FHD_BENCHMARK_INITIAL_CONDITIONS_H
