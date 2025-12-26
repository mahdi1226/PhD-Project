// ============================================================================
// physics/initial_conditions.h - Initial Conditions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.522
// ============================================================================
#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>

/**
 * @brief Initial condition for phase field θ
 *
 * Supports two IC types:
 *   type = 0: Flat pool (Section 6.2, p.522)
 *   type = 1: Circular droplet
 *
 * Flat pool (type = 0):
 *   θ₀(x,y) = tanh((y - pool_depth) / (ε√2))
 *   θ ≈ -1 (non-magnetic) for y > pool_depth
 *   θ ≈ +1 (ferrofluid) for y < pool_depth
 *
 * Circular droplet (type = 1):
 *   θ₀(x,y) = tanh((r - radius) / (ε√2))
 *   where r = distance from center
 *   θ ≈ -1 inside droplet
 *   θ ≈ +1 outside droplet
 */
template <int dim>
class InitialTheta : public dealii::Function<dim>
{
public:
    /**
     * @brief Constructor for flat pool (type = 0)
     */
    InitialTheta(double pool_depth, double epsilon)
        : dealii::Function<dim>(1)
        , type_(0)
        , pool_depth_(pool_depth)
        , epsilon_(epsilon)
        , center_x_(0.0)
        , center_y_(0.0)
        , radius_(0.0)
    {}

    /**
     * @brief Constructor for circular droplet (type = 1)
     */
    InitialTheta(double epsilon, double center_x, double center_y, double radius,
                 int /*type_tag*/)  // type_tag just distinguishes constructors
        : dealii::Function<dim>(1)
        , type_(1)
        , pool_depth_(0.0)
        , epsilon_(epsilon)
        , center_x_(center_x)
        , center_y_(center_y)
        , radius_(radius)
    {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double interface_width = epsilon_ * std::sqrt(2.0);

        if (type_ == 0)  // Flat pool
        {
            const double y = p[1];
            return std::tanh((y - pool_depth_) / interface_width);
        }
        else  // Circular droplet (type_ == 1)
        {
            const double x = p[0];
            const double y = p[1];
            const double dist = std::sqrt((x - center_x_) * (x - center_x_) +
                                          (y - center_y_) * (y - center_y_));
            // θ = -1 inside droplet (ferrofluid), +1 outside
            return std::tanh((dist - radius_) / interface_width);
        }
    }

private:
    int type_;           // 0 = flat pool, 1 = circular droplet
    double pool_depth_;  // for flat pool
    double epsilon_;     // interface thickness
    double center_x_;    // for droplet
    double center_y_;    // for droplet
    double radius_;      // for droplet
};

/**
 * @brief Initial condition for chemical potential ψ
 *
 * From equilibrium (Eq. 14b with ∂θ/∂t = 0):
 *   ψ = εΔθ - (1/ε)f(θ)
 *
 * For smooth tanh profile, approximately:
 *   ψ₀ ≈ -(1/ε)(θ³ - θ)
 *
 * Supports both flat pool and circular droplet ICs.
 */
template <int dim>
class InitialPsi : public dealii::Function<dim>
{
public:
    /**
     * @brief Constructor for flat pool (type = 0)
     */
    InitialPsi(double pool_depth, double epsilon)
        : dealii::Function<dim>(1)
        , type_(0)
        , pool_depth_(pool_depth)
        , epsilon_(epsilon)
        , center_x_(0.0)
        , center_y_(0.0)
        , radius_(0.0)
    {}

    /**
     * @brief Constructor for circular droplet (type = 1)
     */
    InitialPsi(double epsilon, double center_x, double center_y, double radius,
               int /*type_tag*/)  // type_tag just distinguishes constructors
        : dealii::Function<dim>(1)
        , type_(1)
        , pool_depth_(0.0)
        , epsilon_(epsilon)
        , center_x_(center_x)
        , center_y_(center_y)
        , radius_(radius)
    {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double interface_width = epsilon_ * std::sqrt(2.0);

        double theta;
        if (type_ == 0)  // Flat pool
        {
            const double y = p[1];
            theta = std::tanh((y - pool_depth_) / interface_width);
        }
        else  // Circular droplet
        {
            const double x = p[0];
            const double y = p[1];
            const double dist = std::sqrt((x - center_x_) * (x - center_x_) +
                                          (y - center_y_) * (y - center_y_));
            theta = std::tanh((dist - radius_) / interface_width);
        }

        // ψ₀ = -(1/ε)(θ³ - θ) from equilibrium
        const double f_theta = theta * theta * theta - theta;
        return -f_theta / epsilon_;
    }

private:
    int type_;           // 0 = flat pool, 1 = circular droplet
    double pool_depth_;  // for flat pool
    double epsilon_;     // interface thickness
    double center_x_;    // for droplet
    double center_y_;    // for droplet
    double radius_;      // for droplet
};

/**
 * @brief Initial condition for velocity u
 *
 * u₀ = 0 (fluid at rest)
 * Paper states (p.522): "ferrofluid pool ... at rest at t = 0"
 */
template <int dim>
class InitialVelocity : public dealii::Function<dim>
{
public:
    InitialVelocity() : dealii::Function<dim>(dim) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)p;
        (void)component;
        return 0.0;
    }

    virtual void vector_value(const dealii::Point<dim>& p,
                              dealii::Vector<double>& values) const override
    {
        (void)p;
        for (unsigned int d = 0; d < dim; ++d)
            values(d) = 0.0;
    }
};

/**
 * @brief Initial condition for magnetization m
 *
 * In quasi-static model: m₀ = χ_θ h_a(0)
 * At t = 0, dipole intensity α_s(0) = 0, so h_a(0) = 0.
 * Therefore: m₀ = 0
 */
template <int dim>
class InitialMagnetization : public dealii::Function<dim>
{
public:
    InitialMagnetization() : dealii::Function<dim>(dim) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)p;
        (void)component;
        return 0.0;
    }
};

#endif // INITIAL_CONDITIONS_H