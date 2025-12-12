// ============================================================================
// physics/boundary_conditions.h - Boundary Conditions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 15, p.501
// ============================================================================
#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

/**
 * @brief Boundary conditions for the ferrofluid system
 *
 * Boundary conditions (Eq. 15, p.501):
 *
 *   ∂_n θ = 0     on Γ × (0, t_F]    (homogeneous Neumann)
 *   ∂_n ψ = 0     on Γ × (0, t_F]    (homogeneous Neumann)
 *   u = 0         on Γ × (0, t_F]    (no-slip)
 *   ∂_n φ = (h_a - m)·n  on Γ × (0, t_F]  (Neumann, handled in assembly)
 *
 * Notes:
 *   - Neumann BCs for θ, ψ are natural (no constraints needed)
 *   - No-slip for u requires Dirichlet constraints
 *   - Magnetization m has no BCs (advection-reaction, no Laplacian)
 *   - Poisson BC is natural Neumann (handled in variational form)
 */

/**
 * @brief No-slip boundary condition for velocity u = 0
 */
template <int dim>
class NoSlipBoundary : public dealii::Function<dim>
{
public:
    NoSlipBoundary() : dealii::Function<dim>(dim) {}
    
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
        for (unsigned int i = 0; i < dim; ++i)
            values(i) = 0.0;
    }
};

/**
 * @brief Zero function for homogeneous Dirichlet/Neumann
 */
template <int dim>
class ZeroFunction : public dealii::Function<dim>
{
public:
    ZeroFunction() : dealii::Function<dim>(1) {}
    
    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)p;
        (void)component;
        return 0.0;
    }
};

#endif // BOUNDARY_CONDITIONS_H