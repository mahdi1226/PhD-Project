// ============================================================================
// mms/magnetic/magnetic_mms.h - Combined Magnetic MMS Header
//
// Convenience header that includes both Poisson and Magnetization MMS
// exact solutions and error computation utilities.
//
// Also defines `MagneticExactSolution<dim>` — a unified Function with
// (dim+1) components in the monolithic [Mx, My, ..., phi] ordering used
// by the FESystem (FE_DGQ^dim + FE_Q) in the production solver. Used by
// the coupled-MMS tests for IC interpolation onto the joint mag_dof.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_MMS_H
#define MAGNETIC_MMS_H

#include "mms/magnetic/poisson_mms.h"
#include "mms/magnetic/magnetization_mms.h"

// ============================================================================
// Unified exact solution for the monolithic [Mx, My, ..., phi] FESystem.
//
// Component layout (matches setup/magnetic_setup.cc / phase_field.h):
//   components 0..dim-1:  M components from MagExactMx, MagExactMy
//   component  dim:       phi from PoissonExactSolution
//
// This single Function<dim> is what VectorTools::interpolate expects when
// interpolating the IC onto the joint mag_dof_handler.
// ============================================================================
template <int dim>
class MagneticExactSolution : public dealii::Function<dim>
{
public:
    MagneticExactSolution(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(/*n_components=*/dim + 1)
        , time_(time)
        , L_y_(L_y)
        , Mx_(time, L_y)
        , My_(time, L_y)
        , phi_(time, L_y)
    {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        if (component == 0)   return Mx_.value(p);
        if (component == 1)   return My_.value(p);
        if (component == dim) return phi_.value(p);
        return 0.0;
    }

    virtual void vector_value(const dealii::Point<dim>& p,
                              dealii::Vector<double>& values) const override
    {
        Assert(values.size() == dim + 1,
               dealii::ExcDimensionMismatch(values.size(), dim + 1));
        values[0] = Mx_.value(p);
        if constexpr (dim >= 2) values[1] = My_.value(p);
        values[dim] = phi_.value(p);
    }

    virtual dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim>& p,
             const unsigned int component = 0) const override
    {
        if (component == 0)   return Mx_.gradient(p);
        if (component == 1)   return My_.gradient(p);
        if (component == dim) return phi_.gradient(p);
        return dealii::Tensor<1, dim>();
    }

    void set_time(double t) override
    {
        dealii::Function<dim>::set_time(t);
        time_ = t;
        Mx_.set_time(t);
        My_.set_time(t);
        phi_.set_time(t);
    }

private:
    double time_;
    double L_y_;
    MagExactMx<dim>           Mx_;
    MagExactMy<dim>           My_;
    PoissonExactSolution<dim> phi_;
};

#endif // MAGNETIC_MMS_H
