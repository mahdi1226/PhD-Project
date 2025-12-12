// ============================================================================
// physics/applied_field.cc - Applied Magnetic Field Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 96-98, p.519; Section 6.2, p.522
// ============================================================================

#include "applied_field.h"
#include "output/logger.h"
#include <cmath>

template <int dim>
AppliedField<dim>::AppliedField(
    const std::vector<dealii::Point<dim>>& positions,
    const dealii::Tensor<1, dim>& direction,
    double intensity_max,
    double ramp_time)
    : positions_(positions)
    , direction_(direction)
    , intensity_max_(intensity_max)
    , ramp_time_(ramp_time)
{
}

// ============================================================================
// get_intensity()
//
// Intensity ramp (p.522):
//   α_s(t) = 6000 * t/1.6    for t ≤ 1.6
//          = 6000            for t > 1.6
// ============================================================================
template <int dim>
double AppliedField<dim>::get_intensity(double time) const
{
    return intensity_max_ * std::min(time / ramp_time_, 1.0);
}

// ============================================================================
// compute_potential()
//
// 2D Dipole potential (Eq. 97, p.519):
//   φ_s(x) = d · (x_s - x) / |x_s - x|²
// ============================================================================
template <int dim>
double AppliedField<dim>::compute_potential(const dealii::Point<dim>& x,
                                             double time) const
{
    const double alpha = get_intensity(time);
    double potential = 0.0;

    for (const auto& x_s : positions_)
    {
        dealii::Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = x_s[d] - x[d];

        const double r_norm = r.norm();

        if (r_norm > 1e-10)
        {
            double d_dot_r = direction_ * r;

            if constexpr (dim == 2)
                potential += alpha * d_dot_r / (r_norm * r_norm);
            else
                potential += alpha * d_dot_r / (r_norm * r_norm * r_norm);
        }
    }

    return potential;
}

// ============================================================================
// compute_field()
//
// h_a = Σ_s α_s ∇φ_s  (Eq. 98)
//
// 2D gradient: ∇_x φ_s = -d/|r|² - 2(d·r)r/|r|⁴
//
// ASSUMPTION: We derived ∇_x φ_s = -d/|r|² - 2(d·r)r/|r|⁴ (2D) from φ_s = d·r/|r|² (Eq.97)
// BASIS: Derivative w.r.t. x with r = x_s - x, so ∇_x r = -I.
// QUESTION: Is our derived gradient ∇_x φ_s correct? Paper gives φ_s but not ∇φ_s explicitly.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> AppliedField<dim>::compute_field(
    const dealii::Point<dim>& x,
    double time) const
{
    const double alpha = get_intensity(time);
    dealii::Tensor<1, dim> h_a;

    for (const auto& x_s : positions_)
    {
        dealii::Tensor<1, dim> r;
        for (unsigned int d = 0; d < dim; ++d)
            r[d] = x_s[d] - x[d];

        const double r_norm = r.norm();

        if (r_norm > 1e-10)
        {
            const double d_dot_r = direction_ * r;
            const double r2 = r_norm * r_norm;

            if constexpr (dim == 2)
            {
                const double r4 = r2 * r2;
                for (unsigned int d = 0; d < dim; ++d)
                    h_a[d] += alpha * (-direction_[d] / r2 - 2.0 * d_dot_r * r[d] / r4);
            }
            else
            {
                const double r3 = r2 * r_norm;
                const double r5 = r3 * r2;
                for (unsigned int d = 0; d < dim; ++d)
                    h_a[d] += alpha * (-direction_[d] / r3 - 3.0 * d_dot_r * r[d] / r5);
            }
        }
    }

    return h_a;
}

template class AppliedField<2>;
// template class AppliedField<3>;  // Parameters uses Point<2> for dipoles