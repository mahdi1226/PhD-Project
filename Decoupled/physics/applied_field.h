// ============================================================================
// physics/applied_field.h - Applied Magnetic Field h_a
//
// Two modes:
//   1. Uniform field (Zhang, He & Yang, CMAME 371 (2020)):
//      h_a(x, t) = direction * intensity_max * min(t/ramp_time, 1)
//      Spatially constant, with optional linear time ramp.
//
//   2. Dipole sources (Nochetto, Salgado & Tomas, CMAME 309 (2016)):
//      h_a from 2D line dipoles (Eq. 97-98, p.529).
//
// Used by:
//   - Poisson RHS (Eq. 42d):        (h_a - M, ∇χ)
//   - Magnetization RHS (Eq. 42c):  H_total = h_a + ∇φ
// ============================================================================
#ifndef APPLIED_FIELD_H
#define APPLIED_FIELD_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "utilities/parameters.h"

#include <cmath>
#include <algorithm>

// ============================================================================
// Check whether any applied field is configured
// ============================================================================
inline bool has_applied_field(const Parameters& params)
{
    if (params.uniform_field.enabled &&
        (params.uniform_field.intensity_max > 0.0 || params.uniform_field.ramp_slope > 0.0))
        return true;
    if (!params.dipoles.positions.empty() &&
        (params.dipoles.intensity_max > 0.0 || params.dipoles.ramp_slope > 0.0))
        return true;
    return false;
}

// ============================================================================
// Compute h_a at a point
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_applied_field(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    dealii::Tensor<1, dim> h_a;

    // ---- Mode 1: Uniform field (Zhang, He & Yang 2020) ----
    if (params.uniform_field.enabled)
    {
        // Two ramp modes:
        //   1. ramp_slope > 0: α(t) = slope * t, capped at intensity_max
        //   2. ramp_time > 0:  α(t) = intensity_max * min(t/ramp_time, 1)
        //   3. both zero:      α = intensity_max (instant)
        double alpha;
        if (params.uniform_field.ramp_slope > 0.0)
        {
            alpha = params.uniform_field.ramp_slope * current_time;
            if (params.uniform_field.intensity_max > 0.0)
                alpha = std::min(alpha, params.uniform_field.intensity_max);
        }
        else
        {
            const double ramp_factor = (params.uniform_field.ramp_time > 0.0)
                ? std::min(current_time / params.uniform_field.ramp_time, 1.0)
                : 1.0;
            alpha = params.uniform_field.intensity_max * ramp_factor;
        }

        for (unsigned int d = 0; d < dim && d < params.uniform_field.direction.size(); ++d)
            h_a[d] = params.uniform_field.direction[d] * alpha;

        (void)p;  // uniform: independent of position
        return h_a;
    }

    // ---- Mode 2: Dipole sources (Nochetto et al. 2016) ----
    if constexpr (dim != 2)
    {
        (void)p; (void)params; (void)current_time;
        return h_a;
    }
    else
    {
        if (params.dipoles.positions.empty())
            return h_a;

        // Two ramp modes:
        //   1. ramp_slope > 0: α(t) = slope * t, capped at intensity_max
        //   2. ramp_time > 0:  α(t) = intensity_max * min(t/ramp_time, 1)
        //   3. both zero:      α = intensity_max (instant)
        double alpha;
        if (params.dipoles.ramp_slope > 0.0)
        {
            alpha = params.dipoles.ramp_slope * current_time;
            if (params.dipoles.intensity_max > 0.0)
                alpha = std::min(alpha, params.dipoles.intensity_max);
        }
        else
        {
            const double ramp_factor = (params.dipoles.ramp_time > 0.0)
                ? std::min(current_time / params.dipoles.ramp_time, 1.0)
                : 1.0;
            alpha = params.dipoles.intensity_max * ramp_factor;
        }
        const double d_x = params.dipoles.direction[0];
        const double d_y = params.dipoles.direction[1];

        const double min_dipole_dist = std::abs(params.dipoles.positions[0][1]);
        const double delta = 0.01 * min_dipole_dist;
        const double delta_sq = delta * delta;

        h_a[0] = 0.0;
        h_a[1] = 0.0;

        for (const auto& dipole_pos : params.dipoles.positions)
        {
            const double rx = dipole_pos[0] - p[0];
            const double ry = dipole_pos[1] - p[1];
            const double r_sq = rx * rx + ry * ry;
            const double r_eff_sq = r_sq + delta_sq;

            if (r_eff_sq < 1e-12)
                continue;

            const double r_eff_sq_sq = r_eff_sq * r_eff_sq;
            const double d_dot_r = d_x * rx + d_y * ry;

            h_a[0] += alpha * (-d_x / r_eff_sq + 2.0 * d_dot_r * rx / r_eff_sq_sq);
            h_a[1] += alpha * (-d_y / r_eff_sq + 2.0 * d_dot_r * ry / r_eff_sq_sq);
        }

        return h_a;
    }
}

#endif // APPLIED_FIELD_H
