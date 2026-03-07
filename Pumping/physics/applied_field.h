// ============================================================================
// physics/applied_field.h - Applied Magnetic Field h_a
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
//
// Two modes:
//   1. Uniform field:
//      h_a(x, t) = direction * intensity_max * min(t/ramp_time, 1)
//      Spatially constant, with optional linear time ramp.
//
//   2. Point dipole sources (Nochetto Eq. 101-103):
//      Scalar potential φ_s(x) = d · (x_s − x) / |x_s − x|²   (2D)
//      Applied field:   h_a = Σ α_s ∇φ_s
//
// Used by:
//   - Poisson RHS (Eq. 42d):        (h_a − M, ∇X)
//
// Note: h = ∇φ is the TOTAL magnetic field (paper p.8: "use that h = ∇φ").
// The applied field h_a is encoded into φ via the Poisson RHS.
// Assemblers for NS, magnetization, and angular momentum use H = ∇φ only.
// ============================================================================
#ifndef FHD_APPLIED_FIELD_H
#define FHD_APPLIED_FIELD_H

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
// Compute ramp intensity α(t)
//
// Two ramp modes:
//   1. ramp_slope > 0: α(t) = slope * t, capped at intensity_max
//   2. ramp_time > 0:  α(t) = intensity_max * min(t/ramp_time, 1)
//   3. both zero:      α = intensity_max (instant)
// ============================================================================
inline double compute_ramp_intensity(double intensity_max,
                                     double ramp_time,
                                     double ramp_slope,
                                     double current_time)
{
    if (ramp_slope > 0.0)
    {
        double alpha = ramp_slope * current_time;
        if (intensity_max > 0.0)
            alpha = std::min(alpha, intensity_max);
        return alpha;
    }

    const double ramp_factor = (ramp_time > 0.0)
        ? std::min(current_time / ramp_time, 1.0)
        : 1.0;
    return intensity_max * ramp_factor;
}

// ============================================================================
// Compute h_a at a point
//
// Nochetto Eq. 101-103 (dipole mode):
//   φ_s(x) = d · (x_s − x) / |x_s − x|²
//   h_a = Σ α_s ∇φ_s
//
// Gradient of φ_s:
//   ∂φ_s/∂x_i = α [-d_i/R² + 2(d·r)r_i/R⁴]
//   where r = x_s − x, R² = |r|²
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_applied_field(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    dealii::Tensor<1, dim> h_a;

    // ---- Mode 1: Uniform field ----
    if (params.uniform_field.enabled)
    {
        const double alpha = compute_ramp_intensity(
            params.uniform_field.intensity_max,
            params.uniform_field.ramp_time,
            params.uniform_field.ramp_slope,
            current_time);

        for (unsigned int d = 0; d < dim && d < params.uniform_field.direction.size(); ++d)
            h_a[d] = params.uniform_field.direction[d] * alpha;

        (void)p;
        return h_a;
    }

    // ---- Mode 2: Point dipole sources (2D only) ----
    if constexpr (dim != 2)
    {
        (void)p; (void)params; (void)current_time;
        return h_a;
    }
    else
    {
        if (params.dipoles.positions.empty())
            return h_a;

        // Shared intensity (fallback when per-dipole intensities not set)
        const double alpha_shared = compute_ramp_intensity(
            params.dipoles.intensity_max,
            params.dipoles.ramp_time,
            params.dipoles.ramp_slope,
            current_time);

        const bool has_per_dipole_dir = !params.dipoles.directions.empty();
        const bool has_per_dipole_int = !params.dipoles.intensities.empty();

        h_a[0] = 0.0;
        h_a[1] = 0.0;

        for (std::size_t s = 0; s < params.dipoles.positions.size(); ++s)
        {
            const auto& dipole_pos = params.dipoles.positions[s];
            const double rx = dipole_pos[0] - p[0];
            const double ry = dipole_pos[1] - p[1];
            const double r_sq = rx * rx + ry * ry;

            if (r_sq < 1e-12)
                continue;

            const double d_x = has_per_dipole_dir
                ? params.dipoles.directions[s][0] : params.dipoles.direction[0];
            const double d_y = has_per_dipole_dir
                ? params.dipoles.directions[s][1] : params.dipoles.direction[1];
            const double alpha_s = has_per_dipole_int
                ? params.dipoles.intensities[s] : alpha_shared;

            const double r_sq_sq = r_sq * r_sq;
            const double d_dot_r = d_x * rx + d_y * ry;

            // h_a += α_s ∇φ_s = α_s [-d/R² + 2(d·r)r/R⁴]
            h_a[0] += alpha_s * (-d_x / r_sq + 2.0 * d_dot_r * rx / r_sq_sq);
            h_a[1] += alpha_s * (-d_y / r_sq + 2.0 * d_dot_r * ry / r_sq_sq);
        }

        return h_a;
    }
}

#endif // FHD_APPLIED_FIELD_H
