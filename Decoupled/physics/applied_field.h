// ============================================================================
// physics/applied_field.h - Applied Magnetic Field h_a
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 97-98, p.529
//
// Computes the applied magnetic field h_a from line dipole sources.
// Used by:
//   - Poisson RHS (Eq. 42d):        (h_a - M, ∇χ)
//   - Magnetization RHS (Eq. 42c):  H_total = h_a + ∇φ
//
// 2D line dipole formula (Eq. 97-98):
//   h_a(x) = α Σ_i [ -d/|r_i|² + 2(d·r_i)r_i/|r_i|⁴ ]
//
// where:
//   α = intensity × ramp(t)
//   d = dipole direction (unit vector)
//   r_i = dipole_position_i - x
//   Regularization: |r|² → |r|² + δ² to avoid singularity
//
// 3D: Not yet implemented (requires point dipole formula).
// ============================================================================
#ifndef APPLIED_FIELD_H
#define APPLIED_FIELD_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "utilities/parameters.h"

#include <cmath>
#include <algorithm>

// ============================================================================
// Compute h_a at a point from all configured dipoles
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_applied_field(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    dealii::Tensor<1, dim> h_a;

    if constexpr (dim != 2)
    {
        // 3D point dipole: not yet implemented
        // static_assert will be removed when 3D formula is added
        (void)p; (void)params; (void)current_time;
        return h_a;
    }
    else
    {
        // Guard: no applied field if no dipoles configured
        if (params.dipoles.positions.empty())
            return h_a;

        // Time ramp: linearly increase from 0 to full over ramp_time
        const double ramp_factor = (params.dipoles.ramp_time > 0.0)
            ? std::min(current_time / params.dipoles.ramp_time, 1.0)
            : 1.0;

        const double alpha = params.dipoles.intensity_max * ramp_factor;
        const double d_x = params.dipoles.direction[0];
        const double d_y = params.dipoles.direction[1];

        // Regularization: δ = 1% of minimum dipole distance
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

            // 2D line dipole (Eq. 97-98)
            h_a[0] += alpha * (-d_x / r_eff_sq + 2.0 * d_dot_r * rx / r_eff_sq_sq);
            h_a[1] += alpha * (-d_y / r_eff_sq + 2.0 * d_dot_r * ry / r_eff_sq_sq);
        }

        return h_a;
    }
}

#endif // APPLIED_FIELD_H
