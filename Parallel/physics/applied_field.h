// ============================================================================
// utilities/applied_field.h - Applied Magnetic Field Computation
//
// Computes the applied magnetic field h_a from dipole sources.
// Used by both Poisson and Magnetization assemblers.
//
// PAPER REFERENCE: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 97-98
// ============================================================================
#ifndef APPLIED_FIELD_H
#define APPLIED_FIELD_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include "utilities/parameters.h"

#include <cmath>
#include <algorithm>

// ============================================================================
// Compute applied magnetic field h_a at a point
//
// For 2D: Computes dipole field contribution from all configured dipoles
// For 3D: Returns zero (not implemented)
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_applied_field(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    dealii::Tensor<1, dim> h_a;

    // Only implemented for 2D
    if constexpr (dim != 2)
    {
        // Return zero for 3D (not implemented)
        return h_a;
    }
    else
    {
        const double ramp_factor = (params.dipoles.ramp_time > 0.0)
            ? std::min(current_time / params.dipoles.ramp_time, 1.0)
            : 1.0;

        const double alpha = params.dipoles.intensity_max * ramp_factor;
        const double d_x = params.dipoles.direction[0];
        const double d_y = params.dipoles.direction[1];

        h_a[0] = 0.0;
        h_a[1] = 0.0;

        // FINAL CHECK--- Guard: no applied field if no dipoles ---
        if (params.dipoles.positions.empty())
        {
            return h_a;
        }


        const double min_dipole_dist = std::abs(params.dipoles.positions[0][1]);  // y-distance
        const double delta = 0.01 * min_dipole_dist;  // 1% of minimum distance
        const double delta_sq = delta * delta;

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