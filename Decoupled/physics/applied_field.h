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

        // Zhang uses NO regularization: φ_s(x) = d·(x_s - x)/|x_s - x|²
        // Dipoles are far below domain (y=-15), so no singularity protection needed.

        h_a[0] = 0.0;
        h_a[1] = 0.0;

        for (const auto& dipole_pos : params.dipoles.positions)
        {
            const double rx = dipole_pos[0] - p[0];
            const double ry = dipole_pos[1] - p[1];
            const double r_sq = rx * rx + ry * ry;

            if (r_sq < 1e-12)
                continue;

            const double r_sq_sq = r_sq * r_sq;
            const double d_dot_r = d_x * rx + d_y * ry;

            h_a[0] += alpha * (-d_x / r_sq + 2.0 * d_dot_r * rx / r_sq_sq);
            h_a[1] += alpha * (-d_y / r_sq + 2.0 * d_dot_r * ry / r_sq_sq);
        }

        return h_a;
    }
}

// ============================================================================
// Compute the Jacobian ∇h_a at a point: (∇h_a)_{ij} = ∂(h_a)_i / ∂x_j
//
// For uniform field: ∇h_a = 0 (spatially constant).
// For dipole sources: derived from differentiating the dipole formula.
//
// Dipole field per source s:
//   h_a_i = α (-d_i/r² + 2(d·r)r_i/r⁴)
//   where r = x_s - p,  r² = |r|² (no regularization, Zhang convention)
//
// Jacobian w.r.t. evaluation point p (∂r_k/∂p_j = -δ_{kj}):
//   ∂(h_a_i)/∂p_j = α/r⁴ [-2 d_i r_j - 2 d_j r_i - 2(d·r)δ_{ij}
//                          + 8(d·r) r_i r_j / r²]
//
// Used by Kelvin force: (M·∇)H where H = ∇φ + h_a, ∇H = Hess(φ) + ∇h_a.
// ============================================================================
template <int dim>
inline dealii::Tensor<2, dim> compute_applied_field_gradient(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    dealii::Tensor<2, dim> grad_h_a;  // zero-initialized

    // ---- Mode 1: Uniform field — gradient is zero ----
    if (params.uniform_field.enabled)
    {
        (void)p; (void)current_time;
        return grad_h_a;
    }

    // ---- Mode 2: Dipole sources (2D only) ----
    if constexpr (dim != 2)
    {
        (void)p; (void)params; (void)current_time;
        return grad_h_a;
    }
    else
    {
        if (params.dipoles.positions.empty())
            return grad_h_a;

        // Ramp intensity (same logic as compute_applied_field)
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

        // Zhang uses NO regularization — dipoles are far from domain.

        // d vector as array for indexed access
        const double d_vec[2] = {d_x, d_y};

        for (const auto& dipole_pos : params.dipoles.positions)
        {
            const double rx = dipole_pos[0] - p[0];
            const double ry = dipole_pos[1] - p[1];
            const double r_vec[2] = {rx, ry};
            const double r_sq = rx * rx + ry * ry;

            if (r_sq < 1e-12)
                continue;

            const double r_sq_sq = r_sq * r_sq;
            const double d_dot_r = d_x * rx + d_y * ry;

            // ∂(h_a_i)/∂p_j = α/r⁴ [-2 d_i r_j - 2 d_j r_i
            //                        - 2(d·r)δ_{ij} + 8(d·r)r_i r_j/r²]
            for (unsigned int i = 0; i < 2; ++i)
                for (unsigned int j = 0; j < 2; ++j)
                {
                    grad_h_a[i][j] += alpha / r_sq_sq * (
                        - 2.0 * d_vec[i] * r_vec[j]
                        - 2.0 * d_vec[j] * r_vec[i]
                        - 2.0 * d_dot_r * (i == j ? 1.0 : 0.0)
                        + 8.0 * d_dot_r * r_vec[i] * r_vec[j] / r_sq);
                }
        }

        return grad_h_a;
    }
}

#endif // APPLIED_FIELD_H
