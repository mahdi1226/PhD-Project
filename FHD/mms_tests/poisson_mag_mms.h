// ============================================================================
// mms_tests/poisson_mag_mms.h — Coupled Poisson↔Magnetization MMS
//
// PAPER EQUATIONS 42c,d (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   Poisson:        (∇φ^k, ∇X) = (h_a − M^k, ∇X) + (f_φ, X)
//   Magnetization:  (1/τ + 1/𝒯)(M^k, z) = (1/τ)(M^{k-1}, z)
//                     + (κ₀/𝒯)(h^k, z) + (f_M, z)
//
// where h^k = h_a + ∇φ^k.  With h_a = 0, U = 0, σ = 0:
//   The two subsystems couple through M → Poisson RHS and ∇φ → Mag RHS.
//
// EXACT SOLUTIONS (from standalone MMS headers):
//   φ*(x,y,t) = t · cos(πx) · cos(πy)           (Neumann-compatible)
//   M*(x,y,t) = t · [sin(πx)·sin(πy), cos(πx)·sin(πy)]
//
// COUPLED POISSON SOURCE:
//   Poisson weak form:  (∇φ*, ∇X) = (−M*, ∇X) + (f_φ, X)
//   By integration by parts: f_φ = −Δφ* − ∇·M*
//
// COUPLED MAGNETIZATION SOURCE:
//   f_M = (M*(t_new) − M*(t_old))/τ + M*(t_new)/𝒯 − (κ₀/𝒯)·H_discrete
//
//   Note: uses H_discrete (from Poisson solve), not analytical ∇φ*.
//   This is correct because the assembly already adds (κ₀/𝒯)(H_discrete, z)
//   to the RHS. The MMS source supplies the residual.
//
// EXPECTED RATES:
//   φ: L2 = p+1 = 3, H1 = p = 2  (CG Q2)
//   M: L2 = p+1 = 3              (DG Q2)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================
#ifndef FHD_POISSON_MAG_MMS_H
#define FHD_POISSON_MAG_MMS_H

#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Coupled Poisson MMS source: f_φ = −Δφ* − ∇·M*
//
// With φ* = t·cos(πx)·cos(πy):
//   −Δφ* = 2π²t · cos(πx)·cos(πy)
//
// With M* = [t·sin(πx)·sin(πy), t·cos(πx)·sin(πy)]:
//   ∂Mx*/∂x = tπ · cos(πx)·sin(πy)
//   ∂My*/∂y = tπ · cos(πx)·cos(πy)
//   ∇·M* = tπ [cos(πx)·sin(πy) + cos(πx)·cos(πy)]
//
// f_φ = 2π²t·cos(πx)·cos(πy) − tπ·cos(πx)·sin(πy) − tπ·cos(πx)·cos(πy)
//     = t·cos(πx) · [(2π² − π)·cos(πy) − π·sin(πy)]
// ============================================================================
template <int dim>
double compute_poisson_mms_source_coupled(
    const dealii::Point<dim>& pt,
    double time)
{
    const double x = pt[0];
    const double y = pt[1];

    const double cx = std::cos(M_PI * x);
    const double cy = std::cos(M_PI * y);
    const double sy = std::sin(M_PI * y);

    // −Δφ*
    const double neg_lap_phi = 2.0 * M_PI * M_PI * time * cx * cy;

    // ∇·M*
    const double div_M = time * M_PI * cx * sy
                       + time * M_PI * cx * cy;

    return neg_lap_phi - div_M;
}

// ============================================================================
// Coupled Magnetization MMS source
//
// f_M = (M*(t_new) − M*(t_old))/τ + M*(t_new)/𝒯 − (κ₀/𝒯)·H_discrete
//
// The callback receives H_discrete (from the Poisson solve). The assembly
// already adds (κ₀/𝒯)(H_discrete, z) to the RHS, so the MMS source
// supplies the remaining terms to force the exact solution.
//
// Callback signature matches MagnetizationSubsystem::MmsSourceFunction:
//   (point, t_new, t_old, tau_M, kappa_0, H, U, div_U) → Tensor<1,dim>
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_coupled(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double tau_M, double kappa_0,
    const dealii::Tensor<1, dim>& H_discrete,
    const dealii::Tensor<1, dim>& /*U*/,
    double /*div_U*/,
    const dealii::Tensor<1, dim>& /*M_old_disc*/)
{
    const double tau = t_new - t_old;

    const auto M_new = magnetization_exact<dim>(p, t_new);
    const auto M_old = magnetization_exact<dim>(p, t_old);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
    {
        // Time derivative: (M*(t_new) − M*(t_old)) / τ
        f[d] = (M_new[d] - M_old[d]) / tau;

        // Relaxation residual: M*(t_new)/𝒯 − (κ₀/𝒯)·H_discrete
        f[d] += M_new[d] / tau_M - kappa_0 * H_discrete[d] / tau_M;
    }

    return f;
}

// ============================================================================
// Result struct for a single refinement level
// ============================================================================
struct PoissonMagMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // Poisson errors
    double phi_L2 = 0.0;
    double phi_H1 = 0.0;
    double phi_Linf = 0.0;

    // Magnetization errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double M_Linf = 0.0;

    // Picard iterations (last time step)
    unsigned int picard_iters = 0;

    // Wall time
    double walltime = 0.0;
};

#endif // FHD_POISSON_MAG_MMS_H
