// ============================================================================
// mms_tests/full_coupled_mms.h — Full 4-System Coupled MMS Sources
//
// Nochetto, Salgado & Tomas, arXiv:1511.04381, Algorithm 42
//
// Four coupled PDEs with ALL cross-coupling terms:
//   Poisson (42d):  (grad phi, grad X) = (h_a - M, grad X) + (f_phi, X)
//   Mag (42c):      time + relaxation + transport B_h^m(u; M, z) + IP diff
//   NS (42e):       time + viscous + pressure + micropolar + Kelvin force
//   AngMom (42f):   time + diffusion + reaction + curl + magnetic torque
//
// COUPLING STRUCTURE per time step:
//   1. Picard loop: Poisson(M_relaxed) <-> Mag(M_old, H, u_old)
//   2. NS(u_old, w_old, M, H) — Kelvin force + micropolar
//   3. AngMom(w_old, u_new, M, H) — curl coupling + magnetic torque
//
// EXACT SOLUTIONS (reusing all standalone MMS headers):
//   u* = (t pi sin^2(pi x) sin(2pi y), -t pi sin(2pi x) sin^2(pi y))
//   p* = t cos(pi x) cos(pi y)
//   w* = t sin(pi x) sin(pi y)
//   phi* = t cos(pi x) cos(pi y)
//   M* = (t sin(pi x) sin(pi y), t cos(pi x) sin(pi y))
//
// NOTE: phi* = p* by coincidence (both satisfy Neumann BCs). This is fine
// for MMS since the PDEs are independent — different source terms.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_FULL_COUPLED_MMS_H
#define FHD_FULL_COUPLED_MMS_H

#include "navier_stokes/tests/navier_stokes_mms.h"
#include "angular_momentum/tests/angular_momentum_mms.h"
#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact M* gradient (needed for transport and Kelvin force)
//
// M*_x = t sin(pi x) sin(pi y)
//   dM*_x/dx = pi t cos(pi x) sin(pi y)
//   dM*_x/dy = pi t sin(pi x) cos(pi y)
//
// M*_y = t cos(pi x) sin(pi y)
//   dM*_y/dx = -pi t sin(pi x) sin(pi y)
//   dM*_y/dy = pi t cos(pi x) cos(pi y)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> grad_Mx_exact(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> g;
    g[0] = M_PI * t * std::cos(M_PI * x) * std::sin(M_PI * y);
    g[1] = M_PI * t * std::sin(M_PI * x) * std::cos(M_PI * y);
    return g;
}

template <int dim>
dealii::Tensor<1, dim> grad_My_exact(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> g;
    g[0] = -M_PI * t * std::sin(M_PI * x) * std::sin(M_PI * y);
    g[1] = M_PI * t * std::cos(M_PI * x) * std::cos(M_PI * y);
    return g;
}

template <int dim>
double div_M_exact(const dealii::Point<dim>& p, double t)
{
    const auto gMx = grad_Mx_exact<dim>(p, t);
    const auto gMy = grad_My_exact<dim>(p, t);
    return gMx[0] + gMy[1];
}

// ============================================================================
// Exact H* = grad(phi*) and Hessian(phi*)
//
// phi* = t cos(pi x) cos(pi y)
// H* = (-pi t sin(pi x) cos(pi y), -pi t cos(pi x) sin(pi y))
//
// Hess(phi*)[0][0] = -pi^2 t cos(pi x) cos(pi y)
// Hess(phi*)[0][1] = pi^2 t sin(pi x) sin(pi y)
// Hess(phi*)[1][0] = pi^2 t sin(pi x) sin(pi y)
// Hess(phi*)[1][1] = -pi^2 t cos(pi x) cos(pi y)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> H_exact(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> H;
    H[0] = -M_PI * t * std::sin(M_PI * x) * std::cos(M_PI * y);
    H[1] = -M_PI * t * std::cos(M_PI * x) * std::sin(M_PI * y);
    return H;
}

template <int dim>
dealii::Tensor<2, dim> hess_phi_exact(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    const double cx = std::cos(M_PI * x), sx = std::sin(M_PI * x);
    const double cy = std::cos(M_PI * y), sy = std::sin(M_PI * y);

    dealii::Tensor<2, dim> H;
    H[0][0] = -M_PI * M_PI * t * cx * cy;
    H[0][1] = M_PI * M_PI * t * sx * sy;
    H[1][0] = M_PI * M_PI * t * sx * sy;
    H[1][1] = -M_PI * M_PI * t * cx * cy;
    return H;
}

// ============================================================================
// Analytical Kelvin force: F_K = (M*. grad)H* + 1/2 (div M*) H*
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> kelvin_force_exact(const dealii::Point<dim>& p, double t)
{
    const auto M = magnetization_exact<dim>(p, t);
    const auto hess = hess_phi_exact<dim>(p, t);
    const auto H = H_exact<dim>(p, t);
    const double divM = div_M_exact<dim>(p, t);

    // (M . grad) H[i] = sum_j M[j] * hess[i][j]
    dealii::Tensor<1, dim> M_grad_H;
    for (unsigned int i = 0; i < dim; ++i)
    {
        M_grad_H[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
            M_grad_H[i] += M[j] * hess[i][j];
    }

    dealii::Tensor<1, dim> F;
    for (unsigned int d = 0; d < dim; ++d)
        F[d] = M_grad_H[d] + 0.5 * divM * H[d];
    return F;
}

// ============================================================================
// Analytical magnetic torque: m* x h* (scalar in 2D)
// ============================================================================
template <int dim>
double magnetic_torque_exact(const dealii::Point<dim>& p, double t)
{
    const auto M = magnetization_exact<dim>(p, t);
    const auto H = H_exact<dim>(p, t);
    return M[0] * H[1] - M[1] * H[0];
}

// ============================================================================
// Curl of scalar w in 2D: curl_vec(w) = (dw/dy, -dw/dx)
// w* = t sin(pi x) sin(pi y)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> curl_vec_w(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> curl;
    curl[0] = M_PI * t * std::sin(M_PI * x) * std::cos(M_PI * y);   // dw/dy
    curl[1] = -M_PI * t * std::cos(M_PI * x) * std::sin(M_PI * y);  // -dw/dx
    return curl;
}

// ============================================================================
// Curl of vector u in 2D: curl(u) = duy/dx - dux/dy (scalar)
// ============================================================================
template <int dim>
double curl_u_scalar(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    const double sx = std::sin(M_PI * x), sy = std::sin(M_PI * y);
    const double c2x = std::cos(2.0 * M_PI * x);
    const double c2y = std::cos(2.0 * M_PI * y);
    return -2.0 * M_PI * M_PI * t * c2x * sy * sy
           - 2.0 * M_PI * M_PI * t * sx * sx * c2y;
}

// ============================================================================
// FULL NS MMS SOURCE (Eq. 42e with ALL coupling)
//
// f_u = (u*_new - U_old_disc)/tau - (nu_eff/2) Delta u* + grad p*
//       + [convection: (U_old_disc . grad)u* + 1/2 div_U_old_disc * u*]
//       - 2 nu_r curl_vec(w*(t_old))
//       - mu_0 [(M*.grad)H* + 1/2 (div M*) H*]
//
// CRITICAL: Uses discrete U_old_disc (from assembly) in time derivative to
// avoid 1/τ amplification of (u*_old - U_old_disc) error.
// Convection uses discrete U_old_disc and div_U_old_disc from assembly to
// match the skew-symmetric form b_h(u_old; u, v) exactly.
// Uses analytical w*(t_old) for micropolar and M*, H* for Kelvin.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_full_ns_mms_source(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double nu_eff, double nu_r, double mu_0,
    const dealii::Tensor<1, dim>& U_old_disc,
    double div_U_old_disc = 0.0,
    bool include_convection = false)
{
    const double tau = t_new - t_old;

    const auto U_new = ns_exact_velocity<dim>(p, t_new);
    const auto lap = ns_exact_laplacian<dim>(p, t_new);
    const auto gp = ns_exact_pressure_gradient<dim>(p, t_new);

    // Micropolar: -2 nu_r curl_vec(w*(t_old))
    const auto cw = curl_vec_w<dim>(p, t_old);

    // Kelvin: -mu_0 F_K(t_new) [using converged M, H at current time]
    const auto FK = kelvin_force_exact<dim>(p, t_new);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
        f[d] = (U_new[d] - U_old_disc[d]) / tau
               - (nu_eff / 2.0) * lap[d]
               + gp[d]
               - 2.0 * nu_r * cw[d]
               - mu_0 * FK[d];

    // Convection: (U_old_disc . grad)u*(t_new) + 1/2 div_U_old_disc * u*(t_new)
    // This matches the assembly's skew-symmetric form b_h(u_old; u, v)
    if (include_convection)
    {
        dealii::Tensor<1, dim> grad_ux_new, grad_uy_new;
        ns_exact_velocity_gradient<dim>(p, t_new, grad_ux_new, grad_uy_new);

        // (U_old_disc . grad)u*_d = U_old_disc[0] * du*_d/dx + U_old_disc[1] * du*_d/dy
        f[0] += U_old_disc[0] * grad_ux_new[0] + U_old_disc[1] * grad_ux_new[1]
                + 0.5 * div_U_old_disc * U_new[0];
        if (dim > 1)
            f[1] += U_old_disc[0] * grad_uy_new[0] + U_old_disc[1] * grad_uy_new[1]
                    + 0.5 * div_U_old_disc * U_new[1];
    }

    return f;
}

// ============================================================================
// FULL ANGMOM MMS SOURCE (Eq. 42f with ALL coupling)
//
// f_w = j(w*_new - w_old_disc)/tau - c1 Delta w* + 4 nu_r w*
//       - 2 nu_r curl(u*(t_new))
//       - mu_0 (m*(t_new) x h*(t_new))
//
// CRITICAL: Uses discrete w_old_disc (from assembly) in time derivative.
// ============================================================================
template <int dim>
double compute_full_angmom_mms_source(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double j_micro, double c1, double nu_r, double mu_0,
    double w_old_disc)
{
    const double tau = t_new - t_old;
    const double w_new = angular_momentum_exact<dim>(p, t_new);

    // Time derivative: use DISCRETE w_old
    double f = j_micro * (w_new - w_old_disc) / tau;

    // Diffusion: -c1 Delta w* = +2 pi^2 c1 w*
    f += 2.0 * M_PI * M_PI * c1 * w_new;

    // Reaction: 4 nu_r w*
    f += 4.0 * nu_r * w_new;

    // Curl coupling: -2 nu_r curl(u*(t_new))
    f -= 2.0 * nu_r * curl_u_scalar<dim>(p, t_new);

    // Magnetic torque: -mu_0 (m* x h*)
    f -= mu_0 * magnetic_torque_exact<dim>(p, t_new);

    return f;
}

// ============================================================================
// FULL POISSON MMS SOURCE (Eq. 42d with M coupling)
//
// f_phi = -Delta phi* - div M*
// (Same as poisson_mag_mms.h but using the functions above)
// ============================================================================
template <int dim>
double compute_full_poisson_mms_source(const dealii::Point<dim>& p, double t)
{
    const double x = p[0], y = p[1];
    const double cx = std::cos(M_PI * x), cy = std::cos(M_PI * y);

    // -Delta phi* = 2 pi^2 t cos(pi x) cos(pi y)
    const double neg_lap = 2.0 * M_PI * M_PI * t * cx * cy;

    // div M*
    const double divM = div_M_exact<dim>(p, t);

    return neg_lap - divM;
}

// ============================================================================
// FULL MAGNETIZATION MMS SOURCE (Eq. 42c with velocity transport)
//
// f_M = (M* - M*_old)/tau + M*/T_relax - (kappa_0/T_relax) H_disc
//       + (u*_old . grad) M*
//
// Transport (u*.grad)M* uses analytical velocity and M gradient.
// Assembly adds discrete transport B_h^m(U_disc; M, z) on LHS;
// the pollution from u* vs U_disc converges.
//
// Note: div(u*) = 0 so the 1/2(div u)M skew term vanishes analytically.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_full_mag_mms_source(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double tau_M, double kappa_0,
    const dealii::Tensor<1, dim>& H_discrete,
    const dealii::Tensor<1, dim>& U_disc,
    double div_U_disc,
    const dealii::Tensor<1, dim>& M_old_disc)
{
    const double tau = t_new - t_old;

    const auto M_new = magnetization_exact<dim>(p, t_new);

    // Gradient of M* at t_new
    const auto gMx = grad_Mx_exact<dim>(p, t_new);
    const auto gMy = grad_My_exact<dim>(p, t_new);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
    {
        // Time derivative: use DISCRETE M_old to match assembly RHS
        f[d] = (M_new[d] - M_old_disc[d]) / tau;

        // Relaxation residual
        f[d] += M_new[d] / tau_M - kappa_0 * H_discrete[d] / tau_M;
    }

    // Transport: use DISCRETE velocity to match the assembly's skew form
    // B_h^m(U; M, z) = (U · ∇M, z) + ½(∇·U)(M, z)
    // For smooth M*, the face jump [[M*]] = 0 so face terms vanish.
    // Source must match: (U_disc · ∇)M* + ½(∇·U_disc) M*
    f[0] += U_disc[0] * gMx[0] + U_disc[1] * gMx[1]
          + 0.5 * div_U_disc * M_new[0];
    f[1] += U_disc[0] * gMy[0] + U_disc[1] * gMy[1]
          + 0.5 * div_U_disc * M_new[1];

    return f;
}

// ============================================================================
// Result structure for full 4-system test
// ============================================================================
struct FullCoupledMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // NS errors
    double U_L2 = 0.0, U_H1 = 0.0, p_L2 = 0.0;

    // AngMom errors
    double w_L2 = 0.0, w_H1 = 0.0;

    // Poisson errors
    double phi_L2 = 0.0, phi_H1 = 0.0;

    // Magnetization errors
    double M_L2 = 0.0;

    // Picard info
    unsigned int picard_iters = 0;

    double walltime = 0.0;
};

#endif // FHD_FULL_COUPLED_MMS_H
