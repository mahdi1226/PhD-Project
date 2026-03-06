// ============================================================================
// mms_tests/full_ch_fhd_mms.h — Full 6-Subsystem Coupled MMS with chi(phi), nu(phi)
//
// Phase B: All 4 FHD subsystems + Cahn-Hilliard with phase-dependent
// material properties:
//   chi(phi) = chi_0 * (phi+1)/2   in magnetization relaxation
//   nu(phi) = nu_w*(1-Phi) + nu_f*Phi  in NS viscous term
//   sigma * mu * grad(phi)  capillary force in NS
//   u * grad(phi)           convection in CH
//
// ALGORITHM per time step:
//   1. Picard loop: Mag(M_old, H, u_old, ch_old) <-> Poisson(M_relaxed)
//      - chi(phi_old) per-QP in Mag relaxation term
//   2. NS(u_old, w_old, M, H, ch_old) — nu(phi_old) + capillary + Kelvin + micropolar
//   3. AngMom(w_old, u_new, M, H)
//   4. CH(ch_old, u_new) — convection
//
// EXACT SOLUTIONS (reusing all standalone MMS):
//   u*, p*, w*, phi_mag*, M*       from full_coupled_mms.h
//   theta*(phi), mu_ch*(mu)        from cahn_hilliard_mms.h
//
// MMS SOURCE CONSTRUCTION:
//   NS: Phase A source (Kelvin + micropolar + convection) + capillary force
//   Mag: Phase A source (chi_q passed per-QP by assembler, replacing kappa_0)
//   CH: Coupled source with NS convection (from ch_ns_mms.h)
//   Poisson, AngMom: unchanged from Phase A
//
// NOTE ON VARIABLE VISCOSITY:
//   The NS MMS source uses -(nu_eff/2)*Delta_u* which is only correct for
//   spatially constant viscosity. With variable nu(phi), the strong form is
//   -div[nu D(u*)] = -(nu/2)*Delta_u* - grad(nu) . D(u*). The extra term
//   vanishes when nu_carrier = nu_ferro. For truly variable viscosity MMS,
//   the gradient correction term must be added.
//
// EXPECTED CONVERGENCE (CG Q2 vel/pot/ang/ch, DG Q2 mag, DG P1 pressure):
//   U_L2 ~ 3, U_H1 ~ 2, p_L2 ~ 2
//   w_L2 ~ 3, w_H1 ~ 2
//   phi_mag_L2 ~ 3, phi_mag_H1 ~ 2
//   M_L2 ~ 3
//   theta_L2 ~ 3, theta_H1 ~ 2
//   mu_L2 ~ 3, mu_H1 ~ 2
//
// Reference: Nochetto, Salgado & Tomas (2015, 2016)
// ============================================================================
#ifndef FHD_FULL_CH_FHD_MMS_H
#define FHD_FULL_CH_FHD_MMS_H

#include "mms_tests/full_coupled_mms.h"
#include "mms_tests/ch_ns_mms.h"

// ============================================================================
// Combined NS MMS source: Phase A + Phase B capillary force
//
// f_u = [Phase A: time + viscous + pressure + convection + micropolar + Kelvin]
//       - sigma * mu*(t_old) * grad(theta*(t_old))
//
// The assembler adds +sigma * mu_disc * grad(theta_disc) * v on the RHS.
// The source subtracts the exact capillary analytically.
// Pollution: sigma * (mu*grad(theta*) - mu_disc*grad(theta_disc)) -> O(h^2) L2
//   -> O(h^3) velocity L2 via Aubin-Nitsche.
//
// nu_eff is passed per-QP by the assembler as nu(theta_disc) + nu_r.
// With nu_carrier = nu_ferro, this is constant -> no grad(nu) correction needed.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_full_ch_fhd_ns_source(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double nu_eff, double nu_r, double mu_0, double sigma_cap,
    const dealii::Tensor<1, dim>& U_old_disc,
    double div_U_old_disc,
    bool include_convection)
{
    // Phase A source: time + viscous + pressure + convection + micropolar + Kelvin
    // Note: Phase A uses full skew form: (M·∇)H + ½div(M)H
    auto f = compute_full_ns_mms_source<dim>(
        p, t_new, t_old, nu_eff, nu_r, mu_0,
        U_old_disc, div_U_old_disc, include_convection);

    // Two-phase Kelvin correction: remove the ½div(M)H skew term
    // Phase A source subtracted -μ₀[(M·∇)H + ½div(M)H]. The two-phase
    // assembler uses only (M·∇)H, so we add back +μ₀·½div(M)H to
    // make the net source consistent: -μ₀(M·∇)H only.
    {
        const double divM = div_M_exact<dim>(p, t_new);
        const auto H = H_exact<dim>(p, t_new);
        for (unsigned int d = 0; d < dim; ++d)
            f[d] += mu_0 * 0.5 * divM * H[d];
    }

    // Phase B capillary force: -sigma * mu*(t_old) * grad(theta*(t_old))
    const double mu_ch = ch_exact_mu<dim>(p, t_old);
    const auto grad_theta = ch_exact_grad_phi<dim>(p, t_old);

    for (unsigned int d = 0; d < dim; ++d)
        f[d] -= sigma_cap * mu_ch * grad_theta[d];

    return f;
}

// ============================================================================
// Result structure for full 6-subsystem test
// ============================================================================
struct FullCHFHDMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // NS errors
    double U_L2 = 0.0, U_H1 = 0.0, p_L2 = 0.0;

    // AngMom errors
    double w_L2 = 0.0, w_H1 = 0.0;

    // Poisson (magnetic potential) errors
    double phi_mag_L2 = 0.0, phi_mag_H1 = 0.0;

    // Magnetization errors
    double M_L2 = 0.0;

    // CH (phase field + chemical potential) errors
    double theta_L2 = 0.0, theta_H1 = 0.0;
    double mu_L2 = 0.0, mu_H1 = 0.0;

    // Picard info
    unsigned int picard_iters = 0;

    double walltime = 0.0;
};

#endif // FHD_FULL_CH_FHD_MMS_H
