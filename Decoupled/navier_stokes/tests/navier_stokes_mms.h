// ============================================================================
// navier_stokes/tests/navier_stokes_mms.h - MMS Definitions for NS Facade
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-42f
//
// Exact solutions (dim=2, domain [0,x_max]×[0,L_y]):
//   ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
//   uy = -t·π·sin(2πx)·sin²(πy/L_y)
//   p  = t·cos(πx)·cos(πy/L_y)
//
// Properties:  ∇·U = 0 exactly,  U = 0 on all boundaries
//
// Four MMS phases for staged verification:
//   Phase A: Steady Stokes     — f = -ν∇²U + ∇p
//   Phase B: Unsteady Stokes   — f = (U^n−U^{n−1})/dt − ν∇²U^n + ∇p^n
//   Phase C: Steady NS         — f = (U·∇)U − ν∇²U + ∇p
//   Phase D: Unsteady NS       — f = (U^n−U^{n−1})/dt + (U^{n−1}·∇)U^n − ν∇²U^n + ∇p^n
//
// Strong form PDE (for ∇·U = 0):
//   ∂U/∂t + (U·∇)U − ν∇²U + ∇p = f
//
// The weak form ν/2(T(U),T(V)) with T(U)=∇U+(∇U)^T corresponds to −ν∇²U
// in the strong form when ∇·U = 0, because:
//   ∇·T(U) = ∇²U + ∇(∇·U) = ∇²U
//
// Usage with facade:
//   NSSubsystem<dim> ns(params, mpi_comm, triangulation);
//   ns.setup();
//   NSMMSInitialUx<dim> ic_ux(t_init, L_y);
//   NSMMSInitialUy<dim> ic_uy(t_init, L_y);
//   ns.initialize_velocity(ic_ux, ic_uy);
//   // time loop body:
//   auto src = [&](const Point<dim>& p, double t) {
//       return NSMMS::source_phase_D<dim>(p, t, t_old, nu, L_y);
//   };
//   ns.assemble_stokes(dt, nu, true, true, &src, current_time);
//   ns.solve();
//   ns.update_ghosts();
//   auto errors = compute_ns_mms_errors(ns, t, Ly, comm);
// ============================================================================
#ifndef NAVIER_STOKES_MMS_H
#define NAVIER_STOKES_MMS_H

#include "navier_stokes/navier_stokes.h"

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================================
// Namespace: exact solutions, gradients, source terms
// ============================================================================
namespace NSMMS
{
    // ========================================================================
    // Exact solutions
    // ========================================================================
    template <int dim>
    inline double ux_val(const dealii::Point<dim>& p, double t, double Ly)
    {
        const double s = std::sin(M_PI * p[0]);
        return t * (M_PI / Ly) * s * s * std::sin(2.0 * M_PI * p[1] / Ly);
    }

    template <int dim>
    inline double uy_val(const dealii::Point<dim>& p, double t, double Ly)
    {
        const double s = std::sin(M_PI * p[1] / Ly);
        return -t * M_PI * std::sin(2.0 * M_PI * p[0]) * s * s;
    }

    template <int dim>
    inline double p_val(const dealii::Point<dim>& p, double t, double Ly)
    {
        return t * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1] / Ly);
    }

    // ========================================================================
    // Gradients (for H1 error computation and convection source)
    // ========================================================================
    template <int dim>
    inline dealii::Tensor<1, dim> ux_grad(const dealii::Point<dim>& p, double t, double Ly)
    {
        const double s = std::sin(M_PI * p[0]);
        dealii::Tensor<1, dim> g;
        g[0] = t * (M_PI * M_PI / Ly) * std::sin(2.0 * M_PI * p[0])
                 * std::sin(2.0 * M_PI * p[1] / Ly);
        g[1] = t * (2.0 * M_PI * M_PI / (Ly * Ly)) * s * s
                 * std::cos(2.0 * M_PI * p[1] / Ly);
        return g;
    }

    template <int dim>
    inline dealii::Tensor<1, dim> uy_grad(const dealii::Point<dim>& p, double t, double Ly)
    {
        const double s = std::sin(M_PI * p[1] / Ly);
        dealii::Tensor<1, dim> g;
        g[0] = -t * 2.0 * M_PI * M_PI * std::cos(2.0 * M_PI * p[0]) * s * s;
        g[1] = -t * (M_PI * M_PI / Ly) * std::sin(2.0 * M_PI * p[0])
                  * std::sin(2.0 * M_PI * p[1] / Ly);
        return g;
    }


    // ========================================================================
    // Shared building blocks for source terms
    //
    // ∇²ux = t·(2π³/Ly)·cos(2πx)·sin(2πy/Ly)
    //       − t·(4π³/Ly³)·sin²(πx)·sin(2πy/Ly)
    //
    // ∇²uy = t·4π³·sin(2πx)·sin²(πy/Ly)
    //       − t·(2π³/Ly²)·sin(2πx)·(cos²(πy/Ly) − sin²(πy/Ly))
    //
    // ∂p/∂x = −t·π·sin(πx)·cos(πy/Ly)
    // ∂p/∂y = −t·(π/Ly)·cos(πx)·sin(πy/Ly)
    // ========================================================================

    /** @brief Laplacian of velocity at (pt, t). Returns (∇²ux, ∇²uy). */
    template <int dim>
    inline dealii::Tensor<1, dim> laplacian_U(
        const dealii::Point<dim>& pt, double t, double Ly)
    {
        const double pi3 = M_PI * M_PI * M_PI;
        const double x = pt[0], y = pt[1];
        const double sp  = std::sin(M_PI * x);
        const double spy = std::sin(M_PI * y / Ly);
        const double cpy = std::cos(M_PI * y / Ly);
        const double s2x = std::sin(2.0 * M_PI * x);
        const double c2x = std::cos(2.0 * M_PI * x);
        const double s2y = std::sin(2.0 * M_PI * y / Ly);

        dealii::Tensor<1, dim> lap;
        lap[0] = t * (2.0 * pi3 / Ly) * c2x * s2y
               - t * (4.0 * pi3 / (Ly * Ly * Ly)) * sp * sp * s2y;
        lap[1] = t * 4.0 * pi3 * s2x * spy * spy
               - t * (2.0 * pi3 / (Ly * Ly)) * s2x * (cpy * cpy - spy * spy);
        return lap;
    }

    /** @brief Pressure gradient at (pt, t). Returns (∂p/∂x, ∂p/∂y). */
    template <int dim>
    inline dealii::Tensor<1, dim> grad_p(
        const dealii::Point<dim>& pt, double t, double Ly)
    {
        const double x = pt[0], y = pt[1];
        const double sp  = std::sin(M_PI * x);
        const double cp  = std::cos(M_PI * x);
        const double spy = std::sin(M_PI * y / Ly);
        const double cpy = std::cos(M_PI * y / Ly);

        dealii::Tensor<1, dim> gp;
        gp[0] = -t * M_PI * sp * cpy;
        gp[1] = -t * (M_PI / Ly) * cp * spy;
        return gp;
    }

    /** @brief Convection (U·∇)U at (pt, t), same time for both. Used by Phase C. */
    template <int dim>
    inline dealii::Tensor<1, dim> convection_U(
        const dealii::Point<dim>& pt, double t, double Ly)
    {
        const double ux = ux_val<dim>(pt, t, Ly);
        const double uy = uy_val<dim>(pt, t, Ly);
        const auto gux = ux_grad<dim>(pt, t, Ly);
        const auto guy = uy_grad<dim>(pt, t, Ly);

        dealii::Tensor<1, dim> conv;
        conv[0] = ux * gux[0] + uy * gux[1];
        conv[1] = ux * guy[0] + uy * guy[1];
        return conv;
    }


    // ========================================================================
    // Phase A source: Steady Stokes
    //   f_A = −ν∇²U + ∇p       evaluated at t_eval
    // Assembler: assemble_stokes(dt, nu, false, false, &src, t_eval)
    // ========================================================================
    template <int dim>
    inline dealii::Tensor<1, dim> source_phase_A(
        const dealii::Point<dim>& pt, double t_eval,
        double nu, double Ly)
    {
        const auto lap = laplacian_U<dim>(pt, t_eval, Ly);
        const auto gp  = grad_p<dim>(pt, t_eval, Ly);

        dealii::Tensor<1, dim> f;
        f[0] = -nu * lap[0] + gp[0];
        f[1] = -nu * lap[1] + gp[1];
        return f;
    }


    // ========================================================================
    // Phase B source: Unsteady Stokes
    //   f_B = (U^n − U^{n−1})/dt − ν∇²U^n + ∇p^n
    // Assembler: assemble_stokes(dt, nu, true, false, &src, t_new)
    // ========================================================================
    template <int dim>
    inline dealii::Tensor<1, dim> source_phase_B(
        const dealii::Point<dim>& pt, double t_new, double t_old,
        double nu, double Ly)
    {
        const double dt = t_new - t_old;
        const double dux_dt = (ux_val<dim>(pt, t_new, Ly) - ux_val<dim>(pt, t_old, Ly)) / dt;
        const double duy_dt = (uy_val<dim>(pt, t_new, Ly) - uy_val<dim>(pt, t_old, Ly)) / dt;

        const auto lap = laplacian_U<dim>(pt, t_new, Ly);
        const auto gp  = grad_p<dim>(pt, t_new, Ly);

        dealii::Tensor<1, dim> f;
        f[0] = dux_dt - nu * lap[0] + gp[0];
        f[1] = duy_dt - nu * lap[1] + gp[1];
        return f;
    }


    // ========================================================================
    // Phase C source: Steady NS
    //   f_C = (U·∇)U − ν∇²U + ∇p       evaluated at t_eval
    // Assembler: set_old_velocity(exact), assemble_stokes(dt, nu, false, true, &src, t)
    // ========================================================================
    template <int dim>
    inline dealii::Tensor<1, dim> source_phase_C(
        const dealii::Point<dim>& pt, double t_eval,
        double nu, double Ly)
    {
        const auto conv = convection_U<dim>(pt, t_eval, Ly);
        const auto lap  = laplacian_U<dim>(pt, t_eval, Ly);
        const auto gp   = grad_p<dim>(pt, t_eval, Ly);

        dealii::Tensor<1, dim> f;
        f[0] = conv[0] - nu * lap[0] + gp[0];
        f[1] = conv[1] - nu * lap[1] + gp[1];
        return f;
    }


    // ========================================================================
    // Phase D source: Unsteady NS (semi-implicit convection)
    //   f_D = (U^n − U^{n−1})/dt + (U^{n−1}·∇)U^n − ν∇²U^n + ∇p^n
    // Assembler: assemble_stokes(dt, nu, true, true, &src, t_new)
    //
    // NOTE: Original proven-working implementation with inline math.
    //       Kept byte-for-byte to preserve verified correctness.
    // ========================================================================
    template <int dim>
    inline dealii::Tensor<1, dim> source_phase_D(
        const dealii::Point<dim>& pt, double t_new, double t_old,
        double nu, double Ly)
    {
        const double dt = t_new - t_old;
        const double pi3 = M_PI * M_PI * M_PI;
        const double x = pt[0], y = pt[1];
        const double sp = std::sin(M_PI * x), cp = std::cos(M_PI * x);
        const double spy = std::sin(M_PI * y / Ly), cpy = std::cos(M_PI * y / Ly);
        const double s2x = std::sin(2.0 * M_PI * x), c2x = std::cos(2.0 * M_PI * x);
        const double s2y = std::sin(2.0 * M_PI * y / Ly);

        const double dux_dt = (ux_val<dim>(pt, t_new, Ly) - ux_val<dim>(pt, t_old, Ly)) / dt;
        const double duy_dt = (uy_val<dim>(pt, t_new, Ly) - uy_val<dim>(pt, t_old, Ly)) / dt;

        const auto gux_n = ux_grad<dim>(pt, t_new, Ly);
        const auto guy_n = uy_grad<dim>(pt, t_new, Ly);
        const double ux_o = ux_val<dim>(pt, t_old, Ly);
        const double uy_o = uy_val<dim>(pt, t_old, Ly);

        // Laplacians at t_new
        const double lap_ux = t_new * (2.0 * pi3 / Ly) * c2x * s2y
                            - t_new * (4.0 * pi3 / (Ly * Ly * Ly)) * sp * sp * s2y;
        const double lap_uy = t_new * 4.0 * pi3 * s2x * spy * spy
                            - t_new * (2.0 * pi3 / (Ly * Ly)) * s2x * (cpy * cpy - spy * spy);

        // Pressure gradient at t_new
        const double dpx = -t_new * M_PI * sp * cpy;
        const double dpy = -t_new * (M_PI / Ly) * cp * spy;

        dealii::Tensor<1, dim> f;
        f[0] = dux_dt + (ux_o * gux_n[0] + uy_o * gux_n[1]) - nu * lap_ux + dpx;
        f[1] = duy_dt + (ux_o * guy_n[0] + uy_o * guy_n[1]) - nu * lap_uy + dpy;
        return f;
    }
} // namespace NSMMS


// ============================================================================
// Function<dim> wrappers — for initialize_velocity() / set_old_velocity()
// ============================================================================
template <int dim>
class NSMMSInitialUx : public dealii::Function<dim>
{
public:
    NSMMSInitialUx(double t_init, double Ly)
        : dealii::Function<dim>(1), t_(t_init), Ly_(Ly) {}
    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    { return NSMMS::ux_val<dim>(p, t_, Ly_); }
private:
    double t_, Ly_;
};

template <int dim>
class NSMMSInitialUy : public dealii::Function<dim>
{
public:
    NSMMSInitialUy(double t_init, double Ly)
        : dealii::Function<dim>(1), t_(t_init), Ly_(Ly) {}
    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    { return NSMMS::uy_val<dim>(p, t_, Ly_); }
private:
    double t_, Ly_;
};


// ============================================================================
// Function<dim> wrapper for exact pressure (for interpolation/diagnostics)
// ============================================================================
template <int dim>
class NSMMSExactP : public dealii::Function<dim>
{
public:
    NSMMSExactP(double t, double Ly)
        : dealii::Function<dim>(1), t_(t), Ly_(Ly) {}
    double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    { return NSMMS::p_val<dim>(p, t_, Ly_); }
private:
    double t_, Ly_;
};


// ============================================================================
// MMS error structure — all norms + pressure diagnostics
//
// Everything computed in a single unified 2-pass call to
// compute_ns_mms_errors().  No separate diagnostics function needed.
// ============================================================================
struct NSMMSErrors
{
    // --- Velocity L2 and H1 seminorms ---
    double ux_L2 = 0, ux_H1 = 0;
    double uy_L2 = 0, uy_H1 = 0;

    // --- Velocity L∞ ---
    double ux_Linf = 0, uy_Linf = 0;

    // --- Pressure (mean-corrected) ---
    double p_L2   = 0;
    double p_Linf = 0;

    // --- Divergence ---
    double div_U_L2 = 0;

    // --- Pressure diagnostics (computed alongside errors) ---
    double p_mean_h     = 0;   // ∫p_h dx / |Ω|
    double p_mean_exact = 0;   // ∫p*  dx / |Ω|
    double p_offset     = 0;   // p_mean_h − p_mean_exact
    double p_min_h      = 0, p_max_h     = 0;
    double p_min_exact  = 0, p_max_exact = 0;
    double p_raw_L2     = 0;   // ||p_h − p*||_L2 (no mean correction)
    double p_cell_range_max = 0;  // max cell-local pressure range

    /**
     * @brief Print pressure diagnostics to stdout (rank 0 only).
     *
     * Distinguishes:
     *   (a) constant offset: p_range ≈ exact range, large offset
     *   (b) spatially varying garbage: p_range >> exact range
     *   (c) error computation bug: stats look fine but L2 is wrong
     */
    void print_pressure_diagnostics(unsigned int refinement, int rank) const
    {
        if (rank != 0) return;

        const double p_range_h = p_max_h - p_min_h;
        const double pe_range  = p_max_exact - p_min_exact;

        std::cout << std::scientific << std::setprecision(3);
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "NUMERICAL: min=" << p_min_h
                  << ", max=" << p_max_h
                  << ", mean=" << p_mean_h
                  << ", range=" << p_range_h << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "EXACT:     min=" << p_min_exact
                  << ", max=" << p_max_exact
                  << ", mean=" << p_mean_exact
                  << ", range=" << pe_range << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "||p_h - p*||_L2 (raw)            = " << p_raw_L2 << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "||p_h - p*||_L2 (mean-corrected) = " << p_L2 << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "||p_h - p*||_Linf (mean-corr)    = " << p_Linf << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "max cell-local p range = " << p_cell_range_max << "\n";
        std::cout << "  [P-DIAG ref " << refinement << "] "
                  << "offset = " << std::showpos << p_offset
                  << std::noshowpos << "\n";
        std::cout << std::defaultfloat;
    }
};


// ============================================================================
// Unified error computation
//
// Takes NSSubsystem facade reference directly (no raw DoFHandler unpacking).
// Ghosted vectors must be up-to-date (call ns.update_ghosts() first).
//
// Two-pass algorithm:
//   Pass 1: pressure means + min/max (needed for mean-corrected norms)
//   Pass 2: all L2, H1, L∞ errors + pressure diagnostics
// ============================================================================
template <int dim>
NSMMSErrors compute_ns_mms_errors(
    const NSSubsystem<dim>& ns,
    double time, double Ly, MPI_Comm comm)
{
    const auto& ux_dh  = ns.get_ux_dof_handler();
    const auto& uy_dh  = ns.get_uy_dof_handler();
    const auto& p_dh   = ns.get_p_dof_handler();
    const auto& ux_sol = ns.get_ux_relevant();
    const auto& uy_sol = ns.get_uy_relevant();
    const auto& p_sol  = ns.get_p_relevant();

    dealii::QGauss<dim> quad(ux_dh.get_fe().degree + 2);
    const unsigned int nq = quad.size();

    dealii::FEValues<dim> fv_ux(ux_dh.get_fe(), quad,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fv_uy(uy_dh.get_fe(), quad,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> fv_p(p_dh.get_fe(), quad,
        dealii::update_values | dealii::update_quadrature_points |
        dealii::update_JxW_values);

    std::vector<double> ux_v(nq), uy_v(nq), p_v(nq);
    std::vector<dealii::Tensor<1, dim>> gux(nq), guy(nq);

    // ====================================================================
    // Pass 1: pressure means + min/max (for mean correction)
    // ====================================================================
    double loc_p_int = 0, loc_pe_int = 0, loc_vol = 0;
    double loc_p_min = +1e30, loc_p_max = -1e30;
    double loc_pe_min = +1e30, loc_pe_max = -1e30;

    {
        auto uc = ux_dh.begin_active();
        auto pc = p_dh.begin_active();
        for (; uc != ux_dh.end(); ++uc, ++pc)
        {
            if (!uc->is_locally_owned()) continue;
            fv_p.reinit(pc);
            fv_p.get_function_values(p_sol, p_v);
            for (unsigned int q = 0; q < nq; ++q)
            {
                const double w  = fv_p.JxW(q);
                const double ph = p_v[q];
                const double pe = NSMMS::p_val<dim>(
                    fv_p.quadrature_point(q), time, Ly);

                loc_p_int  += ph * w;
                loc_pe_int += pe * w;
                loc_vol    += w;
                loc_p_min   = std::min(loc_p_min, ph);
                loc_p_max   = std::max(loc_p_max, ph);
                loc_pe_min  = std::min(loc_pe_min, pe);
                loc_pe_max  = std::max(loc_pe_max, pe);
            }
        }
    }

    double g_p_int, g_pe_int, g_vol;
    double g_p_min, g_p_max, g_pe_min, g_pe_max;
    MPI_Allreduce(&loc_p_int,  &g_p_int,  1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&loc_pe_int, &g_pe_int, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&loc_vol,    &g_vol,    1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&loc_p_min,  &g_p_min,  1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&loc_p_max,  &g_p_max,  1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&loc_pe_min, &g_pe_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&loc_pe_max, &g_pe_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    const double p_mean  = g_p_int  / g_vol;
    const double ep_mean = g_pe_int / g_vol;

    // ====================================================================
    // Pass 2: all errors (L2, H1, L∞, divergence, pressure diagnostics)
    // ====================================================================
    // L2 accumulators (SUM reduction)
    double lux2 = 0, luxh = 0, luy2 = 0, luyh = 0;
    double lp2 = 0, lp_raw2 = 0, ldiv = 0;

    // L∞ accumulators (MAX reduction)
    double lux_inf = 0, luy_inf = 0, lp_inf = 0;

    // Pressure cell range (MAX reduction)
    double loc_cell_range_max = 0;

    {
        auto uc = ux_dh.begin_active();
        auto yc = uy_dh.begin_active();
        auto pc = p_dh.begin_active();
        for (; uc != ux_dh.end(); ++uc, ++yc, ++pc)
        {
            if (!uc->is_locally_owned()) continue;
            fv_ux.reinit(uc);  fv_uy.reinit(yc);  fv_p.reinit(pc);
            fv_ux.get_function_values(ux_sol, ux_v);
            fv_ux.get_function_gradients(ux_sol, gux);
            fv_uy.get_function_values(uy_sol, uy_v);
            fv_uy.get_function_gradients(uy_sol, guy);
            fv_p.get_function_values(p_sol, p_v);

            double cell_p_min = +1e30, cell_p_max = -1e30;

            for (unsigned int q = 0; q < nq; ++q)
            {
                const double w = fv_ux.JxW(q);
                const auto& xq = fv_ux.quadrature_point(q);

                // --- Velocity errors ---
                const double eu = ux_v[q] - NSMMS::ux_val<dim>(xq, time, Ly);
                const double ev = uy_v[q] - NSMMS::uy_val<dim>(xq, time, Ly);

                lux2    += eu * eu * w;
                luy2    += ev * ev * w;
                lux_inf  = std::max(lux_inf, std::abs(eu));
                luy_inf  = std::max(luy_inf, std::abs(ev));

                // --- Velocity gradient errors (H1 seminorm) ---
                const auto geu = gux[q] - NSMMS::ux_grad<dim>(xq, time, Ly);
                const auto gev = guy[q] - NSMMS::uy_grad<dim>(xq, time, Ly);
                luxh += geu * geu * w;
                luyh += gev * gev * w;

                // --- Divergence ---
                const double d = gux[q][0] + guy[q][1];
                ldiv += d * d * w;

                // --- Pressure errors (mean-corrected + raw) ---
                const double ph = p_v[q];
                const double pe = NSMMS::p_val<dim>(xq, time, Ly);
                const double ep_mc  = (ph - p_mean) - (pe - ep_mean);
                const double ep_raw = ph - pe;

                lp2     += ep_mc  * ep_mc  * w;
                lp_raw2 += ep_raw * ep_raw * w;
                lp_inf   = std::max(lp_inf, std::abs(ep_mc));

                cell_p_min = std::min(cell_p_min, ph);
                cell_p_max = std::max(cell_p_max, ph);
            }

            loc_cell_range_max = std::max(loc_cell_range_max,
                                          cell_p_max - cell_p_min);
        }
    }

    // --- Global reductions ---
    double loc_sum[7] = {lux2, luxh, luy2, luyh, lp2, lp_raw2, ldiv};
    double glb_sum[7];
    MPI_Allreduce(loc_sum, glb_sum, 7, MPI_DOUBLE, MPI_SUM, comm);

    double loc_max[4] = {lux_inf, luy_inf, lp_inf, loc_cell_range_max};
    double glb_max[4];
    MPI_Allreduce(loc_max, glb_max, 4, MPI_DOUBLE, MPI_MAX, comm);

    // --- Pack results ---
    NSMMSErrors e;

    e.ux_L2    = std::sqrt(glb_sum[0]);
    e.ux_H1    = std::sqrt(glb_sum[1]);
    e.uy_L2    = std::sqrt(glb_sum[2]);
    e.uy_H1    = std::sqrt(glb_sum[3]);
    e.p_L2     = std::sqrt(glb_sum[4]);
    e.div_U_L2 = std::sqrt(glb_sum[6]);

    e.ux_Linf  = glb_max[0];
    e.uy_Linf  = glb_max[1];
    e.p_Linf   = glb_max[2];

    e.p_mean_h         = p_mean;
    e.p_mean_exact     = ep_mean;
    e.p_offset         = p_mean - ep_mean;
    e.p_min_h          = g_p_min;
    e.p_max_h          = g_p_max;
    e.p_min_exact      = g_pe_min;
    e.p_max_exact      = g_pe_max;
    e.p_raw_L2         = std::sqrt(glb_sum[5]);
    e.p_cell_range_max = glb_max[3];

    return e;
}

#endif // NAVIER_STOKES_MMS_H