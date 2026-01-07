// ============================================================================
// assembly/magnetization_assembler.h - DG Magnetization Assembler Header
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c, Eq. 57
//
// EQUATION 42c (rearranged):
//   (1/τ + 1/T)(M^k, Z) - B_h^m(U^{k-1}, Z, M^k) = (1/T)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// ============================================================================
#ifndef MAGNETIZATION_ASSEMBLER_H
#define MAGNETIZATION_ASSEMBLER_H

#include "utilities/parameters.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Compute applied magnetic field h_a at a point (2D) [Eq. 97-98]
//
// This belongs here because h_a enters the magnetization equation:
//   M ≈ χ(h_a + h_d) where h_d = ∇φ is the demagnetizing field from Poisson
//
// Poisson only sees ∇·M (demagnetizing), not h_a directly.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_applied_field(
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

        const double delta_sq = 0.04;  // Softening parameter

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

template <int dim>
class MagnetizationAssembler
{
public:
    /// Constructor
    MagnetizationAssembler(
        const Parameters& params,
        const dealii::DoFHandler<dim>& M_dof,
        const dealii::DoFHandler<dim>& U_dof,
        const dealii::DoFHandler<dim>& phi_dof,
        const dealii::DoFHandler<dim>& theta_dof);

    /// Assemble the DG magnetization system
    /// @param system_matrix Output sparse matrix (same for Mx and My)
    /// @param rhs_x Output RHS vector for Mx component
    /// @param rhs_y Output RHS vector for My component
    /// @param Ux x-velocity solution
    /// @param Uy y-velocity solution
    /// @param phi Magnetic potential solution
    /// @param theta Phase field solution
    /// @param Mx_old Previous time step Mx
    /// @param My_old Previous time step My
    /// @param dt Time step size
    /// @param current_time Current time (for MMS source term)
    void assemble(
        dealii::SparseMatrix<double>& system_matrix,
        dealii::Vector<double>& rhs_x,
        dealii::Vector<double>& rhs_y,
        const dealii::Vector<double>& Ux,
        const dealii::Vector<double>& Uy,
        const dealii::Vector<double>& phi,
        const dealii::Vector<double>& theta,
        const dealii::Vector<double>& Mx_old,
        const dealii::Vector<double>& My_old,
        double dt,
        double current_time = 0.0) const;

    /// Assemble only the RHS (for fixed matrix reuse)
    void assemble_rhs_only(
        dealii::Vector<double>& rhs_x,
        dealii::Vector<double>& rhs_y,
        const dealii::Vector<double>& phi,
        const dealii::Vector<double>& theta,
        const dealii::Vector<double>& Mx_old,
        const dealii::Vector<double>& My_old,
        double dt) const;

private:
    const Parameters& params_;
    const dealii::DoFHandler<dim>& M_dof_handler_;
    const dealii::DoFHandler<dim>& U_dof_handler_;
    const dealii::DoFHandler<dim>& phi_dof_handler_;
    const dealii::DoFHandler<dim>& theta_dof_handler_;

    /// Susceptibility function: χ(θ)
    double chi(double theta_val) const;
};

#endif // MAGNETIZATION_ASSEMBLER_H