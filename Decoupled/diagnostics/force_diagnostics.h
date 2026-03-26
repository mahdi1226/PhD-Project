// ============================================================================
// diagnostics/force_diagnostics.h — Force Magnitude Diagnostics (Parallel)
//
// Computes L2 norms and pointwise maxima of the three body forces in NS:
//   - Capillary force:  F_cap = θ ∇ψ               (Zhang Eq 3.11)
//   - Kelvin force:     F_kel = μ₀[(M·∇)H + ½(∇·M)H]  (Zhang Eq 2.3/3.11)
//   - Gravity force:    F_grav = ρ(θ) g             (Zhang Eq 3.11)
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
//
// NOTE: The Kelvin diagnostic computes the cell-interior part of the skew
// form B_h^m (Eq. 38 line 1 from Nochetto). The face-integral correction
// (line 2) is excluded from diagnostics since it's a discrete consistency
// term, not a physical force.
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef FORCE_DIAGNOSTICS_H
#define FORCE_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "physics/kelvin_force.h"
#include "physics/material_properties.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <mpi.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// Force Diagnostic Data
// ============================================================================
struct ForceDiagnostics
{
    double F_cap_L2  = 0.0;   // ||F_cap||_L2
    double F_mag_L2  = 0.0;   // ||F_kel||_L2
    double F_grav_L2 = 0.0;   // ||F_grav||_L2

    double F_cap_max  = 0.0;  // max|F_cap|
    double F_mag_max  = 0.0;  // max|F_kel|
    double F_grav_max = 0.0;  // max|F_grav|
};

// ============================================================================
// Compute force diagnostics (parallel, Zhang formulation)
//
// Requires synchronized DoFHandlers on the same triangulation.
// Uses the same multi-DoFHandler iteration pattern as the NS assembler
// (navier_stokes_assemble.cc, lines 360-370).
// ============================================================================
template <int dim>
ForceDiagnostics compute_force_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>& mag_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    const Parameters& params,
    MPI_Comm comm)
{
    const double mu0 = params.physics.mu_0;
    const double eps = params.physics.epsilon;

    // Gravity vector
    dealii::Tensor<1, dim> gravity;
    if (params.enable_gravity)
    {
        for (unsigned int d = 0; d < dim; ++d)
            gravity[d] = params.physics.gravity_magnitude
                       * params.physics.gravity_direction[d];
    }

    // FEValues for each DoFHandler
    const auto& theta_fe = theta_dof_handler.get_fe();
    const auto& phi_fe   = phi_dof_handler.get_fe();
    const auto& mag_fe   = mag_dof_handler.get_fe();

    // Use highest degree + 1 for quadrature
    const unsigned int quad_degree = std::max({theta_fe.degree, phi_fe.degree, mag_fe.degree}) + 1;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    // θ needs values + gradients (for ψ)
    dealii::FEValues<dim> theta_fv(theta_fe, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    // φ needs gradients (H = ∇φ) + hessians (∇H for Kelvin)
    dealii::FEValues<dim> phi_fv(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_hessians);

    // M needs values only (Kelvin uses M and Hessian of φ)
    dealii::FEValues<dim> mag_fv(mag_fe, quadrature,
        dealii::update_values);

    // Scratch arrays
    std::vector<double>                    theta_vals(n_q);
    std::vector<dealii::Tensor<1, dim>>    psi_grads(n_q);
    std::vector<dealii::Tensor<1, dim>>    phi_grads(n_q);
    std::vector<dealii::Tensor<2, dim>>    phi_hessians(n_q);
    std::vector<double>                    Mx_vals(n_q);
    std::vector<double>                    My_vals(n_q);

    // Local accumulators
    double local_cap_sq = 0.0, local_mag_sq = 0.0, local_grav_sq = 0.0;
    double local_cap_max = 0.0, local_mag_max = 0.0, local_grav_max = 0.0;

    // Synchronized cell iteration across all DoFHandlers
    auto theta_cell = theta_dof_handler.begin_active();
    auto phi_cell   = phi_dof_handler.begin_active();
    auto mag_cell   = mag_dof_handler.begin_active();

    for (; theta_cell != theta_dof_handler.end();
         ++theta_cell, ++phi_cell, ++mag_cell)
    {
        if (!theta_cell->is_locally_owned())
            continue;

        theta_fv.reinit(theta_cell);
        phi_fv.reinit(phi_cell);
        mag_fv.reinit(mag_cell);

        // Extract field values at quadrature points
        theta_fv.get_function_values(theta_relevant, theta_vals);
        theta_fv.get_function_gradients(psi_relevant, psi_grads);
        phi_fv.get_function_gradients(phi_relevant, phi_grads);
        phi_fv.get_function_hessians(phi_relevant, phi_hessians);
        mag_fv.get_function_values(Mx_relevant, Mx_vals);
        mag_fv.get_function_values(My_relevant, My_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW   = theta_fv.JxW(q);
            const double theta = theta_vals[q];

            // ==============================================================
            // Capillary force: F_cap = θ · ∇ψ  (Zhang Eq 3.11)
            // ==============================================================
            dealii::Tensor<1, dim> F_cap = theta * psi_grads[q];

            const double F_cap_mag = F_cap.norm();
            local_cap_sq  += F_cap_mag * F_cap_mag * JxW;
            local_cap_max  = std::max(local_cap_max, F_cap_mag);

            // ==============================================================
            // Kelvin force: μ₀[(M·∇)H + ½(∇·M)H]
            // Cell-interior part of DG skew form B_h^m (Eq. 38, line 1)
            // ==============================================================
            if (params.enable_magnetic)
            {
                dealii::Tensor<1, dim> M;
                M[0] = Mx_vals[q];
                M[1] = My_vals[q];

                // (M·∇)H from Hessian of φ
                const dealii::Tensor<1, dim> M_grad_H =
                    KelvinForce::compute_M_grad_H<dim>(M, phi_hessians[q]);

                // Kelvin: μ₀(M·∇)H
                dealii::Tensor<1, dim> F_kel;
                for (unsigned int d = 0; d < dim; ++d)
                    F_kel[d] = mu0 * M_grad_H[d];

                const double F_kel_mag = F_kel.norm();
                local_mag_sq  += F_kel_mag * F_kel_mag * JxW;
                local_mag_max  = std::max(local_mag_max, F_kel_mag);
            }

            // ==============================================================
            // Gravity force: F_grav = ρ(θ) g
            // ==============================================================
            if (params.enable_gravity)
            {
                const double rho = density_ratio(theta, eps, params.physics.r);

                dealii::Tensor<1, dim> F_grav = rho * gravity;

                const double F_grav_mag = F_grav.norm();
                local_grav_sq  += F_grav_mag * F_grav_mag * JxW;
                local_grav_max  = std::max(local_grav_max, F_grav_mag);
            }
        }
    }

    // MPI reductions
    ForceDiagnostics result;

    double global_cap_sq  = local_cap_sq;
    double global_mag_sq  = local_mag_sq;
    double global_grav_sq = local_grav_sq;

    MPI_Allreduce(MPI_IN_PLACE, &global_cap_sq,  1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_mag_sq,  1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_grav_sq, 1, MPI_DOUBLE, MPI_SUM, comm);

    result.F_cap_L2  = std::sqrt(global_cap_sq);
    result.F_mag_L2  = std::sqrt(global_mag_sq);
    result.F_grav_L2 = std::sqrt(global_grav_sq);

    double global_cap_max  = local_cap_max;
    double global_mag_max  = local_mag_max;
    double global_grav_max = local_grav_max;

    MPI_Allreduce(MPI_IN_PLACE, &global_cap_max,  1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_mag_max,  1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &global_grav_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    result.F_cap_max  = global_cap_max;
    result.F_mag_max  = global_mag_max;
    result.F_grav_max = global_grav_max;

    return result;
}

#endif // FORCE_DIAGNOSTICS_H
