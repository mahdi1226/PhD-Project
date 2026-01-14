// ============================================================================
// diagnostics/force_diagnostics.h - Force Magnitude Diagnostics (Parallel)
//
// Computes L2 norms of the three force terms in Navier-Stokes:
//   - Capillary force: F_cap = (λ/ε) θ ∇ψ
//   - Kelvin force:    F_mag = (μ₀/2) ∇(χ(θ)|H|²)  [Paper Eq. 36]
//   - Gravity force:   F_grav = ρ(θ) g
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef FORCE_DIAGNOSTICS_H
#define FORCE_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>
#include <algorithm>

// ============================================================================
// Force Diagnostic Data
// ============================================================================
struct ForceDiagnostics
{
    double F_cap_L2 = 0.0;   // ||F_cap||_L2
    double F_mag_L2 = 0.0;   // ||F_mag||_L2
    double F_grav_L2 = 0.0;  // ||F_grav||_L2

    double F_cap_max = 0.0;  // max|F_cap|
    double F_mag_max = 0.0;  // max|F_mag|
    double F_grav_max = 0.0; // max|F_grav|
};

// ============================================================================
// Helper: Smooth Heaviside function
// ============================================================================
namespace force_detail
{
    inline double sigmoid(double x)
    {
        if (x > 20.0) return 1.0;
        if (x < -20.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }
}

// ============================================================================
// Compute force diagnostics (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
ForceDiagnostics compute_force_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const dealii::TrilinosWrappers::MPI::Vector* phi_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const double eps = params.physics.epsilon;
    const double lam = params.physics.lambda;
    const double chi0 = params.physics.chi_0;
    const double g_val = params.enable_gravity ? params.physics.gravity : 0.0;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    // Local accumulators
    double local_F_cap_sq = 0.0;
    double local_F_mag_sq = 0.0;
    double local_F_grav_sq = 0.0;
    double local_F_cap_max = 0.0;
    double local_F_mag_max = 0.0;
    double local_F_grav_max = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);
        fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (phi_solution != nullptr)
            fe_values.get_function_gradients(*phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double JxW = fe_values.JxW(q);

            // ================================================================
            // Capillary force: F_cap = (λ/ε) θ ∇ψ
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            for (unsigned int d = 0; d < dim; ++d)
                F_cap[d] = (lam / eps) * theta * psi_gradients[q][d];

            const double F_cap_mag = F_cap.norm();
            local_F_cap_sq += F_cap_mag * F_cap_mag * JxW;
            local_F_cap_max = std::max(local_F_cap_max, F_cap_mag);

            // ================================================================
            // Kelvin force: F_mag ≈ (μ₀/2) χ'(θ) |H|² ∇θ
            // ================================================================
            if (phi_solution != nullptr)
            {
                dealii::Tensor<1, dim> H = phi_gradients[q];
                const double H_sq = H.norm_square();

                // χ'(θ) = (χ₀/ε) * H(θ/ε) * (1 - H(θ/ε))
                const double H_sigmoid = force_detail::sigmoid(theta / eps);
                const double chi_prime = (chi0 / eps) * H_sigmoid * (1.0 - H_sigmoid);

                dealii::Tensor<1, dim> F_mag;
                for (unsigned int d = 0; d < dim; ++d)
                    F_mag[d] = 0.5 * params.physics.mu_0 * chi_prime * H_sq * theta_gradients[q][d];

                const double F_mag_mag = F_mag.norm();
                local_F_mag_sq += F_mag_mag * F_mag_mag * JxW;
                local_F_mag_max = std::max(local_F_mag_max, F_mag_mag);
            }

            // ================================================================
            // Gravity force: F_grav = ρ(θ) g
            // ================================================================
            if (params.enable_gravity)
            {
                const double H_val = force_detail::sigmoid(theta / eps);
                const double rho = 1.0 + params.physics.r * H_val;

                dealii::Tensor<1, dim> F_grav;
                F_grav[0] = 0.0;
                F_grav[1] = -rho * g_val;

                const double F_grav_mag = F_grav.norm();
                local_F_grav_sq += F_grav_mag * F_grav_mag * JxW;
                local_F_grav_max = std::max(local_F_grav_max, F_grav_mag);
            }
        }
    }

    // MPI reductions
    ForceDiagnostics result;

    double global_F_cap_sq = MPIUtils::reduce_sum(local_F_cap_sq, comm);
    double global_F_mag_sq = MPIUtils::reduce_sum(local_F_mag_sq, comm);
    double global_F_grav_sq = MPIUtils::reduce_sum(local_F_grav_sq, comm);

    result.F_cap_L2 = std::sqrt(global_F_cap_sq);
    result.F_mag_L2 = std::sqrt(global_F_mag_sq);
    result.F_grav_L2 = std::sqrt(global_F_grav_sq);

    result.F_cap_max = MPIUtils::reduce_max(local_F_cap_max, comm);
    result.F_mag_max = MPIUtils::reduce_max(local_F_mag_max, comm);
    result.F_grav_max = MPIUtils::reduce_max(local_F_grav_max, comm);

    return result;
}

// ============================================================================
// Compute force diagnostics (serial version with deal.II vectors)
// ============================================================================
template <int dim>
ForceDiagnostics compute_force_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params)
{
    ForceDiagnostics result;

    const double eps = params.physics.epsilon;
    const double lam = params.physics.lambda;
    const double chi0 = params.physics.chi_0;
    const double g_val = params.enable_gravity ? params.physics.gravity : 0.0;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    double F_cap_sq = 0.0;
    double F_mag_sq = 0.0;
    double F_grav_sq = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);
        fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (phi_solution != nullptr)
            fe_values.get_function_gradients(*phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double JxW = fe_values.JxW(q);

            // Capillary force
            dealii::Tensor<1, dim> F_cap;
            for (unsigned int d = 0; d < dim; ++d)
                F_cap[d] = (lam / eps) * theta * psi_gradients[q][d];

            const double F_cap_mag = F_cap.norm();
            F_cap_sq += F_cap_mag * F_cap_mag * JxW;
            result.F_cap_max = std::max(result.F_cap_max, F_cap_mag);

            // Kelvin force
            if (phi_solution != nullptr)
            {
                dealii::Tensor<1, dim> H = phi_gradients[q];
                const double H_sq = H.norm_square();

                const double H_sigmoid = force_detail::sigmoid(theta / eps);
                const double chi_prime = (chi0 / eps) * H_sigmoid * (1.0 - H_sigmoid);

                dealii::Tensor<1, dim> F_mag;
                for (unsigned int d = 0; d < dim; ++d)
                    F_mag[d] = 0.5 * params.physics.mu_0 * chi_prime * H_sq * theta_gradients[q][d];

                const double F_mag_mag = F_mag.norm();
                F_mag_sq += F_mag_mag * F_mag_mag * JxW;
                result.F_mag_max = std::max(result.F_mag_max, F_mag_mag);
            }

            // Gravity force
            if (params.enable_gravity)
            {
                const double H_val = force_detail::sigmoid(theta / eps);
                const double rho = 1.0 + params.physics.r * H_val;

                dealii::Tensor<1, dim> F_grav;
                F_grav[0] = 0.0;
                F_grav[1] = -rho * g_val;

                const double F_grav_mag = F_grav.norm();
                F_grav_sq += F_grav_mag * F_grav_mag * JxW;
                result.F_grav_max = std::max(result.F_grav_max, F_grav_mag);
            }
        }
    }

    result.F_cap_L2 = std::sqrt(F_cap_sq);
    result.F_mag_L2 = std::sqrt(F_mag_sq);
    result.F_grav_L2 = std::sqrt(F_grav_sq);

    return result;
}

#endif // FORCE_DIAGNOSTICS_H