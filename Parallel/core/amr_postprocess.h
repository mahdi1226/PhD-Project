// ============================================================================
// amr/amr_postprocess.h - Clean AMR Post-Processing for Phase Field
//
// After SolutionTransfer, the interpolated solutions violate physical
// constraints and constitutive relations. This module fixes them.
//
// Usage in refine_mesh():
//   // ... after SolutionTransfer and constraints.distribute() ...
//   AMRPostProcessor<dim> postproc(params_);
//   postproc.process(theta_solution_, theta_old_solution_,
//                    psi_solution_, theta_dof_handler_,
//                    psi_dof_handler_, theta_constraints_,
//                    psi_constraints_);
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef AMR_POSTPROCESS_H
#define AMR_POSTPROCESS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include "utilities/parameters.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * @brief Diagnostic data from AMR post-processing
 */
struct AMRPostProcessData
{
    // Before corrections
    double theta_min_raw = 0.0;
    double theta_max_raw = 0.0;
    double theta_old_min_raw = 0.0;
    double theta_old_max_raw = 0.0;

    // After corrections
    double theta_min_fixed = 0.0;
    double theta_max_fixed = 0.0;

    // Clamping stats
    unsigned int n_theta_clamped = 0;
    unsigned int n_theta_old_clamped = 0;

    // Psi reprojection
    unsigned int psi_cg_iterations = 0;
    double psi_residual = 0.0;

    // Flags
    bool had_violations = false;

    void print() const
    {
        std::cout << "[AMR Post-Process] Results:\n";
        std::cout << "  θ raw:     [" << std::setprecision(6) << theta_min_raw
                  << ", " << theta_max_raw << "]";
        if (had_violations)
            std::cout << " *** VIOLATED ***";
        std::cout << "\n";

        std::cout << "  θ_old raw: [" << theta_old_min_raw
                  << ", " << theta_old_max_raw << "]\n";

        if (n_theta_clamped > 0 || n_theta_old_clamped > 0)
        {
            std::cout << "  Clamped:   " << n_theta_clamped << " θ DoFs, "
                      << n_theta_old_clamped << " θ_old DoFs\n";
        }

        std::cout << "  θ fixed:   [" << theta_min_fixed
                  << ", " << theta_max_fixed << "]\n";
        std::cout << "  ψ reproject: " << psi_cg_iterations << " CG iters\n";
    }
};

/**
 * @brief AMR post-processor for Cahn-Hilliard variables
 *
 * Performs three critical corrections after SolutionTransfer:
 * 1. Clamp θ to [-1, 1] to prevent W'(θ) explosion
 * 2. Clamp θ_old similarly
 * 3. Reproject ψ from θ to restore constitutive relation ψ = W'(θ) - ε²Δθ
 */
template <int dim>
class AMRPostProcessor
{
public:
    explicit AMRPostProcessor(const Parameters& params, bool verbose = true)
        : params_(params)
        , verbose_(verbose)
    {}

    /**
     * @brief Main entry point - run all post-processing steps
     */
    AMRPostProcessData process(
        dealii::Vector<double>& theta_solution,
        dealii::Vector<double>& theta_old_solution,
        dealii::Vector<double>& psi_solution,
        const dealii::DoFHandler<dim>& theta_dof_handler,
        const dealii::DoFHandler<dim>& psi_dof_handler,
        const dealii::AffineConstraints<double>& theta_constraints,
        const dealii::AffineConstraints<double>& psi_constraints)
    {
        AMRPostProcessData data;

        // =====================================================================
        // Step 1: Record raw bounds (diagnostic)
        // =====================================================================
        data.theta_min_raw = *std::min_element(theta_solution.begin(), theta_solution.end());
        data.theta_max_raw = *std::max_element(theta_solution.begin(), theta_solution.end());
        data.theta_old_min_raw = *std::min_element(theta_old_solution.begin(), theta_old_solution.end());
        data.theta_old_max_raw = *std::max_element(theta_old_solution.begin(), theta_old_solution.end());

        data.had_violations = (data.theta_min_raw < -1.001 || data.theta_max_raw > 1.001 ||
                               data.theta_old_min_raw < -1.001 || data.theta_old_max_raw > 1.001);

        // =====================================================================
        // Step 2: Clamp θ to [-1, 1]
        //
        // WHY: Interpolation causes overshoot. For |θ| > 1:
        //   W'(θ) = θ³ - θ grows rapidly
        //   θ = 1.5 → W'(θ) = 1.875 (bad)
        //   θ = 2.0 → W'(θ) = 6.0   (explosive)
        // =====================================================================
        data.n_theta_clamped = clamp_vector(theta_solution, -1.0, 1.0);
        data.n_theta_old_clamped = clamp_vector(theta_old_solution, -1.0, 1.0);

        // Re-apply constraints after clamping
        theta_constraints.distribute(theta_solution);
        theta_constraints.distribute(theta_old_solution);

        // Record fixed bounds
        data.theta_min_fixed = *std::min_element(theta_solution.begin(), theta_solution.end());
        data.theta_max_fixed = *std::max_element(theta_solution.begin(), theta_solution.end());

        // =====================================================================
        // Step 3: Reproject ψ from θ
        //
        // The constitutive relation ψ = W'(θ) - ε²Δθ is violated after
        // interpolation. Solve: (ψ, v) = (W'(θ), v) + ε²(∇θ, ∇v)
        // =====================================================================
        reproject_psi(theta_solution, psi_solution,
                      theta_dof_handler, psi_dof_handler,
                      psi_constraints,
                      data.psi_cg_iterations, data.psi_residual);

        if (verbose_)
            data.print();

        return data;
    }

private:
    const Parameters& params_;
    bool verbose_;

    /**
     * @brief Clamp all values in vector to [min_val, max_val]
     * @return Number of values that were clamped
     */
    static unsigned int clamp_vector(dealii::Vector<double>& vec,
                                     double min_val, double max_val)
    {
        unsigned int count = 0;
        for (unsigned int i = 0; i < vec.size(); ++i)
        {
            if (vec[i] < min_val)
            {
                vec[i] = min_val;
                ++count;
            }
            else if (vec[i] > max_val)
            {
                vec[i] = max_val;
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Reproject ψ from θ via L² projection
     *
     * Solves: M ψ = b where
     *   M_ij = (φ_j, φ_i)
     *   b_i  = (W'(θ), φ_i) + ε²(∇θ, ∇φ_i)
     */
    void reproject_psi(
        const dealii::Vector<double>& theta_solution,
        dealii::Vector<double>& psi_solution,
        const dealii::DoFHandler<dim>& theta_dof_handler,
        const dealii::DoFHandler<dim>& psi_dof_handler,
        const dealii::AffineConstraints<double>& psi_constraints,
        unsigned int& cg_iterations,
        double& final_residual)
    {
        const double epsilon = params_.physics.epsilon;
        const double eps2 = epsilon * epsilon;

        const auto& fe = psi_dof_handler.get_fe();
        const unsigned int n_dofs = psi_dof_handler.n_dofs();
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        dealii::QGauss<dim> quadrature(fe.degree + 1);
        const unsigned int n_q_points = quadrature.size();

        dealii::FEValues<dim> fe_values(fe, quadrature,
            dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

        // Build sparsity pattern
        dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);
        dealii::DoFTools::make_sparsity_pattern(psi_dof_handler, dsp, psi_constraints, false);
        dealii::SparsityPattern sparsity;
        sparsity.copy_from(dsp);

        // Allocate matrix and vectors
        dealii::SparseMatrix<double> mass_matrix(sparsity);
        dealii::Vector<double> rhs(n_dofs);
        dealii::Vector<double> psi_new(n_dofs);

        // Local storage
        dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        dealii::Vector<double> cell_rhs(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<double> theta_values(n_q_points);
        std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

        // Assemble
        for (const auto& cell : psi_dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell_matrix = 0;
            cell_rhs = 0;
            cell->get_dof_indices(local_dof_indices);

            // Get θ values at quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                theta_values[q] = 0.0;
                theta_gradients[q] = 0.0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    theta_values[q] += theta_solution[local_dof_indices[i]] *
                                       fe_values.shape_value(i, q);
                    theta_gradients[q] += theta_solution[local_dof_indices[i]] *
                                          fe_values.shape_grad(i, q);
                }
            }

            // Local assembly
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const double JxW = fe_values.JxW(q);
                const double theta_q = theta_values[q];
                const auto& grad_theta_q = theta_gradients[q];

                // W'(θ) = θ³ - θ for double-well W(θ) = (1/4)(θ²-1)²
                const double W_prime = theta_q * theta_q * theta_q - theta_q;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const double phi_i = fe_values.shape_value(i, q);
                    const auto& grad_phi_i = fe_values.shape_grad(i, q);

                    // RHS: (W'(θ), φ_i) + ε²(∇θ, ∇φ_i)
                    cell_rhs(i) += (W_prime * phi_i + eps2 * (grad_theta_q * grad_phi_i)) * JxW;

                    // Mass matrix
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        cell_matrix(i, j) += phi_i * fe_values.shape_value(j, q) * JxW;
                    }
                }
            }

            psi_constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices, mass_matrix, rhs);
        }

        // Solve M ψ = rhs
        dealii::SolverControl solver_control(1000, 1e-12 * rhs.l2_norm());
        dealii::SolverCG<dealii::Vector<double>> cg(solver_control);
        dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(mass_matrix, 1.2);

        psi_new = 0;
        cg.solve(mass_matrix, psi_new, rhs, preconditioner);

        psi_constraints.distribute(psi_new);
        psi_solution = psi_new;

        cg_iterations = solver_control.last_step();
        final_residual = solver_control.last_value();
    }
};

#endif // AMR_POSTPROCESS_H