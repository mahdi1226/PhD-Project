// ============================================================================
// core/nsch_problem_solvers.cc - Solver methods with BlockVector adapters
//
// REFACTORED VERSION: Uses separate scalar solutions internally, but packs
// them into BlockVectors for compatibility with existing assemblers.
//
// FIXED: Added condense() calls before solving to properly apply constraints
//
// Based on: Nochetto, Salgado & Tomas (2016)
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#include "core/nsch_problem.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include "assembly/ch_assembler.h"
#include "assembly/ns_assembler.h"
#include "assembly/poisson_assembler.h"
#include "solvers/poisson_solver.h"
#include "utilities/nsch_mms.h"
#include "utilities/nsch_linear_algebra.h"
#include "utilities/nsch_block_structure.h"

#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Adapter functions: Pack separate scalar vectors into BlockVectors
// ============================================================================

// Pack c_solution and mu_solution into CHVector (BlockVector with 2 blocks)
template <int dim>
CHVector NSCHProblem<dim>::pack_ch_solution(const dealii::Vector<double>& c,
                                             const dealii::Vector<double>& mu) const
{
    CHVector ch_vec(2);
    ch_vec.block(0).reinit(c.size());
    ch_vec.block(1).reinit(mu.size());
    ch_vec.collect_sizes();

    ch_vec.block(0) = c;
    ch_vec.block(1) = mu;
    return ch_vec;
}

// Unpack CHVector back to separate c and mu vectors
template <int dim>
void NSCHProblem<dim>::unpack_ch_solution(const CHVector& ch_vec,
                                           dealii::Vector<double>& c,
                                           dealii::Vector<double>& mu) const
{
    c = ch_vec.block(0);
    mu = ch_vec.block(1);
}

// Pack ux, uy, p into NSVector (BlockVector with 2 blocks: [velocity, pressure])
// Note: The old NS system uses FESystem with [ux, uy] as a vector, then p
// For adapter, we need to interleave ux and uy into velocity block
template <int dim>
NSVector NSCHProblem<dim>::pack_ns_solution(const dealii::Vector<double>& ux,
                                             const dealii::Vector<double>& uy,
                                             const dealii::Vector<double>& p) const
{
    // Old NS uses FESystem<dim>(FE_Q(2), dim, FE_Q(1), 1) with component-wise renumbering
    // After renumbering: block 0 = all velocity dofs (ux1, uy1, ux2, uy2, ...), block 1 = pressure
    // This is NOT what we have (we have separate ux, uy vectors)

    // Since our assembler expects FESystem-based solution, we need to create compatible format
    // The FESystem has (dim + 1) components with component-wise renumbering
    // Block 0: size = n_ux + n_uy (interleaved per component renumbering)
    // Block 1: size = n_p

    // Actually, after DoFRenumbering::component_wise, the DoFs are ordered by component:
    // [all ux DoFs | all uy DoFs | all p DoFs]

    NSVector ns_vec(2);
    ns_vec.block(0).reinit(ux.size() + uy.size());  // velocity block
    ns_vec.block(1).reinit(p.size());                // pressure block
    ns_vec.collect_sizes();

    // Copy ux and uy into velocity block
    // With component_wise renumbering: first all ux, then all uy
    for (unsigned int i = 0; i < ux.size(); ++i)
        ns_vec.block(0)[i] = ux[i];
    for (unsigned int i = 0; i < uy.size(); ++i)
        ns_vec.block(0)[ux.size() + i] = uy[i];

    ns_vec.block(1) = p;
    return ns_vec;
}

// Unpack NSVector back to separate ux, uy, p vectors
template <int dim>
void NSCHProblem<dim>::unpack_ns_solution(const NSVector& ns_vec,
                                           dealii::Vector<double>& ux,
                                           dealii::Vector<double>& uy,
                                           dealii::Vector<double>& p) const
{
    // Reverse of pack: velocity block has [all ux | all uy]
    for (unsigned int i = 0; i < ux.size(); ++i)
        ux[i] = ns_vec.block(0)[i];
    for (unsigned int i = 0; i < uy.size(); ++i)
        uy[i] = ns_vec.block(0)[ux.size() + i];

    p = ns_vec.block(1);
}

// ============================================================================
// Dipole potential BC function (needed for Poisson)
// ============================================================================
template <int dim>
class DipolePotentialBC : public dealii::Function<dim>
{
public:
    static constexpr int n_dipoles = 5;
    double dipole_x[n_dipoles] = {-0.5, 0.0, 0.5, 1.0, 1.5};
    double dipole_y;
    double dir_x, dir_y;
    double max_intensity;
    double ramp_time;
    double current_time;

    DipolePotentialBC(double intensity, double ramp, double y_pos, double t)
        : dealii::Function<dim>(1)
        , dipole_y(y_pos)
        , dir_x(0.0)
        , dir_y(1.0)
        , max_intensity(intensity)
        , ramp_time(ramp)
        , current_time(t)
    {}

    virtual double value(const dealii::Point<dim>& p, const unsigned int = 0) const override
    {
        // Time ramping: intensity goes from 0 to max_intensity over ramp_time
        double alpha = max_intensity;
        if (current_time < ramp_time && ramp_time > 0)
            alpha = max_intensity * (current_time / ramp_time);

        double phi_total = 0.0;
        for (int s = 0; s < n_dipoles; ++s)
        {
            // r = point - dipole_position (vector FROM dipole TO point)
            double rx = dipole_x[s] - p[0];
            double ry = dipole_y - p[1];
            double r2 = rx * rx + ry * ry;
            if (r2 < 1e-10) continue;
            
            // 2D dipole potential: φ = (m / 2π) × (d · r) / |r|²
            // where d is the dipole direction (pointing upward: (0,1))
            double d_dot_r = dir_x * rx + dir_y * ry;
            phi_total += alpha * d_dot_r / r2;
        }
        return phi_total;
    }
};

// ============================================================================
// Update Poisson constraints with dipole field at given time
// ============================================================================
template <int dim>
void NSCHProblem<dim>::update_phi_constraints_dipole(double current_time)
{
    if (!params_.enable_magnetic)
        return;

    phi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);

    DipolePotentialBC<dim> dipole_bc(
        params_.dipole_intensity,
        params_.dipole_ramp_time,
        params_.dipole_y_position,
        current_time);

    for (unsigned int boundary_id = 0; boundary_id < 4; ++boundary_id)
    {
        dealii::VectorTools::interpolate_boundary_values(
            phi_dof_handler_,
            boundary_id,
            dipole_bc,
            phi_constraints_);
    }

    phi_constraints_.close();
}

// ============================================================================
// Update MMS boundary conditions
// ============================================================================
template <int dim>
void NSCHProblem<dim>::update_mms_boundary_conditions(double new_time)
{
    if (!params_.mms_mode)
        return;

    // Update ux constraints
    ux_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);

    // MMS exact ux on all boundaries
    class MMSExactUx : public dealii::Function<dim>
    {
    public:
        double t;
        MMSExactUx(double time) : dealii::Function<dim>(1), t(time) {}
        virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
        {
            // From nsch_mms.h: u_x = t^4 * sin(pi*x) * sin(pi*y)
            return std::pow(t, 4) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
        }
    };

    MMSExactUx exact_ux(new_time);
    for (unsigned int b = 0; b < 4; ++b)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_, b, exact_ux, ux_constraints_);
    }
    ux_constraints_.close();

    // Update uy constraints
    uy_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);

    class MMSExactUy : public dealii::Function<dim>
    {
    public:
        double t;
        MMSExactUy(double time) : dealii::Function<dim>(1), t(time) {}
        virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
        {
            // u_y = -t^4 * sin(pi*x) * sin(pi*y) (for div-free)
            return -std::pow(t, 4) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
        }
    };

    MMSExactUy exact_uy(new_time);
    for (unsigned int b = 0; b < 4; ++b)
    {
        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_, b, exact_uy, uy_constraints_);
    }
    uy_constraints_.close();

    // Update c constraints
    c_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(c_dof_handler_, c_constraints_);

    class MMSExactC : public dealii::Function<dim>
    {
    public:
        double t;
        MMSExactC(double time) : dealii::Function<dim>(1), t(time) {}
        virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
        {
            return std::pow(t, 4) * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
        }
    };

    MMSExactC exact_c(new_time);
    for (unsigned int b = 0; b < 4; ++b)
    {
        dealii::VectorTools::interpolate_boundary_values(
            c_dof_handler_, b, exact_c, c_constraints_);
    }
    c_constraints_.close();
}

// ============================================================================
// Solve Cahn-Hilliard using direct assembly with scalar DoFHandlers
// FIXED: Uses combined constraints that match the sparsity pattern
// ============================================================================
template <int dim>
void NSCHProblem<dim>::solve_cahn_hilliard(double dt)
{
    // Zero the system
    ch_matrix_ = 0;
    ch_rhs_ = 0;

    // Assemble CH system directly using our scalar DoFHandlers
    assemble_ch_system(dt, time_ + dt);

    // Apply constraints to matrix and RHS BEFORE solving
    // Use the member combined constraints that were built during setup
    ch_combined_constraints_.condense(ch_matrix_, ch_rhs_);

    // Solve
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(ch_matrix_);

    dealii::Vector<double> ch_solution(ch_rhs_.size());
    solver.vmult(ch_solution, ch_rhs_);

    // Apply constraints to solution (distribute constrained DoFs)
    ch_combined_constraints_.distribute(ch_solution);

    // Unpack solution back to separate vectors
    for (unsigned int i = 0; i < c_solution_.size(); ++i)
        c_solution_[i] = ch_solution[c_to_ch_map_[i]];
    for (unsigned int i = 0; i < mu_solution_.size(); ++i)
        mu_solution_[i] = ch_solution[mu_to_ch_map_[i]];

    // Apply individual constraints to ensure consistency
    c_constraints_.distribute(c_solution_);
    mu_constraints_.distribute(mu_solution_);
}

// ============================================================================
// Assemble CH system using scalar DoFHandlers directly
// ============================================================================
template <int dim>
void NSCHProblem<dim>::assemble_ch_system(double dt, double current_time)
{
    (void)current_time;  // May be used for MMS source terms in future

    // Quadrature and FEValues
    dealii::QGauss<dim> quadrature(params_.fe_degree_phase + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> c_fe_values(fe_Q2_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> mu_fe_values(fe_Q2_, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> ux_fe_values(fe_Q2_, quadrature, dealii::update_values);
    dealii::FEValues<dim> uy_fe_values(fe_Q2_, quadrature, dealii::update_values);

    const unsigned int dofs_per_cell = fe_Q2_.n_dofs_per_cell();

    dealii::FullMatrix<double> local_matrix_cc(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_cm(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_mc(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_matrix_mm(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_c(dofs_per_cell);
    dealii::Vector<double> local_rhs_m(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> c_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> mu_local_dofs(dofs_per_cell);

    std::vector<double> c_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> c_old_gradients(n_q_points);
    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);

    const double epsilon = params_.epsilon;
    const double gamma = params_.mobility;

    // Iterate over cells
    auto c_cell = c_dof_handler_.begin_active();
    auto mu_cell = mu_dof_handler_.begin_active();
    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();

    for (; c_cell != c_dof_handler_.end(); ++c_cell, ++mu_cell, ++ux_cell, ++uy_cell)
    {
        c_fe_values.reinit(c_cell);
        mu_fe_values.reinit(mu_cell);
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        local_matrix_cc = 0;
        local_matrix_cm = 0;
        local_matrix_mc = 0;
        local_matrix_mm = 0;
        local_rhs_c = 0;
        local_rhs_m = 0;

        // Get old solution values
        c_fe_values.get_function_values(c_old_solution_, c_old_values);
        c_fe_values.get_function_gradients(c_old_solution_, c_old_gradients);
        ux_fe_values.get_function_values(ux_solution_, ux_values);
        uy_fe_values.get_function_values(uy_solution_, uy_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = c_fe_values.JxW(q);
            const double c_old = c_old_values[q];

            // Velocity at quadrature point
            dealii::Tensor<1, dim> u;
            u[0] = ux_values[q];
            u[1] = uy_values[q];

            // Double-well potential derivatives
            const double f_old = c_old * c_old * c_old - c_old;
            const double fp_old = 3.0 * c_old * c_old - 1.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = c_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_i = c_fe_values.shape_grad(i, q);
                const double chi_i = mu_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_chi_i = mu_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = c_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = c_fe_values.shape_grad(j, q);
                    const double chi_j = mu_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_chi_j = mu_fe_values.shape_grad(j, q);

                    // Eq (14a): c_t + div(uc) - gamma * Delta(mu) = 0
                    // Weak form: (1/dt)(c - c_old, phi) + (u . grad(c), phi) + gamma*(grad_mu, grad_phi) = 0
                    // Rearranged: (1/dt)(c, phi) + (u . grad(c), phi) + gamma*(grad_mu, grad_phi) = (1/dt)(c_old, phi)

                    // c-c block: (1/dt)(c, phi) + (u . grad(c), phi)
                    // Using conservative form: -(c*u, grad_phi) for advection
                    local_matrix_cc(i, j) += (1.0 / dt) * phi_i * phi_j * JxW;
                    local_matrix_cc(i, j) -= phi_j * (u * grad_phi_i) * JxW;

                    // c-mu block: -gamma*(grad_mu, grad_phi)
                    // From IBP of +gamma*Delta(mu) term
                    local_matrix_cm(i, j) -= gamma * (grad_chi_j * grad_phi_i) * JxW;

                    // Eq (14b): mu - epsilon*Delta(c) - (1/epsilon)*f(c) = 0
                    // With linearization: f(c) ≈ f(c_old) + f'(c_old)*(c - c_old)
                    // Weak form: (mu, chi) + epsilon*(grad_c, grad_chi) - (1/epsilon)*(f'(c_old)*c, chi)
                    //          = (1/epsilon)*(f(c_old) - f'(c_old)*c_old, chi)

                    // mu-mu block: (mu, chi)
                    local_matrix_mm(i, j) += chi_i * chi_j * JxW;

                    // mu-c block: epsilon*(grad_c, grad_chi) + (1/epsilon)*(f'*c, chi)
                    local_matrix_mc(i, j) += epsilon * (grad_phi_j * grad_chi_i) * JxW;
                    local_matrix_mc(i, j) += (1.0 / epsilon) * fp_old * phi_j * chi_i * JxW;
                }

                // RHS for c equation: (1/dt)(c_old, phi)
                local_rhs_c(i) += (1.0 / dt) * c_old * phi_i * JxW;

                // RHS for mu equation: -(1/epsilon)*(f(c_old) - f'(c_old)*c_old, chi)
                // Note: f - f'*c_old = c_old^3 - c_old - (3*c_old^2 - 1)*c_old = -2*c_old^3
                local_rhs_m(i) -= (1.0 / epsilon) * (f_old - fp_old * c_old) * chi_i * JxW;
            }
        }

        // Get DoF indices
        c_cell->get_dof_indices(c_local_dofs);
        mu_cell->get_dof_indices(mu_local_dofs);

        // Assemble into global matrix
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const auto gi_c = c_to_ch_map_[c_local_dofs[i]];
            const auto gi_m = mu_to_ch_map_[mu_local_dofs[i]];

            ch_rhs_(gi_c) += local_rhs_c(i);
            ch_rhs_(gi_m) += local_rhs_m(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const auto gj_c = c_to_ch_map_[c_local_dofs[j]];
                const auto gj_m = mu_to_ch_map_[mu_local_dofs[j]];

                ch_matrix_.add(gi_c, gj_c, local_matrix_cc(i, j));
                ch_matrix_.add(gi_c, gj_m, local_matrix_cm(i, j));
                ch_matrix_.add(gi_m, gj_c, local_matrix_mc(i, j));
                ch_matrix_.add(gi_m, gj_m, local_matrix_mm(i, j));
            }
        }
    }
}

// ============================================================================
// Solve Poisson for magnetic potential
// ============================================================================
template <int dim>
void NSCHProblem<dim>::solve_poisson()
{
    if (!params_.enable_magnetic)
        return;

    // Update constraints with current time dipole field
    update_phi_constraints_dipole(time_);

    // Rebuild sparsity pattern for new constraints
    dealii::DynamicSparsityPattern dsp(phi_dof_handler_.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(phi_dof_handler_, dsp, phi_constraints_, false);
    phi_sparsity_.copy_from(dsp);
    phi_matrix_.reinit(phi_sparsity_);
    phi_rhs_ = 0;

    // Assemble Poisson system using scalar assembler
    assemble_poisson_system();

    // Solve
    solve_poisson_system(phi_matrix_, phi_rhs_, phi_solution_, phi_constraints_);
}

// ============================================================================
// Assemble Poisson system using scalar DoFHandlers
// ============================================================================
template <int dim>
void NSCHProblem<dim>::assemble_poisson_system()
{
    phi_matrix_ = 0;
    phi_rhs_ = 0;

    dealii::QGauss<dim> quadrature(params_.fe_degree_potential + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(fe_Q2_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> c_fe_values(fe_Q2_, quadrature, dealii::update_values);

    const unsigned int dofs_per_cell = fe_Q2_.n_dofs_per_cell();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> c_values(n_q_points);

    const double kappa_0 = params_.chi_m;
    const double epsilon = params_.epsilon;

    auto phi_cell = phi_dof_handler_.begin_active();
    auto c_cell = c_dof_handler_.begin_active();

    for (; phi_cell != phi_dof_handler_.end(); ++phi_cell, ++c_cell)
    {
        phi_fe_values.reinit(phi_cell);
        c_fe_values.reinit(c_cell);

        local_matrix = 0;
        local_rhs = 0;

        c_fe_values.get_function_values(c_solution_, c_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const double c = c_values[q];

            // Permeability: mu(theta) = 1 + kappa_0 * H(theta/epsilon)
            const double sigmoid_val = 1.0 / (1.0 + std::exp(-c / epsilon));
            const double mu = 1.0 + kappa_0 * sigmoid_val;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const dealii::Tensor<1, dim> grad_psi_i = phi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const dealii::Tensor<1, dim> grad_psi_j = phi_fe_values.shape_grad(j, q);
                    local_matrix(i, j) += mu * (grad_psi_i * grad_psi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints_.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices, phi_matrix_, phi_rhs_);
    }
}

// ============================================================================
// Solve Navier-Stokes
// FIXED: Uses combined constraints that match the sparsity pattern
// FIXED: Pressure pinning is added during setup
// ============================================================================
template <int dim>
void NSCHProblem<dim>::solve_navier_stokes(double dt)
{
    // Zero the system
    ns_matrix_ = 0;
    ns_rhs_ = 0;

    // Assemble NS system using scalar DoFHandlers
    assemble_ns_system(dt, time_ + dt);

    // Apply constraints to matrix and RHS BEFORE solving
    // Use the member combined constraints that were built during setup
    ns_combined_constraints_.condense(ns_matrix_, ns_rhs_);

    // Solve
    dealii::SparseDirectUMFPACK solver;
    solver.initialize(ns_matrix_);

    dealii::Vector<double> ns_solution(ns_rhs_.size());
    solver.vmult(ns_solution, ns_rhs_);

    // Apply constraints to solution
    ns_combined_constraints_.distribute(ns_solution);

    // Unpack solution
    for (unsigned int i = 0; i < ux_solution_.size(); ++i)
        ux_solution_[i] = ns_solution[ux_to_ns_map_[i]];
    for (unsigned int i = 0; i < uy_solution_.size(); ++i)
        uy_solution_[i] = ns_solution[uy_to_ns_map_[i]];
    for (unsigned int i = 0; i < p_solution_.size(); ++i)
        p_solution_[i] = ns_solution[p_to_ns_map_[i]];

    // Apply individual constraints to ensure consistency
    ux_constraints_.distribute(ux_solution_);
    uy_constraints_.distribute(uy_solution_);
    p_constraints_.distribute(p_solution_);
}

// ============================================================================
// Assemble NS system using scalar DoFHandlers
// ============================================================================
template <int dim>
void NSCHProblem<dim>::assemble_ns_system(double dt, double current_time)
{
    (void)current_time;  // May be used for MMS source terms in future

    dealii::QGauss<dim> quadrature(params_.fe_degree_velocity + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(fe_Q2_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_fe_values(fe_Q2_, quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> p_fe_values(fe_Q1_, quadrature,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> c_fe_values(fe_Q2_, quadrature,
        dealii::update_values);
    dealii::FEValues<dim> mu_fe_values(fe_Q2_, quadrature,
        dealii::update_gradients);

    // For magnetic field
    dealii::FEValues<dim> phi_fe_values(fe_Q2_, quadrature,
        dealii::update_gradients | dealii::update_hessians);

    const unsigned int dofs_per_cell_Q2 = fe_Q2_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = fe_Q1_.n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> ux_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_local_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_local_dofs(dofs_per_cell_Q1);

    // Old velocity values
    std::vector<double> ux_old_values(n_q_points);
    std::vector<double> uy_old_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_old_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_old_gradients(n_q_points);

    // Phase field values
    std::vector<double> c_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> mu_gradients(n_q_points);

    // Magnetic potential
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<2, dim>> phi_hessians(n_q_points);

    // Physical parameters
    const double nu_w = params_.nu_water;
    const double nu_f = params_.nu_ferro;
    const double epsilon = params_.epsilon;
    const double lambda = params_.lambda;
    const double theta_time = params_.theta;

    // Magnetic parameters
    const double mu_0 = params_.mu_0;
    const double kappa_0 = params_.chi_m;

    // Gravity parameters
    const bool use_gravity = params_.enable_gravity;
    const double g_mag = params_.gravity;
    const double g_angle = params_.gravity_angle * M_PI / 180.0;
    const double r_density = params_.density_ratio;
    dealii::Tensor<1, dim> g_direction;
    g_direction[0] = std::cos(g_angle);
    g_direction[1] = std::sin(g_angle);

    // Sigmoid function for smooth Heaviside
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };

    // Cell iterators
    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();
    auto p_cell = p_dof_handler_.begin_active();
    auto c_cell = c_dof_handler_.begin_active();
    auto mu_cell = mu_dof_handler_.begin_active();
    auto phi_cell = phi_dof_handler_.begin_active();

    for (; ux_cell != ux_dof_handler_.end();
         ++ux_cell, ++uy_cell, ++p_cell, ++c_cell, ++mu_cell, ++phi_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);
        c_fe_values.reinit(c_cell);
        mu_fe_values.reinit(mu_cell);
        if (params_.enable_magnetic)
            phi_fe_values.reinit(phi_cell);

        ux_cell->get_dof_indices(ux_local_dofs);
        uy_cell->get_dof_indices(uy_local_dofs);
        p_cell->get_dof_indices(p_local_dofs);

        // Get old values
        ux_fe_values.get_function_values(ux_old_solution_, ux_old_values);
        uy_fe_values.get_function_values(uy_old_solution_, uy_old_values);
        ux_fe_values.get_function_gradients(ux_old_solution_, ux_old_gradients);
        uy_fe_values.get_function_gradients(uy_old_solution_, uy_old_gradients);

        c_fe_values.get_function_values(c_solution_, c_values);
        mu_fe_values.get_function_gradients(mu_solution_, mu_gradients);

        if (params_.enable_magnetic)
        {
            phi_fe_values.get_function_gradients(phi_solution_, phi_gradients);
            phi_fe_values.get_function_hessians(phi_solution_, phi_hessians);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);

            const double ux_old = ux_old_values[q];
            const double uy_old = uy_old_values[q];
            const dealii::Tensor<1, dim>& grad_ux_old = ux_old_gradients[q];
            const dealii::Tensor<1, dim>& grad_uy_old = uy_old_gradients[q];

            const double c = c_values[q];
            const dealii::Tensor<1, dim>& grad_mu = mu_gradients[q];

            // Phase-dependent viscosity: nu(c) = nu_w + (nu_f - nu_w) * H(c/eps)
            const double H_c = sigmoid(c / epsilon);
            const double nu = nu_w + (nu_f - nu_w) * H_c;

            // Capillary force: F_cap = (lambda/epsilon) * c * grad(mu)
            dealii::Tensor<1, dim> F_cap;
            if (!params_.mms_mode)
            {
                double coeff_cap = lambda / epsilon;
                F_cap[0] = coeff_cap * c * grad_mu[0];
                F_cap[1] = coeff_cap * c * grad_mu[1];
            }

            // Magnetic force
            dealii::Tensor<1, dim> F_mag;
            if (params_.enable_magnetic && !params_.mms_mode)
            {
                const dealii::Tensor<1, dim>& grad_phi = phi_gradients[q];
                const dealii::Tensor<2, dim>& hess_phi = phi_hessians[q];

                // h = -grad(phi)
                dealii::Tensor<1, dim> h;
                h[0] = -grad_phi[0];
                h[1] = -grad_phi[1];

                // grad_h = -Hess(phi)
                dealii::Tensor<2, dim> grad_h;
                for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                        grad_h[i][j] = -hess_phi[i][j];

                // kappa_theta = kappa_0 * H(c/epsilon)
                double kappa_theta = kappa_0 * sigmoid(c / epsilon);

                // Kelvin force: mu_0 * kappa_theta * (h . grad)h
                double coeff_mag = mu_0 * kappa_theta;
                for (unsigned int i = 0; i < dim; ++i)
                {
                    F_mag[i] = 0.0;
                    for (unsigned int j = 0; j < dim; ++j)
                        F_mag[i] += h[j] * grad_h[i][j];
                    F_mag[i] *= coeff_mag;
                }
            }

            // Gravity force
            dealii::Tensor<1, dim> F_grav;
            if (use_gravity && !params_.mms_mode)
            {
                double H_theta = sigmoid(c / epsilon);
                double grav_factor = 1.0 + r_density * H_theta;
                F_grav[0] = grav_factor * g_mag * g_direction[0];
                F_grav[1] = grav_factor * g_mag * g_direction[1];
            }

            // Convection term: (u_old . grad)u_old
            double conv_x = ux_old * grad_ux_old[0] + uy_old * grad_ux_old[1];
            double conv_y = ux_old * grad_uy_old[0] + uy_old * grad_uy_old[1];

            // Assemble contributions
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_ux_i = ux_fe_values.shape_value(i, q);
                const double phi_uy_i = uy_fe_values.shape_value(i, q);
                const dealii::Tensor<1, dim> grad_phi_ux_i = ux_fe_values.shape_grad(i, q);
                const dealii::Tensor<1, dim> grad_phi_uy_i = uy_fe_values.shape_grad(i, q);

                // RHS contributions
                double rhs_ux = (1.0/dt) * ux_old * phi_ux_i;
                double rhs_uy = (1.0/dt) * uy_old * phi_uy_i;

                // Explicit viscosity term
                rhs_ux -= (1.0 - theta_time) * nu * (grad_ux_old * grad_phi_ux_i);
                rhs_uy -= (1.0 - theta_time) * nu * (grad_uy_old * grad_phi_uy_i);

                // Explicit convection
                rhs_ux -= conv_x * phi_ux_i;
                rhs_uy -= conv_y * phi_uy_i;

                // Forces
                if (!params_.mms_mode)
                {
                    rhs_ux += F_cap[0] * phi_ux_i + F_mag[0] * phi_ux_i + F_grav[0] * phi_ux_i;
                    rhs_uy += F_cap[1] * phi_uy_i + F_mag[1] * phi_uy_i + F_grav[1] * phi_uy_i;
                }

                ns_rhs_(ux_to_ns_map_[ux_local_dofs[i]]) += rhs_ux * JxW;
                ns_rhs_(uy_to_ns_map_[uy_local_dofs[i]]) += rhs_uy * JxW;

                // Matrix contributions
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_ux_j = ux_fe_values.shape_value(j, q);
                    const double phi_uy_j = uy_fe_values.shape_value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    // Mass term
                    double val_ux_ux = (1.0/dt) * phi_ux_i * phi_ux_j;
                    double val_uy_uy = (1.0/dt) * phi_uy_i * phi_uy_j;

                    // Implicit viscosity
                    val_ux_ux += theta_time * nu * (grad_phi_ux_i * grad_phi_ux_j);
                    val_uy_uy += theta_time * nu * (grad_phi_uy_i * grad_phi_uy_j);

                    // Grad-div stabilization: γ * (div u, div v)
                    // div u = ∂ux/∂x + ∂uy/∂y, div v = ∂vx/∂x + ∂vy/∂y
                    // This creates coupling between ux and uy blocks
                    if (params_.grad_div_gamma > 0.0)
                    {
                        const double gamma_gd = params_.grad_div_gamma;
                        // ux-ux: γ * ∂ux/∂x * ∂vx/∂x
                        val_ux_ux += gamma_gd * grad_phi_ux_i[0] * grad_phi_ux_j[0];
                        // uy-uy: γ * ∂uy/∂y * ∂vy/∂y
                        val_uy_uy += gamma_gd * grad_phi_uy_i[1] * grad_phi_uy_j[1];
                    }

                    ns_matrix_.add(ux_to_ns_map_[ux_local_dofs[i]], ux_to_ns_map_[ux_local_dofs[j]], val_ux_ux * JxW);
                    ns_matrix_.add(uy_to_ns_map_[uy_local_dofs[i]], uy_to_ns_map_[uy_local_dofs[j]], val_uy_uy * JxW);

                    // Grad-div cross-coupling terms
                    if (params_.grad_div_gamma > 0.0)
                    {
                        const double gamma_gd = params_.grad_div_gamma;
                        // ux-uy: γ * ∂uy/∂y * ∂vx/∂x (trial uy, test vx)
                        ns_matrix_.add(ux_to_ns_map_[ux_local_dofs[i]], uy_to_ns_map_[uy_local_dofs[j]],
                                       gamma_gd * grad_phi_ux_i[0] * grad_phi_uy_j[1] * JxW);
                        // uy-ux: γ * ∂ux/∂x * ∂vy/∂y (trial ux, test vy)
                        ns_matrix_.add(uy_to_ns_map_[uy_local_dofs[i]], ux_to_ns_map_[ux_local_dofs[j]],
                                       gamma_gd * grad_phi_uy_i[1] * grad_phi_ux_j[0] * JxW);
                    }
                }

                // Pressure gradient in momentum equation
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double phi_p_j = p_fe_values.shape_value(j, q);

                    // -p * div(v) term: comes from integration by parts of pressure
                    // In weak form: -(p, div v) = (grad p, v) after IBP
                    // Standard formulation: -p * (d(phi_ux)/dx + d(phi_uy)/dy)

                    // Actually for separate velocity components:
                    // For ux equation: -dp/dx * phi_ux_i -> -(p, d(phi_ux)/dx)
                    ns_matrix_.add(ux_to_ns_map_[ux_local_dofs[i]], p_to_ns_map_[p_local_dofs[j]],
                                   -phi_p_j * grad_phi_ux_i[0] * JxW);
                    ns_matrix_.add(uy_to_ns_map_[uy_local_dofs[i]], p_to_ns_map_[p_local_dofs[j]],
                                   -phi_p_j * grad_phi_uy_i[1] * JxW);
                }
            }

            // Continuity equation: div(u) = 0
            // (div u, q) = 0  ->  (d(ux)/dx + d(uy)/dy, q) = 0
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const double phi_p_i = p_fe_values.shape_value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const dealii::Tensor<1, dim> grad_phi_ux_j = ux_fe_values.shape_grad(j, q);
                    const dealii::Tensor<1, dim> grad_phi_uy_j = uy_fe_values.shape_grad(j, q);

                    ns_matrix_.add(p_to_ns_map_[p_local_dofs[i]], ux_to_ns_map_[ux_local_dofs[j]],
                                   -grad_phi_ux_j[0] * phi_p_i * JxW);
                    ns_matrix_.add(p_to_ns_map_[p_local_dofs[i]], uy_to_ns_map_[uy_local_dofs[j]],
                                   -grad_phi_uy_j[1] * phi_p_i * JxW);
                }
            }
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class NSCHProblem<2>;