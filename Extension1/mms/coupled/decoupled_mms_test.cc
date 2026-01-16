// ============================================================================
// mms/coupled/decoupled_mms_test.cc - MINIMAL Decoupled MMS Tests
//
// GOAL: Test each subsystem INDEPENDENTLY to isolate bugs
//
// Test 1: Poisson with EXACT M* (no coupling to discrete M_h)
//   -Δφ = -∇·M* + f_φ   where M* is exact, f_φ = -Δφ*
//   Expected: φ_h → φ* with optimal rates
//
// Test 2: Magnetization with EXACT H* (no coupling to discrete φ_h)
//   (M^n - M^{n-1})/dt + M^n/τ = χH*/τ + f_M
//   where H* is exact, f_M computed from exact M*
//   Expected: M_h → M* with optimal rates
//
// This AVOIDS the circular dependency that plagues coupled tests.
// ============================================================================

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int dim = 2;

// ============================================================================
// EXACT SOLUTIONS
// ============================================================================

// φ* = t·cos(πx)·cos(πy/L_y)
double exact_phi(double x, double y, double t, double L_y)
{
    return t * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ∇φ*
void exact_grad_phi(double x, double y, double t, double L_y, double& dpdx, double& dpdy)
{
    dpdx = -t * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
    dpdy = -t * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
}

// H* = -∇φ*
void exact_H(double x, double y, double t, double L_y, double& Hx, double& Hy)
{
    double dpdx, dpdy;
    exact_grad_phi(x, y, t, L_y, dpdx, dpdy);
    Hx = -dpdx;  // = t·π·sin(πx)·cos(πy/L_y)
    Hy = -dpdy;  // = t·(π/L_y)·cos(πx)·sin(πy/L_y)
}

// Mx* = t·sin(πx)·sin(πy/L_y)
double exact_Mx(double x, double y, double t, double L_y)
{
    return t * std::sin(M_PI * x) * std::sin(M_PI * y / L_y);
}

// My* = t·cos(πx)·sin(πy/L_y)
double exact_My(double x, double y, double t, double L_y)
{
    return t * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
}

// ∇·M* = ∂Mx*/∂x + ∂My*/∂y
double exact_div_M(double x, double y, double t, double L_y)
{
    double dMx_dx = t * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
    double dMy_dy = t * (M_PI / L_y) * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
    return dMx_dx + dMy_dy;
}

// -Δφ* = π²(1 + 1/L_y²)·t·cos(πx)·cos(πy/L_y)
double exact_neg_laplacian_phi(double x, double y, double t, double L_y)
{
    return t * M_PI * M_PI * (1.0 + 1.0/(L_y*L_y)) * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// TEST 1: POISSON WITH EXACT M*
//
// Strong form: -Δφ = -∇·M* + f_φ
// For φ_h → φ*: f_φ = -Δφ* + ∇·M* (so that -Δφ* = -∇·M* + f_φ)
//
// Weak form: (∇φ, ∇χ) = (-M*, ∇χ) + (f_φ, χ)
// where M* is EXACT (interpolated), not discrete
// ============================================================================
struct PoissonTestResult
{
    double h;
    double L2_error;
    double H1_error;
};

PoissonTestResult run_poisson_test(
    unsigned int refinement,
    double t,
    double L_y,
    MPI_Comm mpi_comm)
{
    PoissonTestResult result;
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Create mesh
    dealii::parallel::distributed::Triangulation<dim> tria(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    const unsigned int n_cells_x = 10;
    const unsigned int n_cells_y = static_cast<unsigned int>(std::round(n_cells_x * L_y));
    
    dealii::GridGenerator::subdivided_hyper_rectangle(
        tria, {n_cells_x, n_cells_y},
        dealii::Point<dim>(0.0, 0.0),
        dealii::Point<dim>(1.0, L_y));

    tria.refine_global(refinement);
    result.h = 1.0 / (n_cells_x * std::pow(2.0, refinement));

    // FE space (Q2 for phi)
    dealii::FE_Q<dim> fe(2);
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet owned = dof_handler.locally_owned_dofs();
    dealii::IndexSet relevant = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Constraints (pure Neumann - no Dirichlet)
    dealii::AffineConstraints<double> constraints;
    constraints.reinit(owned, relevant);
    // Pin one DOF to fix the constant (required for pure Neumann)
    if (owned.is_element(0))
        constraints.add_line(0);
    constraints.close();

    // Sparsity pattern
    dealii::DynamicSparsityPattern dsp(relevant);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, owned, mpi_comm, relevant);

    dealii::TrilinosWrappers::SparseMatrix matrix;
    matrix.reinit(owned, owned, dsp, mpi_comm);

    dealii::TrilinosWrappers::MPI::Vector rhs(owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector solution(owned, mpi_comm);

    // Assemble matrix and RHS
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& pt = fe_values.quadrature_point(q);
            const double x = pt[0], y = pt[1];

            // Exact M* at this point
            const double Mx_exact = exact_Mx(x, y, t, L_y);
            const double My_exact = exact_My(x, y, t, L_y);

            // MMS source: f_φ = -Δφ* - ∇·M*
            // Weak form: (∇φ,∇χ) = (-M,∇χ) + (f_φ,χ)
            // Strong form: -Δφ = ∇·M + f_φ  (after integrating (-M,∇χ) by parts)
            // For φ_h → φ*: f_φ = -Δφ* - ∇·M*
            const double f_phi = exact_neg_laplacian_phi(x, y, t, L_y) - exact_div_M(x, y, t, L_y);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double chi_i = fe_values.shape_value(i, q);
                const auto& grad_chi_i = fe_values.shape_grad(i, q);

                // RHS: (-M*, ∇χ) + (f_φ, χ)
                cell_rhs(i) += (-Mx_exact * grad_chi_i[0] - My_exact * grad_chi_i[1]) * JxW;
                cell_rhs(i) += f_phi * chi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = fe_values.shape_grad(j, q);
                    // Matrix: (∇φ, ∇χ)
                    cell_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dofs);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dofs, matrix, rhs);
    }

    matrix.compress(dealii::VectorOperation::add);
    rhs.compress(dealii::VectorOperation::add);

    // Solve with GMRES + AMG (more robust for near-singular Neumann problems)
    dealii::TrilinosWrappers::PreconditionAMG preconditioner;
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.elliptic = true;
    amg_data.higher_order_elements = true;
    amg_data.smoother_sweeps = 2;
    preconditioner.initialize(matrix, amg_data);

    // Use relaxed tolerance - the singular mode can cause precision issues
    const double tol = std::max(1e-10, 1e-8 * rhs.l2_norm());
    dealii::SolverControl solver_control(2000, tol);
    dealii::TrilinosWrappers::SolverGMRES solver(solver_control);
    
    try {
        solver.solve(matrix, solution, rhs, preconditioner);
    } catch (const std::exception& e) {
        if (this_rank == 0)
            std::cerr << "  WARNING: Solver did not converge: " << e.what() << "\n";
    }
    constraints.distribute(solution);

    if (this_rank == 0)
        std::cout << "  Poisson solve: " << solver_control.last_step() << " GMRES iterations\n";

    // Compute errors with mean correction
    dealii::TrilinosWrappers::MPI::Vector solution_rel(owned, relevant, mpi_comm);
    solution_rel = solution;

    std::vector<double> phi_vals(n_q);
    std::vector<dealii::Tensor<1,dim>> grad_vals(n_q);

    double local_vol = 0, local_mean_diff = 0, local_H1_sq = 0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values.reinit(cell);
        fe_values.get_function_values(solution_rel, phi_vals);
        fe_values.get_function_gradients(solution_rel, grad_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& pt = fe_values.quadrature_point(q);
            const double x = pt[0], y = pt[1];

            const double phi_ex = exact_phi(x, y, t, L_y);
            double grad_ex_x, grad_ex_y;
            exact_grad_phi(x, y, t, L_y, grad_ex_x, grad_ex_y);

            local_mean_diff += (phi_vals[q] - phi_ex) * JxW;
            local_vol += JxW;

            double grad_err_x = grad_vals[q][0] - grad_ex_x;
            double grad_err_y = grad_vals[q][1] - grad_ex_y;
            local_H1_sq += (grad_err_x*grad_err_x + grad_err_y*grad_err_y) * JxW;
        }
    }

    double global_vol, global_mean_diff, global_H1_sq;
    MPI_Allreduce(&local_vol, &global_vol, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_mean_diff, &global_mean_diff, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_H1_sq, &global_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    double c_shift = global_mean_diff / global_vol;

    // Second pass for L2 with mean correction
    double local_L2_sq = 0;
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values.reinit(cell);
        fe_values.get_function_values(solution_rel, phi_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& pt = fe_values.quadrature_point(q);
            double phi_ex = exact_phi(pt[0], pt[1], t, L_y);
            double err = (phi_vals[q] - c_shift) - phi_ex;
            local_L2_sq += err * err * JxW;
        }
    }

    double global_L2_sq;
    MPI_Allreduce(&local_L2_sq, &global_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    result.L2_error = std::sqrt(global_L2_sq);
    result.H1_error = std::sqrt(global_H1_sq);

    return result;
}

// ============================================================================
// TEST 2: MAGNETIZATION WITH EXACT H*
//
// Time-discrete: (M^n - M^{n-1})/dt + M^n/τ = χH*/τ + f_M
// For M_h → M*: f_M = (M*^n - M*^{n-1})/dt + M*^n/τ - χH*/τ
//
// Rearranged: (1/dt + 1/τ)M^n = M^{n-1}/dt + χH*/τ + f_M
// ============================================================================
struct MagTestResult
{
    double h;
    double L2_error;
};

MagTestResult run_magnetization_test(
    unsigned int refinement,
    double t_start,
    double t_end,
    unsigned int n_steps,
    double tau_M,
    double chi_0,
    double L_y,
    MPI_Comm mpi_comm)
{
    MagTestResult result;
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const double dt = (t_end - t_start) / n_steps;

    // Create mesh
    dealii::parallel::distributed::Triangulation<dim> tria(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    const unsigned int n_cells_x = 10;
    const unsigned int n_cells_y = static_cast<unsigned int>(std::round(n_cells_x * L_y));
    
    dealii::GridGenerator::subdivided_hyper_rectangle(
        tria, {n_cells_x, n_cells_y},
        dealii::Point<dim>(0.0, 0.0),
        dealii::Point<dim>(1.0, L_y));

    tria.refine_global(refinement);
    result.h = 1.0 / (n_cells_x * std::pow(2.0, refinement));

    // DG-Q1 space for magnetization
    dealii::FE_DGQ<dim> fe(1);
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet owned = dof_handler.locally_owned_dofs();
    dealii::IndexSet relevant = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    // DG has no inter-element constraints, but we need block-diagonal mass matrix
    // For simplicity, solve element-by-element (DG allows this)
    
    dealii::TrilinosWrappers::MPI::Vector Mx_old(owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector My_old(owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector Mx_new(owned, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector My_new(owned, mpi_comm);

    // Initialize with exact solution at t_start
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    dealii::FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    dealii::Vector<double> local_sol(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    double current_time = t_start;

    // Initialize M at t_start via L2 projection
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);
        local_mass = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& pt = fe_values.quadrature_point(q);
            const double x = pt[0], y = pt[1];

            const double Mx_ex = exact_Mx(x, y, current_time, L_y);
            const double My_ex = exact_My(x, y, current_time, L_y);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                local_rhs_x(i) += Mx_ex * phi_i * JxW;
                local_rhs_y(i) += My_ex * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    local_mass(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * JxW;
                }
            }
        }

        local_mass_inv.invert(local_mass);
        
        local_mass_inv.vmult(local_sol, local_rhs_x);
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            Mx_old[local_dofs[i]] = local_sol(i);

        local_mass_inv.vmult(local_sol, local_rhs_y);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            My_old[local_dofs[i]] = local_sol(i);
    }
    Mx_old.compress(dealii::VectorOperation::insert);
    My_old.compress(dealii::VectorOperation::insert);

    // Time stepping
    const double coeff = 1.0/dt + 1.0/tau_M;  // LHS coefficient

    for (unsigned int step = 0; step < n_steps; ++step)
    {
        double t_old = current_time;
        current_time += dt;
        double t_new = current_time;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;

            fe_values.reinit(cell);
            cell->get_dof_indices(local_dofs);

            // Build LHS mass matrix scaled by coeff
            local_mass = 0;
            local_rhs_x = 0;
            local_rhs_y = 0;

            // Get old values at quadrature points
            std::vector<double> Mx_old_vals(n_q), My_old_vals(n_q);
            for (unsigned int q = 0; q < n_q; ++q)
            {
                Mx_old_vals[q] = 0;
                My_old_vals[q] = 0;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    Mx_old_vals[q] += fe_values.shape_value(j, q) * Mx_old[local_dofs[j]];
                    My_old_vals[q] += fe_values.shape_value(j, q) * My_old[local_dofs[j]];
                }
            }

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW = fe_values.JxW(q);
                const auto& pt = fe_values.quadrature_point(q);
                const double x = pt[0], y = pt[1];

                // Exact values
                const double Mx_new_ex = exact_Mx(x, y, t_new, L_y);
                const double My_new_ex = exact_My(x, y, t_new, L_y);
                const double Mx_old_ex = exact_Mx(x, y, t_old, L_y);
                const double My_old_ex = exact_My(x, y, t_old, L_y);

                double Hx_ex, Hy_ex;
                exact_H(x, y, t_new, L_y, Hx_ex, Hy_ex);

                // MMS source: f_M = (M*^n - M*^{n-1})/dt + M*^n/τ - χH*/τ
                const double f_Mx = (Mx_new_ex - Mx_old_ex)/dt + Mx_new_ex/tau_M - chi_0*Hx_ex/tau_M;
                const double f_My = (My_new_ex - My_old_ex)/dt + My_new_ex/tau_M - chi_0*Hy_ex/tau_M;

                // RHS: M^{n-1}/dt + χH*/τ + f_M
                const double rhs_x = Mx_old_vals[q]/dt + chi_0*Hx_ex/tau_M + f_Mx;
                const double rhs_y = My_old_vals[q]/dt + chi_0*Hy_ex/tau_M + f_My;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const double phi_i = fe_values.shape_value(i, q);
                    local_rhs_x(i) += rhs_x * phi_i * JxW;
                    local_rhs_y(i) += rhs_y * phi_i * JxW;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        local_mass(i, j) += coeff * fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * JxW;
                    }
                }
            }

            local_mass_inv.invert(local_mass);

            local_mass_inv.vmult(local_sol, local_rhs_x);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                Mx_new[local_dofs[i]] = local_sol(i);

            local_mass_inv.vmult(local_sol, local_rhs_y);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                My_new[local_dofs[i]] = local_sol(i);
        }

        Mx_new.compress(dealii::VectorOperation::insert);
        My_new.compress(dealii::VectorOperation::insert);

        // Swap for next step
        Mx_old = Mx_new;
        My_old = My_new;
    }

    // Compute error at final time
    dealii::TrilinosWrappers::MPI::Vector Mx_rel(owned, relevant, mpi_comm);
    dealii::TrilinosWrappers::MPI::Vector My_rel(owned, relevant, mpi_comm);
    Mx_rel = Mx_new;
    My_rel = My_new;

    std::vector<double> Mx_vals(n_q), My_vals(n_q);
    double local_L2_sq = 0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(Mx_rel, Mx_vals);
        fe_values.get_function_values(My_rel, My_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& pt = fe_values.quadrature_point(q);

            const double Mx_ex = exact_Mx(pt[0], pt[1], current_time, L_y);
            const double My_ex = exact_My(pt[0], pt[1], current_time, L_y);

            double err_x = Mx_vals[q] - Mx_ex;
            double err_y = My_vals[q] - My_ex;
            local_L2_sq += (err_x*err_x + err_y*err_y) * JxW;
        }
    }

    double global_L2_sq;
    MPI_Allreduce(&local_L2_sq, &global_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    result.L2_error = std::sqrt(global_L2_sq);

    return result;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    const double L_y = 0.6;
    const double t = 0.5;  // Evaluation time for Poisson
    const double t_start = 0.1, t_end = 0.2;
    const double tau_M = 1.0;
    const double chi_0 = 0.5;

    if (this_rank == 0)
    {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  DECOUPLED MMS TESTS\n";
        std::cout << "========================================\n";
        std::cout << "Domain: [0,1] x [0," << L_y << "]\n";
        std::cout << "τ_M = " << tau_M << ", χ_0 = " << chi_0 << "\n\n";
    }

    // ========================================================================
    // TEST 1: POISSON
    // ========================================================================
    if (this_rank == 0)
    {
        std::cout << "--- TEST 1: Poisson with exact M* ---\n";
        std::cout << "Evaluation time: t = " << t << "\n";
        std::cout << "Expected rates: L2 ~ 3, H1 ~ 2 (Q2 elements)\n\n";
    }

    std::vector<unsigned int> refs = {2, 3, 4};
    std::vector<PoissonTestResult> poisson_results;

    for (unsigned int ref : refs)
    {
        if (this_rank == 0)
            std::cout << "ref=" << ref << ":\n";
        
        PoissonTestResult r = run_poisson_test(ref, t, L_y, mpi_comm);
        poisson_results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "  h=" << std::scientific << std::setprecision(3) << r.h
                      << "  L2=" << r.L2_error
                      << "  H1=" << r.H1_error << "\n";
        }
    }

    // Compute rates
    if (this_rank == 0)
    {
        std::cout << "\nPoisson Convergence Rates:\n";
        std::cout << std::setw(5) << "ref" << std::setw(12) << "h" 
                  << std::setw(12) << "L2" << std::setw(8) << "rate"
                  << std::setw(12) << "H1" << std::setw(8) << "rate" << "\n";
        std::cout << std::string(57, '-') << "\n";

        for (size_t i = 0; i < poisson_results.size(); ++i)
        {
            double L2_rate = 0, H1_rate = 0;
            if (i > 0)
            {
                L2_rate = std::log(poisson_results[i-1].L2_error / poisson_results[i].L2_error) /
                          std::log(poisson_results[i-1].h / poisson_results[i].h);
                H1_rate = std::log(poisson_results[i-1].H1_error / poisson_results[i].H1_error) /
                          std::log(poisson_results[i-1].h / poisson_results[i].h);
            }

            std::cout << std::setw(5) << refs[i]
                      << std::setw(12) << std::scientific << std::setprecision(2) << poisson_results[i].h
                      << std::setw(12) << poisson_results[i].L2_error
                      << std::setw(8) << std::fixed << std::setprecision(2) << L2_rate
                      << std::setw(12) << std::scientific << poisson_results[i].H1_error
                      << std::setw(8) << std::fixed << H1_rate << "\n";
        }

        bool poisson_pass = (poisson_results.size() >= 2) &&
            (std::log(poisson_results[poisson_results.size()-2].L2_error / poisson_results.back().L2_error) /
             std::log(poisson_results[poisson_results.size()-2].h / poisson_results.back().h) > 2.5);
        
        std::cout << (poisson_pass ? "[PASS]" : "[FAIL]") << " Poisson test\n\n";
    }

    // ========================================================================
    // TEST 2: MAGNETIZATION
    // ========================================================================
    if (this_rank == 0)
    {
        std::cout << "--- TEST 2: Magnetization with exact H* ---\n";
        std::cout << "Time interval: [" << t_start << ", " << t_end << "]\n";
        std::cout << "Expected rate: L2 ~ 2 (DG-Q1 elements)\n\n";
    }

    // Use more time steps for finer meshes to avoid temporal error dominance
    std::vector<unsigned int> mag_steps = {50, 200, 800};  // Scale with h^2
    std::vector<MagTestResult> mag_results;

    for (size_t i = 0; i < refs.size(); ++i)
    {
        unsigned int ref = refs[i];
        unsigned int steps = mag_steps[i];

        if (this_rank == 0)
            std::cout << "ref=" << ref << " (steps=" << steps << "):\n";

        MagTestResult r = run_magnetization_test(ref, t_start, t_end, steps, tau_M, chi_0, L_y, mpi_comm);
        mag_results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "  h=" << std::scientific << std::setprecision(3) << r.h
                      << "  L2=" << r.L2_error << "\n";
        }
    }

    // Compute rates
    if (this_rank == 0)
    {
        std::cout << "\nMagnetization Convergence Rates:\n";
        std::cout << std::setw(5) << "ref" << std::setw(12) << "h"
                  << std::setw(12) << "L2" << std::setw(8) << "rate" << "\n";
        std::cout << std::string(37, '-') << "\n";

        for (size_t i = 0; i < mag_results.size(); ++i)
        {
            double L2_rate = 0;
            if (i > 0)
            {
                L2_rate = std::log(mag_results[i-1].L2_error / mag_results[i].L2_error) /
                          std::log(mag_results[i-1].h / mag_results[i].h);
            }

            std::cout << std::setw(5) << refs[i]
                      << std::setw(12) << std::scientific << std::setprecision(2) << mag_results[i].h
                      << std::setw(12) << mag_results[i].L2_error
                      << std::setw(8) << std::fixed << std::setprecision(2) << L2_rate << "\n";
        }

        bool mag_pass = (mag_results.size() >= 2) &&
            (std::log(mag_results[mag_results.size()-2].L2_error / mag_results.back().L2_error) /
             std::log(mag_results[mag_results.size()-2].h / mag_results.back().h) > 1.7);

        std::cout << (mag_pass ? "[PASS]" : "[FAIL]") << " Magnetization test\n\n";
    }

    if (this_rank == 0)
        std::cout << "========================================\n\n";

    return 0;
}