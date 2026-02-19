// ============================================================================
// poisson/poisson_assemble.cc - Matrix and RHS Assembly
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   (∇φ^k, ∇X) = (h_a^k − M^k, ∇X)    ∀X ∈ X_h
//
// Matrix: (∇φ, ∇X) — constant-coefficient Laplacian, assembled ONCE
// RHS:    (h_a − M, ∇X) — changes each Picard iteration (M) / timestep (h_a)
// AMG:    built ONCE after matrix assembly, reused for all solves
//
// MMS source term: added as (f_mms, X) when enable_mms = true
// ============================================================================

#include "poisson/poisson.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <chrono>

// ============================================================================
// assemble_matrix — called ONCE from setup()
//
// Assembles: (∇φ, ∇X) for all X ∈ X_h
// Quadrature: degree + 1 (exact for bilinear form of Q1)
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::assemble_matrix()
{
    pcout_ << "[Poisson] Assembling Laplacian matrix (once)...\n";

    system_matrix_ = 0;

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int quad_degree = fe_.degree + 1;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        local_matrix = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_X_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_X_j = fe_values.shape_grad(j, q);

                    // LHS Eq. 42d: (∇φ, ∇X)
                    local_matrix(i, j) += (grad_X_i * grad_X_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints_.distribute_local_to_global(
            local_matrix, local_dof_indices, system_matrix_);
    }

    system_matrix_.compress(dealii::VectorOperation::add);
}

// ============================================================================
// initialize_preconditioner — called ONCE from setup(), after assemble_matrix
//
// AMG settings tuned for scalar Poisson:
//   elliptic = true, higher_order_elements = true (even for Q1, safe default)
//   smoother_sweeps = 2, aggregation_threshold = 1e-4
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::initialize_preconditioner()
{
    pcout_ << "[Poisson] Building AMG preconditioner (once)...\n";

    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 1e-4;
    amg_data.elliptic = true;
    amg_data.higher_order_elements = true;

    try
    {
        amg_preconditioner_.initialize(system_matrix_, amg_data);
        amg_initialized_ = true;
    }
    catch (std::exception& e)
    {
        pcout_ << "[Poisson] WARNING: AMG init failed: " << e.what()
               << " — will fall back to Jacobi\n";
        amg_initialized_ = false;
    }
}

// ============================================================================
// assemble_rhs — called every Picard iteration / timestep
//
// RHS Eq. 42d: (h_a^k − M^k, ∇X) + MMS source
//
// Inputs:
//   M_x_relevant, M_y_relevant: magnetization components (DG, ghosted)
//   M_dof_handler: DoFHandler for M (DG elements)
//   current_time: for h_a ramp and MMS
//
// If M has size 0: assembles with M = 0 (standalone Poisson test)
// If MMS mode: adds (f_mms, X) volumetric source
// ============================================================================
template <int dim>
void PoissonSubsystem<dim>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector& M_x_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& M_y_relevant,
    const dealii::DoFHandler<dim>& M_dof_handler,
    double current_time)
{
    auto start = std::chrono::high_resolution_clock::now();

    const bool has_M = (M_x_relevant.size() > 0);
    const bool has_applied_field =
        (!params_.enable_mms &&
         params_.enable_magnetic &&
         !params_.dipoles.positions.empty());

    system_rhs_ = 0;

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    // FEValues for M (DG elements, different FE space)
    std::unique_ptr<dealii::FEValues<dim>> M_fe_values_ptr;
    if (has_M)
    {
        M_fe_values_ptr = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler.get_fe(), quadrature, dealii::update_values);
    }

    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    // Synchronized cell iteration over φ and M meshes
    auto phi_cell = dof_handler_.begin_active();
    auto M_cell = has_M
        ? M_dof_handler.begin_active()
        : decltype(M_dof_handler.begin_active())();

    for (; phi_cell != dof_handler_.end(); ++phi_cell)
    {
        if (!phi_cell->is_locally_owned())
        {
            if (has_M) ++M_cell;
            continue;
        }

        fe_values.reinit(phi_cell);
        local_rhs = 0;

        if (has_M)
        {
            M_fe_values_ptr->reinit(M_cell);
            M_fe_values_ptr->get_function_values(M_x_relevant, mx_values);
            M_fe_values_ptr->get_function_values(M_y_relevant, my_values);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const dealii::Point<dim>& x_q = fe_values.quadrature_point(q);

            // Magnetization M^k
            dealii::Tensor<1, dim> M;
            if (has_M)
            {
                M[0] = mx_values[q];
                if constexpr (dim >= 2) M[1] = my_values[q];
            }

            // Applied field h_a(x, t)
            dealii::Tensor<1, dim> h_a;
            if (has_applied_field)
                h_a = compute_applied_field<dim>(x_q, params_, current_time);

            // RHS Eq. 42d: source = h_a − M
            dealii::Tensor<1, dim> source = h_a - M;

            // MMS volumetric source: (f_mms, X)
            // f_mms is set via set_mms_source() before assembly
            double f_mms = 0.0;
            if (params_.enable_mms && mms_source_)
            {
                f_mms = mms_source_(x_q, current_time);
            }

            // Assemble: (source, ∇X_i) + (f_mms, X_i)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_X_i = fe_values.shape_grad(i, q);

                local_rhs(i) += (source * grad_X_i) * JxW;

                if (params_.enable_mms)
                {
                    const double X_i = fe_values.shape_value(i, q);
                    local_rhs(i) += f_mms * X_i * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        constraints_.distribute_local_to_global(
            local_rhs, local_dof_indices, system_rhs_);

        if (has_M) ++M_cell;
    }

    system_rhs_.compress(dealii::VectorOperation::add);

    auto end = std::chrono::high_resolution_clock::now();
    last_assemble_time_ = std::chrono::duration<double>(end - start).count();

    ghosts_valid_ = false;
}

// ============================================================================
// Explicit instantiations (methods defined in THIS file)
// ============================================================================
template void PoissonSubsystem<2>::assemble_matrix();
template void PoissonSubsystem<3>::assemble_matrix();

template void PoissonSubsystem<2>::initialize_preconditioner();
template void PoissonSubsystem<3>::initialize_preconditioner();

template void PoissonSubsystem<2>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&,
    double);
template void PoissonSubsystem<3>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&,
    double);