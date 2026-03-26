// ============================================================================
// navier_stokes/navier_stokes_setup.cc — Setup Implementation
//
// Implements NSSubsystem<dim>::setup():
//   1. Distribute DoFs for ux (Q2), uy (Q2), p (Q1)
//   2. Extract per-component index sets
//   3. Build velocity constraints (hanging nodes + Dirichlet u=0)
//   4. Build pressure constraints (hanging nodes, mean subtraction)
//   5. Build 3 separate sparsity patterns (no monolithic system)
//   6. Allocate matrices and vectors
//   7. Assemble lumped velocity mass matrix diagonal
//
// Pressure-correction projection method (Zhang Algorithm 3.1).
// Each subsystem (ux, uy, p) has its own matrix and solver.
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// setup() — call once after mesh is ready
// ============================================================================
template <int dim>
void NSSubsystem<dim>::setup()
{
    // Reset cached state (mesh may have changed via AMR)
    ux_amg_valid_ = false;
    uy_amg_valid_ = false;
    p_amg_valid_  = false;
    p_matrix_assembled_ = false;

    // ========================================================================
    // Step 1: Distribute DoFs
    //
    // Velocity:  FE_Q<dim>(2) — Q2 continuous
    // Pressure:  FE_Q<dim>(1) — Q1 continuous (for pressure Poisson)
    // ========================================================================
    ux_dof_handler_.distribute_dofs(fe_velocity_);
    uy_dof_handler_.distribute_dofs(fe_velocity_);
    p_dof_handler_.distribute_dofs(fe_pressure_);

    // Cuthill-McKee renumbering for all 3 (all CG now)
    if (params_.renumber_dofs)
    {
        dealii::DoFRenumbering::Cuthill_McKee(ux_dof_handler_);
        dealii::DoFRenumbering::Cuthill_McKee(uy_dof_handler_);
        dealii::DoFRenumbering::Cuthill_McKee(p_dof_handler_);
    }

    // ========================================================================
    // Step 2: Extract per-component index sets
    // ========================================================================
    ux_locally_owned_    = ux_dof_handler_.locally_owned_dofs();
    ux_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler_);
    uy_locally_owned_    = uy_dof_handler_.locally_owned_dofs();
    uy_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler_);
    p_locally_owned_     = p_dof_handler_.locally_owned_dofs();
    p_locally_relevant_  = dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler_);

    pcout_ << "[NS Setup] DoFs: ux=" << ux_dof_handler_.n_dofs()
           << ", uy=" << uy_dof_handler_.n_dofs()
           << ", p=" << p_dof_handler_.n_dofs() << "\n";

    // ========================================================================
    // Step 3: Build velocity constraints
    //
    // Hanging nodes + homogeneous Dirichlet u=0 on all boundaries (IDs 0-3).
    // ========================================================================
    ux_constraints_.clear();
    ux_constraints_.reinit(ux_locally_owned_, ux_locally_relevant_);
    uy_constraints_.clear();
    uy_constraints_.reinit(uy_locally_owned_, uy_locally_relevant_);

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);

    for (dealii::types::boundary_id bid = 0; bid <= 3; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_, bid,
            dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_, bid,
            dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    }

    ux_constraints_.close();
    uy_constraints_.close();

    // ========================================================================
    // Step 4: Build pressure constraints
    //
    // CG Q1 pressure: hanging node constraints required.
    // No Dirichlet BCs (Neumann natural).
    // Pressure uniqueness via post-solve mean subtraction.
    // ========================================================================
    p_constraints_.clear();
    p_constraints_.reinit(p_locally_owned_, p_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);
    p_constraints_.close();

    // ========================================================================
    // Step 5: Build 3 separate sparsity patterns
    //
    // Each system has its own sparsity pattern — much simpler than monolithic.
    // ========================================================================

    // --- ux sparsity ---
    {
        dealii::DynamicSparsityPattern dsp(ux_locally_relevant_);
        dealii::DoFTools::make_sparsity_pattern(ux_dof_handler_, dsp, ux_constraints_, false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp, ux_locally_owned_, mpi_comm_, ux_locally_relevant_);
        dealii::TrilinosWrappers::SparsityPattern sp;
        sp.reinit(ux_locally_owned_, ux_locally_owned_, dsp, mpi_comm_);
        ux_matrix_.reinit(sp);
        vel_mass_matrix_.reinit(sp);  // same sparsity as ux (same FE, same DoFHandler)
    }

    // --- uy sparsity (same FE as ux, but separate DoFHandler) ---
    {
        dealii::DynamicSparsityPattern dsp(uy_locally_relevant_);
        dealii::DoFTools::make_sparsity_pattern(uy_dof_handler_, dsp, uy_constraints_, false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp, uy_locally_owned_, mpi_comm_, uy_locally_relevant_);
        dealii::TrilinosWrappers::SparsityPattern sp;
        sp.reinit(uy_locally_owned_, uy_locally_owned_, dsp, mpi_comm_);
        uy_matrix_.reinit(sp);
    }

    // --- p sparsity ---
    {
        dealii::DynamicSparsityPattern dsp(p_locally_relevant_);
        dealii::DoFTools::make_sparsity_pattern(p_dof_handler_, dsp, p_constraints_, false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp, p_locally_owned_, mpi_comm_, p_locally_relevant_);
        dealii::TrilinosWrappers::SparsityPattern sp;
        sp.reinit(p_locally_owned_, p_locally_owned_, dsp, mpi_comm_);
        p_matrix_.reinit(sp);
    }

    pcout_ << "[NS Setup] 3 separate matrices: ux("
           << ux_matrix_.m() << "x" << ux_matrix_.n() << "), uy("
           << uy_matrix_.m() << "x" << uy_matrix_.n() << "), p("
           << p_matrix_.m() << "x" << p_matrix_.n() << ")\n";

    // ========================================================================
    // Step 6: Allocate vectors
    // ========================================================================

    // --- RHS vectors (owned) ---
    ux_rhs_.reinit(ux_locally_owned_, mpi_comm_);
    uy_rhs_.reinit(uy_locally_owned_, mpi_comm_);
    p_rhs_.reinit(p_locally_owned_, mpi_comm_);

    // --- Component solutions (owned) ---
    ux_solution_.reinit(ux_locally_owned_, mpi_comm_);
    ux_old_solution_.reinit(ux_locally_owned_, mpi_comm_);
    uy_solution_.reinit(uy_locally_owned_, mpi_comm_);
    uy_old_solution_.reinit(uy_locally_owned_, mpi_comm_);
    p_solution_.reinit(p_locally_owned_, mpi_comm_);
    p_old_solution_.reinit(p_locally_owned_, mpi_comm_);

    // --- Ghosted vectors (for assembly and inter-subsystem reads) ---
    ux_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_comm_);
    uy_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_comm_);
    p_relevant_.reinit(p_locally_owned_, p_locally_relevant_, mpi_comm_);
    ux_old_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_comm_);
    uy_old_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_comm_);
    p_old_relevant_.reinit(p_locally_owned_, p_locally_relevant_, mpi_comm_);

    // --- Lumped velocity mass diagonal (for velocity correction) ---
    vel_mass_lumped_.reinit(ux_locally_owned_, mpi_comm_);

    // ========================================================================
    // Step 7: Assemble lumped velocity mass matrix diagonal
    // ========================================================================
    assemble_lumped_mass();
}

// ============================================================================
// assemble_lumped_mass() — Velocity mass matrix + lumped diagonal
//
// Assembles both:
//   1. Consistent mass matrix: M(i,j) = ∫ φ_i φ_j dx  (for CG solve)
//   2. Lumped diagonal:        M_L(i)  = Σ_j M(i,j)    (for preconditioner)
//
// The consistent mass is used in velocity correction (Step 4) via CG solve:
//   M * δu = δt * ∫ δp ∇φ dx
// The lumped diagonal serves as a Jacobi preconditioner for the CG solve.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_lumped_mass()
{
    vel_mass_matrix_ = 0;
    vel_mass_lumped_ = 0;

    const unsigned int dofs_per_cell = fe_velocity_.n_dofs_per_cell();
    dealii::QGauss<dim> quadrature(fe_velocity_.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe_velocity_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    dealii::FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_lumped(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : ux_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        local_mass = 0;
        local_lumped = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double mass_ij = fe_values.shape_value(i, q) *
                                           fe_values.shape_value(j, q) *
                                           fe_values.JxW(q);
                    local_mass(i, j)  += mass_ij;
                    local_lumped(i)   += mass_ij;  // row-sum lumping
                }
            }

        cell->get_dof_indices(local_dof_indices);

        // Consistent mass: use constraints for hanging nodes + Dirichlet BCs
        // (Dirichlet rows become identity → delta_u = 0 there, which is correct)
        ux_constraints_.distribute_local_to_global(
            local_mass, local_dof_indices, vel_mass_matrix_);

        // Lumped mass: distribute manually (diagonal, no constraints needed)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            if (ux_locally_owned_.is_element(local_dof_indices[i]))
                vel_mass_lumped_(local_dof_indices[i]) += local_lumped(i);
        }
    }

    vel_mass_matrix_.compress(dealii::VectorOperation::add);
    vel_mass_lumped_.compress(dealii::VectorOperation::add);
}

// ============================================================================
// subtract_mean_pressure() — Remove mean pressure for uniqueness
//
// p ← p − (∫p dx / |Ω|)
// ============================================================================
template <int dim>
void NSSubsystem<dim>::subtract_mean_pressure()
{
    // Compute ∫p dx and |Ω| via quadrature
    dealii::QGauss<dim> quadrature(fe_pressure_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_pressure_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> p_values(n_q_points);

    // Need ghosted pressure for evaluation
    dealii::TrilinosWrappers::MPI::Vector p_ghosted(
        p_locally_owned_, p_locally_relevant_, mpi_comm_);
    p_ghosted = p_solution_;

    double local_integral = 0;
    double local_area = 0;

    for (const auto& cell : p_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(p_ghosted, p_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            local_integral += p_values[q] * fe_values.JxW(q);
            local_area     += fe_values.JxW(q);
        }
    }

    double global_integral = 0, global_area = 0;
    MPI_Allreduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    MPI_Allreduce(&local_area,     &global_area,     1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    const double mean_p = (global_area > 0) ? global_integral / global_area : 0;

    // Subtract mean from owned vector
    // Use a temporary buffer to avoid state-tracking issues
    dealii::TrilinosWrappers::MPI::Vector p_correction(p_locally_owned_, mpi_comm_);
    p_correction = mean_p;
    p_solution_.add(-1.0, p_correction);
}

// ============================================================================
// Explicit instantiations (for methods defined in THIS file only)
// ============================================================================
template void NSSubsystem<2>::setup();
template void NSSubsystem<3>::setup();
template void NSSubsystem<2>::assemble_lumped_mass();
template void NSSubsystem<3>::assemble_lumped_mass();
template void NSSubsystem<2>::subtract_mean_pressure();
template void NSSubsystem<3>::subtract_mean_pressure();
