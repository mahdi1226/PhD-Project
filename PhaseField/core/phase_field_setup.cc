// ============================================================================
// core/phase_field_setup.cc - Setup Methods for PhaseFieldProblem
//
// FIXED VERSION: Added index maps and combined constraints for coupled systems
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "phase_field.h"
#include "output/logger.h"
#include "physics/initial_conditions.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

// ============================================================================
// setup_mesh()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_mesh()
{
    Logger::info("    setup_mesh() started");

    const double x_min = params_.domain.x_min;
    const double x_max = params_.domain.x_max;
    const double y_min = params_.domain.y_min;
    const double y_max = params_.domain.y_max;

    Logger::info("      Domain: [" + std::to_string(x_min) + ", " +
                 std::to_string(x_max) + "] × [" +
                 std::to_string(y_min) + ", " + std::to_string(y_max) + "]");

    // Create rectangular mesh
    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation_,
        {params_.domain.nx_base, params_.domain.ny_base},
        dealii::Point<dim>(x_min, y_min),
        dealii::Point<dim>(x_max, y_max));

    Logger::info("      Base mesh: " + std::to_string(params_.domain.nx_base) +
                 " × " + std::to_string(params_.domain.ny_base) + " elements");

    // Assign boundary IDs: 0=bottom, 1=right, 2=top, 3=left
    const double tol = 1e-10;
    for (auto& cell : triangulation_.active_cell_iterators())
    {
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (cell->face(f)->at_boundary())
            {
                const auto center = cell->face(f)->center();
                if (std::abs(center[1] - y_min) < tol)
                    cell->face(f)->set_boundary_id(0);  // bottom
                else if (std::abs(center[0] - x_max) < tol)
                    cell->face(f)->set_boundary_id(1);  // right
                else if (std::abs(center[1] - y_max) < tol)
                    cell->face(f)->set_boundary_id(2);  // top
                else if (std::abs(center[0] - x_min) < tol)
                    cell->face(f)->set_boundary_id(3);  // left
            }
        }
    }

    // Initial refinement
    triangulation_.refine_global(params_.domain.initial_refinement);

    Logger::info("      Initial refinement level: " +
                 std::to_string(params_.domain.initial_refinement));
    Logger::info("      Total cells: " +
                 std::to_string(triangulation_.n_active_cells()));

    Logger::success("    setup_mesh() completed");
}

// ============================================================================
// setup_dof_handlers()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_dof_handlers()
{
    Logger::info("    setup_dof_handlers() started");

    // Distribute DoFs for all fields
    theta_dof_handler_.distribute_dofs(fe_Q2_);
    psi_dof_handler_.distribute_dofs(fe_Q2_);
    mx_dof_handler_.distribute_dofs(fe_Q2_);
    my_dof_handler_.distribute_dofs(fe_Q2_);
    phi_dof_handler_.distribute_dofs(fe_Q2_);
    ux_dof_handler_.distribute_dofs(fe_Q2_);
    uy_dof_handler_.distribute_dofs(fe_Q2_);
    p_dof_handler_.distribute_dofs(fe_Q1_);

    Logger::info("      θ DoFs: " + std::to_string(theta_dof_handler_.n_dofs()));
    Logger::info("      ψ DoFs: " + std::to_string(psi_dof_handler_.n_dofs()));
    Logger::info("      m_x, m_y DoFs: " + std::to_string(mx_dof_handler_.n_dofs()) + " each");
    Logger::info("      φ DoFs: " + std::to_string(phi_dof_handler_.n_dofs()));
    Logger::info("      u_x, u_y DoFs: " + std::to_string(ux_dof_handler_.n_dofs()) + " each");
    Logger::info("      p DoFs: " + std::to_string(p_dof_handler_.n_dofs()));

    const unsigned int total = theta_dof_handler_.n_dofs() + psi_dof_handler_.n_dofs()
                             + 2 * mx_dof_handler_.n_dofs() + phi_dof_handler_.n_dofs()
                             + 2 * ux_dof_handler_.n_dofs() + p_dof_handler_.n_dofs();
    Logger::info("      Total DoFs: " + std::to_string(total));

    Logger::success("    setup_dof_handlers() completed");
}

// ============================================================================
// setup_constraints()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_constraints()
{
    Logger::info("    setup_constraints() started");

    // θ: Neumann (no constraints except hanging nodes)
    theta_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);
    theta_constraints_.close();

    // ψ: Neumann
    psi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);
    psi_constraints_.close();

    // m_x, m_y: Neumann
    mx_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(mx_dof_handler_, mx_constraints_);
    mx_constraints_.close();

    my_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(my_dof_handler_, my_constraints_);
    my_constraints_.close();

    // φ: Neumann (pinning handled in assembler)
    phi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);
    phi_constraints_.close();

    // u_x, u_y: No-slip (zero Dirichlet on all boundaries)
    ux_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    for (unsigned int b = 0; b < 4; ++b)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_, b,
            dealii::Functions::ZeroFunction<dim>(),
            ux_constraints_);
    }
    ux_constraints_.close();

    uy_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    for (unsigned int b = 0; b < 4; ++b)
    {
        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_, b,
            dealii::Functions::ZeroFunction<dim>(),
            uy_constraints_);
    }
    uy_constraints_.close();

    // p: No Dirichlet (determined up to constant)
    p_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);
    p_constraints_.close();

    Logger::info("      θ constraints: " + std::to_string(theta_constraints_.n_constraints()));
    Logger::info("      ψ constraints: " + std::to_string(psi_constraints_.n_constraints()));
    Logger::info("      m constraints: " + std::to_string(mx_constraints_.n_constraints()) + " each");
    Logger::info("      φ constraints: " + std::to_string(phi_constraints_.n_constraints()));
    Logger::info("      u constraints: " + std::to_string(ux_constraints_.n_constraints()) + " each");
    Logger::info("      p constraints: " + std::to_string(p_constraints_.n_constraints()));

    Logger::success("    setup_constraints() completed");
}

// ============================================================================
// setup_sparsity_patterns() - WITH INDEX MAPS AND COMBINED CONSTRAINTS
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_sparsity_patterns()
{
    Logger::info("    setup_sparsity_patterns() started");

    // =========================================================================
    // CH system: [θ | ψ] coupled
    // =========================================================================
    Logger::info("      Creating CH sparsity pattern...");
    {
        const unsigned int n_theta = theta_dof_handler_.n_dofs();
        const unsigned int n_psi = psi_dof_handler_.n_dofs();
        const unsigned int n_ch = n_theta + n_psi;

        // Build index maps
        theta_to_ch_map_.resize(n_theta);
        psi_to_ch_map_.resize(n_psi);
        for (unsigned int i = 0; i < n_theta; ++i)
            theta_to_ch_map_[i] = i;
        for (unsigned int i = 0; i < n_psi; ++i)
            psi_to_ch_map_[i] = n_theta + i;

        // Build combined constraints
        ch_combined_constraints_.clear();

        // Copy theta constraints with mapping
        for (unsigned int i = 0; i < n_theta; ++i)
        {
            if (theta_constraints_.is_constrained(i))
            {
                const unsigned int mapped_i = theta_to_ch_map_[i];
                const auto* entries = theta_constraints_.get_constraint_entries(i);
                const double inhom = theta_constraints_.get_inhomogeneity(i);

                ch_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& e : *entries)
                        ch_combined_constraints_.add_entry(mapped_i, theta_to_ch_map_[e.first], e.second);
                }
                ch_combined_constraints_.set_inhomogeneity(mapped_i, inhom);
            }
        }

        // Copy psi constraints with mapping
        for (unsigned int i = 0; i < n_psi; ++i)
        {
            if (psi_constraints_.is_constrained(i))
            {
                const unsigned int mapped_i = psi_to_ch_map_[i];
                const auto* entries = psi_constraints_.get_constraint_entries(i);
                const double inhom = psi_constraints_.get_inhomogeneity(i);

                ch_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& e : *entries)
                        ch_combined_constraints_.add_entry(mapped_i, psi_to_ch_map_[e.first], e.second);
                }
                ch_combined_constraints_.set_inhomogeneity(mapped_i, inhom);
            }
        }
        ch_combined_constraints_.close();

        // Build sparsity pattern
        dealii::DynamicSparsityPattern dsp(n_ch, n_ch);

        // θ-θ coupling
        for (const auto& cell : theta_dof_handler_.active_cell_iterators())
        {
            std::vector<dealii::types::global_dof_index> dofs(fe_Q2_.n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            for (const auto& i : dofs)
                for (const auto& j : dofs)
                    dsp.add(theta_to_ch_map_[i], theta_to_ch_map_[j]);
        }

        // ψ-ψ coupling
        for (const auto& cell : psi_dof_handler_.active_cell_iterators())
        {
            std::vector<dealii::types::global_dof_index> dofs(fe_Q2_.n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            for (const auto& i : dofs)
                for (const auto& j : dofs)
                    dsp.add(psi_to_ch_map_[i], psi_to_ch_map_[j]);
        }

        // θ-ψ and ψ-θ coupling
        auto theta_cell = theta_dof_handler_.begin_active();
        auto psi_cell = psi_dof_handler_.begin_active();
        for (; theta_cell != theta_dof_handler_.end(); ++theta_cell, ++psi_cell)
        {
            std::vector<dealii::types::global_dof_index> theta_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> psi_dofs(fe_Q2_.n_dofs_per_cell());
            theta_cell->get_dof_indices(theta_dofs);
            psi_cell->get_dof_indices(psi_dofs);

            for (const auto& i : theta_dofs)
                for (const auto& j : psi_dofs)
                {
                    dsp.add(theta_to_ch_map_[i], psi_to_ch_map_[j]);
                    dsp.add(psi_to_ch_map_[j], theta_to_ch_map_[i]);
                }
        }

        // Condense with combined constraints
        ch_combined_constraints_.condense(dsp);

        ch_sparsity_.copy_from(dsp);
        ch_matrix_.reinit(ch_sparsity_);
        ch_rhs_.reinit(n_ch);

        Logger::info("        CH system size: " + std::to_string(n_ch) +
                     ", nnz: " + std::to_string(ch_sparsity_.n_nonzero_elements()));
    }

    // =========================================================================
    // NS system: [u_x | u_y | p] coupled
    // =========================================================================
    Logger::info("      Creating NS sparsity pattern...");
    {
        const unsigned int n_ux = ux_dof_handler_.n_dofs();
        const unsigned int n_uy = uy_dof_handler_.n_dofs();
        const unsigned int n_p = p_dof_handler_.n_dofs();
        const unsigned int n_ns = n_ux + n_uy + n_p;

        // Build index maps
        ux_to_ns_map_.resize(n_ux);
        uy_to_ns_map_.resize(n_uy);
        p_to_ns_map_.resize(n_p);
        for (unsigned int i = 0; i < n_ux; ++i)
            ux_to_ns_map_[i] = i;
        for (unsigned int i = 0; i < n_uy; ++i)
            uy_to_ns_map_[i] = n_ux + i;
        for (unsigned int i = 0; i < n_p; ++i)
            p_to_ns_map_[i] = n_ux + n_uy + i;

        // Build combined constraints
        ns_combined_constraints_.clear();

        // Copy ux constraints
        for (unsigned int i = 0; i < n_ux; ++i)
        {
            if (ux_constraints_.is_constrained(i))
            {
                const unsigned int mapped_i = ux_to_ns_map_[i];
                const auto* entries = ux_constraints_.get_constraint_entries(i);
                const double inhom = ux_constraints_.get_inhomogeneity(i);

                ns_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& e : *entries)
                        ns_combined_constraints_.add_entry(mapped_i, ux_to_ns_map_[e.first], e.second);
                }
                ns_combined_constraints_.set_inhomogeneity(mapped_i, inhom);
            }
        }

        // Copy uy constraints
        for (unsigned int i = 0; i < n_uy; ++i)
        {
            if (uy_constraints_.is_constrained(i))
            {
                const unsigned int mapped_i = uy_to_ns_map_[i];
                const auto* entries = uy_constraints_.get_constraint_entries(i);
                const double inhom = uy_constraints_.get_inhomogeneity(i);

                ns_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& e : *entries)
                        ns_combined_constraints_.add_entry(mapped_i, uy_to_ns_map_[e.first], e.second);
                }
                ns_combined_constraints_.set_inhomogeneity(mapped_i, inhom);
            }
        }

        // Copy p constraints
        for (unsigned int i = 0; i < n_p; ++i)
        {
            if (p_constraints_.is_constrained(i))
            {
                const unsigned int mapped_i = p_to_ns_map_[i];
                const auto* entries = p_constraints_.get_constraint_entries(i);
                const double inhom = p_constraints_.get_inhomogeneity(i);

                ns_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& e : *entries)
                        ns_combined_constraints_.add_entry(mapped_i, p_to_ns_map_[e.first], e.second);
                }
                ns_combined_constraints_.set_inhomogeneity(mapped_i, inhom);
            }
        }
        ns_combined_constraints_.close();

        // Build sparsity pattern
        dealii::DynamicSparsityPattern dsp(n_ns, n_ns);

        // ux-ux, uy-uy coupling
        for (const auto& cell : ux_dof_handler_.active_cell_iterators())
        {
            std::vector<dealii::types::global_dof_index> dofs(fe_Q2_.n_dofs_per_cell());
            cell->get_dof_indices(dofs);
            for (const auto& i : dofs)
                for (const auto& j : dofs)
                {
                    dsp.add(ux_to_ns_map_[i], ux_to_ns_map_[j]);
                    dsp.add(uy_to_ns_map_[i], uy_to_ns_map_[j]);
                }
        }

        // ux-uy, uy-ux coupling (for grad-div stabilization)
        auto ux_cell = ux_dof_handler_.begin_active();
        auto uy_cell = uy_dof_handler_.begin_active();
        for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++uy_cell)
        {
            std::vector<dealii::types::global_dof_index> ux_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> uy_dofs(fe_Q2_.n_dofs_per_cell());
            ux_cell->get_dof_indices(ux_dofs);
            uy_cell->get_dof_indices(uy_dofs);

            for (const auto& i : ux_dofs)
                for (const auto& j : uy_dofs)
                {
                    dsp.add(ux_to_ns_map_[i], uy_to_ns_map_[j]);
                    dsp.add(uy_to_ns_map_[j], ux_to_ns_map_[i]);
                }
        }

        // Velocity-pressure coupling
        ux_cell = ux_dof_handler_.begin_active();
        auto p_cell = p_dof_handler_.begin_active();
        for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++p_cell)
        {
            std::vector<dealii::types::global_dof_index> ux_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> p_dofs(fe_Q1_.n_dofs_per_cell());
            ux_cell->get_dof_indices(ux_dofs);
            p_cell->get_dof_indices(p_dofs);

            for (const auto& i : ux_dofs)
                for (const auto& j : p_dofs)
                {
                    dsp.add(ux_to_ns_map_[i], p_to_ns_map_[j]);
                    dsp.add(uy_to_ns_map_[i], p_to_ns_map_[j]);
                    dsp.add(p_to_ns_map_[j], ux_to_ns_map_[i]);
                    dsp.add(p_to_ns_map_[j], uy_to_ns_map_[i]);
                }
        }

        // Condense with combined constraints
        ns_combined_constraints_.condense(dsp);

        ns_sparsity_.copy_from(dsp);
        ns_matrix_.reinit(ns_sparsity_);
        ns_rhs_.reinit(n_ns);

        Logger::info("        NS system size: " + std::to_string(n_ns) +
                     ", nnz: " + std::to_string(ns_sparsity_.n_nonzero_elements()));
    }

    // =========================================================================
    // Magnetization system (lumped mass - diagonal)
    // =========================================================================
    Logger::info("      Creating magnetization sparsity pattern...");
    {
        dealii::DynamicSparsityPattern dsp(mx_dof_handler_.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(mx_dof_handler_, dsp, mx_constraints_, false);
        mag_sparsity_.copy_from(dsp);
        mag_matrix_.reinit(mag_sparsity_);
        mag_rhs_x_.reinit(mx_dof_handler_.n_dofs());
        mag_rhs_y_.reinit(my_dof_handler_.n_dofs());

        Logger::info("        Mag system size: " + std::to_string(mx_dof_handler_.n_dofs()) +
                     ", nnz: " + std::to_string(mag_sparsity_.n_nonzero_elements()));
    }

    // =========================================================================
    // Poisson system
    // =========================================================================
    Logger::info("      Creating Poisson sparsity pattern...");
    {
        dealii::DynamicSparsityPattern dsp(phi_dof_handler_.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(phi_dof_handler_, dsp, phi_constraints_, false);
        poisson_sparsity_.copy_from(dsp);
        poisson_matrix_.reinit(poisson_sparsity_);
        poisson_rhs_.reinit(phi_dof_handler_.n_dofs());

        Logger::info("        Poisson system size: " + std::to_string(phi_dof_handler_.n_dofs()) +
                     ", nnz: " + std::to_string(poisson_sparsity_.n_nonzero_elements()));
    }

    Logger::success("    setup_sparsity_patterns() completed");
}

// ============================================================================
// initialize_solutions()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::initialize_solutions()
{
    Logger::info("    initialize_solutions() started");

    // Resize all solution vectors
    Logger::info("      Resizing solution vectors...");
    theta_solution_.reinit(theta_dof_handler_.n_dofs());
    theta_old_solution_.reinit(theta_dof_handler_.n_dofs());
    psi_solution_.reinit(psi_dof_handler_.n_dofs());
    mx_solution_.reinit(mx_dof_handler_.n_dofs());
    my_solution_.reinit(my_dof_handler_.n_dofs());
    mx_old_solution_.reinit(mx_dof_handler_.n_dofs());
    my_old_solution_.reinit(my_dof_handler_.n_dofs());
    phi_solution_.reinit(phi_dof_handler_.n_dofs());
    ux_solution_.reinit(ux_dof_handler_.n_dofs());
    uy_solution_.reinit(uy_dof_handler_.n_dofs());
    ux_old_solution_.reinit(ux_dof_handler_.n_dofs());
    uy_old_solution_.reinit(uy_dof_handler_.n_dofs());
    p_solution_.reinit(p_dof_handler_.n_dofs());

    // Initialize θ with tanh profile (ferrofluid pool)
    Logger::info("      Initializing θ (tanh profile)...");
    InitialTheta<dim> initial_theta(params_.ic.pool_depth, params_.ch.epsilon);
    dealii::VectorTools::interpolate(theta_dof_handler_, initial_theta, theta_solution_);

    // Initialize ψ to equilibrium value
    Logger::info("      Initializing ψ (equilibrium)...");
    InitialPsi<dim> initial_psi(params_.ic.pool_depth, params_.ch.epsilon);
    dealii::VectorTools::interpolate(psi_dof_handler_, initial_psi, psi_solution_);

    // Initialize m to zero (will be computed from equilibrium)
    Logger::info("      Initializing m (zero)...");
    mx_solution_ = 0;
    my_solution_ = 0;

    // Initialize u to zero
    Logger::info("      Initializing u (zero)...");
    ux_solution_ = 0;
    uy_solution_ = 0;
    p_solution_ = 0;

    // Initialize φ to zero (will be computed from Poisson)
    Logger::info("      Initializing φ (zero)...");
    phi_solution_ = 0;

    // Copy to old solutions
    Logger::info("      Copying to old solutions...");
    theta_old_solution_ = theta_solution_;
    mx_old_solution_ = mx_solution_;
    my_old_solution_ = my_solution_;
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;

    // Apply constraints
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);
    mx_constraints_.distribute(mx_solution_);
    my_constraints_.distribute(my_solution_);
    phi_constraints_.distribute(phi_solution_);
    ux_constraints_.distribute(ux_solution_);
    uy_constraints_.distribute(uy_solution_);
    p_constraints_.distribute(p_solution_);

    Logger::success("    initialize_solutions() completed");
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;
// template class PhaseFieldProblem<3>;