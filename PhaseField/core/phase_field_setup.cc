// ============================================================================
// core/phase_field_setup.cc - Setup Methods for PhaseFieldProblem
//
// This file contains:
//   - setup_mesh()
//   - setup_dof_handlers()
//   - setup_constraints()
//   - setup_sparsity_patterns()
//   - initialize_solutions()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2 for domain and mesh parameters
// ============================================================================

#include "phase_field.h"
#include "output/logger.h"
#include "physics/initial_conditions.h"
#include "physics/boundary_conditions.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <filesystem>

// ============================================================================
// setup_mesh()
//
// Domain: Ω = [0, 1] × [0, 0.6] (Section 6.2, p.520)
// Initial mesh: subdivisions to get ~10 × 6 base elements
// Then refine globally to desired level
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_mesh()
{
    Logger::info("    setup_mesh() started");

    // Create rectangular domain
    // Ω = [x_min, x_max] × [y_min, y_max]
    // Default: [0, 1] × [0, 0.6] per Section 6.2, p.520
    const dealii::Point<dim> p1(params_.domain.x_min, params_.domain.y_min);
    const dealii::Point<dim> p2(params_.domain.x_max, params_.domain.y_max);

    Logger::info("      Domain: [" + std::to_string(params_.domain.x_min) + ", " +
                 std::to_string(params_.domain.x_max) + "] × [" +
                 std::to_string(params_.domain.y_min) + ", " +
                 std::to_string(params_.domain.y_max) + "]");

    // Subdivisions: roughly 10 × 6 base mesh for [0,1] × [0,0.6]
    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = 10;  // x-direction
    if constexpr (dim >= 2)
        subdivisions[1] = 6;   // y-direction
    if constexpr (dim >= 3)
        subdivisions[2] = 6;   // z-direction

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation_,
                                                       subdivisions,
                                                       p1, p2);

    Logger::info("      Base mesh: " + std::to_string(subdivisions[0]) + " × " +
                 std::to_string(subdivisions[1]) + " elements");

    // Initial global refinement
    triangulation_.refine_global(params_.domain.initial_refinement);

    Logger::info("      Initial refinement level: " +
                 std::to_string(params_.domain.initial_refinement));
    Logger::info("      Total cells: " +
                 std::to_string(triangulation_.n_active_cells()));

    // Create output directory
    std::filesystem::create_directories(params_.output.folder);

    Logger::success("    setup_mesh() completed");
}

// ============================================================================
// setup_dof_handlers()
//
// Each scalar field has its own DoFHandler sharing the same triangulation.
// FE degrees: Q2 for θ, ψ, m, φ, u; Q1 for p (Section 6.2, p.519)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_dof_handlers()
{
    Logger::info("    setup_dof_handlers() started");

    // θ (theta) - Phase field, Q2
    theta_dof_handler_.distribute_dofs(fe_Q2_);
    Logger::info("      θ DoFs: " + std::to_string(theta_dof_handler_.n_dofs()));

    // ψ (psi) - Chemical potential, Q2
    psi_dof_handler_.distribute_dofs(fe_Q2_);
    Logger::info("      ψ DoFs: " + std::to_string(psi_dof_handler_.n_dofs()));

    // m_x, m_y - Magnetization components, Q2
    mx_dof_handler_.distribute_dofs(fe_Q2_);
    my_dof_handler_.distribute_dofs(fe_Q2_);
    Logger::info("      m_x, m_y DoFs: " + std::to_string(mx_dof_handler_.n_dofs()) + " each");

    // φ (phi) - Magnetic potential, Q2
    phi_dof_handler_.distribute_dofs(fe_Q2_);
    Logger::info("      φ DoFs: " + std::to_string(phi_dof_handler_.n_dofs()));

    // u_x, u_y - Velocity components, Q2
    ux_dof_handler_.distribute_dofs(fe_Q2_);
    uy_dof_handler_.distribute_dofs(fe_Q2_);
    Logger::info("      u_x, u_y DoFs: " + std::to_string(ux_dof_handler_.n_dofs()) + " each");

    // p - Pressure, Q1
    p_dof_handler_.distribute_dofs(fe_Q1_);
    Logger::info("      p DoFs: " + std::to_string(p_dof_handler_.n_dofs()));

    // Total DoFs
    const unsigned int total = theta_dof_handler_.n_dofs() + psi_dof_handler_.n_dofs() +
                               mx_dof_handler_.n_dofs() + my_dof_handler_.n_dofs() +
                               phi_dof_handler_.n_dofs() +
                               ux_dof_handler_.n_dofs() + uy_dof_handler_.n_dofs() +
                               p_dof_handler_.n_dofs();
    Logger::info("      Total DoFs: " + std::to_string(total));

    Logger::success("    setup_dof_handlers() completed");
}

// ============================================================================
// setup_constraints()
//
// Boundary conditions (Eq. 15, p.501):
//   ∂_n θ = 0, ∂_n ψ = 0  (homogeneous Neumann - natural, no constraints)
//   u = 0                  (no-slip Dirichlet on all boundaries)
//   ∂_n φ = (h_a - m)·n   (Neumann - handled in assembly)
//   m: no BC needed        (advection-reaction, no Laplacian term)
//   p: no explicit BC      (determined up to constant)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_constraints()
{
    Logger::info("    setup_constraints() started");

    // θ constraints - homogeneous Neumann (natural BC, only hanging nodes)
    theta_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);
    theta_constraints_.close();
    Logger::info("      θ constraints: " + std::to_string(theta_constraints_.n_constraints()));

    // ψ constraints - homogeneous Neumann (natural BC, only hanging nodes)
    psi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);
    psi_constraints_.close();
    Logger::info("      ψ constraints: " + std::to_string(psi_constraints_.n_constraints()));

    // m constraints - no BC (only hanging nodes)
    mx_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(mx_dof_handler_, mx_constraints_);
    mx_constraints_.close();

    my_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(my_dof_handler_, my_constraints_);
    my_constraints_.close();
    Logger::info("      m constraints: " + std::to_string(mx_constraints_.n_constraints()) + " each");

    // φ constraints - Neumann (natural BC, only hanging nodes)
    // Note: Poisson with pure Neumann needs pinning - handled in solver
    phi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);
    phi_constraints_.close();
    Logger::info("      φ constraints: " + std::to_string(phi_constraints_.n_constraints()));

    // u constraints - no-slip Dirichlet on ALL boundaries (Eq. 15)
    ux_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::VectorTools::interpolate_boundary_values(ux_dof_handler_,
                                                      0,  // boundary_id (all)
                                                      dealii::Functions::ZeroFunction<dim>(),
                                                      ux_constraints_);
    ux_constraints_.close();

    uy_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    dealii::VectorTools::interpolate_boundary_values(uy_dof_handler_,
                                                      0,
                                                      dealii::Functions::ZeroFunction<dim>(),
                                                      uy_constraints_);
    uy_constraints_.close();
    Logger::info("      u constraints: " + std::to_string(ux_constraints_.n_constraints()) + " each");

    // p constraints - only hanging nodes (pressure determined up to constant)
    p_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);
    p_constraints_.close();
    Logger::info("      p constraints: " + std::to_string(p_constraints_.n_constraints()));

    Logger::success("    setup_constraints() completed");
}

// ============================================================================
// setup_sparsity_patterns()
//
// Create sparsity patterns for coupled systems:
//   - CH system: 2×2 block for (θ, ψ) coupling
//   - NS system: 3×3 block for (u_x, u_y, p) coupling
//   - Magnetization: m_x, m_y (may be decoupled or equilibrium)
//   - Poisson: scalar system for φ
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_sparsity_patterns()
{
    Logger::info("    setup_sparsity_patterns() started");

    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_psi = psi_dof_handler_.n_dofs();
    const unsigned int n_ux = ux_dof_handler_.n_dofs();
    const unsigned int n_uy = uy_dof_handler_.n_dofs();
    const unsigned int n_p = p_dof_handler_.n_dofs();
    const unsigned int n_phi = phi_dof_handler_.n_dofs();
    const unsigned int n_mx = mx_dof_handler_.n_dofs();

    // -------------------------------------------------------------------------
    // CH sparsity pattern (θ-ψ coupled system)
    // Size: (n_theta + n_psi) × (n_theta + n_psi)
    // -------------------------------------------------------------------------
    {
        Logger::info("      Creating CH sparsity pattern...");
        const unsigned int n_ch = n_theta + n_psi;
        dealii::DynamicSparsityPattern dsp(n_ch, n_ch);

        // θ-θ block (mass + possibly stiffness from stabilization)
        // ψ-ψ block (mass)
        // θ-ψ block (coupling: γ∇ψ·∇Λ term)
        // ψ-θ block (coupling: ε∇θ·∇Υ + (1/η)θ·Υ terms)

        // For now, use full coupling - both fields share same mesh/FE
        // so their sparsity patterns are identical
        for (const auto& cell : theta_dof_handler_.active_cell_iterators())
        {
            std::vector<dealii::types::global_dof_index> theta_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> psi_dofs(fe_Q2_.n_dofs_per_cell());

            cell->get_dof_indices(theta_dofs);

            // Get corresponding psi cell (same cell, different handler)
            typename dealii::DoFHandler<dim>::active_cell_iterator
                psi_cell(&triangulation_, cell->level(), cell->index(), &psi_dof_handler_);
            psi_cell->get_dof_indices(psi_dofs);

            // All 4 blocks: θ-θ, θ-ψ, ψ-θ, ψ-ψ
            for (unsigned int i = 0; i < fe_Q2_.n_dofs_per_cell(); ++i)
            {
                for (unsigned int j = 0; j < fe_Q2_.n_dofs_per_cell(); ++j)
                {
                    // θ-θ block (upper left)
                    dsp.add(theta_dofs[i], theta_dofs[j]);
                    // θ-ψ block (upper right)
                    dsp.add(theta_dofs[i], n_theta + psi_dofs[j]);
                    // ψ-θ block (lower left)
                    dsp.add(n_theta + psi_dofs[i], theta_dofs[j]);
                    // ψ-ψ block (lower right)
                    dsp.add(n_theta + psi_dofs[i], n_theta + psi_dofs[j]);
                }
            }
        }

        ch_sparsity_.copy_from(dsp);
        ch_matrix_.reinit(ch_sparsity_);
        ch_rhs_.reinit(n_ch);

        Logger::info("        CH system size: " + std::to_string(n_ch) +
                     ", nnz: " + std::to_string(ch_sparsity_.n_nonzero_elements()));
    }

    // -------------------------------------------------------------------------
    // NS sparsity pattern (u_x-u_y-p coupled system)
    // Size: (n_ux + n_uy + n_p) × (n_ux + n_uy + n_p)
    // -------------------------------------------------------------------------
    {
        Logger::info("      Creating NS sparsity pattern...");
        const unsigned int n_ns = n_ux + n_uy + n_p;
        dealii::DynamicSparsityPattern dsp(n_ns, n_ns);

        // Build all blocks
        for (const auto& cell : ux_dof_handler_.active_cell_iterators())
        {
            std::vector<dealii::types::global_dof_index> ux_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> uy_dofs(fe_Q2_.n_dofs_per_cell());
            std::vector<dealii::types::global_dof_index> p_dofs(fe_Q1_.n_dofs_per_cell());

            cell->get_dof_indices(ux_dofs);

            typename dealii::DoFHandler<dim>::active_cell_iterator
                uy_cell(&triangulation_, cell->level(), cell->index(), &uy_dof_handler_);
            uy_cell->get_dof_indices(uy_dofs);

            typename dealii::DoFHandler<dim>::active_cell_iterator
                p_cell(&triangulation_, cell->level(), cell->index(), &p_dof_handler_);
            p_cell->get_dof_indices(p_dofs);

            // Velocity-velocity blocks (Q2-Q2)
            for (unsigned int i = 0; i < fe_Q2_.n_dofs_per_cell(); ++i)
            {
                for (unsigned int j = 0; j < fe_Q2_.n_dofs_per_cell(); ++j)
                {
                    // u_x - u_x
                    dsp.add(ux_dofs[i], ux_dofs[j]);
                    // u_x - u_y
                    dsp.add(ux_dofs[i], n_ux + uy_dofs[j]);
                    // u_y - u_x
                    dsp.add(n_ux + uy_dofs[i], ux_dofs[j]);
                    // u_y - u_y
                    dsp.add(n_ux + uy_dofs[i], n_ux + uy_dofs[j]);
                }
            }

            // Velocity-pressure blocks (Q2-Q1)
            for (unsigned int i = 0; i < fe_Q2_.n_dofs_per_cell(); ++i)
            {
                for (unsigned int j = 0; j < fe_Q1_.n_dofs_per_cell(); ++j)
                {
                    // u_x - p
                    dsp.add(ux_dofs[i], n_ux + n_uy + p_dofs[j]);
                    // u_y - p
                    dsp.add(n_ux + uy_dofs[i], n_ux + n_uy + p_dofs[j]);
                    // p - u_x
                    dsp.add(n_ux + n_uy + p_dofs[j], ux_dofs[i]);
                    // p - u_y
                    dsp.add(n_ux + n_uy + p_dofs[j], n_ux + uy_dofs[i]);
                }
            }

            // Pressure-pressure block (Q1-Q1) - for stabilization if needed
            for (unsigned int i = 0; i < fe_Q1_.n_dofs_per_cell(); ++i)
            {
                for (unsigned int j = 0; j < fe_Q1_.n_dofs_per_cell(); ++j)
                {
                    dsp.add(n_ux + n_uy + p_dofs[i], n_ux + n_uy + p_dofs[j]);
                }
            }
        }

        ns_sparsity_.copy_from(dsp);
        ns_matrix_.reinit(ns_sparsity_);
        ns_rhs_.reinit(n_ns);

        Logger::info("        NS system size: " + std::to_string(n_ns) +
                     ", nnz: " + std::to_string(ns_sparsity_.n_nonzero_elements()));
    }

    // -------------------------------------------------------------------------
    // Magnetization sparsity (for full PDE; equilibrium mode doesn't need this)
    // -------------------------------------------------------------------------
    {
        Logger::info("      Creating magnetization sparsity pattern...");
        dealii::DynamicSparsityPattern dsp(n_mx);
        dealii::DoFTools::make_sparsity_pattern(mx_dof_handler_, dsp);

        mag_sparsity_.copy_from(dsp);
        mag_matrix_.reinit(mag_sparsity_);
        mag_rhs_.reinit(n_mx);

        Logger::info("        Mag system size: " + std::to_string(n_mx) +
                     ", nnz: " + std::to_string(mag_sparsity_.n_nonzero_elements()));
    }

    // -------------------------------------------------------------------------
    // Poisson sparsity (scalar φ system)
    // -------------------------------------------------------------------------
    {
        Logger::info("      Creating Poisson sparsity pattern...");
        dealii::DynamicSparsityPattern dsp(n_phi);
        dealii::DoFTools::make_sparsity_pattern(phi_dof_handler_, dsp);

        poisson_sparsity_.copy_from(dsp);
        poisson_matrix_.reinit(poisson_sparsity_);
        poisson_rhs_.reinit(n_phi);

        Logger::info("        Poisson system size: " + std::to_string(n_phi) +
                     ", nnz: " + std::to_string(poisson_sparsity_.n_nonzero_elements()));
    }

    Logger::success("    setup_sparsity_patterns() completed");
}

// ============================================================================
// initialize_solutions()
//
// Initial conditions (Eq. 41, p.505):
//   θ^0 = tanh((y - 0.2)/(ε√2))  - Pool of ferrofluid at bottom
//   ψ^0 = -(1/ε)(θ³ - θ)         - From equilibrium
//   m^0 = 0                       - Since h_a(0) = 0 (α_s ramps from 0)
//   u^0 = 0                       - Fluid at rest
//   p^0 = 0                       - Reference pressure
//   φ^0 = 0                       - Will be computed from Poisson
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

    // Initialize θ - tanh profile for ferrofluid pool
    // θ₀(x,y) = tanh((y - pool_depth)/(ε√2))
    // θ ≈ +1 below interface (ferrofluid), θ ≈ -1 above (non-magnetic)
    Logger::info("      Initializing θ (tanh profile)...");
    {
        InitialTheta<dim> ic_theta(params_.ic.pool_depth, params_.ch.epsilon);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_theta, theta_solution_);
        theta_constraints_.distribute(theta_solution_);
    }

    // Initialize ψ - equilibrium chemical potential
    // ψ₀ ≈ -(1/ε)(θ³ - θ)
    Logger::info("      Initializing ψ (equilibrium)...");
    {
        InitialPsi<dim> ic_psi(params_.ic.pool_depth, params_.ch.epsilon);
        dealii::VectorTools::interpolate(psi_dof_handler_, ic_psi, psi_solution_);
        psi_constraints_.distribute(psi_solution_);
    }

    // Initialize m - zero (since h_a(0) = 0 when dipole intensity starts at 0)
    Logger::info("      Initializing m (zero)...");
    mx_solution_ = 0.0;
    my_solution_ = 0.0;

    // Initialize u - zero (fluid at rest)
    Logger::info("      Initializing u (zero)...");
    ux_solution_ = 0.0;
    uy_solution_ = 0.0;

    // Initialize p - zero
    Logger::info("      Initializing p (zero)...");
    p_solution_ = 0.0;

    // Initialize φ - zero (or could solve Poisson, but h_a(0) ≈ 0)
    Logger::info("      Initializing φ (zero)...");
    phi_solution_ = 0.0;

    // Copy to old solutions for time stepping
    Logger::info("      Copying to old solutions...");
    theta_old_solution_ = theta_solution_;
    mx_old_solution_ = mx_solution_;
    my_old_solution_ = my_solution_;
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;

    Logger::success("    initialize_solutions() completed");
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;
//template class PhaseFieldProblem<3>;