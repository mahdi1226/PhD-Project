// ============================================================================
// core/phase_field_setup.cc - Setup Methods for Phase Field Problem
//
// Setup for CH + Poisson + NS systems.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "diagnostics/ch_mms.h"
#include "setup/ch_setup.h"
#include "setup/poisson_setup.h"
#include "setup/ns_setup.h"
#include "assembly/poisson_assembler.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// setup_mesh()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_mesh()
{
    // Create rectangular domain
    dealii::Point<dim> p1(params_.domain.x_min, params_.domain.y_min);
    dealii::Point<dim> p2(params_.domain.x_max, params_.domain.y_max);

    dealii::GridGenerator::hyper_rectangle(triangulation_, p1, p2);

    // Assign boundary IDs:
    //   0 = bottom (y = y_min)
    //   1 = right  (x = x_max)
    //   2 = top    (y = y_max)
    //   3 = left   (x = x_min)
    for (auto& face : triangulation_.active_face_iterators())
    {
        if (!face->at_boundary())
            continue;

        const auto center = face->center();
        const double tol = 1e-10;

        if (std::abs(center[1] - params_.domain.y_min) < tol)
            face->set_boundary_id(0);  // bottom
        else if (std::abs(center[0] - params_.domain.x_max) < tol)
            face->set_boundary_id(1);  // right
        else if (std::abs(center[1] - params_.domain.y_max) < tol)
            face->set_boundary_id(2);  // top
        else if (std::abs(center[0] - params_.domain.x_min) < tol)
            face->set_boundary_id(3);  // left
    }

    // Global refinement
    triangulation_.refine_global(params_.domain.initial_refinement);

    if (params_.output.verbose)
    {
        std::cout << "[Setup] Mesh created: "
                  << triangulation_.n_active_cells() << " cells\n";
    }
}

// ============================================================================
// setup_dof_handlers()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_dof_handlers()
{
    // ========================================================================
    // CH fields (θ, ψ) - Q2 elements
    // ========================================================================
    theta_dof_handler_.distribute_dofs(fe_Q2_);
    psi_dof_handler_.distribute_dofs(fe_Q2_);

    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_psi = psi_dof_handler_.n_dofs();

    theta_solution_.reinit(n_theta);
    theta_old_solution_.reinit(n_theta);
    psi_solution_.reinit(n_psi);

    // ========================================================================
    // Poisson (φ) - Q2 elements (if magnetic enabled)
    // ========================================================================
    if (params_.magnetic.enabled)
    {
        phi_dof_handler_.distribute_dofs(fe_Q2_);
        phi_solution_.reinit(phi_dof_handler_.n_dofs());
    }

    // ========================================================================
    // NS fields (ux, uy, p) - Q2 velocity, Q1 pressure (Taylor-Hood)
    // ========================================================================
    if (params_.ns.enabled)
    {
        ux_dof_handler_.distribute_dofs(fe_Q2_);
        uy_dof_handler_.distribute_dofs(fe_Q2_);
        p_dof_handler_.distribute_dofs(fe_Q1_);

        const unsigned int n_ux = ux_dof_handler_.n_dofs();
        const unsigned int n_uy = uy_dof_handler_.n_dofs();
        const unsigned int n_p = p_dof_handler_.n_dofs();

        ux_solution_.reinit(n_ux);
        ux_old_solution_.reinit(n_ux);
        uy_solution_.reinit(n_uy);
        uy_old_solution_.reinit(n_uy);
        p_solution_.reinit(n_p);

        // Initialize velocity to zero
        ux_solution_ = 0;
        ux_old_solution_ = 0;
        uy_solution_ = 0;
        uy_old_solution_ = 0;
        p_solution_ = 0;

        if (params_.output.verbose)
        {
            std::cout << "[Setup] DoFs: θ = " << n_theta
                      << ", ψ = " << n_psi
                      << ", total CH = " << (n_theta + n_psi);
            if (params_.magnetic.enabled)
                std::cout << ", φ = " << phi_dof_handler_.n_dofs();
            std::cout << ", ux = " << n_ux
                      << ", uy = " << n_uy
                      << ", p = " << n_p
                      << ", total NS = " << (n_ux + n_uy + n_p) << "\n";
        }
    }
    else
    {
        // Create dummy velocity vectors for CH assembly (zero velocity)
        ux_old_solution_.reinit(n_theta);
        uy_old_solution_.reinit(n_theta);
        ux_old_solution_ = 0;
        uy_old_solution_ = 0;

        if (params_.output.verbose)
        {
            std::cout << "[Setup] DoFs: θ = " << n_theta
                      << ", ψ = " << n_psi
                      << ", total CH = " << (n_theta + n_psi);
            if (params_.magnetic.enabled)
                std::cout << ", φ = " << phi_dof_handler_.n_dofs();
            std::cout << "\n";
        }
    }
}

// ============================================================================
// setup_constraints()
//
// For physical runs: Neumann BCs (natural, no explicit constraints)
// For MMS verification: Dirichlet BCs (θ = θ_exact, ψ = ψ_exact on ∂Ω)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_constraints()
{
    // θ constraints
    theta_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(
        theta_dof_handler_, theta_constraints_);

    // ψ constraints
    psi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(
        psi_dof_handler_, psi_constraints_);

    // For MMS: add Dirichlet BCs at initial time
    if (params_.mms.enabled)
    {
        apply_ch_mms_boundary_constraints<dim>(
            theta_dof_handler_,
            psi_dof_handler_,
            theta_constraints_,
            psi_constraints_,
            params_.mms.t_init);  // Initial time for MMS

        // Set initial time
        const_cast<double&>(time_) = params_.mms.t_init;
    }

    theta_constraints_.close();
    psi_constraints_.close();
}

// ============================================================================
// setup_ch_system()
//
// Creates coupled θ-ψ system by calling the free function in setup/.
// This keeps the setup logic modular and reusable.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ch_system()
{
    // Call the free function to build index maps, constraints, and sparsity
    setup_ch_coupled_system<dim>(
        theta_dof_handler_,
        psi_dof_handler_,
        theta_constraints_,
        psi_constraints_,
        theta_to_ch_map_,
        psi_to_ch_map_,
        ch_combined_constraints_,
        ch_sparsity_,
        params_.output.verbose);

    // Initialize matrix and RHS
    const unsigned int n_total = theta_dof_handler_.n_dofs() + psi_dof_handler_.n_dofs();
    ch_matrix_.reinit(ch_sparsity_);
    ch_rhs_.reinit(n_total);
}

// ============================================================================
// setup_poisson_system()
//
// Sets up the magnetostatic Poisson system for φ.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_poisson_system()
{
    // Initialize constraints (hanging nodes + Dirichlet BCs)
    phi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);

    // Apply dipole Dirichlet BCs at initial time
    apply_poisson_dirichlet_bcs<dim>(
        phi_dof_handler_,
        params_,
        0.0,  // Initial time
        phi_constraints_);

    phi_constraints_.close();

    // Build sparsity pattern
    build_poisson_sparsity<dim>(
        phi_dof_handler_,
        phi_constraints_,
        phi_sparsity_,
        params_.output.verbose);

    // Initialize matrix and RHS
    phi_matrix_.reinit(phi_sparsity_);
    phi_rhs_.reinit(phi_dof_handler_.n_dofs());
}

// ============================================================================
// setup_ns_system()
//
// Sets up the Navier-Stokes system for (ux, uy, p).
// Uses Taylor-Hood elements: Q2 velocity, Q1 pressure.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ns_system()
{
    // ========================================================================
    // Individual field constraints (hanging nodes + no-slip BCs)
    // ========================================================================
    ux_constraints_.clear();
    uy_constraints_.clear();
    p_constraints_.clear();

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);

    // No-slip BCs: u = 0 on all boundaries (boundary IDs 0, 1, 2, 3)
    for (unsigned int boundary_id = 0; boundary_id <= 3; ++boundary_id)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_,
            boundary_id,
            dealii::Functions::ZeroFunction<dim>(),
            ux_constraints_);

        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_,
            boundary_id,
            dealii::Functions::ZeroFunction<dim>(),
            uy_constraints_);
    }

    // No pressure BC (pressure is determined up to a constant)
    // Could fix one pressure DoF if needed for uniqueness

    ux_constraints_.close();
    uy_constraints_.close();
    p_constraints_.close();

    // ========================================================================
    // Build coupled NS system
    // ========================================================================
    setup_ns_coupled_system<dim>(
        ux_dof_handler_,
        uy_dof_handler_,
        p_dof_handler_,
        ux_constraints_,
        uy_constraints_,
        p_constraints_,
        ux_to_ns_map_,
        uy_to_ns_map_,
        p_to_ns_map_,
        ns_combined_constraints_,
        ns_sparsity_,
        params_.output.verbose);

    // Initialize matrix and vectors
    const unsigned int n_ns = ux_dof_handler_.n_dofs() +
                               uy_dof_handler_.n_dofs() +
                               p_dof_handler_.n_dofs();
    ns_matrix_.reinit(ns_sparsity_);
    ns_rhs_.reinit(n_ns);
    ns_solution_.reinit(n_ns);
}

// ============================================================================
// initialize_solutions()
//
// IC types for physical runs (p.522):
//   0 = Circular droplet (for testing)
//   1 = Flat ferrofluid layer (Rosensweig baseline)
//   2 = Perturbed layer (Rosensweig with perturbation)
//
// For MMS: use exact solutions at t_init
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::initialize_solutions()
{
    // MMS mode: use exact solutions at t_init
    if (params_.mms.enabled)
    {
        apply_ch_mms_initial_conditions<dim>(
            theta_dof_handler_,
            psi_dof_handler_,
            theta_solution_,
            psi_solution_,
            params_.mms.t_init);

        theta_old_solution_ = theta_solution_;

        if (params_.output.verbose)
        {
            std::cout << "[Setup] MMS IC at t = " << params_.mms.t_init << "\n";
            std::cout << "[Setup] Initial mass = " << compute_mass() << "\n";
        }
        return;
    }

    // Physical IC modes
    const double epsilon = params_.ch.epsilon;
    const int ic_type = params_.ic.type;
    const double pool_depth = params_.ic.pool_depth;
    const double y_max = params_.domain.y_max;
    const double y_min = params_.domain.y_min;

    // Interface position for layer IC
    const double interface_y = y_min + pool_depth * (y_max - y_min);

    // IC based on type
    if (ic_type == 0)
    {
        // Circular droplet centered in domain
        const double cx = 0.5 * (params_.domain.x_min + params_.domain.x_max);
        const double cy = 0.5 * (params_.domain.y_min + params_.domain.y_max);
        const double radius = 0.2;

        class DropletIC : public dealii::Function<dim>
        {
        public:
            double cx_, cy_, radius_, eps_;
            DropletIC(double cx, double cy, double r, double e)
                : dealii::Function<dim>(1), cx_(cx), cy_(cy), radius_(r), eps_(e) {}

            double value(const dealii::Point<dim>& p, unsigned int = 0) const override
            {
                const double dist = std::sqrt((p[0]-cx_)*(p[0]-cx_) + (p[1]-cy_)*(p[1]-cy_));
                return -std::tanh((dist - radius_) / (std::sqrt(2.0) * eps_));
            }
        };

        DropletIC ic_func(cx, cy, radius, epsilon);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);
    }
    else if (ic_type == 1)
    {
        // Flat ferrofluid layer: θ = +1 below interface, θ = -1 above
        class FlatLayerIC : public dealii::Function<dim>
        {
        public:
            double interface_y_, eps_;
            FlatLayerIC(double y, double e)
                : dealii::Function<dim>(1), interface_y_(y), eps_(e) {}

            double value(const dealii::Point<dim>& p, unsigned int = 0) const override
            {
                return -std::tanh((p[1] - interface_y_) / (std::sqrt(2.0) * eps_));
            }
        };

        FlatLayerIC ic_func(interface_y, epsilon);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);
    }
    else if (ic_type == 2)
    {
        // Perturbed layer (Rosensweig)
        const double perturbation = params_.ic.perturbation;
        const int n_modes = params_.ic.perturbation_modes;
        const double Lx = params_.domain.x_max - params_.domain.x_min;

        class PerturbedLayerIC : public dealii::Function<dim>
        {
        public:
            double interface_y_, eps_, amp_, Lx_, x_min_;
            int n_modes_;

            PerturbedLayerIC(double y, double e, double a, double L, double xm, int n)
                : dealii::Function<dim>(1)
                , interface_y_(y), eps_(e), amp_(a), Lx_(L), x_min_(xm), n_modes_(n) {}

            double value(const dealii::Point<dim>& p, unsigned int = 0) const override
            {
                // Sum of cosine modes
                double perturbation = 0.0;
                for (int k = 1; k <= n_modes_; ++k)
                {
                    perturbation += amp_ * std::cos(2.0 * M_PI * k * (p[0] - x_min_) / Lx_);
                }
                const double y_interface = interface_y_ + perturbation;
                return -std::tanh((p[1] - y_interface) / (std::sqrt(2.0) * eps_));
            }
        };

        PerturbedLayerIC ic_func(interface_y, epsilon, perturbation, Lx,
                                  params_.domain.x_min, n_modes);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);
    }

    // Copy to old solution
    theta_old_solution_ = theta_solution_;

    // Initialize ψ = 0 (will be computed in first solve)
    psi_solution_ = 0;

    if (params_.output.verbose)
    {
        std::cout << "[Setup] IC type " << ic_type << " initialized\n";
        std::cout << "[Setup] Initial mass = " << compute_mass() << "\n";
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;