// ============================================================================
// core/nsch_problem_setup.cc - Setup and initialization methods
//
// REFACTORED VERSION: Separate DoFHandlers for each scalar field
//
// Based on: Nochetto, Salgado & Tomas (2016)
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#include "core/nsch_problem.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include "utilities/nsch_mms.h"

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Setup mesh (shared between all DoFHandlers)
// ============================================================================
// Boundary IDs for rectangular domain:
//   0 = bottom (y = y_min)
//   1 = right  (x = x_max)
//   2 = top    (y = y_max)
//   3 = left   (x = x_min)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_mesh()
{
    dealii::GridGenerator::hyper_rectangle(
        triangulation_,
        dealii::Point<dim>(params_.x_min, params_.y_min),
        dealii::Point<dim>(params_.x_max, params_.y_max));

    // Assign boundary IDs: bottom=0, right=1, top=2, left=3
    const double tol = 1e-10;
    for (auto& cell : triangulation_.active_cell_iterators())
    {
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (cell->face(f)->at_boundary())
            {
                const dealii::Point<dim> face_center = cell->face(f)->center();

                if (std::abs(face_center[1] - params_.y_min) < tol)
                    cell->face(f)->set_boundary_id(0);  // bottom
                else if (std::abs(face_center[0] - params_.x_max) < tol)
                    cell->face(f)->set_boundary_id(1);  // right
                else if (std::abs(face_center[1] - params_.y_max) < tol)
                    cell->face(f)->set_boundary_id(2);  // top
                else if (std::abs(face_center[0] - params_.x_min) < tol)
                    cell->face(f)->set_boundary_id(3);  // left
            }
        }
    }

    // For AMR: use n_refinements as initial uniform refinement
    unsigned int initial_refinement = params_.n_refinements;
    if (params_.use_amr)
        initial_refinement = std::max(initial_refinement, params_.amr_min_level);

    triangulation_.refine_global(initial_refinement);

    if (params_.verbose)
        std::cout << "[INFO] Mesh created: "
                  << triangulation_.n_active_cells() << " cells\n";
}

// ============================================================================
// Setup concentration (c) system - Q2 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_c_system()
{
    c_dof_handler_.distribute_dofs(fe_Q2_);

    if (params_.verbose)
        std::cout << "[INFO] c DoFs: " << c_dof_handler_.n_dofs() << "\n";

    c_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(c_dof_handler_, c_constraints_);

    // Cahn-Hilliard typically has homogeneous Neumann BCs (no-flux)
    // MMS mode may need Dirichlet
    if (params_.mms_mode)
    {
        // MMS BC handled in update_mms_boundary_conditions
    }
    c_constraints_.close();

    c_solution_.reinit(c_dof_handler_.n_dofs());
    c_old_solution_.reinit(c_dof_handler_.n_dofs());
}

// ============================================================================
// Setup chemical potential (mu) system - Q2 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_mu_system()
{
    mu_dof_handler_.distribute_dofs(fe_Q2_);

    if (params_.verbose)
        std::cout << "[INFO] mu DoFs: " << mu_dof_handler_.n_dofs() << "\n";

    mu_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(mu_dof_handler_, mu_constraints_);
    mu_constraints_.close();

    mu_solution_.reinit(mu_dof_handler_.n_dofs());
}

// ============================================================================
// Setup velocity x (ux) system - Q2 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_ux_system()
{
    ux_dof_handler_.distribute_dofs(fe_Q2_);

    if (params_.verbose)
        std::cout << "[INFO] ux DoFs: " << ux_dof_handler_.n_dofs() << "\n";

    ux_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);

    if (params_.mms_mode)
    {
        // MMS: set exact velocity on all boundaries
        // Handled in update_mms_boundary_conditions
    }
    else
    {
        // No-slip: ux = 0 on all boundaries
        for (unsigned int b = 0; b < 4; ++b)
        {
            dealii::VectorTools::interpolate_boundary_values(
                ux_dof_handler_,
                b,
                dealii::Functions::ZeroFunction<dim>(),
                ux_constraints_);
        }
    }
    ux_constraints_.close();

    ux_solution_.reinit(ux_dof_handler_.n_dofs());
    ux_old_solution_.reinit(ux_dof_handler_.n_dofs());
}

// ============================================================================
// Setup velocity y (uy) system - Q2 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_uy_system()
{
    uy_dof_handler_.distribute_dofs(fe_Q2_);

    if (params_.verbose)
        std::cout << "[INFO] uy DoFs: " << uy_dof_handler_.n_dofs() << "\n";

    uy_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);

    if (params_.mms_mode)
    {
        // MMS: set exact velocity on all boundaries
        // Handled in update_mms_boundary_conditions
    }
    else
    {
        // No-slip: uy = 0 on all boundaries
        for (unsigned int b = 0; b < 4; ++b)
        {
            dealii::VectorTools::interpolate_boundary_values(
                uy_dof_handler_,
                b,
                dealii::Functions::ZeroFunction<dim>(),
                uy_constraints_);
        }
    }
    uy_constraints_.close();

    uy_solution_.reinit(uy_dof_handler_.n_dofs());
    uy_old_solution_.reinit(uy_dof_handler_.n_dofs());
}

// ============================================================================
// Setup pressure (p) system - Q1 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_p_system()
{
    p_dof_handler_.distribute_dofs(fe_Q1_);

    if (params_.verbose)
        std::cout << "[INFO] p DoFs: " << p_dof_handler_.n_dofs() << "\n";

    p_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);
    // Pressure has no Dirichlet BC (determined up to a constant, handled post-solve)
    p_constraints_.close();

    p_solution_.reinit(p_dof_handler_.n_dofs());
}

// ============================================================================
// Setup magnetic potential (phi) system - Q2 elements
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_phi_system()
{
    if (!params_.enable_magnetic)
        return;

    phi_dof_handler_.distribute_dofs(fe_Q2_);

    if (params_.verbose)
        std::cout << "[INFO] phi DoFs: " << phi_dof_handler_.n_dofs() << "\n";

    // Initial constraints with dipole BCs at t=0
    update_phi_constraints_dipole(0.0);

    phi_solution_.reinit(phi_dof_handler_.n_dofs());

    // Setup sparsity pattern for Poisson system
    dealii::DynamicSparsityPattern dsp(phi_dof_handler_.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(phi_dof_handler_, dsp, phi_constraints_, false);
    phi_sparsity_.copy_from(dsp);
    phi_matrix_.reinit(phi_sparsity_);
    phi_rhs_.reinit(phi_dof_handler_.n_dofs());
}

// ============================================================================
// Setup coupled CH system (sparsity pattern for [c, mu] monolithic solve)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_ch_coupled_system()
{
    // The CH system couples c and mu equations
    // Total size = n_c + n_mu (same since both use Q2)
    const unsigned int n_c = c_dof_handler_.n_dofs();
    const unsigned int n_mu = mu_dof_handler_.n_dofs();
    const unsigned int n_total = n_c + n_mu;

    // Build index maps: local scalar DoF -> coupled system index
    c_to_ch_map_.resize(n_c);
    mu_to_ch_map_.resize(n_mu);

    for (unsigned int i = 0; i < n_c; ++i)
        c_to_ch_map_[i] = i;
    for (unsigned int i = 0; i < n_mu; ++i)
        mu_to_ch_map_[i] = n_c + i;

    // Create coupled sparsity pattern
    dealii::DynamicSparsityPattern dsp(n_total, n_total);

    // c-c coupling (from c equation time derivative and advection)
    for (const auto& cell : c_dof_handler_.active_cell_iterators())
    {
        std::vector<dealii::types::global_dof_index> local_dofs(fe_Q2_.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        for (const auto& i : local_dofs)
            for (const auto& j : local_dofs)
                dsp.add(c_to_ch_map_[i], c_to_ch_map_[j]);
    }

    // mu-mu coupling (identity in mu equation)
    for (const auto& cell : mu_dof_handler_.active_cell_iterators())
    {
        std::vector<dealii::types::global_dof_index> local_dofs(fe_Q2_.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        for (const auto& i : local_dofs)
            for (const auto& j : local_dofs)
                dsp.add(mu_to_ch_map_[i], mu_to_ch_map_[j]);
    }

    // c-mu coupling (diffusion term in c equation couples to mu)
    // mu-c coupling (Laplacian and potential terms couple mu to c)
    // These share the same mesh, so iterate once
    auto c_cell = c_dof_handler_.begin_active();
    auto mu_cell = mu_dof_handler_.begin_active();
    for (; c_cell != c_dof_handler_.end(); ++c_cell, ++mu_cell)
    {
        std::vector<dealii::types::global_dof_index> c_dofs(fe_Q2_.n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> mu_dofs(fe_Q2_.n_dofs_per_cell());
        c_cell->get_dof_indices(c_dofs);
        mu_cell->get_dof_indices(mu_dofs);

        for (const auto& i : c_dofs)
            for (const auto& j : mu_dofs)
            {
                dsp.add(c_to_ch_map_[i], mu_to_ch_map_[j]);   // c-mu
                dsp.add(mu_to_ch_map_[j], c_to_ch_map_[i]);   // mu-c
            }
    }

    // ========================================================================
    // CREATE COMBINED CONSTRAINTS AND CONDENSE THE SPARSITY PATTERN
    // This is the proper way to handle hanging nodes (per step-28)
    // ========================================================================
    ch_combined_constraints_.clear();

    // Copy c constraints with index mapping
    for (unsigned int i = 0; i < n_c; ++i)
    {
        if (c_constraints_.is_constrained(i))
        {
            const unsigned int mapped_i = c_to_ch_map_[i];
            const auto* entries = c_constraints_.get_constraint_entries(i);
            const double inhomogeneity = c_constraints_.get_inhomogeneity(i);

            std::vector<std::pair<dealii::types::global_dof_index, double>> mapped_entries;
            if (entries)
            {
                for (const auto& entry : *entries)
                    mapped_entries.emplace_back(c_to_ch_map_[entry.first], entry.second);
            }
            ch_combined_constraints_.add_line(mapped_i);
            for (const auto& entry : mapped_entries)
                ch_combined_constraints_.add_entry(mapped_i, entry.first, entry.second);
            ch_combined_constraints_.set_inhomogeneity(mapped_i, inhomogeneity);
        }
    }

    // Copy mu constraints with index mapping
    for (unsigned int i = 0; i < n_mu; ++i)
    {
        if (mu_constraints_.is_constrained(i))
        {
            const unsigned int mapped_i = mu_to_ch_map_[i];
            const auto* entries = mu_constraints_.get_constraint_entries(i);
            const double inhomogeneity = mu_constraints_.get_inhomogeneity(i);

            std::vector<std::pair<dealii::types::global_dof_index, double>> mapped_entries;
            if (entries)
            {
                for (const auto& entry : *entries)
                    mapped_entries.emplace_back(mu_to_ch_map_[entry.first], entry.second);
            }
            ch_combined_constraints_.add_line(mapped_i);
            for (const auto& entry : mapped_entries)
                ch_combined_constraints_.add_entry(mapped_i, entry.first, entry.second);
            ch_combined_constraints_.set_inhomogeneity(mapped_i, inhomogeneity);
        }
    }
    ch_combined_constraints_.close();

    // Condense the sparsity pattern with combined constraints
    ch_combined_constraints_.condense(dsp);

    ch_sparsity_.copy_from(dsp);
    ch_matrix_.reinit(ch_sparsity_);
    ch_rhs_.reinit(n_total);
}

// ============================================================================
// Setup coupled NS system (sparsity pattern for [ux, uy, p] monolithic solve)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_ns_coupled_system()
{
    const unsigned int n_ux = ux_dof_handler_.n_dofs();
    const unsigned int n_uy = uy_dof_handler_.n_dofs();
    const unsigned int n_p = p_dof_handler_.n_dofs();
    const unsigned int n_total = n_ux + n_uy + n_p;

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

    // Create coupled sparsity pattern
    dealii::DynamicSparsityPattern dsp(n_total, n_total);

    // ux-ux, uy-uy coupling (viscosity, convection, time derivative)
    for (const auto& cell : ux_dof_handler_.active_cell_iterators())
    {
        std::vector<dealii::types::global_dof_index> local_dofs(fe_Q2_.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        for (const auto& i : local_dofs)
            for (const auto& j : local_dofs)
            {
                dsp.add(ux_to_ns_map_[i], ux_to_ns_map_[j]);
                dsp.add(uy_to_ns_map_[i], uy_to_ns_map_[j]);
            }
    }

    // ux-uy coupling (if any from convection linearization)
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

    // p-p coupling (pressure stabilization if any, usually sparse)
    for (const auto& cell : p_dof_handler_.active_cell_iterators())
    {
        std::vector<dealii::types::global_dof_index> local_dofs(fe_Q1_.n_dofs_per_cell());
        cell->get_dof_indices(local_dofs);
        for (const auto& i : local_dofs)
            for (const auto& j : local_dofs)
                dsp.add(p_to_ns_map_[i], p_to_ns_map_[j]);
    }

    // ux-p, uy-p coupling (pressure gradient in momentum, divergence in continuity)
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
                dsp.add(p_to_ns_map_[j], ux_to_ns_map_[i]);
            }
    }

    uy_cell = uy_dof_handler_.begin_active();
    p_cell = p_dof_handler_.begin_active();
    for (; uy_cell != uy_dof_handler_.end(); ++uy_cell, ++p_cell)
    {
        std::vector<dealii::types::global_dof_index> uy_dofs(fe_Q2_.n_dofs_per_cell());
        std::vector<dealii::types::global_dof_index> p_dofs(fe_Q1_.n_dofs_per_cell());
        uy_cell->get_dof_indices(uy_dofs);
        p_cell->get_dof_indices(p_dofs);

        for (const auto& i : uy_dofs)
            for (const auto& j : p_dofs)
            {
                dsp.add(uy_to_ns_map_[i], p_to_ns_map_[j]);
                dsp.add(p_to_ns_map_[j], uy_to_ns_map_[i]);
            }
    }

    // ========================================================================
    // CREATE COMBINED CONSTRAINTS AND CONDENSE THE SPARSITY PATTERN
    // This is the proper way to handle hanging nodes (per step-28)
    // ========================================================================
    ns_combined_constraints_.clear();

    // Helper lambda to copy constraints with index mapping
    auto copy_constraints = [this](
        const dealii::AffineConstraints<double>& src,
        const std::vector<unsigned int>& index_map,
        unsigned int n_dofs)
    {
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
            if (src.is_constrained(i))
            {
                const unsigned int mapped_i = index_map[i];
                const auto* entries = src.get_constraint_entries(i);
                const double inhomogeneity = src.get_inhomogeneity(i);

                ns_combined_constraints_.add_line(mapped_i);
                if (entries)
                {
                    for (const auto& entry : *entries)
                        ns_combined_constraints_.add_entry(mapped_i, index_map[entry.first], entry.second);
                }
                ns_combined_constraints_.set_inhomogeneity(mapped_i, inhomogeneity);
            }
        }
    };

    // Copy all constraints with their index mappings
    copy_constraints(ux_constraints_, ux_to_ns_map_, n_ux);
    copy_constraints(uy_constraints_, uy_to_ns_map_, n_uy);
    copy_constraints(p_constraints_, p_to_ns_map_, n_p);

    // Pin pressure at one node to remove singularity
    // In a closed domain with no-flux BCs, pressure is determined only up to a constant.
    // Without this constraint, the linear system is singular and UMFPACK produces garbage.
    // We pin the first pressure DoF to zero (p(DoF 0) = 0).
    {
        const dealii::types::global_dof_index p_pin_local = 0;
        const dealii::types::global_dof_index p_pin_global = p_to_ns_map_[p_pin_local];

        // Only add if not already constrained
        if (!ns_combined_constraints_.is_constrained(p_pin_global))
        {
            ns_combined_constraints_.add_line(p_pin_global);
            ns_combined_constraints_.set_inhomogeneity(p_pin_global, 0.0);
        }
    }

    ns_combined_constraints_.close();

    // Condense the sparsity pattern with combined constraints
    ns_combined_constraints_.condense(dsp);

    ns_sparsity_.copy_from(dsp);
    ns_matrix_.reinit(ns_sparsity_);
    ns_rhs_.reinit(n_total);
}

// ============================================================================
// Setup all systems
// ============================================================================
template <int dim>
void NSCHProblem<dim>::setup_all_systems()
{
    setup_c_system();
    setup_mu_system();
    setup_ux_system();
    setup_uy_system();
    setup_p_system();
    setup_phi_system();

    setup_ch_coupled_system();
    setup_ns_coupled_system();
}

// ============================================================================
// Initialize concentration (c)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::initialize_c()
{
    double epsilon = params_.epsilon;

    if (params_.mms_mode)
    {
        // MMS: zero initial condition
        c_solution_ = 0.0;
        c_old_solution_ = 0.0;
        std::cout << "[INFO] c initialized to zero (MMS mode)\n";
        return;
    }

    if (params_.ic_type == 0)
    {
        // Circular droplet
        class InitialC_Droplet : public dealii::Function<dim>
        {
        public:
            double eps;
            InitialC_Droplet(double e) : dealii::Function<dim>(1), eps(e) {}
            virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
            {
                const double x = p[0] - 0.5;
                const double y = p[1] - 0.5;
                const double r = std::sqrt(x*x + y*y);
                const double R = 0.25;
                return std::tanh((R - r) / (std::sqrt(2.0) * eps));
            }
        };

        InitialC_Droplet initial_c(epsilon);
        dealii::VectorTools::interpolate(c_dof_handler_, initial_c, c_solution_);
        std::cout << "[INFO] c initialized (circular droplet)\n";
    }
    else if (params_.ic_type == 1 || params_.ic_type == 2)
    {
        // Rosensweig layer (flat or perturbed)
        class InitialC_Rosensweig : public dealii::Function<dim>
        {
        public:
            double eps, layer_h, pert_amp, domain_w;
            int n_modes;
            bool perturb;

            InitialC_Rosensweig(double e, double h, double amp, int modes, double w, bool p)
                : dealii::Function<dim>(1), eps(e), layer_h(h), pert_amp(amp),
                  domain_w(w), n_modes(modes), perturb(p) {}

            virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
            {
                const double x = p[0];
                const double y = p[1];

                double interface_y = layer_h;
                if (perturb)
                {
                    for (int k = 1; k <= n_modes; ++k)
                    {
                        double wn = 2.0 * M_PI * k / domain_w;
                        double amp = pert_amp / k;
                        double phase = M_PI * k / 3.0;
                        interface_y += amp * std::sin(wn * x + phase);
                    }
                }

                double dist = y - interface_y;
                return std::tanh(-dist / (std::sqrt(2.0) * eps));
            }
        };

        double domain_width = params_.x_max - params_.x_min;
        double layer_height = params_.y_min + params_.rosensweig_layer_height * (params_.y_max - params_.y_min);
        bool add_perturbation = (params_.ic_type == 2);

        InitialC_Rosensweig initial_c(epsilon, layer_height, params_.rosensweig_perturbation,
                                       params_.rosensweig_perturbation_modes, domain_width, add_perturbation);
        dealii::VectorTools::interpolate(c_dof_handler_, initial_c, c_solution_);

        std::cout << "[INFO] c initialized (Rosensweig " << (add_perturbation ? "perturbed" : "flat") << ")\n";
        std::cout << "[INFO]   Layer height: " << layer_height << ", eps: " << epsilon << "\n";
    }

    c_old_solution_ = c_solution_;
}

// ============================================================================
// Initialize chemical potential (mu)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::initialize_mu()
{
    if (params_.mms_mode)
    {
        mu_solution_ = 0.0;
        std::cout << "[INFO] mu initialized to zero (MMS mode)\n";
        return;
    }

    // mu = f(c) = c^3 - c at equilibrium
    class InitialMu : public dealii::Function<dim>
    {
    public:
        const dealii::DoFHandler<dim>& c_dh;
        const dealii::Vector<double>& c_sol;

        InitialMu(const dealii::DoFHandler<dim>& dh, const dealii::Vector<double>& sol)
            : dealii::Function<dim>(1), c_dh(dh), c_sol(sol) {}

        virtual double value(const dealii::Point<dim>& /*p*/, unsigned int = 0) const override
        {
            // For simplicity, use c^3 - c evaluated at this point
            // This is approximate; a more accurate way would be to project
            // For initialization, this is sufficient
            return 0.0;  // Start with zero; will equilibrate quickly
        }
    };

    // Simple initialization: mu = c^3 - c
    for (unsigned int i = 0; i < mu_solution_.size(); ++i)
    {
        double c = c_solution_[i];
        mu_solution_[i] = c * c * c - c;
    }

    std::cout << "[INFO] mu initialized (equilibrium mu = c^3 - c)\n";
}

// ============================================================================
// Initialize velocity
// ============================================================================
template <int dim>
void NSCHProblem<dim>::initialize_velocity()
{
    ux_solution_ = 0.0;
    ux_old_solution_ = 0.0;
    uy_solution_ = 0.0;
    uy_old_solution_ = 0.0;
    p_solution_ = 0.0;

    std::cout << "[INFO] Velocity initialized to zero\n";
}

// ============================================================================
// Initialize magnetic potential (phi)
// ============================================================================
template <int dim>
void NSCHProblem<dim>::initialize_phi()
{
    if (!params_.enable_magnetic)
        return;

    // Initialize with dipole potential at t=0
    class DipolePotentialIC : public dealii::Function<dim>
    {
    public:
        double dipole_x[5] = {-0.5, 0.0, 0.5, 1.0, 1.5};
        double dipole_y, dir_x, dir_y, intensity, ramp_time;

        DipolePotentialIC(double I, double ramp, double y_pos)
            : dealii::Function<dim>(1), dipole_y(y_pos), dir_x(0.0), dir_y(1.0),
              intensity(I), ramp_time(ramp) {}

        virtual double value(const dealii::Point<dim>& p, unsigned int = 0) const override
        {
            double alpha = (ramp_time > 0) ? 0.0 : intensity;  // At t=0, intensity is 0 if ramping

            double phi_total = 0.0;
            for (int s = 0; s < 5; ++s)
            {
                double rx = dipole_x[s] - p[0];
                double ry = dipole_y - p[1];
                double r2 = rx * rx + ry * ry;
                if (r2 < 1e-10) continue;
                double d_dot_r = dir_x * rx + dir_y * ry;
                phi_total += alpha * d_dot_r / r2;
            }
            return phi_total;
        }
    };

    DipolePotentialIC initial_phi(params_.dipole_intensity, params_.dipole_ramp_time,
                                   params_.dipole_y_position);
    dealii::VectorTools::interpolate(phi_dof_handler_, initial_phi, phi_solution_);

    std::cout << "[INFO] phi initialized (dipole field at t=0)\n";
}

// ============================================================================
// Initialize all fields
// ============================================================================
template <int dim>
void NSCHProblem<dim>::initialize_all()
{
    initialize_c();
    initialize_mu();
    initialize_velocity();
    initialize_phi();

    // Apply constraints
    c_constraints_.distribute(c_solution_);
    c_constraints_.distribute(c_old_solution_);
    mu_constraints_.distribute(mu_solution_);
    ux_constraints_.distribute(ux_solution_);
    ux_constraints_.distribute(ux_old_solution_);
    uy_constraints_.distribute(uy_solution_);
    uy_constraints_.distribute(uy_old_solution_);
    p_constraints_.distribute(p_solution_);
    if (params_.enable_magnetic)
        phi_constraints_.distribute(phi_solution_);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class NSCHProblem<2>;