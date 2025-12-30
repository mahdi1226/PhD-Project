// ============================================================================
// core/phase_field_setup.cc - Setup Methods for Phase Field Problem
//
// Setup for CH + Poisson + Magnetization (DG) + NS systems.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "mms/ch_mms.h"
#include "setup/ch_setup.h"
#include "setup/poisson_setup.h"
#include "setup/ns_setup.h"
#include "solvers/ns_solver.h"
#include "mms/ns_mms.h"
#include "physics/material_properties.h"
#include "solvers/solver_info.h"


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
PhaseFieldProblem<dim>::PhaseFieldProblem(const Parameters& params)
    : params_(params)
    , fe_Q2_(params.fe.degree_velocity)        // Q2 for velocity, phase fields
    , fe_Q1_(params.fe.degree_pressure)        // Q1 for pressure
    , fe_DG_(params.fe.degree_magnetization)   // DG for magnetization (default: DG0)
    , theta_dof_handler_(triangulation_)
    , psi_dof_handler_(triangulation_)
    , phi_dof_handler_(triangulation_)
    , mx_dof_handler_(triangulation_)
    , my_dof_handler_(triangulation_)
    , ux_dof_handler_(triangulation_)
    , uy_dof_handler_(triangulation_)
    , p_dof_handler_(triangulation_)
    , time_(params.mms_t_init)                 // Supports MMS with non-zero initial time
    , timestep_number_(0)
{
}

// ============================================================================
// setup_mesh()
//
// Creates rectangular domain with specified number of initial cells,
// then applies global refinement.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_mesh()
{

    // Create rectangular domain with subdivisions
    dealii::Point<dim> p1(params_.domain.x_min, params_.domain.y_min);
    dealii::Point<dim> p2(params_.domain.x_max, params_.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params_.domain.initial_cells_x;
    subdivisions[1] = params_.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation_, subdivisions, p1, p2);

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
    triangulation_.refine_global(params_.mesh.initial_refinement);

    if (params_.output.verbose)
    {
        std::cout << "[Setup] Base mesh: " << params_.domain.initial_cells_x
                  << " × " << params_.domain.initial_cells_y << " cells\n";
        std::cout << "[Setup] After " << params_.mesh.initial_refinement
                  << " refinements: " << triangulation_.n_active_cells() << " cells\n";
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
    // Poisson (φ) - Q2 elements
    // ========================================================================
    if (params_.enable_magnetic)
    {
        phi_dof_handler_.distribute_dofs(fe_Q2_);
        phi_solution_.reinit(phi_dof_handler_.n_dofs());
    }

    // ========================================================================
    // Magnetization (Mx, My) - DG elements (NEW!)
    //
    // CRITICAL: M must be DG for the energy identity B_h^m(H,H,M) = 0.
    // Do NOT use CG or projected fields!
    // ========================================================================
    if (params_.enable_magnetic)
    {
        mx_dof_handler_.distribute_dofs(fe_DG_);
        my_dof_handler_.distribute_dofs(fe_DG_);

        const unsigned int n_M = mx_dof_handler_.n_dofs();

        mx_solution_.reinit(n_M);
        my_solution_.reinit(n_M);
        mx_old_solution_.reinit(n_M);
        my_old_solution_.reinit(n_M);

        // Initialize to zero (will be set properly in initialize_solutions)
        mx_solution_ = 0;
        my_solution_ = 0;
        mx_old_solution_ = 0;
        my_old_solution_ = 0;

        if (params_.output.verbose)
        {
            std::cout << "[Setup] Magnetization DoFs: " << n_M << " (DG"
                      << fe_DG_.degree << ")\n";
            // Verify FE space compatibility for energy stability
            std::cout << "[Setup] M space: DG" << fe_DG_.degree
                      << " (need DG" << (fe_Q2_.degree - 1) << " for ∇X_h ⊂ M_h)\n";
            Assert(fe_DG_.degree >= fe_Q2_.degree - 1,
                   dealii::ExcMessage("Magnetization degree too low for energy stability!"));
        }
    }

    // ========================================================================
    // NS fields (ux, uy, p) - Q2 velocity, Q1 pressure (Taylor-Hood)
    // ========================================================================
    if (params_.enable_ns)
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
                      << ", ψ = " << n_psi;
            if (params_.enable_magnetic)
                std::cout << ", φ = " << phi_dof_handler_.n_dofs()
                          << ", M = " << mx_dof_handler_.n_dofs();
            std::cout << ", ux = " << n_ux
                      << ", uy = " << n_uy
                      << ", p = " << n_p << "\n";
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
                      << ", ψ = " << n_psi;
            if (params_.enable_magnetic)
                std::cout << ", φ = " << phi_dof_handler_.n_dofs()
                          << ", M = " << mx_dof_handler_.n_dofs();
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
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);

    // ψ constraints
    psi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);

    // For MMS: add Dirichlet BCs at initial time
    if (params_.enable_mms)
    {
        apply_ch_mms_boundary_constraints<dim>(
            theta_dof_handler_,
            psi_dof_handler_,
            theta_constraints_,
            psi_constraints_,
            params_.mms_t_init);

        // Set initial time
        const_cast<double&>(time_) = params_.mms_t_init;
    }

    theta_constraints_.close();
    psi_constraints_.close();
}

// ============================================================================
// setup_ch_system()
//
// Creates coupled θ-ψ system using the setup free function.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ch_system()
{
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

    // Initialize matrix and vectors
    const unsigned int n_ch = theta_dof_handler_.n_dofs() +
                               psi_dof_handler_.n_dofs();
    ch_matrix_.reinit(ch_sparsity_);
    ch_rhs_.reinit(n_ch);
    ch_solution_.reinit(n_ch);
}

// ============================================================================
// setup_poisson_system()
//
// Poisson for magnetostatic potential: (∇φ, ∇χ) = (h_a - M, ∇χ)
// Pure Neumann → pin DoF 0 to fix constant.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_poisson_system()
{

    // Constraints: hanging nodes + pin DoF 0
    phi_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler_, phi_constraints_);

    // Pin DoF 0 to zero (fixes the constant for pure Neumann)
    if (phi_dof_handler_.n_dofs() > 0 && !phi_constraints_.is_constrained(0))
    {
        phi_constraints_.add_line(0);
        phi_constraints_.set_inhomogeneity(0, 0.0);
    }
    phi_constraints_.close();

    // Sparsity pattern WITH constraints
    dealii::DynamicSparsityPattern dsp(phi_dof_handler_.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(phi_dof_handler_, dsp,
                                     phi_constraints_,
                                     /*keep_constrained_dofs=*/false);
    phi_sparsity_.copy_from(dsp);

    // Initialize matrix and vectors
    phi_matrix_.reinit(phi_sparsity_);
    phi_rhs_.reinit(phi_dof_handler_.n_dofs());

    if (params_.output.verbose)
    {
        std::cout << "[Setup] Poisson sparsity: "
                  << phi_sparsity_.n_nonzero_elements() << " nonzeros\n";
    }
}

// ============================================================================
// setup_magnetization_system() - NEW!
//
// DG transport for magnetization M = (Mx, My).
// Paper Eq. 42d: ∂M/∂t + (u·∇)M = (1/τ_M)(χ(θ)H - M)
//
// CRITICAL: Must use DG elements and flux sparsity pattern!
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_magnetization_system()
{

    const unsigned int n_M = mx_dof_handler_.n_dofs();

    // DG sparsity: includes face coupling (for upwind flux)
    dealii::DynamicSparsityPattern dsp_M(n_M, n_M);
    dealii::DoFTools::make_flux_sparsity_pattern(mx_dof_handler_, dsp_M);
    mx_sparsity_.copy_from(dsp_M);
    my_sparsity_.copy_from(dsp_M);  // Same pattern for My

    // Initialize matrices and vectors
    mx_matrix_.reinit(mx_sparsity_);
    my_matrix_.reinit(my_sparsity_);
    mx_rhs_.reinit(n_M);
    my_rhs_.reinit(n_M);

    if (params_.output.verbose)
    {
        std::cout << "[Setup] Magnetization sparsity: "
                  << mx_sparsity_.n_nonzero_elements() << " nonzeros (DG flux)\n";
    }
}

// ============================================================================
// setup_ns_system()
//
// Taylor-Hood Q2-Q1 for velocity-pressure.
// No-slip BCs on all boundaries.
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ns_system()
{

    // ========================================================================
    // Individual field constraints
    // ========================================================================
    ux_constraints_.clear();
    uy_constraints_.clear();
    p_constraints_.clear();

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);

    // No-slip BCs: u = 0 on bottom, left, right (NOT top - free surface)
    for (unsigned int boundary_id : {0, 1, 3})  // Exclude 2 (top)
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
    // Top (boundary_id = 2): free surface (no Dirichlet BC = natural/stress-free)

    // ========================================================================
    // CRITICAL: Pin one pressure DoF to fix the constant!
    // ========================================================================
    if (p_dof_handler_.n_dofs() > 0 && !p_constraints_.is_constrained(0))
    {
        p_constraints_.add_line(0);
        p_constraints_.set_inhomogeneity(0, 0.0);
    }

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

    // ========================================================================
    // Assemble pressure mass matrix (for Schur complement preconditioner)
    // ========================================================================
    assemble_pressure_mass_matrix<dim>(
    p_dof_handler_,
    p_constraints_,
    pressure_mass_sparsity_,
    pressure_mass_matrix_);
}

// ============================================================================
// initialize_solutions()
//
// IC types for physical runs:
//   0 = Flat ferrofluid layer (DEFAULT)
//   1 = Perturbed layer
//   2 = Circular droplet (testing)
//
// For MMS: use exact solutions at t_init
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::initialize_solutions()
{

    // MMS mode: use exact solutions at t_init
    if (params_.enable_mms)
    {
        apply_ch_mms_initial_conditions<dim>(
            theta_dof_handler_,
            psi_dof_handler_,
            theta_solution_,
            psi_solution_,
            params_.mms_t_init);

        theta_old_solution_ = theta_solution_;

        // NS MMS IC
        if (params_.enable_ns)
        {
            const double L_y = params_.domain.y_max - params_.domain.y_min;
            const double t_init = params_.mms_t_init;

            NSExactVelocityX<dim> exact_ux(t_init, L_y);
            NSExactVelocityY<dim> exact_uy(t_init, L_y);
            NSExactPressure<dim> exact_p(t_init, L_y);

            dealii::VectorTools::interpolate(ux_dof_handler_, exact_ux, ux_solution_);
            dealii::VectorTools::interpolate(uy_dof_handler_, exact_uy, uy_solution_);
            dealii::VectorTools::interpolate(p_dof_handler_, exact_p, p_solution_);

            ux_old_solution_ = ux_solution_;
            uy_old_solution_ = uy_solution_;
        }

        if (params_.output.verbose)
        {
            std::cout << "[Setup] MMS IC at t = " << params_.mms_t_init << "\n";
        }
        return;
    }

    // Physical IC modes
    const int ic_type = params_.ic.type;
    const double interface_y = params_.ic.pool_depth;

    if (ic_type == 0)
    {
        // ================================================================
        // Flat ferrofluid layer
        // θ = +1 below interface (ferrofluid)
        // θ = -1 above interface (non-magnetic)
        // ================================================================
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

        FlatLayerIC ic_func(interface_y, params_.physics.epsilon);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);

        if (params_.output.verbose)
            std::cout << "[Setup] Flat layer IC: interface at y = " << interface_y << "\n";
    }
    else if (ic_type == 1)
    {
        // ================================================================
        // Perturbed layer
        // ================================================================
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
                double pert = 0.0;
                for (int k = 1; k <= n_modes_; ++k)
                    pert += amp_ * std::cos(2.0 * M_PI * k * (p[0] - x_min_) / Lx_);
                const double y_interface = interface_y_ + pert;
                return -std::tanh((p[1] - y_interface) / (std::sqrt(2.0) * eps_));
            }
        };

        PerturbedLayerIC ic_func(interface_y, params_.physics.epsilon, perturbation, Lx,
                                  params_.domain.x_min, n_modes);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);

        if (params_.output.verbose)
            std::cout << "[Setup] Perturbed layer IC: interface at y = " << interface_y << "\n";
    }
    else if (ic_type == 2)
    {
        // ================================================================
        // Circular droplet (testing only)
        // ================================================================
        const double cx = params_.ic.droplet_center_x;
        const double cy = params_.ic.droplet_center_y;
        const double radius = params_.ic.droplet_radius;

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

        DropletIC ic_func(cx, cy, radius, params_.physics.epsilon);
        dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);

        if (params_.output.verbose)
            std::cout << "[Setup] Circular droplet IC\n";
    }
    else
    {
        std::cerr << "[Setup] Unknown IC type: " << ic_type << "\n";
        std::exit(1);
    }

    // Copy to old solution
    theta_old_solution_ = theta_solution_;

    // Initialize ψ = 0 (will be computed in first solve)
    psi_solution_ = 0;

    // ========================================================================
    // Initialize magnetization M = χ(θ) H
    //
    // At t=0, we don't have H yet, so either:
    //   (a) Set M = 0 and let it evolve
    //   (b) Solve Poisson once to get H, then set M = χ(θ) H
    //
    // Option (b) is more physical but requires Poisson to be set up first.
    // For simplicity, we use (a) here; M will quickly relax to equilibrium.
    // ========================================================================
    if (params_.enable_magnetic)
    {
        mx_solution_ = 0;
        my_solution_ = 0;
        mx_old_solution_ = 0;
        my_old_solution_ = 0;

        if (params_.output.verbose)
            std::cout << "[Setup] Magnetization initialized to zero (will relax)\n";
    }
}

// ============================================================================
// get_min_h() - Minimum cell diameter
// ============================================================================
template <int dim>
double PhaseFieldProblem<dim>::get_min_h() const
{
    return dealii::GridTools::minimal_cell_diameter(triangulation_);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;