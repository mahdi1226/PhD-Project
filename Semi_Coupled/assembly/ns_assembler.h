// ============================================================================
// assembly/ns_assembler.h - Parallel Navier-Stokes Assembler (Production)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e-42f (discrete scheme)
//
// PRODUCTION assembler with:
//   - Skew-symmetric convection via skew_forms.h (Eq. 37)
//   - Kelvin force mu_0(M.grad)H via kelvin_force.h (Eq. 38)
//   - Capillary force (lambda/epsilon) theta grad(psi)
//   - Gravity force r*H(theta/epsilon)*g
//   - Optional MMS support via enable_mms flag
//
// Architecture:
//   - Single unified entry point: assemble_ns_system_unified()
//   - NSForceData struct bundles optional force parameters
//   - Legacy wrappers delegate to unified function for backward compatibility
//
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "utilities/parameters.h"

#include <functional>
#include <vector>
#include <mpi.h>

// ============================================================================
// NSForceData: Bundles optional force parameters for unified NS assembly
//
// Usage:
//   NSForceData<dim> forces;  // default: no forces
//
//   // Add Kelvin force:
//   forces.enable_kelvin(phi_dof, M_dof, phi_sol, Mx_sol, My_sol, mu_0, params, time);
//
//   // Add capillary force:
//   forces.enable_capillary(theta_dof, psi_dof, theta_sol, psi_sol, lambda, epsilon);
//
//   // Add gravity:
//   forces.enable_gravity_force(r, gravity_mag, gravity_dir);
//
//   // Add variable viscosity:
//   forces.enable_variable_viscosity(theta_dof, theta_sol, epsilon, nu_water, nu_ferro);
//
//   // Add body force (e.g. for MMS):
//   forces.set_body_force(&body_force_fn, time);
// ============================================================================
template <int dim>
struct NSForceData
{
    // ---- Body force (external, e.g. MMS) ----
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>*
        body_force = nullptr;
    double body_force_time = 0.0;

    // ---- Kelvin force: mu_0(M.grad)H ----
    bool has_kelvin = false;
    const dealii::DoFHandler<dim>* phi_dof_handler = nullptr;
    const dealii::DoFHandler<dim>* M_dof_handler = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* phi_solution = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* Mx_solution = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* My_solution = nullptr;
    double mu_0 = 1.0;
    const Parameters* kelvin_params = nullptr;   // for dipoles, reduced field, etc.
    double kelvin_time = 0.0;

    // ---- Capillary force: (lambda/epsilon) theta grad(psi) ----
    bool has_capillary = false;
    const dealii::DoFHandler<dim>* theta_cap_dof_handler = nullptr;
    const dealii::DoFHandler<dim>* psi_dof_handler = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* theta_cap_solution = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* psi_solution = nullptr;
    double lambda = 1.0;
    double epsilon_cap = 0.01;

    // ---- Gravity force: r*H(theta/epsilon)*g ----
    bool has_gravity = false;
    double r = 0.0;
    double gravity_mag = 0.0;
    dealii::Tensor<1, dim> gravity_dir;

    // ---- Variable viscosity: nu(theta) = nu_water + (nu_ferro - nu_water)*H(theta/epsilon) ----
    bool has_variable_viscosity = false;
    const dealii::DoFHandler<dim>* theta_visc_dof_handler = nullptr;
    const dealii::TrilinosWrappers::MPI::Vector* theta_visc_solution = nullptr;
    double epsilon_visc = 0.01;
    double nu_water = 1.0;
    double nu_ferro = 2.0;

    // ---- Convenience setters ----
    void set_body_force(
        const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>* bf,
        double time)
    {
        body_force = bf;
        body_force_time = time;
    }

    void enable_kelvin(
        const dealii::DoFHandler<dim>& phi_dof,
        const dealii::DoFHandler<dim>& M_dof,
        const dealii::TrilinosWrappers::MPI::Vector& phi_sol,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_sol,
        const dealii::TrilinosWrappers::MPI::Vector& My_sol,
        double mu0,
        const Parameters& params,
        double time)
    {
        has_kelvin = true;
        phi_dof_handler = &phi_dof;
        M_dof_handler = &M_dof;
        phi_solution = &phi_sol;
        Mx_solution = &Mx_sol;
        My_solution = &My_sol;
        mu_0 = mu0;
        kelvin_params = &params;
        kelvin_time = time;
    }

    void enable_capillary(
        const dealii::DoFHandler<dim>& theta_dof,
        const dealii::DoFHandler<dim>& psi_dof,
        const dealii::TrilinosWrappers::MPI::Vector& theta_sol,
        const dealii::TrilinosWrappers::MPI::Vector& psi_sol,
        double lam,
        double eps)
    {
        has_capillary = true;
        theta_cap_dof_handler = &theta_dof;
        psi_dof_handler = &psi_dof;
        theta_cap_solution = &theta_sol;
        psi_solution = &psi_sol;
        lambda = lam;
        epsilon_cap = eps;
    }

    void enable_gravity_force(double r_val, double grav_mag,
                              const dealii::Tensor<1, dim>& grav_dir)
    {
        has_gravity = true;
        r = r_val;
        gravity_mag = grav_mag;
        gravity_dir = grav_dir;
    }

    void enable_variable_viscosity(
        const dealii::DoFHandler<dim>& theta_dof,
        const dealii::TrilinosWrappers::MPI::Vector& theta_sol,
        double eps, double nu_w, double nu_f)
    {
        has_variable_viscosity = true;
        theta_visc_dof_handler = &theta_dof;
        theta_visc_solution = &theta_sol;
        epsilon_visc = eps;
        nu_water = nu_w;
        nu_ferro = nu_f;
    }
};


// ============================================================================
// Unified NS assembly function
//
// Assembles the complete Navier-Stokes system with all optional forces
// controlled by the NSForceData struct. This is the single entry point
// that all other assembly functions delegate to.
//
// Core assembly (always):
//   - Time derivative: (U/dt, V)
//   - Viscous: nu/4 (T(U), T(V)) with T = grad + grad^T
//   - Convection: B_h(U_old, U, V) skew-symmetric
//   - Pressure: -(p, div V), (div U, q)
//   - MMS source (if enable_mms)
//   - Body force (if forces.body_force != nullptr)
//
// Optional forces (via NSForceData):
//   - Kelvin: mu_0(M.grad)H + face terms
//   - Capillary: (lambda/epsilon) theta grad(psi)
//   - Gravity: r*H(theta/epsilon)*g
//   - Variable viscosity: nu(theta) per quadrature point
// ============================================================================
template <int dim>
void assemble_ns_system_unified(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    const NSForceData<dim>& forces = NSForceData<dim>{},
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0,
    bool mms_analytical_dt = false);


// ============================================================================
// Legacy wrapper: Basic NS assembly (no forces)
// ============================================================================
template <int dim>
void assemble_ns_system_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0,
    bool mms_analytical_dt = false);

// ============================================================================
// Legacy wrapper: NS assembly with body force
// ============================================================================
template <int dim>
void assemble_ns_system_with_body_force_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>& body_force,
    double current_time,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0,
    bool mms_analytical_dt = false);

// ============================================================================
// Legacy wrapper: NS assembly with Kelvin force only (used by MMS tests)
// ============================================================================
template <int dim>
void assemble_ns_system_with_kelvin_force_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0,
    bool mms_analytical_dt = false);

// ============================================================================
// Legacy wrapper: Full ferrofluid NS assembly (Kelvin + Capillary + Gravity)
// ============================================================================
template <int dim>
void assemble_ns_system_ferrofluid_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_old,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old,
    double nu,
    double dt,
    bool include_time_derivative,
    bool include_convection,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ns_matrix,
    dealii::TrilinosWrappers::MPI::Vector& ns_rhs,
    MPI_Comm mpi_comm,
    // Kelvin force inputs (magnetic)
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double mu_0,
    // Capillary force inputs (phase field)
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    double lambda,
    double epsilon,
    // Variable viscosity inputs (Paper Eq. 17)
    double nu_water,
    double nu_ferro,
    // Gravity force inputs
    bool enable_gravity,
    double r,
    double gravity_mag,
    const dealii::Tensor<1, dim>& gravity_dir,
    // Simulation parameters (dipoles, reduced field, MMS)
    const Parameters& params,
    double current_time,
    // MMS options
    bool enable_mms = false,
    double mms_time = 0.0,
    double mms_time_old = 0.0,
    double mms_L_y = 1.0);

#endif // NS_ASSEMBLER_H
