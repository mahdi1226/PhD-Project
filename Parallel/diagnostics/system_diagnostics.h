// ============================================================================
// diagnostics/system_diagnostics.h - System Diagnostics Orchestrator
//
// Orchestrates all subsystem diagnostics and fills a StepData struct.
// This is the single entry point for computing all diagnostics each step.
//
// Usage:
//   StepData data = compute_system_diagnostics<2>(
//       theta_dof, theta_solution,
//       phi_dof, phi_solution,
//       ux_dof, ux_solution,
//       uy_dof, uy_solution,
//       p_dof, p_solution,
//       params, step, time, dt, h_min, prev_data, comm);
// ============================================================================
#ifndef SYSTEM_DIAGNOSTICS_H
#define SYSTEM_DIAGNOSTICS_H

#include "diagnostics/step_data.h"
#include "diagnostics/ch_diagnostics.h"
#include "diagnostics/ns_diagnostics.h"
#include "diagnostics/poisson_diagnostics.h"
#include "diagnostics/interface_tracking.h"
#include "diagnostics/force_diagnostics.h"
#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/distributed/tria.h>

// ============================================================================
// Compute all system diagnostics (parallel version)
// ============================================================================
template <int dim>
StepData compute_system_diagnostics(
    // CH
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    // Poisson (can be nullptr if magnetic disabled)
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector* phi_solution,
    // NS (can be nullptr if NS disabled)
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::DoFHandler<dim>* p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector* ux_solution,
    const dealii::TrilinosWrappers::MPI::Vector* uy_solution,
    const dealii::TrilinosWrappers::MPI::Vector* p_solution,
    // Parameters and state
    const Parameters& params,
    unsigned int step,
    double time,
    double dt,
    double h_min,
    const StepData& prev_data,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    StepData data;
    data.step = step;
    data.time = time;
    data.dt = dt;

    // ========================================================================
    // Cahn-Hilliard diagnostics
    // ========================================================================
    CHDiagnostics ch = compute_ch_diagnostics<dim>(
        theta_dof_handler, theta_solution, params, comm);

    data.theta_min = ch.theta_min;
    data.theta_max = ch.theta_max;
    data.mass = ch.mass;
    data.E_CH = ch.energy;
    data.theta_bounds_violated = ch.bounds_violated;

    // ========================================================================
    // Poisson/Magnetic diagnostics
    // ========================================================================
    if (params.enable_magnetic && phi_dof_handler != nullptr && phi_solution != nullptr)
    {
        PoissonDiagnostics poisson = compute_poisson_diagnostics<dim>(
            *phi_dof_handler, *phi_solution,
            theta_dof_handler, theta_solution,
            params, comm);

        data.phi_min = poisson.phi_min;
        data.phi_max = poisson.phi_max;
        data.H_max = poisson.H_max;
        data.M_max = poisson.M_max;
        data.E_mag = poisson.magnetic_energy;
        data.mu_min = poisson.mu_min;
        data.mu_max = poisson.mu_max;
    }

    // ========================================================================
    // Navier-Stokes diagnostics
    // ========================================================================
    if (params.enable_ns &&
        ux_dof_handler != nullptr && uy_dof_handler != nullptr && p_dof_handler != nullptr &&
        ux_solution != nullptr && uy_solution != nullptr && p_solution != nullptr)
    {
        NSDiagnostics ns = compute_ns_diagnostics<dim>(
            *ux_dof_handler, *uy_dof_handler, *p_dof_handler,
            *ux_solution, *uy_solution, *p_solution,
            dt, h_min, comm);

        data.ux_min = ns.ux_min;
        data.ux_max = ns.ux_max;
        data.uy_min = ns.uy_min;
        data.uy_max = ns.uy_max;
        data.U_max = ns.U_max;
        data.E_kin = ns.kinetic_energy;
        data.divU_L2 = ns.div_U_L2;
        data.divU_Linf = ns.div_U_max;
        data.CFL = ns.cfl;
        data.p_min = ns.p_min;
        data.p_max = ns.p_max;
    }

    // ========================================================================
    // Interface tracking
    // ========================================================================
    InterfacePosition iface = compute_interface_position<dim>(
        theta_dof_handler, theta_solution, comm);

    data.interface_y_min = iface.y_min;
    data.interface_y_max = iface.y_max;
    data.interface_y_mean = iface.y_mean;

    // Store initial interface position on first step
    if (step == 0)
    {
        data.interface_y_initial = iface.y_max;
    }
    else
    {
        data.interface_y_initial = prev_data.interface_y_initial;
    }

    // Estimate spike count (rough)
    double domain_width = params.domain.x_max - params.domain.x_min;
    data.spike_count = estimate_spike_count(iface, domain_width);

    // ========================================================================
    // Compute derived quantities
    // ========================================================================
    data.compute_derived();
    data.compute_energy_rates(prev_data);

    return data;
}

// ============================================================================
// Compute all system diagnostics (serial version)
// ============================================================================
template <int dim>
StepData compute_system_diagnostics(
    // CH
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    // Poisson (can be nullptr if magnetic disabled)
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>* phi_solution,
    // NS (can be nullptr if NS disabled)
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::DoFHandler<dim>* p_dof_handler,
    const dealii::Vector<double>* ux_solution,
    const dealii::Vector<double>* uy_solution,
    const dealii::Vector<double>* p_solution,
    // Parameters and state
    const Parameters& params,
    unsigned int step,
    double time,
    double dt,
    double h_min,
    const StepData& prev_data)
{
    StepData data;
    data.step = step;
    data.time = time;
    data.dt = dt;

    // CH diagnostics
    CHDiagnostics ch = compute_ch_diagnostics<dim>(
        theta_dof_handler, theta_solution, params);

    data.theta_min = ch.theta_min;
    data.theta_max = ch.theta_max;
    data.mass = ch.mass;
    data.E_CH = ch.energy;
    data.theta_bounds_violated = ch.bounds_violated;

    // Poisson diagnostics
    if (params.enable_magnetic && phi_dof_handler != nullptr && phi_solution != nullptr)
    {
        PoissonDiagnostics poisson = compute_poisson_diagnostics<dim>(
            *phi_dof_handler, *phi_solution,
            theta_dof_handler, theta_solution,
            params);

        data.phi_min = poisson.phi_min;
        data.phi_max = poisson.phi_max;
        data.H_max = poisson.H_max;
        data.M_max = poisson.M_max;
        data.E_mag = poisson.magnetic_energy;
        data.mu_min = poisson.mu_min;
        data.mu_max = poisson.mu_max;
    }

    // NS diagnostics
    if (params.enable_ns &&
        ux_dof_handler != nullptr && uy_dof_handler != nullptr && p_dof_handler != nullptr &&
        ux_solution != nullptr && uy_solution != nullptr && p_solution != nullptr)
    {
        NSDiagnostics ns = compute_ns_diagnostics<dim>(
            *ux_dof_handler, *uy_dof_handler, *p_dof_handler,
            *ux_solution, *uy_solution, *p_solution,
            dt, h_min);

        data.ux_min = ns.ux_min;
        data.ux_max = ns.ux_max;
        data.uy_min = ns.uy_min;
        data.uy_max = ns.uy_max;
        data.U_max = ns.U_max;
        data.E_kin = ns.kinetic_energy;
        data.divU_L2 = ns.div_U_L2;
        data.divU_Linf = ns.div_U_max;
        data.CFL = ns.cfl;
        data.p_min = ns.p_min;
        data.p_max = ns.p_max;
    }

    // Interface tracking
    InterfacePosition iface = compute_interface_position<dim>(
        theta_dof_handler, theta_solution);

    data.interface_y_min = iface.y_min;
    data.interface_y_max = iface.y_max;
    data.interface_y_mean = iface.y_mean;

    if (step == 0)
        data.interface_y_initial = iface.y_max;
    else
        data.interface_y_initial = prev_data.interface_y_initial;

    double domain_width = params.domain.x_max - params.domain.x_min;
    data.spike_count = estimate_spike_count(iface, domain_width);

    data.compute_derived();
    data.compute_energy_rates(prev_data);

    return data;
}

// ============================================================================
// Update StepData with solver info (call after each solve)
// ============================================================================
inline void update_ch_solver_info(StepData& data,
                                   unsigned int iterations,
                                   double residual,
                                   double solve_time,
                                   bool fallback_used = false)
{
    data.ch_iterations = iterations;
    data.ch_residual = residual;
    data.ch_time = solve_time;
    data.solver_fallback_used = data.solver_fallback_used || fallback_used;
}

inline void update_poisson_solver_info(StepData& data,
                                        unsigned int iterations,
                                        double residual,
                                        double solve_time)
{
    data.poisson_iterations = iterations;
    data.poisson_residual = residual;
    data.poisson_time = solve_time;
}

inline void update_mag_solver_info(StepData& data,
                                    unsigned int iterations,
                                    double residual,
                                    double solve_time)
{
    data.mag_iterations = iterations;
    data.mag_residual = residual;
    data.mag_time = solve_time;
}

inline void update_ns_solver_info(StepData& data,
                                   unsigned int outer_iterations,
                                   unsigned int inner_iterations,
                                   double residual,
                                   double solve_time,
                                   bool fallback_used = false)
{
    data.ns_outer_iterations = outer_iterations;
    data.ns_inner_iterations = inner_iterations;
    data.ns_residual = residual;
    data.ns_time = solve_time;
    data.solver_fallback_used = data.solver_fallback_used || fallback_used;
}

inline void update_timing_info(StepData& data,
                                double step_wall_time,
                                double total_wall_time)
{
    data.wall_time_step = step_wall_time;
    data.wall_time_total = total_wall_time;
}

inline void update_mesh_info(StepData& data,
                              unsigned int n_active_cells,
                              unsigned int n_dofs_total)
{
    data.n_active_cells = n_active_cells;
    data.n_dofs_total = n_dofs_total;
}

// ============================================================================
// Update StepData with force diagnostics (call separately if needed)
// Requires psi_solution which isn't always available in main diagnostics call
// ============================================================================
template <int dim>
void update_force_diagnostics(
    StepData& data,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& psi_solution,
    const dealii::TrilinosWrappers::MPI::Vector* phi_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    ForceDiagnostics forces = compute_force_diagnostics<dim>(
        theta_dof_handler, theta_solution, psi_solution, phi_solution, params, comm);

    data.F_cap_max = forces.F_cap_max;
    data.F_mag_max = forces.F_mag_max;
    data.F_grav_max = forces.F_grav_max;
}

template <int dim>
void update_force_diagnostics(
    StepData& data,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params)
{
    ForceDiagnostics forces = compute_force_diagnostics<dim>(
        theta_dof_handler, theta_solution, psi_solution, phi_solution, params);

    data.F_cap_max = forces.F_cap_max;
    data.F_mag_max = forces.F_mag_max;
    data.F_grav_max = forces.F_grav_max;
}

#endif // SYSTEM_DIAGNOSTICS_H