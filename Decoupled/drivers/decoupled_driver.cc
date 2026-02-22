// ============================================================================
// drivers/decoupled_driver.cc — Strategy A: Fully Decoupled Ferrofluid Solver
//
// Gauss-Seidel splitting: each subsystem solved ONCE per timestep.
// No Picard iteration — all cross-subsystem coupling is lagged.
//
// Algorithm (per timestep n):
//   1. Cahn-Hilliard:   θ^n, ψ^n  using U^{n-1}
//   2. Poisson:         φ^n        using M^{n-1}
//      Compute H^n = ∇φ^n
//   3. Magnetization:   M^n        using H^n, U^{n-1}, θ^{n-1}
//   4. Navier-Stokes:   U^n, p^n   using H^n, M^n, θ^{n-1}
//      (Kelvin force from current H, M; viscosity from lagged θ)
//
// Characteristics:
//   - 4 solves per step (cheapest strategy)
//   - O(dt) splitting error
//   - Energy-stable with lagged coefficients (Theorem 4.1)
//
// Usage:
//   mpirun -np 4 ./ferrofluid_decoupled --rosensweig --refinement 5
//   mpirun -np 4 ./ferrofluid_decoupled --rosensweig -r 4 --t_final 0.5
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42f
// ============================================================================

#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "cahn_hilliard/cahn_hilliard.h"
#include "navier_stokes/navier_stokes.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <string>
#include <sys/stat.h>

constexpr int dim = 2;


// ============================================================================
// Discrete energy (Theorem 4.1, Eq. 40)
//
// E^n = ½||U^n||² + λε/2||∇θ^n||² + (λ/ε)(F(θ^n), 1)
//     + μ₀/(2τ_M)||M^n||² − μ₀(M^n, H^n)
// ============================================================================
static double compute_discrete_energy(
    const CahnHilliardSubsystem<dim>::Diagnostics& ch_diag,
    const NSSubsystem<dim>::Diagnostics& ns_diag,
    const MagnetizationSubsystem<dim>::Diagnostics& /*mag_diag*/,
    const PoissonSubsystem<dim>::Diagnostics& /*poi_diag*/,
    const Parameters& /*params*/)
{
    // E_kin = ½∫|U|² dΩ  (from NS diagnostics)
    // E_CH = λε/2∫|∇θ|² dΩ + λ/ε∫F(θ) dΩ  (from CH diagnostics)
    // Magnetic energy terms would require quadrature over M·H — approximate
    return ns_diag.E_kin + ch_diag.E_total;
}


// ============================================================================
// Multi-file CSV logging system
//
// Output files (all in output_dir/):
//   run_info.txt         — Full parameter dump (once at startup)
//   diagnostics.csv      — Primary per-step summary (compact: key physics)
//   energy.csv           — Energy components and rates
//   timing.csv           — Per-subsystem wall times (assemble + solve)
//   solver.csv           — Iterations, residuals, convergence
//   flow.csv             — CFL, div(U), vorticity, pressure, Reynolds
//   magnetic.csv         — M/H statistics, equilibrium departure, alignment
//
// Design: each file is a focused "family" for easy post-processing.
// ============================================================================

// Helper: write the run config stamp as a CSV comment header
static void write_csv_stamp(std::ofstream& ofs, const Parameters& params,
                            unsigned int n_ranks, unsigned int n_cells,
                            unsigned int n_dofs)
{
    ofs << "# date=" << get_timestamp()
        << ",preset=" << (params.validation_test.empty() ? "production" : params.validation_test)
        << ",dt=" << params.time.dt
        << ",t_final=" << params.time.t_final
        << ",ref=" << params.mesh.initial_refinement
        << ",eps=" << params.physics.epsilon
        << ",lambda=" << params.physics.lambda
        << ",chi0=" << params.physics.chi_0
        << ",ranks=" << n_ranks
        << ",cells=" << n_cells
        << ",dofs=" << n_dofs
        << ",coupling=decoupled"
        << "\n";
}


class CSVLoggerFamily
{
public:
    CSVLoggerFamily(const std::string& output_dir, MPI_Comm comm,
                    const Parameters& params,
                    unsigned int n_cells, unsigned int n_dofs)
        : dir_(output_dir)
        , rank_(dealii::Utilities::MPI::this_mpi_process(comm))
        , n_ranks_(dealii::Utilities::MPI::n_mpi_processes(comm))
        , initial_mass_(0.0)
        , prev_E_total_(0.0)
        , prev_E_CH_(0.0)
        , prev_E_kin_(0.0)
        , step_count_(0)
    {
        if (rank_ != 0) return;

        // ---- run_info.txt ----
        {
            std::ofstream ri(dir_ + "/run_info.txt");
            ri << "Ferrofluid Decoupled Driver (Strategy A)\n";
            ri << "========================================\n\n";
            ri << "Date:       " << get_timestamp() << "\n";
            ri << "MPI ranks:  " << n_ranks_ << "\n";
            ri << "Cells:      " << n_cells << "\n";
            ri << "Total DoFs: " << n_dofs << "\n\n";
            ri << "[Domain]\n";
            ri << "  x: [" << params.domain.x_min << ", " << params.domain.x_max << "]\n";
            ri << "  y: [" << params.domain.y_min << ", " << params.domain.y_max << "]\n";
            ri << "  initial_cells: " << params.domain.initial_cells_x
               << " x " << params.domain.initial_cells_y << "\n";
            ri << "  refinement: " << params.mesh.initial_refinement << "\n\n";
            ri << "[Time]\n";
            ri << "  dt:        " << params.time.dt << "\n";
            ri << "  t_final:   " << params.time.t_final << "\n";
            ri << "  max_steps: " << params.time.max_steps << "\n\n";
            ri << "[Physics]\n";
            ri << "  epsilon:   " << params.physics.epsilon << "\n";
            ri << "  lambda:    " << params.physics.lambda << "\n";
            ri << "  mobility:  " << params.physics.mobility << "\n";
            ri << "  chi_0:     " << params.physics.chi_0 << "\n";
            ri << "  tau_M:     " << params.physics.tau_M << "\n";
            ri << "  nu_water:  " << params.physics.nu_water << "\n";
            ri << "  nu_ferro:  " << params.physics.nu_ferro << "\n";
            ri << "  r (density ratio): " << params.physics.r << "\n";
            ri << "  gravity:   " << params.physics.gravity_magnitude << "\n\n";
            ri << "[Enables]\n";
            ri << "  magnetic:  " << (params.enable_magnetic ? "ON" : "OFF") << "\n";
            ri << "  gravity:   " << (params.enable_gravity ? "ON" : "OFF") << "\n";
            ri << "  NS:        " << (params.enable_ns ? "ON" : "OFF") << "\n\n";
            ri << "[Validation]\n";
            ri << "  test: " << (params.validation_test.empty() ? "none" : params.validation_test) << "\n\n";
            ri << "[Applied Field]\n";
            if (params.uniform_field.enabled)
            {
                ri << "  type: uniform\n";
                ri << "  intensity_max: " << params.uniform_field.intensity_max << "\n";
                ri << "  ramp_time: " << params.uniform_field.ramp_time << "\n";
                ri << "  direction: (" << params.uniform_field.direction[0]
                   << ", " << params.uniform_field.direction[1] << ")\n";
            }
            else if (!params.dipoles.positions.empty())
            {
                ri << "  type: dipoles\n";
                ri << "  count: " << params.dipoles.positions.size() << "\n";
                ri << "  intensity_max: " << params.dipoles.intensity_max << "\n";
                ri << "  ramp_time: " << params.dipoles.ramp_time << "\n";
                for (unsigned int i = 0; i < params.dipoles.positions.size(); ++i)
                    ri << "  pos[" << i << "]: (" << params.dipoles.positions[i] << ")\n";
            }
            else
                ri << "  type: none\n";
            ri << "\n[Output]\n";
            ri << "  vtk_interval: " << params.output.vtk_interval << "\n";
            ri << "  folder: " << params.output.folder << "\n";
        }

        // ---- diagnostics.csv ---- (compact per-step summary)
        diag_.open(dir_ + "/diagnostics.csv");
        write_csv_stamp(diag_, params, n_ranks_, n_cells, n_dofs);
        diag_ << "step,time,dt,"
              << "theta_min,theta_max,theta_mass,mass_drift_rel,"
              << "E_CH,E_kin,E_mag,E_total,"
              << "U_max,CFL,divU_L2,"
              << "H_max,M_mag_max,"
              << "wall_time_s"
              << "\n";

        // ---- energy.csv ---- (all energy components + rates)
        energy_.open(dir_ + "/energy.csv");
        write_csv_stamp(energy_, params, n_ranks_, n_cells, n_dofs);
        energy_ << "step,time,"
                << "E_CH_grad,E_CH_bulk,E_CH,"
                << "E_kin,E_mag,E_total,"
                << "dE_CH_dt,dE_kin_dt,dE_total_dt"
                << "\n";

        // ---- timing.csv ---- (per-subsystem performance)
        timing_.open(dir_ + "/timing.csv");
        write_csv_stamp(timing_, params, n_ranks_, n_cells, n_dofs);
        timing_ << "step,time,"
                << "poi_assemble_s,poi_solve_s,"
                << "mag_assemble_s,mag_solve_s,"
                << "ch_assemble_s,ch_solve_s,"
                << "ns_assemble_s,ns_solve_s,"
                << "wall_step_s"
                << "\n";

        // ---- solver.csv ---- (iterations, residuals, convergence)
        solver_.open(dir_ + "/solver.csv");
        write_csv_stamp(solver_, params, n_ranks_, n_cells, n_dofs);
        solver_ << "step,time,"
                << "poi_iters,poi_residual,"
                << "mag_Mx_iters,mag_Mx_residual,"
                << "mag_My_iters,mag_My_residual,"
                << "ch_iters,ch_residual,"
                << "ns_iters,ns_residual,"
                << "gauss_law_residual"
                << "\n";

        // ---- flow.csv ---- (CFL, incompressibility, vorticity, pressure)
        flow_.open(dir_ + "/flow.csv");
        write_csv_stamp(flow_, params, n_ranks_, n_cells, n_dofs);
        flow_ << "step,time,"
              << "ux_min,ux_max,uy_min,uy_max,U_max,"
              << "E_kin,CFL,"
              << "divU_L2,divU_Linf,"
              << "p_min,p_max,p_mean,"
              << "omega_L2,omega_Linf,enstrophy,"
              << "Re_max"
              << "\n";

        // ---- magnetic.csv ---- (M/H statistics, equilibrium, alignment)
        magnetic_.open(dir_ + "/magnetic.csv");
        write_csv_stamp(magnetic_, params, n_ranks_, n_cells, n_dofs);
        magnetic_ << "step,time,"
                  << "phi_min,phi_max,H_max,H_L2,E_mag,"
                  << "mu_min,mu_max,"
                  << "M_mag_mean,M_mag_min,M_mag_max,"
                  << "Mx_mean,My_mean,"
                  << "M_eq_departure_L2,M_air_max,"
                  << "M_H_alignment,M_cross_H_L2,"
                  << "Mx_integral,My_integral"
                  << "\n";

        // ---- phase_field.csv ---- (detailed CH diagnostics)
        phase_.open(dir_ + "/phase_field.csv");
        write_csv_stamp(phase_, params, n_ranks_, n_cells, n_dofs);
        phase_ << "step,time,"
               << "theta_min,theta_max,theta_mean,theta_mass,"
               << "mass_drift_rel,"
               << "psi_min,psi_max,psi_L2,"
               << "E_CH_grad,E_CH_bulk,E_CH,"
               << "interface_length"
               << "\n";
    }


    void log(unsigned int step, double time, double dt,
             const CahnHilliardSubsystem<dim>::Diagnostics& ch,
             const PoissonSubsystem<dim>::Diagnostics& poi,
             const MagnetizationSubsystem<dim>::Diagnostics& mag,
             const NSSubsystem<dim>::Diagnostics& ns,
             double E_total, double wall_s)
    {
        if (rank_ != 0) return;

        // Track initial mass for drift computation
        if (step_count_ == 0)
            initial_mass_ = ch.mass_integral;
        ++step_count_;

        // Mass drift relative
        const double mass_drift = (std::abs(initial_mass_) > 1e-15)
            ? std::abs(ch.mass_integral - initial_mass_) / std::abs(initial_mass_)
            : std::abs(ch.mass_integral - initial_mass_);

        // Energy rates
        const double dE_CH_dt   = (step_count_ > 1) ? (ch.E_total - prev_E_CH_) / dt : 0.0;
        const double dE_kin_dt  = (step_count_ > 1) ? (ns.E_kin - prev_E_kin_) / dt : 0.0;
        const double dE_total_dt = (step_count_ > 1) ? (E_total - prev_E_total_) / dt : 0.0;
        prev_E_CH_   = ch.E_total;
        prev_E_kin_  = ns.E_kin;
        prev_E_total_ = E_total;

        // Scientific format helper
        auto S = [](std::ofstream& o) -> std::ofstream& {
            o << std::scientific << std::setprecision(6);
            return o;
        };

        // ---- diagnostics.csv ---- (compact summary)
        S(diag_) << step << "," << time << "," << dt << ","
                 << ch.theta_min << "," << ch.theta_max << ","
                 << ch.mass_integral << "," << mass_drift << ","
                 << ch.E_total << "," << ns.E_kin << "," << poi.E_mag << ","
                 << E_total << ","
                 << ns.U_max << "," << ns.CFL << "," << ns.divU_L2 << ","
                 << poi.H_max << "," << mag.M_magnitude_max << ","
                 << std::fixed << std::setprecision(2) << wall_s
                 << "\n";
        diag_.flush();

        // ---- energy.csv ----
        S(energy_) << step << "," << time << ","
                   << ch.E_gradient << "," << ch.E_bulk << "," << ch.E_total << ","
                   << ns.E_kin << "," << poi.E_mag << "," << E_total << ","
                   << dE_CH_dt << "," << dE_kin_dt << "," << dE_total_dt
                   << "\n";
        energy_.flush();

        // ---- timing.csv ----
        S(timing_) << step << "," << time << ","
                   << poi.assemble_time << "," << poi.solve_time << ","
                   << mag.assemble_time << "," << mag.solve_time << ","
                   << ch.assemble_time << "," << ch.solve_time << ","
                   << ns.assemble_time << "," << ns.solve_time << ","
                   << std::fixed << std::setprecision(4) << wall_s
                   << "\n";
        timing_.flush();

        // ---- solver.csv ----
        S(solver_) << step << "," << time << ","
                   << poi.iterations << "," << poi.residual << ","
                   << mag.Mx_iterations << "," << mag.Mx_residual << ","
                   << mag.My_iterations << "," << mag.My_residual << ","
                   << ch.iterations << "," << ch.residual << ","
                   << ns.iterations << "," << ns.residual << ","
                   << poi.gauss_law_residual
                   << "\n";
        solver_.flush();

        // ---- flow.csv ----
        S(flow_) << step << "," << time << ","
                 << ns.ux_min << "," << ns.ux_max << ","
                 << ns.uy_min << "," << ns.uy_max << "," << ns.U_max << ","
                 << ns.E_kin << "," << ns.CFL << ","
                 << ns.divU_L2 << "," << ns.divU_Linf << ","
                 << ns.p_min << "," << ns.p_max << "," << ns.p_mean << ","
                 << ns.omega_L2 << "," << ns.omega_Linf << ","
                 << ns.enstrophy << ","
                 << ns.Re_max
                 << "\n";
        flow_.flush();

        // ---- magnetic.csv ----
        S(magnetic_) << step << "," << time << ","
                     << poi.phi_min << "," << poi.phi_max << ","
                     << poi.H_max << "," << poi.H_L2 << "," << poi.E_mag << ","
                     << poi.mu_min << "," << poi.mu_max << ","
                     << mag.M_magnitude_mean << "," << mag.M_magnitude_min << ","
                     << mag.M_magnitude_max << ","
                     << mag.Mx_mean << "," << mag.My_mean << ","
                     << mag.M_equilibrium_departure_L2 << ","
                     << mag.M_air_phase_max << ","
                     << mag.M_H_alignment_mean << "," << mag.M_cross_H_L2 << ","
                     << mag.Mx_integral << "," << mag.My_integral
                     << "\n";
        magnetic_.flush();

        // ---- phase_field.csv ----
        S(phase_) << step << "," << time << ","
                  << ch.theta_min << "," << ch.theta_max << ","
                  << ch.theta_mean << "," << ch.mass_integral << ","
                  << mass_drift << ","
                  << ch.psi_min << "," << ch.psi_max << "," << ch.psi_L2 << ","
                  << ch.E_gradient << "," << ch.E_bulk << "," << ch.E_total << ","
                  << ch.interface_length
                  << "\n";
        phase_.flush();
    }

private:
    std::string dir_;
    unsigned int rank_;
    unsigned int n_ranks_;

    double initial_mass_;
    double prev_E_total_, prev_E_CH_, prev_E_kin_;
    unsigned int step_count_;

    std::ofstream diag_;
    std::ofstream energy_;
    std::ofstream timing_;
    std::ofstream solver_;
    std::ofstream flow_;
    std::ofstream magnetic_;
    std::ofstream phase_;
};


// ============================================================================
// Initial condition: flat interface at y = y_interface
//
// Zhang Eq 4.3: Φ₀ = 1 if y ≤ y_interface, Φ₀ = 0 otherwise (SHARP STEP)
// In {-1,+1} convention: θ = +1 if y ≤ y_interface, θ = -1 otherwise
//
// Maps to θ = +1 (ferrofluid, below) and θ = -1 (air, above)
// ============================================================================
class FlatInterfaceIC : public dealii::Function<dim>
{
public:
    FlatInterfaceIC(double y_interface, double /*epsilon*/,
                    double /*perturb_amp*/ = 0.0, double /*perturb_k*/ = 0.0)
        : dealii::Function<dim>(1)
        , y_if_(y_interface)
    {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/) const override
    {
        // Sharp step: Zhang Eq 4.3
        return (p[1] <= y_if_) ? 1.0 : -1.0;
    }

private:
    double y_if_;
};

class ZeroFunction : public dealii::Function<dim>
{
public:
    ZeroFunction() : dealii::Function<dim>(1) {}
    double value(const dealii::Point<dim>& /*p*/,
                 const unsigned int /*component*/) const override
    { return 0.0; }
};


// ============================================================================
// Circular droplet IC (validation: --validation droplet)
//
// θ = tanh((R - |x - x_c|) / (√2 ε))
// Ferrofluid (θ=+1) inside circle, air (θ=-1) outside.
// In absence of magnetic field, circle should remain circular.
// ============================================================================
class CircularDropletIC : public dealii::Function<dim>
{
public:
    CircularDropletIC(dealii::Point<dim> center, double radius, double /*epsilon*/)
        : dealii::Function<dim>(1)
        , center_(center)
        , R_(radius)
    {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/) const override
    {
        // Sharp step: Zhang Eq 4.8
        const double dist = center_.distance(p);
        return (dist <= R_) ? 1.0 : -1.0;
    }

private:
    dealii::Point<dim> center_;
    double R_;
};


// ============================================================================
// Diamond (rotated square) droplet IC (validation: --validation square)
//
// θ = tanh((R - d) / (√2 ε))
// where d = (|dx + dy| + |dx - dy|) / 2  is the L1 (diamond) distance.
// Ferrofluid (θ=+1) inside diamond, air (θ=-1) outside.
// Under magnetic field, the diamond should deform toward an elongated shape.
// ============================================================================
class DiamondDropletIC : public dealii::Function<dim>
{
public:
    DiamondDropletIC(dealii::Point<dim> center, double radius, double epsilon)
        : dealii::Function<dim>(1)
        , center_(center)
        , R_(radius)
        , eps_(epsilon)
    {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int /*component*/) const override
    {
        const double dx = p[0] - center_[0];
        const double dy = p[1] - center_[1];
        // L1 / diamond distance: d = |dx| + |dy|
        const double d = std::abs(dx) + std::abs(dy);
        return std::tanh((R_ - d) / (std::sqrt(2.0) * eps_));
    }

private:
    dealii::Point<dim> center_;
    double R_;
    double eps_;
};


// ============================================================================
// Combined VTU writer — single file with all subsystem fields
//
// Fields:
//   theta, psi           — Cahn-Hilliard (CG Q2)
//   phi                  — Poisson potential (CG Q1)
//   Mx, My               — Magnetization (DG Q1)
//   ux, uy               — Velocity (CG Q2)
//   p                    — Pressure (DG P1)
//   H_x, H_y, H_mag     — Derived: H = ∇φ (DG Q0, cell-averaged)
//   M_mag                — Derived: |M| (DG Q0, cell-averaged)
//   subdomain            — MPI partition
//
// Output: {output_dir}/solution_{step:05d}.pvtu
// ============================================================================
static void write_combined_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time,
    const CahnHilliardSubsystem<dim>&  ch,
    const PoissonSubsystem<dim>&       poisson,
    const MagnetizationSubsystem<dim>& mag,
    const NSSubsystem<dim>&            ns,
    const dealii::parallel::distributed::Triangulation<dim>& triangulation,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    // Ensure output directory exists
    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm);

    // --- DG Q0 space for cell-averaged derived fields ---
    FE_DGQ<dim> fe_dg0(0);
    DoFHandler<dim> dg0_dof(triangulation);
    dg0_dof.distribute_dofs(fe_dg0);

    IndexSet dg0_owned = dg0_dof.locally_owned_dofs();
    const IndexSet dg0_relevant =
        DoFTools::extract_locally_relevant_dofs(dg0_dof);

    TrilinosWrappers::MPI::Vector H_x_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector H_y_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector H_mag_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector M_mag_vec(dg0_owned, mpi_comm);

    // Compute H = ∇φ (cell-averaged) and |M| (cell-averaged)
    {
        const auto& fe_poi = poisson.get_dof_handler().get_fe();
        QGauss<dim> quadrature(fe_poi.degree + 1);
        FEValues<dim> fe_vals_poi(fe_poi, quadrature,
            update_gradients | update_JxW_values);

        const auto& fe_mag = mag.get_dof_handler().get_fe();
        FEValues<dim> fe_vals_mag(fe_mag, quadrature,
            update_values | update_JxW_values);

        const unsigned int n_q = quadrature.size();
        std::vector<Tensor<1, dim>> grad_phi(n_q);
        std::vector<double> Mx_vals(n_q), My_vals(n_q);

        auto cell_poi = poisson.get_dof_handler().begin_active();
        auto cell_mag = mag.get_dof_handler().begin_active();
        auto cell_dg0 = dg0_dof.begin_active();

        for (; cell_poi != poisson.get_dof_handler().end();
             ++cell_poi, ++cell_mag, ++cell_dg0)
        {
            if (!cell_poi->is_locally_owned())
                continue;

            // H = ∇φ
            fe_vals_poi.reinit(cell_poi);
            fe_vals_poi.get_function_gradients(
                poisson.get_solution_relevant(), grad_phi);

            // M components
            fe_vals_mag.reinit(cell_mag);
            fe_vals_mag.get_function_values(
                mag.get_Mx_relevant(), Mx_vals);
            fe_vals_mag.get_function_values(
                mag.get_My_relevant(), My_vals);

            Tensor<1, dim> avg_H;
            double avg_M_mag = 0.0;
            double vol = 0.0;

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW = fe_vals_poi.JxW(q);
                avg_H += grad_phi[q] * JxW;
                avg_M_mag += std::sqrt(Mx_vals[q] * Mx_vals[q]
                                     + My_vals[q] * My_vals[q]) * JxW;
                vol += JxW;
            }
            avg_H /= vol;
            avg_M_mag /= vol;

            std::vector<types::global_dof_index> dg0_idx(1);
            cell_dg0->get_dof_indices(dg0_idx);
            const auto idx = dg0_idx[0];

            H_x_vec(idx)   = avg_H[0];
            H_y_vec(idx)   = avg_H[1];
            H_mag_vec(idx) = avg_H.norm();
            M_mag_vec(idx) = avg_M_mag;
        }
    }

    H_x_vec.compress(VectorOperation::insert);
    H_y_vec.compress(VectorOperation::insert);
    H_mag_vec.compress(VectorOperation::insert);
    M_mag_vec.compress(VectorOperation::insert);

    // Ghosted versions for output
    TrilinosWrappers::MPI::Vector H_x_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector H_y_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector H_mag_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector M_mag_rel(dg0_owned, dg0_relevant, mpi_comm);
    H_x_rel   = H_x_vec;
    H_y_rel   = H_y_vec;
    H_mag_rel = H_mag_vec;
    M_mag_rel = M_mag_vec;

    // --- Build combined DataOut with all subsystem fields ---
    DataOut<dim> data_out;

    // Primary handler: CH theta (CG Q2 — highest polynomial degree)
    data_out.attach_dof_handler(ch.get_theta_dof_handler());
    data_out.add_data_vector(ch.get_theta_relevant(), "theta");

    // CH: psi (same FE space as theta, separate DoFHandler)
    data_out.add_data_vector(
        ch.get_psi_dof_handler(), ch.get_psi_relevant(), "psi");

    // Poisson: phi (CG Q1)
    data_out.add_data_vector(
        poisson.get_dof_handler(), poisson.get_solution_relevant(), "phi");

    // Magnetization: Mx, My (DG Q1)
    data_out.add_data_vector(
        mag.get_dof_handler(), mag.get_Mx_relevant(), "Mx");
    data_out.add_data_vector(
        mag.get_dof_handler(), mag.get_My_relevant(), "My");

    // NS: ux, uy (CG Q2), p (DG P1)
    data_out.add_data_vector(
        ns.get_ux_dof_handler(), ns.get_ux_relevant(), "ux");
    data_out.add_data_vector(
        ns.get_uy_dof_handler(), ns.get_uy_relevant(), "uy");
    data_out.add_data_vector(
        ns.get_p_dof_handler(), ns.get_p_relevant(), "p");

    // Derived: H components and magnitudes (DG Q0)
    data_out.add_data_vector(dg0_dof, H_x_rel,   "H_x");
    data_out.add_data_vector(dg0_dof, H_y_rel,   "H_y");
    data_out.add_data_vector(dg0_dof, H_mag_rel, "H_mag");
    data_out.add_data_vector(dg0_dof, M_mag_rel, "M_mag");

    // Subdomain coloring
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    // Build patches at highest FE degree (Q2)
    data_out.build_patches(2);

    // VTK flags
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.time = time;
    vtk_flags.cycle = step;
    vtk_flags.compression_level =
        DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    // Write parallel VTU + PVTU
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "solution", step, mpi_comm, /*n_digits=*/5);

    if (rank == 0)
    {
        std::cout << "  [VTK] " << output_dir << "/solution_"
                  << std::setfill('0') << std::setw(5) << step
                  << std::setfill(' ')
                  << ".pvtu (t=" << std::scientific << std::setprecision(3)
                  << time << ", all fields)\n" << std::defaultfloat;
    }
}


// ============================================================================
// Simplified VTK writer — CH fields only (for pure CH validation tests)
// ============================================================================
static void write_ch_only_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time,
    const CahnHilliardSubsystem<dim>& ch,
    const dealii::parallel::distributed::Triangulation<dim>& triangulation,
    MPI_Comm mpi_comm)
{
    using namespace dealii;
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(ch.get_theta_dof_handler());
    data_out.add_data_vector(ch.get_theta_relevant(), "theta");
    data_out.add_data_vector(
        ch.get_psi_dof_handler(), ch.get_psi_relevant(), "psi");

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(2);

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.time = time;
    vtk_flags.cycle = step;
    vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "solution", step, mpi_comm, /*n_digits=*/5);

    if (rank == 0)
    {
        std::cout << "  [VTK] " << output_dir << "/solution_"
                  << std::setfill('0') << std::setw(5) << step
                  << std::setfill(' ')
                  << ".pvtu (t=" << std::scientific << std::setprecision(3)
                  << time << ", CH only)\n" << std::defaultfloat;
    }
}


// ============================================================================
// Combined VTU writer — algebraic M mode (all fields, M computed from θ,φ)
//
// Fields:
//   theta, psi           — Cahn-Hilliard (CG Q2)
//   phi                  — Poisson potential (CG Q2)
//   ux, uy               — Velocity (CG Q2)
//   p                    — Pressure (DG P1)
//   H_x, H_y, H_mag     — Derived: H = ∇φ (DG Q0, cell-averaged)
//   M_x, M_y, M_mag     — Derived: M = χ(θ)H (DG Q0, cell-averaged)
//   chi                  — Derived: χ(θ) (DG Q0, cell-averaged)
//   subdomain            — MPI partition
// ============================================================================
static void write_algebraic_M_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time,
    const CahnHilliardSubsystem<dim>&  ch,
    const PoissonSubsystem<dim>&       poisson,
    const NSSubsystem<dim>&            ns,
    const Parameters&                  params,
    const dealii::parallel::distributed::Triangulation<dim>& triangulation,
    MPI_Comm mpi_comm)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);

    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm);

    // --- DG Q0 space for cell-averaged derived fields ---
    FE_DGQ<dim> fe_dg0(0);
    DoFHandler<dim> dg0_dof(triangulation);
    dg0_dof.distribute_dofs(fe_dg0);

    IndexSet dg0_owned = dg0_dof.locally_owned_dofs();
    const IndexSet dg0_relevant =
        DoFTools::extract_locally_relevant_dofs(dg0_dof);

    TrilinosWrappers::MPI::Vector H_x_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector H_y_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector H_mag_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector M_x_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector M_y_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector M_mag_vec(dg0_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector chi_vec(dg0_owned, mpi_comm);

    // Compute H = ∇φ, M = χ(θ)(∇φ + h_a) at cell-averaged quadrature points
    {
        const auto& fe_poi = poisson.get_dof_handler().get_fe();
        const auto& fe_ch  = ch.get_theta_dof_handler().get_fe();
        QGauss<dim> quadrature(fe_poi.degree + 1);
        FEValues<dim> fe_vals_poi(fe_poi, quadrature,
            update_gradients | update_quadrature_points | update_JxW_values);
        FEValues<dim> fe_vals_ch(fe_ch, quadrature,
            update_values | update_JxW_values);

        const unsigned int n_q = quadrature.size();
        std::vector<Tensor<1, dim>> grad_phi(n_q);
        std::vector<double> theta_vals(n_q);

        const double eps   = params.physics.epsilon;
        const double chi_0 = params.physics.chi_0;

        auto cell_poi = poisson.get_dof_handler().begin_active();
        auto cell_ch  = ch.get_theta_dof_handler().begin_active();
        auto cell_dg0 = dg0_dof.begin_active();

        for (; cell_poi != poisson.get_dof_handler().end();
             ++cell_poi, ++cell_ch, ++cell_dg0)
        {
            if (!cell_poi->is_locally_owned())
                continue;

            fe_vals_poi.reinit(cell_poi);
            fe_vals_poi.get_function_gradients(
                poisson.get_solution_relevant(), grad_phi);

            fe_vals_ch.reinit(cell_ch);
            fe_vals_ch.get_function_values(
                ch.get_theta_relevant(), theta_vals);

            Tensor<1, dim> avg_H, avg_M;
            double avg_chi = 0.0;
            double vol = 0.0;

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double JxW = fe_vals_poi.JxW(q);

                // H = ∇φ + h_a
                Tensor<1, dim> h_a = compute_applied_field(
                    fe_vals_poi.quadrature_point(q), params, time);
                Tensor<1, dim> H_total = grad_phi[q] + h_a;

                // χ(θ) and M = χ(θ) * H_total
                const double chi_q = susceptibility(theta_vals[q], eps, chi_0);
                Tensor<1, dim> M_q = chi_q * H_total;

                avg_H += grad_phi[q] * JxW;  // H_field = ∇φ (without h_a for consistency)
                avg_M += M_q * JxW;
                avg_chi += chi_q * JxW;
                vol += JxW;
            }
            avg_H /= vol;
            avg_M /= vol;
            avg_chi /= vol;

            std::vector<types::global_dof_index> dg0_idx(1);
            cell_dg0->get_dof_indices(dg0_idx);
            const auto idx = dg0_idx[0];

            H_x_vec(idx)   = avg_H[0];
            H_y_vec(idx)   = avg_H[1];
            H_mag_vec(idx) = avg_H.norm();
            M_x_vec(idx)   = avg_M[0];
            M_y_vec(idx)   = avg_M[1];
            M_mag_vec(idx) = avg_M.norm();
            chi_vec(idx)   = avg_chi;
        }
    }

    H_x_vec.compress(VectorOperation::insert);
    H_y_vec.compress(VectorOperation::insert);
    H_mag_vec.compress(VectorOperation::insert);
    M_x_vec.compress(VectorOperation::insert);
    M_y_vec.compress(VectorOperation::insert);
    M_mag_vec.compress(VectorOperation::insert);
    chi_vec.compress(VectorOperation::insert);

    // Ghosted versions for output
    TrilinosWrappers::MPI::Vector H_x_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector H_y_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector H_mag_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector M_x_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector M_y_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector M_mag_rel(dg0_owned, dg0_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector chi_rel(dg0_owned, dg0_relevant, mpi_comm);
    H_x_rel   = H_x_vec;
    H_y_rel   = H_y_vec;
    H_mag_rel = H_mag_vec;
    M_x_rel   = M_x_vec;
    M_y_rel   = M_y_vec;
    M_mag_rel = M_mag_vec;
    chi_rel   = chi_vec;

    // --- Build combined DataOut with all subsystem fields ---
    DataOut<dim> data_out;

    // Primary handler: CH theta (CG Q2)
    data_out.attach_dof_handler(ch.get_theta_dof_handler());
    data_out.add_data_vector(ch.get_theta_relevant(), "theta");

    // CH: psi
    data_out.add_data_vector(
        ch.get_psi_dof_handler(), ch.get_psi_relevant(), "psi");

    // Poisson: phi
    data_out.add_data_vector(
        poisson.get_dof_handler(), poisson.get_solution_relevant(), "phi");

    // NS: ux, uy, p
    data_out.add_data_vector(
        ns.get_ux_dof_handler(), ns.get_ux_relevant(), "ux");
    data_out.add_data_vector(
        ns.get_uy_dof_handler(), ns.get_uy_relevant(), "uy");
    data_out.add_data_vector(
        ns.get_p_dof_handler(), ns.get_p_relevant(), "p");

    // Derived: H, M, χ (DG Q0)
    data_out.add_data_vector(dg0_dof, H_x_rel,   "H_x");
    data_out.add_data_vector(dg0_dof, H_y_rel,   "H_y");
    data_out.add_data_vector(dg0_dof, H_mag_rel, "H_mag");
    data_out.add_data_vector(dg0_dof, M_x_rel,   "M_x");
    data_out.add_data_vector(dg0_dof, M_y_rel,   "M_y");
    data_out.add_data_vector(dg0_dof, M_mag_rel, "M_mag");
    data_out.add_data_vector(dg0_dof, chi_rel,    "chi");

    // Subdomain coloring
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(2);

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.time = time;
    vtk_flags.cycle = step;
    vtk_flags.compression_level =
        DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "solution", step, mpi_comm, /*n_digits=*/5);

    if (rank == 0)
    {
        std::cout << "  [VTK] " << output_dir << "/solution_"
                  << std::setfill('0') << std::setw(5) << step
                  << std::setfill(' ')
                  << ".pvtu (t=" << std::scientific << std::setprecision(3)
                  << time << ", algebraic M)\n" << std::defaultfloat;
    }
}


// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    // ----------------------------------------------------------------
    // Parse command line
    // ----------------------------------------------------------------
    Parameters params = Parameters::parse_command_line(argc, argv);

    const double dt      = params.time.dt;
    const double t_final = params.time.t_final;
    const unsigned int max_steps = params.time.max_steps;
    const unsigned int refinement = params.mesh.initial_refinement;

    const unsigned int vtk_interval = params.output.vtk_interval;

    pcout << "\n"
          << "============================================================\n"
          << "  Ferrofluid Solver — Strategy A: Fully Decoupled\n"
          << "  Gauss-Seidel Splitting (no Picard iteration)\n"
          << "============================================================\n"
          << "  MPI ranks:    " << n_procs << "\n"
          << "  Refinement:   " << refinement << "\n"
          << "  Domain:       [" << params.domain.x_min << "," << params.domain.x_max
          << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n"
          << "  dt:           " << dt << "\n"
          << "  t_final:      " << t_final << "\n"
          << "  max_steps:    " << max_steps << "\n"
          << "  VTK interval: " << vtk_interval << "\n"
          << "  Physics:\n"
          << "    epsilon:    " << params.physics.epsilon << "\n"
          << "    chi_0:      " << params.physics.chi_0 << "\n"
          << "    tau_M:      " << params.physics.tau_M << "\n"
          << "    mobility:   " << params.physics.mobility << "\n"
          << "    lambda:     " << params.physics.lambda << "\n"
          << "    nu_w/nu_f:  " << params.physics.nu_water << " / "
                                << params.physics.nu_ferro << "\n"
          << "    gravity:    " << params.physics.gravity_magnitude << "\n"
          << "    magnetic:   " << (params.enable_magnetic ? "ON" : "OFF") << "\n"
          << "    NS:         " << (params.enable_ns ? "ON" : "OFF") << "\n";
    if (params.uniform_field.enabled)
    {
        pcout << "    H_a:        uniform (" << params.uniform_field.intensity_max << ")";
        if (params.uniform_field.ramp_slope > 0.0)
            pcout << ", slope=" << params.uniform_field.ramp_slope;
        else
            pcout << ", ramp=" << params.uniform_field.ramp_time;
        pcout << "\n";
    }
    else if (!params.dipoles.positions.empty())
    {
        pcout << "    H_a:        " << params.dipoles.positions.size() << " dipoles";
        if (params.dipoles.ramp_slope > 0.0)
            pcout << ", slope=" << params.dipoles.ramp_slope;
        else
            pcout << ", intensity=" << params.dipoles.intensity_max
                  << ", ramp=" << params.dipoles.ramp_time;
        pcout << "\n";
    }
    pcout << "  Validation:   "
          << (params.validation_test.empty() ? "none" : params.validation_test) << "\n"
          << "============================================================\n\n";

    // ----------------------------------------------------------------
    // 1. Create shared triangulation
    // ----------------------------------------------------------------
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    GridGenerator::subdivided_hyper_rectangle(
        triangulation,
        {params.domain.initial_cells_x, params.domain.initial_cells_y},
        Point<dim>(params.domain.x_min, params.domain.y_min),
        Point<dim>(params.domain.x_max, params.domain.y_max));

    triangulation.refine_global(refinement);

    pcout << "  Mesh: " << triangulation.n_global_active_cells()
          << " cells (global)\n";

    // ----------------------------------------------------------------
    // 2. Create and setup all 4 subsystems
    // ----------------------------------------------------------------
    PoissonSubsystem<dim>        poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim>  mag(params, mpi_comm, triangulation);
    CahnHilliardSubsystem<dim>   ch(params, mpi_comm, triangulation);
    NSSubsystem<dim>             ns(params, mpi_comm, triangulation);

    ch.setup();
    if (params.enable_magnetic)
    {
        poisson.setup();
        if (!params.use_algebraic_magnetization)
            mag.setup();
    }
    if (params.enable_ns)
        ns.setup();

    pcout << "  DoFs: CH(θ+ψ)=" << ch.get_theta_dof_handler().n_dofs()
                            + ch.get_psi_dof_handler().n_dofs();
    if (params.enable_magnetic)
    {
        pcout << "  Poisson=" << poisson.get_dof_handler().n_dofs();
        if (!params.use_algebraic_magnetization)
            pcout << "  Mag(x2)=" << 2 * mag.get_dof_handler().n_dofs();
        else
            pcout << "  Mag=algebraic";
    }
    if (params.enable_ns)
        pcout << "  NS(ux+uy+p)=" << ns.get_ux_dof_handler().n_dofs()
                                    + ns.get_uy_dof_handler().n_dofs()
                                    + ns.get_p_dof_handler().n_dofs();
    if (params.use_sav)
        pcout << "  SAV=ON";
    pcout << "\n\n";

    // ----------------------------------------------------------------
    // 3. Initial conditions
    // ----------------------------------------------------------------
    pcout << "  Initializing fields...\n";

    // Select IC based on validation mode or default Rosensweig
    {
        const double eps = params.physics.epsilon;
        const double cx = 0.5 * (params.domain.x_min + params.domain.x_max);
        const double cy = 0.5 * (params.domain.y_min + params.domain.y_max);

        if (params.validation_test == "square")
        {
            // Diamond (L1 ball) at center, half-diagonal = 0.25
            // (Zhang et al. Section 4.2: vertices at (0.5±0.25, 0.5) and (0.5, 0.5±0.25))
            const double R = 0.25;
            pcout << "  IC: Diamond droplet (R=" << R
                  << ", center=(" << cx << "," << cy << "))\n";
            DiamondDropletIC theta_ic(Point<dim>(cx, cy), R, eps);
            ZeroFunction psi_ic;
            ch.project_initial_condition(theta_ic, psi_ic);
        }
        else if (params.validation_test == "droplet" || params.validation_test == "droplet_nofield")
        {
            // Circular droplet at center, R = 0.1
            // (Zhang et al. Section 4.5, Eq 4.8, Fig 4.14)
            const double R = 0.1;
            pcout << "  IC: Circular droplet (R=" << R
                  << ", center=(" << cx << "," << cy << "))\n";
            CircularDropletIC theta_ic(Point<dim>(cx, cy), R, eps);
            ZeroFunction psi_ic;
            ch.project_initial_condition(theta_ic, psi_ic);
        }
        else
        {
            // Default: flat interface (Rosensweig)
            // Zhang et al. SIAM J. Sci. Comput. 43(1), 2021, Eq 4.3:
            // Flat interface at y=0.2, NO perturbation.
            // Rosensweig instability emerges naturally from the magnetic field.
            const double y_interface = 0.2;
            pcout << "  IC: Flat interface (y=" << y_interface << ")\n";
            FlatInterfaceIC theta_ic(y_interface, eps);
            ZeroFunction psi_ic;
            ch.project_initial_condition(theta_ic, psi_ic);
        }
        ch.update_ghosts();
    }

    // NS: zero velocity (only if NS is enabled)
    if (params.enable_ns)
    {
        ns.initialize_zero();
        ns.update_ghosts();
    }

    // Poisson + Magnetization initialization (only if magnetic is enabled)
    if (params.enable_magnetic)
    {
        if (params.use_algebraic_magnetization)
        {
            // Algebraic M mode: solve nonlinear Poisson with theta-dependent coefficient
            poisson.assemble_nonlinear(
                ch.get_theta_relevant(), ch.get_theta_dof_handler(), 0.0);
            poisson.solve();
            poisson.update_ghosts();
            // M is computed algebraically at quadrature points — no initialization needed
        }
        else
        {
            // PDE M mode: Poisson with M = 0 initially, then equilibrium magnetization
            IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
            IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(
                mag.get_dof_handler());
            TrilinosWrappers::MPI::Vector Mx_zero(M_owned, M_relevant, mpi_comm);
            TrilinosWrappers::MPI::Vector My_zero(M_owned, M_relevant, mpi_comm);
            Mx_zero = 0;
            My_zero = 0;

            poisson.assemble_rhs(Mx_zero, My_zero, mag.get_dof_handler(), 0.0);
            poisson.solve();
            poisson.update_ghosts();

            mag.initialize_equilibrium(
                poisson.get_solution_relevant(), poisson.get_dof_handler(),
                ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                0.0);
            mag.update_ghosts();
        }
    }

    pcout << "  Initialization complete.\n\n";

    // ----------------------------------------------------------------
    // 4. Output directory and diagnostics logger
    // ----------------------------------------------------------------
    std::string output_dir;
    if (rank == 0)
    {
        // Determine descriptive test name for folder
        std::string test_name;
        if (params.validation_test == "square")
            test_name = "square";
        else if (params.validation_test == "droplet" || params.validation_test == "droplet_nofield")
        {
            if (params.enable_magnetic)
                test_name = "droplet_wfield";
            else
                test_name = "droplet_wofield";
        }
        else if (params.enable_magnetic && params.enable_gravity)
            test_name = "rosensweig";
        else if (params.enable_magnetic)
            test_name = "magnetic";
        else if (params.enable_ns)
            test_name = "flow";
        else
            test_name = "ch_only";

        // Format: ddmmyy_hhmmss_testname_r<N>
        output_dir = params.output.folder + "/"
                   + get_timestamp() + "_" + test_name
                   + "_r" + std::to_string(refinement);
        mkdir(output_dir.c_str(), 0755);
    }
    // Broadcast output_dir to all ranks
    {
        unsigned int len = output_dir.size();
        MPI_Bcast(&len, 1, MPI_UNSIGNED, 0, mpi_comm);
        output_dir.resize(len);
        MPI_Bcast(&output_dir[0], len, MPI_CHAR, 0, mpi_comm);
    }

    unsigned int total_dofs = ch.get_theta_dof_handler().n_dofs()
                            + ch.get_psi_dof_handler().n_dofs();
    if (params.enable_magnetic)
    {
        total_dofs += poisson.get_dof_handler().n_dofs();
        if (!params.use_algebraic_magnetization)
            total_dofs += 2 * mag.get_dof_handler().n_dofs();
    }
    if (params.enable_ns)
        total_dofs += ns.get_ux_dof_handler().n_dofs()
                    + ns.get_uy_dof_handler().n_dofs()
                    + ns.get_p_dof_handler().n_dofs();

    CSVLoggerFamily logger(output_dir, mpi_comm, params,
                           triangulation.n_global_active_cells(), total_dofs);

    // Write initial VTK
    const bool full_vtk = params.enable_magnetic && params.enable_ns;
    const bool has_mag_vectors = full_vtk && !params.use_algebraic_magnetization;
    const bool has_algebraic_M = full_vtk && params.use_algebraic_magnetization;
    if (has_mag_vectors)
        write_combined_vtu(output_dir, 0, 0.0,
                           ch, poisson, mag, ns, triangulation, mpi_comm);
    else if (has_algebraic_M)
        write_algebraic_M_vtu(output_dir, 0, 0.0,
                              ch, poisson, ns, params, triangulation, mpi_comm);
    else
        write_ch_only_vtu(output_dir, 0, 0.0, ch, triangulation, mpi_comm);

    // ----------------------------------------------------------------
    // 4b. When NS is disabled, create zero velocity vectors using CH's
    //     θ DoFHandler (CG Q2, same FE space as velocity)
    // ----------------------------------------------------------------
    TrilinosWrappers::MPI::Vector zero_ux_relevant, zero_uy_relevant;
    if (!params.enable_ns)
    {
        IndexSet vel_owned = ch.get_theta_dof_handler().locally_owned_dofs();
        IndexSet vel_relevant = DoFTools::extract_locally_relevant_dofs(
            ch.get_theta_dof_handler());
        zero_ux_relevant.reinit(vel_owned, vel_relevant, mpi_comm);
        zero_uy_relevant.reinit(vel_owned, vel_relevant, mpi_comm);
        zero_ux_relevant = 0;
        zero_uy_relevant = 0;
    }

    pcout << "  Output directory: " << output_dir << "\n\n";

    // ----------------------------------------------------------------
    // 5. SAV initialization (if enabled)
    // ----------------------------------------------------------------
    double sav_r = 1.0;           // SAV variable r(t)
    double sav_E1_old = 0.0;      // E1(theta^n) cached for SAV update
    double sav_S1 = params.sav.S1;
    double sav_S2 = params.sav.S2;
    const bool adaptive_S2 = (params.sav.S2 <= 0.0) && params.enable_magnetic
                           && params.enable_ns && params.use_sav;

    if (params.use_sav)
    {
        // Auto-compute S1 if not set: S = lambda/(4*epsilon)
        // Zhang p.B182: "we choose L = 1/(2*epsilon), thus S = lambda/(4*epsilon)"
        if (sav_S1 <= 0.0)
            sav_S1 = params.physics.lambda / (4.0 * params.physics.epsilon);

        // Compute initial bulk energy and SAV variable
        sav_E1_old = ch.compute_bulk_energy(ch.get_theta_relevant());
        sav_r = std::sqrt(sav_E1_old + params.sav.C0);

        pcout << "  SAV initialization:\n"
              << "    S1 = " << sav_S1 << " (CH stabilization)\n"
              << "    S2 = " << sav_S2 << " (NS stabilization"
              << (adaptive_S2 ? ", adaptive" : "") << ")\n"
              << "    C0 = " << params.sav.C0 << "\n"
              << "    E1(theta^0) = " << sav_E1_old << "\n"
              << "    r^0 = " << sav_r << "\n\n";
    }

    // ----------------------------------------------------------------
    // 6. Time-stepping loop
    // ----------------------------------------------------------------
    double current_time = 0.0;
    auto wall_start_total = std::chrono::high_resolution_clock::now();

    for (unsigned int step = 1; step <= max_steps; ++step)
    {
        current_time += dt;
        if (current_time > t_final + 1e-14 * dt)
            break;

        auto wall_start = std::chrono::high_resolution_clock::now();

        // ============================================================
        // Step 1: Cahn-Hilliard (θ^n, ψ^n) using U^{n-1}
        //
        // SAV mode: uses S₁ stabilization and SAV nonlinear scaling
        // Standard mode: original Nochetto scheme
        //
        // IMPORTANT: Save θ^{n-1} BEFORE CH solve — needed for NS lagging.
        // Per Nochetto Eq 65d: all θ-dependent coefficients in NS (ν, ρ, χ,
        // capillary θ factor) must use θ^{k-1} for energy stability.
        // ============================================================
        TrilinosWrappers::MPI::Vector theta_old(ch.get_theta_relevant());

        {
            const auto& vel_ux = params.enable_ns
                ? ns.get_ux_old_relevant() : zero_ux_relevant;
            const auto& vel_uy = params.enable_ns
                ? ns.get_uy_old_relevant() : zero_uy_relevant;
            const auto& vel_dof = params.enable_ns
                ? ns.get_ux_dof_handler() : ch.get_theta_dof_handler();

            std::vector<const TrilinosWrappers::MPI::Vector*> vel_comps = {
                &vel_ux, &vel_uy
            };

            if (params.use_sav)
            {

                // SAV factor: r^n / sqrt(E1(theta^n) + C0)
                const double denom = std::sqrt(sav_E1_old + params.sav.C0);
                const double sav_factor = (denom > 1e-15) ? sav_r / denom : 1.0;

                ch.assemble_sav(ch.get_theta_relevant(), vel_comps, vel_dof,
                                dt, current_time, sav_S1, sav_factor);
                ch.solve();
                ch.update_ghosts();

                // SAV update: r^{n+1} = r^n + inner_product / (2*sqrt(E1_old + C0))
                if (denom > 1e-15)
                {
                    const double inner_prod = ch.compute_sav_inner_product(
                        ch.get_theta_relevant(), theta_old);
                    sav_r += inner_prod / (2.0 * denom);
                }

                // Update E1 for next timestep
                sav_E1_old = ch.compute_bulk_energy(ch.get_theta_relevant());
            }
            else
            {
                ch.assemble(ch.get_theta_relevant(), vel_comps, vel_dof,
                            dt, current_time);
                ch.solve();
                ch.update_ghosts();
            }
        }

        // ============================================================
        // Step 2: Poisson (φ^n) using θ^n (algebraic M) or M^{n-1} (PDE M)
        //   Skipped if magnetic field is disabled (pure CH test)
        // ============================================================
        if (params.enable_magnetic)
        {
            if (params.use_algebraic_magnetization)
            {
                // Nonlinear Poisson: ((1+chi(theta)) grad phi, grad X) = ((1-chi(theta)) h_a, grad X)
                // Picard iteration for convergence
                // Nonlinear Poisson: ((1+chi(theta)) grad phi, grad X) = ((1-chi(theta)) h_a, grad X)
                // For fixed theta, this is a linear system in phi — one solve is exact.
                // (Picard iteration would be needed if phi appeared in theta, which it doesn't here)
                poisson.assemble_nonlinear(
                    ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                    current_time);
                poisson.solve();
                poisson.update_ghosts();
            }
            else
            {
                // PDE-M path: Under-relaxed Picard sub-iteration for
                // Poisson ↔ Magnetization coupling.
                //
                // Without sub-iteration, the explicit coupling M^{n-1}→φ^n→H^n→M^n
                // is unstable when χ₀ is large and τ_M << dt.
                // For χ₀=2, τ_M=1e-4, dt=1e-3: amplification = 1.73 > 1.
                //
                // Under-relaxed Picard with ω < 1/(1+χ₀) makes the map contractive.
                // The time derivative always uses M^{n-1} (saved at timestep start),
                // not M^k — this is the correct semi-implicit scheme.
                //
                // Algorithm per timestep:
                //   save M^{n-1}
                //   for k = 0, 1, ..., max_picard-1:
                //     1. Poisson:  (∇φ^k, ∇X) = (h_a - M^k, ∇X)
                //     2. Mag solve: (1/dt + 1/τ_M) M_raw + B_h^m(U, M_raw, Z)
                //                   = (1/τ_M) χ H^k · Z + (1/dt) M^{n-1} · Z + β[...]
                //     3. Under-relax: M^{k+1} = ω·M_raw + (1-ω)·M^k
                //     4. Convergence: ||M^{k+1} - M^k|| / ||M^{k+1}|| < tol

                const auto& vel_ux = params.enable_ns
                    ? ns.get_ux_old_relevant() : zero_ux_relevant;
                const auto& vel_uy = params.enable_ns
                    ? ns.get_uy_old_relevant() : zero_uy_relevant;
                const auto& vel_dof = params.enable_ns
                    ? ns.get_ux_dof_handler() : ch.get_theta_dof_handler();

                const unsigned int max_picard = params.picard_iterations;
                const double picard_tol   = params.picard_tolerance;
                const double picard_omega = params.picard_relaxation;

                // Save M^{n-1} for the time derivative term (1/dt)*M^{n-1}
                mag.save_old_solution();

                // ---- Iteration k=0: full assembly (matrix + RHS) ----
                poisson.assemble_rhs(mag.get_Mx_relevant(),
                                     mag.get_My_relevant(),
                                     mag.get_dof_handler(),
                                     current_time);
                poisson.solve();
                poisson.update_ghosts();

                // Full assembly: matrix depends on U^{n-1} (set once per timestep).
                // RHS uses M^{n-1} for time derivative and β-term.
                mag.assemble(
                    mag.get_Mx_old_relevant(),   // M^{n-1}: time derivative + β
                    mag.get_My_old_relevant(),
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    ch.get_theta_relevant(),         ch.get_theta_dof_handler(),
                    vel_ux, vel_uy, vel_dof,
                    dt, current_time);
                mag.solve();
                mag.apply_under_relaxation(picard_omega);
                mag.update_ghosts();

                // ---- Iterations k=1, ..., max_picard-1 ----
                unsigned int picard_iters = 1;
                for (unsigned int k = 1; k < max_picard; ++k)
                {
                    // M^k norms for convergence check (owned vectors)
                    const double Mx_k_norm = mag.get_Mx_solution().l2_norm();
                    const double My_k_norm = mag.get_My_solution().l2_norm();
                    const double M_k_norm = std::sqrt(Mx_k_norm * Mx_k_norm
                                                    + My_k_norm * My_k_norm);

                    // Re-solve Poisson with M^k
                    poisson.assemble_rhs(mag.get_Mx_relevant(),
                                         mag.get_My_relevant(),
                                         mag.get_dof_handler(),
                                         current_time);
                    poisson.solve();
                    poisson.update_ghosts();

                    // RHS only (matrix frozen): time derivative uses M^{n-1},
                    // H^k = ∇φ^k + h_a from updated Poisson.
                    mag.assemble_rhs_only(
                        poisson.get_solution_relevant(), poisson.get_dof_handler(),
                        ch.get_theta_relevant(),         ch.get_theta_dof_handler(),
                        mag.get_Mx_old_relevant(),       // M^{n-1}: time derivative + β
                        mag.get_My_old_relevant(),
                        dt, current_time);
                    mag.solve();
                    mag.apply_under_relaxation(picard_omega);
                    mag.update_ghosts();

                    picard_iters = k + 1;

                    // Convergence: relative change in ||M||
                    const double Mx_new_norm = mag.get_Mx_solution().l2_norm();
                    const double My_new_norm = mag.get_My_solution().l2_norm();
                    const double M_new_norm = std::sqrt(Mx_new_norm * Mx_new_norm
                                                      + My_new_norm * My_new_norm);
                    const double rel_change = (M_new_norm > 1e-15)
                        ? std::abs(M_new_norm - M_k_norm) / M_new_norm
                        : 0.0;

                    if (step % 100 == 0 || step <= 5)
                        pcout << "    Picard k=" << k
                              << ": rel_change=" << rel_change << "\n";

                    if (rel_change < picard_tol)
                        break;
                }

                if (step % 100 == 0 || step <= 5)
                    pcout << "    Picard: " << picard_iters << " iters"
                          << " (ω=" << picard_omega << ")\n";
            }
        }

        // ============================================================
        // Step 3: Magnetization (M^n)
        //   Algebraic M: SKIP (M already computed inside Picard loop above)
        //   PDE M: already solved inside Picard loop above
        // ============================================================
        // (Nothing to do here — magnetization was solved in the Picard
        //  sub-iteration above when enable_magnetic && !algebraic_M)

        // ============================================================
        // Step 4: Navier-Stokes (U^n, p^n)
        //   Algebraic M + SAV: uses assemble_coupled_algebraic_M with S₂
        //   Standard: original assemble_coupled
        //   Skipped if NS is disabled
        //
        // Adaptive S₂: When S₂ is not user-specified (= 0), we compute it
        // each timestep from the current field strength via Zhang's bound:
        //   S₂ ≥ μ₀² C_M² / (4 ν_min)
        // where C_M = χ₀ |H_max| bounds the Kelvin force coupling.
        // A safety factor of 1.5 is applied for robustness.
        // ============================================================
        if (adaptive_S2 && params.enable_magnetic)
        {
            // Quick Poisson diagnostics to get current |H|_max
            auto poi_diag_quick = poisson.compute_diagnostics(
                ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                current_time);
            const double H_max_now = poi_diag_quick.H_max;
            const double mu0  = params.physics.mu_0;
            const double chi0 = params.physics.chi_0;
            const double nu_min = std::min(params.physics.nu_water,
                                           params.physics.nu_ferro);
            // C_M = chi_0 * H_max (bound on |M·∇H| coupling)
            const double C_M = chi0 * H_max_now;
            // Zhang's bound: S2 >= mu0^2 * C_M^2 / (4 * nu_min)
            // Safety factor 1.5 for robustness of the decoupled splitting
            sav_S2 = 1.5 * mu0 * mu0 * C_M * C_M / (4.0 * nu_min);
        }

        if (params.enable_ns)
        {
            if (!params.enable_magnetic)
            {
                // NS without magnetic field: constant viscosity Stokes/NS
                // Capillary force (λψ∇θ) still present via body_force if needed,
                // but assemble_stokes uses constant ν — fine when ν_f == ν_w.
                const double nu = params.physics.nu_water;
                ns.assemble_stokes(dt, nu,
                    /*include_time_derivative=*/true,
                    /*include_convection=*/true);
            }
            else if (params.use_algebraic_magnetization)
            {
                // NS uses θ^{n-1} (lagged) for ν, ρ, χ, capillary θ
                // per Nochetto Eq 65d. ψ^n is current (from CH solve).
                ns.assemble_coupled_algebraic_M(
                    dt,
                    theta_old,                       ch.get_theta_dof_handler(),
                    ch.get_psi_relevant(),           ch.get_psi_dof_handler(),
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    current_time,
                    sav_S2,
                    /*include_convection=*/true);
            }
            else
            {
                // NS uses θ^{n-1} (lagged) — same as algebraic M path
                // S2 stabilization per Zhang Theorem 4.1 for energy stability
                ns.assemble_coupled(
                    dt,
                    theta_old,                       ch.get_theta_dof_handler(),
                    ch.get_psi_relevant(),           ch.get_psi_dof_handler(),
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    mag.get_Mx_relevant(),
                    mag.get_My_relevant(),
                    mag.get_dof_handler(),
                    current_time,
                    sav_S2,
                    /*include_convection=*/true);
            }
            ns.solve();
            ns.advance_time();
            ns.update_ghosts();
        }

        // ============================================================
        // Diagnostics
        // ============================================================
        auto wall_end = std::chrono::high_resolution_clock::now();
        double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

        auto ch_diag = ch.compute_diagnostics();

        PoissonSubsystem<dim>::Diagnostics poi_diag;
        MagnetizationSubsystem<dim>::Diagnostics mag_diag;
        NSSubsystem<dim>::Diagnostics ns_diag;

        if (params.enable_magnetic)
        {
            poi_diag = poisson.compute_diagnostics(
                ch.get_theta_relevant(), ch.get_theta_dof_handler(), current_time);
            if (!params.use_algebraic_magnetization)
            {
                mag_diag = mag.compute_diagnostics(
                    poisson.get_solution_relevant(), poisson.get_dof_handler(),
                    ch.get_theta_relevant(), ch.get_theta_dof_handler(),
                    current_time);
            }
            // When using algebraic M, mag_diag stays default (zeros).
            // The key physics are in poi_diag (H_max, E_mag, etc.)
        }
        if (params.enable_ns)
            ns_diag = ns.compute_diagnostics(dt);

        double E_total = compute_discrete_energy(
            ch_diag, ns_diag, mag_diag, poi_diag, params);

        logger.log(step, current_time, dt,
                   ch_diag, poi_diag, mag_diag, ns_diag,
                   E_total, wall_s);

        // Console output
        if (step % 10 == 0 || step == 1)
        {
            pcout << "  step " << std::setw(5) << step
                  << "  t=" << std::scientific << std::setprecision(3)
                  << current_time
                  << "  θ=[" << std::setprecision(2) << ch_diag.theta_min
                  << "," << ch_diag.theta_max << "]"
                  << "  E_CH=" << std::setprecision(4) << ch_diag.E_total;
            if (params.enable_ns)
                pcout << "  |U|=" << std::setprecision(2) << ns_diag.U_max;
            if (params.enable_magnetic)
            {
                if (!params.use_algebraic_magnetization)
                    pcout << "  |M|=" << std::setprecision(2) << mag_diag.M_magnitude_max;
                pcout << "  |H|=" << std::setprecision(2) << poi_diag.H_max;
            }
            if (params.use_sav)
            {
                pcout << "  r=" << std::setprecision(4) << sav_r;
                if (adaptive_S2)
                    pcout << "  S2=" << std::setprecision(1) << sav_S2;
            }
            pcout << "  wall=" << std::fixed << std::setprecision(1) << wall_s << "s"
                  << "\n";
        }

        // VTK output
        if (step % vtk_interval == 0)
        {
            if (has_mag_vectors)
                write_combined_vtu(output_dir, step, current_time,
                                   ch, poisson, mag, ns, triangulation, mpi_comm);
            else if (has_algebraic_M)
                write_algebraic_M_vtu(output_dir, step, current_time,
                                      ch, poisson, ns, params, triangulation, mpi_comm);
            else
                write_ch_only_vtu(output_dir, step, current_time,
                                  ch, triangulation, mpi_comm);
        }
    }

    // ----------------------------------------------------------------
    // 6. Final output
    // ----------------------------------------------------------------
    auto wall_end_total = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(
        wall_end_total - wall_start_total).count();

    // Write final VTK
    unsigned int final_step = static_cast<unsigned int>(
        std::min(current_time / dt, static_cast<double>(max_steps)));
    if (has_mag_vectors)
        write_combined_vtu(output_dir, final_step, current_time,
                           ch, poisson, mag, ns, triangulation, mpi_comm);
    else if (has_algebraic_M)
        write_algebraic_M_vtu(output_dir, final_step, current_time,
                              ch, poisson, ns, params, triangulation, mpi_comm);
    else
        write_ch_only_vtu(output_dir, final_step, current_time,
                          ch, triangulation, mpi_comm);

    pcout << "\n"
          << "============================================================\n"
          << "  Simulation complete.\n"
          << "  Final time:    " << current_time << "\n"
          << "  Total steps:   " << final_step << "\n"
          << "  Wall time:     " << std::fixed << std::setprecision(1)
          << total_s << " s (" << total_s / 60.0 << " min)\n"
          << "  Avg step time: " << std::setprecision(2)
          << total_s / final_step << " s/step\n"
          << "  Output:        " << output_dir << "\n"
          << "============================================================\n";

    return 0;
}
