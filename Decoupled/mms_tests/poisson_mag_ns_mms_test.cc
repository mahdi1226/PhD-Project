// ============================================================================
// mms_tests/poisson_mag_ns_mms_test.cc — Poisson + Mag + NS Coupled MMS
//
// Intermediate test: validates Kelvin force coupling (μ₀ > 0) between
// NS and Poisson/Mag subsystems. No CH (θ = +1 constant, ψ = 0).
//
// Time loop mirrors production driver:
//   Step 1: NS (using φ^{n-1}, M^{n-1})  → U^n, p^n
//   Step 2: Poisson/Mag Picard (using U^n) → φ^n, M^n
//
// Usage:
//   mpirun -np 2 ./test_poisson_mag_ns_mms [--refs 2 3 4 5] [--steps 10]
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms_tests/poisson_mag_ns_mms.h"

#include "navier_stokes/navier_stokes.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"

#include "navier_stokes/tests/navier_stokes_mms.h"
#include "poisson/tests/poisson_mms.h"
#include "magnetization/tests/magnetization_mms.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

constexpr int dim = 2;


// ============================================================================
// Single refinement level
// ============================================================================
static PoissonMagNSResult run_single_level(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm,
    bool use_projected_velocity = false,
    bool mag_only = false)
{
    using namespace dealii;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    PoissonMagNSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------------
    // Domain and time
    // ----------------------------------------------------------------
    const double L_y     = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end   = 0.2;
    const double dt      = (t_end - t_start) / n_time_steps;

    pcout << "\n  [ref=" << refinement << "] dt=" << dt
          << "  n_steps=" << n_time_steps
          << "  mu_0=" << params.physics.mu_0 << "\n";

    // ----------------------------------------------------------------
    // 1. Shared triangulation
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
    result.h = 1.0 / (params.domain.initial_cells_x * std::pow(2.0, refinement));

    // ----------------------------------------------------------------
    // 2. Setup subsystems
    //
    // When use_projected_velocity = true, we skip NS entirely and
    // interpolate exact U* onto CG-Q2 vectors. This isolates whether
    // DG transport convergence failure is due to NS coupling vs
    // something fundamental.
    // ----------------------------------------------------------------
    std::unique_ptr<NSSubsystem<dim>> ns_ptr;
    if (!use_projected_velocity)
    {
        ns_ptr = std::make_unique<NSSubsystem<dim>>(params, mpi_comm, triangulation);
        ns_ptr->setup();
    }

    PoissonSubsystem<dim>       poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();

    // CG-Q2 DoFHandler for velocity (used in both modes)
    FE_Q<dim> fe_vel(params.fe.degree_velocity);
    DoFHandler<dim> vel_dof(triangulation);
    vel_dof.distribute_dofs(fe_vel);
    IndexSet vel_owned    = vel_dof.locally_owned_dofs();
    IndexSet vel_relevant = DoFTools::extract_locally_relevant_dofs(vel_dof);

    // Velocity vectors for projected mode
    TrilinosWrappers::MPI::Vector ux_proj_owned(vel_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_proj_owned(vel_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector ux_proj(vel_owned, vel_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_proj(vel_owned, vel_relevant, mpi_comm);

    const unsigned int total_dofs =
        (ns_ptr ? ns_ptr->get_ux_dof_handler().n_dofs()
                  + ns_ptr->get_uy_dof_handler().n_dofs()
                  + ns_ptr->get_p_dof_handler().n_dofs()
                : 2 * vel_dof.n_dofs())
        + poisson.get_dof_handler().n_dofs()
        + 2 * mag.get_dof_handler().n_dofs();
    result.n_dofs = total_dofs;

    pcout << "  Total DoFs: " << total_dofs
          << (use_projected_velocity ? " (projected U)" : " (NS solve)") << "\n";

    // ----------------------------------------------------------------
    // 3. Dummy θ = +1, ψ = 0 fields (no CH subsystem)
    //
    // NS assembly reads theta for ν(θ), ρ(θ) and psi for capillary.
    // With θ = +1 (ferrofluid) and ψ = 0, capillary vanishes.
    // ----------------------------------------------------------------
    FE_Q<dim> fe_dummy(1);

    DoFHandler<dim> theta_dof(triangulation);
    theta_dof.distribute_dofs(fe_dummy);
    IndexSet theta_owned    = theta_dof.locally_owned_dofs();
    IndexSet theta_relevant = DoFTools::extract_locally_relevant_dofs(theta_dof);

    TrilinosWrappers::MPI::Vector theta_owned_vec(theta_owned, mpi_comm);
    theta_owned_vec = 1.0;  // ferrofluid everywhere
    TrilinosWrappers::MPI::Vector theta_vec(theta_owned, theta_relevant, mpi_comm);
    theta_vec = theta_owned_vec;

    // ψ = 0 (use same DoFHandler pattern)
    DoFHandler<dim> psi_dof(triangulation);
    psi_dof.distribute_dofs(fe_dummy);
    IndexSet psi_owned    = psi_dof.locally_owned_dofs();
    IndexSet psi_relevant = DoFTools::extract_locally_relevant_dofs(psi_dof);

    TrilinosWrappers::MPI::Vector psi_vec(psi_owned, psi_relevant, mpi_comm);
    psi_vec = 0;  // no capillary force

    // ----------------------------------------------------------------
    // 4. MMS source injection
    // ----------------------------------------------------------------
    const double nu = params.physics.nu_ferro;  // constant viscosity
    const double mu0 = params.physics.mu_0;

    // NS source: with Kelvin term 1 + term 2 curl + b_stab
    PoissonMagNSSourceU<dim> ns_source(dt, L_y, nu, mu0);
    if (ns_ptr)
    {
        ns_ptr->set_mms_source(
            [&](const Point<dim>& pt, double t) -> Tensor<1, dim> {
                return ns_source(pt, t);
            });
    }

    // Poisson source: −Δφ* − ∇·M* (same as Poisson-Mag test)
    poisson.set_mms_source(
        [L_y](const Point<dim>& pt, double time) -> double {
            return compute_poisson_mms_source_coupled<dim>(pt, time, L_y);
        });

    // Magnetization source: with transport by U* + spin-vorticity correction
    mag.set_mms_source(
        [L_y](const Point<dim>& pt,
              double t_new, double t_old,
              double tau_M, double chi_val,
              const Tensor<1, dim>& H,
              const Tensor<1, dim>& U,
              double div_U) -> Tensor<1, dim>
        {
            auto f = compute_mag_mms_source_with_transport<dim>(
                pt, t_new, t_old, tau_M, chi_val, H, U, div_U, L_y);

            // Spin-vorticity correction (D3 fix):
            // Assembler adds +½·ω_z·(-My_old, Mx_old) to RHS.
            // MMS source must subtract this exact contribution.
            // ω_z = curl(U*) at t_new, M* at t_old.
            const auto gux = NSMMS::ux_grad<dim>(pt, t_new, L_y);
            const auto guy = NSMMS::uy_grad<dim>(pt, t_new, L_y);
            const double omega_z = guy[0] - gux[1];
            const auto M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);
            f[0] -= 0.5 * omega_z * (-M_old[1]);
            f[1] -= 0.5 * omega_z * ( M_old[0]);

            return f;
        });

    // ----------------------------------------------------------------
    // 5. Initial conditions at t = t_start
    // ----------------------------------------------------------------
    // NS velocity (or projected velocity)
    if (ns_ptr)
    {
        NSMMSInitialUx<dim> ux_ic(t_start, L_y);
        NSMMSInitialUy<dim> uy_ic(t_start, L_y);
        ns_ptr->initialize_velocity(ux_ic, uy_ic);
        ns_ptr->update_ghosts();
    }
    else
    {
        // Project exact U*(t_start) onto CG-Q2
        NSMMSInitialUx<dim> ux_ic(t_start, L_y);
        NSMMSInitialUy<dim> uy_ic(t_start, L_y);
        VectorTools::interpolate(vel_dof, ux_ic, ux_proj_owned);
        VectorTools::interpolate(vel_dof, uy_ic, uy_proj_owned);
        ux_proj = ux_proj_owned;
        uy_proj = uy_proj_owned;
    }

    // Magnetization
    {
        MagExactMx<dim> exact_Mx(t_start, L_y);
        MagExactMy<dim> exact_My(t_start, L_y);
        mag.project_initial_condition(exact_Mx, exact_My);
    }
    mag.update_ghosts();

    // Poisson: initial solve consistent with M*(t_start)
    poisson.assemble_rhs(mag.get_Mx_relevant(), mag.get_My_relevant(),
                         mag.get_dof_handler(), t_start);
    poisson.solve();
    poisson.update_ghosts();

    // ----------------------------------------------------------------
    // 6. Workspace vectors for Picard iteration
    // ----------------------------------------------------------------
    IndexSet M_owned    = mag.get_dof_handler().locally_owned_dofs();
    IndexSet M_relevant = DoFTools::extract_locally_relevant_dofs(
        mag.get_dof_handler());

    TrilinosWrappers::MPI::Vector Mx_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_old(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed(M_owned, M_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_relaxed_owned(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector Mx_prev(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_prev(M_owned, mpi_comm);

    Mx_relaxed = mag.get_Mx_relevant();
    My_relaxed = mag.get_My_relevant();

    // ----------------------------------------------------------------
    // 6b. DIAGNOSTIC: measure initial projection error ||e^0||_L2
    // ----------------------------------------------------------------
    {
        MagMMSError e0_err = compute_mag_mms_errors_parallel<dim>(
            mag.get_dof_handler(),
            mag.get_Mx_relevant(),
            mag.get_My_relevant(),
            t_start, L_y, mpi_comm);
        pcout << "  DIAG-INIT: ||e^0||_M_L2=" << std::scientific
              << std::setprecision(4) << e0_err.M_L2
              << "  Mx_L2=" << e0_err.Mx_L2
              << "  My_L2=" << e0_err.My_L2
              << "  M_Linf=" << e0_err.M_Linf << "\n";
    }

    // ----------------------------------------------------------------
    // 7. Time stepping
    //
    // Step 1: NS (uses φ^{n-1}, M^{n-1} → computes U^n)
    // Step 2: Poisson/Mag Picard (uses U^n → computes φ^n, M^n)
    // ----------------------------------------------------------------
    const unsigned int max_picard = 50;
    const double picard_tol       = 1e-10;
    const double picard_omega     = 0.35;

    double current_time = t_start;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // ==== Step 1: Navier-Stokes (or project exact U*) ====
        if (ns_ptr)
        {
            ns_ptr->assemble_coupled(
                dt,
                theta_vec,                             // θ = +1
                theta_vec,                             // θ_old = +1 (same)
                theta_dof,
                psi_vec,                               // ψ = 0
                psi_dof,
                poisson.get_solution_relevant(),       // φ^{n-1}
                poisson.get_dof_handler(),
                mag.get_Mx_relevant(),                 // M^{n-1}
                mag.get_My_relevant(),
                mag.get_dof_handler(),
                current_time,                          // for MMS source
                true);                                 // include convection

            ns_ptr->solve();

            // Pressure correction (Zhang Steps 3-4)
            ns_ptr->assemble_pressure_poisson(dt);
            ns_ptr->solve_pressure();
            ns_ptr->velocity_correction(dt);

            ns_ptr->advance_time();
            ns_ptr->update_ghosts();
        }
        else
        {
            // Project exact U* at current_time onto CG-Q2
            NSMMSInitialUx<dim> ux_exact(current_time, L_y);
            NSMMSInitialUy<dim> uy_exact(current_time, L_y);
            VectorTools::interpolate(vel_dof, ux_exact, ux_proj_owned);
            VectorTools::interpolate(vel_dof, uy_exact, uy_proj_owned);
            // DIAGNOSTIC: set --zero-u to force U=0 in both matrix and MMS
            if (mag_only && std::getenv("ZERO_U"))
            {
                ux_proj_owned = 0;
                uy_proj_owned = 0;
            }
            ux_proj = ux_proj_owned;
            uy_proj = uy_proj_owned;
        }

        // ==== Step 2: Poisson/Mag Picard (or mag-only) ====
        // Snapshot M^{n-1}
        Mx_old = mag.get_Mx_relevant();
        My_old = mag.get_My_relevant();
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        // Velocity references (constant during Picard)
        const auto& ux_for_mag = ns_ptr
            ? ns_ptr->get_ux_old_relevant() : ux_proj;
        const auto& uy_for_mag = ns_ptr
            ? ns_ptr->get_uy_old_relevant() : uy_proj;
        const auto& u_dof_for_mag = ns_ptr
            ? ns_ptr->get_ux_dof_handler() : vel_dof;

        if (mag_only)
        {
            // ============================================================
            // MAG-ONLY MODE: no Picard, no Poisson update.
            // Single mag assemble+solve with φ=0 and projected U.
            // Isolates DG transport convergence.
            // ============================================================
            mag.assemble(
                Mx_old, My_old,
                poisson.get_solution_relevant(), poisson.get_dof_handler(),
                theta_vec, theta_dof,
                ux_for_mag, uy_for_mag,
                u_dof_for_mag,
                dt, current_time);

            // DIAGNOSTIC: MMS residual check (first step only)
            if (step == 0)
            {
                // L2-project M*(t_new) onto DG Q1
                MagExactMx<dim> exact_Mx_tnew(current_time, L_y);
                MagExactMy<dim> exact_My_tnew(current_time, L_y);

                // Cell-local L2 projection
                TrilinosWrappers::MPI::Vector Mstar_x(M_owned, mpi_comm);
                TrilinosWrappers::MPI::Vector Mstar_y(M_owned, mpi_comm);
                Mstar_x = 0; Mstar_y = 0;
                {
                    const auto& fe_M = mag.get_dof_handler().get_fe();
                    QGauss<dim> q_proj(fe_M.degree + 2);
                    FEValues<dim> fv(fe_M, q_proj,
                        update_values | update_quadrature_points | update_JxW_values);
                    const unsigned int nq = q_proj.size();
                    const unsigned int dpc = fe_M.n_dofs_per_cell();
                    std::vector<types::global_dof_index> dofs(dpc);
                    FullMatrix<double> cell_mass(dpc, dpc);
                    Vector<double> cell_rhs_x(dpc), cell_rhs_y(dpc);
                    Vector<double> cell_sol_x(dpc), cell_sol_y(dpc);

                    for (const auto& cell : mag.get_dof_handler().active_cell_iterators())
                    {
                        if (!cell->is_locally_owned()) continue;
                        fv.reinit(cell);
                        cell->get_dof_indices(dofs);
                        cell_mass = 0; cell_rhs_x = 0; cell_rhs_y = 0;
                        for (unsigned int q = 0; q < nq; ++q)
                        {
                            const auto& xq = fv.quadrature_point(q);
                            double JxW = fv.JxW(q);
                            double mx = exact_Mx_tnew.value(xq);
                            double my = exact_My_tnew.value(xq);
                            for (unsigned int i = 0; i < dpc; ++i)
                            {
                                double Zi = fv.shape_value(i, q);
                                cell_rhs_x(i) += mx * Zi * JxW;
                                cell_rhs_y(i) += my * Zi * JxW;
                                for (unsigned int j = 0; j < dpc; ++j)
                                    cell_mass(i, j) += Zi * fv.shape_value(j, q) * JxW;
                            }
                        }
                        cell_mass.gauss_jordan();
                        cell_mass.vmult(cell_sol_x, cell_rhs_x);
                        cell_mass.vmult(cell_sol_y, cell_rhs_y);
                        for (unsigned int i = 0; i < dpc; ++i)
                        {
                            Mstar_x[dofs[i]] = cell_sol_x(i);
                            Mstar_y[dofs[i]] = cell_sol_y(i);
                        }
                    }
                    Mstar_x.compress(VectorOperation::insert);
                    Mstar_y.compress(VectorOperation::insert);
                }

                // Compute residual: r = A * π_h(M*) - b
                const auto& A = mag.get_system_matrix();
                const auto& bx = mag.get_Mx_rhs();
                const auto& by = mag.get_My_rhs();
                TrilinosWrappers::MPI::Vector res_x(M_owned, mpi_comm);
                TrilinosWrappers::MPI::Vector res_y(M_owned, mpi_comm);
                A.vmult(res_x, Mstar_x);
                res_x -= bx;
                A.vmult(res_y, Mstar_y);
                res_y -= by;

                // Also compute: A*M_h^0 - b (defect of the old solution)
                TrilinosWrappers::MPI::Vector Mold_owned_x(M_owned, mpi_comm);
                TrilinosWrappers::MPI::Vector Mold_owned_y(M_owned, mpi_comm);
                // Copy relevant→owned for vmult
                for (const auto idx : M_owned)
                {
                    Mold_owned_x[idx] = Mx_old[idx];
                    Mold_owned_y[idx] = My_old[idx];
                }
                TrilinosWrappers::MPI::Vector def_x(M_owned, mpi_comm);
                TrilinosWrappers::MPI::Vector def_y(M_owned, mpi_comm);
                A.vmult(def_x, Mold_owned_x);
                def_x -= bx;
                A.vmult(def_y, Mold_owned_y);
                def_y -= by;

                pcout << "  DIAG-RESID: ||A*π(M*)-b||  Mx=" << std::scientific
                      << std::setprecision(4) << res_x.l2_norm()
                      << "  My=" << res_y.l2_norm() << "\n";
                pcout << "  DIAG-DEFECT: ||A*M_old-b||  Mx=" << def_x.l2_norm()
                      << "  My=" << def_y.l2_norm() << "\n";
                pcout << "  DIAG-NORMS: ||b_x||=" << bx.l2_norm()
                      << "  ||b_y||=" << by.l2_norm()
                      << "  ||π(M*x)||=" << Mstar_x.l2_norm()
                      << "  ||M_old_x||=" << Mold_owned_x.l2_norm() << "\n";
            }

            mag.solve();
            mag.update_ghosts();

            // DIAGNOSTIC: error after this step
            if (step < 3)
            {
                MagMMSError step_err = compute_mag_mms_errors_parallel<dim>(
                    mag.get_dof_handler(),
                    mag.get_Mx_relevant(),
                    mag.get_My_relevant(),
                    current_time, L_y, mpi_comm);
                pcout << "    step " << step << ": M_L2="
                      << std::scientific << std::setprecision(4)
                      << step_err.M_L2
                      << "  M_Linf=" << step_err.M_Linf << "\n";
            }

            result.picard_iters = 0;
            pcout << "    step " << step << ": mag-only (no Picard)\n";
        }
        else
        {
        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Poisson: φ^k using relaxed M
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Magnetization: M^k using φ^k and U^n
            // After advance_time(), get_ux_old_relevant() = U^n
            //
            // NOTE: Always use full assemble() (not assemble_rhs_only)
            // because assemble_rhs_only passes U=0 to the MMS source,
            // which drops the transport term from the manufactured source.
            mag.assemble(
                Mx_old, My_old,
                poisson.get_solution_relevant(), poisson.get_dof_handler(),
                theta_vec, theta_dof,
                ux_for_mag, uy_for_mag,
                u_dof_for_mag,
                dt, current_time);
            mag.solve();
            mag.update_ghosts();

            // Under-relax
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

            Mx_relaxed_owned  = mag.get_Mx_solution();
            Mx_relaxed_owned *= picard_omega;
            Mx_relaxed_owned.add(1.0 - picard_omega, Mx_prev);

            My_relaxed_owned  = mag.get_My_solution();
            My_relaxed_owned *= picard_omega;
            My_relaxed_owned.add(1.0 - picard_omega, My_prev);

            Mx_relaxed = Mx_relaxed_owned;
            My_relaxed = My_relaxed_owned;

            // Convergence check
            TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_comm);
            TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_comm);
            Mx_diff  = Mx_relaxed_owned;
            Mx_diff -= Mx_prev;
            My_diff  = My_relaxed_owned;
            My_diff -= My_prev;

            const double change = Mx_diff.l2_norm() + My_diff.l2_norm();
            const double norm   = Mx_relaxed_owned.l2_norm()
                                + My_relaxed_owned.l2_norm() + 1e-14;
            const double residual = change / norm;

            if (residual < picard_tol)
            {
                result.picard_iters = k + 1;
                pcout << "    step " << step << ": Picard converged, iter="
                      << k + 1 << ", res=" << std::scientific
                      << std::setprecision(2) << residual << "\n";
                break;
            }

            if (k == max_picard - 1)
            {
                result.picard_iters = max_picard;
                pcout << "    step " << step
                      << ": WARNING Picard did not converge, res="
                      << std::scientific << std::setprecision(2)
                      << residual << "\n";
            }
        }
        } // end else (Picard mode)
    }

    // ----------------------------------------------------------------
    // 8. Compute errors at final time
    // ----------------------------------------------------------------
    if (ns_ptr) ns_ptr->update_ghosts();
    poisson.update_ghosts();
    mag.update_ghosts();

    // NS errors
    if (ns_ptr)
    {
        NSMMSErrors ns_err = compute_ns_mms_errors<dim>(
            *ns_ptr, current_time, L_y, mpi_comm);

        result.ux_L2 = ns_err.ux_L2;
        result.uy_L2 = ns_err.uy_L2;
        result.p_L2  = ns_err.p_L2;
    }
    else
    {
        result.ux_L2 = 0.0;
        result.uy_L2 = 0.0;
        result.p_L2  = 0.0;
    }

    // Poisson errors
    {
        PoissonMMSErrors phi_err = compute_poisson_mms_errors<dim>(
            poisson.get_dof_handler(),
            poisson.get_solution_relevant(),
            current_time, L_y, mpi_comm);

        result.phi_L2   = phi_err.L2;
        result.phi_H1   = phi_err.H1;
        result.phi_Linf = phi_err.Linf;
    }

    // Magnetization errors
    {
        MagMMSError mag_err = compute_mag_mms_errors_parallel<dim>(
            mag.get_dof_handler(),
            mag.get_Mx_relevant(),
            mag.get_My_relevant(),
            current_time, L_y, mpi_comm);

        result.M_L2   = mag_err.M_L2;
        result.M_Linf = mag_err.M_Linf;
        result.Mx_L2  = mag_err.Mx_L2;
        result.My_L2  = mag_err.My_L2;
    }

    // ----------------------------------------------------------------
    // 8b. Residual diagnostics (MMS consistency check)
    //
    // Compute AFTER errors so we can safely overwrite solution vectors.
    //   (a) Solver residual:      ||A * M_h - b||  (should be ~0 for MUMPS)
    //   (b) Consistency residual:  ||A * Π_h(M*) - b||  (MMS mismatch)
    // ----------------------------------------------------------------
    {
        const auto& A   = mag.get_system_matrix();
        const auto& b_x = mag.get_Mx_rhs();
        const auto& b_y = mag.get_My_rhs();

        // (a) Solver residual: should be ~machine epsilon for direct solver
        TrilinosWrappers::MPI::Vector r_sol_x(M_owned, mpi_comm);
        TrilinosWrappers::MPI::Vector r_sol_y(M_owned, mpi_comm);
        A.vmult(r_sol_x, mag.get_Mx_solution());
        r_sol_x -= b_x;
        A.vmult(r_sol_y, mag.get_My_solution());
        r_sol_y -= b_y;

        pcout << "  DIAG: solver residual  ||A*Mh-b||  Mx=" << std::scientific
              << std::setprecision(3) << r_sol_x.l2_norm()
              << "  My=" << r_sol_y.l2_norm() << "\n";

        // (b) Consistency with L2 PROJECTION: A * Π_h(M*) - b
        MagExactMx<dim> exact_Mx_now(current_time, L_y);
        MagExactMy<dim> exact_My_now(current_time, L_y);
        mag.project_initial_condition(exact_Mx_now, exact_My_now);

        TrilinosWrappers::MPI::Vector r_con_x(M_owned, mpi_comm);
        TrilinosWrappers::MPI::Vector r_con_y(M_owned, mpi_comm);
        A.vmult(r_con_x, mag.get_Mx_solution());
        r_con_x -= b_x;
        A.vmult(r_con_y, mag.get_My_solution());
        r_con_y -= b_y;

        pcout << "  DIAG: L2proj residual  ||A*Πm-b||  Mx=" << std::scientific
              << std::setprecision(3) << r_con_x.l2_norm()
              << "  My=" << r_con_y.l2_norm() << "\n";

        // (c) Consistency with INTERPOLATION: A * I_h(M*) - b
        //     DG-Q1 interpolation sets DoFs = M*(vertex). Since adjacent cells
        //     share vertex positions, the interpolation has [[I_h M*]] = 0,
        //     eliminating face flux residuals.
        TrilinosWrappers::MPI::Vector mx_interp(M_owned, mpi_comm);
        TrilinosWrappers::MPI::Vector my_interp(M_owned, mpi_comm);
        VectorTools::interpolate(mag.get_dof_handler(), exact_Mx_now, mx_interp);
        VectorTools::interpolate(mag.get_dof_handler(), exact_My_now, my_interp);

        TrilinosWrappers::MPI::Vector r_int_x(M_owned, mpi_comm);
        TrilinosWrappers::MPI::Vector r_int_y(M_owned, mpi_comm);
        A.vmult(r_int_x, mx_interp);
        r_int_x -= b_x;
        A.vmult(r_int_y, my_interp);
        r_int_y -= b_y;

        pcout << "  DIAG: interp residual  ||A*Im-b||  Mx=" << std::scientific
              << std::setprecision(3) << r_int_x.l2_norm()
              << "  My=" << r_int_y.l2_norm() << "\n";

        // (d) Difference: L2 proj vs interpolation
        TrilinosWrappers::MPI::Vector proj_diff_x(M_owned, mpi_comm);
        proj_diff_x  = mag.get_Mx_solution();
        proj_diff_x -= mx_interp;
        pcout << "  DIAG: ||Πm - Im||_Mx=" << proj_diff_x.l2_norm() << "\n";

        pcout << "  DIAG: ||b_x||=" << b_x.l2_norm()
              << "  ||b_y||=" << b_y.l2_norm() << "\n";

        // Linf norms of residuals
        pcout << "  DIAG: interp Linf  Mx=" << r_int_x.linfty_norm()
              << "  My=" << r_int_y.linfty_norm() << "\n";

        // ||A*I_h|| = ||r_int + b||
        {
            TrilinosWrappers::MPI::Vector A_Ih_x(M_owned, mpi_comm);
            A.vmult(A_Ih_x, mx_interp);
            pcout << "  DIAG: ||A*Ih_x||=" << A_Ih_x.l2_norm()
                  << "  ||b_x||=" << b_x.l2_norm()
                  << "  ||mx_interp||=" << mx_interp.l2_norm() << "\n";

            // Decompose: mass action vs transport
            // mass_action_i = mass_coeff * ∫_K I_h * Z_i dx (by quadrature)
            const double dt_val = current_time - t_start;  // dt for this run
            const double mass_coeff = 1.0/dt_val + 1.0/params.physics.tau_M;
            pcout << "  DIAG: mass_coeff=" << mass_coeff
                  << " (1/dt=" << 1.0/dt_val
                  << " + 1/tau_M=" << 1.0/params.physics.tau_M << ")\n";

            // Compute mass matrix action: M_DG * c_interp
            TrilinosWrappers::MPI::Vector mass_Ih(M_owned, mpi_comm);
            mass_Ih = 0;
            {
                const auto& fe_M = mag.get_dof_handler().get_fe();
                QGauss<dim> q_mass(fe_M.degree + 2);
                FEValues<dim> fv(fe_M, q_mass,
                    update_values | update_JxW_values);
                const unsigned int nq = q_mass.size();
                const unsigned int dpc = fe_M.n_dofs_per_cell();
                std::vector<double> ih_vals(nq);
                std::vector<types::global_dof_index> dofs(dpc);

                for (const auto& cell : mag.get_dof_handler().active_cell_iterators())
                {
                    if (!cell->is_locally_owned()) continue;
                    fv.reinit(cell);
                    fv.get_function_values(mx_interp, ih_vals);
                    cell->get_dof_indices(dofs);
                    for (unsigned int q = 0; q < nq; ++q)
                    {
                        double v = ih_vals[q] * fv.JxW(q);
                        for (unsigned int i = 0; i < dpc; ++i)
                            mass_Ih[dofs[i]] += v * fv.shape_value(i, q);
                    }
                }
                mass_Ih.compress(VectorOperation::add);
            }
            pcout << "  DIAG: ||M_DG*c_interp||=" << mass_Ih.l2_norm()
                  << "  ||mass_coeff*M_DG*c||="
                  << (mass_coeff * mass_Ih.l2_norm()) << "\n";

            // Transport action = A*I_h - mass_coeff*M*I_h
            TrilinosWrappers::MPI::Vector transport_Ih(A_Ih_x);
            transport_Ih.add(-mass_coeff, mass_Ih);
            pcout << "  DIAG: ||A_transport*Ih||=" << transport_Ih.l2_norm() << "\n";

            // ---- Cell IBP transport: ∫_K [-(U·∇Z_i)*I_h - 0.5*(∇·U)*Z_i*I_h] dx
            //   For CG: no face flux (continuity enforced by FE space).
            //   transport_Ih = cell_IBP (strong form).
            {
                const auto& diag_ux2 = ns_ptr
                    ? ns_ptr->get_ux_old_relevant() : ux_proj;
                const auto& diag_uy2 = ns_ptr
                    ? ns_ptr->get_uy_old_relevant() : uy_proj;
                const auto& diag_u_dof2 = ns_ptr
                    ? ns_ptr->get_ux_dof_handler() : vel_dof;

                const auto& fe_M2 = mag.get_dof_handler().get_fe();
                QGauss<dim> q_ibp(fe_M2.degree + 2);
                const unsigned int nq_ibp = q_ibp.size();
                const unsigned int dpc2 = fe_M2.n_dofs_per_cell();

                FEValues<dim> fv_M_ibp(fe_M2, q_ibp,
                    update_values | update_gradients | update_JxW_values);
                FEValues<dim> fv_U_ibp(diag_u_dof2.get_fe(), q_ibp,
                    update_values | update_gradients);

                std::vector<double> ih_ibp(nq_ibp);
                std::vector<double> ux_ibp(nq_ibp), uy_ibp(nq_ibp);
                std::vector<Tensor<1,dim>> gux_ibp(nq_ibp), guy_ibp(nq_ibp);
                std::vector<types::global_dof_index> dofs_ibp(dpc2);

                TrilinosWrappers::MPI::Vector cell_IBP_x(M_owned, mpi_comm);
                cell_IBP_x = 0;

                auto cM = mag.get_dof_handler().begin_active();
                auto cU = diag_u_dof2.begin_active();
                for (; cM != mag.get_dof_handler().end(); ++cM, ++cU)
                {
                    if (!cM->is_locally_owned()) continue;
                    fv_M_ibp.reinit(cM);
                    fv_U_ibp.reinit(cU);

                    fv_M_ibp.get_function_values(mx_interp, ih_ibp);
                    fv_U_ibp.get_function_values(diag_ux2, ux_ibp);
                    fv_U_ibp.get_function_values(diag_uy2, uy_ibp);
                    fv_U_ibp.get_function_gradients(diag_ux2, gux_ibp);
                    fv_U_ibp.get_function_gradients(diag_uy2, guy_ibp);
                    cM->get_dof_indices(dofs_ibp);

                    for (unsigned int q = 0; q < nq_ibp; ++q)
                    {
                        const double JxW = fv_M_ibp.JxW(q);
                        Tensor<1,dim> U_q;
                        U_q[0] = ux_ibp[q]; U_q[1] = uy_ibp[q];
                        double div_U = gux_ibp[q][0] + guy_ibp[q][1];

                        for (unsigned int i = 0; i < dpc2; ++i)
                        {
                            const double Z_i = fv_M_ibp.shape_value(i, q);
                            const Tensor<1,dim>& gZ_i = fv_M_ibp.shape_grad(i, q);

                            // IBP form: -(U·∇Z_i)*I_h - 0.5*(∇·U)*Z_i*I_h
                            double v = -(U_q * gZ_i) * ih_ibp[q]
                                       - 0.5 * div_U * Z_i * ih_ibp[q];
                            cell_IBP_x[dofs_ibp[i]] += v * JxW;
                        }
                    }
                }
                cell_IBP_x.compress(VectorOperation::add);

                // face_flux = transport_Ih - cell_IBP
                TrilinosWrappers::MPI::Vector face_flux_x(transport_Ih);
                face_flux_x -= cell_IBP_x;

                pcout << "  DIAG: ||cell_IBP_x||=" << cell_IBP_x.l2_norm()
                      << "  ||face_flux_x||=" << face_flux_x.l2_norm() << "\n";
                pcout << "  DIAG: expect: cell_IBP + face_flux = transport_Ih = "
                      << transport_Ih.l2_norm() << "\n";
                pcout << "  DIAG:         cell_IBP - strong_form = boundary_integral\n";

                // Also compute cell_IBP - strong_form to get boundary integral
                // (strong_form computed later, but we can compute it inline)
                TrilinosWrappers::MPI::Vector sf_inline(M_owned, mpi_comm);
                sf_inline = 0;
                cM = mag.get_dof_handler().begin_active();
                cU = diag_u_dof2.begin_active();
                std::vector<Tensor<1,dim>> gih_ibp(nq_ibp);
                for (; cM != mag.get_dof_handler().end(); ++cM, ++cU)
                {
                    if (!cM->is_locally_owned()) continue;
                    fv_M_ibp.reinit(cM);
                    fv_U_ibp.reinit(cU);
                    fv_M_ibp.get_function_values(mx_interp, ih_ibp);
                    fv_M_ibp.get_function_gradients(mx_interp, gih_ibp);
                    fv_U_ibp.get_function_values(diag_ux2, ux_ibp);
                    fv_U_ibp.get_function_values(diag_uy2, uy_ibp);
                    fv_U_ibp.get_function_gradients(diag_ux2, gux_ibp);
                    fv_U_ibp.get_function_gradients(diag_uy2, guy_ibp);
                    cM->get_dof_indices(dofs_ibp);
                    for (unsigned int q = 0; q < nq_ibp; ++q)
                    {
                        const double JxW = fv_M_ibp.JxW(q);
                        Tensor<1,dim> U_q;
                        U_q[0] = ux_ibp[q]; U_q[1] = uy_ibp[q];
                        double div_U = gux_ibp[q][0] + guy_ibp[q][1];
                        double UgIh = U_q[0]*gih_ibp[q][0] + U_q[1]*gih_ibp[q][1];
                        for (unsigned int i = 0; i < dpc2; ++i)
                        {
                            double Z_i = fv_M_ibp.shape_value(i, q);
                            double v = (UgIh + 0.5*div_U*ih_ibp[q]) * Z_i;
                            sf_inline[dofs_ibp[i]] += v * JxW;
                        }
                    }
                }
                sf_inline.compress(VectorOperation::add);

                TrilinosWrappers::MPI::Vector bdy_int(cell_IBP_x);
                bdy_int -= sf_inline;
                // bdy_int should be -∫_∂K (U·n)*Z_i*I_h dS
                pcout << "  DIAG: ||strong_form_inline||=" << sf_inline.l2_norm()
                      << "  ||boundary_integral||=" << bdy_int.l2_norm() << "\n";
                pcout << "  DIAG: verify: boundary_integral + face_flux should ≈ 0: "
                      << "||bdy+face||=" << [&]{
                          TrilinosWrappers::MPI::Vector sum(bdy_int);
                          sum += face_flux_x;
                          return sum.l2_norm();
                      }() << "\n";
            }

            // Compare with b decomposition
            // b = old_coeff*(M_old,Z) + relax_coeff*(chi*H,Z) + (f_MMS,Z)
            // Compute just the MMS source integral separately
            TrilinosWrappers::MPI::Vector mass_Mstar(M_owned, mpi_comm);
            mass_Mstar = 0;
            {
                MagExactMx<dim> exact_Mx_t(current_time, L_y);
                const auto& fe_M = mag.get_dof_handler().get_fe();
                QGauss<dim> q_mass(fe_M.degree + 2);
                FEValues<dim> fv(fe_M, q_mass,
                    update_values | update_quadrature_points |
                    update_JxW_values);
                const unsigned int nq = q_mass.size();
                const unsigned int dpc = fe_M.n_dofs_per_cell();
                std::vector<types::global_dof_index> dofs(dpc);

                for (const auto& cell : mag.get_dof_handler().active_cell_iterators())
                {
                    if (!cell->is_locally_owned()) continue;
                    fv.reinit(cell);
                    cell->get_dof_indices(dofs);
                    for (unsigned int q = 0; q < nq; ++q)
                    {
                        const auto& xq = fv.quadrature_point(q);
                        double Mstar = exact_Mx_t.value(xq);
                        double v = Mstar * fv.JxW(q);
                        for (unsigned int i = 0; i < dpc; ++i)
                            mass_Mstar[dofs[i]] += v * fv.shape_value(i, q);
                    }
                }
                mass_Mstar.compress(VectorOperation::add);
            }
            pcout << "  DIAG: ||(M*,Z)||=" << mass_Mstar.l2_norm()
                  << "  ||mass_coeff*(M*,Z)||="
                  << (mass_coeff * mass_Mstar.l2_norm()) << "\n";

            // Difference: mass_coeff*(I_h - M*, Z)
            TrilinosWrappers::MPI::Vector mass_diff(mass_Ih);
            mass_diff -= mass_Mstar;
            pcout << "  DIAG: ||mass_coeff*(Ih-M*,Z)||="
                  << (mass_coeff * mass_diff.l2_norm()) << "\n";
        }

        // (e) Manual "strong form" cell transport for the interpolant
        //     Compute ∫_K [U·∇I_h(Mx*)] Z_i dx  cell by cell
        //     For continuous I_h, this should equal A_transport * I_h(Mx*)
        //     (face contributions vanish since [[I_h]] = 0)
        {
            // Velocity references (same logic as time loop)
            const auto& diag_ux = ns_ptr
                ? ns_ptr->get_ux_old_relevant() : ux_proj;
            const auto& diag_uy = ns_ptr
                ? ns_ptr->get_uy_old_relevant() : uy_proj;
            const auto& diag_u_dof = ns_ptr
                ? ns_ptr->get_ux_dof_handler() : vel_dof;

            const auto& fe_M = mag.get_dof_handler().get_fe();
            QGauss<dim> quad_diag(fe_M.degree + 2);
            const unsigned int nq_d = quad_diag.size();
            const unsigned int dpc = fe_M.n_dofs_per_cell();

            FEValues<dim> fv_M(fe_M, quad_diag,
                update_values | update_gradients |
                update_quadrature_points | update_JxW_values);
            FEValues<dim> fv_U(diag_u_dof.get_fe(), quad_diag,
                update_values | update_gradients);

            std::vector<double>          imx_v(nq_d), imy_v(nq_d);
            std::vector<Tensor<1, dim>>  imx_g(nq_d), imy_g(nq_d);
            std::vector<double>          uxv(nq_d), uyv(nq_d);
            std::vector<Tensor<1, dim>>  uxg(nq_d), uyg(nq_d);
            std::vector<types::global_dof_index> ldofs(dpc);

            // Accumulate strong-form transport into a vector
            TrilinosWrappers::MPI::Vector sf_transport_x(M_owned, mpi_comm);
            sf_transport_x = 0;

            // Also compute some statistics
            double max_UgradErr = 0, max_UgradExact = 0, max_gradErr = 0;
            double max_divU = 0;

            auto cell_M2 = mag.get_dof_handler().begin_active();
            auto cell_U2 = diag_u_dof.begin_active();
            for (; cell_M2 != mag.get_dof_handler().end();
                 ++cell_M2, ++cell_U2)
            {
                if (!cell_M2->is_locally_owned()) continue;

                fv_M.reinit(cell_M2);
                fv_U.reinit(cell_U2);

                fv_M.get_function_values(mx_interp, imx_v);
                fv_M.get_function_gradients(mx_interp, imx_g);
                fv_U.get_function_values(diag_ux, uxv);
                fv_U.get_function_values(diag_uy, uyv);
                fv_U.get_function_gradients(diag_ux, uxg);
                fv_U.get_function_gradients(diag_uy, uyg);

                cell_M2->get_dof_indices(ldofs);

                for (unsigned int q = 0; q < nq_d; ++q)
                {
                    const auto& xq = fv_M.quadrature_point(q);
                    const double JxW = fv_M.JxW(q);

                    Tensor<1, dim> U_d;
                    U_d[0] = uxv[q]; U_d[1] = uyv[q];
                    double divU = uxg[q][0] + uyg[q][1];
                    max_divU = std::max(max_divU, std::abs(divU));

                    // Strong form of interp transport
                    double Ugm_interp = U_d[0]*imx_g[q][0]
                                      + U_d[1]*imx_g[q][1];
                    double sf = Ugm_interp + 0.5*divU*imx_v[q];

                    // Strong form of exact transport
                    double gMx_x = current_time * M_PI
                        * std::cos(M_PI*xq[0])
                        * std::sin(M_PI*xq[1]/L_y);
                    double gMx_y = current_time * (M_PI/L_y)
                        * std::sin(M_PI*xq[0])
                        * std::cos(M_PI*xq[1]/L_y);
                    double Mx_ex = current_time * std::sin(M_PI*xq[0])
                        * std::sin(M_PI*xq[1]/L_y);
                    double Ugm_exact = U_d[0]*gMx_x + U_d[1]*gMx_y;

                    // Gradient error
                    double ge0 = imx_g[q][0] - gMx_x;
                    double ge1 = imx_g[q][1] - gMx_y;
                    max_gradErr = std::max(max_gradErr,
                        std::sqrt(ge0*ge0 + ge1*ge1));
                    max_UgradErr = std::max(max_UgradErr,
                        std::abs(Ugm_interp - Ugm_exact));
                    max_UgradExact = std::max(max_UgradExact,
                        std::abs(Ugm_exact));

                    for (unsigned int i = 0; i < dpc; ++i)
                    {
                        double Zi = fv_M.shape_value(i, q);
                        sf_transport_x[ldofs[i]] += sf * Zi * JxW;
                    }
                }
            }
            sf_transport_x.compress(VectorOperation::add);

            // Compare: the matrix transport on the interpolant
            // A * I_h(Mx*) = A_mass*I_h + A_transport*I_h
            // A_transport * I_h = A*I_h - mass_coeff * M_mass * I_h
            // For now, just print norms
            pcout << "  DIAG: ||strong_form_transport_x||=" << sf_transport_x.l2_norm() << "\n";
            pcout << "  DIAG: max|∇(Ih-M*)|=" << max_gradErr
                  << "  max|U·∇(Ih-M*)|=" << max_UgradErr
                  << "  max|U·∇M*|=" << max_UgradExact
                  << "  max|∇·U|=" << max_divU << "\n";

            // Compare matrix action minus mass with strong form:
            // A*I_h - b = [A_mass + A_transport]*I_h - b
            // If we subtract mass from both sides:
            // A_transport*I_h = (A*I_h - b) + (b - A_mass*I_h)
            // = r_int + [b - mass_coeff*(I_h, Z)]
            // The quantity b - mass_coeff*(I_h, Z) is the RHS minus the mass
            // action. This includes old_coeff*(M_old, Z) + relax + source
            // - mass_coeff*(I_h, Z).
            // We can also directly compare sf_transport with r_int:
            // If [[I_h]] = 0: A_transport*I_h = sf_transport + face_contrib
            //   where face_contrib = 0 (since [[I_h]]=0 means face flux is
            //   just the cell boundary cancellation).
            // So A_transport*I_h ≈ sf_transport.
            // And: r_int = A_transport*I_h + A_mass*I_h - b
            //     = sf_transport + mass_stuff - b
            //     = sf_transport - (source_transport, Z)
            // since (mass_stuff - b_mass) = 0 (mass is consistent).
            // So: sf_transport - (source, Z) = r_int
            // I.e., ||r_int|| = ||sf_transport - (source_transport, Z)||
            // We already know sf_transport and we know (source_transport,Z)≈0
            // for Mx. So ||r_int|| ≈ ||sf_transport||.
            // Let's verify:
            pcout << "  DIAG: compare r_int_x=" << r_int_x.l2_norm()
                  << " vs sf_transport_x=" << sf_transport_x.l2_norm() << "\n";
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.time_s = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "  |U|_L2=" << std::scientific << std::setprecision(2)
          << result.ux_L2
          << "  p_L2=" << result.p_L2
          << "  φ_L2=" << result.phi_L2
          << "  M_L2=" << result.M_L2
          << "  time=" << std::fixed << std::setprecision(1)
          << result.time_s << "s\n";

    return result;
}


// ============================================================================
// Convergence study across refinement levels
// ============================================================================
PoissonMagNSConvergenceResult run_poisson_mag_ns_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm,
    bool use_projected_velocity = false,
    bool mag_only = false)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    // Local copy with MMS overrides
    Parameters p = params;
    p.enable_mms = true;
    p.enable_ns = true;
    p.enable_magnetic = true;
    p.use_algebraic_magnetization = false;
    p.enable_gravity = false;
    p.dipoles.intensity_max = 0.0;
    p.dipoles.positions.clear();

    // Use default domain [0,1]×[0,0.6] — same as standalone Poisson-Mag test
    // (Default from Parameters: x_min=0, x_max=1, y_min=0, y_max=0.6,
    //  initial_cells_x=10, initial_cells_y=6)

    // Key: enable Kelvin force with μ₀ > 0
    p.physics.mu_0 = 0.1;

    // Constant viscosity (no ∇ν correction needed)
    p.physics.nu_water = 1.0;
    p.physics.nu_ferro = 1.0;

    // Relaxation time: larger value to reduce stiffness and amplify transport
    p.physics.tau_M = 1.0;  // DIAGNOSTIC: default 1e-6 is very stiff

    PoissonMagNSConvergenceResult result;

    if (rank == 0)
    {
        std::cout << "\n============================================================\n";
        std::cout << "  Poisson + Magnetization + NS MMS (Kelvin Force)\n";
        std::cout << "============================================================\n";
        std::cout << "  MPI ranks:  "
                  << dealii::Utilities::MPI::n_mpi_processes(mpi_comm) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  Picard:     max=50  tol=1e-10  omega=0.35\n";
        std::cout << "  Physics:    mu_0=" << p.physics.mu_0
                  << "  nu=" << p.physics.nu_ferro
                  << "  chi_0=" << p.physics.chi_0
                  << "  tau_M=" << p.physics.tau_M << "\n";
        std::cout << "  FE:         u=Q" << p.fe.degree_velocity
                  << "  p=DG-P" << p.fe.degree_pressure
                  << "  phi=Q" << p.fe.degree_potential
                  << "  M=DG-Q" << p.fe.degree_magnetization << "\n";
        std::cout << "  Expected:   u_L2=" << result.expected_ux_L2
                  << "  p_L2=" << result.expected_p_L2
                  << "  phi_L2=" << result.expected_phi_L2
                  << "  M_L2=" << result.expected_M_L2 << "\n";
        std::cout << "  Coupling:   theta=+1 (no CH), psi=0 (no capillary)\n";
        std::cout << "  Velocity:   " << (use_projected_velocity
            ? "PROJECTED (exact U* interpolated onto CG-Q2)"
            : "NS SOLVE") << "\n";
        std::cout << "  Refs:      ";
        for (auto r : refinements) std::cout << " " << r;
        std::cout << "\n";
    }

    for (unsigned int ref : refinements)
        result.results.push_back(
            run_single_level(ref, p, n_time_steps, mpi_comm,
                             use_projected_velocity, mag_only));

    result.compute_rates();
    return result;
}


// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    // Defaults
    std::vector<unsigned int> refinements = {2, 3, 4, 5};
    unsigned int n_time_steps = 10;
    bool use_projected_velocity = false;
    bool mag_only = false;  // skip Poisson Picard, test DG transport alone

    // Parse args
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--refs" && i + 1 < argc)
        {
            refinements.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                refinements.push_back(std::stoi(argv[++i]));
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            n_time_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--project-u")
        {
            use_projected_velocity = true;
        }
        else if (arg == "--mag-only")
        {
            use_projected_velocity = true;
            mag_only = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            if (rank == 0)
                std::cout << "Usage: mpirun -np N " << argv[0]
                          << " [--refs 2 3 4 5] [--steps 10]"
                          << " [--project-u] [--mag-only]\n";
            return 0;
        }
    }

    Parameters params;

    try
    {
        auto result = run_poisson_mag_ns_mms(
            refinements, params, n_time_steps, MPI_COMM_WORLD,
            use_projected_velocity, mag_only);

        if (rank == 0)
        {
            result.print();

            std::cout << "\n============================================================\n";
            if (result.passes())
                std::cout << "  [PASS] All convergence rates within tolerance.\n";
            else
                std::cout << "  [FAIL] Some convergence rates below expected.\n";
            std::cout << "============================================================\n";

            result.write_csv("poisson_mag_ns_mms_rates.csv");
        }

        return result.passes() ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
            std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
