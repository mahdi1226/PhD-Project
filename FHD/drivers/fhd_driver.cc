// ============================================================================
// drivers/fhd_driver.cc — Production FHD Simulation Driver
//
// Nochetto, Salgado & Tomas, Algorithm 42 (arXiv:1511.04381):
//   Per time step:
//     1. Picard loop: Poisson(M_relaxed) <-> Mag(M_old, H, u_old)
//     2. NS(u_old, w_old, M, phi) — Kelvin force + micropolar + convection
//     3. AngMom(w_old, u_new, M, phi) — curl coupling + magnetic torque
//
// Usage:
//   mpirun -np 4 ./fhd_driver --spinning-magnet -r 5 --dt 0.01 -v
//   mpirun -np 4 ./fhd_driver --nu 0.1 --kappa_0 5 -r 4 --t_final 2.0
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "passive_scalar/passive_scalar.h"
#include "mesh/mesh.h"
#include "utilities/timestamp.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr int dim = 2;

// ============================================================================
// Section 7.1: Spinning Magnet — time-dependent dipole update
//
// t ∈ [0,1]: ramp intensity 0→10, static position (0.5, -0.4), d=(0,1)
// t ∈ [1,2]: constant field, fluid rests
// t ∈ [2,4]: dipole orbits (0.5, 0.5) on circle R=0.9, d toward center
//            CCW, one full orbit in 2s → ω = π rad/s
// ============================================================================
void update_spinning_magnet(Parameters& params, double t)
{
    if (t <= 2.0)
        return;  // phases 1-2: static (preset handles ramp)

    const double cx = 0.5, cy = 0.5;
    const double R = 0.9;
    const double omega = M_PI;  // π rad/s
    const double theta = -M_PI / 2.0 + omega * (t - 2.0);

    params.dipoles.positions[0] = dealii::Point<2>(
        cx + R * std::cos(theta),
        cy + R * std::sin(theta));

    params.dipoles.direction = {-std::cos(theta), -std::sin(theta)};
}

// ============================================================================
// Section 7.2: Ferrofluid Pumping — per-dipole intensity update
//
// α_s(t) = |sin(ωt − κx_s)|^{2q}
// ω = 2πf = 20π (f=10 Hz), κ = 2π/λ = 2π (λ=1), q = 5
// Creates traveling pulses from left to right
// ============================================================================
void update_pumping_dipoles(Parameters& params, double t)
{
    const double f = 10.0;
    const double omega = 2.0 * M_PI * f;
    const double lambda = 1.0;
    const double kappa = 2.0 * M_PI / lambda;
    const int q = 5;

    for (std::size_t s = 0; s < params.dipoles.positions.size(); ++s)
    {
        const double x_s = params.dipoles.positions[s][0];
        const double arg = std::sin(omega * t - kappa * x_s);
        params.dipoles.intensities[s] = std::pow(std::abs(arg), 2 * q);
    }
}

// ============================================================================
// Section 7.3, Approach 1 (Eq. 105): Two alternating dipoles
//
// α₁ = α₀ sin(ωt), α₂ = α₀ sin(ωt + π/2)
// f = 20Hz, α₀ = 5.0
// ============================================================================
void update_stirring_approach1(Parameters& params, double t)
{
    const double alpha_0 = params.dipoles.intensity_max;  // 5.0
    const double f = params.dipoles.frequency;             // 20Hz default
    const double omega = 2.0 * M_PI * f;

    params.dipoles.intensities[0] = alpha_0 * std::sin(omega * t);
    params.dipoles.intensities[1] = alpha_0 * std::sin(omega * t + M_PI / 2.0);
}

// ============================================================================
// Section 7.3, Approach 2 (Eq. 106): Eight-dipole traveling wave
//
// α_s = α₀ |sin(ωt − κx_s)|, κ = 2π/λ, λ = 0.8, f = 20Hz, α₀ = 5.0
// ============================================================================
void update_stirring_approach2(Parameters& params, double t)
{
    const double alpha_0 = params.dipoles.intensity_max;  // 5.0
    const double f = params.dipoles.frequency;             // 20Hz default
    const double omega = 2.0 * M_PI * f;
    const double lambda = 0.8;
    const double kappa = 2.0 * M_PI / lambda;

    for (std::size_t s = 0; s < params.dipoles.positions.size(); ++s)
    {
        const double x_s = params.dipoles.positions[s][0];
        params.dipoles.intensities[s] =
            alpha_0 * std::abs(std::sin(omega * t - kappa * x_s));
    }
}

// ============================================================================
// Step function IC for passive scalar: c = 1 for y < threshold, 0 otherwise
// ============================================================================
template <int dim>
class StepFunctionY : public dealii::Function<dim>
{
public:
    StepFunctionY(double y_threshold) : y_threshold_(y_threshold) {}
    double value(const dealii::Point<dim>& p,
                 unsigned int /*component*/ = 0) const override
    {
        return (p[1] < y_threshold_) ? 1.0 : 0.0;
    }
private:
    double y_threshold_;
};

int main(int argc, char* argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    // ================================================================
    // 1. Parse parameters
    // ================================================================
    Parameters params = Parameters::parse_command_line(argc, argv);
    params.enable_mms = false;

    const double dt = params.time.dt;
    const double t_final = params.time.t_final;
    const unsigned int max_steps = (params.run.steps > 0)
        ? static_cast<unsigned int>(params.run.steps)
        : params.time.max_steps;
    const unsigned int vtk_interval = params.output.vtk_interval;
    const bool verbose = params.output.verbose;

    const unsigned int max_picard = params.picard_iterations;
    const double picard_tol = params.picard_tolerance;
    const double omega = params.picard_relaxation;

    // Create timestamped output directory: Results/{timestamp}_{experiment}_r{ref}/
    std::string output_dir;
    {
        char dir_buf[512];
        std::memset(dir_buf, 0, sizeof(dir_buf));

        if (rank == 0)
        {
            const std::string ts = get_timestamp();
            const std::string exp_name = params.experiment_name.empty()
                ? "generic" : params.experiment_name;
            output_dir = params.output.folder + "/" + ts + "_" + exp_name
                         + "_r" + std::to_string(params.mesh.initial_refinement);

            std::filesystem::create_directories(output_dir);
            std::strncpy(dir_buf, output_dir.c_str(), sizeof(dir_buf) - 1);
        }

        MPI_Bcast(dir_buf, 512, MPI_CHAR, 0, mpi_comm);
        output_dir = std::string(dir_buf);
    }

    pcout << "================================================================\n"
          << "  FHD SIMULATION (Nochetto Algorithm 42)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << Utilities::MPI::n_mpi_processes(mpi_comm) << "\n"
          << "  dt:             " << std::scientific << std::setprecision(3) << dt << "\n"
          << "  t_final:        " << t_final << "\n"
          << "  max_steps:      " << max_steps << "\n"
          << "  Picard:         max=" << max_picard
          << "  tol=" << picard_tol << "  omega=" << omega << "\n"
          << "  Refinement:     " << params.mesh.initial_refinement << "\n"
          << "  Physics:        nu=" << params.physics.nu
          << "  nu_r=" << params.physics.nu_r
          << "  mu_0=" << params.physics.mu_0 << "\n"
          << "                  kappa_0=" << params.physics.kappa_0
          << "  T_relax=" << params.physics.T_relax
          << "  sigma=" << params.physics.sigma << "\n"
          << "  Experiment:     " << (params.experiment_name.empty()
                                     ? "generic" : params.experiment_name) << "\n"
          << "  Output:         " << output_dir
          << "  vtk_interval=" << vtk_interval << "\n"
          << "================================================================\n\n";

    // ================================================================
    // 2. Create mesh
    // ================================================================
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    FHDMesh::create_mesh<dim>(triangulation, params);

    const auto mesh_info = FHDMesh::get_mesh_info<dim>(triangulation);
    pcout << "  Mesh: " << mesh_info.n_global_active_cells << " cells"
          << ", h_min=" << std::scientific << std::setprecision(3) << mesh_info.h_min
          << ", h_max=" << mesh_info.h_max << "\n";

    // ================================================================
    // 3. Setup subsystems
    // ================================================================
    PoissonSubsystem<dim> poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    NavierStokesSubsystem<dim> ns(params, mpi_comm, triangulation);
    AngularMomentumSubsystem<dim> am(params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();
    ns.setup();
    am.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs = mag.get_dof_handler().n_dofs();
    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int w_dofs = am.get_dof_handler().n_dofs();
    const unsigned int total_dofs = phi_dofs + 2 * M_dofs + 2 * vel_dofs + p_dofs + w_dofs;

    // Passive scalar (Section 7.3, optional)
    std::unique_ptr<PassiveScalarSubsystem<dim>> scalar;
    unsigned int c_dofs = 0;
    if (params.enable_passive_scalar)
    {
        scalar = std::make_unique<PassiveScalarSubsystem<dim>>(
            params, mpi_comm, triangulation);
        scalar->setup();
        scalar->initialize(StepFunctionY<dim>(0.5));
        scalar->update_ghosts();
        c_dofs = scalar->get_dof_handler().n_dofs();
    }

    pcout << "  DoFs: phi=" << phi_dofs << " M=" << M_dofs << "(x2)"
          << " vel=" << vel_dofs << "(x2) p=" << p_dofs
          << " w=" << w_dofs;
    if (scalar)
        pcout << " c=" << c_dofs;
    pcout << " total=" << total_dofs + c_dofs << "\n\n";

    // ================================================================
    // 4. Initialize solutions to zero
    // ================================================================

    // Velocity: u_old = 0
    IndexSet ux_relevant = DoFTools::extract_locally_relevant_dofs(
        ns.get_ux_dof_handler());
    IndexSet uy_relevant = DoFTools::extract_locally_relevant_dofs(
        ns.get_uy_dof_handler());

    TrilinosWrappers::MPI::Vector ux_old_rel(
        ns.get_ux_dof_handler().locally_owned_dofs(), ux_relevant, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_old_rel(
        ns.get_uy_dof_handler().locally_owned_dofs(), uy_relevant, mpi_comm);
    ux_old_rel = 0.0;
    uy_old_rel = 0.0;

    // Angular velocity: w_old = 0
    IndexSet w_relevant_set = DoFTools::extract_locally_relevant_dofs(
        am.get_dof_handler());
    TrilinosWrappers::MPI::Vector w_old_rel(
        am.get_dof_handler().locally_owned_dofs(), w_relevant_set, mpi_comm);
    w_old_rel = 0.0;

    // Magnetization: M_old = 0
    IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
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

    Mx_old = 0.0;
    My_old = 0.0;
    Mx_relaxed = 0.0;
    My_relaxed = 0.0;

    // Passive scalar old solution
    TrilinosWrappers::MPI::Vector c_old_rel;
    if (scalar)
    {
        IndexSet c_relevant_set = DoFTools::extract_locally_relevant_dofs(
            scalar->get_dof_handler());
        c_old_rel.reinit(scalar->get_dof_handler().locally_owned_dofs(),
                         c_relevant_set, mpi_comm);
        c_old_rel = scalar->get_relevant();
    }

    // Initial Poisson solve (with M=0 → phi determined by h_a only)
    poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                         mag.get_dof_handler(), 0.0);
    poisson.solve();
    poisson.update_ghosts();

    // ================================================================
    // Unified VTK output helper (lambda)
    // ================================================================
    auto write_unified_vtu = [&](unsigned int out_step, double out_time)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(ns.get_ux_dof_handler());

        // Velocity (CG Q2, attached DoF handler)
        data_out.add_data_vector(ns.get_ux_relevant(), "ux");
        data_out.add_data_vector(ns.get_uy_relevant(), "uy");

        // Pressure (DG P1, different DoF handler)
        data_out.add_data_vector(ns.get_p_dof_handler(),
                                 ns.get_p_relevant(), "p");

        // Magnetic potential (CG Q2)
        data_out.add_data_vector(poisson.get_dof_handler(),
                                 poisson.get_solution_relevant(), "phi");

        // Angular velocity (CG Q2)
        data_out.add_data_vector(am.get_dof_handler(),
                                 am.get_relevant(), "w");

        // Magnetization (DG Q2)
        data_out.add_data_vector(mag.get_dof_handler(),
                                 mag.get_Mx_relevant(), "Mx");
        data_out.add_data_vector(mag.get_dof_handler(),
                                 mag.get_My_relevant(), "My");

        // Cell-averaged derived quantities
        const unsigned int n_cells = triangulation.n_active_cells();
        dealii::Vector<float> U_mag_cell(n_cells);
        dealii::Vector<float> M_mag_cell(n_cells);
        dealii::Vector<float> H_mag_cell(n_cells);

        {
            const dealii::QMidpoint<dim> q_mid;

            // U_mag from velocity
            dealii::FEValues<dim> fe_vel(ns.get_ux_dof_handler().get_fe(),
                                         q_mid, dealii::update_values);
            unsigned int idx = 0;
            for (const auto& cell : ns.get_ux_dof_handler().active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_vel.reinit(cell);
                    std::vector<double> ux_val(1), uy_val(1);
                    fe_vel.get_function_values(ns.get_ux_relevant(), ux_val);
                    fe_vel.get_function_values(ns.get_uy_relevant(), uy_val);
                    U_mag_cell[idx] = static_cast<float>(
                        std::sqrt(ux_val[0]*ux_val[0] + uy_val[0]*uy_val[0]));
                }
                ++idx;
            }

            // M_mag from magnetization
            dealii::FEValues<dim> fe_mag(mag.get_dof_handler().get_fe(),
                                         q_mid, dealii::update_values);
            idx = 0;
            for (const auto& cell : mag.get_dof_handler().active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_mag.reinit(cell);
                    std::vector<double> mx_val(1), my_val(1);
                    fe_mag.get_function_values(mag.get_Mx_relevant(), mx_val);
                    fe_mag.get_function_values(mag.get_My_relevant(), my_val);
                    M_mag_cell[idx] = static_cast<float>(
                        std::sqrt(mx_val[0]*mx_val[0] + my_val[0]*my_val[0]));
                }
                ++idx;
            }

            // H_mag from Poisson gradient
            dealii::FEValues<dim> fe_poi(poisson.get_dof_handler().get_fe(),
                                         q_mid, dealii::update_gradients);
            idx = 0;
            for (const auto& cell : poisson.get_dof_handler().active_cell_iterators())
            {
                if (cell->is_locally_owned())
                {
                    fe_poi.reinit(cell);
                    std::vector<dealii::Tensor<1, dim>> grad_phi(1);
                    fe_poi.get_function_gradients(poisson.get_solution_relevant(),
                                                   grad_phi);
                    H_mag_cell[idx] = static_cast<float>(grad_phi[0].norm());
                }
                ++idx;
            }
        }

        data_out.add_data_vector(U_mag_cell, "U_mag",
                                 dealii::DataOut<dim>::type_cell_data);
        data_out.add_data_vector(M_mag_cell, "M_mag",
                                 dealii::DataOut<dim>::type_cell_data);
        data_out.add_data_vector(H_mag_cell, "H_mag",
                                 dealii::DataOut<dim>::type_cell_data);

        // Passive scalar concentration (if enabled)
        if (scalar)
            data_out.add_data_vector(scalar->get_dof_handler(),
                                     scalar->get_relevant(), "c");

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "solution_",
            out_step, mpi_comm, 4, 0);
    };

    // Write initial VTK (step 0)
    write_unified_vtu(0, 0.0);

    // ================================================================
    // Open CSV diagnostics file (rank 0 only)
    // ================================================================
    std::ofstream csv_file;
    if (rank == 0)
    {
        csv_file.open(output_dir + "/diagnostics.csv");
        csv_file << "step,time,dt,picard_iters,picard_res,"
                 << "ux_min,ux_max,uy_min,uy_max,U_max,E_kin,divU_L2,"
                 << "p_min,p_max,"
                 << "w_min,w_max,w_max_abs,"
                 << "Mx_min,Mx_max,My_min,My_max,M_max,"
                 << "phi_min,phi_max,H_max,"
                 << "CFL,n_cells,n_dofs_total,wall_s";
        if (scalar)
            csv_file << ",c_min,c_max,c_mass";
        csv_file << "\n";
    }

    pcout << "  Step   Time         Picard  NS_res       AM_res       Wall(s)\n"
          << "  ----   ----------   ------  ----------   ----------   -------\n";

    // ================================================================
    // 5. Time stepping
    // ================================================================
    auto wall_start = std::chrono::high_resolution_clock::now();
    double current_time = 0.0;
    unsigned int step = 0;

    while (current_time < t_final - 1e-14 * dt && step < max_steps)
    {
        current_time += dt;
        ++step;

        Timer step_timer;
        step_timer.start();

        // ============================================================
        // Update time-dependent applied field
        // ============================================================
        if (params.experiment_name == "spinning_magnet")
            update_spinning_magnet(params, current_time);
        else if (params.experiment_name == "pumping")
            update_pumping_dipoles(params, current_time);
        else if (params.experiment_name == "stirring_approach1")
            update_stirring_approach1(params, current_time);
        else if (params.experiment_name == "stirring_approach2" ||
                 params.experiment_name == "stirring_approach2_enhanced")
            update_stirring_approach2(params, current_time);

        // ============================================================
        // Phase 1: Picard iteration — Poisson <-> Magnetization
        // ============================================================
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        unsigned int picard_iters = 0;
        double picard_res_final = 0.0;

        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Step A: Poisson -> phi using relaxed M
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Step B: Magnetization -> M_raw using phi, u_old, w_old
            //   Includes spin-magnetization coupling: -(M_old × W, z) on RHS
            mag.assemble(Mx_old, My_old,
                         poisson.get_solution_relevant(),
                         poisson.get_dof_handler(),
                         ux_old_rel, uy_old_rel,
                         ns.get_ux_dof_handler(),
                         dt, current_time,
                         w_old_rel, &am.get_dof_handler());
            mag.solve();
            mag.update_ghosts();

            // Step C: Under-relax M
            Mx_prev = Mx_relaxed;
            My_prev = My_relaxed;

            Mx_relaxed_owned = mag.get_Mx_solution();
            Mx_relaxed_owned *= omega;
            Mx_relaxed_owned.add(1.0 - omega, Mx_prev);

            My_relaxed_owned = mag.get_My_solution();
            My_relaxed_owned *= omega;
            My_relaxed_owned.add(1.0 - omega, My_prev);

            Mx_relaxed = Mx_relaxed_owned;
            My_relaxed = My_relaxed_owned;

            // Step D: Convergence check
            TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_comm);
            TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_comm);
            Mx_diff = Mx_relaxed_owned;
            Mx_diff -= Mx_prev;
            My_diff = My_relaxed_owned;
            My_diff -= My_prev;

            const double change = Mx_diff.l2_norm() + My_diff.l2_norm();
            const double norm = Mx_relaxed_owned.l2_norm()
                              + My_relaxed_owned.l2_norm() + 1e-14;
            picard_res_final = change / norm;

            picard_iters = k + 1;

            if (picard_res_final < picard_tol)
                break;

            if (k == max_picard - 1 && verbose)
            {
                pcout << "  WARNING: Picard did not converge at step "
                      << step << ", res=" << std::scientific
                      << std::setprecision(2) << picard_res_final << "\n";
            }
        }

        // Final Poisson solve consistent with converged M
        poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                             mag.get_dof_handler(), current_time);
        poisson.solve();
        poisson.update_ghosts();

        // ============================================================
        // Phase 2: NS with Kelvin force + micropolar + convection
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/true,
                    w_old_rel, am.get_dof_handler(),
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler());

        auto ns_info = ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Phase 3: AngMom with curl coupling + magnetic torque + convection
        //   Paper Eq. 52c: j b_h(U^k, W^k, X) is included
        // ============================================================
        am.assemble(w_old_rel,
                    dt, current_time,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler(),
                    /*include_convection=*/true,
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler());

        auto am_info = am.solve();
        am.update_ghosts();

        // ============================================================
        // Phase 4: Passive scalar transport (Section 7.3, Eq. 104)
        // ============================================================
        if (scalar)
        {
            scalar->assemble(c_old_rel, dt,
                             ns.get_ux_relevant(), ns.get_uy_relevant(),
                             ns.get_ux_dof_handler());
            scalar->solve();
            scalar->update_ghosts();
        }

        // ============================================================
        // Advance old solutions
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();
        w_old_rel = am.get_relevant();
        Mx_old = Mx_relaxed;
        My_old = My_relaxed;
        if (scalar)
            c_old_rel = scalar->get_relevant();

        step_timer.stop();
        const double step_wall = step_timer.wall_time();

        // ============================================================
        // Compute diagnostics from all subsystems
        // ============================================================
        auto ns_diag  = ns.compute_diagnostics();
        auto mag_diag = mag.compute_diagnostics();
        auto poi_diag = poisson.compute_diagnostics();
        auto am_diag  = am.compute_diagnostics();

        // Scalar diagnostics must be computed by ALL ranks (uses MPI::sum)
        typename PassiveScalarSubsystem<dim>::Diagnostics sc_diag;
        if (scalar)
            sc_diag = scalar->compute_diagnostics();

        const double CFL = (mesh_info.h_min > 0.0)
            ? ns_diag.U_max * dt / mesh_info.h_min : 0.0;

        // ============================================================
        // CSV diagnostics (every step, rank 0 only)
        // ============================================================
        if (rank == 0)
        {
            csv_file << step << ","
                     << std::scientific << std::setprecision(8)
                     << current_time << ","
                     << dt << ","
                     << picard_iters << ","
                     << picard_res_final << ","
                     << ns_diag.ux_min << "," << ns_diag.ux_max << ","
                     << ns_diag.uy_min << "," << ns_diag.uy_max << ","
                     << ns_diag.U_max << ","
                     << ns_diag.E_kin << ","
                     << ns_diag.divU_L2 << ","
                     << ns_diag.p_min << "," << ns_diag.p_max << ","
                     << am_diag.w_min << "," << am_diag.w_max << ","
                     << am_diag.w_max_abs << ","
                     << mag_diag.Mx_min << "," << mag_diag.Mx_max << ","
                     << mag_diag.My_min << "," << mag_diag.My_max << ","
                     << mag_diag.M_max << ","
                     << poi_diag.phi_min << "," << poi_diag.phi_max << ","
                     << poi_diag.H_max << ","
                     << CFL << ","
                     << mesh_info.n_global_active_cells << ","
                     << total_dofs << ","
                     << std::fixed << std::setprecision(2)
                     << step_wall;
            if (scalar)
            {
                csv_file << "," << std::scientific << std::setprecision(8)
                         << sc_diag.c_min << ","
                         << sc_diag.c_max << ","
                         << sc_diag.c_mass;
            }
            csv_file << "\n";
            csv_file << std::flush;
        }

        // ============================================================
        // Console diagnostics
        // ============================================================
        pcout << "  " << std::setw(4) << step
              << "   " << std::scientific << std::setprecision(4) << current_time
              << "   " << std::setw(5) << picard_iters
              << "   " << std::scientific << std::setprecision(2)
              << ns_info.residual
              << "   " << am_info.residual
              << "   " << std::fixed << std::setprecision(1)
              << step_wall << "\n";

        // ============================================================
        // Unified VTK output
        // ============================================================
        if (step % vtk_interval == 0 || current_time >= t_final - 1e-14 * dt)
        {
            write_unified_vtu(step, current_time);

            if (verbose)
                pcout << "         → VTK output written (step " << step << ")\n";
        }
    }

    // ================================================================
    // 6. Summary
    // ================================================================
    if (rank == 0 && csv_file.is_open())
        csv_file.close();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();

    pcout << "\n================================================================\n"
          << "  SIMULATION COMPLETE\n"
          << "================================================================\n"
          << "  Steps:      " << step << "\n"
          << "  Final time: " << std::scientific << std::setprecision(4) << current_time << "\n"
          << "  Wall time:  " << std::fixed << std::setprecision(1) << wall_time << " s\n"
          << "  Output:     " << output_dir << "/\n"
          << "================================================================\n";

    return 0;
}
