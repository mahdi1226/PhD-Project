// ============================================================================
// drivers/ch_benchmark_driver.cc — Phase B: Droplet & Square Benchmarks
//
// CH + NS two-phase coupling (no magnetics):
//   Per time step (sequential, no Picard):
//     1. Solve NS for (u, p) with capillary force from (φ_old, μ_old)
//     2. Solve CH for (φ, μ) with convection from u
//
// Benchmarks:
//   --droplet : Circular droplet equilibrium (Young-Laplace validation)
//   --square  : Square relaxation to circle (energy decay validation)
//
// Usage:
//   mpirun -np 2 ./ch_benchmark --droplet -r 5 --dt 1e-3 --t-final 1.0
//   mpirun -np 2 ./ch_benchmark --square  -r 5 --dt 1e-3 --t-final 2.0
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "cahn_hilliard/cahn_hilliard.h"
#include "mesh/mesh.h"
#include "utilities/timestamp.h"
#include "physics/benchmark_initial_conditions.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>

constexpr int dim = 2;

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    // ================================================================
    // CLI defaults
    // ================================================================
    std::string benchmark = "droplet";
    unsigned int refinement = 5;
    double dt = 1e-3;
    double t_final = 1.0;
    unsigned int vtk_interval = 10;
    double epsilon = 0.02;
    double sigma = 1.0;
    double gamma_ch = 1.0;
    double nu = 1.0;
    double radius = 0.25;
    bool verbose = false;

    // ================================================================
    // Parse CLI
    // ================================================================
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--droplet") == 0)
            benchmark = "droplet";
        else if (std::strcmp(argv[i], "--square") == 0)
        {
            benchmark = "square";
            t_final = 2.0;
        }
        else if ((std::strcmp(argv[i], "-r") == 0 ||
                  std::strcmp(argv[i], "--refinement") == 0) && i+1 < argc)
            refinement = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--dt") == 0 && i+1 < argc)
            dt = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--t-final") == 0 && i+1 < argc)
            t_final = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-interval") == 0 && i+1 < argc)
            vtk_interval = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i+1 < argc)
            epsilon = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--sigma") == 0 && i+1 < argc)
            sigma = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--gamma") == 0 && i+1 < argc)
            gamma_ch = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--nu") == 0 && i+1 < argc)
            nu = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--radius") == 0 && i+1 < argc)
            radius = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0 ||
                 std::strcmp(argv[i], "--verbose") == 0)
            verbose = true;
    }

    // ================================================================
    // Build Parameters
    // ================================================================
    Parameters params;
    params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
    params.domain.y_min = 0.0;  params.domain.y_max = 1.0;
    params.domain.initial_cells_x = 1;
    params.domain.initial_cells_y = 1;
    params.mesh.initial_refinement = refinement;

    params.physics.nu = nu;
    params.physics.nu_r = 0.0;    // no micropolar

    params.cahn_hilliard_params.epsilon = epsilon;
    params.cahn_hilliard_params.gamma   = gamma_ch;
    params.cahn_hilliard_params.sigma   = sigma;
    params.enable_cahn_hilliard = true;

    params.time.dt = dt;
    params.time.t_final = t_final;
    params.time.max_steps = static_cast<unsigned int>(
        std::ceil(t_final / dt) + 10);

    params.output.folder = SOURCE_DIR "/Results";
    params.output.vtk_interval = vtk_interval;
    params.output.verbose = verbose;
    params.experiment_name = benchmark;

    // ================================================================
    // Derived quantities
    // ================================================================
    const unsigned int max_steps = params.time.max_steps;
    const double nu_eff = params.physics.nu + params.physics.nu_r;
    const double expected_p_jump = (benchmark == "droplet") ? sigma / radius : 0.0;

    // ================================================================
    // 1. Mesh
    // ================================================================
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    FHDMesh::create_mesh<dim>(triangulation, params);

    const double h_min = GridTools::minimal_cell_diameter(triangulation);
    const unsigned int n_cells = triangulation.n_global_active_cells();

    // ================================================================
    // 2. Subsystems
    // ================================================================
    NavierStokesSubsystem<dim> ns(params, mpi_comm, triangulation);
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);

    ns.setup();
    ch.setup();

    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int ch_dofs = ch.get_dof_handler().n_dofs();
    const unsigned int total_dofs = 2*vel_dofs + p_dofs + ch_dofs;

    // Dummy angular velocity DoFHandler (NS signature requires it)
    FE_Q<dim> fe_dummy(params.fe.degree_angular);
    DoFHandler<dim> w_dof_handler(triangulation);
    w_dof_handler.distribute_dofs(fe_dummy);
    IndexSet w_owned = w_dof_handler.locally_owned_dofs();
    IndexSet w_rel_set = DoFTools::extract_locally_relevant_dofs(w_dof_handler);
    TrilinosWrappers::MPI::Vector w_dummy(w_owned, w_rel_set, mpi_comm);
    w_dummy = 0.0;

    // ================================================================
    // 3. Initial conditions
    // ================================================================
    // Velocity: zero (fluid at rest)
    IndexSet ux_rel_set = DoFTools::extract_locally_relevant_dofs(
        ns.get_ux_dof_handler());
    IndexSet uy_rel_set = DoFTools::extract_locally_relevant_dofs(
        ns.get_uy_dof_handler());

    TrilinosWrappers::MPI::Vector ux_old_rel(
        ns.get_ux_dof_handler().locally_owned_dofs(), ux_rel_set, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_old_rel(
        ns.get_uy_dof_handler().locally_owned_dofs(), uy_rel_set, mpi_comm);
    ux_old_rel = 0.0;
    uy_old_rel = 0.0;

    // Phase field: droplet or square
    if (benchmark == "droplet")
    {
        Point<dim> center(0.5, 0.5);
        CircularDropletIC<dim> ic(center, radius, epsilon);
        ch.initialize(ic);
    }
    else // square
    {
        Point<dim> lower(0.25, 0.25);
        Point<dim> upper(0.75, 0.75);
        SquareRegionIC<dim> ic(lower, upper, epsilon);
        ch.initialize(ic);
    }
    ch.update_ghosts();
    ch.save_old_solution();

    // ================================================================
    // 4. Output directory
    // ================================================================
    std::string output_dir;
    {
        char dir_buf[512];
        std::memset(dir_buf, 0, sizeof(dir_buf));

        if (rank == 0)
        {
            const std::string ts = get_timestamp();
            output_dir = params.output.folder + "/" + ts + "_" + benchmark
                         + "_r" + std::to_string(refinement);
            std::filesystem::create_directories(output_dir);
            std::strncpy(dir_buf, output_dir.c_str(), sizeof(dir_buf) - 1);
        }

        MPI_Bcast(dir_buf, 512, MPI_CHAR, 0, mpi_comm);
        output_dir = std::string(dir_buf);
    }

    // ================================================================
    // 5. VTK output lambda
    // ================================================================
    auto write_vtu = [&](unsigned int out_step, double out_time)
    {
        // NS fields: velocity, pressure, |u|
        {
            DataOut<dim> data_out;
            data_out.attach_dof_handler(ns.get_ux_dof_handler());

            data_out.add_data_vector(ns.get_ux_relevant(), "ux");
            data_out.add_data_vector(ns.get_uy_relevant(), "uy");

            data_out.add_data_vector(ns.get_p_dof_handler(),
                                     ns.get_p_relevant(), "p");

            // Cell-averaged |u|
            const unsigned int nc = triangulation.n_active_cells();
            Vector<float> U_mag_cell(nc);
            {
                const QMidpoint<dim> q_mid;
                FEValues<dim> fe_vel(ns.get_ux_dof_handler().get_fe(),
                                      q_mid, update_values);
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
            }
            data_out.add_data_vector(U_mag_cell, "U_mag",
                                     DataOut<dim>::type_cell_data);

            data_out.build_patches();
            data_out.write_vtu_with_pvtu_record(
                output_dir + "/", "ns_",
                out_step, mpi_comm, 4, 0);
        }

        // CH fields: phi, mu (separate file — FESystem has non-contiguous DoFs)
        ch.write_vtu(output_dir, out_step, out_time);
    };

    // ================================================================
    // Initial diagnostics + VTK at step 0
    // ================================================================
    ns.update_ghosts();

    auto ch_diag0 = ch.compute_diagnostics();
    const double initial_mass = ch_diag0.phi_mass;
    const double initial_energy = ch_diag0.free_energy;

    write_vtu(0, 0.0);

    // ================================================================
    // Header
    // ================================================================
    pcout << "\n"
          << "================================================================\n"
          << "  CH + NS BENCHMARK: "
          << (benchmark == "droplet" ? "Circular Droplet" : "Square Relaxation")
          << "\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  Refinement:     " << refinement
          << "  (" << n_cells << " cells, h_min=" << std::scientific
          << std::setprecision(3) << h_min << ")\n"
          << "  Total DoFs:     " << total_dofs
          << "  (vel=" << vel_dofs << "x2 + p=" << p_dofs
          << " + ch=" << ch_dofs << ")\n"
          << "  dt:             " << dt << "\n"
          << "  t_final:        " << t_final << "\n"
          << "  epsilon:        " << epsilon
          << "  (eps/h=" << std::fixed << std::setprecision(2)
          << epsilon / h_min << ")\n" << std::scientific
          << "  sigma:          " << std::setprecision(3) << sigma << "\n"
          << "  gamma:          " << gamma_ch << "\n"
          << "  nu:             " << nu << "  (nu_eff=" << nu_eff << ")\n";
    if (benchmark == "droplet")
        pcout << "  radius:         " << radius
              << "  (Young-Laplace: dp=" << expected_p_jump << ")\n";
    pcout << "  Output:         " << output_dir << "\n"
          << "  Initial mass:   " << std::scientific << std::setprecision(6)
          << initial_mass << "\n"
          << "  Initial energy: " << initial_energy << "\n"
          << "================================================================\n\n";

    pcout << "  Step   Time         U_max        p_jump       phi_mass     "
          << "E_free       Wall(s)\n"
          << "  ----   ----------   ----------   ----------   ----------   "
          << "----------   -------\n";

    // ================================================================
    // 6. Open CSV
    // ================================================================
    std::ofstream csv_file;
    if (rank == 0)
    {
        csv_file.open(output_dir + "/diagnostics.csv");
        csv_file << "step,time,dt,"
                 << "phi_min,phi_max,phi_mass,free_energy,"
                 << "mu_min,mu_max,mu_L2,"
                 << "ux_min,ux_max,uy_min,uy_max,U_max,E_kin,divU_L2,"
                 << "p_min,p_max,p_jump,"
                 << "ch_iters,ns_iters,wall_s\n";
    }

    // ================================================================
    // 7. Time stepping
    // ================================================================
    auto wall_start = std::chrono::high_resolution_clock::now();
    double current_time = 0.0;
    unsigned int step = 0;
    bool energy_monotone = true;
    double prev_energy = initial_energy;

    while (current_time < t_final - 1e-14*dt && step < max_steps)
    {
        current_time += dt;
        ++step;

        Timer step_timer;
        step_timer.start();

        // ============================================================
        // Step 1: Solve NS with capillary force from old CH
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/true,
                    w_dummy, w_dof_handler,
                    /*Mx=*/{}, /*My=*/{}, /*M_dh=*/nullptr,
                    /*phi=*/{}, /*phi_dh=*/nullptr,
                    ch.get_old_relevant(),
                    &ch.get_dof_handler());

        auto ns_info = ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Step 2: Solve CH with convection from new velocity
        // ============================================================
        ch.assemble(ch.get_old_relevant(), dt,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler());

        auto ch_info = ch.solve();
        ch.update_ghosts();
        ch.save_old_solution();

        // ============================================================
        // Advance old velocity
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();

        step_timer.stop();
        const double step_wall = step_timer.wall_time();

        // ============================================================
        // Diagnostics
        // ============================================================
        auto ns_diag = ns.compute_diagnostics();
        auto ch_diag = ch.compute_diagnostics();

        const double p_jump = ns_diag.p_max - ns_diag.p_min;

        if (ch_diag.free_energy > prev_energy + 1e-12)
            energy_monotone = false;
        prev_energy = ch_diag.free_energy;

        // ============================================================
        // CSV output (every step)
        // ============================================================
        if (rank == 0)
        {
            csv_file << step << ","
                     << std::scientific << std::setprecision(8)
                     << current_time << "," << dt << ","
                     << ch_diag.phi_min << "," << ch_diag.phi_max << ","
                     << ch_diag.phi_mass << "," << ch_diag.free_energy << ","
                     << ch_diag.mu_min << "," << ch_diag.mu_max << ","
                     << ch_diag.mu_L2 << ","
                     << ns_diag.ux_min << "," << ns_diag.ux_max << ","
                     << ns_diag.uy_min << "," << ns_diag.uy_max << ","
                     << ns_diag.U_max << "," << ns_diag.E_kin << ","
                     << ns_diag.divU_L2 << ","
                     << ns_diag.p_min << "," << ns_diag.p_max << ","
                     << p_jump << ","
                     << ch_info.iterations << "," << ns_info.iterations << ","
                     << std::fixed << std::setprecision(2) << step_wall
                     << "\n";
            csv_file << std::flush;
        }

        // ============================================================
        // Console output (every step or every vtk_interval)
        // ============================================================
        if (step % vtk_interval == 0 || step <= 5 ||
            current_time >= t_final - 1e-14*dt)
        {
            pcout << "  " << std::setw(4) << step
                  << "   " << std::scientific << std::setprecision(4)
                  << current_time
                  << "   " << std::setprecision(2) << ns_diag.U_max
                  << "   " << p_jump
                  << "   " << ch_diag.phi_mass
                  << "   " << ch_diag.free_energy
                  << "   " << std::fixed << std::setprecision(1)
                  << step_wall << "\n";
        }

        // ============================================================
        // VTK output
        // ============================================================
        if (step % vtk_interval == 0 || current_time >= t_final - 1e-14*dt)
        {
            write_vtu(step, current_time);

            if (verbose)
                pcout << "         → VTK output (step " << step << ")\n";
        }
    }

    // ================================================================
    // 8. Summary
    // ================================================================
    if (rank == 0 && csv_file.is_open())
        csv_file.close();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();

    auto ch_diag_final = ch.compute_diagnostics();
    auto ns_diag_final = ns.compute_diagnostics();
    const double final_mass = ch_diag_final.phi_mass;
    const double final_energy = ch_diag_final.free_energy;
    const double final_p_jump = ns_diag_final.p_max - ns_diag_final.p_min;
    const double mass_rel_change = (std::abs(initial_mass) > 1e-15)
        ? std::abs(final_mass - initial_mass) / std::abs(initial_mass) : 0.0;

    pcout << "\n"
          << "================================================================\n"
          << "  BENCHMARK COMPLETE: "
          << (benchmark == "droplet" ? "Circular Droplet" : "Square Relaxation")
          << "\n"
          << "================================================================\n"
          << "  Steps:          " << step << "\n"
          << "  Final time:     " << std::scientific << std::setprecision(4)
          << current_time << "\n"
          << "  Wall time:      " << std::fixed << std::setprecision(1)
          << wall_time << " s\n"
          << "\n"
          << "  Mass conservation:\n"
          << "    Initial:      " << std::scientific << std::setprecision(6)
          << initial_mass << "\n"
          << "    Final:        " << final_mass << "\n"
          << "    Rel. change:  " << mass_rel_change << "\n"
          << "\n"
          << "  Energy:\n"
          << "    Initial:      " << initial_energy << "\n"
          << "    Final:        " << final_energy << "\n"
          << "    Monotone:     " << (energy_monotone ? "YES" : "NO") << "\n"
          << "\n";

    if (benchmark == "droplet")
    {
        const double p_err = (expected_p_jump > 0)
            ? std::abs(final_p_jump - expected_p_jump) / expected_p_jump : 0.0;

        pcout << "  Young-Laplace pressure jump:\n"
              << "    Computed:     " << final_p_jump << "\n"
              << "    Expected:     " << expected_p_jump
              << "  (sigma/R = " << sigma << "/" << radius << ")\n"
              << "    Rel. error:   " << p_err << "\n"
              << "\n";
    }

    pcout << "  Final velocity:\n"
          << "    U_max:        " << ns_diag_final.U_max << "\n"
          << "    E_kin:        " << ns_diag_final.E_kin << "\n"
          << "\n"
          << "  Phase field:\n"
          << "    phi_min:      " << ch_diag_final.phi_min << "\n"
          << "    phi_max:      " << ch_diag_final.phi_max << "\n"
          << "\n"
          << "  Output:         " << output_dir << "\n"
          << "================================================================\n\n";

    return 0;
}
