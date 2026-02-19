// ============================================================================
// poisson/poisson_main.cc — Standalone Poisson Driver
//
// Modes:
//   mms       MMS spatial convergence (2D), refs 2-6
//   2d        2D dipole field solve with VTK output
//   3d        3D MMS solve with VTK output
//   temporal  N/A — Poisson is time-independent
//
// Usage:
//   mpirun -np 4 ./poisson_main --mode mms
//   mpirun -np 4 ./poisson_main --mode 2d --refinement 4
//   mpirun -np 4 ./poisson_main --mode 3d --refinement 2
//   mpirun -np 4 ./poisson_main --mode temporal
//   mpirun -np 4 ./poisson_main --ref 2 3 4 5
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42d
// ============================================================================

#include "poisson/poisson.h"
#include "poisson/tests/poisson_mms.h"
#include "physics/applied_field.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_dgq.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace dealii;


// ============================================================================
// Single MMS refinement (reusable for mms loop)
// ============================================================================
struct PoissonMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;
    double L2 = 0.0, H1 = 0.0, Linf = 0.0;
    int iterations = 0;
    double walltime = 0.0;
};

template <int dim>
PoissonMMSResult run_poisson_mms_single(
    unsigned int refinement,
    const Parameters& params,
    MPI_Comm mpi_comm,
    double L_y,
    double L_z = 1.0)
{
    PoissonMMSResult result;
    result.refinement = refinement;
    auto wall_start = std::chrono::high_resolution_clock::now();

    const double mms_time = 1.0;

    // Mesh
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1, p2;
    std::vector<unsigned int> subs(dim);
    p1[0] = params.domain.x_min;
    p2[0] = params.domain.x_max;
    subs[0] = params.domain.initial_cells_x;
    if constexpr (dim >= 2)
    {
        p1[1] = params.domain.y_min;
        p2[1] = params.domain.y_max;
        subs[1] = params.domain.initial_cells_y;
    }
    if constexpr (dim >= 3)
    {
        p1[2] = 0.0;
        p2[2] = L_z;
        subs[2] = params.domain.initial_cells_x;
    }

    GridGenerator::subdivided_hyper_rectangle(triangulation, subs, p1, p2);
    triangulation.refine_global(refinement);

    // Compute min h
    double local_h_min = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h_min = std::min(local_h_min, cell->diameter());
    MPI_Allreduce(&local_h_min, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);

    // Setup
    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.enable_magnetic = false;

    PoissonSubsystem<dim> poisson(mms_params, mpi_comm, triangulation);
    poisson.setup();
    result.n_dofs = poisson.get_dof_handler().n_dofs();

    // MMS source
    poisson.set_mms_source(
        [L_y, L_z, mms_time](const Point<dim>& p, double /*t*/) -> double {
            return compute_poisson_mms_source_standalone<dim>(p, mms_time, L_y, L_z);
        });

    // Solve
    TrilinosWrappers::MPI::Vector empty_Mx, empty_My;
    FE_DGQ<dim> dummy_fe(0);
    DoFHandler<dim> dummy_dof(triangulation);
    dummy_dof.distribute_dofs(dummy_fe);

    poisson.assemble_rhs(empty_Mx, empty_My, dummy_dof, mms_time);
    SolverInfo info = poisson.solve();
    poisson.update_ghosts();
    result.iterations = info.iterations;

    // Errors
    PoissonMMSErrors errors = compute_poisson_mms_errors<dim>(
        poisson.get_dof_handler(),
        poisson.get_solution_relevant(),
        mms_time, L_y, mpi_comm, L_z);

    result.L2 = errors.L2;
    result.H1 = errors.H1;
    result.Linf = errors.Linf;

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    return result;
}


// ============================================================================
// 2D dipole mode
// ============================================================================
void run_2d_dipole(const Parameters& params, MPI_Comm mpi_comm)
{
    constexpr int dim = 2;
    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    Parameters p = params;
    p.setup_rosensweig();
    p.enable_mms = false;
    p.enable_magnetic = true;
    if (params.mesh.initial_refinement != 0)
        p.mesh.initial_refinement = params.mesh.initial_refinement;

    const double run_time = p.dipoles.ramp_time;

    pcout << "\n================================================================\n"
          << "  POISSON STANDALONE: 2D Dipole Field\n"
          << "================================================================\n"
          << "  Refinement:     " << p.mesh.initial_refinement << "\n"
          << "  Dipole(s):      " << p.dipoles.positions.size() << "\n"
          << "================================================================\n\n";

    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1(p.domain.x_min, p.domain.y_min);
    Point<dim> p2(p.domain.x_max, p.domain.y_max);
    std::vector<unsigned int> subs = {p.domain.initial_cells_x, p.domain.initial_cells_y};
    GridGenerator::subdivided_hyper_rectangle(triangulation, subs, p1, p2);
    triangulation.refine_global(p.mesh.initial_refinement);

    PoissonSubsystem<dim> poisson(p, mpi_comm, triangulation);
    poisson.setup();

    pcout << "  DoFs: " << poisson.get_dof_handler().n_dofs() << "\n";

    FE_DGQ<dim> dummy_fe(0);
    DoFHandler<dim> dummy_dof(triangulation);
    dummy_dof.distribute_dofs(dummy_fe);
    TrilinosWrappers::MPI::Vector empty_Mx, empty_My;

    poisson.assemble_rhs(empty_Mx, empty_My, dummy_dof, run_time);
    SolverInfo info = poisson.solve();
    poisson.update_ghosts();

    pcout << "  Solve: " << info.iterations << " iterations\n";

    const std::string out_dir = "../poisson_results/vtk/"
        + timestamped_filename_mpi("poisson_2d_dipole", "", mpi_comm);
    poisson.write_vtu(out_dir, 0, run_time);
    pcout << "  VTK: " << out_dir << "\n\n";
}


// ============================================================================
// 3D solve with VTK output
// ============================================================================
void run_3d_vtk(const Parameters& params, MPI_Comm mpi_comm)
{
    constexpr int dim = 3;
    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    const double L_y = params.domain.y_max - params.domain.y_min;
    const double L_z = 1.0;
    const double mms_time = 1.0;
    const unsigned int ref = params.mesh.initial_refinement;

    pcout << "\n================================================================\n"
          << "  POISSON STANDALONE: 3D Solve with VTK\n"
          << "================================================================\n"
          << "  Refinement:     " << ref << "\n"
          << "  Domain:         [0,1] x [0," << L_y << "] x [0," << L_z << "]\n"
          << "================================================================\n\n";

    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    Point<dim> p1(0.0, 0.0, 0.0);
    Point<dim> p2(1.0, L_y, L_z);
    std::vector<unsigned int> subs = {10, 6, 10};
    GridGenerator::subdivided_hyper_rectangle(triangulation, subs, p1, p2);
    triangulation.refine_global(ref);

    Parameters mms_params = params;
    mms_params.enable_mms = true;
    mms_params.enable_magnetic = false;

    PoissonSubsystem<dim> poisson(mms_params, mpi_comm, triangulation);
    poisson.setup();

    pcout << "  DoFs: " << poisson.get_dof_handler().n_dofs() << "\n";

    poisson.set_mms_source(
        [L_y, L_z, mms_time](const Point<dim>& p, double /*t*/) -> double {
            return compute_poisson_mms_source_standalone<dim>(p, mms_time, L_y, L_z);
        });

    FE_DGQ<dim> dummy_fe(0);
    DoFHandler<dim> dummy_dof(triangulation);
    dummy_dof.distribute_dofs(dummy_fe);
    TrilinosWrappers::MPI::Vector empty_Mx, empty_My;

    poisson.assemble_rhs(empty_Mx, empty_My, dummy_dof, mms_time);
    SolverInfo info = poisson.solve();
    poisson.update_ghosts();

    pcout << "  Solve: " << info.iterations << " iterations\n";

    const std::string out_dir = "../poisson_results/vtk/"
        + timestamped_filename_mpi("poisson_3d", "", mpi_comm);
    poisson.write_vtu(out_dir, 0, mms_time);
    pcout << "  VTK: " << out_dir << "\n\n";
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    try
    {
        Parameters params = Parameters::parse_command_line(argc, argv);
        const std::string& mode = params.run.mode;

        // ================================================================
        // Mode: mms — spatial convergence study (2D)
        // ================================================================
        if (mode == "mms")
        {
            constexpr int dim = 2;
            params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
            params.domain.y_min = 0.0;  params.domain.y_max = 0.6;
            params.domain.initial_cells_x = 10;
            params.domain.initial_cells_y = 6;
            params.fe.degree_potential = 1;

            const double L_y = params.domain.y_max - params.domain.y_min;
            const unsigned int fe_degree = params.fe.degree_potential;

            pcout << "\n================================================================\n"
                  << "  POISSON MMS CONVERGENCE (Q" << fe_degree << ")\n"
                  << "================================================================\n"
                  << "  Refinements: ";
            for (auto r : params.run.refs) pcout << r << " ";
            pcout << "\n================================================================\n\n";

            std::vector<PoissonMMSResult> results;
            for (unsigned int ref : params.run.refs)
            {
                pcout << "  Refinement " << ref << "... " << std::flush;
                PoissonMMSResult r = run_poisson_mms_single<dim>(ref, params, mpi_comm, L_y);
                results.push_back(r);
                pcout << "L2=" << std::scientific << std::setprecision(2) << r.L2
                      << ", H1=" << r.H1
                      << ", its=" << r.iterations
                      << ", wall=" << std::fixed << std::setprecision(2) << r.walltime << "s\n";
            }

            bool pass = false;
            if (rank == 0 && results.size() >= 2)
            {
                pcout << "\n  Convergence rates:\n";
                for (size_t i = 1; i < results.size(); ++i)
                {
                    double log_h = std::log(results[i-1].h / results[i].h);
                    double L2_rate = std::log(results[i-1].L2 / results[i].L2) / log_h;
                    double H1_rate = std::log(results[i-1].H1 / results[i].H1) / log_h;
                    pcout << "    ref " << results[i].refinement
                          << ": L2_rate=" << std::fixed << std::setprecision(2) << L2_rate
                          << ", H1_rate=" << H1_rate << "\n";
                    if (i == results.size() - 1)
                        pass = (L2_rate >= fe_degree + 1 - 0.3)
                            && (H1_rate >= fe_degree - 0.3);
                }
                if (pass) pcout << "  [PASS]\n";
                else      pcout << "  [FAIL]\n";
            }
            return pass ? 0 : 1;
        }
        // ================================================================
        // Mode: 2d — dipole field with VTK
        // ================================================================
        else if (mode == "2d")
        {
            run_2d_dipole(params, mpi_comm);
            return 0;
        }
        // ================================================================
        // Mode: 3d — 3D solve with VTK output
        // ================================================================
        else if (mode == "3d")
        {
            if (params.mesh.initial_refinement == 0)
                params.mesh.initial_refinement = 2;
            params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
            params.domain.y_min = 0.0;  params.domain.y_max = 0.6;

            run_3d_vtk(params, mpi_comm);
            return 0;
        }
        // ================================================================
        // Mode: temporal — N/A for Poisson
        // ================================================================
        else if (mode == "temporal")
        {
            pcout << "\n================================================================\n"
                  << "  Poisson is time-independent — no temporal convergence test.\n"
                  << "================================================================\n\n";
            return 0;
        }
        else
        {
            pcout << "Unknown mode: " << mode
                  << " (use mms, 2d, 3d, or temporal)\n";
            return 1;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "\n[Error] " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n[Error] Unknown exception!\n";
        return 1;
    }
}
