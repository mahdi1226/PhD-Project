// ============================================================================
// cahn_hilliard/tests/cahn_hilliard_mms_test.cc - MMS Convergence Study
//
// Uses CahnHilliardSubsystem facade — all setup, assembly, and solve
// are encapsulated. The test only provides:
//   1. Mesh creation
//   2. MMS source terms (via set_mms_source callback)
//   3. Dirichlet BCs (via apply_dirichlet_boundary)
//   4. Time-stepping loop
//   5. Error computation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42b
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "cahn_hilliard/tests/cahn_hilliard_mms.h"
#include "utilities/parameters.h"
#include "utilities/timestamp.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

// ============================================================================
// Result structures
// ============================================================================
struct CHMMSResult
{
    unsigned int refinement = 0;
    double h = 0.0;
    unsigned int n_dofs = 0;
    double theta_L2   = 0.0;
    double theta_H1   = 0.0;
    double theta_Linf = 0.0;
    double psi_L2     = 0.0;
    double psi_Linf   = 0.0;
    double total_time  = 0.0;
};

struct CHMMSConvergenceResult
{
    std::vector<CHMMSResult> results;
    std::vector<double> theta_L2_rates;
    std::vector<double> theta_H1_rates;
    std::vector<double> theta_Linf_rates;
    std::vector<double> psi_L2_rates;
    std::vector<double> psi_Linf_rates;
    unsigned int fe_degree = 2;
    unsigned int n_time_steps = 10;

    void compute_rates()
    {
        theta_L2_rates.clear();
        theta_H1_rates.clear();
        theta_Linf_rates.clear();
        psi_L2_rates.clear();
        psi_Linf_rates.clear();
        for (size_t i = 1; i < results.size(); ++i)
        {
            const double log_h = std::log(results[i-1].h / results[i].h);
            auto rate = [&](double e_coarse, double e_fine) {
                return (e_coarse > 1e-15 && e_fine > 1e-15)
                    ? std::log(e_coarse / e_fine) / log_h : 0.0;
            };
            theta_L2_rates.push_back(rate(results[i-1].theta_L2, results[i].theta_L2));
            theta_H1_rates.push_back(rate(results[i-1].theta_H1, results[i].theta_H1));
            theta_Linf_rates.push_back(rate(results[i-1].theta_Linf, results[i].theta_Linf));
            psi_L2_rates.push_back(rate(results[i-1].psi_L2, results[i].psi_L2));
            psi_Linf_rates.push_back(rate(results[i-1].psi_Linf, results[i].psi_Linf));
        }
    }

    void print() const
    {
        std::cout << "\n--- CH MMS Convergence (CG Q" << fe_degree << ") ---\n";
        std::cout << std::left
                  << std::setw(5) << "Ref"
                  << std::setw(12) << "h"
                  << std::setw(12) << "θ_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "θ_H1"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "θ_Linf"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ψ_L2"
                  << std::setw(8)  << "rate"
                  << std::setw(12) << "ψ_Linf"
                  << std::setw(8)  << "rate"
                  << std::setw(10) << "wall(s)"
                  << "\n";
        std::cout << std::string(127, '-') << "\n";

        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            std::cout << std::left << std::setw(5) << r.refinement
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.h
                      << std::setw(12) << r.theta_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? theta_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.theta_H1
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? theta_H1_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.theta_Linf
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? theta_Linf_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.psi_L2
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? psi_L2_rates[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << r.psi_Linf
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? psi_Linf_rates[i-1] : 0.0)
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.total_time
                      << "\n";
        }
    }

    void write_csv(const std::string& filepath) const
    {
        std::ofstream f(filepath);
        f << "refinement,h,n_dofs,theta_L2,theta_L2_rate,theta_H1,theta_H1_rate,"
          << "theta_Linf,theta_Linf_rate,psi_L2,psi_L2_rate,"
          << "psi_Linf,psi_Linf_rate,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            f << r.refinement << ","
              << std::scientific << std::setprecision(6) << r.h << ","
              << r.n_dofs << ","
              << r.theta_L2 << ","
              << std::fixed << std::setprecision(3)
              << (i > 0 ? theta_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.theta_H1 << ","
              << std::fixed << std::setprecision(3)
              << (i > 0 ? theta_H1_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.theta_Linf << ","
              << std::fixed << std::setprecision(3)
              << (i > 0 ? theta_Linf_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.psi_L2 << ","
              << std::fixed << std::setprecision(3)
              << (i > 0 ? psi_L2_rates[i-1] : 0.0) << ","
              << std::scientific << std::setprecision(6) << r.psi_Linf << ","
              << std::fixed << std::setprecision(3)
              << (i > 0 ? psi_Linf_rates[i-1] : 0.0) << ","
              << std::fixed << std::setprecision(4) << r.total_time << "\n";
        }
        std::cout << "  CSV written: " << filepath << "\n";
    }

    bool passes(double tol = 0.3) const
    {
        if (theta_L2_rates.empty()) return false;
        const double expected_L2 = fe_degree + 1;
        const double expected_H1 = fe_degree;
        return (theta_L2_rates.back() >= expected_L2 - tol)
            && (theta_H1_rates.back() >= expected_H1 - tol);
    }
};


// ============================================================================
// Helper: build domain lengths array from Parameters
// ============================================================================
template <int dim>
void fill_domain_lengths(const Parameters& params, double L[dim])
{
    L[0] = params.domain.x_max - params.domain.x_min;
    if constexpr (dim >= 2) L[1] = params.domain.y_max - params.domain.y_min;
    if constexpr (dim >= 3) L[2] = 1.0;  // Default z-extent for 3D
}


// ============================================================================
// Single refinement test
// ============================================================================
template <int dim>
CHMMSResult run_ch_mms_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_comm)
{
    CHMMSResult result;
    result.refinement = refinement;

    auto total_start = std::chrono::high_resolution_clock::now();

    const double t_init  = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;

    double L[dim];
    fill_domain_lengths<dim>(params, L);

    // ========================================================================
    // Create distributed mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1, p2;
    std::vector<unsigned int> subdivisions(dim);

    p1[0] = params.domain.x_min;
    p2[0] = params.domain.x_max;
    subdivisions[0] = params.domain.initial_cells_x;

    if constexpr (dim >= 2)
    {
        p1[1] = params.domain.y_min;
        p2[1] = params.domain.y_max;
        subdivisions[1] = params.domain.initial_cells_y;
    }
    if constexpr (dim >= 3)
    {
        p1[2] = 0.0;
        p2[2] = L[2];
        subdivisions[2] = params.domain.initial_cells_x;  // same as x
    }

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                       subdivisions, p1, p2);
    triangulation.refine_global(refinement);

    // ========================================================================
    // Create facade and setup
    // ========================================================================
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);
    ch.setup();

    result.n_dofs = ch.get_theta_dof_handler().n_dofs() * 2;

    // ========================================================================
    // Create dummy velocity (zero — standalone CH, no convection)
    // ========================================================================
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::DoFHandler<dim> vel_dof_handler(triangulation);
    vel_dof_handler.distribute_dofs(fe_vel);

    dealii::IndexSet vel_owned = vel_dof_handler.locally_owned_dofs();
    dealii::IndexSet vel_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(vel_dof_handler);

    // One zero-velocity vector per spatial dimension
    std::vector<dealii::TrilinosWrappers::MPI::Vector> vel_vectors(dim);
    std::vector<const dealii::TrilinosWrappers::MPI::Vector*> vel_ptrs(dim);
    for (unsigned int d = 0; d < dim; ++d)
    {
        vel_vectors[d].reinit(vel_owned, vel_relevant, mpi_comm);
        vel_vectors[d] = 0;
        vel_ptrs[d] = &vel_vectors[d];
    }

    // ========================================================================
    // Inject MMS source terms
    // ========================================================================
    CHSourceTheta<dim> src_theta(params.physics.mobility, dt, L);
    CHSourcePsi<dim>   src_psi(params.physics.epsilon, dt, L);

    ch.set_mms_source(
        [&](const dealii::Point<dim>& p, double t) -> double {
            src_theta.set_time(t);
            return src_theta.value(p);
        },
        [&](const dealii::Point<dim>& p, double t) -> double {
            src_psi.set_time(t);
            return src_psi.value(p);
        });

    // ========================================================================
    // Project exact initial condition at t_init
    // ========================================================================
    CHMMSInitialTheta<dim> theta_ic(t_init, L);
    CHMMSInitialPsi<dim>   psi_ic(t_init, L);
    ch.project_initial_condition(theta_ic, psi_ic);

    // Apply initial Dirichlet BCs
    CHMMSBoundaryTheta<dim> theta_bc(L);
    CHMMSBoundaryPsi<dim>   psi_bc(L);
    theta_bc.set_time(t_init);
    psi_bc.set_time(t_init);
    ch.apply_dirichlet_boundary(theta_bc, psi_bc);

    // ========================================================================
    // Time stepping loop
    // ========================================================================
    ch.update_ghosts();  // needed for theta_old_relevant

    double current_time = t_init;

    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;

        // Update Dirichlet BCs to current time
        theta_bc.set_time(current_time);
        psi_bc.set_time(current_time);
        ch.apply_dirichlet_boundary(theta_bc, psi_bc);

        // Assemble with θ^{n-1} (ghosted) and zero velocity
        ch.assemble(ch.get_theta_relevant(),
                    vel_ptrs, vel_dof_handler,
                    dt, current_time);

        // Solve coupled system → updates θ, ψ (owned)
        ch.solve();

        // Update ghosts for next step (θ^{n-1} = θ^n)
        ch.update_ghosts();
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    CHMMSErrors errors = compute_ch_mms_errors<dim>(
        ch.get_theta_dof_handler(),
        ch.get_psi_dof_handler(),
        ch.get_theta_relevant(),
        ch.get_psi_relevant(),
        current_time, L, mpi_comm);

    result.theta_L2   = errors.theta_L2;
    result.theta_H1   = errors.theta_H1;
    result.theta_Linf = errors.theta_Linf;
    result.psi_L2     = errors.psi_L2;
    result.psi_Linf   = errors.psi_Linf;

    // Compute minimum h
    double local_min_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    MPI_Allreduce(&local_min_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}


// ============================================================================
// Main: convergence study
// ============================================================================
int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

    // Parse parameters (uses Rosensweig defaults)
    Parameters params = Parameters::parse_command_line(argc, argv);

    // Refinement levels and time steps
    std::vector<unsigned int> refinements = {2, 3, 4, 5, 6};
    const unsigned int n_time_steps = 10;

    constexpr int dim = 2;

    CHMMSConvergenceResult conv;
    conv.fe_degree = params.fe.degree_phase;
    conv.n_time_steps = n_time_steps;

    if (rank == 0)
    {
        std::cout << "\n[CH MMS] Cahn-Hilliard MMS Convergence Study\n";
        std::cout << "  MPI ranks: " << n_ranks << "\n";
        std::cout << "  FE degree: CG Q" << params.fe.degree_phase << "\n";
        std::cout << "  ε = " << params.physics.epsilon
                  << ", γ = " << params.physics.mobility << "\n";
        std::cout << "  Time steps: " << n_time_steps
                  << ", t ∈ [0.1, 0.2]\n";
        std::cout << "  Using CahnHilliardSubsystem facade\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (rank == 0)
            std::cout << "  Refinement " << ref << "... " << std::flush;

        CHMMSResult r = run_ch_mms_single<dim>(ref, params, n_time_steps, mpi_comm);
        conv.results.push_back(r);

        if (rank == 0)
        {
            std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << r.theta_L2
                      << ", θ_H1=" << r.theta_H1
                      << ", θ_Linf=" << r.theta_Linf
                      << ", ψ_L2=" << r.psi_L2
                      << ", ψ_Linf=" << r.psi_Linf
                      << ", time=" << std::fixed << std::setprecision(1)
                      << r.total_time << "s\n";
        }
    }

    conv.compute_rates();

    if (rank == 0)
    {
        conv.print();

        // Write CSV to cahn_hilliard_results/mms/ (relative to build → ../cahn_hilliard_results/mms)
        const std::string out_dir = "../cahn_hilliard_results/mms";
        std::system(("mkdir -p " + out_dir).c_str());

        const std::string csv_name = timestamped_filename(
            "ch_mms_convergence", ".csv");
        const std::string csv_path = out_dir + "/" + csv_name;
        conv.write_csv(csv_path);

        // Summary
        double total_wall = 0.0;
        for (const auto& r : conv.results) total_wall += r.total_time;

        std::cout << "\nExpected: θ_L2 ~ O(h^" << (params.fe.degree_phase + 1)
                  << "), θ_H1 ~ O(h^" << params.fe.degree_phase << ")"
                  << "  |  Total wall time: " << std::fixed << std::setprecision(1)
                  << total_wall << "s\n";

        if (conv.passes())
            std::cout << "[PASS] Convergence rates within tolerance!\n";
        else
            std::cout << "[FAIL] Convergence rates below expected!\n";
    }

    return conv.passes() ? 0 : 1;
}