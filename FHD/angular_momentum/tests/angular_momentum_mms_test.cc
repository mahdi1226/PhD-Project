// ============================================================================
// angular_momentum/tests/angular_momentum_mms_test.cc - MMS Convergence Test
//
// PAPER EQUATION 42f (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//   Standalone angular momentum: reaction-diffusion (no coupling)
//
// Tests: AngularMomentumSubsystem facade using PRODUCTION code paths
//   setup() → assemble(standalone) → solve() → compute errors
//
// Expected convergence (CG Q2):
//   w_L2: O(h³) — rate ≈ 3.0
//   w_H1: O(h²) — rate ≈ 2.0
//
// Usage:
//   mpirun -np 2 ./test_angular_momentum_mms
//   mpirun -np 4 ./test_angular_momentum_mms --refs 2 3 4 5
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================

#include "angular_momentum/angular_momentum.h"
#include "angular_momentum/tests/angular_momentum_mms.h"
#include "mesh/mesh.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

constexpr int dim = 2;

struct MMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;
    double w_L2 = 0.0;
    double w_H1 = 0.0;
    double w_Linf = 0.0;
    int iterations = 0;
    double walltime = 0.0;
};

// ============================================================================
// Exact solution as deal.II Function for interpolation
// ============================================================================
class ExactW : public dealii::Function<dim>
{
public:
    ExactW(double time) : dealii::Function<dim>(1), time_(time) {}

    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    {
        return angular_momentum_exact<dim>(p, time_);
    }
private:
    double time_;
};

MMSResult run_single(unsigned int refinement,
                     const Parameters& base_params,
                     MPI_Comm mpi_comm)
{
    MMSResult result;
    result.refinement = refinement;

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ================================================================
    // Create distributed mesh
    // ================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    Parameters params = base_params;
    params.mesh.initial_refinement = refinement;
    params.enable_mms = true;

    FHDMesh::create_mesh<dim>(triangulation, params);

    result.h = dealii::GridTools::minimal_cell_diameter(triangulation);

    // ================================================================
    // Time-step: τ = h² (to keep temporal error sub-dominant)
    // ================================================================
    const double h = result.h;
    const double dt = h * h;
    const double t_old = 1.0;
    const double t_new = t_old + dt;

    // ================================================================
    // Create and setup Angular Momentum subsystem (PRODUCTION CODE)
    // ================================================================
    AngularMomentumSubsystem<dim> am(params, mpi_comm, triangulation);
    am.setup();

    result.n_dofs = am.get_dof_handler().n_dofs();

    // ================================================================
    // Inject MMS source
    // ================================================================
    am.set_mms_source(compute_angular_mms_source_standalone<dim>);

    // ================================================================
    // Project initial condition w*(t_old) onto CG space
    // ================================================================
    dealii::TrilinosWrappers::MPI::Vector w_old_owned(
        am.get_dof_handler().locally_owned_dofs(), mpi_comm);

    ExactW exact_w(t_old);
    dealii::VectorTools::interpolate(
        am.get_dof_handler(), exact_w, w_old_owned);

    // Create relevant (ghosted) vector for assembly
    dealii::IndexSet locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(am.get_dof_handler());
    dealii::TrilinosWrappers::MPI::Vector w_old_relevant(
        am.get_dof_handler().locally_owned_dofs(),
        locally_relevant, mpi_comm);
    w_old_relevant = w_old_owned;

    // ================================================================
    // Assemble standalone (no velocity coupling)
    // ================================================================
    dealii::TrilinosWrappers::MPI::Vector empty_ux, empty_uy;

    am.assemble(w_old_relevant,
                dt, t_new,
                empty_ux, empty_uy,
                am.get_dof_handler(),  // dummy — not used since has_vel=false
                /*include_convection=*/false);

    // ================================================================
    // Solve (PRODUCTION CODE)
    // ================================================================
    SolverInfo info = am.solve();
    result.iterations = info.iterations;

    // ================================================================
    // Compute errors at t_new
    // ================================================================
    am.update_ghosts();

    AngularMomentumMMSErrors errors = compute_angular_mms_errors<dim>(
        am.get_dof_handler(),
        am.get_relevant(),
        t_new, mpi_comm);

    result.w_L2 = errors.w_L2;
    result.w_H1 = errors.w_H1;
    result.w_Linf = errors.w_Linf;

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.walltime = std::chrono::duration<double>(wall_end - wall_start).count();

    return result;
}

int main(int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
    dealii::ConditionalOStream pcout(std::cout, rank == 0);

    // Parse refinement levels
    std::vector<unsigned int> refinements = {2, 3, 4, 5, 6};

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--refs") == 0)
        {
            refinements.clear();
            for (int j = i + 1; j < argc; ++j)
            {
                if (argv[j][0] == '-') break;
                refinements.push_back(std::stoul(argv[j]));
            }
            break;
        }
    }

    Parameters params;
    params.setup_mms_validation();

    const unsigned int degree = params.fe.degree_angular;

    pcout << "\n"
          << "================================================================\n"
          << "  ANGULAR MOMENTUM MMS CONVERGENCE (Nochetto Eq. 42f)\n"
          << "================================================================\n"
          << "  MPI ranks:      " << n_ranks << "\n"
          << "  FE space:       CG Q" << degree << "\n"
          << "  Expected rates: L2 = " << degree + 1
          << ", H1 = " << degree << "\n"
          << "  Domain:         [0,1]^2\n"
          << "  Standalone:     no coupling, reaction-diffusion\n"
          << "  Time step:      τ = h² (single backward Euler step)\n"
          << "  Refinements:    ";
    for (auto r : refinements) pcout << r << " ";
    pcout << "\n"
          << "================================================================\n\n";

    std::vector<MMSResult> results;

    for (unsigned int ref : refinements)
    {
        pcout << "  Refinement " << ref << "... " << std::flush;

        MMSResult r = run_single(ref, params, mpi_comm);
        results.push_back(r);

        pcout << "DoFs=" << r.n_dofs
              << ", L2=" << std::scientific << std::setprecision(2) << r.w_L2
              << ", H1=" << r.w_H1
              << ", wall=" << std::fixed << std::setprecision(2)
              << r.walltime << "s\n";
    }

    // Compute rates
    std::vector<double> L2_rates, H1_rates;
    for (size_t i = 1; i < results.size(); ++i)
    {
        auto rate = [](double ef, double ec, double hf, double hc) -> double {
            if (ef < 1e-15 || ec < 1e-15 || hf >= hc) return 0.0;
            return std::log(ec / ef) / std::log(hc / hf);
        };
        L2_rates.push_back(rate(results[i].w_L2, results[i-1].w_L2,
                                 results[i].h, results[i-1].h));
        H1_rates.push_back(rate(results[i].w_H1, results[i-1].w_H1,
                                 results[i].h, results[i-1].h));
    }

    // Print table
    pcout << "\n"
          << std::left
          << std::setw(5)  << "Ref"
          << std::setw(10) << "DoFs"
          << std::setw(12) << "h"
          << std::setw(12) << "w_L2"
          << std::setw(8)  << "rate"
          << std::setw(12) << "w_H1"
          << std::setw(8)  << "rate"
          << "\n"
          << std::string(67, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto& r = results[i];
        pcout << std::left
              << std::setw(5)  << r.refinement
              << std::setw(10) << r.n_dofs
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.h
              << std::setw(12) << r.w_L2
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? L2_rates[i-1] : 0.0)
              << std::scientific << std::setprecision(2)
              << std::setw(12) << r.w_H1
              << std::fixed << std::setprecision(2)
              << std::setw(8)  << (i > 0 ? H1_rates[i-1] : 0.0)
              << "\n";
    }

    // Pass/fail
    const double tolerance = 0.3;
    bool pass = false;

    if (!L2_rates.empty())
    {
        const double final_L2 = L2_rates.back();
        const double final_H1 = H1_rates.back();

        const double expected_L2 = degree + 1;
        const double expected_H1 = degree;

        pass = (final_L2 >= expected_L2 - tolerance) &&
               (final_H1 >= expected_H1 - tolerance);

        pcout << "\n"
              << "================================================================\n"
              << "  SUMMARY\n"
              << "================================================================\n"
              << "  Asymptotic rates (finest pair):\n"
              << "    w_L2: " << std::fixed << std::setprecision(2)
              << final_L2 << "  (expected " << expected_L2 << ")\n"
              << "    w_H1: " << final_H1 << "  (expected " << expected_H1 << ")\n"
              << "\n"
              << "  STATUS: " << (pass ? "PASS" : "FAIL") << "\n"
              << "================================================================\n\n";
    }

    // Write CSV
    if (rank == 0)
    {
        std::system("mkdir -p Results/mms");

        std::ofstream csv("Results/mms/angular_momentum_mms.csv");
        csv << "refinement,n_dofs,h,w_L2,w_L2_rate,w_H1,w_H1_rate,"
            << "w_Linf,iterations,walltime\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& r = results[i];
            csv << r.refinement << ","
                << r.n_dofs << ","
                << std::scientific << std::setprecision(6) << r.h << ","
                << r.w_L2 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? L2_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.w_H1 << ","
                << std::fixed << std::setprecision(3)
                << (i > 0 ? H1_rates[i-1] : 0.0) << ","
                << std::scientific << std::setprecision(6) << r.w_Linf << ","
                << r.iterations << ","
                << std::fixed << std::setprecision(4) << r.walltime << "\n";
        }
        pcout << "  Results written to Results/mms/angular_momentum_mms.csv\n\n";
    }

    return pass ? 0 : 1;
}
