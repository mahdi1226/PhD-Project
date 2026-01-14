// ============================================================================
// validation/bubble_benchmark.cc - Rising Bubble Validation
//
// Standalone program to validate Cahn-Hilliard + Navier-Stokes coupling
// against Hysing et al. benchmark data.
//
// Usage:
//   ./bubble_benchmark --test-case 1 --refinement 5
//   ./bubble_benchmark --test-case 2 --refinement 6
//
// Test cases (Hysing et al.):
//   Case 1: Low density/viscosity ratio (easier)
//   Case 2: High density/viscosity ratio (harder)
//
// Output:
//   - Console: Bubble metrics at each output time
//   - CSV: Time series of centroid, circularity, rise velocity
//
// Reference: Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009) 1259-1288
// ============================================================================

#include "validation/validation.h"
#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

// deal.II includes
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace dealii;

// ============================================================================
// Hysing Benchmark Parameters
// ============================================================================
namespace HysingData
{
    // Test Case 1: Low density/viscosity ratio
    struct TestCase1
    {
        static constexpr double rho_1 = 1000.0;     // Bubble density
        static constexpr double rho_2 = 100.0;      // Surrounding fluid density
        static constexpr double mu_1 = 10.0;        // Bubble viscosity
        static constexpr double mu_2 = 1.0;         // Surrounding fluid viscosity
        static constexpr double sigma = 24.5;       // Surface tension
        static constexpr double g = 0.98;           // Gravity

        // Reference values at t = 3.0
        static constexpr double ref_centroid_y = 1.0813;
        static constexpr double ref_rise_velocity = 0.2417;
        static constexpr double ref_circularity = 0.9013;
    };

    // Test Case 2: High density/viscosity ratio
    struct TestCase2
    {
        static constexpr double rho_1 = 1000.0;
        static constexpr double rho_2 = 1.0;
        static constexpr double mu_1 = 10.0;
        static constexpr double mu_2 = 0.1;
        static constexpr double sigma = 1.96;
        static constexpr double g = 0.98;

        // Reference values at t = 3.0
        static constexpr double ref_centroid_y = 1.1380;
        static constexpr double ref_rise_velocity = 0.2502;
        static constexpr double ref_circularity = 0.5144;
    };

    // Domain: [0, 1] x [0, 2]
    static constexpr double domain_width = 1.0;
    static constexpr double domain_height = 2.0;

    // Initial bubble: circle centered at (0.5, 0.5) with radius 0.25
    static constexpr double bubble_center_x = 0.5;
    static constexpr double bubble_center_y = 0.5;
    static constexpr double bubble_radius = 0.25;

    // Simulation parameters
    static constexpr double t_end = 3.0;
}

// ============================================================================
// Initial condition: circular bubble
// ============================================================================
template <int dim>
class BubbleInitialCondition : public Function<dim>
{
public:
    BubbleInitialCondition(double cx, double cy, double r, double epsilon)
        : Function<dim>(1)
        , center_x_(cx), center_y_(cy), radius_(r), epsilon_(epsilon)
    {}

    virtual double value(const Point<dim>& p,
                        const unsigned int /*component*/ = 0) const override
    {
        double dist = std::sqrt((p[0] - center_x_) * (p[0] - center_x_) +
                               (p[1] - center_y_) * (p[1] - center_y_));
        // Smooth transition: θ = tanh((r - dist) / (sqrt(2) * ε))
        return std::tanh((radius_ - dist) / (std::sqrt(2.0) * epsilon_));
    }

private:
    double center_x_, center_y_, radius_, epsilon_;
};

// ============================================================================
// Rising Bubble Benchmark
// ============================================================================
template <int dim>
class BubbleBenchmark
{
public:
    BubbleBenchmark(MPI_Comm comm, int test_case, unsigned int refinement);
    void run();

    const std::vector<BubbleMetrics>& get_metrics_history() const
    { return metrics_history_; }

private:
    void setup_grid();
    void setup_dofs();
    void setup_initial_condition();
    void run_simulation();
    void output_results();
    void validate_results();

    MPI_Comm                                    mpi_comm_;
    ConditionalOStream                          pcout_;

    parallel::distributed::Triangulation<dim>   triangulation_;

    FE_Q<dim>                                   fe_;
    DoFHandler<dim>                             dof_handler_;
    AffineConstraints<double>                   constraints_;

    TrilinosWrappers::MPI::Vector               solution_theta_;

    IndexSet                                    locally_owned_dofs_;
    IndexSet                                    locally_relevant_dofs_;

    int                                         test_case_;
    unsigned int                                refinement_;
    double                                      epsilon_;

    std::vector<BubbleMetrics>                  metrics_history_;
    std::string                                 output_dir_;
};

template <int dim>
BubbleBenchmark<dim>::BubbleBenchmark(MPI_Comm comm, int test_case, unsigned int refinement)
    : mpi_comm_(comm)
    , pcout_(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , triangulation_(comm,
                     typename Triangulation<dim>::MeshSmoothing(
                         Triangulation<dim>::smoothing_on_refinement |
                         Triangulation<dim>::smoothing_on_coarsening))
    , fe_(2)  // Q2 elements
    , dof_handler_(triangulation_)
    , test_case_(test_case)
    , refinement_(refinement)
    , epsilon_(0.01)  // Interface width
    , output_dir_("bubble_output")
{
    pcout_ << "============================================================\n"
           << "Rising Bubble Benchmark (Hysing et al.)\n"
           << "============================================================\n"
           << "  Test case: " << test_case_ << "\n"
           << "  Refinement level: " << refinement_ << "\n";

    if (test_case_ == 1)
    {
        pcout_ << "  Density ratio: " << HysingData::TestCase1::rho_1
               << " / " << HysingData::TestCase1::rho_2 << "\n"
               << "  Viscosity ratio: " << HysingData::TestCase1::mu_1
               << " / " << HysingData::TestCase1::mu_2 << "\n"
               << "  Surface tension: " << HysingData::TestCase1::sigma << "\n";
    }
    else
    {
        pcout_ << "  Density ratio: " << HysingData::TestCase2::rho_1
               << " / " << HysingData::TestCase2::rho_2 << "\n"
               << "  Viscosity ratio: " << HysingData::TestCase2::mu_1
               << " / " << HysingData::TestCase2::mu_2 << "\n"
               << "  Surface tension: " << HysingData::TestCase2::sigma << "\n";
    }

    pcout_ << "============================================================\n\n";
}

template <int dim>
void BubbleBenchmark<dim>::setup_grid()
{
    pcout_ << "Setting up grid... ";

    // Domain: [0, 1] x [0, 2]
    Point<dim> p1(0.0, 0.0);
    Point<dim> p2(HysingData::domain_width, HysingData::domain_height);

    std::vector<unsigned int> repetitions(dim);
    repetitions[0] = 1;
    repetitions[1] = 2;  // Twice as tall

    GridGenerator::subdivided_hyper_rectangle(triangulation_, repetitions, p1, p2);
    triangulation_.refine_global(refinement_);

    pcout_ << triangulation_.n_global_active_cells() << " cells\n";
}

template <int dim>
void BubbleBenchmark<dim>::setup_dofs()
{
    pcout_ << "Setting up DoFs... ";

    dof_handler_.distribute_dofs(fe_);

    locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);

    pcout_ << dof_handler_.n_dofs() << " DoFs\n";

    constraints_.clear();
    constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);
    DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);
    constraints_.close();

    solution_theta_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
}

template <int dim>
void BubbleBenchmark<dim>::setup_initial_condition()
{
    pcout_ << "Setting initial condition... ";

    BubbleInitialCondition<dim> ic(
        HysingData::bubble_center_x,
        HysingData::bubble_center_y,
        HysingData::bubble_radius,
        epsilon_);

    TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs_, mpi_comm_);
    VectorTools::interpolate(dof_handler_, ic, tmp);
    constraints_.distribute(tmp);
    solution_theta_ = tmp;

    pcout_ << "done\n";

    // Record initial metrics
    BubbleMetrics initial = compute_bubble_metrics<dim>(
        dof_handler_, solution_theta_, 0.0, -1e10, 0.0, mpi_comm_);
    metrics_history_.push_back(initial);

    pcout_ << "Initial bubble:\n"
           << "  Centroid: (" << initial.centroid_x << ", " << initial.centroid_y << ")\n"
           << "  Area: " << initial.area << "\n"
           << "  Circularity: " << initial.circularity << "\n\n";
}

template <int dim>
void BubbleBenchmark<dim>::run_simulation()
{
    pcout_ << "Running simulation...\n";
    pcout_ << "NOTE: Connect this to your actual CH+NS solver\n";
    pcout_ << "      The validation functions are ready to use.\n\n";

    // Placeholder: In actual use, this would call your time stepping loop
    // and record metrics at each output interval

    // Example of how to record metrics during time stepping:
    /*
    double dt = 0.001;
    double t = 0.0;
    double t_end = HysingData::t_end;
    double output_interval = 0.1;
    double next_output = output_interval;

    while (t < t_end)
    {
        // Your time stepping code here
        // solve_cahn_hilliard();
        // solve_navier_stokes();

        t += dt;

        if (t >= next_output)
        {
            double prev_y = metrics_history_.empty() ?
                           -1e10 : metrics_history_.back().centroid_y;

            BubbleMetrics m = compute_bubble_metrics<dim>(
                dof_handler_, solution_theta_, t, prev_y, dt, mpi_comm_);
            metrics_history_.push_back(m);

            pcout_ << "t = " << t
                   << ", y_c = " << m.centroid_y
                   << ", v_rise = " << m.rise_velocity
                   << ", circ = " << m.circularity << "\n";

            next_output += output_interval;
        }
    }
    */
}

template <int dim>
void BubbleBenchmark<dim>::validate_results()
{
    pcout_ << "\n============================================================\n"
           << "VALIDATION RESULTS - Test Case " << test_case_ << "\n"
           << "============================================================\n";

    if (metrics_history_.empty())
    {
        pcout_ << "No metrics recorded. Run simulation first.\n";
        return;
    }

    const BubbleMetrics& final_metrics = metrics_history_.back();

    double ref_y, ref_v, ref_c;
    if (test_case_ == 1)
    {
        ref_y = HysingData::TestCase1::ref_centroid_y;
        ref_v = HysingData::TestCase1::ref_rise_velocity;
        ref_c = HysingData::TestCase1::ref_circularity;
    }
    else
    {
        ref_y = HysingData::TestCase2::ref_centroid_y;
        ref_v = HysingData::TestCase2::ref_rise_velocity;
        ref_c = HysingData::TestCase2::ref_circularity;
    }

    pcout_ << "At t = " << final_metrics.time << ":\n"
           << "  Centroid y:     " << final_metrics.centroid_y
           << " (ref: " << ref_y << ")\n"
           << "  Rise velocity:  " << final_metrics.rise_velocity
           << " (ref: " << ref_v << ")\n"
           << "  Circularity:    " << final_metrics.circularity
           << " (ref: " << ref_c << ")\n";

    double err_y = std::abs(final_metrics.centroid_y - ref_y) / ref_y;
    double err_v = std::abs(final_metrics.rise_velocity - ref_v) / ref_v;
    double err_c = std::abs(final_metrics.circularity - ref_c) / ref_c;

    pcout_ << "\nRelative errors:\n"
           << "  Centroid:    " << err_y * 100 << "%\n"
           << "  Velocity:    " << err_v * 100 << "%\n"
           << "  Circularity: " << err_c * 100 << "%\n";

    bool passed = (err_y < 0.05 && err_v < 0.10 && err_c < 0.10);
    pcout_ << "\nStatus: " << (passed ? "PASSED" : "NEEDS REVIEW") << "\n"
           << "============================================================\n";
}

template <int dim>
void BubbleBenchmark<dim>::output_results()
{
    if (Utilities::MPI::this_mpi_process(mpi_comm_) != 0)
        return;

    std::string csv_file = output_dir_ + "/bubble_case" +
                          std::to_string(test_case_) + "_metrics.csv";
    write_bubble_metrics_csv(metrics_history_, csv_file);
    pcout_ << "Wrote metrics to " << csv_file << "\n";
}

template <int dim>
void BubbleBenchmark<dim>::run()
{
    setup_grid();
    setup_dofs();
    setup_initial_condition();
    run_simulation();
    validate_results();
    output_results();
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Default parameters
    int test_case = 1;
    unsigned int refinement = 5;

    // Parse command line
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--test-case" && i + 1 < argc)
            test_case = std::stoi(argv[++i]);
        else if (arg == "--refinement" && i + 1 < argc)
            refinement = std::stoul(argv[++i]);
        else if (arg == "--help")
        {
            if (Utilities::MPI::this_mpi_process(comm) == 0)
            {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --test-case N    Hysing test case (1 or 2)\n"
                          << "  --refinement N   Mesh refinement level\n"
                          << "  --help           Show this help\n";
            }
            return 0;
        }
    }

    try
    {
        BubbleBenchmark<2> benchmark(comm, test_case, refinement);
        benchmark.run();
    }
    catch (std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    return 0;
}