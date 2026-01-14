// ============================================================================
// validation/cavity_benchmark.cc - Lid-Driven Cavity Validation
//
// Standalone program to validate Navier-Stokes solver against Ghia et al.
// benchmark data.
//
// Usage:
//   ./cavity_benchmark --re 100 --refinement 5
//   ./cavity_benchmark --re 1000 --refinement 6
//
// Output:
//   - Console: Pass/fail status with RMS errors
//   - CSV: Detailed centerline comparison
//
// Reference: Ghia, Ghia & Shin, J. Comp. Physics 48 (1982) 387-411
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
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using namespace dealii;

// ============================================================================
// Lid velocity boundary condition
// ============================================================================
template <int dim>
class LidVelocity : public Function<dim>
{
public:
    LidVelocity(double lid_velocity = 1.0)
        : Function<dim>(1), U_lid(lid_velocity) {}

    virtual double value(const Point<dim>& p,
                        const unsigned int component = 0) const override
    {
        (void)p;
        (void)component;
        return U_lid;
    }

private:
    double U_lid;
};

// ============================================================================
// Zero boundary condition
// ============================================================================
template <int dim>
class ZeroFunction : public Function<dim>
{
public:
    ZeroFunction() : Function<dim>(1) {}

    virtual double value(const Point<dim>& /*p*/,
                        const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};

// ============================================================================
// Cavity Benchmark Problem
// ============================================================================
template <int dim>
class CavityBenchmark
{
public:
    CavityBenchmark(MPI_Comm comm, int reynolds, unsigned int refinement);
    void run();

    CavityValidation get_validation_result() const { return validation_result_; }

private:
    void setup_grid();
    void setup_dofs();
    void setup_system();
    void assemble_stokes();
    void solve_stokes();
    void run_to_steady_state();
    void extract_and_validate();
    void output_results();

    MPI_Comm                                    mpi_comm_;
    ConditionalOStream                          pcout_;

    parallel::distributed::Triangulation<dim>   triangulation_;

    FE_Q<dim>                                   fe_velocity_;
    FE_Q<dim>                                   fe_pressure_;

    DoFHandler<dim>                             dof_handler_ux_;
    DoFHandler<dim>                             dof_handler_uy_;
    DoFHandler<dim>                             dof_handler_p_;

    AffineConstraints<double>                   constraints_ux_;
    AffineConstraints<double>                   constraints_uy_;
    AffineConstraints<double>                   constraints_p_;

    TrilinosWrappers::SparseMatrix              system_matrix_ux_;
    TrilinosWrappers::SparseMatrix              system_matrix_uy_;
    TrilinosWrappers::SparseMatrix              system_matrix_p_;

    TrilinosWrappers::MPI::Vector               solution_ux_;
    TrilinosWrappers::MPI::Vector               solution_uy_;
    TrilinosWrappers::MPI::Vector               solution_p_;

    TrilinosWrappers::MPI::Vector               rhs_ux_;
    TrilinosWrappers::MPI::Vector               rhs_uy_;
    TrilinosWrappers::MPI::Vector               rhs_p_;

    IndexSet                                    locally_owned_dofs_velocity_;
    IndexSet                                    locally_relevant_dofs_velocity_;
    IndexSet                                    locally_owned_dofs_pressure_;
    IndexSet                                    locally_relevant_dofs_pressure_;

    int                                         reynolds_;
    unsigned int                                refinement_;
    double                                      viscosity_;
    double                                      dt_;
    double                                      tolerance_;
    unsigned int                                max_iterations_;

    CavityValidation                            validation_result_;
    std::string                                 output_dir_;
};

template <int dim>
CavityBenchmark<dim>::CavityBenchmark(MPI_Comm comm, int reynolds, unsigned int refinement)
    : mpi_comm_(comm)
    , pcout_(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
    , triangulation_(comm,
                     typename Triangulation<dim>::MeshSmoothing(
                         Triangulation<dim>::smoothing_on_refinement |
                         Triangulation<dim>::smoothing_on_coarsening))
    , fe_velocity_(2)  // Q2 for velocity
    , fe_pressure_(1)  // Q1 for pressure
    , dof_handler_ux_(triangulation_)
    , dof_handler_uy_(triangulation_)
    , dof_handler_p_(triangulation_)
    , reynolds_(reynolds)
    , refinement_(refinement)
    , viscosity_(1.0 / reynolds)
    , dt_(0.01)
    , tolerance_(1e-6)
    , max_iterations_(10000)
    , output_dir_("cavity_output")
{
    pcout_ << "============================================================\n"
           << "Lid-Driven Cavity Benchmark\n"
           << "============================================================\n"
           << "  Reynolds number: " << reynolds_ << "\n"
           << "  Refinement level: " << refinement_ << "\n"
           << "  Viscosity: " << viscosity_ << "\n"
           << "============================================================\n\n";
}

template <int dim>
void CavityBenchmark<dim>::setup_grid()
{
    pcout_ << "Setting up grid... ";

    GridGenerator::hyper_cube(triangulation_, 0.0, 1.0);
    triangulation_.refine_global(refinement_);

    pcout_ << triangulation_.n_global_active_cells() << " cells\n";
}

template <int dim>
void CavityBenchmark<dim>::setup_dofs()
{
    pcout_ << "Setting up DoFs... ";

    dof_handler_ux_.distribute_dofs(fe_velocity_);
    dof_handler_uy_.distribute_dofs(fe_velocity_);
    dof_handler_p_.distribute_dofs(fe_pressure_);

    // Velocity DoFs
    locally_owned_dofs_velocity_ = dof_handler_ux_.locally_owned_dofs();
    locally_relevant_dofs_velocity_ = DoFTools::extract_locally_relevant_dofs(dof_handler_ux_);

    // Pressure DoFs
    locally_owned_dofs_pressure_ = dof_handler_p_.locally_owned_dofs();
    locally_relevant_dofs_pressure_ = DoFTools::extract_locally_relevant_dofs(dof_handler_p_);

    pcout_ << dof_handler_ux_.n_dofs() << " velocity DoFs per component, "
           << dof_handler_p_.n_dofs() << " pressure DoFs\n";

    // Boundary conditions for u_x:
    // - Top lid (y=1): u_x = 1
    // - All other walls: u_x = 0
    constraints_ux_.clear();
    constraints_ux_.reinit(locally_owned_dofs_velocity_, locally_relevant_dofs_velocity_);
    DoFTools::make_hanging_node_constraints(dof_handler_ux_, constraints_ux_);

    // Bottom (y=0): u_x = 0
    VectorTools::interpolate_boundary_values(dof_handler_ux_,
                                              0,  // boundary_id (all boundaries)
                                              ZeroFunction<dim>(),
                                              constraints_ux_);

    // We need to set top lid separately - for now use component mask approach
    // In deal.II, boundary_id 0 is all boundaries by default
    // We'll handle top lid by iterating over boundary faces

    constraints_ux_.close();

    // Boundary conditions for u_y: zero everywhere
    constraints_uy_.clear();
    constraints_uy_.reinit(locally_owned_dofs_velocity_, locally_relevant_dofs_velocity_);
    DoFTools::make_hanging_node_constraints(dof_handler_uy_, constraints_uy_);
    VectorTools::interpolate_boundary_values(dof_handler_uy_,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints_uy_);
    constraints_uy_.close();

    // Pressure: pin one value to fix constant
    constraints_p_.clear();
    constraints_p_.reinit(locally_owned_dofs_pressure_, locally_relevant_dofs_pressure_);
    DoFTools::make_hanging_node_constraints(dof_handler_p_, constraints_p_);
    // Pin pressure at corner
    if (locally_owned_dofs_pressure_.is_element(0))
        constraints_p_.add_constraint(0, {}, 0.0);
    constraints_p_.close();
}

template <int dim>
void CavityBenchmark<dim>::setup_system()
{
    pcout_ << "Setting up system matrices... ";

    // Velocity matrices
    {
        DynamicSparsityPattern dsp(locally_relevant_dofs_velocity_);
        DoFTools::make_sparsity_pattern(dof_handler_ux_, dsp, constraints_ux_, false);
        SparsityTools::distribute_sparsity_pattern(dsp,
            dof_handler_ux_.locally_owned_dofs(),
            mpi_comm_,
            locally_relevant_dofs_velocity_);

        system_matrix_ux_.reinit(locally_owned_dofs_velocity_,
                                  locally_owned_dofs_velocity_,
                                  dsp, mpi_comm_);
        system_matrix_uy_.reinit(locally_owned_dofs_velocity_,
                                  locally_owned_dofs_velocity_,
                                  dsp, mpi_comm_);
    }

    // Pressure matrix
    {
        DynamicSparsityPattern dsp(locally_relevant_dofs_pressure_);
        DoFTools::make_sparsity_pattern(dof_handler_p_, dsp, constraints_p_, false);
        SparsityTools::distribute_sparsity_pattern(dsp,
            dof_handler_p_.locally_owned_dofs(),
            mpi_comm_,
            locally_relevant_dofs_pressure_);

        system_matrix_p_.reinit(locally_owned_dofs_pressure_,
                                 locally_owned_dofs_pressure_,
                                 dsp, mpi_comm_);
    }

    // Vectors
    solution_ux_.reinit(locally_owned_dofs_velocity_, locally_relevant_dofs_velocity_,
                        mpi_comm_);
    solution_uy_.reinit(locally_owned_dofs_velocity_, locally_relevant_dofs_velocity_,
                        mpi_comm_);
    solution_p_.reinit(locally_owned_dofs_pressure_, locally_relevant_dofs_pressure_,
                       mpi_comm_);

    rhs_ux_.reinit(locally_owned_dofs_velocity_, mpi_comm_);
    rhs_uy_.reinit(locally_owned_dofs_velocity_, mpi_comm_);
    rhs_p_.reinit(locally_owned_dofs_pressure_, mpi_comm_);

    pcout_ << "done\n";
}

template <int dim>
void CavityBenchmark<dim>::run_to_steady_state()
{
    pcout_ << "\nRunning to steady state...\n";

    // Simple iterative solver (placeholder - replace with your NS solver)
    // This is a simplified version; your actual code has the full implementation

    // For now, just indicate this needs to be connected to your actual solver
    pcout_ << "NOTE: Connect this to your actual Navier-Stokes solver\n";
    pcout_ << "      The validation functions are ready to use.\n\n";

    // Placeholder: set solution to zero
    solution_ux_ = 0.0;
    solution_uy_ = 0.0;
    solution_p_ = 0.0;
}

template <int dim>
void CavityBenchmark<dim>::extract_and_validate()
{
    pcout_ << "Extracting centerlines and validating...\n";

    // Extract vertical centerline: u_x(0.5, y)
    CenterlineData ux_vertical = extract_ux_vertical_centerline<dim>(
        dof_handler_ux_, solution_ux_,
        0.5,    // x = 0.5
        0.0,    // y_min
        1.0,    // y_max
        129,    // n_samples
        mpi_comm_);

    // Extract horizontal centerline: u_y(x, 0.5)
    CenterlineData uy_horizontal = extract_uy_horizontal_centerline<dim>(
        dof_handler_uy_, solution_uy_,
        0.5,    // y = 0.5
        0.0,    // x_min
        1.0,    // x_max
        129,    // n_samples
        mpi_comm_);

    // Validate against Ghia data
    validation_result_ = validate_cavity(ux_vertical, uy_horizontal,
                                          reynolds_, 0.05);

    // Output results
    if (Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
    {
        pcout_ << "\n============================================================\n"
               << "VALIDATION RESULTS - Re = " << reynolds_ << "\n"
               << "============================================================\n"
               << "  u_x RMS error:  " << validation_result_.ux_rms_error << "\n"
               << "  u_x max error:  " << validation_result_.ux_max_error << "\n"
               << "  u_y RMS error:  " << validation_result_.uy_rms_error << "\n"
               << "  u_y max error:  " << validation_result_.uy_max_error << "\n"
               << "  Combined RMS:   " << validation_result_.combined_rms << "\n"
               << "  Status:         " << (validation_result_.passed ? "PASSED" : "FAILED") << "\n"
               << "============================================================\n\n";

        // Write CSV comparison
        std::string csv_file = output_dir_ + "/cavity_Re" +
                              std::to_string(reynolds_) + "_comparison.csv";
        write_cavity_comparison_csv(ux_vertical, uy_horizontal, reynolds_, csv_file);
        pcout_ << "Wrote comparison data to " << csv_file << "\n";
    }
}

template <int dim>
void CavityBenchmark<dim>::output_results()
{
    // VTK output for visualization
    DataOut<dim> data_out;

    // Create vectors on locally owned dofs for output
    TrilinosWrappers::MPI::Vector ux_output(locally_owned_dofs_velocity_, mpi_comm_);
    TrilinosWrappers::MPI::Vector uy_output(locally_owned_dofs_velocity_, mpi_comm_);

    ux_output = solution_ux_;
    uy_output = solution_uy_;

    data_out.attach_dof_handler(dof_handler_ux_);
    data_out.add_data_vector(ux_output, "ux");

    // Note: For proper vector output, you'd want to use a vector-valued FE
    // This is simplified for the benchmark

    data_out.build_patches();

    std::string vtk_file = output_dir_ + "/cavity_Re" +
                          std::to_string(reynolds_) + ".vtu";

    data_out.write_vtu_in_parallel(vtk_file, mpi_comm_);
    pcout_ << "Wrote VTK output to " << vtk_file << "\n";
}

template <int dim>
void CavityBenchmark<dim>::run()
{
    setup_grid();
    setup_dofs();
    setup_system();
    run_to_steady_state();
    extract_and_validate();
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
    int reynolds = 100;
    unsigned int refinement = 5;

    // Parse command line
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--re" && i + 1 < argc)
            reynolds = std::stoi(argv[++i]);
        else if (arg == "--refinement" && i + 1 < argc)
            refinement = std::stoul(argv[++i]);
        else if (arg == "--help")
        {
            if (Utilities::MPI::this_mpi_process(comm) == 0)
            {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --re N          Reynolds number (100, 400, 1000)\n"
                          << "  --refinement N  Mesh refinement level\n"
                          << "  --help          Show this help\n";
            }
            return 0;
        }
    }

    try
    {
        CavityBenchmark<2> benchmark(comm, reynolds, refinement);
        benchmark.run();

        // Return non-zero if validation failed
        return benchmark.get_validation_result().passed ? 0 : 1;
    }
    catch (std::exception& exc)
    {
        std::cerr << "Exception: " << exc.what() << std::endl;
        return 1;
    }

    return 0;
}