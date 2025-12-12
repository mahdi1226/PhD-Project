// ============================================================================
// core/phase_field.cc - Main Orchestrator Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "phase_field.h"
#include "output/logger.h"

#include "assembly/ch_assembler.h"
#include "assembly/ns_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "assembly/poisson_assembler.h"

#include "solvers/ch_solver.h"
#include "solvers/ns_solver.h"
#include "solvers/magnetization_solver.h"
#include "solvers/poisson_solver.h"

#include "output/vtk_writer.h"

#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
PhaseFieldProblem<dim>::PhaseFieldProblem(const Parameters& params)
    : params_(params)
    , triangulation_()
    , fe_Q2_(2)
    , fe_Q1_(1)
    , theta_dof_handler_(triangulation_)
    , psi_dof_handler_(triangulation_)
    , mx_dof_handler_(triangulation_)
    , my_dof_handler_(triangulation_)
    , phi_dof_handler_(triangulation_)
    , ux_dof_handler_(triangulation_)
    , uy_dof_handler_(triangulation_)
    , p_dof_handler_(triangulation_)
    , time_(0.0)
    , timestep_number_(0)
{
    Logger::info("PhaseFieldProblem constructor started");

    // Create assemblers
    Logger::info("  Creating assemblers...");
    ch_assembler_      = std::make_unique<CHAssembler<dim>>(*this);
    ns_assembler_      = std::make_unique<NSAssembler<dim>>(*this);
    mag_assembler_     = std::make_unique<MagnetizationAssembler<dim>>(*this);
    poisson_assembler_ = std::make_unique<PoissonAssembler<dim>>(*this);

    // Create solvers
    Logger::info("  Creating solvers...");
    ch_solver_      = std::make_unique<CHSolver<dim>>(*this);
    ns_solver_      = std::make_unique<NSSolver<dim>>(*this);
    mag_solver_     = std::make_unique<MagnetizationSolver<dim>>(*this);
    poisson_solver_ = std::make_unique<PoissonSolver<dim>>(*this);

    // Create output writer
    Logger::info("  Creating VTK writer...");
    vtk_writer_ = std::make_unique<VTKWriter<dim>>(*this);

    Logger::success("PhaseFieldProblem constructor completed");
}

// ============================================================================
// Destructor
// ============================================================================
template <int dim>
PhaseFieldProblem<dim>::~PhaseFieldProblem()
{
    Logger::info("PhaseFieldProblem destructor called");
}

// ============================================================================
// Helper: generate timestamped folder name
// ============================================================================
std::string generate_run_folder(const std::string& base_folder)
{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time_t);

    std::ostringstream oss;
    oss << base_folder << "/run-"
        << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    return oss.str();
}

// ============================================================================
// run() - Main time loop
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::run()
{
    Logger::info("========================================");
    Logger::info("PhaseFieldProblem::run() started");
    Logger::info("========================================");

    // Create timestamped output folder
    params_.output.folder = generate_run_folder(params_.output.folder);
    std::filesystem::create_directories(params_.output.folder);
    Logger::info("Output folder: " + params_.output.folder);

    // Update VTK writer with new folder
    vtk_writer_->set_output_directory(params_.output.folder);

    // Setup phase
    Logger::info("Setting up mesh...");
    setup_mesh();

    Logger::info("Setting up DoF handlers...");
    setup_dof_handlers();

    Logger::info("Setting up constraints...");
    setup_constraints();

    Logger::info("Setting up sparsity patterns...");
    setup_sparsity_patterns();

    Logger::info("Initializing solutions...");
    initialize_solutions();

    // Output initial condition
    Logger::info("Outputting initial condition...");
    output_results();

    // Time loop
    const double t_final = params_.time.t_final;
    const double dt = params_.time.dt;

    Logger::info("Starting time loop...");
    Logger::info("  t_final = " + std::to_string(t_final));
    Logger::info("  dt      = " + std::to_string(dt));

    while (time_ < t_final - 1e-12)
    {
        ++timestep_number_;
        time_ += dt;

        Logger::info("----------------------------------------");
        Logger::info("Time step " + std::to_string(timestep_number_) +
                     ", t = " + std::to_string(time_));

        do_time_step();

        if (timestep_number_ % params_.output.frequency == 0)
            output_results();

        if (params_.amr.enabled && timestep_number_ % params_.amr.interval == 0)
            refine_mesh();
    }

    // Final output
    Logger::info("Outputting final results...");
    output_results();

    Logger::success("========================================");
    Logger::success("PhaseFieldProblem::run() completed");
    Logger::success("========================================");
}

// ============================================================================
// do_time_step() - Single time step
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::do_time_step()
{
    Logger::info("  do_time_step() started");

    const double dt = params_.time.dt;

    // Step 1: Cahn-Hilliard [Eq. 42a-b]
    Logger::info("    [1/4] Cahn-Hilliard...");
    ch_assembler_->assemble(dt, time_);
    ch_solver_->solve();

    // Step 2: Magnetization [Eq. 42c]
    Logger::info("    [2/4] Magnetization...");
    mag_assembler_->assemble(dt, time_);
    mag_solver_->solve();

    // Step 3: Poisson [Eq. 42d]
    Logger::info("    [3/4] Poisson...");
    poisson_assembler_->assemble(time_);
    poisson_solver_->solve();

    // Step 4: Navier-Stokes [Eq. 42e-f]
    Logger::info("    [4/4] Navier-Stokes...");
    ns_assembler_->assemble(dt, time_);
    ns_solver_->solve();

    // Update old solutions
    theta_old_solution_ = theta_solution_;
    ux_old_solution_ = ux_solution_;
    uy_old_solution_ = uy_solution_;
    mx_old_solution_ = mx_solution_;
    my_old_solution_ = my_solution_;

    Logger::success("  do_time_step() completed");
}

// ============================================================================
// output_results()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::output_results() const
{
    vtk_writer_->write(timestep_number_);
}

template class PhaseFieldProblem<2>;
//template class PhaseFieldProblem<3>;