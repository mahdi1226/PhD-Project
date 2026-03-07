// ============================================================================
// drivers/fhd_ch_driver.cc — Full Coupled FHD + Cahn-Hilliard Driver
//
// Two-phase ferrohydrodynamics: all 6 subsystems
//   Poisson + Magnetization (Picard) + NS + AngMom + Cahn-Hilliard
//
// Phase-dependent material properties:
//   χ(φ): susceptibility  — ferrofluid (φ=+1) responds, carrier (φ=-1) does not
//   ν(φ): viscosity       — linear interpolation between carrier and ferrofluid
//
// Capillary force: σ·μ_CH·∇φ (diffuse-interface surface tension)
// Kelvin force:    μ₀[(M·∇)H + ½div(M)H] — acts only where χ(φ)≠0
//
// Time loop per step:
//   Picard loop (Mag + Poisson only):
//     1. Mag.assemble(M_old, φ_mag, u_old, w_old, ch_old)  ← χ(φ_old)
//     2. Mag.solve() → M
//     3. Poisson.assemble_rhs(M, h_a)
//     4. Poisson.solve() → φ_mag
//     Check Picard convergence on (M, φ_mag)
//
//   Sequential (no Picard):
//     5. NS.assemble(u_old, w_old, M, H, ch_old)  ← ν(φ_old) + capillary + Kelvin
//     6. NS.solve() → u, p
//     7. AngMom.assemble(w_old, u, M, H)
//     8. AngMom.solve() → w
//     9. CH.assemble(ch_old, dt, u)
//    10. CH.solve() → (φ, μ)
//
// CLI presets:
//   --rosensweig-instability : Zhang Fig 4.7 / Nochetto 2016 Fig 1
//   --droplet-deformation    : Zhang Fig 4.15-4.16
//
// Usage:
//   mpirun -np 1 ./fhd_ch_driver --rosensweig-instability -r 5 --block-schur -v
//   mpirun -np 1 ./fhd_ch_driver --rosensweig-instability -r 5 --epsilon 0.01 --field-strength 15 -v
//   mpirun -np 1 ./fhd_ch_driver --droplet-deformation -r 5 --dt 1e-3 -v
//
// CLI flags (override preset values):
//   --perturbation-amp VALUE   : IC perturbation amplitude (default 0.01 for Rosensweig)
//   --perturbation-modes N     : number of cosine modes (default 3)
//   --field-strength VALUE     : uniform field intensity H_max
//   --ramp-time VALUE          : field ramp time
//   --field-direction DX DY    : field direction (default 0 1 = vertical)
//   --epsilon VALUE            : CH interface width parameter ε
//   --gamma VALUE              : CH mobility γ
//   --sigma-ch VALUE           : capillary surface tension σ
//   --chi VALUE                : ferrofluid susceptibility χ₀
//   --block-schur              : use block-Schur preconditioner for NS
//   --amr / --no-amr          : enable/disable adaptive mesh refinement
//   --amr-interval N          : refine every N steps (default 5)
//   --amr-max-level N         : maximum refinement level
//   --amr-upper-fraction F    : fraction of cells to refine (default 0.3)
//   --amr-lower-fraction F    : fraction of cells to coarsen (default 0.1)
//
// References:
//   Nochetto, Salgado & Tomas (2016), CMAME, arXiv:1601.06824
//   Zhang, He, Yang (2021), SIAM J. Sci. Comput.
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "cahn_hilliard/cahn_hilliard.h"
#include "mesh/mesh.h"
#include "utilities/timestamp.h"
#include "utilities/amr.h"
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

// M_PI defined via physics/benchmark_initial_conditions.h

constexpr int dim = 2;

// ============================================================================
// Interface position tracking
//
// Finds the x-range and y-range of the φ=0 contour by scanning cells where
// φ is near the transition. Returns both vertical and horizontal extents.
// For Rosensweig instability, spike height = y_max - y_ref.
// For droplet elongation, aspect ratio = (y_max-y_min)/(x_max-x_min).
// ============================================================================
struct InterfaceMetrics
{
    double y_min = 1e30;    // lowest y of phi=0 contour
    double y_max = -1e30;   // highest y of phi=0 contour
    double x_min = 1e30;    // leftmost x of phi=0 contour
    double x_max = -1e30;   // rightmost x of phi=0 contour
    double amplitude = 0.0; // (y_max - y_min)/2
    double aspect_ratio = 1.0; // (y_max-y_min)/(x_max-x_min)
};

template <int dim_t>
InterfaceMetrics compute_interface_position(
    const dealii::DoFHandler<dim_t>& ch_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ch_solution_relevant,
    MPI_Comm mpi_comm)
{
    // Sub-cell interpolation: find φ=0 contour by detecting sign changes
    // across cell edges and linearly interpolating for sub-cell accuracy.
    // QTrapezoid evaluates at vertices: (0,0),(1,0),(0,1),(1,1)
    const dealii::QTrapezoid<dim_t> q_vert;
    dealii::FEValues<dim_t> fe_values(ch_dof_handler.get_fe(), q_vert,
                                       dealii::update_values |
                                       dealii::update_quadrature_points);

    const dealii::FEValuesExtractors::Scalar phi_extractor(0);

    // Edge connectivity for deal.II quad vertices (2D)
    // Vertex ordering: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    // Edges: bottom(0-1), right(1-3), top(2-3), left(0-2)
    static constexpr unsigned int edges[4][2] = {{0,1}, {1,3}, {2,3}, {0,2}};
    static constexpr unsigned int n_edges = 4;

    double local_y_min = 1e30, local_y_max = -1e30;
    double local_x_min = 1e30, local_x_max = -1e30;

    for (const auto& cell : ch_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        std::vector<double> phi_v(q_vert.size());
        fe_values[phi_extractor].get_function_values(ch_solution_relevant, phi_v);

        for (unsigned int e = 0; e < n_edges; ++e)
        {
            const unsigned int v1 = edges[e][0];
            const unsigned int v2 = edges[e][1];

            // Check for sign change (φ=0 crossing)
            if (phi_v[v1] * phi_v[v2] < 0.0)
            {
                // Linear interpolation: t = φ1/(φ1 - φ2)
                const double t = phi_v[v1] / (phi_v[v1] - phi_v[v2]);
                const auto& p1 = fe_values.quadrature_point(v1);
                const auto& p2 = fe_values.quadrature_point(v2);
                const double xc = p1[0] + t * (p2[0] - p1[0]);
                const double yc = p1[1] + t * (p2[1] - p1[1]);

                local_x_min = std::min(local_x_min, xc);
                local_x_max = std::max(local_x_max, xc);
                local_y_min = std::min(local_y_min, yc);
                local_y_max = std::max(local_y_max, yc);
            }
        }
    }

    // MPI reduce
    double global_y_min = 0.0, global_y_max = 0.0;
    double global_x_min = 0.0, global_x_max = 0.0;
    MPI_Allreduce(&local_y_min, &global_y_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);
    MPI_Allreduce(&local_y_max, &global_y_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);
    MPI_Allreduce(&local_x_min, &global_x_min, 1, MPI_DOUBLE, MPI_MIN, mpi_comm);
    MPI_Allreduce(&local_x_max, &global_x_max, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    InterfaceMetrics m;
    m.y_min = global_y_min;
    m.y_max = global_y_max;
    m.x_min = global_x_min;
    m.x_max = global_x_max;
    m.amplitude = (global_y_max - global_y_min) / 2.0;
    const double x_span = global_x_max - global_x_min;
    const double y_span = global_y_max - global_y_min;
    m.aspect_ratio = (x_span > 1e-12) ? y_span / x_span : 1.0;
    return m;
}

// ============================================================================
// Rosensweig instability preset
//
// Zhang, He & Yang (2021) SIAM J. Sci. Comput., Section 4.3, Eq. (4.4)
//
// Domain: [0,1] x [0,0.6], ferrofluid pool y ∈ [0, 0.2], carrier above.
// 5 dipoles far below (y = -15) approximate a uniform vertical field.
// Intensity ramps linearly 0 → 8000 over t ∈ [0, 1.6], then constant.
// Gravity is essential: g = (0, -6e4) with Boussinesq r = 0.1.
//
// Parameters from Zhang Eq. (4.4):
//   ε=5e-3, M=2e-4, ν_f=2, ν_w=1, μ=1, τ=1e-4,
//   β=1, χ₀=0.5, λ=1, r=0.1, h=1e-2, δt=1e-3, g=(0,-6e4)
// ============================================================================
void setup_rosensweig_instability(Parameters& params)
{
    params.experiment_name = "rosensweig_instability";

    // Domain: [0,1] x [0,0.6]
    // Ferrofluid pool at bottom (y < 0.2), carrier above
    params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
    params.domain.y_min = 0.0;  params.domain.y_max = 0.6;
    params.domain.initial_cells_x = 5;
    params.domain.initial_cells_y = 3;

    // Physics — Zhang Eq. (4.4) exact values
    params.physics.kappa_0 = 0.5;     // χ₀ for single-phase fallback
    params.physics.chi_ferro = 0.5;   // χ₀ = 0.5
    params.physics.nu = 2.0;          // single-phase viscosity (unused with CH)
    params.physics.nu_r = 0.0;        // no micropolar ν_r for Shliomis model
    params.physics.nu_carrier = 1.0;  // ν_w = 1
    params.physics.nu_ferro = 2.0;    // ν_f = 2
    params.physics.mu_0 = 1.0;        // μ = 1
    params.physics.j_micro = 1.0;
    params.physics.c_1 = 1.0;
    params.physics.c_2 = 1.0;
    params.physics.sigma = 0.0;       // no magnetic diffusion
    params.physics.T_relax = 1e-4;    // τ = 1e-4
    params.physics.beta = 1.0;        // β = 1 (Zhang Eq. 2.8)

    // Gravity (Boussinesq): f_g = (1 + r·H(θ/ε)) · g
    // Zhang Sec 4.3: g = (0, -6e4), r = 0.1
    params.physics.gravity_x = 0.0;
    params.physics.gravity_y = -6e4;
    params.physics.boussinesq_r = 0.1;

    // Cahn-Hilliard — Zhang Eq. (4.4)
    params.cahn_hilliard_params.epsilon = 5e-3;   // ε = 5e-3
    params.cahn_hilliard_params.gamma = 2e-4;     // M = 2e-4 (mobility)
    params.cahn_hilliard_params.sigma = 1.0;      // λ = 1 (surface tension)
    params.enable_cahn_hilliard = true;

    // Applied field: 5 dipoles far below, approximating uniform vertical field
    // Zhang Sec 4.3: positions (-0.5,-15),(0,-15),(0.5,-15),(1,-15),(1.5,-15)
    // direction d = (0,1), intensity ramps 0 → 8000 linearly over [0, 1.6]
    params.uniform_field.enabled = false;  // use dipoles, not uniform
    params.dipoles.positions.clear();
    params.dipoles.positions.push_back(dealii::Point<2>(-0.5, -15.0));
    params.dipoles.positions.push_back(dealii::Point<2>( 0.0, -15.0));
    params.dipoles.positions.push_back(dealii::Point<2>( 0.5, -15.0));
    params.dipoles.positions.push_back(dealii::Point<2>( 1.0, -15.0));
    params.dipoles.positions.push_back(dealii::Point<2>( 1.5, -15.0));
    params.dipoles.direction = {0.0, 1.0};
    params.dipoles.intensity_max = 8000.0;
    params.dipoles.ramp_time = 1.6;
    params.dipoles.ramp_slope = 0.0;
    params.dipoles.intensities.clear();   // shared intensity for all dipoles

    // Time
    params.time.dt = 1e-3;
    params.time.t_final = 2.0;

    // Solver: omega=0.7 converges faster than 0.5, tol=1e-5 is sufficient
    // (temporal error O(dt)=O(1e-3) dominates Picard residual)
    params.picard_iterations = 15;
    params.picard_tolerance = 1e-5;
    params.picard_relaxation = 0.7;

    // Output
    params.output.vtk_interval = 50;
}

// ============================================================================
// Droplet deformation preset
//
// Circular ferrofluid droplet under uniform vertical field
// Domain: [0,1] x [0,1], droplet at center, R=0.2
// Compare aspect ratio b/a vs magnetic Bond number
// ============================================================================
void setup_droplet_deformation(Parameters& params)
{
    params.experiment_name = "droplet_deformation";

    // Domain
    params.domain.x_min = 0.0;  params.domain.x_max = 1.0;
    params.domain.y_min = 0.0;  params.domain.y_max = 1.0;
    params.domain.initial_cells_x = 1;
    params.domain.initial_cells_y = 1;

    // Physics
    params.physics.kappa_0 = 1.0;
    params.physics.chi_ferro = 1.0;
    params.physics.nu = 1.0;
    params.physics.nu_r = 0.0;
    params.physics.nu_carrier = 1.0;
    params.physics.nu_ferro = 1.0;
    params.physics.mu_0 = 1.0;
    params.physics.j_micro = 1.0;
    params.physics.c_1 = 1.0;
    params.physics.c_2 = 1.0;
    params.physics.sigma = 0.0;
    params.physics.T_relax = 1e-4;

    // Cahn-Hilliard
    params.cahn_hilliard_params.epsilon = 0.02;
    params.cahn_hilliard_params.gamma = 1e-3;   // O(ε²) mobility for O(1) timescale
    params.cahn_hilliard_params.sigma = 1.0;
    params.enable_cahn_hilliard = true;

    // Uniform vertical field (ramped)
    params.uniform_field.enabled = true;
    params.uniform_field.direction = {0.0, 1.0};
    params.uniform_field.intensity_max = 5.0;
    params.uniform_field.ramp_time = 0.5;
    params.uniform_field.ramp_slope = 0.0;

    // No dipoles
    params.dipoles.positions.clear();
    params.dipoles.intensities.clear();

    // Time
    params.time.dt = 1e-3;
    params.time.t_final = 5.0;

    // Solver
    params.picard_iterations = 10;
    params.picard_tolerance = 1e-6;
    params.picard_relaxation = 0.5;

    // Output
    params.output.vtk_interval = 20;
}

// ============================================================================
// Ferrofluid pool IC for Rosensweig instability
//
// φ = +1 for y < y_interface(x), −1 above
// Smooth transition via tanh profile
//
// y_interface(x) = y0 + A · cos(2π·n·x / L)
//
// The sinusoidal perturbation seeds the normal-field instability.
// Without it, a flat interface under uniform field has no x-variation
// in the Kelvin force and the instability cannot develop.
// ============================================================================
template <int dim_t>
class FerrofluidPoolIC : public dealii::Function<dim_t>
{
public:
    FerrofluidPoolIC(double y_interface, double epsilon,
                     double perturbation_amp = 0.0,
                     unsigned int n_modes = 3,
                     double domain_width = 1.0)
        : dealii::Function<dim_t>(2)
        , y_interface_(y_interface)
        , width_(2.0 * std::sqrt(2.0) * epsilon)
        , perturbation_amp_(perturbation_amp)
        , n_modes_(n_modes)
        , domain_width_(domain_width)
    {}

    double value(const dealii::Point<dim_t>& p,
                 const unsigned int component = 0) const override
    {
        if (component == 0)
        {
            double y_intf = y_interface_;
            if (perturbation_amp_ > 0.0 && n_modes_ > 0)
                y_intf += perturbation_amp_ * std::cos(
                    2.0 * M_PI * n_modes_ * p[0] / domain_width_);
            return std::tanh((y_intf - p[1]) / width_);
        }
        return 0.0;   // mu = 0
    }

private:
    double y_interface_;
    double width_;
    double perturbation_amp_;
    unsigned int n_modes_;
    double domain_width_;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[])
{
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_comm);
    ConditionalOStream pcout(std::cout, rank == 0);

    // ================================================================
    // 1. Parse CLI
    // ================================================================
    Parameters params;
    std::string preset;
    unsigned int refinement = 5;
    bool verbose = false;
    double perturbation_amp = -1.0;    // -1 = use preset default
    int    perturbation_modes = -1;    // -1 = use preset default

    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--rosensweig-instability") == 0)
            preset = "rosensweig";
        else if (std::strcmp(argv[i], "--droplet-deformation") == 0)
            preset = "droplet";
        else if ((std::strcmp(argv[i], "-r") == 0 ||
                  std::strcmp(argv[i], "--refinement") == 0) && i+1 < argc)
            refinement = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0 ||
                 std::strcmp(argv[i], "--verbose") == 0)
            verbose = true;
    }

    // Apply preset first, then re-parse CLI to allow overrides
    if (preset == "rosensweig")
        setup_rosensweig_instability(params);
    else if (preset == "droplet")
        setup_droplet_deformation(params);
    else
    {
        // Default: generic two-phase setup
        params.experiment_name = "fhd_ch_generic";
        params.enable_cahn_hilliard = true;
    }

    // Re-parse CLI args that should override preset values
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--dt") == 0 && i+1 < argc)
            params.time.dt = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--t-final") == 0 && i+1 < argc)
            params.time.t_final = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--steps") == 0 && i+1 < argc)
            params.run.steps = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--vtk-interval") == 0 && i+1 < argc)
            params.output.vtk_interval = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i+1 < argc)
            params.cahn_hilliard_params.epsilon = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--chi") == 0 && i+1 < argc)
        {
            params.physics.chi_ferro = std::stod(argv[++i]);
            params.physics.kappa_0 = params.physics.chi_ferro;
        }
        else if (std::strcmp(argv[i], "--sigma-ch") == 0 && i+1 < argc)
            params.cahn_hilliard_params.sigma = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--field-strength") == 0 && i+1 < argc)
            params.uniform_field.intensity_max = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--ramp-time") == 0 && i+1 < argc)
            params.uniform_field.ramp_time = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--field-direction") == 0 && i+2 < argc)
        {
            params.uniform_field.direction = {std::stod(argv[i+1]),
                                               std::stod(argv[i+2])};
            i += 2;
        }
        else if (std::strcmp(argv[i], "--nu-carrier") == 0 && i+1 < argc)
            params.physics.nu_carrier = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--nu-ferro") == 0 && i+1 < argc)
            params.physics.nu_ferro = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--perturbation-amp") == 0 && i+1 < argc)
            perturbation_amp = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--perturbation-modes") == 0 && i+1 < argc)
            perturbation_modes = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--gamma") == 0 && i+1 < argc)
            params.cahn_hilliard_params.gamma = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--gravity-y") == 0 && i+1 < argc)
            params.physics.gravity_y = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--boussinesq-r") == 0 && i+1 < argc)
            params.physics.boussinesq_r = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--beta") == 0 && i+1 < argc)
            params.physics.beta = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--block-schur") == 0)
            params.solvers.navier_stokes.preconditioner =
                LinearSolverParams::Preconditioner::BlockSchur;
        // AMR flags
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no-amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr-interval") == 0 && i+1 < argc)
            params.mesh.amr_interval = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--amr-max-level") == 0 && i+1 < argc)
            params.mesh.amr_max_level = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--amr-upper-fraction") == 0 && i+1 < argc)
            params.mesh.amr_upper_fraction = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--amr-lower-fraction") == 0 && i+1 < argc)
            params.mesh.amr_lower_fraction = std::stod(argv[++i]);
    }

    // Default perturbation for Rosensweig
    // Zhang uses a step-function IC (no perturbation). The normal-field
    // instability develops from numerical noise once H exceeds the critical
    // value. A small perturbation can help seed it faster.
    if (preset == "rosensweig")
    {
        if (perturbation_amp < 0.0)
            perturbation_amp = 0.0;  // Zhang: none (step IC)
        if (perturbation_modes < 0)
            perturbation_modes = 0;
    }
    else
    {
        if (perturbation_amp < 0.0)
            perturbation_amp = 0.0;
        if (perturbation_modes < 0)
            perturbation_modes = 0;
    }

    // Final overrides (always applied)
    params.mesh.initial_refinement = refinement;
    params.output.verbose = verbose;
    params.output.folder = SOURCE_DIR "/Results";
    params.enable_mms = false;

    const double dt = params.time.dt;
    const double t_final = params.time.t_final;
    const unsigned int max_steps = (params.run.steps > 0)
        ? static_cast<unsigned int>(params.run.steps)
        : static_cast<unsigned int>(std::ceil(t_final / dt) + 10);
    const unsigned int vtk_interval = params.output.vtk_interval;

    const unsigned int max_picard = params.picard_iterations;
    const double picard_tol = params.picard_tolerance;
    const double omega = params.picard_relaxation;

    // ================================================================
    // 2. Output directory
    // ================================================================
    std::string output_dir;
    {
        char dir_buf[512];
        std::memset(dir_buf, 0, sizeof(dir_buf));

        if (rank == 0)
        {
            const std::string ts = get_timestamp();
            output_dir = params.output.folder + "/" + ts + "_"
                         + params.experiment_name
                         + "_r" + std::to_string(refinement);
            std::filesystem::create_directories(output_dir);
            std::strncpy(dir_buf, output_dir.c_str(), sizeof(dir_buf) - 1);
        }

        MPI_Bcast(dir_buf, 512, MPI_CHAR, 0, mpi_comm);
        output_dir = std::string(dir_buf);
    }

    // Save parameters for post-processing (convergence studies need epsilon, H0, etc.)
    if (rank == 0)
    {
        std::ofstream pf(output_dir + "/params.txt");
        pf << "experiment=" << params.experiment_name << "\n"
           << "refinement=" << refinement << "\n"
           << "dt=" << dt << "\n"
           << "t_final=" << t_final << "\n"
           << "epsilon=" << params.cahn_hilliard_params.epsilon << "\n"
           << "gamma=" << params.cahn_hilliard_params.gamma << "\n"
           << "sigma=" << params.cahn_hilliard_params.sigma << "\n"
           << "chi_ferro=" << params.physics.chi_ferro << "\n"
           << "nu_carrier=" << params.physics.nu_carrier << "\n"
           << "nu_ferro=" << params.physics.nu_ferro << "\n"
           << "mu_0=" << params.physics.mu_0 << "\n"
           << "T_relax=" << params.physics.T_relax << "\n"
           << "beta=" << params.physics.beta << "\n"
           << "H_max=" << params.uniform_field.intensity_max << "\n"
           << "ramp_time=" << params.uniform_field.ramp_time << "\n"
           << "field_dir_x=" << params.uniform_field.direction[0] << "\n"
           << "field_dir_y=" << params.uniform_field.direction[1] << "\n";
        pf.close();
    }

    // ================================================================
    // 3. Mesh
    // ================================================================
    parallel::distributed::Triangulation<dim> triangulation(
        mpi_comm,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening));

    FHDMesh::create_mesh<dim>(triangulation, params);

    const auto mesh_info = FHDMesh::get_mesh_info<dim>(triangulation);

    // ================================================================
    // 4. Setup all 6 subsystems
    // ================================================================
    PoissonSubsystem<dim> poisson(params, mpi_comm, triangulation);
    MagnetizationSubsystem<dim> mag(params, mpi_comm, triangulation);
    NavierStokesSubsystem<dim> ns(params, mpi_comm, triangulation);
    AngularMomentumSubsystem<dim> am(params, mpi_comm, triangulation);
    CahnHilliardSubsystem<dim> ch(params, mpi_comm, triangulation);

    poisson.setup();
    mag.setup();
    ns.setup();
    am.setup();
    ch.setup();

    const unsigned int phi_dofs = poisson.get_dof_handler().n_dofs();
    const unsigned int M_dofs = mag.get_dof_handler().n_dofs();
    const unsigned int vel_dofs = ns.get_ux_dof_handler().n_dofs();
    const unsigned int p_dofs = ns.get_p_dof_handler().n_dofs();
    const unsigned int w_dofs = am.get_dof_handler().n_dofs();
    const unsigned int ch_dofs = ch.get_dof_handler().n_dofs();
    const unsigned int total_dofs = phi_dofs + 2*M_dofs + 2*vel_dofs + p_dofs
                                  + w_dofs + ch_dofs;

    pcout << "\n"
          << "================================================================\n"
          << "  FHD + CAHN-HILLIARD: Two-Phase Ferrofluid Simulation\n"
          << "================================================================\n"
          << "  MPI ranks:      " << Utilities::MPI::n_mpi_processes(mpi_comm) << "\n"
          << "  Experiment:     " << params.experiment_name << "\n"
          << "  Domain:         [" << params.domain.x_min << ", " << params.domain.x_max
          << "] x [" << params.domain.y_min << ", " << params.domain.y_max << "]\n"
          << "  Refinement:     " << refinement
          << "  (" << mesh_info.n_global_active_cells << " cells"
          << ", h_min=" << std::scientific << std::setprecision(3) << mesh_info.h_min << ")\n"
          << "  Total DoFs:     " << total_dofs << "\n"
          << "    phi_mag=" << phi_dofs << " M=" << M_dofs << "x2"
          << " vel=" << vel_dofs << "x2 p=" << p_dofs
          << " w=" << w_dofs << " ch=" << ch_dofs << "\n"
          << "  dt:             " << dt << "\n"
          << "  t_final:        " << params.time.t_final << "\n"
          << "  max_steps:      " << max_steps << "\n"
          << "  Picard:         max=" << max_picard
          << "  tol=" << picard_tol << "  omega=" << omega << "\n"
          << "  Physics:\n"
          << "    chi_ferro=" << params.physics.chi_ferro
          << "  nu_carrier=" << params.physics.nu_carrier
          << "  nu_ferro=" << params.physics.nu_ferro << "\n"
          << "    mu_0=" << params.physics.mu_0
          << "  T_relax=" << params.physics.T_relax
          << "  beta=" << params.physics.beta << "\n"
          << "  Gravity:\n"
          << "    g=(" << params.physics.gravity_x << ", "
          << params.physics.gravity_y << ")"
          << "  r=" << params.physics.boussinesq_r << "\n"
          << "  CH params:\n"
          << "    epsilon=" << params.cahn_hilliard_params.epsilon
          << "  gamma=" << params.cahn_hilliard_params.gamma
          << "  sigma=" << params.cahn_hilliard_params.sigma << "\n"
          << "  Perturbation:   amp=" << perturbation_amp
          << "  modes=" << perturbation_modes << "\n"
          << "  Field:          H_max=" << params.uniform_field.intensity_max
          << "  ramp=" << params.uniform_field.ramp_time
          << "  dir=(" << params.uniform_field.direction[0]
          << ", " << params.uniform_field.direction[1] << ")\n"
          << "  Output:         " << output_dir << "\n"
          << "================================================================\n\n";

    // ================================================================
    // 5. Initialize solutions
    // ================================================================

    // Velocity: u_old = 0
    IndexSet ux_rel = DoFTools::extract_locally_relevant_dofs(
        ns.get_ux_dof_handler());
    IndexSet uy_rel = DoFTools::extract_locally_relevant_dofs(
        ns.get_uy_dof_handler());

    TrilinosWrappers::MPI::Vector ux_old_rel(
        ns.get_ux_dof_handler().locally_owned_dofs(), ux_rel, mpi_comm);
    TrilinosWrappers::MPI::Vector uy_old_rel(
        ns.get_uy_dof_handler().locally_owned_dofs(), uy_rel, mpi_comm);
    ux_old_rel = 0.0;
    uy_old_rel = 0.0;

    // Angular velocity: w_old = 0
    IndexSet w_rel_set = DoFTools::extract_locally_relevant_dofs(
        am.get_dof_handler());
    TrilinosWrappers::MPI::Vector w_old_rel(
        am.get_dof_handler().locally_owned_dofs(), w_rel_set, mpi_comm);
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
    TrilinosWrappers::MPI::Vector Mx_diff(M_owned, mpi_comm);
    TrilinosWrappers::MPI::Vector My_diff(M_owned, mpi_comm);

    Mx_old = 0.0;  My_old = 0.0;
    Mx_relaxed = 0.0;  My_relaxed = 0.0;

    // Cahn-Hilliard: initial condition
    if (preset == "rosensweig")
    {
        // Zhang Eq. (4.3): Φ(0) = 1 for y ≤ 0.2, Φ(0) = 0 for y > 0.2
        // Our convention: θ=+1 (ferro) at bottom, θ=−1 (carrier) above
        // FerrofluidPoolIC uses tanh profile for smooth transition
        const double y_interface = 0.2;
        const double domain_width = params.domain.x_max - params.domain.x_min;
        FerrofluidPoolIC<dim> ic(y_interface,
                                 params.cahn_hilliard_params.epsilon,
                                 perturbation_amp,
                                 static_cast<unsigned int>(perturbation_modes),
                                 domain_width);
        ch.initialize(ic);
    }
    else if (preset == "droplet")
    {
        Point<dim> center(0.5, 0.5);
        CircularDropletIC<dim> ic(center, 0.2,
                                  params.cahn_hilliard_params.epsilon);
        ch.initialize(ic);
    }
    else
    {
        // Default: droplet at center
        Point<dim> center(0.5, 0.5);
        CircularDropletIC<dim> ic(center, 0.25,
                                  params.cahn_hilliard_params.epsilon);
        ch.initialize(ic);
    }
    ch.update_ghosts();
    ch.save_old_solution();

    // Initial Poisson solve (M=0, determines h from h_a)
    poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                         mag.get_dof_handler(), 0.0);
    poisson.solve();
    poisson.update_ghosts();

    // ================================================================
    // 6. VTK output helper — ALL fields in a single file
    // ================================================================
    auto write_vtu = [&](unsigned int out_step, double /*out_time*/)
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(ns.get_ux_dof_handler());

        // NS fields
        data_out.add_data_vector(ns.get_ux_relevant(), "ux");
        data_out.add_data_vector(ns.get_uy_relevant(), "uy");
        data_out.add_data_vector(ns.get_p_dof_handler(),
                                 ns.get_p_relevant(), "p");

        // Magnetic potential, angular velocity
        data_out.add_data_vector(poisson.get_dof_handler(),
                                 poisson.get_solution_relevant(), "phi");
        data_out.add_data_vector(am.get_dof_handler(),
                                 am.get_relevant(), "w");

        // Magnetization
        data_out.add_data_vector(mag.get_dof_handler(),
                                 mag.get_Mx_relevant(), "Mx");
        data_out.add_data_vector(mag.get_dof_handler(),
                                 mag.get_My_relevant(), "My");

        // Cahn-Hilliard: theta (phase field) + mu (chemical potential)
        {
            const std::vector<std::string> ch_names = {"theta", "mu"};
            const std::vector<DataComponentInterpretation::DataComponentInterpretation>
                ch_interp(2, DataComponentInterpretation::component_is_scalar);
            data_out.add_data_vector(ch.get_dof_handler(),
                                     ch.get_relevant(),
                                     ch_names, ch_interp);
        }

        // Cell-averaged derived quantities
        const unsigned int n_cells = triangulation.n_active_cells();
        Vector<float> U_mag_cell(n_cells);
        Vector<float> M_mag_cell(n_cells);
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

            FEValues<dim> fe_mag(mag.get_dof_handler().get_fe(),
                                  q_mid, update_values);
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
        }
        data_out.add_data_vector(U_mag_cell, "U_mag",
                                 DataOut<dim>::type_cell_data);
        data_out.add_data_vector(M_mag_cell, "M_mag",
                                 DataOut<dim>::type_cell_data);

        // Subdomain
        Vector<float> subdomain(n_cells);
        for (const auto& cell : triangulation.active_cell_iterators())
            if (cell->is_locally_owned())
                subdomain[cell->active_cell_index()] =
                    static_cast<float>(cell->subdomain_id());
        data_out.add_data_vector(subdomain, "subdomain",
                                 DataOut<dim>::type_cell_data);

        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            output_dir + "/", "solution_",
            out_step, mpi_comm, 4, 0);
    };

    // Write initial VTK
    ns.update_ghosts();
    write_vtu(0, 0.0);

    // Initial CH diagnostics
    auto ch_diag0 = ch.compute_diagnostics();
    const double initial_mass = ch_diag0.phi_mass;
    const double initial_energy = ch_diag0.free_energy;

    pcout << "  Initial phi_mass: " << std::scientific << std::setprecision(6)
          << initial_mass << "\n"
          << "  Initial energy:   " << initial_energy << "\n\n";

    // ================================================================
    // 7. CSV diagnostics file
    // ================================================================
    std::ofstream csv_file;
    if (rank == 0)
    {
        csv_file.open(output_dir + "/diagnostics.csv");
        csv_file << "step,time,dt,picard_iters,picard_res,"
                 << "U_max,E_kin,divU_L2,"
                 << "p_min,p_max,"
                 << "kelvin_cell_L2,kelvin_Fx,kelvin_Fy,"
                 << "w_max_abs,M_max,H_max,"
                 << "phi_min,phi_max,phi_mass,free_energy,"
                 << "mu_L2,CFL,"
                 << "intf_x_min,intf_x_max,intf_y_min,intf_y_max,"
                 << "intf_amplitude,aspect_ratio,"
                 << "wall_s\n";
    }

    pcout << "  Step   Time         Picard  U_max        phi_mass     Energy       Wall(s)\n"
          << "  ----   ----------   ------  ----------   ----------   ----------   -------\n";

    // ================================================================
    // 8. Time stepping
    // ================================================================
    auto wall_start = std::chrono::high_resolution_clock::now();
    double current_time = 0.0;
    unsigned int step = 0;
    bool energy_monotone = true;
    double prev_energy = initial_energy;

    while (current_time < t_final - 1e-14 * dt && step < max_steps)
    {
        current_time += dt;
        ++step;

        Timer step_timer;
        step_timer.start();

        // ============================================================
        // AMR: Adaptive Mesh Refinement (if enabled)
        // ============================================================
        if (params.mesh.use_amr && step > 1 &&
            step % params.mesh.amr_interval == 0)
        {
            perform_amr<dim>(
                triangulation, params, mpi_comm,
                ch, ns, am, poisson, mag,
                ux_old_rel, uy_old_rel, w_old_rel,
                Mx_old, My_old, Mx_relaxed, My_relaxed);

            // Reinitialize Picard scratch vectors with new DoF distribution
            const IndexSet M_owned_new = mag.get_dof_handler().locally_owned_dofs();
            Mx_relaxed_owned.reinit(M_owned_new, mpi_comm);
            My_relaxed_owned.reinit(M_owned_new, mpi_comm);
            Mx_prev.reinit(M_owned_new, mpi_comm);
            My_prev.reinit(M_owned_new, mpi_comm);
            Mx_diff.reinit(M_owned_new, mpi_comm);
            My_diff.reinit(M_owned_new, mpi_comm);
        }

        // ============================================================
        // Phase 1: Picard iteration — Poisson <-> Magnetization
        //   with χ(φ_old) from Cahn-Hilliard
        // ============================================================
        Mx_relaxed = Mx_old;
        My_relaxed = My_old;

        unsigned int picard_iters = 0;
        double picard_res_final = 0.0;

        for (unsigned int k = 0; k < max_picard; ++k)
        {
            // Poisson: -Δφ = div(M - h_a)
            poisson.assemble_rhs(Mx_relaxed, My_relaxed,
                                 mag.get_dof_handler(),
                                 current_time);
            poisson.solve();
            poisson.update_ghosts();

            // Magnetization: with χ(φ_old) from CH
            mag.assemble(Mx_old, My_old,
                         poisson.get_solution_relevant(),
                         poisson.get_dof_handler(),
                         ux_old_rel, uy_old_rel,
                         ns.get_ux_dof_handler(),
                         dt, current_time,
                         w_old_rel, &am.get_dof_handler(),
                         ch.get_old_relevant(),
                         &ch.get_dof_handler());
            mag.solve();
            mag.update_ghosts();

            // Under-relax M
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

            // Convergence check
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

        // ============================================================
        // Phase 2: NS with ν(φ_old) + capillary + Kelvin
        // ============================================================
        ns.assemble(ux_old_rel, uy_old_rel,
                    dt, current_time,
                    /*include_convection=*/true,
                    w_old_rel, am.get_dof_handler(),
                    Mx_relaxed, My_relaxed,
                    &mag.get_dof_handler(),
                    poisson.get_solution_relevant(),
                    &poisson.get_dof_handler(),
                    ch.get_old_relevant(),
                    &ch.get_dof_handler());

        ns.solve();
        ns.update_ghosts();

        // ============================================================
        // Phase 3: Angular momentum
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

        am.solve();
        am.update_ghosts();

        // ============================================================
        // Phase 4: Cahn-Hilliard with convection from new velocity
        // ============================================================
        ch.assemble(ch.get_old_relevant(), dt,
                    ns.get_ux_relevant(), ns.get_uy_relevant(),
                    ns.get_ux_dof_handler());

        ch.solve();
        ch.update_ghosts();
        ch.save_old_solution();

        // ============================================================
        // Advance old solutions
        // ============================================================
        ux_old_rel = ns.get_ux_relevant();
        uy_old_rel = ns.get_uy_relevant();
        w_old_rel = am.get_relevant();
        Mx_old = Mx_relaxed;
        My_old = My_relaxed;

        step_timer.stop();
        const double step_wall = step_timer.wall_time();

        // ============================================================
        // Diagnostics
        // ============================================================
        auto ns_diag = ns.compute_diagnostics();
        auto mag_diag = mag.compute_diagnostics();
        auto poi_diag = poisson.compute_diagnostics();
        auto am_diag = am.compute_diagnostics();
        auto ch_diag = ch.compute_diagnostics();

        // Interface position tracking
        auto intf = compute_interface_position<dim>(
            ch.get_dof_handler(), ch.get_relevant(), mpi_comm);

        if (ch_diag.free_energy > prev_energy + 1e-12)
            energy_monotone = false;
        prev_energy = ch_diag.free_energy;

        const double CFL = (mesh_info.h_min > 0.0)
            ? ns_diag.U_max * dt / mesh_info.h_min : 0.0;

        // ============================================================
        // CSV output (every step)
        // ============================================================
        if (rank == 0)
        {
            csv_file << step << ","
                     << std::scientific << std::setprecision(8)
                     << current_time << "," << dt << ","
                     << picard_iters << "," << picard_res_final << ","
                     << ns_diag.U_max << "," << ns_diag.E_kin << ","
                     << ns_diag.divU_L2 << ","
                     << ns_diag.p_min << "," << ns_diag.p_max << ","
                     << ns_diag.kelvin_cell_L2 << ","
                     << ns_diag.kelvin_Fx << "," << ns_diag.kelvin_Fy << ","
                     << am_diag.w_max_abs << ","
                     << mag_diag.M_max << ","
                     << poi_diag.H_max << ","
                     << ch_diag.phi_min << "," << ch_diag.phi_max << ","
                     << ch_diag.phi_mass << "," << ch_diag.free_energy << ","
                     << ch_diag.mu_L2 << ","
                     << CFL << ","
                     << intf.x_min << "," << intf.x_max << ","
                     << intf.y_min << "," << intf.y_max << ","
                     << intf.amplitude << ","
                     << std::fixed << std::setprecision(4) << intf.aspect_ratio << ","
                     << std::fixed << std::setprecision(2) << step_wall
                     << "\n";
            csv_file << std::flush;
        }

        // ============================================================
        // Console output
        // ============================================================
        if (step % vtk_interval == 0 || step <= 5 ||
            current_time >= t_final - 1e-14*dt)
        {
            pcout << "  " << std::setw(4) << step
                  << "   " << std::scientific << std::setprecision(4)
                  << current_time
                  << "   " << std::setw(5) << picard_iters
                  << "   " << std::setprecision(2) << ns_diag.U_max
                  << "   " << ch_diag.phi_mass
                  << "   " << ch_diag.free_energy
                  << "   x=[" << std::fixed << std::setprecision(4)
                  << intf.x_min << "," << intf.x_max << "]"
                  << " y=[" << intf.y_min << "," << intf.y_max << "]"
                  << " AR=" << std::setprecision(3) << intf.aspect_ratio
                  << "   " << std::setprecision(1) << step_wall << "\n";
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
    // 9. Summary
    // ================================================================
    if (rank == 0 && csv_file.is_open())
        csv_file.close();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_time = std::chrono::duration<double>(wall_end - wall_start).count();

    auto ch_diag_final = ch.compute_diagnostics();
    auto ns_diag_final = ns.compute_diagnostics();
    const double mass_rel_change = (std::abs(initial_mass) > 1e-15)
        ? std::abs(ch_diag_final.phi_mass - initial_mass) / std::abs(initial_mass)
        : 0.0;

    pcout << "\n"
          << "================================================================\n"
          << "  SIMULATION COMPLETE: " << params.experiment_name << "\n"
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
          << "    Final:        " << ch_diag_final.phi_mass << "\n"
          << "    Rel. change:  " << mass_rel_change << "\n"
          << "\n"
          << "  Energy:\n"
          << "    Initial:      " << initial_energy << "\n"
          << "    Final:        " << ch_diag_final.free_energy << "\n"
          << "    Monotone:     " << (energy_monotone ? "YES" : "NO") << "\n"
          << "\n"
          << "  Final velocity: U_max = " << ns_diag_final.U_max << "\n"
          << "  Final M_max:    " << mag.compute_diagnostics().M_max << "\n"
          << "  Phase field:    [" << ch_diag_final.phi_min << ", "
          << ch_diag_final.phi_max << "]\n"
          << "\n"
          << "  Output:         " << output_dir << "\n"
          << "================================================================\n\n";

    return 0;
}
