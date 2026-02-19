// ============================================================================
// utilities/parameters.cc - Runtime Configuration
//
// Includes presets and CLI options for all subsystems:
//   Poisson, Magnetization, Cahn-Hilliard, Navier-Stokes
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Zhang, He & Yang, CMAME 371 (2020) — β-term extension
// ============================================================================

#include "utilities/parameters.h"

#include <cstring>
#include <iostream>
#include <mpi.h>

// ============================================================================
// Rosensweig preset (Section 6.2, p.520-522)
// ============================================================================
void Parameters::setup_rosensweig()
{
    // Domain (Section 6.2, p.522)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;

    // Physics — shared
    physics.epsilon = 0.01;
    physics.chi_0 = 0.5;
    physics.mu_0 = 1.0;

    // Physics — Magnetization (Eq. 42c, Section 6.2)
    physics.tau_M = 1e-6;
    physics.beta = 0.0;
    physics.enable_beta_term = false;

    // Physics — Cahn-Hilliard (Eq. 42a-b, Section 6.2)
    physics.mobility = 0.0002;     // γ
    physics.lambda = 0.05;         // λ (surface tension)

    // Physics — Navier-Stokes (Eq. 42e-f, Section 6.2, p.520)
    //   "We choose the viscosities ν_w = 1.0 and ν_f = 2.0"
    //   "the density ρ implicitly taken ... to be unitary"
    //   "r = 0.1"
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;
    physics.gravity_magnitude = 30000.0;
    physics.gravity_direction = {0.0, -1.0};

    // Dipoles: 5 line dipoles below domain (Section 6.2, p.522)
    dipoles.positions = {
        dealii::Point<2>(-0.5, -15.0),
        dealii::Point<2>( 0.0, -15.0),
        dealii::Point<2>( 0.5, -15.0),
        dealii::Point<2>( 1.0, -15.0),
        dealii::Point<2>( 1.5, -15.0)
    };
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 6000.0;
    dipoles.ramp_time = 1.6;

    // Time stepping (Section 6.2)
    time.dt = 5e-4;
    time.t_final = 2.0;
    time.max_steps = 4000;

    // Mesh
    mesh.initial_refinement = 5;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;
    use_reduced_magnetic_field = false;

    // Picard
    picard_iterations = 7;
    picard_tolerance = 0.01;
}

// ============================================================================
// Command line parsing
// ============================================================================
Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // ---- Run mode / multi-ref / steps ----
        if (std::strcmp(argv[i], "--mode") == 0)
        {
            if (++i >= argc) { std::cerr << "--mode requires a value\n"; std::exit(1); }
            params.run.mode = argv[i];
        }
        else if (std::strcmp(argv[i], "--ref") == 0)
        {
            params.run.refs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                params.run.refs.push_back(std::stoul(argv[++i]));
            if (params.run.refs.empty()) { std::cerr << "--ref requires at least one value\n"; std::exit(1); }
        }
        else if (std::strcmp(argv[i], "--steps") == 0)
        {
            if (++i >= argc) { std::cerr << "--steps requires a value\n"; std::exit(1); }
            params.run.steps = std::stoi(argv[i]);
        }

        // ---- Presets ----
        else if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();

        // ---- Mesh ----
        else if (std::strcmp(argv[i], "--refinement") == 0 ||
                 std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            params.mesh.initial_refinement = std::stoul(argv[i]);
        }

        // ---- Time stepping ----
        else if (std::strcmp(argv[i], "--dt") == 0)
        {
            if (++i >= argc) { std::cerr << "--dt requires a value\n"; std::exit(1); }
            params.time.dt = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--t_final") == 0)
        {
            if (++i >= argc) { std::cerr << "--t_final requires a value\n"; std::exit(1); }
            params.time.t_final = std::stod(argv[i]);
        }

        // ---- Magnetization physics ----
        else if (std::strcmp(argv[i], "--tau_M") == 0)
        {
            if (++i >= argc) { std::cerr << "--tau_M requires a value\n"; std::exit(1); }
            params.physics.tau_M = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--beta") == 0)
        {
            if (++i >= argc) { std::cerr << "--beta requires a value\n"; std::exit(1); }
            params.physics.beta = std::stod(argv[i]);
            params.physics.enable_beta_term = (params.physics.beta != 0.0);
        }
        else if (std::strcmp(argv[i], "--enable_beta") == 0)
            params.physics.enable_beta_term = true;

        // ---- Cahn-Hilliard physics ----
        else if (std::strcmp(argv[i], "--mobility") == 0)
        {
            if (++i >= argc) { std::cerr << "--mobility requires a value\n"; std::exit(1); }
            params.physics.mobility = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--lambda") == 0)
        {
            if (++i >= argc) { std::cerr << "--lambda requires a value\n"; std::exit(1); }
            params.physics.lambda = std::stod(argv[i]);
        }

        // ---- Navier-Stokes physics ----
        else if (std::strcmp(argv[i], "--nu_water") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu_water requires a value\n"; std::exit(1); }
            params.physics.nu_water = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--nu_ferro") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu_ferro requires a value\n"; std::exit(1); }
            params.physics.nu_ferro = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--density_ratio") == 0)
        {
            if (++i >= argc) { std::cerr << "--density_ratio requires a value\n"; std::exit(1); }
            params.physics.r = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--gravity") == 0)
        {
            if (++i >= argc) { std::cerr << "--gravity requires a value\n"; std::exit(1); }
            params.physics.gravity_magnitude = std::stod(argv[i]);
        }

        // ---- Picard ----
        else if (std::strcmp(argv[i], "--picard_iters") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_iters requires a value\n"; std::exit(1); }
            params.picard_iterations = std::stoul(argv[i]);
        }

        // ---- Subsystem enables ----
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.enable_mms = true;
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.enable_magnetic = false;
        else if (std::strcmp(argv[i], "--reduced_field") == 0)
            params.use_reduced_magnetic_field = true;
        else if (std::strcmp(argv[i], "--no_ns") == 0)
            params.enable_ns = false;
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.enable_gravity = false;

        // ---- Output ----
        else if (std::strcmp(argv[i], "--verbose") == 0 ||
                 std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // ---- Help ----
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Ferrofluid Solver — Poisson + Magnetization + CH + NS\n";
            std::cout << "Reference: Nochetto et al. CMAME 309 (2016) Eq. 42a-42f\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "  Run mode:\n";
            std::cout << "    --mode <mms|2d|3d|temporal>  Run mode (default: mms)\n";
            std::cout << "    --ref 2 3 4 5 6    Refinement levels for mms/temporal\n";
            std::cout << "    --steps N          Override number of time steps\n\n";
            std::cout << "  Presets:\n";
            std::cout << "    --rosensweig        Rosensweig preset (Section 6.2)\n\n";
            std::cout << "  Mesh:\n";
            std::cout << "    --refinement N      Mesh refinement level (2d/3d modes)\n\n";
            std::cout << "  Time stepping:\n";
            std::cout << "    --dt VALUE          Time step size\n";
            std::cout << "    --t_final VALUE     Final simulation time\n\n";
            std::cout << "  Cahn-Hilliard:\n";
            std::cout << "    --mobility VALUE    CH mobility γ (default: 0.0002)\n";
            std::cout << "    --lambda VALUE      Surface tension λ (default: 0.05)\n\n";
            std::cout << "  Magnetization:\n";
            std::cout << "    --tau_M VALUE       Relaxation time (default: 1e-6)\n";
            std::cout << "    --beta VALUE        Landau-Lifshitz damping coefficient\n";
            std::cout << "    --enable_beta       Enable β-term even if β=0\n\n";
            std::cout << "  Navier-Stokes:\n";
            std::cout << "    --nu_water VALUE    Non-magnetic viscosity (default: 1.0)\n";
            std::cout << "    --nu_ferro VALUE    Ferrofluid viscosity (default: 2.0)\n";
            std::cout << "    --density_ratio V   Density ratio r (default: 0.1)\n";
            std::cout << "    --gravity VALUE     Gravity magnitude (default: 30000)\n\n";
            std::cout << "  Coupling:\n";
            std::cout << "    --picard_iters N    Max Picard iterations (default: 7)\n\n";
            std::cout << "  Subsystems:\n";
            std::cout << "    --mms               MMS verification mode\n";
            std::cout << "    --no_magnetic       Disable applied field\n";
            std::cout << "    --reduced_field     H = h_a only (dome setup)\n";
            std::cout << "    --no_ns             Disable Navier-Stokes\n";
            std::cout << "    --no_gravity        Disable gravity body force\n\n";
            std::cout << "  Output:\n";
            std::cout << "    --verbose           Verbose output\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
                  << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "  epsilon=" << params.physics.epsilon << "\n";
        std::cout << "  CH: mobility=" << params.physics.mobility
                  << ", lambda=" << params.physics.lambda << "\n";
        std::cout << "  Mag: chi_0=" << params.physics.chi_0
                  << ", tau_M=" << params.physics.tau_M << "\n";
        std::cout << "  beta=" << params.physics.beta
                  << " (enabled=" << (params.physics.enable_beta_term ? "yes" : "no") << ")\n";
        std::cout << "  NS: nu_w=" << params.physics.nu_water
                  << ", nu_f=" << params.physics.nu_ferro
                  << ", r=" << params.physics.r
                  << ", |g|=" << params.physics.gravity_magnitude << "\n";
        std::cout << "  dt=" << params.time.dt
                  << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Picard: " << params.picard_iterations << " iters\n";
        std::cout << "  Magnetic: " << (params.enable_magnetic ? "ON" : "OFF") << "\n";
        std::cout << "  NS: " << (params.enable_ns ? "ON" : "OFF")
                  << ", Gravity: " << (params.enable_gravity ? "ON" : "OFF") << "\n";
        std::cout << "  MMS: " << (params.enable_mms ? "ON" : "OFF") << "\n";
        std::cout << "=====================\n";
    }

    return params;
}