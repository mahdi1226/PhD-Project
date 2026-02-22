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
// Rosensweig preset (Zhang, He & Yang, SIAM J. Sci. Comput. 43 (2021),
// Section 4.3, Eq 4.4, Fig 4.7-4.9)
//
// Flat ferrofluid interface, applied H field from 5 dipoles below domain.
// Spikes form via Rosensweig instability.
//
// Domain: [0,1] × [0,0.6]
// IC: flat interface at y = 0.3, θ = tanh((0.3 - y)/(√2 ε))
// ε = 5e-3, M (mobility) = 5e-4, β = 1, λ = 1
// ν_f = ν_w = 1, μ₀ = 0.1, χ₀ = 2, g = 10
// Applied field: 5 dipoles at y=-15, slope=1000
// dt = 1e-4, h = 1/128
// ============================================================================
void Parameters::setup_rosensweig()
{
    // Domain
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;

    // Physics — Zhang Eq 4.4 (SIAM J. Sci. Comput. 43(1), 2021, p.B186)
    physics.epsilon = 5e-3;        // ε = 5e-3
    physics.chi_0 = 0.5;          // χ₀ = 0.5
    physics.mu_0 = 1.0;           // μ = 1

    // Physics — Magnetization (Zhang Eq 4.4: full PDE with β=1, τ=1e-4)
    physics.tau_M = 1e-4;         // Zhang Eq 4.4: τ = 1e-4
    physics.beta = 1.0;           // Zhang Eq 4.4: β = 1 (Landau-Lifshitz damping)
    physics.enable_beta_term = true;

    // Physics — Cahn-Hilliard
    physics.mobility = 2e-4;      // Zhang Eq 4.4: M = 2e-4
    physics.lambda = 1.0;         // Zhang Eq 4.4: λ = 1

    // Physics — Navier-Stokes
    physics.nu_water = 1.0;       // Zhang Eq 4.4: ν_w = 1
    physics.nu_ferro = 2.0;       // Zhang Eq 4.4: ν_f = 2
    physics.r = 0.1;              // Zhang Eq 4.4: r = 0.1 (density ratio)
    physics.gravity_magnitude = 6e4;    // Zhang Eq 4.4: g = 6e4
    physics.gravity_direction = {0.0, -1.0};

    // Applied field: 5 dipoles far below domain (y = -15)
    uniform_field.enabled = false;
    dipoles.positions.clear();
    dipoles.positions.push_back(dealii::Point<2>(-0.5, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 0.0, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 0.5, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 1.0, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 1.5, -15.0));
    dipoles.direction    = {0.0, 1.0};
    dipoles.ramp_slope   = 5000.0;   // Zhang: α from 0→8000 over [0,1.6], slope=8000/1.6=5000
    dipoles.ramp_time    = 0.0;
    dipoles.intensity_max = 8000.0;  // Zhang: cap at 8000 for t > 1.6

    // Time stepping — Zhang Eq 4.4: δt = 1e-3
    time.dt = 1e-3;
    time.t_final = 2.0;
    time.max_steps = 2000;

    // Mesh — Zhang Eq 4.4: h = 1e-2
    // r=4: 10*16=160 cells in x (h_x=1/160≈0.00625), finer than h=0.01
    mesh.initial_refinement = 4;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;
    use_reduced_magnetic_field = false;

    // Picard (not used in decoupled driver, but set for completeness)
    picard_iterations = 7;
    picard_tolerance = 0.01;

    // Zhang's SAV scheme — use FULL magnetization PDE (not algebraic)
    use_algebraic_magnetization = false;  // Zhang solves mag PDE (Eq 3.15-3.16)
    use_sav = true;
    sav.C0 = 1.0;
    sav.S1 = 0.0;   // auto-computed: S1 = lambda/(4*epsilon) = 50
    sav.S2 = 0.0;   // start with 0, increase if needed
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
        else if (std::strcmp(argv[i], "--max_steps") == 0)
        {
            if (++i >= argc) { std::cerr << "--max_steps requires a value\n"; std::exit(1); }
            params.time.max_steps = std::stoul(argv[i]);
        }

        // ---- Uniform applied field ----
        else if (std::strcmp(argv[i], "--uniform_field") == 0)
        {
            if (++i >= argc) { std::cerr << "--uniform_field requires intensity value\n"; std::exit(1); }
            params.uniform_field.enabled = true;
            params.uniform_field.intensity_max = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--field_ramp_time") == 0)
        {
            if (++i >= argc) { std::cerr << "--field_ramp_time requires a value\n"; std::exit(1); }
            params.uniform_field.ramp_time = std::stod(argv[i]);
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
        else if (std::strcmp(argv[i], "--picard_omega") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_omega requires a value\n"; std::exit(1); }
            params.picard_relaxation = std::stod(argv[i]);
        }

        // ---- SAV + Algebraic Magnetization ----
        else if (std::strcmp(argv[i], "--algebraic_M") == 0)
            params.use_algebraic_magnetization = true;
        else if (std::strcmp(argv[i], "--no_algebraic_M") == 0)
            params.use_algebraic_magnetization = false;
        else if (std::strcmp(argv[i], "--sav") == 0)
            params.use_sav = true;
        else if (std::strcmp(argv[i], "--no_sav") == 0)
            params.use_sav = false;
        else if (std::strcmp(argv[i], "--sav_S1") == 0)
        {
            if (++i >= argc) { std::cerr << "--sav_S1 requires a value\n"; std::exit(1); }
            params.sav.S1 = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--sav_S2") == 0)
        {
            if (++i >= argc) { std::cerr << "--sav_S2 requires a value\n"; std::exit(1); }
            params.sav.S2 = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--sav_C0") == 0)
        {
            if (++i >= argc) { std::cerr << "--sav_C0 requires a value\n"; std::exit(1); }
            params.sav.C0 = std::stod(argv[i]);
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

        // ---- Validation presets (Zhang, He & Yang, CMAME 371 (2020)) ----
        else if (std::strcmp(argv[i], "--validation") == 0)
        {
            if (++i >= argc) { std::cerr << "--validation requires a value (square|droplet)\n"; std::exit(1); }
            params.validation_test = argv[i];

            if (params.validation_test == "square")
            {
                // ============================================================
                // Square test (Section 4.2, Fig 4.3):
                // Pure CH relaxation — diamond → circle, NO magnetic, NO NS.
                //
                // Domain: [0,1]², h = 1/64 (r=6 on 1-cell base grid)
                // IC: diamond (L1 ball) radius 0.25 at center
                // ε = 0.01, M(θ) = γε² with γ = 1 → mobility = ε² = 1e-4
                // λ = 1/ε = 100 (for the energy functional)
                // dt = 1e-3, t_final = 5.0
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 1.0;
                params.domain.y_min = 0.0;
                params.domain.y_max = 1.0;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = false;
                params.enable_ns       = false;
                params.enable_gravity  = false;

                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.intensity_max = 0.0;

                params.physics.epsilon  = 0.01;
                params.physics.mobility = 1e-4;   // γε² with γ=1
                params.physics.lambda   = 100.0;  // 1/ε

                params.time.dt        = 1e-3;
                params.time.t_final   = 5.0;
                params.time.max_steps = 5000;

                params.mesh.initial_refinement = 6;  // h = 1/64
            }
            else if (params.validation_test == "droplet")
            {
                // ============================================================
                // Droplet deformation test (Zhang, He & Yang, SIAM J. Sci.
                // Comput. 43 (2021), Section 4.5, Eq 4.8, Fig 4.14-4.16):
                //
                // Circle in applied field → elongates vertically.
                //
                // Domain: [0,1]², h = 1/128 (r=7)
                // IC: circle at center, R = 0.1
                // ε = 2e-3, M (mobility) = 2e-4, β = 1, λ = 1
                // ν_f = ν_w = 1, ρ uniform
                // μ₀ = 0.1, χ₀ = 2, τ_M = 1e-6 (fast relaxation)
                // Applied field: 5 dipoles at y = -15, direction (0,1),
                //   intensity α(t) = 1000*t (linear ramp, slope=1000)
                // No gravity
                // dt = 1e-3, t_final = 1.5
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 1.0;
                params.domain.y_min = 0.0;
                params.domain.y_max = 1.0;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = true;
                params.enable_ns       = true;
                params.enable_gravity  = false;

                // Applied field: 5 dipoles far below domain (y = -15)
                // This produces a nearly uniform vertical field inside [0,1]²
                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.positions.push_back(dealii::Point<2>(-0.5, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 0.0, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 0.5, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 1.0, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 1.5, -15.0));
                params.dipoles.direction    = {0.0, 1.0};
                params.dipoles.ramp_slope   = 1000.0;   // α(t) = 1000*t
                params.dipoles.ramp_time    = 0.0;
                params.dipoles.intensity_max = 0.0;      // unused with ramp_slope

                params.physics.epsilon  = 2e-3;    // Zhang Eq 4.8
                params.physics.chi_0    = 2.0;     // Zhang Eq 4.8
                params.physics.mu_0     = 0.1;     // Zhang Eq 4.8: μ = 0.1
                params.physics.tau_M    = 1e-4;    // Zhang Eq 4.8: τ = 1e-4
                params.physics.beta     = 1.0;     // Zhang Eq 4.8: β = 1
                params.physics.enable_beta_term = true;
                params.physics.mobility = 2e-4;    // Zhang Eq 4.8: M = 2e-4
                params.physics.lambda   = 1.0;     // Zhang Eq 4.8
                params.physics.nu_water = 1.0;     // Zhang Eq 4.8: ν_w = 1
                params.physics.nu_ferro = 1.0;     // Zhang Eq 4.8: ν_f = 1
                params.physics.r        = 0.0;     // uniform density
                params.physics.gravity_magnitude = 0.0;

                params.time.dt        = 1e-3;      // Zhang Eq 4.8: τ = 1e-3
                params.time.t_final   = 1.5;
                params.time.max_steps = 1500;

                params.mesh.initial_refinement = 7;  // h = 1/128

                // Zhang's SAV scheme — use FULL magnetization PDE
                params.use_algebraic_magnetization = false;  // Zhang solves mag PDE
                params.use_sav = true;
                params.sav.C0 = 1.0;
                params.sav.S1 = 0.0;   // auto-computed: S1 = lambda/(4*epsilon) = 125
                params.sav.S2 = 0.0;   // start with 0
            }
            else if (params.validation_test == "droplet_nofield")
            {
                // ============================================================
                // Droplet baseline test — NO magnetic field.
                // Same as droplet (Zhang Eq 4.8) but magnetic OFF.
                // Circle should remain circular (pure CH + NS relaxation).
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 1.0;
                params.domain.y_min = 0.0;
                params.domain.y_max = 1.0;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = false;
                params.enable_ns       = true;
                params.enable_gravity  = false;

                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.intensity_max = 0.0;

                params.physics.epsilon  = 2e-3;
                params.physics.chi_0    = 2.0;
                params.physics.mu_0     = 0.1;
                params.physics.tau_M    = 1e-6;
                params.physics.mobility = 2e-4;
                params.physics.lambda   = 1.0;
                params.physics.nu_water = 1.0;
                params.physics.nu_ferro = 1.0;
                params.physics.r        = 0.0;
                params.physics.gravity_magnitude = 0.0;

                params.time.dt        = 1e-3;
                params.time.t_final   = 1.5;
                params.time.max_steps = 1500;

                params.mesh.initial_refinement = 7;

                params.use_algebraic_magnetization = true;
                params.use_sav = true;
                params.sav.C0 = 1.0;
                params.sav.S1 = 0.0;
                params.sav.S2 = 0.0;
            }
            else
            {
                std::cerr << "Unknown validation test: " << params.validation_test
                          << " (use 'square', 'droplet', or 'droplet_nofield')\n";
                std::exit(1);
            }
        }

        // ---- Output ----
        else if (std::strcmp(argv[i], "--vtk_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--vtk_interval requires a value\n"; std::exit(1); }
            params.output.vtk_interval = std::stoul(argv[i]);
        }
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
            std::cout << "    --rosensweig        Rosensweig preset (Section 6.2)\n";
            std::cout << "    --validation MODE   Validation test (droplet|square)\n\n";
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
            std::cout << "    --picard_iters N    Max Picard iterations (default: 7)\n";
            std::cout << "    --picard_omega V    Picard under-relaxation (default: 0.3)\n\n";
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
        std::cout << "  Picard: " << params.picard_iterations << " iters, ω="
                  << params.picard_relaxation << "\n";
        std::cout << "  Magnetic: " << (params.enable_magnetic ? "ON" : "OFF") << "\n";
        std::cout << "  NS: " << (params.enable_ns ? "ON" : "OFF")
                  << ", Gravity: " << (params.enable_gravity ? "ON" : "OFF") << "\n";
        std::cout << "  MMS: " << (params.enable_mms ? "ON" : "OFF") << "\n";
        if (params.uniform_field.enabled)
            std::cout << "  Uniform field: ON, |H_a|=" << params.uniform_field.intensity_max
                      << ", ramp=" << params.uniform_field.ramp_time << "\n";
        std::cout << "=====================\n";
    }

    return params;
}