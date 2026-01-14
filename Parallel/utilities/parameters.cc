// ============================================================================
// utilities/parameters.cc - Runtime Configuration
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "utilities/parameters.h"
#include <array>
#include <cstring>
#include <mpi.h>

void Parameters::setup_rosensweig()
{
    preset_name = "rosen";  // For auto-generating run_name

    // Domain (Section 6.2, p.522)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;
    ic.pool_depth = 0.2;

    // Physical parameters (Section 6.2, p.520-522)
    physics.epsilon = 0.01;       // interface thickness
    physics.mobility = 0.0002;    // γ
    physics.lambda = 0.05;        // capillary coefficient
    physics.chi_0 = 0.5;          // susceptibility
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;              // density ratio
    physics.gravity = 30000.0;    // non-dimensional gravity

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

    // Time-stepping
    time.dt = 5e-4;
    time.t_final = 2.0;
    time.max_steps = 4000;
    time.use_adaptive_dt = false;  // PAPER_MATCH: Paper uses fixed dt

    // Mesh
    mesh.initial_refinement = 5;
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    // AMR levels (Paper Section 6.1: maintain ~20 elements across interface)
    mesh.amr_min_level = mesh.initial_refinement - 2;  // Don't coarsen below level 3
    mesh.amr_max_level = mesh.initial_refinement + 2;  // Allow refinement up to level 7

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Output
    output.frequency = 10;
}

void Parameters::setup_hedgehog()
{
    preset_name = "hedge";  // For auto-generating run_name

    // Domain (Section 6.3, p.527)
    // Same rectangular domain as Rosensweig: Ω = (0,1) × (0,0.6)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    ic.pool_depth = 0.11;

    // Physical parameters (Section 6.3, p.527) - DIFFERENT from Rosensweig!
    physics.epsilon = 0.005;      // interface thickness (HALF of Rosensweig)
    physics.mobility = 0.0002;    // γ (same)
    physics.lambda = 0.025;       // capillary coefficient (HALF of Rosensweig)
    physics.chi_0 = 0.9;          // susceptibility (HIGHER than Rosensweig: 0.5)
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;              // density ratio
    physics.gravity = 30000.0;    // non-dimensional gravity

    // Dipoles (Section 6.3, p.527-528)
    // 42 dipoles arranged in 3 rows × 14 columns
    // Approximating a bar magnet of 0.4 width × 0.5 height
    // Rows at y = -0.5, -0.75, -1.0
    // X-positions: equi-distributed from x = 0.3 to x = 0.7 (centered on domain)
    dipoles.positions.clear();
    const double x_min_dipole = 0.3;
    const double x_max_dipole = 0.7;
    const double x_spacing = (x_max_dipole - x_min_dipole) / 13.0;
    const std::array<double, 3> y_rows = {-0.5, -0.75, -1.0};

    for (double y : y_rows)
    {
        for (int j = 0; j < 14; ++j)
        {
            double x = x_min_dipole + j * x_spacing;
            dipoles.positions.push_back(dealii::Point<2>(x, y));
        }
    }

    dipoles.direction = {0.0, 1.0};
    // FIXED: Paper Section 6.3 specifies αs = 4.3 (NOT 6000!)
    // "αs is increased linearly in time from αs = 0 at time t = 0,
    //  to its maximum value αs = 4.3 at time t = 4.2"
    dipoles.intensity_max = 4.3;
    dipoles.ramp_time = 4.2;

    // Time-stepping
    // Paper Section 6.3: No explicit dt given, but using similar approach to Rosensweig
    // With t_final=6.0, use dt=0.001 (similar to Rosensweig ref5)
    time.dt = 0.001;
    time.t_final = 6.0;
    time.max_steps = 6000;
    time.use_adaptive_dt = false;  // PAPER_MATCH: Paper uses fixed dt

    // Mesh
    mesh.initial_refinement = 6;
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    // AMR levels (Paper Section 6.1)
    mesh.amr_min_level = mesh.initial_refinement - 2;  // Don't coarsen below level 4
    mesh.amr_max_level = mesh.initial_refinement + 2;  // Allow refinement up to level 8

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Output
    output.frequency = 10;
}

void Parameters::setup_dome()
{
    preset_name = "dome";  // For auto-generating run_name

    // Same as hedgehog but WITHOUT demagnetizing field
    // Results in dome shape (Fig. 7) instead of spikes (Fig. 6)

    // Domain (same as hedgehog)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    ic.pool_depth = 0.11;
    time.use_adaptive_dt = false;

    // Physical parameters (same as Hedgehog, Section 6.3)
    physics.epsilon = 0.005;
    physics.mobility = 0.0002;
    physics.lambda = 0.025;
    physics.chi_0 = 0.9;
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;
    physics.gravity = 30000.0;

    // Same dipoles as hedgehog
    dipoles.positions.clear();
    const double x_min_dipole = 0.3;
    const double x_max_dipole = 0.7;
    const double x_spacing = (x_max_dipole - x_min_dipole) / 13.0;
    const std::array<double, 3> y_rows = {-0.5, -0.75, -1.0};

    for (double y : y_rows)
    {
        for (int j = 0; j < 14; ++j)
        {
            double x = x_min_dipole + j * x_spacing;
            dipoles.positions.push_back(dealii::Point<2>(x, y));
        }
    }

    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 4.3;
    dipoles.ramp_time = 4.2;

    // KEY DIFFERENCE: Use reduced field (h = ha only, skip Poisson for hd)
    use_reduced_magnetic_field = true;  // NEW FLAG

    // Time-stepping (use paper's values!)
    time.dt = 0.001;           // Paper: 2000 steps for t_final=6 → dt=0.003
    time.t_final = 6.0;
    time.max_steps = 6000;
    time.use_adaptive_dt = false;  // Disable adaptive!

    // Same mesh as hedgehog (Paper Section 6.3)
    mesh.initial_refinement = 6;
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    mesh.amr_min_level = mesh.initial_refinement - 2;  // Level 4
    mesh.amr_max_level = mesh.initial_refinement + 2;  // Level 8

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Output more frequently to see evolution
    output.frequency = 10;
}

void Parameters::setup_droplet()
{
    preset_name = "drop";  // For auto-generating run_name

    // Domain: unit square
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 1.0;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 10;

    // Initial condition: circular droplet
    ic.type = 2;
    ic.droplet_center_x = 0.5;
    ic.droplet_center_y = 0.5;
    ic.droplet_radius = 0.25;

    // Physical parameters (same as Rosensweig defaults)
    physics.epsilon = 0.01;
    physics.mobility = 0.0002;
    physics.lambda = 0.05;
    physics.chi_0 = 0.5;
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;
    physics.gravity = 30000.0;

    // No dipoles (field-free test case)
    dipoles.positions.clear();
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 0.0;
    dipoles.ramp_time = 0.0;

    // Time-stepping (larger dt for faster testing)
    time.dt = 1e-3;
    time.t_final = 1.0;
    time.max_steps = 1000;

    // Mesh
    mesh.initial_refinement = 5;
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    // AMR levels
    mesh.amr_min_level = mesh.initial_refinement - 2;
    mesh.amr_max_level = mesh.initial_refinement + 2;

    // Subsystems: NS only, no magnetic or gravity
    enable_magnetic = false;
    enable_ns = true;
    enable_gravity = false;

    // Output
    output.frequency = 10;
}

// ============================================================================
// finalize_run_name() - Build run_name from preset + refinement + amr
// Call after all command line parsing is complete
// ============================================================================
void Parameters::finalize_run_name()
{
    // If user specified --run_name, use it as-is
    if (!output.run_name.empty())
        return;

    // Auto-generate: preset-rN[-amr]
    output.run_name = preset_name + "-r" + std::to_string(mesh.initial_refinement);

    if (mesh.use_amr)
        output.run_name += "-amr";
    // Update AMR levels if they weren't explicitly set (still at default 0)
    // This handles --refinement override case
    if (mesh.use_amr && mesh.amr_min_level == 0 && mesh.amr_max_level == 0)
    {
        mesh.amr_min_level = (mesh.initial_refinement > 2) ? mesh.initial_refinement - 2 : 0;
        mesh.amr_max_level = mesh.initial_refinement + 2;
    }
}

Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // Presets
        if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();
        else if (std::strcmp(argv[i], "--hedgehog") == 0)
            params.setup_hedgehog();
        else if (std::strcmp(argv[i], "--droplet") == 0)
            params.setup_droplet();
        else if (std::strcmp(argv[i], "--dome") == 0)
            params.setup_dome();

        // Overrides
        else if (std::strcmp(argv[i], "--refinement") == 0 || std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            params.mesh.initial_refinement = std::stoul(argv[i]);
        }
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

        // AMR
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no_amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr_interval requires a value\n"; std::exit(1); }
            params.mesh.amr_interval = std::stoul(argv[i]);
        }

        // Adaptive time stepping
        else if (std::strcmp(argv[i], "--adaptive_dt") == 0)
            params.time.use_adaptive_dt = true;
        else if (std::strcmp(argv[i], "--no_adaptive_dt") == 0)
            params.time.use_adaptive_dt = false;

        // Solver
        else if (std::strcmp(argv[i], "--direct") == 0)
        {
            params.solvers.ns.use_iterative = false;
            params.solvers.ch.use_iterative = false;  // Also use direct for CH
        }

        else if (std::strcmp(argv[i], "--dg_transport") == 0)
            params.use_dg_transport = true;
        else if (std::strcmp(argv[i], "--no_dg_transport") == 0)
            params.use_dg_transport = false;

        // Run name (NEW)
        else if (std::strcmp(argv[i], "--run_name") == 0)
        {
            if (++i >= argc) { std::cerr << "--run_name requires a value\n"; std::exit(1); }
            params.output.run_name = argv[i];
        }

        // Debugging
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.enable_mms = true;
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.enable_magnetic = false;
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.enable_gravity = false;
        else if (std::strcmp(argv[i], "--no_ns") == 0)
            params.enable_ns = false;
        else if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // Help
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Ferrofluid Phase Field Solver\n";
            std::cout << "Reference: Nochetto et al. CMAME 309 (2016) 497-531\n\n";
            std::cout << "Usage: " << argv[0] << " --rosensweig|--hedgehog|--droplet [options]\n";
            std::cout << "  (Presets set defaults; later options override them)\n\n";

            std::cout << "PRESETS (pick one):\n";
            std::cout << "  --rosensweig    Rosensweig instability (Section 6.2)\n";
            std::cout << "  --hedgehog      Hedgehog instability (Section 6.3)\n";
            std::cout << "  --dome          Hedgehog with h=ha only (Fig. 7 - dome)\n";
            std::cout << "  --droplet       Simple droplet (no magnetic, no gravity)\n\n";

            std::cout << "OVERRIDES:\n";
            std::cout << "  --refinement N  Mesh refinement level\n";
            std::cout << "  --dt DT         Time step size\n";
            std::cout << "  --t_final T     Final simulation time\n";
            std::cout << "  --max_steps N   Maximum number of steps\n\n";

            std::cout << "AMR:\n";
            std::cout << "  --amr / --no_amr      Enable/disable AMR\n";
            std::cout << "  --amr_interval N      Refine every N steps\n\n";

            std::cout << "TIME STEPPING:\n";
            std::cout << "  --adaptive_dt         Enable adaptive time stepping\n";
            std::cout << "  --no_adaptive_dt      Disable adaptive time stepping (paper default)\n\n";

            std::cout << "SOLVER:\n";
            std::cout << "  --direct              Use direct solver (recommended)\n";
            std::cout << "  --dg_transport        Enable DG magnetization transport\n";
            std::cout << "  --no_dg_transport     Disable DG transport (quasi-equilibrium)\n\n";

            std::cout << "OUTPUT:\n";
            std::cout << "  --run_name NAME       Custom run name (default: auto-generated)\n";
            std::cout << "                        e.g., rosen-r5-amr, dome-r4, hedge-r5\n";
            std::cout << "  Results saved to: ../Results/<run_name>-<timestamp>/\n\n";

            std::cout << "DEBUGGING:\n";
            std::cout << "  --mms                 MMS verification mode\n";
            std::cout << "  --no_magnetic         Disable magnetic forces\n";
            std::cout << "  --no_gravity          Disable gravity\n";
            std::cout << "  --no_ns               Disable Navier-Stokes\n";
            std::cout << "  --verbose             Verbose output\n";

            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Finalize run_name after all parsing is done
    params.finalize_run_name();

    // Only print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Run name: " << params.output.run_name << "\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  dt=" << params.time.dt << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Adaptive dt: " << (params.time.use_adaptive_dt ? "ON" : "OFF") << "\n";
        std::cout << "  AMR: " << (params.mesh.use_amr ? "ON" : "OFF");
        if (params.mesh.use_amr)
            std::cout << " (every " << params.mesh.amr_interval << " steps)";
        std::cout << "\n";
        std::cout << "  Physics: epsilon=" << params.physics.epsilon
                  << ", lambda=" << params.physics.lambda
                  << ", chi_0=" << params.physics.chi_0 << "\n";
        std::cout << "  Solver: " << (params.solvers.ns.use_iterative ? "Iterative" : "Direct") << "\n";
        std::cout << "  Subsystems: "
                  << (params.enable_magnetic ? "Magnetic " : "")
                  << (params.enable_ns ? "NS " : "")
                  << (params.enable_gravity ? "Gravity " : "")
                  << (params.enable_mms ? "MMS " : "") << "\n";
        std::cout << "=====================\n";
    }

    return params;
}