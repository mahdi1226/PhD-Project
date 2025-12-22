// ============================================================================
// utilities/parameters.cc - Runtime Configuration
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "utilities/parameters.h"
#include <cstring>

void Parameters::setup_rosensweig()
{
    // Domain (Section 6.2, p.522)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;
    ic.pool_depth = 0.2;

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

    // Mesh
    mesh.initial_refinement = 5;
    mesh.use_amr = true;
    mesh.amr_interval = 5;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Output
    output.frequency = 25;
}

void Parameters::setup_hedgehog()
{
    // Domain (Section 6.3)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    ic.pool_depth = 0.11;

    // Time-stepping
    time.dt = 0.00025;
    time.t_final = 6.0;
    time.max_steps = 24000;

    // Mesh
    mesh.initial_refinement = 6;
    mesh.use_amr = true;
    mesh.amr_interval = 5;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Output
    output.frequency = 25;
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

        // Overrides
        else if (std::strcmp(argv[i], "--refinement") == 0 || std::strcmp(argv[i], "-r") == 0)
            params.mesh.initial_refinement = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--dt") == 0)
            params.time.dt = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--t_final") == 0)
            params.time.t_final = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--max_steps") == 0)
            params.time.max_steps = std::stoul(argv[++i]);

        // AMR
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no_amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr_interval") == 0)
            params.mesh.amr_interval = std::stoul(argv[++i]);

        // Solver
        else if (std::strcmp(argv[i], "--direct") == 0)
            params.solvers.ns.use_iterative = false;

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
            std::cout << "Usage: " << argv[0] << " --rosensweig|--hedgehog [options]\n\n";

            std::cout << "PRESETS (pick one):\n";
            std::cout << "  --rosensweig    Rosensweig instability (Section 6.2)\n";
            std::cout << "  --hedgehog      Hedgehog instability (Section 6.3)\n\n";

            std::cout << "OVERRIDES:\n";
            std::cout << "  --refinement N  Mesh refinement level\n";
            std::cout << "  --dt DT         Time step size\n";
            std::cout << "  --t_final T     Final simulation time\n";
            std::cout << "  --max_steps N   Maximum number of steps\n\n";

            std::cout << "AMR:\n";
            std::cout << "  --amr / --no_amr      Enable/disable AMR\n";
            std::cout << "  --amr_interval N      Refine every N steps\n\n";

            std::cout << "SOLVER:\n";
            std::cout << "  --direct              Use direct solver (recommended)\n\n";

            std::cout << "DEBUGGING:\n";
            std::cout << "  --mms                 MMS verification mode\n";
            std::cout << "  --no_magnetic         Disable magnetic forces\n";
            std::cout << "  --no_gravity          Disable gravity\n";
            std::cout << "  --no_ns               Disable Navier-Stokes\n";
            std::cout << "  --verbose             Verbose output\n\n";

            std::cout << "OUTPUT:\n";
            std::cout << "  Results saved to: ../Results/run-<timestamp>/\n";

            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    if (params.output.verbose)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  dt=" << params.time.dt << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  AMR: " << (params.mesh.use_amr ? "ON" : "OFF");
        if (params.mesh.use_amr)
            std::cout << " (every " << params.mesh.amr_interval << " steps)";
        std::cout << "\n";
        std::cout << "  Solver: " << (params.solvers.ns.use_iterative ? "Iterative" : "Direct") << "\n";
        std::cout << "=====================\n";
    }

    return params;
}