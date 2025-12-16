// ============================================================================
// utilities/parameters.cc - Parameter Parsing
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "utilities/parameters.h"
#include <cstring>
#include <stdexcept>

Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // ====================================================================
        // Preset configurations (USE THESE)
        // ====================================================================
        if (std::strcmp(argv[i], "--rosensweig") == 0)
        {
            params.setup_rosensweig();
        }
        else if (std::strcmp(argv[i], "--hedgehog") == 0)
        {
            params.setup_hedgehog();
        }

        // ====================================================================
        // Domain
        // ====================================================================
        else if (std::strcmp(argv[i], "--refinement") == 0 || std::strcmp(argv[i], "-r") == 0)
        {
            params.domain.initial_refinement = std::stoul(argv[++i]);
            params.mesh.initial_refinement = params.domain.initial_refinement;
        }
        else if (std::strcmp(argv[i], "--x_min") == 0)
            params.domain.x_min = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--x_max") == 0)
            params.domain.x_max = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--y_min") == 0)
            params.domain.y_min = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--y_max") == 0)
            params.domain.y_max = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--pool_depth") == 0 || std::strcmp(argv[i], "--layer_height") == 0)
        {
            params.ic.pool_depth = std::stod(argv[++i]);
            params.domain.layer_height = params.ic.pool_depth;
        }

        // ====================================================================
        // MMS
        // ====================================================================
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.mms.enabled = true;
        else if (std::strcmp(argv[i], "--mms_t_init") == 0 || std::strcmp(argv[i], "--t_init") == 0)
            params.mms.t_init = std::stod(argv[++i]);

        // ====================================================================
        // Cahn-Hilliard
        // ====================================================================
        else if (std::strcmp(argv[i], "--epsilon") == 0)
        {
            params.ch.epsilon = std::stod(argv[++i]);
            params.ch.eta = params.ch.epsilon;
        }
        else if (std::strcmp(argv[i], "--lambda") == 0)
            params.ch.lambda = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--mobility") == 0 || std::strcmp(argv[i], "--gamma") == 0)
            params.ch.gamma = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--eta") == 0)
            params.ch.eta = std::stod(argv[++i]);

        // ====================================================================
        // Magnetization
        // ====================================================================
        else if (std::strcmp(argv[i], "--chi_0") == 0)
            params.magnetization.chi_0 = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--tau_M") == 0)
        {
            double val = std::stod(argv[++i]);
            params.magnetization.tau_M = val;
            params.magnetization.T_relax = val;
        }

        // ====================================================================
        // Navier-Stokes
        // ====================================================================
        else if (std::strcmp(argv[i], "--ns") == 0)
            params.ns.enabled = true;
        else if (std::strcmp(argv[i], "--no_ns") == 0)
            params.ns.enabled = false;
        else if (std::strcmp(argv[i], "--nu_water") == 0)
            params.ns.nu_water = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--nu_ferro") == 0)
            params.ns.nu_ferro = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--mu_0") == 0)
            params.ns.mu_0 = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--r") == 0)
            params.ns.r = std::stod(argv[++i]);

        // ====================================================================
        // Gravity
        // ====================================================================
        else if (std::strcmp(argv[i], "--gravity") == 0)
            params.gravity.enabled = true;
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.gravity.enabled = false;
        else if (std::strcmp(argv[i], "--g") == 0)
            params.gravity.magnitude = std::stod(argv[++i]);

        // ====================================================================
        // Dipoles
        // ====================================================================
        else if (std::strcmp(argv[i], "--dipole_intensity") == 0)
            params.dipoles.intensity_max = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--dipole_ramp") == 0)
            params.dipoles.ramp_time = std::stod(argv[++i]);

        // ====================================================================
        // Magnetic
        // ====================================================================
        else if (std::strcmp(argv[i], "--magnetic") == 0)
            params.magnetic.enabled = true;
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.magnetic.enabled = false;
        else if (std::strcmp(argv[i], "--simplified") == 0)
            params.magnetic.use_simplified = true;
        else if (std::strcmp(argv[i], "--dg_transport") == 0)
            params.magnetic.use_dg_transport = true;
        else if (std::strcmp(argv[i], "--quasi_equilibrium") == 0)
            params.magnetic.use_dg_transport = false;

        // ====================================================================
        // Time
        // ====================================================================
        else if (std::strcmp(argv[i], "--dt") == 0)
            params.time.dt = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--t_final") == 0)
            params.time.t_final = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--max_steps") == 0)
            params.time.max_steps = std::stoul(argv[++i]);

        // ====================================================================
        // Mesh/AMR
        // ====================================================================
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no_amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr_min") == 0)
            params.mesh.amr_min_level = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--amr_max") == 0)
            params.mesh.amr_max_level = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--amr_interval") == 0)
            params.mesh.amr_interval = std::stoul(argv[++i]);

        // ====================================================================
        // Output
        // ====================================================================
        else if (std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0)
            params.output.folder = argv[++i];
        else if (std::strcmp(argv[i], "--output_frequency") == 0)
            params.output.frequency = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // ====================================================================
        // Solver
        // ====================================================================
        else if (std::strcmp(argv[i], "--direct") == 0)
            params.solver.use_direct = true;
        else if (std::strcmp(argv[i], "--iterative") == 0)
            params.solver.use_direct = false;
        else if (std::strcmp(argv[i], "--tol") == 0)
            params.solver.tolerance = std::stod(argv[++i]);

        // ====================================================================
        // Help
        // ====================================================================
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Ferrofluid Phase Field Solver\n";
            std::cout << "Reference: Nochetto et al. CMAME 309 (2016) 497-531\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";

            std::cout << "PRESET CONFIGURATIONS:\n";
            std::cout << "  --rosensweig         Rosensweig instability (Section 6.2)\n";
            std::cout << "  --hedgehog           Hedgehog instability (Section 6.3)\n\n";

            std::cout << "DOMAIN:\n";
            std::cout << "  --refinement N       Mesh refinement level\n";
            std::cout << "  --pool_depth H       Ferrofluid pool depth\n\n";

            std::cout << "CAHN-HILLIARD:\n";
            std::cout << "  --epsilon E          Interface thickness\n";
            std::cout << "  --lambda L           Capillary coefficient\n";
            std::cout << "  --gamma G            Mobility\n\n";

            std::cout << "MAGNETIC:\n";
            std::cout << "  --chi_0 C            Susceptibility\n";
            std::cout << "  --dipole_intensity I Dipole intensity\n";
            std::cout << "  --dipole_ramp T      Ramp time\n\n";

            std::cout << "TIME:\n";
            std::cout << "  --dt DT              Time step\n";
            std::cout << "  --t_final T          Final time\n";
            std::cout << "  --max_steps N        Maximum steps\n\n";

            std::cout << "OUTPUT:\n";
            std::cout << "  --output DIR         Output directory\n";
            std::cout << "  --output_frequency N Output every N steps\n";
            std::cout << "  --verbose            Verbose output\n\n";

            std::cout << "AMR:\n";
            std::cout << "  --amr / --no_amr     Enable/disable AMR\n";
            std::cout << "  --amr_interval N     Refine every N steps\n";

            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Sync aliases
    params.magnetization.T_relax = params.magnetization.tau_M;

    // Print configuration
    if (params.output.verbose)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Domain: [" << params.domain.x_min << ", " << params.domain.x_max
                  << "] x [" << params.domain.y_min << ", " << params.domain.y_max << "]\n";
        std::cout << "  Pool depth: " << params.ic.pool_depth << "\n";
        std::cout << "  Refinement: " << params.domain.initial_refinement << "\n";
        std::cout << "  epsilon=" << params.ch.epsilon << ", lambda=" << params.ch.lambda
                  << ", gamma=" << params.ch.gamma << "\n";
        std::cout << "  chi_0=" << params.magnetization.chi_0 << "\n";
        std::cout << "  Magnetic: " << (params.magnetic.enabled ? "ON" : "OFF") << "\n";
        std::cout << "  NS: " << (params.ns.enabled ? "ON" : "OFF") << "\n";
        std::cout << "  Gravity: " << (params.gravity.enabled ? "ON" : "OFF") << "\n";
        std::cout << "  AMR: " << (params.mesh.use_amr ? "ON" : "OFF") << "\n";
        std::cout << "  dt=" << params.time.dt << ", t_final=" << params.time.t_final << "\n";
        std::cout << "=====================\n";
    }

    return params;
}