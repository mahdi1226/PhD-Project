// ============================================================================
// utilities/parameters.cc - Parameter Parsing Implementation
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
        // Domain parameters
        if (std::strcmp(argv[i], "--refinement") == 0 ||
            std::strcmp(argv[i], "-r") == 0)
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
        else if (std::strcmp(argv[i], "--layer_height") == 0)
        {
            params.domain.layer_height = std::stod(argv[++i]);
            params.ic.pool_depth = params.domain.layer_height;
        }

        // Initial condition parameters
        else if (std::strcmp(argv[i], "--ic_type") == 0)
            params.ic.type = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pool_depth") == 0)
            params.ic.pool_depth = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--perturbation") == 0)
            params.ic.perturbation = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--perturbation_modes") == 0)
            params.ic.perturbation_modes = std::stoi(argv[++i]);

        // MMS parameters
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.mms.enabled = true;
        else if (std::strcmp(argv[i], "--mms_t_init") == 0 ||
                 std::strcmp(argv[i], "--t_init") == 0)
            params.mms.t_init = std::stod(argv[++i]);

        // Cahn-Hilliard parameters
        else if (std::strcmp(argv[i], "--epsilon") == 0)
        {
            params.ch.epsilon = std::stod(argv[++i]);
            params.ch.eta = params.ch.epsilon;  // Default: η = ε
        }
        else if (std::strcmp(argv[i], "--lambda") == 0)
            params.ch.lambda = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--mobility") == 0 ||
                 std::strcmp(argv[i], "--gamma") == 0)
            params.ch.gamma = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--eta") == 0)
            params.ch.eta = std::stod(argv[++i]);

        // Magnetization parameters
        else if (std::strcmp(argv[i], "--chi_0") == 0)
            params.magnetization.chi_0 = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--T_relax") == 0)
            params.magnetization.T_relax = std::stod(argv[++i]);

        // Navier-Stokes parameters
        else if (std::strcmp(argv[i], "--ns") == 0)
            params.ns.enabled = true;
        else if (std::strcmp(argv[i], "--nu_water") == 0)
            params.ns.nu_water = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--nu_ferro") == 0)
            params.ns.nu_ferro = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--mu_0") == 0)
            params.ns.mu_0 = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--density_ratio") == 0 ||
                 std::strcmp(argv[i], "--r") == 0)
        {
            params.ns.r = std::stod(argv[++i]);
            params.ns.density_ratio = params.ns.r;
        }
        else if (std::strcmp(argv[i], "--grad_div") == 0)
            params.ns.grad_div = std::stod(argv[++i]);

        // Gravity parameters
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.gravity.enabled = false;
        else if (std::strcmp(argv[i], "--g") == 0 ||
                 std::strcmp(argv[i], "--gravity") == 0)
            params.gravity.magnitude = std::stod(argv[++i]);

        // Dipole parameters
        else if (std::strcmp(argv[i], "--dipole_intensity") == 0)
            params.dipoles.intensity_max = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--dipole_ramp") == 0)
            params.dipoles.ramp_time = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--dipole_y") == 0)
        {
            double y = std::stod(argv[++i]);
            // Update all dipole y-positions
            for (auto& pos : params.dipoles.positions)
                pos[1] = y;
        }

        // Magnetic model options
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.magnetic.enabled = false;
        else if (std::strcmp(argv[i], "--simplified") == 0)
            params.magnetic.use_simplified = true;

        // Time parameters
        else if (std::strcmp(argv[i], "--dt") == 0)
            params.time.dt = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--t_final") == 0)
            params.time.t_final = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--max_steps") == 0)
            params.time.max_steps = std::stoul(argv[++i]);
        else if (std::strcmp(argv[i], "--time_theta") == 0)
            params.time.theta = std::stod(argv[++i]);

        // Mesh/AMR parameters
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

        // Output parameters
        else if (std::strcmp(argv[i], "--output") == 0 ||
                 std::strcmp(argv[i], "-o") == 0)
        {
            params.output.folder = argv[++i];
            params.output.output_dir = params.output.folder;
        }
        else if (std::strcmp(argv[i], "--output_frequency") == 0 ||
                 std::strcmp(argv[i], "--output_interval") == 0)
        {
            params.output.frequency = std::stoul(argv[++i]);
            params.output.output_interval = params.output.frequency;
        }
        else if (std::strcmp(argv[i], "--verbose") == 0 ||
                 std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // Solver parameters
        else if (std::strcmp(argv[i], "--direct") == 0)
            params.solver.use_direct = true;
        else if (std::strcmp(argv[i], "--tol") == 0)
            params.solver.tolerance = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--max_iter") == 0)
            params.solver.max_iterations = std::stoul(argv[++i]);

        // Preset configurations
        else if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();
        else if (std::strcmp(argv[i], "--hedgehog") == 0)
            params.setup_hedgehog();

        // Help
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Preset configurations:\n"
                      << "  --rosensweig         Rosensweig instability (Section 6.2)\n"
                      << "  --hedgehog           Hedgehog instability (Section 6.3)\n\n"
                      << "Domain:\n"
                      << "  --refinement N       Initial mesh refinement level\n"
                      << "  --x_min, --x_max     Domain x-bounds\n"
                      << "  --y_min, --y_max     Domain y-bounds\n"
                      << "  --layer_height H     Ferrofluid pool depth\n\n"
                      << "Cahn-Hilliard:\n"
                      << "  --epsilon E          Interface thickness\n"
                      << "  --lambda L           Capillary coefficient\n"
                      << "  --mobility G         Mobility γ\n\n"
                      << "Magnetic:\n"
                      << "  --chi_0 C            Susceptibility (≤4)\n"
                      << "  --dipole_intensity I Maximum dipole intensity\n"
                      << "  --dipole_ramp T      Ramp time\n"
                      << "  --dipole_y Y         Dipole y-position\n"
                      << "  --simplified         Use h := h_a (skip Poisson)\n\n"
                      << "Navier-Stokes:\n"
                      << "  --ns                 Enable NS coupling\n"
                      << "  --nu_water, --nu_ferro  Viscosities\n"
                      << "  --g G                Gravity magnitude\n\n"
                      << "Time:\n"
                      << "  --dt DT              Time step\n"
                      << "  --t_final T          Final time\n\n"
                      << "Output:\n"
                      << "  --output DIR         Output directory\n"
                      << "  --output_frequency N Output every N steps\n"
                      << "  --verbose            Verbose output\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Print configuration summary
    if (params.output.verbose)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Domain: [" << params.domain.x_min << ", " << params.domain.x_max
                  << "] x [" << params.domain.y_min << ", " << params.domain.y_max << "]\n";
        std::cout << "  Refinement: " << params.domain.initial_refinement << "\n";
        std::cout << "  ε = " << params.ch.epsilon << ", λ = " << params.ch.lambda
                  << ", γ = " << params.ch.gamma << "\n";
        std::cout << "  χ₀ = " << params.magnetization.chi_0 << "\n";
        std::cout << "  dt = " << params.time.dt << ", t_final = " << params.time.t_final << "\n";

        if (params.mms.enabled)
            std::cout << "  MMS: ENABLED (t_init=" << params.mms.t_init << ")\n";
        else
            std::cout << "  IC type: " << params.ic.type << "\n";

        std::cout << "=====================\n";
    }

    return params;
}