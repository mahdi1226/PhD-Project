// ============================================================================
// utilities/parameters.cc - Parameter Parsing Implementation
//
// Simplified version without Logger dependency for standalone testing.
// ============================================================================

#include "parameters.h"

#include <iostream>
#include <cstdlib>
#include <cstring>

Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // Domain
        if (arg == "--refinement" && i + 1 < argc)
            params.domain.initial_refinement = std::stoul(argv[++i]);

        // Time
        else if (arg == "--dt" && i + 1 < argc)
            params.time.dt = std::stod(argv[++i]);
        else if ((arg == "--t_final" || arg == "--t-final") && i + 1 < argc)
            params.time.t_final = std::stod(argv[++i]);

        // CH
        else if (arg == "--epsilon" && i + 1 < argc)
            params.ch.epsilon = std::stod(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc)
            params.ch.gamma = std::stod(argv[++i]);
        else if (arg == "--eta" && i + 1 < argc)
            params.ch.eta = std::stod(argv[++i]);

        // IC
        else if (arg == "--ic_type" && i + 1 < argc)
            params.ic.type = std::stoi(argv[++i]);
        else if (arg == "--pool_depth" && i + 1 < argc)
            params.ic.pool_depth = std::stod(argv[++i]);
        else if (arg == "--perturbation" && i + 1 < argc)
            params.ic.perturbation = std::stod(argv[++i]);

        // MMS
        else if (arg == "--mms" || arg == "--mms_enabled")
            params.mms.enabled = true;
        else if (arg == "--mms_t_init" && i + 1 < argc)
            params.mms.t_init = std::stod(argv[++i]);

        // Magnetic
        else if (arg == "--magnetic")
            params.magnetic.enabled = true;
        else if (arg == "--chi_m" && i + 1 < argc)
            params.magnetization.chi_0 = std::stod(argv[++i]);
        else if (arg == "--dipole_intensity" && i + 1 < argc)
            params.dipoles.intensity_max = std::stod(argv[++i]);
        else if (arg == "--dipole_ramp" && i + 1 < argc)
            params.dipoles.ramp_time = std::stod(argv[++i]);

        // Navier-Stokes
        else if (arg == "--ns")
            params.ns.enabled = true;
        else if (arg == "--nu_water" && i + 1 < argc)
            params.ns.nu_water = std::stod(argv[++i]);
        else if (arg == "--nu_ferro" && i + 1 < argc)
            params.ns.nu_ferro = std::stod(argv[++i]);
        else if (arg == "--gravity")
            params.gravity.enabled = true;
        else if (arg == "--no-gravity")
            params.gravity.enabled = false;
        else if (arg == "--g_mag" && i + 1 < argc)
            params.gravity.magnitude = std::stod(argv[++i]);

        // Output
        else if (arg == "--output_dir" && i + 1 < argc)
            params.output.folder = argv[++i];
        else if (arg == "--output_frequency" && i + 1 < argc)
            params.output.frequency = std::stoul(argv[++i]);
        else if (arg == "--verbose")
            params.output.verbose = true;
        else if (arg == "--quiet")
            params.output.verbose = false;

        // Help
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Phase Field Ferrofluid Solver\n\n";
            std::cout << "Usage: ./ferrofluid [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --refinement <n>    Mesh refinement (default: 5)\n";
            std::cout << "  --dt <val>          Time step (default: 5e-4)\n";
            std::cout << "  --t-final <val>     Final time (default: 2.0)\n";
            std::cout << "  --epsilon <val>     Interface thickness (default: 0.01)\n";
            std::cout << "  --gamma <val>       Mobility (default: 0.0002)\n";
            std::cout << "  --ic_type <n>       0=droplet, 1=flat, 2=perturbed\n";
            std::cout << "\nMMS Verification:\n";
            std::cout << "  --mms               Enable MMS verification mode\n";
            std::cout << "  --mms_t_init <val>  MMS initial time (default: 0.1)\n";
            std::cout << "\nMagnetic Field:\n";
            std::cout << "  --magnetic          Enable magnetostatic Poisson solve\n";
            std::cout << "  --chi_m <val>       Magnetic susceptibility (default: 0.5)\n";
            std::cout << "  --dipole_intensity <val>  Dipole intensity (default: 6000)\n";
            std::cout << "  --dipole_ramp <val>       Ramp time (default: 1.6)\n";
            std::cout << "\nNavier-Stokes:\n";
            std::cout << "  --ns                Enable Navier-Stokes solve\n";
            std::cout << "  --nu_water <val>    Water viscosity (default: 1.0)\n";
            std::cout << "  --nu_ferro <val>    Ferrofluid viscosity (default: 2.0)\n";
            std::cout << "  --gravity / --no-gravity  Enable/disable gravity\n";
            std::cout << "  --g_mag <val>       Gravity magnitude (default: 30000)\n";
            std::cout << "\nOutput:\n";
            std::cout << "  --output_dir <path> Output directory\n";
            std::cout << "  --output_frequency <n> Output every N steps\n";
            std::cout << "  --verbose / --quiet Verbosity control\n";
            std::exit(0);
        }
    }

    // Validation warnings
    if (params.ch.eta > params.ch.epsilon)
    {
        std::cerr << "[Warning] eta > epsilon violates stability condition\n";
    }

    // Print summary
    if (params.output.verbose)
    {
        std::cout << "[Parameters]\n";
        std::cout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
                  << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "  Refinement: " << params.domain.initial_refinement << "\n";
        std::cout << "  CH: epsilon=" << params.ch.epsilon
                  << ", gamma=" << params.ch.gamma << "\n";
        std::cout << "  Time: dt=" << params.time.dt
                  << ", t_final=" << params.time.t_final << "\n";
        if (params.mms.enabled)
            std::cout << "  MMS: ENABLED (t_init=" << params.mms.t_init << ")\n";
        else
            std::cout << "  IC type: " << params.ic.type << "\n";
        if (params.magnetic.enabled)
            std::cout << "  Magnetic: ENABLED (χ₀=" << params.magnetization.chi_0 << ")\n";
        if (params.ns.enabled)
            std::cout << "  Navier-Stokes: ENABLED (ν_w=" << params.ns.nu_water
                      << ", ν_f=" << params.ns.nu_ferro << ")\n";
    }

    return params;
}