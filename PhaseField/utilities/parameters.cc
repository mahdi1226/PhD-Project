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
            std::cout << "Cahn-Hilliard Standalone Test\n\n";
            std::cout << "Usage: ./ferrofluid [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --refinement <n>    Mesh refinement (default: 5)\n";
            std::cout << "  --mms               Enable MMS verification mode\n";
            std::cout << "  --mms_t_init <val>  MMS initial time (default: 0.1)\n";
            std::cout << "  --dt <val>          Time step (default: 5e-4)\n";
            std::cout << "  --t-final <val>     Final time (default: 2.0)\n";
            std::cout << "  --epsilon <val>     Interface thickness (default: 0.01)\n";
            std::cout << "  --gamma <val>       Mobility (default: 0.0002)\n";
            std::cout << "  --ic_type <n>       0=droplet, 1=flat, 2=perturbed\n";
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
        std::cout << "  IC type: " << params.ic.type << "\n";
        if (params.mms.enabled)
            std::cout << "  MMS: ENABLED (t_init=" << params.mms.t_init << ")\n";
        std::cout << "  Refinement: " << params.domain.initial_refinement << "\n";
        std::cout << "  CH: epsilon=" << params.ch.epsilon
                  << ", gamma=" << params.ch.gamma << "\n";
        std::cout << "  Time: dt=" << params.time.dt
                  << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  IC type: " << params.ic.type << "\n";
    }

    return params;
}