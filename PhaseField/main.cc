// ============================================================================
// main.cc - Entry Point for Ferrofluid Phase Field Solver
//
// Full solver implementing Nochetto, Salgado & Tomas, CMAME 309 (2016)
//
// Subsystems:
//   - Cahn-Hilliard (θ, ψ): phase separation with convection
//   - Poisson (φ): magnetostatic potential, H = ∇φ
//   - Magnetization (Mx, My): DG transport of M
//   - Navier-Stokes (ux, uy, p): fluid flow with Kelvin force B_h^m
//
// Run modes:
//   ./ferrofluid                           # Default parameters
//   ./ferrofluid --help                    # Show all options
//   ./ferrofluid --mms                     # MMS verification mode
//   ./ferrofluid --magnetic --ns           # Full ferrofluid simulation
//   ./ferrofluid --ic_type 0 --t_final 1.0 # Custom IC and runtime
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "utilities/parameters.h"

#include <deal.II/base/logstream.h>

#include <iostream>
#include <exception>

int main(int argc, char* argv[])
{
    try
    {
        // Suppress deal.II console output (keeps our output clean)
        dealii::deallog.depth_console(0);

        // Parse command line parameters
        Parameters params = Parameters::parse_command_line(argc, argv);

        // Print solver configuration
        std::cout << "============================================================\n";
        std::cout << "  Ferrofluid Phase Field Solver\n";
        std::cout << "  Reference: Nochetto et al. CMAME 309 (2016) 497-531\n";
        std::cout << "============================================================\n";
        std::cout << "\n";
        std::cout << "Configuration:\n";
        std::cout << "  Cahn-Hilliard:  ENABLED\n";
        std::cout << "  Magnetic:       " << (params.magnetic.enabled ? "ENABLED" : "disabled") << "\n";
        std::cout << "  Navier-Stokes:  " << (params.ns.enabled ? "ENABLED" : "disabled") << "\n";
        std::cout << "  MMS mode:       " << (params.mms.enabled ? "ENABLED" : "disabled") << "\n";
        std::cout << "  Gravity:        " << (params.gravity.enabled ? "ENABLED" : "disabled") << "\n";
        std::cout << "\n";

        // Create and run problem
        PhaseFieldProblem<2> problem(params);
        problem.run();

        std::cout << "\n[Success] Simulation completed.\n";
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "\n[Error] " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n[Error] Unknown exception!\n";
        return 1;
    }
}