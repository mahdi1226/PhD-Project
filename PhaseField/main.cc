// ============================================================================
// main.cc - Entry Point for Cahn-Hilliard Standalone Test
//
// This is a standalone test for the CH subsystem before integrating NS/Poisson.
// With u = 0, this tests pure diffuse interface dynamics.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "utilities/parameters.h"

#include <iostream>
#include <exception>

int main(int argc, char* argv[])
{
    try
    {
        // Suppress deal.II console output
        dealii::deallog.depth_console(0);

        // Parse parameters
        Parameters params = Parameters::parse_command_line(argc, argv);

        // Create and run problem
        PhaseFieldProblem<2> problem(params);
        problem.run();

        std::cout << "\n[Success] CH standalone test completed.\n";
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