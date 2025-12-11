// ============================================================================
// main.cc - Coupled NS/CH/Poisson Ferrofluid Solver Entry Point
// ============================================================================
#include "core/nsch_problem.h"
#include "utilities/nsch_parameters.h"

#include <iostream>
#include <exception>

int main(int argc, char* argv[])
{
    try
    {
        dealii::deallog.depth_console(0);

        // Parse command line using existing infrastructure
        NSCHParameters params = parse_command_line(argc, argv);

        // Create and run problem
        NSCHProblem<2> problem(params);
        problem.run();
    }
    catch (std::exception& exc)
    {
        std::cerr << "\n----------------------------------------------------\n"
                  << "Exception: " << exc.what() << "\n"
                  << "Aborting!\n"
                  << "----------------------------------------------------\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "\nUnknown exception! Aborting!\n";
        return 1;
    }

    return 0;
}