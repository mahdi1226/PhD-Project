// ============================================================================
// main.cc - Entry Point for Two-Phase Ferrofluid Solver
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "utilities/parameters.h"
#include "utilities/questions.h"
#include "output/logger.h"

#include <iostream>

void print_help(const char* prog)
{
    std::cout << "\nTwo-Phase Ferrofluid Solver\n";
    std::cout << "Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016)\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "  --help, -h       Show this help\n";
    std::cout << "  --questions, -q  Print open questions\n\n";
}

int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_help(argv[0]); return 0; }
        if (arg == "--questions" || arg == "-q") { print_questions(); return 0; }
    }
    
    try
    {
        Parameters params = Parameters::parse_command_line(argc, argv);
        
        Logger::info("========================================");
        Logger::info("Two-Phase Ferrofluid Solver");
        Logger::info("========================================");
        
        PhaseFieldProblem<2> problem(params);
        problem.run();
        
        Logger::success("Simulation completed!");
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}