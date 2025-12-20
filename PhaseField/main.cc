// ============================================================================
// main.cc - Entry Point for Ferrofluid Phase Field Solver
//
// Full solver implementing Nochetto, Salgado & Tomas, CMAME 309 (2016)
//
// Subsystems:
//   - Cahn-Hilliard (θ, ψ): phase separation with convection
//   - Poisson (φ): magnetostatic potential, H = -∇φ
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
        dealii::deallog.depth_console(0);

        Parameters params = Parameters::parse_command_line(argc, argv);

        PhaseFieldProblem<2> problem(params);
        problem.run();

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