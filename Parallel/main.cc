// ============================================================================
// main.cc - Entry Point for Ferrofluid Phase Field Solver (PARALLEL)
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
//   mpirun -np 4 ./ferrofluid                    # Default parameters
//   mpirun -np 4 ./ferrofluid --help             # Show all options
//   mpirun -np 4 ./ferrofluid --mms              # MMS verification mode
//   mpirun -np 4 ./ferrofluid --magnetic --ns    # Full ferrofluid simulation
//
// ============================================================================
#include "core/phase_field.h"
#include "utilities/parameters.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <exception>

int main(int argc, char* argv[])
{
    // Initialize MPI (required for Trilinos AMG)
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    try
    {
        dealii::deallog.depth_console(0);

        const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

        Parameters params = Parameters::parse_command_line(argc, argv);

        PhaseFieldProblem<2> problem(params);
        problem.run();

        if (rank == 0)
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