// ============================================================================
// cahn_hilliard/cahn_hilliard_output.cc - VTK Output
//
// Writes parallel VTU/PVTU containing: phi (phase field), mu (chemical potential)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>

template <int dim>
void CahnHilliardSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double /*time*/) const
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);

    std::vector<std::string> names = {"phi", "mu"};
    data_out.add_data_vector(solution_relevant_, names);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "cahn_hilliard_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void CahnHilliardSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void CahnHilliardSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
