// ============================================================================
// passive_scalar/passive_scalar_output.cc - VTK Output
//
// Writes parallel VTU/PVTU containing: c (concentration)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "passive_scalar/passive_scalar.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>

template <int dim>
void PassiveScalarSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double /*time*/) const
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(c_relevant_, "c");

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "passive_scalar_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void PassiveScalarSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void PassiveScalarSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
