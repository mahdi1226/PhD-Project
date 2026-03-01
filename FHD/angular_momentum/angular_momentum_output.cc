// ============================================================================
// angular_momentum/angular_momentum_output.cc - VTK Output
//
// Writes parallel VTU/PVTU containing: w (angular velocity)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "angular_momentum/angular_momentum.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>

template <int dim>
void AngularMomentumSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double /*time*/) const
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(w_relevant_, "w");

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "angular_momentum_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void AngularMomentumSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void AngularMomentumSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
