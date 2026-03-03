// ============================================================================
// navier_stokes/navier_stokes_output.cc - VTK Output
//
// Writes parallel VTU/PVTU containing: ux, uy, U_mag, p
// Velocity uses CG Q2 DoFHandler, pressure uses DG P1.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <filesystem>

template <int dim>
void NavierStokesSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double /*time*/) const
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    // Velocity output (CG Q2)
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(ux_dof_handler_);
    data_out.add_data_vector(ux_relevant_, "ux");
    data_out.add_data_vector(uy_relevant_, "uy");

    // Compute |U| per cell
    dealii::Vector<float> U_mag(triangulation_.n_active_cells());
    const dealii::QMidpoint<dim> q_mid;
    dealii::FEValues<dim> fe_values(fe_velocity_, q_mid, dealii::update_values);

    unsigned int idx = 0;
    for (const auto& cell : ux_dof_handler_.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            std::vector<double> ux(1), uy(1);
            fe_values.get_function_values(ux_relevant_, ux);
            fe_values.get_function_values(uy_relevant_, uy);
            U_mag[idx] = static_cast<float>(
                std::sqrt(ux[0] * ux[0] + uy[0] * uy[0]));
        }
        ++idx;
    }
    data_out.add_data_vector(U_mag, "U_mag",
                             dealii::DataOut<dim>::type_cell_data);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "navier_stokes_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void NavierStokesSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void NavierStokesSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
