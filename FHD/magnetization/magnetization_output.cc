// ============================================================================
// magnetization/magnetization_output.cc - VTK Output for Magnetization
//
// Writes parallel VTU/PVTU containing: Mx, My, M_mag
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>

template <int dim>
void MagnetizationSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double /*time*/) const
{
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(Mx_relevant_, "Mx");
    data_out.add_data_vector(My_relevant_, "My");

    // Compute |M| per cell
    dealii::Vector<float> M_mag(triangulation_.n_active_cells());
    const dealii::QMidpoint<dim> q_mid;
    dealii::FEValues<dim> fe_values(fe_, q_mid, dealii::update_values);

    unsigned int idx = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            std::vector<double> mx(1), my(1);
            fe_values.get_function_values(Mx_relevant_, mx);
            fe_values.get_function_values(My_relevant_, my);
            M_mag[idx] = static_cast<float>(
                std::sqrt(mx[0] * mx[0] + my[0] * my[0]));
        }
        ++idx;
    }
    data_out.add_data_vector(M_mag, "M_mag",
                             dealii::DataOut<dim>::type_cell_data);

    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "magnetization_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void MagnetizationSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void MagnetizationSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
