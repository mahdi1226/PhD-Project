// ============================================================================
// magnetization/magnetization_output.cc — VTK Output Implementation
//
// Implements MagnetizationSubsystem<dim>::write_vtu():
//   Writes parallel VTU/PVTU files for Mx, My, |M|.
//
// VTK fields:
//   Mx, My         Magnetization components (DG Q1, nodal)
//   M_magnitude    |M| (DG Q1, nodal)
//   subdomain      MPI partition coloring
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cmath>

template <int dim>
void MagnetizationSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time) const
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    // --- Ensure output directory exists ---
    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    // --- Compute |M| on DG DoFs ---
    dealii::IndexSet locally_owned = dof_handler_.locally_owned_dofs();
    const dealii::IndexSet locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_);

    dealii::TrilinosWrappers::MPI::Vector M_mag_owned(locally_owned, mpi_comm_);

    for (auto it = locally_owned.begin(); it != locally_owned.end(); ++it)
    {
        const auto i = *it;
        const double mx = Mx_relevant_[i];
        const double my = My_relevant_[i];
        M_mag_owned[i] = std::sqrt(mx * mx + my * my);
    }
    M_mag_owned.compress(dealii::VectorOperation::insert);

    dealii::TrilinosWrappers::MPI::Vector M_mag_rel(locally_owned, locally_relevant, mpi_comm_);
    M_mag_rel = M_mag_owned;

    // --- DataOut ---
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);

    data_out.add_data_vector(Mx_relevant_, "Mx",
                             dealii::DataOut<dim>::type_dof_data);
    data_out.add_data_vector(My_relevant_, "My",
                             dealii::DataOut<dim>::type_dof_data);
    data_out.add_data_vector(M_mag_rel, "M_magnitude",
                             dealii::DataOut<dim>::type_dof_data);

    // Subdomain coloring
    dealii::Vector<float> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain",
                             dealii::DataOut<dim>::type_cell_data);

    data_out.build_patches(fe_.degree);

    // --- VTK flags ---
    dealii::DataOutBase::VtkFlags vtk_flags;
    vtk_flags.time = time;
    vtk_flags.cycle = step;
    vtk_flags.compression_level =
        dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    // --- Write parallel VTU + PVTU ---
    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "solution", step, mpi_comm_, /*n_digits=*/5);

    if (rank == 0)
    {
        std::cout << "  [VTK] " << output_dir << "/solution_"
                  << std::setfill('0') << std::setw(5) << step
                  << std::setfill(' ')
                  << ".pvtu (t=" << std::scientific << std::setprecision(3)
                  << time << ")\n" << std::defaultfloat;
    }
}


// ============================================================================
// Explicit instantiations
// ============================================================================
template void MagnetizationSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;

template void MagnetizationSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
