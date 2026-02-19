// ============================================================================
// navier_stokes/navier_stokes_output.cc — VTK Output Implementation
//
// Implements NSSubsystem<dim>::write_vtu():
//   Writes parallel VTU/PVTU files for the NS solution.
//   Uses deal.II DataOut with multiple DoFHandlers (v9.4+ feature).
//
// Each field (ux, uy, p) lives on its own DoFHandler with its own FE space:
//   ux, uy: FE_Q<dim>(2)   — Q2 continuous
//   p:      FE_DGP<dim>(1) — DG P1 discontinuous
//
// The caller provides a timestamped output directory, e.g.:
//   navier_stokes_results/vtk/021826_160500_navier_stokes/
// Inside it, deal.II default naming is used:
//   solution_00002.pvtu, solution_00002.0.vtu, ...
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>
#include <iostream>
#include <iomanip>

template <int dim>
void NSSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time) const
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    // --- Ensure output directory exists ---
    if (rank == 0)
    {
        std::filesystem::create_directories(output_dir);
    }
    MPI_Barrier(mpi_comm_);

    // --- Build DataOut with multiple DoFHandlers ---
    dealii::DataOut<dim> data_out;

    // Attach primary DoFHandler (velocity ux space)
    data_out.attach_dof_handler(ux_dof_handler_);

    // Add velocity components (ux uses the attached DoFHandler)
    data_out.add_data_vector(ux_relevant_, "ux");

    // Add uy using its own DoFHandler (same FE space, separate handler)
    data_out.add_data_vector(uy_dof_handler_, uy_relevant_, "uy");

    // Add pressure using its own DoFHandler (FE_DGP, different space)
    data_out.add_data_vector(p_dof_handler_, p_relevant_, "p");

    // --- Add subdomain IDs for parallel visualization ---
    dealii::Vector<float> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    // --- Build patches ---
    data_out.build_patches(fe_velocity_.degree);

    // --- Set VTK flags ---
    dealii::DataOutBase::VtkFlags vtk_flags;
    vtk_flags.time = time;
    vtk_flags.cycle = step;
    vtk_flags.compression_level =
        dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    // --- Write parallel VTU + PVTU files ---
    // Produces: {output_dir}/solution_{step:05d}.{rank}.vtu
    //           {output_dir}/solution_{step:05d}.pvtu
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
template void NSSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;

template void NSSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
