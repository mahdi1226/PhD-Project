// ============================================================================
// output/vtk_writer.cc - VTK Output Writer
// ============================================================================

#include "vtk_writer.h"
#include "logger.h"

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

template <int dim>
VTKWriter<dim>::VTKWriter(const PhaseFieldProblem<dim>& problem)
    : problem_(problem)
    , output_dir_(problem.get_params().output.folder)
{
    Logger::info("      VTKWriter constructed, output to: " + output_dir_);
}

template <int dim>
void VTKWriter<dim>::set_output_directory(const std::string& dir)
{
    output_dir_ = dir;
}

template <int dim>
void VTKWriter<dim>::write(unsigned int step) const
{
    std::filesystem::create_directories(output_dir_);

    // Use θ's DoFHandler as primary (all Q2 fields share same mesh)
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(problem_.theta_dof_handler_);

    // Add all fields
    data_out.add_data_vector(problem_.theta_solution_, "theta");
    data_out.add_data_vector(problem_.psi_solution_, "psi");
    data_out.add_data_vector(problem_.phi_solution_, "phi");
    data_out.add_data_vector(problem_.mx_solution_, "mx");
    data_out.add_data_vector(problem_.my_solution_, "my");
    data_out.add_data_vector(problem_.ux_solution_, "ux");
    data_out.add_data_vector(problem_.uy_solution_, "uy");

    // Note: p is Q1, different size - output separately or skip for now

    data_out.build_patches();

    // Generate filename
    std::ostringstream filename;
    filename << output_dir_ << "/solution-"
             << std::setfill('0') << std::setw(5) << step << ".vtu";

    std::ofstream output(filename.str());
    data_out.write_vtu(output);

    Logger::info("      Wrote: " + filename.str());
}

template class VTKWriter<2>;
template class VTKWriter<3>;