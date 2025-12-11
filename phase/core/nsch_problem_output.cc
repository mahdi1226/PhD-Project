// ============================================================================
// core/nsch_problem_output.cc - VTK output for visualization
//
// Outputs three VTK files per timestep:
//   - ns_solution_XXXX.vtu  : velocity (ux, uy, magnitude) + pressure
//   - ch_solution_XXXX.vtu  : phase field (c) + chemical potential (mu)
//   - phi_solution_XXXX.vtu : magnetic potential (phi) - only if magnetic enabled
// ============================================================================
#include "core/nsch_problem.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <filesystem>

// ============================================================================
// Output results to VTK files
// ============================================================================
template <int dim>
void NSCHProblem<dim>::output_results(unsigned int output_number) const
{
    // Create output directory if it doesn't exist
    if (!params_.output_dir.empty())
        std::filesystem::create_directories(params_.output_dir);

    const std::string suffix = dealii::Utilities::int_to_string(output_number, 4);
    const std::string base_path = params_.output_dir.empty() ? "" : params_.output_dir + "/";

    // ========================================================================
    // 1. Navier-Stokes solution: ux, uy, |u| (on Q2 mesh)
    //    Pressure is on Q1 mesh, output separately or interpolate
    // ========================================================================
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(ux_dof_handler_);

        // Add velocity components (all Q2, same DoF structure)
        data_out.add_data_vector(ux_solution_, "ux");
        data_out.add_data_vector(uy_solution_, "uy");

        // Compute velocity magnitude
        dealii::Vector<double> velocity_magnitude(ux_solution_.size());
        for (unsigned int i = 0; i < ux_solution_.size(); ++i)
        {
            velocity_magnitude[i] = std::sqrt(ux_solution_[i] * ux_solution_[i] +
                                              uy_solution_[i] * uy_solution_[i]);
        }
        data_out.add_data_vector(velocity_magnitude, "velocity_magnitude");

        data_out.build_patches(params_.fe_degree_velocity);

        std::ofstream output(base_path + "ns_solution_" + suffix + ".vtu");
        data_out.write_vtu(output);
    }

    // Pressure on Q1 mesh (separate file for clean visualization)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(p_dof_handler_);

        data_out.add_data_vector(p_solution_, "pressure");

        data_out.build_patches(params_.fe_degree_pressure);

        std::ofstream output(base_path + "pressure_" + suffix + ".vtu");
        data_out.write_vtu(output);
    }

    // ========================================================================
    // 2. Cahn-Hilliard solution: c, mu
    // ========================================================================
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(c_dof_handler_);

        data_out.add_data_vector(c_solution_, "c");
        data_out.add_data_vector(mu_solution_, "mu");

        data_out.build_patches(params_.fe_degree_phase);

        std::ofstream output(base_path + "ch_solution_" + suffix + ".vtu");
        data_out.write_vtu(output);
    }

    // ========================================================================
    // 3. Poisson solution: phi (magnetic potential) - only if enabled
    // ========================================================================
    if (params_.enable_magnetic)
    {
        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(phi_dof_handler_);

        data_out.add_data_vector(phi_solution_, "phi");

        data_out.build_patches(params_.fe_degree_phase);

        std::ofstream output(base_path + "phi_solution_" + suffix + ".vtu");
        data_out.write_vtu(output);
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void NSCHProblem<2>::output_results(unsigned int) const;