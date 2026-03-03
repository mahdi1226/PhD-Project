// ============================================================================
// poisson/poisson_output.cc - VTK Output for Poisson Subsystem
//
// Writes parallel VTU/PVTU files containing:
//   - phi:    scalar potential φ
//   - H_x:    ∂φ/∂x (x-component of demagnetizing field)
//   - H_y:    ∂φ/∂y (y-component of demagnetizing field)
//   - H_mag:  |∇φ| (magnitude of demagnetizing field)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Eq. 42d
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>

#include <filesystem>
#include <fstream>

template <int dim>
void PoissonSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time) const
{
    // Create output directory if needed
    if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_) == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(solution_relevant_, "phi");

    // Compute derived fields: H = ∇φ
    // Use cell-averaged DG0 projection for derived quantities
    dealii::Vector<float> H_x(triangulation_.n_active_cells());
    dealii::Vector<float> H_y(triangulation_.n_active_cells());
    dealii::Vector<float> H_mag(triangulation_.n_active_cells());

    const dealii::QMidpoint<dim> q_midpoint;
    dealii::FEValues<dim> fe_values(fe_, q_midpoint,
                                    dealii::update_gradients);

    unsigned int cell_index = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            std::vector<dealii::Tensor<1, dim>> grad_phi(1);
            fe_values.get_function_gradients(solution_relevant_, grad_phi);

            H_x[cell_index] = static_cast<float>(grad_phi[0][0]);
            H_y[cell_index] = static_cast<float>(grad_phi[0][1]);
            H_mag[cell_index] = static_cast<float>(grad_phi[0].norm());
        }
        ++cell_index;
    }

    data_out.add_data_vector(H_x, "H_x",
                             dealii::DataOut<dim>::type_cell_data);
    data_out.add_data_vector(H_y, "H_y",
                             dealii::DataOut<dim>::type_cell_data);
    data_out.add_data_vector(H_mag, "H_mag",
                             dealii::DataOut<dim>::type_cell_data);

    // Subdomain for parallel visualization
    dealii::Vector<float> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain",
                             dealii::DataOut<dim>::type_cell_data);

    data_out.build_patches();

    const std::string filename = output_dir + "/poisson_" +
        dealii::Utilities::int_to_string(step, 6);

    data_out.write_vtu_with_pvtu_record(
        output_dir + "/", "poisson_",
        step, mpi_comm_, 4, 0);
}

// Explicit instantiations
template void PoissonSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;
template void PoissonSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
