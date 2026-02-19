// ============================================================================
// poisson/poisson_output.cc — VTK Output Implementation
//
// Implements PoissonSubsystem<dim>::write_vtu():
//   Writes parallel VTU/PVTU files for phi and derived H = grad(phi).
//
// VTK fields:
//   phi        Magnetostatic potential (CG Q1, nodal)
//   H_x, H_y  Demagnetizing field components (DG Q0, cell-averaged)
//   H_z        (3D only)
//   H_mag      |grad(phi)| (DG Q0, cell-averaged)
//   subdomain  MPI partition coloring
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "poisson/poisson.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <filesystem>
#include <iostream>
#include <iomanip>

template <int dim>
void PoissonSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time) const
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    // --- Ensure output directory exists ---
    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    // --- DG Q0 space for cell-averaged gradient fields ---
    dealii::FE_DGQ<dim> fe_dg0(0);
    dealii::DoFHandler<dim> dg0_dof(triangulation_);
    dg0_dof.distribute_dofs(fe_dg0);

    dealii::IndexSet dg0_owned = dg0_dof.locally_owned_dofs();
    const dealii::IndexSet dg0_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(dg0_dof);

    // Vectors for derived fields
    dealii::TrilinosWrappers::MPI::Vector H_x(dg0_owned, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_y(dg0_owned, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_z(dg0_owned, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_mag(dg0_owned, mpi_comm_);

    // FEValues for gradient evaluation on CG space
    const auto& fe_cg = dof_handler_.get_fe();
    dealii::QGauss<dim> quadrature(fe_cg.degree + 1);
    dealii::FEValues<dim> fe_values(fe_cg, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<dealii::Tensor<1, dim>> grad_phi(n_q);

    auto cell_cg  = dof_handler_.begin_active();
    auto cell_dg0 = dg0_dof.begin_active();

    for (; cell_cg != dof_handler_.end(); ++cell_cg, ++cell_dg0)
    {
        if (!cell_cg->is_locally_owned())
            continue;

        fe_values.reinit(cell_cg);
        fe_values.get_function_gradients(solution_relevant_, grad_phi);

        dealii::Tensor<1, dim> avg_grad;
        double vol = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            avg_grad += grad_phi[q] * JxW;
            vol += JxW;
        }
        avg_grad /= vol;

        std::vector<dealii::types::global_dof_index> dg0_idx(1);
        cell_dg0->get_dof_indices(dg0_idx);
        const auto idx = dg0_idx[0];

        H_x(idx) = avg_grad[0];
        H_y(idx) = avg_grad[1];
        if constexpr (dim == 3)
            H_z(idx) = avg_grad[2];
        H_mag(idx) = avg_grad.norm();
    }

    H_x.compress(dealii::VectorOperation::insert);
    H_y.compress(dealii::VectorOperation::insert);
    H_z.compress(dealii::VectorOperation::insert);
    H_mag.compress(dealii::VectorOperation::insert);

    // Ghosted versions for output
    dealii::TrilinosWrappers::MPI::Vector H_x_rel(dg0_owned, dg0_relevant, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_y_rel(dg0_owned, dg0_relevant, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_z_rel(dg0_owned, dg0_relevant, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector H_mag_rel(dg0_owned, dg0_relevant, mpi_comm_);
    H_x_rel   = H_x;
    H_y_rel   = H_y;
    H_z_rel   = H_z;
    H_mag_rel = H_mag;

    // --- DataOut ---
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_);
    data_out.add_data_vector(solution_relevant_, "phi");

    data_out.add_data_vector(dg0_dof, H_x_rel,   "H_x");
    data_out.add_data_vector(dg0_dof, H_y_rel,   "H_y");
    if constexpr (dim == 3)
        data_out.add_data_vector(dg0_dof, H_z_rel, "H_z");
    data_out.add_data_vector(dg0_dof, H_mag_rel, "H_mag");

    // Subdomain coloring
    dealii::Vector<float> subdomain(triangulation_.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation_.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(fe_cg.degree);

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
template void PoissonSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;

template void PoissonSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
