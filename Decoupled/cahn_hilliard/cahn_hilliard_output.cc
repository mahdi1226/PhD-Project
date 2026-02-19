// ============================================================================
// cahn_hilliard/cahn_hilliard_output.cc — VTK Output Implementation
//
// Implements CahnHilliardSubsystem<dim>::write_vtu():
//   Writes parallel VTU/PVTU files for theta, psi, derived fields.
//
// VTK fields:
//   theta          Phase field (CG Q2, nodal)
//   psi            Chemical potential (CG Q2, nodal)
//   grad_theta     |nabla theta| — interface indicator (DG Q0, cell-averaged)
//   energy_density eps/2|nabla theta|^2 + (1/eps)F(theta) per cell (DG Q0)
//   subdomain      MPI partition coloring
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <filesystem>
#include <iostream>
#include <iomanip>

template <int dim>
void CahnHilliardSubsystem<dim>::write_vtu(
    const std::string& output_dir,
    unsigned int step,
    double time) const
{
    const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);

    // --- Ensure output directory exists ---
    if (rank == 0)
        std::filesystem::create_directories(output_dir);
    MPI_Barrier(mpi_comm_);

    // --- DG Q0 space for cell-averaged derived fields ---
    dealii::FE_DGQ<dim> fe_dg0(0);
    dealii::DoFHandler<dim> dg0_dof(triangulation_);
    dg0_dof.distribute_dofs(fe_dg0);

    dealii::IndexSet dg0_owned = dg0_dof.locally_owned_dofs();
    const dealii::IndexSet dg0_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(dg0_dof);

    dealii::TrilinosWrappers::MPI::Vector grad_theta_mag(dg0_owned, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector energy_density(dg0_owned, mpi_comm_);

    // FEValues for gradient/value evaluation on CG space
    dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> theta_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> theta_grads(n_q);

    const double eps = params_.physics.epsilon;

    auto cell_cg  = theta_dof_handler_.begin_active();
    auto cell_dg0 = dg0_dof.begin_active();

    for (; cell_cg != theta_dof_handler_.end(); ++cell_cg, ++cell_dg0)
    {
        if (!cell_cg->is_locally_owned())
            continue;

        fe_values.reinit(cell_cg);
        fe_values.get_function_values(theta_relevant_, theta_vals);
        fe_values.get_function_gradients(theta_relevant_, theta_grads);

        double avg_grad_mag = 0.0;
        double avg_energy = 0.0;
        double vol = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const double grad_norm = theta_grads[q].norm();
            const double th = theta_vals[q];
            const double F = 0.25 * (th * th - 1.0) * (th * th - 1.0);

            avg_grad_mag += grad_norm * JxW;
            avg_energy += (0.5 * eps * grad_norm * grad_norm + F / eps) * JxW;
            vol += JxW;
        }

        std::vector<dealii::types::global_dof_index> dg0_idx(1);
        cell_dg0->get_dof_indices(dg0_idx);
        const auto idx = dg0_idx[0];

        grad_theta_mag(idx) = avg_grad_mag / vol;
        energy_density(idx) = avg_energy / vol;
    }

    grad_theta_mag.compress(dealii::VectorOperation::insert);
    energy_density.compress(dealii::VectorOperation::insert);

    dealii::TrilinosWrappers::MPI::Vector grad_rel(dg0_owned, dg0_relevant, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector energy_rel(dg0_owned, dg0_relevant, mpi_comm_);
    grad_rel   = grad_theta_mag;
    energy_rel = energy_density;

    // --- DataOut ---
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(theta_dof_handler_);
    data_out.add_data_vector(theta_relevant_, "theta");
    data_out.add_data_vector(psi_dof_handler_, psi_relevant_, "psi");

    data_out.add_data_vector(dg0_dof, grad_rel, "grad_theta");
    data_out.add_data_vector(dg0_dof, energy_rel, "energy_density");

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
template void CahnHilliardSubsystem<2>::write_vtu(
    const std::string&, unsigned int, double) const;

template void CahnHilliardSubsystem<3>::write_vtu(
    const std::string&, unsigned int, double) const;
