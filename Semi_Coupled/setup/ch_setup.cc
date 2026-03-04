// ============================================================================
// setup/ch_setup.cc - Cahn-Hilliard Coupled System Setup (Parallel Version)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/ch_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <iostream>

template <int dim>
void setup_ch_coupled_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& ch_locally_relevant,
    std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ch_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout)
{
    Assert(theta_dof_handler.n_dofs() == psi_dof_handler.n_dofs(),
           dealii::ExcMessage("θ and ψ DoF counts must match"));

    const unsigned int n_theta = theta_dof_handler.n_dofs();
    const unsigned int n_psi = psi_dof_handler.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // Get individual field IndexSets
    const dealii::IndexSet theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    const dealii::IndexSet theta_locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    const dealii::IndexSet psi_locally_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler);

    // ========================================================================
    // Step 1: Build index maps
    //   θ occupies [0, n_theta)
    //   ψ occupies [n_theta, n_total)
    // ========================================================================
    theta_to_ch_map.resize(n_theta);
    psi_to_ch_map.resize(n_psi);

    for (unsigned int i = 0; i < n_theta; ++i)
        theta_to_ch_map[i] = i;

    for (unsigned int i = 0; i < n_psi; ++i)
        psi_to_ch_map[i] = n_theta + i;

    // ========================================================================
    // Step 2: Build combined constraints
    //
    // CRITICAL for parallel: Must reinit with (owned, relevant) IndexSets
    // Only loop over locally_relevant DoFs
    // ========================================================================
    ch_combined_constraints.clear();
    ch_combined_constraints.reinit(ch_locally_owned, ch_locally_relevant);

    // Map θ constraints to coupled system (only locally relevant)
    for (auto idx = theta_locally_relevant.begin();
         idx != theta_locally_relevant.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (theta_constraints.is_constrained(i))
        {
            const auto* entries = theta_constraints.get_constraint_entries(i);
            const double inhomogeneity = theta_constraints.get_inhomogeneity(i);
            const auto coupled_i = theta_to_ch_map[i];

            if (entries != nullptr && !entries->empty())
            {
                // Hanging node: constrained to other DoFs
                std::vector<std::pair<dealii::types::global_dof_index, double>> coupled_entries;
                for (const auto& entry : *entries)
                    coupled_entries.emplace_back(theta_to_ch_map[entry.first], entry.second);

                ch_combined_constraints.add_line(coupled_i);
                ch_combined_constraints.add_entries(coupled_i, coupled_entries);
                ch_combined_constraints.set_inhomogeneity(coupled_i, inhomogeneity);
            }
            else
            {
                // Dirichlet BC
                ch_combined_constraints.add_line(coupled_i);
                ch_combined_constraints.set_inhomogeneity(coupled_i, inhomogeneity);
            }
        }
    }

    // Map ψ constraints to coupled system (only locally relevant)
    for (auto idx = psi_locally_relevant.begin();
         idx != psi_locally_relevant.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (psi_constraints.is_constrained(i))
        {
            const auto* entries = psi_constraints.get_constraint_entries(i);
            const double inhomogeneity = psi_constraints.get_inhomogeneity(i);
            const auto coupled_i = psi_to_ch_map[i];

            if (entries != nullptr && !entries->empty())
            {
                std::vector<std::pair<dealii::types::global_dof_index, double>> coupled_entries;
                for (const auto& entry : *entries)
                    coupled_entries.emplace_back(psi_to_ch_map[entry.first], entry.second);

                ch_combined_constraints.add_line(coupled_i);
                ch_combined_constraints.add_entries(coupled_i, coupled_entries);
                ch_combined_constraints.set_inhomogeneity(coupled_i, inhomogeneity);
            }
            else
            {
                ch_combined_constraints.add_line(coupled_i);
                ch_combined_constraints.set_inhomogeneity(coupled_i, inhomogeneity);
            }
        }
    }

    ch_combined_constraints.close();

    // ========================================================================
    // Step 3: Build distributed sparsity pattern for coupled system
    //
    // Block structure:
    //   [θ-θ  θ-ψ]
    //   [ψ-θ  ψ-ψ]
    //
    // Since θ and ψ use the same FE on the same mesh, all 4 blocks
    // have identical sparsity structure. Build once, reuse 4 times.
    // ========================================================================
    dealii::DynamicSparsityPattern dsp(n_total, n_total, ch_locally_relevant);

    // Build base sparsity for single field (only locally relevant rows)
    dealii::DynamicSparsityPattern base_dsp(n_theta, n_theta, theta_locally_relevant);
    dealii::DoFTools::make_sparsity_pattern(
        theta_dof_handler,
        base_dsp,
        theta_constraints,
        /*keep_constrained_dofs=*/true);

    // Map to all 4 blocks (only for locally relevant rows)
    for (auto idx = theta_locally_relevant.begin();
         idx != theta_locally_relevant.end(); ++idx)
    {
        const unsigned int i = *idx;
        for (auto j = base_dsp.begin(i); j != base_dsp.end(i); ++j)
        {
            const unsigned int col = j->column();
            // θ-θ block
            dsp.add(theta_to_ch_map[i], theta_to_ch_map[col]);
            // θ-ψ block
            dsp.add(theta_to_ch_map[i], psi_to_ch_map[col]);
            // ψ-θ block
            dsp.add(psi_to_ch_map[i], theta_to_ch_map[col]);
            // ψ-ψ block
            dsp.add(psi_to_ch_map[i], psi_to_ch_map[col]);
        }
    }

    // Distribute sparsity pattern across MPI ranks
    dealii::SparsityTools::distribute_sparsity_pattern(
        dsp,
        ch_locally_owned,
        mpi_communicator,
        ch_locally_relevant);

    // ========================================================================
    // Step 4: Initialize Trilinos matrix
    // ========================================================================
    ch_matrix.reinit(ch_locally_owned,
                     ch_locally_owned,
                     dsp,
                     mpi_communicator);

    pcout << "[CH Setup] n_dofs = " << n_total
          << ", locally_owned = " << ch_locally_owned.n_elements()
          << ", nnz = " << ch_matrix.n_nonzero_elements() << "\n";
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void setup_ch_coupled_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm,
    dealii::ConditionalOStream&);