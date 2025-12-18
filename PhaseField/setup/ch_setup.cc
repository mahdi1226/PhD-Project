// ============================================================================
// setup/ch_setup.cc - Cahn-Hilliard Coupled System Setup Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/ch_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

template <int dim>
void setup_ch_coupled_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    dealii::SparsityPattern& ch_sparsity,
    bool verbose)
{
    Assert(theta_dof_handler.n_dofs() == psi_dof_handler.n_dofs(),
           dealii::ExcMessage("θ and ψ DoF counts must match"));

    const unsigned int n_theta = theta_dof_handler.n_dofs();
    const unsigned int n_psi = psi_dof_handler.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

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
    // This is CRITICAL for AMR! Hanging node constraints from both θ and ψ
    // must be mapped to the coupled system indices.
    // ========================================================================
    ch_combined_constraints.clear();

    // Map θ constraints to coupled system
    for (unsigned int i = 0; i < n_theta; ++i)
    {
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

    // Map ψ constraints to coupled system
    for (unsigned int i = 0; i < n_psi; ++i)
    {
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
    // Step 3: Build sparsity pattern for coupled system
    //
    // Block structure:
    //   [θ-θ  θ-ψ]
    //   [ψ-θ  ψ-ψ]
    //
    // Since θ and ψ use the same FE on the same mesh, all 4 blocks
    // have identical sparsity structure. Build once, reuse 4 times.
    // ========================================================================
    dealii::DynamicSparsityPattern dsp(n_total, n_total);

    // Build base sparsity for single field
    dealii::DynamicSparsityPattern base_dsp(n_theta, n_theta);
    dealii::DoFTools::make_sparsity_pattern(
        theta_dof_handler,
        base_dsp,
        theta_constraints,
        /*keep_constrained_dofs=*/false);
    // All 4 blocks use the same structure
    for (unsigned int i = 0; i < n_theta; ++i)
    {
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

    // Condense sparsity pattern with combined constraints
    ch_combined_constraints.condense(dsp);

    // Copy to final sparsity pattern
    ch_sparsity.copy_from(dsp);

    if (verbose)
    {
        std::cout << "[Setup] CH sparsity: " << ch_sparsity.n_nonzero_elements()
                  << " nonzeros\n";
    }
}



// ============================================================================
// Explicit instantiation
// ============================================================================
template void setup_ch_coupled_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);