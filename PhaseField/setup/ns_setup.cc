// ============================================================================
// setup/ns_setup.cc - Navier-Stokes Coupled System Setup Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/ns_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

template <int dim>
void setup_ns_coupled_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& ux_constraints,
    const dealii::AffineConstraints<double>& uy_constraints,
    const dealii::AffineConstraints<double>& p_constraints,
    std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::AffineConstraints<double>& ns_combined_constraints,
    dealii::SparsityPattern& ns_sparsity,
    bool verbose)
{
    const unsigned int n_ux = ux_dof_handler.n_dofs();
    const unsigned int n_uy = uy_dof_handler.n_dofs();
    const unsigned int n_p = p_dof_handler.n_dofs();
    const unsigned int n_total = n_ux + n_uy + n_p;

    // ========================================================================
    // Step 1: Build index maps
    // ========================================================================
    // Layout: [ux | uy | p]
    ux_to_ns_map.resize(n_ux);
    uy_to_ns_map.resize(n_uy);
    p_to_ns_map.resize(n_p);

    for (unsigned int i = 0; i < n_ux; ++i)
        ux_to_ns_map[i] = i;

    for (unsigned int i = 0; i < n_uy; ++i)
        uy_to_ns_map[i] = n_ux + i;

    for (unsigned int i = 0; i < n_p; ++i)
        p_to_ns_map[i] = n_ux + n_uy + i;

    // ========================================================================
    // Step 2: Build sparsity pattern
    // ========================================================================
    // The NS system has 9 blocks:
    //   [ux-ux, ux-uy, ux-p ]
    //   [uy-ux, uy-uy, uy-p ]
    //   [p-ux,  p-uy,  p-p  ]
    //
    // Note: p-p block is empty (no pressure-pressure coupling)

    dealii::DynamicSparsityPattern dsp(n_total, n_total);

    // Build base sparsity patterns for Q2-Q2 and Q1-Q1 couplings
    dealii::DynamicSparsityPattern dsp_Q2(n_ux, n_ux);
    dealii::DoFTools::make_sparsity_pattern(ux_dof_handler, dsp_Q2);

    dealii::DynamicSparsityPattern dsp_Q1(n_p, n_p);
    dealii::DoFTools::make_sparsity_pattern(p_dof_handler, dsp_Q1);

    // Q2-Q1 coupling (velocity-pressure)
    // For Taylor-Hood, we need the coupling between Q2 velocity and Q1 pressure
    dealii::DynamicSparsityPattern dsp_Q2_Q1(n_ux, n_p);
    dealii::DoFTools::make_sparsity_pattern(ux_dof_handler, p_dof_handler, dsp_Q2_Q1);

    // Fill the 9 blocks
    // Block (0,0): ux-ux (Q2-Q2)
    for (unsigned int i = 0; i < n_ux; ++i)
        for (auto j = dsp_Q2.begin(i); j != dsp_Q2.end(i); ++j)
            dsp.add(ux_to_ns_map[i], ux_to_ns_map[j->column()]);

    // Block (0,1): ux-uy (Q2-Q2) - for grad-div stabilization
    for (unsigned int i = 0; i < n_ux; ++i)
        for (auto j = dsp_Q2.begin(i); j != dsp_Q2.end(i); ++j)
            dsp.add(ux_to_ns_map[i], uy_to_ns_map[j->column()]);

    // Block (0,2): ux-p (Q2-Q1)
    for (unsigned int i = 0; i < n_ux; ++i)
        for (auto j = dsp_Q2_Q1.begin(i); j != dsp_Q2_Q1.end(i); ++j)
            dsp.add(ux_to_ns_map[i], p_to_ns_map[j->column()]);

    // Block (1,0): uy-ux (Q2-Q2) - for grad-div stabilization
    for (unsigned int i = 0; i < n_uy; ++i)
        for (auto j = dsp_Q2.begin(i); j != dsp_Q2.end(i); ++j)
            dsp.add(uy_to_ns_map[i], ux_to_ns_map[j->column()]);

    // Block (1,1): uy-uy (Q2-Q2)
    for (unsigned int i = 0; i < n_uy; ++i)
        for (auto j = dsp_Q2.begin(i); j != dsp_Q2.end(i); ++j)
            dsp.add(uy_to_ns_map[i], uy_to_ns_map[j->column()]);

    // Block (1,2): uy-p (Q2-Q1)
    for (unsigned int i = 0; i < n_uy; ++i)
        for (auto j = dsp_Q2_Q1.begin(i); j != dsp_Q2_Q1.end(i); ++j)
            dsp.add(uy_to_ns_map[i], p_to_ns_map[j->column()]);

    // Block (2,0): p-ux (Q1-Q2) - transpose of ux-p
    for (unsigned int i = 0; i < n_ux; ++i)
        for (auto j = dsp_Q2_Q1.begin(i); j != dsp_Q2_Q1.end(i); ++j)
            dsp.add(p_to_ns_map[j->column()], ux_to_ns_map[i]);

    // Block (2,1): p-uy (Q1-Q2) - transpose of uy-p
    for (unsigned int i = 0; i < n_uy; ++i)
        for (auto j = dsp_Q2_Q1.begin(i); j != dsp_Q2_Q1.end(i); ++j)
            dsp.add(p_to_ns_map[j->column()], uy_to_ns_map[i]);

    // Block (2,2): p-p - empty (no pressure stabilization by default)
    // Could add pressure stabilization here if needed

    ns_sparsity.copy_from(dsp);

    // ========================================================================
    // Step 3: Build combined constraints
    // ========================================================================
    ns_combined_constraints.clear();

    // Helper lambda to map constraints
    auto map_constraints = [&](const dealii::AffineConstraints<double>& src,
                               const std::vector<dealii::types::global_dof_index>& index_map,
                               unsigned int n_dofs)
    {
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
            if (src.is_constrained(i))
            {
                const auto coupled_i = index_map[i];
                const auto* entries = src.get_constraint_entries(i);
                const double inhom = src.get_inhomogeneity(i);

                ns_combined_constraints.add_line(coupled_i);

                if (entries != nullptr && !entries->empty())
                {
                    for (const auto& entry : *entries)
                    {
                        const auto coupled_j = index_map[entry.first];
                        ns_combined_constraints.add_entry(coupled_i, coupled_j, entry.second);
                    }
                }
                ns_combined_constraints.set_inhomogeneity(coupled_i, inhom);
            }
        }
    };

    map_constraints(ux_constraints, ux_to_ns_map, n_ux);
    map_constraints(uy_constraints, uy_to_ns_map, n_uy);
    map_constraints(p_constraints, p_to_ns_map, n_p);

    ns_combined_constraints.close();

    if (verbose)
    {
        std::cout << "[Setup] NS system: " << n_ux << " + " << n_uy << " + " << n_p
                  << " = " << n_total << " DoFs\n";
        std::cout << "[Setup] NS sparsity: " << ns_sparsity.n_nonzero_elements()
                  << " nonzeros\n";
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void setup_ns_coupled_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);