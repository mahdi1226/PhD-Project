// ============================================================================
// mms/mms_context.cc - MMS Test Context Implementation (Parallel Version)
//
// CH-ONLY VERSION: Other subsystems not yet converted to parallel.
//
// PARALLEL VERSION:
//   - Uses parallel::distributed::Triangulation
//   - Uses TrilinosWrappers types
//   - Initializes IndexSets for owned/relevant DoFs
//   - Uses MPI reductions where needed
// ============================================================================

#include "mms/mms_context.h"

// Production setup functions - CH only
#include "setup/ch_setup.h"

// MMS exact solutions and BC functions - CH only
#include "mms/ch/ch_mms.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MMSContext<dim>::MMSContext(MPI_Comm mpi_comm)
    : mpi_communicator(mpi_comm)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    , triangulation(mpi_comm,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                        dealii::Triangulation<dim>::smoothing_on_refinement |
                        dealii::Triangulation<dim>::smoothing_on_coarsening))
    , theta_dof_handler(triangulation)
    , psi_dof_handler(triangulation)
{
}

// ============================================================================
// setup_mesh() - Creates distributed mesh
// ============================================================================
template <int dim>
void MMSContext<dim>::setup_mesh(const Parameters& params, unsigned int refinement)
{
    // Create rectangular domain with subdivisions
    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params.domain.initial_cells_x;
    subdivisions[1] = params.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    // Assign boundary IDs - SAME as production:
    //   0 = bottom (y = y_min)
    //   1 = right  (x = x_max)
    //   2 = top    (y = y_max)
    //   3 = left   (x = x_min)
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (const auto& face : cell->face_iterators())
        {
            if (!face->at_boundary())
                continue;

            const auto center = face->center();
            const double tol = 1e-10;

            if (std::abs(center[1] - params.domain.y_min) < tol)
                face->set_boundary_id(0);  // bottom
            else if (std::abs(center[0] - params.domain.x_max) < tol)
                face->set_boundary_id(1);  // right
            else if (std::abs(center[1] - params.domain.y_max) < tol)
                face->set_boundary_id(2);  // top
            else if (std::abs(center[0] - params.domain.x_min) < tol)
                face->set_boundary_id(3);  // left
        }
    }

    // Refine mesh
    triangulation.refine_global(refinement);

    pcout << "[MMSContext] Mesh: " << triangulation.n_global_active_cells()
          << " cells (local: " << triangulation.n_locally_owned_active_cells() << ")\n";
}

// ============================================================================
// setup_ch() - Uses PRODUCTION setup_ch_coupled_system()
// ============================================================================
template <int dim>
void MMSContext<dim>::setup_ch(const Parameters& params, double initial_time)
{
    // Create finite element
    fe_phase = std::make_unique<dealii::FE_Q<dim>>(params.fe.degree_phase);

    // Distribute DoFs for both fields
    theta_dof_handler.distribute_dofs(*fe_phase);
    psi_dof_handler.distribute_dofs(*fe_phase);

    const unsigned int n_theta = theta_dof_handler.n_dofs();
    const unsigned int n_psi = psi_dof_handler.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // Build IndexSets for individual fields
    theta_locally_owned = theta_dof_handler.locally_owned_dofs();
    theta_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler);
    psi_locally_owned = psi_dof_handler.locally_owned_dofs();
    psi_locally_relevant = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler);

    // Build combined IndexSets for coupled system
    // θ in [0, n_theta), ψ in [n_theta, n_total)
    ch_locally_owned.set_size(n_total);
    ch_locally_relevant.set_size(n_total);

    for (auto idx = theta_locally_owned.begin(); idx != theta_locally_owned.end(); ++idx)
        ch_locally_owned.add_index(*idx);
    for (auto idx = psi_locally_owned.begin(); idx != psi_locally_owned.end(); ++idx)
        ch_locally_owned.add_index(n_theta + *idx);

    for (auto idx = theta_locally_relevant.begin(); idx != theta_locally_relevant.end(); ++idx)
        ch_locally_relevant.add_index(*idx);
    for (auto idx = psi_locally_relevant.begin(); idx != psi_locally_relevant.end(); ++idx)
        ch_locally_relevant.add_index(n_theta + *idx);

    ch_locally_owned.compress();
    ch_locally_relevant.compress();

    // Reinit vectors
    theta_owned.reinit(theta_locally_owned, mpi_communicator);
    theta_relevant.reinit(theta_locally_owned, theta_locally_relevant, mpi_communicator);
    theta_old_owned.reinit(theta_locally_owned, mpi_communicator);
    theta_old_relevant.reinit(theta_locally_owned, theta_locally_relevant, mpi_communicator);
    psi_owned.reinit(psi_locally_owned, mpi_communicator);
    psi_relevant.reinit(psi_locally_owned, psi_locally_relevant, mpi_communicator);
    ch_rhs.reinit(ch_locally_owned, mpi_communicator);

    // Setup index maps
    theta_to_ch_map.resize(n_theta);
    psi_to_ch_map.resize(n_psi);
    for (unsigned int i = 0; i < n_theta; ++i)
        theta_to_ch_map[i] = i;
    for (unsigned int i = 0; i < n_psi; ++i)
        psi_to_ch_map[i] = n_theta + i;

    // Setup individual field constraints with MMS BCs
    theta_constraints.clear();
    theta_constraints.reinit(theta_locally_owned, theta_locally_relevant);
    psi_constraints.clear();
    psi_constraints.reinit(psi_locally_owned, psi_locally_relevant);

    // Apply MMS boundary conditions
    CHMMSBoundaryTheta<dim> theta_bc;
    CHMMSBoundaryPsi<dim> psi_bc;
    theta_bc.set_time(initial_time);
    psi_bc.set_time(initial_time);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> theta_bc_map;
    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> psi_bc_map;
    for (unsigned int i = 0; i < 2 * dim; ++i)
    {
        theta_bc_map[i] = &theta_bc;
        psi_bc_map[i] = &psi_bc;
    }

    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler, theta_bc_map, theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler, psi_bc_map, psi_constraints);

    theta_constraints.close();
    psi_constraints.close();

    // PRODUCTION: Setup coupled system - builds ch_constraints, ch_matrix
    setup_ch_coupled_system<dim>(
        theta_dof_handler, psi_dof_handler,
        theta_constraints, psi_constraints,
        ch_locally_owned, ch_locally_relevant,
        theta_to_ch_map, psi_to_ch_map,
        ch_constraints, ch_matrix,
        mpi_communicator, pcout);

    pcout << "[MMSContext] CH DoFs: θ = " << n_theta
          << ", ψ = " << n_psi
          << ", coupled = " << n_total
          << ", local = " << ch_locally_owned.n_elements() << "\n";
}

// ============================================================================
// apply_ch_initial_conditions() - MMS initial conditions
// ============================================================================
template <int dim>
void MMSContext<dim>::apply_ch_initial_conditions(const Parameters& /*params*/, double t_init)
{
    CHMMSInitialTheta<dim> theta_ic(t_init);
    CHMMSInitialPsi<dim> psi_ic(t_init);

    // Interpolate into owned vectors
    dealii::VectorTools::interpolate(theta_dof_handler, theta_ic, theta_owned);
    dealii::VectorTools::interpolate(psi_dof_handler, psi_ic, psi_owned);

    // Update ghost values
    theta_relevant = theta_owned;
    psi_relevant = psi_owned;

    // Copy to old
    theta_old_owned = theta_owned;
    theta_old_relevant = theta_relevant;

    pcout << "[MMSContext] Applied CH MMS initial conditions at t = " << t_init << "\n";
}

// ============================================================================
// update_ch_constraints() - Update BCs for new time
// ============================================================================
template <int dim>
void MMSContext<dim>::update_ch_constraints(const Parameters& /*params*/, double current_time)
{
    // Clear and reinit individual constraints
    theta_constraints.clear();
    theta_constraints.reinit(theta_locally_owned, theta_locally_relevant);
    psi_constraints.clear();
    psi_constraints.reinit(psi_locally_owned, psi_locally_relevant);

    // Apply MMS BCs at new time
    CHMMSBoundaryTheta<dim> theta_bc;
    CHMMSBoundaryPsi<dim> psi_bc;
    theta_bc.set_time(current_time);
    psi_bc.set_time(current_time);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> theta_bc_map;
    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> psi_bc_map;
    for (unsigned int i = 0; i < 2 * dim; ++i)
    {
        theta_bc_map[i] = &theta_bc;
        psi_bc_map[i] = &psi_bc;
    }

    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler, theta_bc_map, theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler, psi_bc_map, psi_constraints);

    theta_constraints.close();
    psi_constraints.close();

    // Rebuild combined constraints
    const unsigned int n_theta = theta_dof_handler.n_dofs();

    ch_constraints.clear();
    ch_constraints.reinit(ch_locally_owned, ch_locally_relevant);

    // Map θ constraints to coupled system
    for (auto idx = theta_locally_relevant.begin(); idx != theta_locally_relevant.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (theta_constraints.is_constrained(i))
        {
            const auto* entries = theta_constraints.get_constraint_entries(i);
            if (entries == nullptr || entries->empty())
            {
                // Inhomogeneous constraint
                ch_constraints.add_line(theta_to_ch_map[i]);
                ch_constraints.set_inhomogeneity(theta_to_ch_map[i],
                    theta_constraints.get_inhomogeneity(i));
            }
        }
    }

    // Map ψ constraints to coupled system
    for (auto idx = psi_locally_relevant.begin(); idx != psi_locally_relevant.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (psi_constraints.is_constrained(i))
        {
            const auto* entries = psi_constraints.get_constraint_entries(i);
            if (entries == nullptr || entries->empty())
            {
                ch_constraints.add_line(psi_to_ch_map[i]);
                ch_constraints.set_inhomogeneity(psi_to_ch_map[i],
                    psi_constraints.get_inhomogeneity(i));
            }
        }
    }

    ch_constraints.close();
}

// ============================================================================
// get_min_h() - Minimum cell diameter (MPI reduction)
// ============================================================================
template <int dim>
double MMSContext<dim>::get_min_h() const
{
    double local_min_h = std::numeric_limits<double>::max();

    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (cell->is_locally_owned())
            local_min_h = std::min(local_min_h, cell->diameter());
    }

    double global_min_h = 0.0;
    MPI_Allreduce(&local_min_h, &global_min_h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    return global_min_h;
}

// ============================================================================
// n_ch_dofs() - Total CH DoFs (global)
// ============================================================================
template <int dim>
unsigned int MMSContext<dim>::n_ch_dofs() const
{
    return theta_dof_handler.n_dofs() + psi_dof_handler.n_dofs();
}

// ============================================================================
// Ghost update helpers
// ============================================================================
template <int dim>
void MMSContext<dim>::update_theta_ghosts()
{
    theta_relevant = theta_owned;
}

template <int dim>
void MMSContext<dim>::update_theta_old_ghosts()
{
    theta_old_relevant = theta_old_owned;
}

template <int dim>
void MMSContext<dim>::update_psi_ghosts()
{
    psi_relevant = psi_owned;
}

template <int dim>
void MMSContext<dim>::update_ch_ghosts()
{
    update_theta_ghosts();
    update_theta_old_ghosts();
    update_psi_ghosts();
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MMSContext<2>;