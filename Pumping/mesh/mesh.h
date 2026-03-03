// ============================================================================
// mesh/mesh.h - Mesh Creation and Boundary Identification
//
// Provides helper functions for creating standard meshes used in FHD.
// The driver owns the triangulation; these functions configure it.
//
// Boundary IDs (deal.II convention for subdivided_hyper_rectangle):
//   0: x = x_min (left)     1: x = x_max (right)
//   2: y = y_min (bottom)   3: y = y_max (top)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_MESH_H
#define FHD_MESH_H

#include "utilities/parameters.h"

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <vector>

namespace FHDMesh
{

// ============================================================================
// Boundary IDs — for use by subsystems when setting BCs
// ============================================================================
enum BoundaryID : unsigned int
{
    left   = 0,
    right  = 1,
    bottom = 2,
    top    = 3
};

// ============================================================================
// Create rectangular mesh from parameters
//
// Creates a subdivided_hyper_rectangle with the domain extents and
// initial cell counts from params.domain, then refines globally.
//
// @param triangulation  The p4est triangulation to populate (must be empty)
// @param params         Parameters with domain and mesh settings
// ============================================================================
template <int dim>
void create_mesh(
    dealii::parallel::distributed::Triangulation<dim>& triangulation,
    const Parameters& params)
{
    static_assert(dim == 2, "FHD is currently 2D only");

    const dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    const dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);

    const std::vector<unsigned int> repetitions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y
    };

    dealii::GridGenerator::subdivided_hyper_rectangle(
        triangulation, repetitions, p1, p2, /*colorize=*/true);

    triangulation.refine_global(params.mesh.initial_refinement);
}

// ============================================================================
// Get mesh statistics (for logging)
// ============================================================================
template <int dim>
struct MeshInfo
{
    unsigned int n_active_cells;
    unsigned int n_global_active_cells;
    double       h_min;
    double       h_max;
};

template <int dim>
MeshInfo<dim> get_mesh_info(
    const dealii::parallel::distributed::Triangulation<dim>& triangulation)
{
    MeshInfo<dim> info;
    info.n_active_cells = triangulation.n_active_cells();
    info.n_global_active_cells = triangulation.n_global_active_cells();
    info.h_min = dealii::GridTools::minimal_cell_diameter(triangulation);
    info.h_max = dealii::GridTools::maximal_cell_diameter(triangulation);
    return info;
}

} // namespace FHDMesh

#endif // FHD_MESH_H
