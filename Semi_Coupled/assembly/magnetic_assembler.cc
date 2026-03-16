// ============================================================================
// assembly/magnetic_assembler.cc - Monolithic Magnetics Assembler (PARALLEL)
//
// Full PDE assembler for Paper Eq. 42c-42d (Nochetto et al. CMAME 2016).
//
// Two modes (selected by --full_mag flag):
//   SIMPLIFIED (default): Relaxation uses h_a (Section 5/6). No C_M_phi coupling.
//   FULL (--full_mag):    Relaxation uses ∇φ (Eq 42c). C_M_phi coupling on LHS.
//
// Cell terms:
//   A_M:       (1/dt + 1/tau_M)(M, Z) + (U·∇M)·Z + ½(∇·U)(M·Z)   [mass+transport]
//   C_phi_M:   +(M, grad X)                                         [Poisson coupling]
//   A_phi:     (grad phi, grad X)                                    [Laplacian]
//
// Face terms (DG transport, Eq. 57 second line):
//   -(U·n) [[M]]·{Z}   (assembled componentwise for M, no phi face terms)
//
// RHS:
//   f_M:   (1/dt)(M^{k-1}, Z) + (1/tau_M) chi(theta) (h_a, Z)   [time + relaxation]
//   f_phi: (h_a, grad X)                                          [applied field]
//
// Sign derivation for C_phi_M:
//   Paper Eq 42d: (grad phi, grad X) = (h_a - M, grad X)
//   Rearranged:   (grad phi, grad X) + (M, grad X) = (h_a, grad X)
//   So C_phi_M = +(M, grad X) on LHS (POSITIVE sign)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/magnetic_assembler.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"
#include "physics/skew_forms.h"
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagneticAssembler<dim>::MagneticAssembler(
    const Parameters& params,
    const dealii::DoFHandler<dim>& mag_dof,
    const dealii::DoFHandler<dim>& U_dof,
    const dealii::DoFHandler<dim>& theta_dof,
    const dealii::AffineConstraints<double>& mag_constraints,
    MPI_Comm mpi_communicator)
    : params_(params)
    , mag_dof_handler_(mag_dof)
    , U_dof_handler_(U_dof)
    , theta_dof_handler_(theta_dof)
    , mag_constraints_(mag_constraints)
{
    (void)mpi_communicator;  // Reserved for future use
    // Precompute M component local DoF mapping.
    // For FESystem(FE_DGQ^dim, FE_Q): components 0..dim-1 are M, component dim is phi.
    // M_comp_local_dofs_[d][i] = local DoF index for the i-th DG basis function
    // of M component d. Since all M components share the same DG basis, the
    // shape functions are identical — only the global DoF indices differ.
    const auto& fe = mag_dof.get_fe();
    M_comp_local_dofs_.resize(dim);
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
        const unsigned int comp = fe.system_to_component_index(i).first;
        if (comp < static_cast<unsigned int>(dim))
            M_comp_local_dofs_[comp].push_back(i);
    }
}

// ============================================================================
// Main assembly (PARALLEL)
// ============================================================================
template <int dim>
void MagneticAssembler<dim>::assemble(
    dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& system_rhs,
    const dealii::TrilinosWrappers::MPI::Vector& Ux,
    const dealii::TrilinosWrappers::MPI::Vector& Uy,
    const dealii::TrilinosWrappers::MPI::Vector& theta,
    const dealii::TrilinosWrappers::MPI::Vector& mag_old,
    double dt,
    double current_time) const
{
    const bool mms_mode = params_.enable_mms;
    const double L_y = params_.domain.y_max - params_.domain.y_min;
    const double t_old = current_time - dt;

    const auto& fe_mag = mag_dof_handler_.get_fe();
    const auto& fe_U = U_dof_handler_.get_fe();
    const auto& fe_theta = theta_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_mag.n_dofs_per_cell();
    const unsigned int n_M = M_comp_local_dofs_[0].size();  // DG DoFs per component

    // Extractors for the FESystem components
    const dealii::FEValuesExtractors::Vector M(0);     // components 0, 1
    const dealii::FEValuesExtractors::Scalar phi(dim); // component dim

    // Quadrature
    const unsigned int quad_degree = std::max(fe_mag.degree, 2u) + 2;
    dealii::QGauss<dim> quadrature_cell(quad_degree);
    dealii::QGauss<dim - 1> quadrature_face(quad_degree);
    const unsigned int n_q_cell = quadrature_cell.size();
    const unsigned int n_q_face = quadrature_face.size();

    // FEValues for cells
    dealii::FEValues<dim> fe_values_mag(fe_mag, quadrature_cell,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_U(fe_U, quadrature_cell,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
        dealii::update_values);

    // FEInterfaceValues for DG face terms (on the FESystem)
    dealii::FEInterfaceValues<dim> fe_interface_mag(fe_mag, quadrature_face,
        dealii::update_values | dealii::update_JxW_values |
        dealii::update_normal_vectors);

    // FEFaceValues for U at faces (CG velocity, continuous across faces)
    dealii::FEFaceValues<dim> fe_face_U(fe_U, quadrature_face,
        dealii::update_values);

    // Pre-allocated storage for cell field values
    std::vector<double> theta_vals(n_q_cell);
    std::vector<double> Ux_vals(n_q_cell), Uy_vals(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> grad_Ux(n_q_cell), grad_Uy(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> M_old_vals(n_q_cell);  // Vector M^{k-1}

    // Face field values
    std::vector<double> Ux_face(n_q_face), Uy_face(n_q_face);

    // Local contributions
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    // Face matrices (scalar, n_M × n_M) — shared by all M components
    dealii::FullMatrix<double> face_hh(n_M, n_M);
    dealii::FullMatrix<double> face_ht(n_M, n_M);
    dealii::FullMatrix<double> face_th(n_M, n_M);
    dealii::FullMatrix<double> face_tt(n_M, n_M);
    std::vector<dealii::types::global_dof_index> dofs_here(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> dofs_there(dofs_per_cell);

    // Physical parameters
    // mass_coeff: (1/dt + 1/tau_M) — combines time derivative and relaxation mass
    // old_coeff:  1/dt             — time derivative RHS
    // relax_coeff: 1/tau_M         — relaxation coupling to grad phi
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff = 1.0 / dt + ((tau_M_val > 0.0) ? 1.0 / tau_M_val : 0.0);
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 0.0;
    const double old_coeff = 1.0 / dt;

    // Initialize
    system_matrix = 0;
    system_rhs = 0;

    // ========================================================================
    // Helper lambda: assemble face flux matrices and distribute to system.
    // Called after fe_interface_mag, fe_face_U, dofs_here, dofs_there are
    // initialized for the current face.
    // ========================================================================
    auto assemble_face_flux = [&]()
    {
        face_hh = 0;
        face_ht = 0;
        face_th = 0;
        face_tt = 0;

        for (unsigned int q = 0; q < n_q_face; ++q)
        {
            const double JxW = fe_interface_mag.JxW(q);
            const auto& normal = fe_interface_mag.normal(q);
            const double U_dot_n = Ux_face[q] * normal[0] + Uy_face[q] * normal[1];

            for (unsigned int i = 0; i < n_M; ++i)
            {
                // Local DoF index in FESystem for component 0's i-th DG basis
                const unsigned int li_h = M_comp_local_dofs_[0][i];
                // Shape function values on each side
                const double phi_i_h = fe_interface_mag.shape_value(true, li_h, q);
                const double phi_i_t = fe_interface_mag.shape_value(false, dofs_per_cell + li_h, q);

                for (unsigned int j = 0; j < n_M; ++j)
                {
                    const unsigned int lj_h = M_comp_local_dofs_[0][j];
                    const double phi_j_h = fe_interface_mag.shape_value(true, lj_h, q);
                    const double phi_j_t = fe_interface_mag.shape_value(false, dofs_per_cell + lj_h, q);

                    // Face flux: -(U·n) [[V]] {W}  (Eq. 57, second line)
                    // V = phi_j (trial, transported), W = phi_i (test)
                    face_hh(i, j) += skew_magnetic_face_value_scalar_interface(
                        U_dot_n, phi_j_h, 0.0, phi_i_h, 0.0) * JxW;
                    face_ht(i, j) += skew_magnetic_face_value_scalar_interface(
                        U_dot_n, 0.0, phi_j_t, phi_i_h, 0.0) * JxW;
                    face_th(i, j) += skew_magnetic_face_value_scalar_interface(
                        U_dot_n, phi_j_h, 0.0, 0.0, phi_i_t) * JxW;
                    face_tt(i, j) += skew_magnetic_face_value_scalar_interface(
                        U_dot_n, 0.0, phi_j_t, 0.0, phi_i_t) * JxW;
                }
            }
        }

        // Distribute scalar face matrix to global matrix for each M component.
        // All M components share the same DG basis, so the scalar face matrix
        // is identical — only the global DoF indices differ per component.
        for (unsigned int d = 0; d < static_cast<unsigned int>(dim); ++d)
        {
            for (unsigned int i = 0; i < n_M; ++i)
            {
                const auto gi_h = dofs_here[M_comp_local_dofs_[d][i]];
                const auto gi_t = dofs_there[M_comp_local_dofs_[d][i]];
                for (unsigned int j = 0; j < n_M; ++j)
                {
                    const auto gj_h = dofs_here[M_comp_local_dofs_[d][j]];
                    const auto gj_t = dofs_there[M_comp_local_dofs_[d][j]];

                    system_matrix.add(gi_h, gj_h, face_hh(i, j));
                    system_matrix.add(gi_h, gj_t, face_ht(i, j));
                    system_matrix.add(gi_t, gj_h, face_th(i, j));
                    system_matrix.add(gi_t, gj_t, face_tt(i, j));
                }
            }
        }
    };

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    auto cell_mag = mag_dof_handler_.begin_active();
    auto cell_U = U_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_mag != mag_dof_handler_.end();
         ++cell_mag, ++cell_U, ++cell_theta)
    {
        if (!cell_mag->is_locally_owned())
            continue;

        fe_values_mag.reinit(cell_mag);
        fe_values_U.reinit(cell_U);
        fe_values_theta.reinit(cell_theta);

        cell_matrix = 0;
        cell_rhs = 0;

        // Get field values at quadrature points
        fe_values_theta.get_function_values(theta, theta_vals);
        fe_values_U.get_function_values(Ux, Ux_vals);
        fe_values_U.get_function_values(Uy, Uy_vals);
        fe_values_U.get_function_gradients(Ux, grad_Ux);
        fe_values_U.get_function_gradients(Uy, grad_Uy);

        // Extract M component from combined mag_old (FESystem)
        fe_values_mag[M].get_function_values(mag_old, M_old_vals);

        // ====================================================================
        // Cell integrals: all 4 blocks + transport
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_mag.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_mag.quadrature_point(q);

            // Velocity and divergence
            dealii::Tensor<1, dim> U_vec;
            U_vec[0] = Ux_vals[q];
            U_vec[1] = Uy_vals[q];
            const double div_U = grad_Ux[q][0] + grad_Uy[q][1];

            // Susceptibility chi(theta)
            const double chi_theta = susceptibility(
                theta_vals[q], params_.physics.epsilon, params_.physics.chi_0);

            // Applied field (zero in MMS mode)
            dealii::Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);

            // MMS sources
            double f_mms_phi = 0.0;
            dealii::Tensor<1, dim> f_mms_M;
            if (mms_mode)
            {
                // phi source: coupled (-Delta phi* - div M*)
                f_mms_phi = compute_poisson_mms_source_coupled<dim>(
                    x_q, current_time, L_y);

                // M source: full PDE with transport (Eq. 42c)
                // Uses exact H = grad(phi*) for the relaxation term
                dealii::Tensor<1, dim> H_exact = poisson_mms_exact_H<dim>(
                    x_q, current_time, L_y);
                f_mms_M = compute_mag_mms_source_with_transport<dim>(
                    x_q, current_time, t_old, tau_M_val, chi_theta,
                    H_exact, U_vec, div_U, L_y);
            }

            // ================================================================
            // Matrix entries: loop over test (i) and trial (j) DoFs
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Test function values via extractors
                const dealii::Tensor<1, dim> Z_i = fe_values_mag[M].value(i, q);
                const double X_i = fe_values_mag[phi].value(i, q);
                const dealii::Tensor<1, dim> grad_X_i = fe_values_mag[phi].gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Trial function values via extractors
                    const dealii::Tensor<1, dim> M_j = fe_values_mag[M].value(j, q);
                    const dealii::Tensor<2, dim> grad_M_j = fe_values_mag[M].gradient(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = fe_values_mag[phi].gradient(j, q);

                    double val = 0.0;

                    // A_M: mass (1/dt + 1/tau_M)(M_j, Z_i)
                    val += mass_coeff * (M_j * Z_i);

                    // A_M: transport cell (Eq. 57, first line)
                    // (U·∇M_j)·Z_i + ½(∇·U)(M_j·Z_i)
                    // (U·∇)M_j = grad_M_j * U_vec  (Tensor<2,dim> * Tensor<1,dim>)
                    val += (grad_M_j * U_vec) * Z_i + 0.5 * div_U * (M_j * Z_i);

                    // C_M_phi: Relaxation coupling — depends on --full_mag flag
                    if (params_.use_full_mag_model)
                    {
                        // Full model (Eq 42c): -(1/tau_M) chi(theta) (∇φ_j, Z_i)
                        // M relaxes toward chi(theta) * ∇φ (full H field)
                        val += -relax_coeff * chi_theta * (grad_phi_j * Z_i);
                    }
                    // else: simplified model (Section 5/6) — no M-phi coupling,
                    // relaxation source (h_a) on RHS instead

                    // C_phi_M: +(M_j, grad X_i)
                    val += M_j * grad_X_i;

                    // A_phi: (grad phi_j, grad X_i)
                    val += grad_phi_j * grad_X_i;

                    cell_matrix(i, j) += val * JxW;
                }

                // ============================================================
                // RHS
                // ============================================================
                double rhs_val = 0.0;

                // M block: (1/dt)(M^{k-1}, Z_i) — time derivative
                rhs_val += old_coeff * (M_old_vals[q] * Z_i);

                // M block: relaxation source (depends on --full_mag)
                if (!params_.use_full_mag_model)
                {
                    // Simplified (Section 5/6): +(1/tau_M) chi(theta) (h_a, Z_i)
                    rhs_val += relax_coeff * chi_theta * (h_a * Z_i);
                }
                // Full model: relaxation is on LHS via C_M_phi coupling (no RHS source)

                // phi block: (h_a, grad X_i)
                rhs_val += h_a * grad_X_i;

                // MMS sources
                if (mms_mode)
                {
                    rhs_val += f_mms_M * Z_i;
                    rhs_val += f_mms_phi * X_i;
                }

                cell_rhs(i) += rhs_val * JxW;
            }
        }

        // Distribute cell contributions (handles phi constraints)
        cell_mag->get_dof_indices(local_dofs);
        mag_constraints_.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dofs,
            system_matrix, system_rhs);

        // ====================================================================
        // FACE CONTRIBUTIONS: DG transport (Eq. 57, second line)
        //
        // Interior faces only — boundary faces have no neighbor, and for
        // periodic/Neumann BCs the face flux is zero (no jump at boundary).
        //
        // Face integrand: -(U·n) [[M]]·{Z}
        // Assembled componentwise: same scalar face matrix for each M component.
        // ====================================================================
        for (unsigned int f = 0; f < cell_mag->n_faces(); ++f)
        {
            if (cell_mag->at_boundary(f))
                continue;

            const auto neighbor_mag = cell_mag->neighbor(f);

            // PARALLEL: For cross-rank faces, only lower-index cell processes
            if (!neighbor_mag->is_locally_owned() &&
                cell_mag->index() > neighbor_mag->index())
                continue;

            // AMR Case 1: Neighbor is coarser — handle from fine side
            if (cell_mag->neighbor_is_coarser(f))
            {
                const auto neighbor_info =
                    cell_mag->neighbor_of_coarser_neighbor(f);
                const unsigned int neighbor_face = neighbor_info.first;
                const unsigned int neighbor_subface = neighbor_info.second;

                fe_interface_mag.reinit(cell_mag, f,
                    dealii::numbers::invalid_unsigned_int,
                    neighbor_mag, neighbor_face, neighbor_subface);

                // U at face quadrature points (CG, continuous across faces)
                fe_face_U.reinit(cell_U, f);
                fe_face_U.get_function_values(Ux, Ux_face);
                fe_face_U.get_function_values(Uy, Uy_face);

                cell_mag->get_dof_indices(dofs_here);
                neighbor_mag->get_dof_indices(dofs_there);

                assemble_face_flux();
            }
            // AMR Case 2: Same level neighbor (must be active)
            else if (cell_mag->level() == neighbor_mag->level() &&
                     neighbor_mag->is_active())
            {
                // Only process once: lower index handles it
                if (cell_mag->index() > neighbor_mag->index())
                    continue;

                const unsigned int neighbor_face =
                    cell_mag->neighbor_of_neighbor(f);

                fe_interface_mag.reinit(cell_mag, f,
                    dealii::numbers::invalid_unsigned_int,
                    neighbor_mag, neighbor_face,
                    dealii::numbers::invalid_unsigned_int);

                // U at face quadrature points
                fe_face_U.reinit(cell_U, f);
                fe_face_U.get_function_values(Ux, Ux_face);
                fe_face_U.get_function_values(Uy, Uy_face);

                cell_mag->get_dof_indices(dofs_here);
                neighbor_mag->get_dof_indices(dofs_there);

                assemble_face_flux();
            }
            // AMR Case 3: Neighbor is finer — skip, handled by fine cells
        }
    } // end cell loop

    // Synchronize parallel contributions
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticAssembler<2>;
