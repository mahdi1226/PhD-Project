// ============================================================================
// assembly/magnetic_assembler.cc - Monolithic Magnetics Assembler (PARALLEL)
//
// Full M transport + Poisson assembler (Paper Eq 42c-42d):
//
//   Eq 42c (M transport):
//     (δM^k/τ, Z) - B_h^m(U^k, Z, M^k) + (1/T)(M^k, Z) = (1/T)(χ_Θ H^k, Z)
//
//   Eq 42d (Poisson):
//     (∇Φ^k, ∇X) + (M^k, ∇X) = (h_a, ∇X)
//
// B_h^m(U, Z, M) is the DG skew-symmetric trilinear form (Eq 57):
//   Cell: (U·∇)Z·M + ½div(U)Z·M
//   Face: -([[Z]]·{M})(U·n) dS
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
#include "mms/magnetic/poisson_mms.h"
#include "mms/magnetic/magnetization_mms.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
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
    , mpi_communicator_(mpi_communicator)
{
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

    const auto& fe_mag = mag_dof_handler_.get_fe();
    const auto& fe_theta = theta_dof_handler_.get_fe();
    const auto& fe_vel = U_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_mag.n_dofs_per_cell();

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
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
        dealii::update_values);
    dealii::FEValues<dim> fe_values_vel(fe_vel, quadrature_cell,
        dealii::update_values | dealii::update_gradients);

    // FEFaceValues for DG convection face integrals
    dealii::FEFaceValues<dim> fe_face_mag_minus(fe_mag, quadrature_face,
        dealii::update_values | dealii::update_normal_vectors |
        dealii::update_JxW_values);
    dealii::FEFaceValues<dim> fe_face_mag_plus(fe_mag, quadrature_face,
        dealii::update_values);
    dealii::FEFaceValues<dim> fe_face_vel(fe_vel, quadrature_face,
        dealii::update_values);

    // FESubfaceValues for AMR: coarser cell side when neighbor is finer
    dealii::FESubfaceValues<dim> fe_subface_mag_minus(fe_mag, quadrature_face,
        dealii::update_values | dealii::update_normal_vectors |
        dealii::update_JxW_values);
    dealii::FESubfaceValues<dim> fe_subface_vel(fe_vel, quadrature_face,
        dealii::update_values);

    // Pre-allocated storage for field values
    std::vector<double> theta_vals(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> Ux_grads(n_q_cell);  // for div(U)
    std::vector<dealii::Tensor<1, dim>> Uy_grads(n_q_cell);
    std::vector<double> Ux_vals(n_q_cell);
    std::vector<double> Uy_vals(n_q_cell);
    // M_old values at quadrature points for time derivative RHS
    std::vector<dealii::Tensor<1, dim>> M_old_vals(n_q_cell);

    // Face quadrature storage
    std::vector<double> Ux_face_vals(n_q_face);
    std::vector<double> Uy_face_vals(n_q_face);

    // Local contributions
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    // Face contributions: combined (2n × 2n) matrix for both cells
    // Rows/cols [0..n-1] = minus cell, [n..2n-1] = plus cell
    const unsigned int total_face_dofs = 2 * dofs_per_cell;
    dealii::FullMatrix<double> face_matrix(total_face_dofs, total_face_dofs);
    std::vector<dealii::types::global_dof_index> face_dof_indices(total_face_dofs);
    std::vector<dealii::types::global_dof_index> neighbor_dofs(dofs_per_cell);

    // Physical parameters
    const double tau_M_val = params_.physics.tau_M;
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 1.0;
    const double time_coeff = (dt > 0.0) ? 1.0 / dt : 0.0;
    // Total mass coefficient: (1/dt + 1/tau_M) for M block
    const double mass_coeff = time_coeff + relax_coeff;

    // Initialize
    system_matrix = 0;
    system_rhs = 0;

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    auto cell_mag = mag_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();
    auto cell_vel = U_dof_handler_.begin_active();

    for (; cell_mag != mag_dof_handler_.end();
         ++cell_mag, ++cell_theta, ++cell_vel)
    {
        if (!cell_mag->is_locally_owned())
            continue;

        fe_values_mag.reinit(cell_mag);
        fe_values_theta.reinit(cell_theta);
        fe_values_vel.reinit(cell_vel);

        cell_matrix = 0;
        cell_rhs = 0;

        // Get field values at quadrature points
        fe_values_theta.get_function_values(theta, theta_vals);
        fe_values_vel.get_function_values(Ux, Ux_vals);
        fe_values_vel.get_function_values(Uy, Uy_vals);
        fe_values_vel.get_function_gradients(Ux, Ux_grads);
        fe_values_vel.get_function_gradients(Uy, Uy_grads);

        // M_old at quadrature points (for time derivative RHS)
        fe_values_mag[M].get_function_values(mag_old, M_old_vals);

        // ====================================================================
        // Cell integrals
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_mag.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_mag.quadrature_point(q);

            // Susceptibility chi(theta)
            const double chi_theta = susceptibility(
                theta_vals[q], params_.physics.epsilon, params_.physics.chi_0);

            // Velocity U^k and div(U^k) at quadrature point
            dealii::Tensor<1, dim> U_q;
            U_q[0] = Ux_vals[q];
            U_q[1] = Uy_vals[q];
            const double div_U = Ux_grads[q][0] + Uy_grads[q][1];

            // Applied field (zero in MMS mode)
            dealii::Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);

            // MMS sources
            double f_mms_phi = 0.0;
            dealii::Tensor<1, dim> f_mms_M;
            if (mms_mode)
            {
                f_mms_phi = compute_poisson_mms_source_coupled<dim>(
                    x_q, current_time, L_y);
                f_mms_M = compute_mag_mms_source_equilibrium<dim>(
                    x_q, current_time, tau_M_val, chi_theta, L_y);
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
                // Gradient of M test function (for convection cell term)
                const dealii::Tensor<2, dim> grad_Z_i = fe_values_mag[M].gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Trial function values via extractors
                    const dealii::Tensor<1, dim> M_j = fe_values_mag[M].value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = fe_values_mag[phi].gradient(j, q);

                    double val = 0.0;

                    // ==== M block (Eq 42c) ====

                    // A_M: (1/dt + 1/tau_M)(M_j, Z_i)
                    val += mass_coeff * (M_j * Z_i);

                    // Convection cell term: -B_h^m(U, Z, M) cell part
                    // -[(U·∇)Z·M + ½div(U)Z·M]
                    // (U·∇)Z is a tensor: (U·∇)Z_i[a][b] = Σ_c U_c ∂Z_i^a/∂x_c
                    // But Z_i is vector-valued. (U·∇)Z_i · M_j = Σ_a [(U·∇)Z_i^a] * M_j^a
                    double U_grad_Z_dot_M = 0.0;
                    for (unsigned int a = 0; a < dim; ++a)
                    {
                        double U_grad_Z_a = 0.0;
                        for (unsigned int c = 0; c < dim; ++c)
                            U_grad_Z_a += U_q[c] * grad_Z_i[a][c];
                        U_grad_Z_dot_M += U_grad_Z_a * M_j[a];
                    }
                    val += -(U_grad_Z_dot_M + 0.5 * div_U * (Z_i * M_j));

                    // C_M_phi: -(1/tau_M) chi (grad phi_j, Z_i)
                    val += -relax_coeff * chi_theta * (grad_phi_j * Z_i);

                    // ==== phi block (Eq 42d) ====

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

                // M block RHS: (1/dt)(M^{k-1}, Z_i)
                rhs_val += time_coeff * (M_old_vals[q] * Z_i);

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
    } // end cell loop

    // ========================================================================
    // FACE LOOP: DG convection face term from -B_h^m(U, Z, M)
    //
    // Face part of -B_h^m(U, Z, M) = +[[Z]]·{M}(U·n) dS
    //
    // For DG test Z and trial M on both cells:
    //   minus-test, minus-trial: +Z_i⁻ · ½M_j⁻ · (U·n⁻)
    //   minus-test, plus-trial:  +Z_i⁻ · ½M_j⁺ · (U·n⁻)
    //   plus-test,  minus-trial: -Z_i⁺ · ½M_j⁻ · (U·n⁻)
    //   plus-test,  plus-trial:  -Z_i⁺ · ½M_j⁺ · (U·n⁻)
    //
    // Only involves M block (phi is CG, no face terms for phi).
    //
    // AMR handling: three cases per face
    //   1. Same-level neighbor: standard face integral (process once)
    //   2. Finer neighbor (face->has_children): loop over subfaces,
    //      use FESubfaceValues on coarser (minus) side
    //   3. Coarser neighbor: skip — coarser cell handles via case 2
    // ========================================================================

    // Lambda: assemble face kernel given FEValuesBase refs for minus/plus sides.
    // fe_minus provides JxW, normals, and minus-cell M test/trial values.
    // fe_plus provides plus-cell M test/trial values.
    // fe_vel_face provides velocity at face quadrature points.
    auto assemble_face_kernel = [&](
        const dealii::FEValuesBase<dim>& fe_minus,
        const dealii::FEValuesBase<dim>& fe_plus,
        const dealii::FEValuesBase<dim>& fe_vel_face,
        const typename dealii::DoFHandler<dim>::active_cell_iterator& cell_minus,
        const typename dealii::DoFHandler<dim>::active_cell_iterator& cell_plus)
    {
        // Get velocity at face quadrature points
        fe_vel_face.get_function_values(Ux, Ux_face_vals);
        fe_vel_face.get_function_values(Uy, Uy_face_vals);

        // Get DoF indices — combined [minus | plus]
        cell_minus->get_dof_indices(local_dofs);
        cell_plus->get_dof_indices(neighbor_dofs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            face_dof_indices[i] = local_dofs[i];
            face_dof_indices[dofs_per_cell + i] = neighbor_dofs[i];
        }

        // Zero combined face matrix
        face_matrix = 0;

        const unsigned int n_q = fe_minus.n_quadrature_points;
        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW_f = fe_minus.JxW(q);
            const dealii::Tensor<1, dim>& normal = fe_minus.normal_vector(q);

            // U·n at face
            dealii::Tensor<1, dim> U_face;
            U_face[0] = Ux_face_vals[q];
            U_face[1] = Uy_face_vals[q];
            const double U_dot_n = U_face * normal;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const dealii::Tensor<1, dim> Z_minus_i = fe_minus[M].value(i, q);
                const dealii::Tensor<1, dim> Z_plus_i = fe_plus[M].value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const dealii::Tensor<1, dim> M_minus_j = fe_minus[M].value(j, q);
                    const dealii::Tensor<1, dim> M_plus_j = fe_plus[M].value(j, q);

                    // mm: +Z⁻ · ½M⁻ · (U·n)
                    face_matrix(i, j) +=
                        0.5 * (Z_minus_i * M_minus_j) * U_dot_n * JxW_f;
                    // mp: +Z⁻ · ½M⁺ · (U·n)
                    face_matrix(i, dofs_per_cell + j) +=
                        0.5 * (Z_minus_i * M_plus_j) * U_dot_n * JxW_f;
                    // pm: -Z⁺ · ½M⁻ · (U·n)
                    face_matrix(dofs_per_cell + i, j) +=
                        -0.5 * (Z_plus_i * M_minus_j) * U_dot_n * JxW_f;
                    // pp: -Z⁺ · ½M⁺ · (U·n)
                    face_matrix(dofs_per_cell + i, dofs_per_cell + j) +=
                        -0.5 * (Z_plus_i * M_plus_j) * U_dot_n * JxW_f;
                }
            }
        }

        // Distribute combined face matrix to global system
        mag_constraints_.distribute_local_to_global(
            face_matrix, face_dof_indices, system_matrix);
    };

    {
        auto cell_mag_f = mag_dof_handler_.begin_active();
        auto cell_vel_f = U_dof_handler_.begin_active();

        for (; cell_mag_f != mag_dof_handler_.end();
             ++cell_mag_f, ++cell_vel_f)
        {
            if (!cell_mag_f->is_locally_owned())
                continue;

            for (unsigned int face_no = 0;
                 face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                if (cell_mag_f->at_boundary(face_no))
                    continue;

                const auto face = cell_mag_f->face(face_no);

                // ==========================================================
                // Case 1: Face has children → neighbor is FINER
                // Process each subface. Use FESubfaceValues on this (coarser)
                // cell, FEFaceValues on the finer neighbor child.
                // ==========================================================
                if (face->has_children())
                {
                    for (unsigned int subface = 0;
                         subface < face->n_children(); ++subface)
                    {
                        const auto mag_child =
                            cell_mag_f->neighbor_child_on_subface(
                                face_no, subface);

                        // Find which face of the child touches us
                        unsigned int child_face_no = 0;
                        for (unsigned int f = 0;
                             f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
                        {
                            if (!mag_child->at_boundary(f) &&
                                mag_child->neighbor(f) == cell_mag_f)
                            {
                                child_face_no = f;
                                break;
                            }
                        }

                        // Find corresponding velocity child
                        const auto vel_child =
                            cell_vel_f->neighbor_child_on_subface(
                                face_no, subface);

                        // Reinit: subface on coarser side, face on finer side
                        fe_subface_mag_minus.reinit(
                            cell_mag_f, face_no, subface);
                        fe_face_mag_plus.reinit(mag_child, child_face_no);
                        fe_subface_vel.reinit(cell_vel_f, face_no, subface);

                        assemble_face_kernel(
                            fe_subface_mag_minus, fe_face_mag_plus,
                            fe_subface_vel, cell_mag_f, mag_child);
                    }
                    continue;
                }

                // ==========================================================
                // Case 2: Neighbor is COARSER → skip
                // The coarser neighbor handles this face via Case 1.
                // ==========================================================
                if (cell_mag_f->neighbor_is_coarser(face_no))
                    continue;

                // ==========================================================
                // Case 3: SAME-LEVEL neighbor
                // Process each face once using cell ID comparison.
                // ==========================================================
                const auto mag_neighbor = cell_mag_f->neighbor(face_no);

                // Process once: skip if same-level neighbor has lower ID
                if (mag_neighbor->is_locally_owned() &&
                    (mag_neighbor->level() < cell_mag_f->level() ||
                     (mag_neighbor->level() == cell_mag_f->level() &&
                      mag_neighbor->index() < cell_mag_f->index())))
                    continue;

                const unsigned int neighbor_face_no =
                    cell_mag_f->neighbor_of_neighbor(face_no);

                // Reinit face FEValues
                fe_face_mag_minus.reinit(cell_mag_f, face_no);
                fe_face_mag_plus.reinit(mag_neighbor, neighbor_face_no);
                fe_face_vel.reinit(cell_vel_f, face_no);

                assemble_face_kernel(
                    fe_face_mag_minus, fe_face_mag_plus,
                    fe_face_vel, cell_mag_f, mag_neighbor);
            } // end face loop
        } // end cell loop (face pass)
    }

    // Synchronize parallel contributions
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticAssembler<2>;
