// ============================================================================
// assemblers/magnetization_assembler.cc - DG Magnetization Equation Assembler
//                                          (PAPER_MATCH v2)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c, Eq. 57
//
// FIX: chi() now uses params_.physics.chi_0 instead of global chi_0.
// MMS: When enable_mms=true, adds MMS source term to RHS.
//
// Assembles the scalar DG magnetization system using skew_forms.h.
//
// EQUATION 42c (rearranged):
//   (1/τ + 1/T)(M^k, Z) - B_h^m(U^{k-1}, Z, M^k) = (1/T)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// where B_h^m is defined by Eq. 57 (see skew_forms.h).
//
// IMPORTANT: Caller must pass U^{k-1} (lagged velocity), NOT U^k.
// This is required for:
//   - Decoupled NS-M solve (paper algorithm)
//   - Energy stability (B_h^m(U,M,M) = 0 globally)
//
// ============================================================================

#include "assembly/magnetization_assembler.h"
#include "mms/magnetization_mms.h"
#include "physics/skew_forms.h"
#include "physics/material_properties.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/quadrature_lib.h>


// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagnetizationAssembler<dim>::MagnetizationAssembler(
    const Parameters& params,
    const dealii::DoFHandler<dim>& M_dof,
    const dealii::DoFHandler<dim>& U_dof,
    const dealii::DoFHandler<dim>& phi_dof,
    const dealii::DoFHandler<dim>& theta_dof)
    : params_(params)
    , M_dof_handler_(M_dof)
    , U_dof_handler_(U_dof)
    , phi_dof_handler_(phi_dof)
    , theta_dof_handler_(theta_dof)
{
    // All DoFHandlers must share the same triangulation
    Assert(&M_dof_handler_.get_triangulation() == &U_dof_handler_.get_triangulation(),
           dealii::ExcMessage("M and U DoFHandlers must share the same triangulation"));
    Assert(&M_dof_handler_.get_triangulation() == &phi_dof_handler_.get_triangulation(),
           dealii::ExcMessage("M and phi DoFHandlers must share the same triangulation"));
    Assert(&M_dof_handler_.get_triangulation() == &theta_dof_handler_.get_triangulation(),
           dealii::ExcMessage("M and theta DoFHandlers must share the same triangulation"));
}

// ============================================================================
// Susceptibility function: χ(θ) = χ₀(1+θ)/2 - uses params_.physics.chi_0!
// ============================================================================
template <int dim>
double MagnetizationAssembler<dim>::chi(double theta_val) const
{
    return susceptibility(theta_val, params_.physics.epsilon, params_.physics.chi_0);
}

// ============================================================================
// Main assembly routine
// ============================================================================
template <int dim>
void MagnetizationAssembler<dim>::assemble(
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& rhs_x,
    dealii::Vector<double>& rhs_y,
    const dealii::Vector<double>& Ux,
    const dealii::Vector<double>& Uy,
    const dealii::Vector<double>& phi,
    const dealii::Vector<double>& theta,
    const dealii::Vector<double>& Mx_old,
    const dealii::Vector<double>& My_old,
    double dt,
    double current_time) const
{
    const bool mms_mode = params_.enable_mms;
    const double L_y = params_.domain.y_max - params_.domain.y_min;
    const double t_old = current_time - dt;

    const dealii::FiniteElement<dim>& fe_M = M_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_U = U_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_phi = phi_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_theta = theta_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_M.dofs_per_cell;

    // Quadrature
    dealii::QGauss<dim> quadrature_cell(fe_M.degree + 2);
    dealii::QGauss<dim-1> quadrature_face(fe_M.degree + 2);

    const unsigned int n_q_cell = quadrature_cell.size();
    const unsigned int n_q_face = quadrature_face.size();

    // FEValues for cells
    dealii::FEValues<dim> fe_values_M(fe_M, quadrature_cell,
                               dealii::update_values | dealii::update_gradients |
                               dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_U(fe_U, quadrature_cell,
                               dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> fe_values_phi(fe_phi, quadrature_cell,
                                 dealii::update_gradients | dealii::update_quadrature_points);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
                                   dealii::update_values);

    // FEInterfaceValues for faces
    dealii::FEInterfaceValues<dim> fe_interface_M(fe_M, quadrature_face,
                                           dealii::update_values | dealii::update_JxW_values | dealii::update_normal_vectors);

    // Storage for field values
    std::vector<double> Ux_vals(n_q_cell), Uy_vals(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> grad_Ux(n_q_cell), grad_Uy(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> grad_phi(n_q_cell);
    std::vector<double> theta_vals(n_q_cell);
    std::vector<double> Mx_old_vals(n_q_cell), My_old_vals(n_q_cell);

    // Local contributions
    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    // Parameters
    const double tau = dt;

    // Coefficients: (1/τ + 1/T) for LHS, 1/T and 1/τ for RHS
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff = (tau_M_val > 0.0) ? (1.0/tau + 1.0/tau_M_val) : 1.0/tau;
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0/tau_M_val : 0.0;
    const double old_coeff = 1.0/tau;

    // Initialize
    system_matrix = 0;
    rhs_x = 0;
    rhs_y = 0;

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    auto cell_M = M_dof_handler_.begin_active();
    auto cell_U = U_dof_handler_.begin_active();
    auto cell_phi = phi_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_M != M_dof_handler_.end();
         ++cell_M, ++cell_U, ++cell_phi, ++cell_theta)
    {
        fe_values_M.reinit(cell_M);
        fe_values_U.reinit(cell_U);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        local_matrix = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        // Get field values
        fe_values_U.get_function_values(Ux, Ux_vals);
        fe_values_U.get_function_values(Uy, Uy_vals);
        fe_values_U.get_function_gradients(Ux, grad_Ux);
        fe_values_U.get_function_gradients(Uy, grad_Uy);
        fe_values_phi.get_function_gradients(phi, grad_phi);
        fe_values_theta.get_function_values(theta, theta_vals);
        fe_values_M.get_function_values(Mx_old, Mx_old_vals);
        fe_values_M.get_function_values(My_old, My_old_vals);

        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_M.quadrature_point(q);

            // U and div(U)
            dealii::Tensor<1, dim> U;
            U[0] = Ux_vals[q];
            U[1] = Uy_vals[q];
            const double div_U = grad_Ux[q][0] + grad_Uy[q][1];

            // H = h_a + h_d where:
            //   h_a = applied field from dipoles/bars
            //   h_d = ∇φ = demagnetizing field from Poisson
            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params_, current_time);
            dealii::Tensor<1, dim> H;
            H[0] = h_a[0] + grad_phi[q][0];
            H[1] = h_a[1] + grad_phi[q][1];

            // χ(θ)
            const double chi_theta = chi(theta_vals[q]);

            // MMS source term
            dealii::Tensor<1, dim> F_mms;
            F_mms = 0;
            if (mms_mode)
            {
                // For MMS: compute source term that makes exact solution satisfy equation
                F_mms = compute_mag_mms_source_with_transport<dim>(
                    x_q, current_time, t_old, tau_M_val, chi_theta, H, U, div_U, L_y);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);
                const dealii::Tensor<1, dim>& grad_phi_i = fe_values_M.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values_M.shape_value(j, q);

                    // Mass: (1/τ + 1/T)(φ_j, φ_i)
                    double mass = mass_coeff * phi_j * phi_i;

                    // Transport cell: -B_h^m cell term with Z=φ_i (test), M=φ_j (trial)
                    // B_h^m cell = (U·∇Z)M + (1/2)(∇·U)(Z M)
                    // We want -B_h^m, so negate
                    double transport = -skew_magnetic_cell_value_scalar<dim>(
                        U, div_U, phi_i, grad_phi_i, phi_j);

                    local_matrix(i, j) += (mass + transport) * JxW;
                }

                // RHS: (1/T)(χ_θ H, φ_i) + (1/τ)(M_old, φ_i) + (F_mms, φ_i)
                local_rhs_x(i) += (relax_coeff * chi_theta * H[0] + old_coeff * Mx_old_vals[q] + F_mms[0]) * phi_i * JxW;
                local_rhs_y(i) += (relax_coeff * chi_theta * H[1] + old_coeff * My_old_vals[q] + F_mms[1]) * phi_i * JxW;
            }
        }

        // Distribute cell contributions
        cell_M->get_dof_indices(local_dofs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            rhs_x(local_dofs[i]) += local_rhs_x(i);
            rhs_y(local_dofs[i]) += local_rhs_y(i);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dofs[i], local_dofs[j], local_matrix(i, j));
        }

        // ====================================================================
        // FACE CONTRIBUTIONS (interior faces only)
        //
        // Paper Eq. 57 face term: -(U·n)[[V]]·{W}
        // We assemble -B_h^m, so the contribution is: +(U·n)[[Z]]·{M}
        //
        // For DG basis functions supported on single cells:
        //   - φ_i on "here" cell: [[φ_i]] = φ_i_here (test contributes +)
        //   - φ_i on "there" cell: [[φ_i]] = -φ_i_there (test contributes -)
        //   - {φ_j} = 0.5 * (value on here + value on there)
        // ====================================================================
        for (unsigned int f = 0; f < cell_M->n_faces(); ++f)
        {
            if (cell_M->at_boundary(f))
                continue;

            const auto neighbor_M = cell_M->neighbor(f);

            // Process each face once
            if (neighbor_M->is_active() && cell_M->index() > neighbor_M->index())
                continue;

            const auto neighbor_U = cell_U->neighbor(f);
            const unsigned int nf = cell_M->neighbor_of_neighbor(f);

            // Reinit interface
            fe_interface_M.reinit(cell_M, f, dealii::numbers::invalid_unsigned_int,
                                   neighbor_M, nf, dealii::numbers::invalid_unsigned_int);

            // U values on face (for computing U·n)
            dealii::FEFaceValues<dim> fe_face_U_here(fe_U, quadrature_face, dealii::update_values);
            dealii::FEFaceValues<dim> fe_face_U_there(fe_U, quadrature_face, dealii::update_values);
            fe_face_U_here.reinit(cell_U, f);
            fe_face_U_there.reinit(neighbor_U, nf);

            std::vector<double> Ux_here(n_q_face), Uy_here(n_q_face);
            std::vector<double> Ux_there(n_q_face), Uy_there(n_q_face);
            fe_face_U_here.get_function_values(Ux, Ux_here);
            fe_face_U_here.get_function_values(Uy, Uy_here);
            fe_face_U_there.get_function_values(Ux, Ux_there);
            fe_face_U_there.get_function_values(Uy, Uy_there);

            // DoF indices
            std::vector<dealii::types::global_dof_index> dofs_here(dofs_per_cell);
            std::vector<dealii::types::global_dof_index> dofs_there(dofs_per_cell);
            cell_M->get_dof_indices(dofs_here);
            neighbor_M->get_dof_indices(dofs_there);

            // Face matrices (4 blocks for cell-cell coupling)
            dealii::FullMatrix<double> face_hh(dofs_per_cell, dofs_per_cell);
            dealii::FullMatrix<double> face_ht(dofs_per_cell, dofs_per_cell);
            dealii::FullMatrix<double> face_th(dofs_per_cell, dofs_per_cell);
            dealii::FullMatrix<double> face_tt(dofs_per_cell, dofs_per_cell);

            face_hh = 0; face_ht = 0; face_th = 0; face_tt = 0;

            for (unsigned int q = 0; q < n_q_face; ++q)
            {
                const double JxW = fe_interface_M.JxW(q);
                const dealii::Tensor<1, dim>& normal = fe_interface_M.normal(q);

                // U·n⁻: evaluated on minus (here) side, consistent with skew_forms.h
                // For CG velocity this equals the trace, but we use minus-side
                // to align with the documented convention.
                const double U_dot_n_minus = Ux_here[q] * normal[0] + Uy_here[q] * normal[1];

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const double phi_i_here = fe_interface_M.shape_value(true, i, q);
                    const double phi_i_there = fe_interface_M.shape_value(false, i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const double phi_j_here = fe_interface_M.shape_value(true, j, q);
                        const double phi_j_there = fe_interface_M.shape_value(false, j, q);

                        // Compute -B_h^m face term using skew_forms.h
                        //
                        // B_h^m face = -(U·n)[[Z]]{M}  (from skew_magnetic_face_value_scalar)
                        // We want: -B_h^m = -[-(U·n)[[Z]]{M}] = (U·n)[[Z]]{M}
                        //        = -skew_magnetic_face_value_scalar_interface(...)
                        //
                        // For basis functions supported on single cells:
                        //   - Test φ_i on here:  Z_here = φ_i_here, Z_there = 0
                        //   - Test φ_i on there: Z_here = 0, Z_there = φ_i_there
                        //   - Trial φ_j on here:  M_here = φ_j_here, M_there = 0
                        //   - Trial φ_j on there: M_here = 0, M_there = φ_j_there

                        // here-here block: test on here, trial on here
                        face_hh(i, j) += -skew_magnetic_face_value_scalar_interface(
                            U_dot_n_minus, phi_i_here, 0.0, phi_j_here, 0.0) * JxW;

                        // here-there block: test on here, trial on there
                        face_ht(i, j) += -skew_magnetic_face_value_scalar_interface(
                            U_dot_n_minus, phi_i_here, 0.0, 0.0, phi_j_there) * JxW;

                        // there-here block: test on there, trial on here
                        face_th(i, j) += -skew_magnetic_face_value_scalar_interface(
                            U_dot_n_minus, 0.0, phi_i_there, phi_j_here, 0.0) * JxW;

                        // there-there block: test on there, trial on there
                        face_tt(i, j) += -skew_magnetic_face_value_scalar_interface(
                            U_dot_n_minus, 0.0, phi_i_there, 0.0, phi_j_there) * JxW;
                    }
                }
            }

            // Distribute face contributions
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    system_matrix.add(dofs_here[i], dofs_here[j], face_hh(i, j));
                    system_matrix.add(dofs_here[i], dofs_there[j], face_ht(i, j));
                    system_matrix.add(dofs_there[i], dofs_here[j], face_th(i, j));
                    system_matrix.add(dofs_there[i], dofs_there[j], face_tt(i, j));
                }
            }
        }
    }
}

// ============================================================================
// RHS-only assembly
// ============================================================================
template <int dim>
void MagnetizationAssembler<dim>::assemble_rhs_only(
    dealii::Vector<double>& rhs_x,
    dealii::Vector<double>& rhs_y,
    const dealii::Vector<double>& phi,
    const dealii::Vector<double>& theta,
    const dealii::Vector<double>& Mx_old,
    const dealii::Vector<double>& My_old,
    double dt) const
{

    const dealii::FiniteElement<dim>& fe_M = M_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_phi = phi_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_theta = theta_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_M.dofs_per_cell;

    dealii::QGauss<dim> quadrature_cell(fe_M.degree + 2);
    const unsigned int n_q_cell = quadrature_cell.size();

    dealii::FEValues<dim> fe_values_M(fe_M, quadrature_cell,
                               dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_phi(fe_phi, quadrature_cell,
                                 dealii::update_gradients);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
                                   dealii::update_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi(n_q_cell);
    std::vector<double> theta_vals(n_q_cell);
    std::vector<double> Mx_old_vals(n_q_cell), My_old_vals(n_q_cell);

    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    const double tau = dt;
    const double tau_M_val = params_.physics.tau_M;
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0/tau_M_val : 0.0;
    const double old_coeff = 1.0/tau;

    rhs_x = 0;
    rhs_y = 0;

    auto cell_M = M_dof_handler_.begin_active();
    auto cell_phi = phi_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_M != M_dof_handler_.end(); ++cell_M, ++cell_phi, ++cell_theta)
    {
        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        local_rhs_x = 0;
        local_rhs_y = 0;

        fe_values_phi.get_function_gradients(phi, grad_phi);
        fe_values_theta.get_function_values(theta, theta_vals);
        fe_values_M.get_function_values(Mx_old, Mx_old_vals);
        fe_values_M.get_function_values(My_old, My_old_vals);

        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_M.JxW(q);

            // H = h_a + h_d (applied + demagnetizing)
            // Note: assemble_rhs_only doesn't have current_time, so h_a = 0
            // This function is only used when matrix is fixed (U=0), so typically MMS/standalone
            dealii::Tensor<1, dim> H;
            H[0] = grad_phi[q][0];
            H[1] = grad_phi[q][1];
            // TO-DO: If h_a is needed in rhs_only, add current_time parameter

            const double chi_theta = chi(theta_vals[q]);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                local_rhs_x(i) += (relax_coeff * chi_theta * H[0] + old_coeff * Mx_old_vals[q]) * phi_i * JxW;
                local_rhs_y(i) += (relax_coeff * chi_theta * H[1] + old_coeff * My_old_vals[q]) * phi_i * JxW;
            }
        }

        cell_M->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            rhs_x(local_dofs[i]) += local_rhs_x(i);
            rhs_y(local_dofs[i]) += local_rhs_y(i);
        }
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class MagnetizationAssembler<2>;
template class MagnetizationAssembler<3>;