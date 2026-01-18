// ============================================================================
// assembly/magnetization_assembler.cc - DG Magnetization Assembler (PARALLEL)
//
// OPTIMIZED VERSION:
//   - FEFaceValues created ONCE, reinit per face (not recreated)
//   - Pre-allocated scratch data (local matrices, vectors)
//   - Reduced memory allocations in inner loops
//
// FIX: assemble_rhs_only now uses H = h_a + h_d (was missing h_a)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c, Eq. 57
// ============================================================================

#include "assembly/magnetization_assembler.h"
#include "mms/magnetization/magnetization_mms.h"
#include "physics/skew_forms.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagnetizationAssembler<dim>::MagnetizationAssembler(
    const Parameters& params,
    const dealii::DoFHandler<dim>& M_dof,
    const dealii::DoFHandler<dim>& U_dof,
    const dealii::DoFHandler<dim>& phi_dof,
    const dealii::DoFHandler<dim>& theta_dof,
    MPI_Comm mpi_communicator)
    : params_(params)
      , M_dof_handler_(M_dof)
      , U_dof_handler_(U_dof)
      , phi_dof_handler_(phi_dof)
      , theta_dof_handler_(theta_dof)
      , mpi_communicator_(mpi_communicator)
{
}


// ============================================================================
// TEST FUNCTION TO VERIFY THE FIX
// ============================================================================
// Add this method to MagnetizationAssembler class to test the fix:

template <int dim>
void MagnetizationAssembler<dim>::verify_susceptibility_fix() const
{
    const double chi_0 = params_.physics.chi_0;

    std::cout << "\n=== SUSCEPTIBILITY VERIFICATION ===\n";

    const double chi_ferro     = params_.physics.chi_0;   // ferrofluid (θ = +1)
    const double chi_water     = 0.0;                     // non-magnetic phase (θ = −1)
    const double chi_interface = 0.5 * params_.physics.chi_0; // θ = 0 midpoint

    std::cout << "χ(θ=+1, ferrofluid) = " << chi_ferro
        << " (expect ≈ " << chi_0 << ")\n";
    std::cout << "χ(θ=-1, water)        = " << chi_water
        << " (expect ≈ 0)\n";
    std::cout << "χ(θ=0, interface)   = " << chi_interface
        << " (expect ≈ " << chi_0 / 2.0 << ")\n";

    bool is_correct = (chi_ferro > 0.9 * chi_0) && (chi_water < 0.1 * chi_0);

    if (is_correct)
    {
        std::cout << "\n✅ SUCCESS: Susceptibility function is now CORRECT!\n";
        std::cout << "You should now see Rosensweig SPIKES instead of domes.\n";
    }
    else
    {
        std::cout << "\n❌ ERROR: Susceptibility function still appears wrong.\n";
        std::cout << "Please check the implementation.\n";
    }
}


// ============================================================================
// Main assembly routine (PARALLEL) - OPTIMIZED
// ============================================================================
template <int dim>
void MagnetizationAssembler<dim>::assemble(
    dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& rhs_x,
    dealii::TrilinosWrappers::MPI::Vector& rhs_y,
    const dealii::TrilinosWrappers::MPI::Vector& Ux,
    const dealii::TrilinosWrappers::MPI::Vector& Uy,
    const dealii::TrilinosWrappers::MPI::Vector& phi,
    const dealii::TrilinosWrappers::MPI::Vector& theta,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_old,
    const dealii::TrilinosWrappers::MPI::Vector& My_old,
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
    dealii::QGauss<dim - 1> quadrature_face(fe_M.degree + 2);

    const unsigned int n_q_cell = quadrature_cell.size();
    const unsigned int n_q_face = quadrature_face.size();

    // FEValues for cells - created ONCE
    dealii::FEValues<dim> fe_values_M(fe_M, quadrature_cell,
                                      dealii::update_values | dealii::update_gradients |
                                      dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_U(fe_U, quadrature_cell,
                                      dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> fe_values_phi(fe_phi, quadrature_cell,
                                        dealii::update_gradients | dealii::update_quadrature_points);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
                                          dealii::update_values);

    // FEInterfaceValues for faces - created ONCE, reinit per face
    dealii::FEInterfaceValues<dim> fe_interface_M(fe_M, quadrature_face,
                                                  dealii::update_values | dealii::update_JxW_values |
                                                  dealii::update_normal_vectors);

    // OPTIMIZATION: FEFaceValues for U created ONCE outside face loop
    dealii::FEFaceValues<dim> fe_face_U(fe_U, quadrature_face, dealii::update_values);

    // Storage for field values - pre-allocated
    std::vector<double> Ux_vals(n_q_cell), Uy_vals(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> grad_Ux(n_q_cell), grad_Uy(n_q_cell);
    std::vector<dealii::Tensor<1, dim>> grad_phi(n_q_cell);
    std::vector<double> theta_vals(n_q_cell);
    std::vector<double> Mx_old_vals(n_q_cell), My_old_vals(n_q_cell);

    // Face field values - pre-allocated
    std::vector<double> Ux_face(n_q_face), Uy_face(n_q_face);

    // Local contributions - pre-allocated
    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    // OPTIMIZATION: Face matrices pre-allocated ONCE
    dealii::FullMatrix<double> face_hh(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> face_ht(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> face_th(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> face_tt(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> dofs_here(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> dofs_there(dofs_per_cell);

    // Parameters
    const double tau = dt;
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff = (tau_M_val > 0.0) ? (1.0 / tau + 1.0 / tau_M_val) : 1.0 / tau;
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 0.0;
    const double old_coeff = 1.0 / tau;

    // Initialize
    system_matrix = 0;
    rhs_x = 0;
    rhs_y = 0;

    // ========================================================================
    // CELL LOOP - Only locally owned cells
    // ========================================================================
    auto cell_M = M_dof_handler_.begin_active();
    auto cell_U = U_dof_handler_.begin_active();
    auto cell_phi = phi_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_M != M_dof_handler_.end();
           ++cell_M, ++cell_U, ++cell_phi, ++cell_theta)
    {
        // PARALLEL: Only process locally owned cells
        if (!cell_M->is_locally_owned())
            continue;

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

        // ====================================================================
        // Cell integral: mass + volume convection + RHS
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_M.quadrature_point(q);

            // Velocity and divergence
            dealii::Tensor<1, dim> U;
            U[0] = Ux_vals[q];
            U[1] = Uy_vals[q];
            const double div_U = grad_Ux[q][0] + grad_Uy[q][1];

            // H = h_a + h_d (applied + demagnetizing from ∇φ)
            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params_, current_time);
            dealii::Tensor<1, dim> H;
            H[0] = h_a[0] + grad_phi[q][0];
            H[1] = h_a[1] + grad_phi[q][1];

            // Susceptibility
            const double chi_theta =
                susceptibility(theta_vals[q],
                               params_.physics.epsilon,
                               params_.physics.chi_0);

            /* BETA TERM, FOR FUTURE
            // ================================================================
            // EXTENSION: β M×(M×H) Landau-Lifshitz damping term
            // Reference: Zhang-He-Yang SIAM J. Sci. Comput. 2021, Eq. 2.8
            //
            // In 2D: M×H = Mx*Hy - My*Hx (z-component, scalar)
            //        M×(M×H) = (M×H)×M = (Mx*Hy - My*Hx) * (-My, Mx)
            //
            // This term goes on the RHS (explicit treatment using M_old)
            // ================================================================
            dealii::Tensor<1, dim> beta_term;
            beta_term[0] = 0.0;
            beta_term[1] = 0.0;

            if (params_.physics.enable_beta_term && params_.physics.beta != 0.0)
            {
                const double Mx = Mx_old_vals[q];
                const double My = My_old_vals[q];
                const double Hx = H[0];
                const double Hy = H[1];

                // M×H (z-component)
                const double MxH_z = Mx * Hy - My * Hx;

                // M×(M×H) = (MxH_z) × M = MxH_z * (-My, Mx)
                // Note: This is the term that gets ADDED to RHS
                // The equation is: ∂M/∂t + ... = ... + β M×(M×H)
                // So we add +β * M×(M×H) to the RHS
                beta_term[0] = params_.physics.beta * (-MxH_z * My);
                beta_term[1] = params_.physics.beta * ( MxH_z * Mx);
            } */

            // MMS source term
            dealii::Tensor<1, dim> F_mms;
            F_mms = 0;
            if (mms_mode)
            {
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

                    // Mass: (1/τ + 1/τ_M)(φ_j, φ_i)
                    double mass = mass_coeff * phi_j * phi_i;

                    // Transport cell: -B_h^m cell term
                    double transport = -skew_magnetic_cell_value_scalar<dim>(
                        U, div_U, phi_i, grad_phi_i, phi_j);

                    local_matrix(i, j) += (mass + transport) * JxW;
                }

                // RHS: (1/τ_M)(χ H, φ_i) + (1/τ)(M^old, φ_i) + β M×(M×H) + (F_mms, φ_i)
                local_rhs_x(i) += (relax_coeff * chi_theta * H[0]
                    + old_coeff * Mx_old_vals[q]
                    + F_mms[0]) * phi_i * JxW;
                local_rhs_y(i) += (relax_coeff * chi_theta * H[1]
                    + old_coeff * My_old_vals[q]
                    + F_mms[1]) * phi_i * JxW;
            }
        }

        // Distribute cell contributions
        cell_M->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            rhs_x[local_dofs[i]] += local_rhs_x(i);
            rhs_y[local_dofs[i]] += local_rhs_y(i);
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dofs[i], local_dofs[j], local_matrix(i, j));
        }

        // ====================================================================
        // FACE CONTRIBUTIONS (interior faces only)
        // For STANDALONE MMS with U=0, these vanish since U·n=0
        // ====================================================================
        for (unsigned int f = 0; f < cell_M->n_faces(); ++f)
        {
            if (cell_M->at_boundary(f))
                continue;

            const auto neighbor_M = cell_M->neighbor(f);

            // PARALLEL: Skip if neighbor is not locally owned AND we're the higher index
            // (the other rank will handle it)
            if (!neighbor_M->is_locally_owned() && cell_M->index() > neighbor_M->index())
                continue;

            // AMR Case 1: Neighbor is coarser - handle from fine side
            if (cell_M->neighbor_is_coarser(f))
            {
                const std::pair<unsigned int, unsigned int> neighbor_info =
                    cell_M->neighbor_of_coarser_neighbor(f);
                const unsigned int neighbor_face = neighbor_info.first;
                const unsigned int neighbor_subface = neighbor_info.second;

                fe_interface_M.reinit(cell_M, f, dealii::numbers::invalid_unsigned_int,
                                      neighbor_M, neighbor_face, neighbor_subface);

                // OPTIMIZATION: Reinit pre-allocated FEFaceValues
                fe_face_U.reinit(cell_U, f);
                fe_face_U.get_function_values(Ux, Ux_face);
                fe_face_U.get_function_values(Uy, Uy_face);

                // DoF indices
                cell_M->get_dof_indices(dofs_here);
                neighbor_M->get_dof_indices(dofs_there);

                // OPTIMIZATION: Reset pre-allocated face matrices
                face_hh = 0;
                face_ht = 0;
                face_th = 0;
                face_tt = 0;

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    const double JxW = fe_interface_M.JxW(q);
                    const dealii::Tensor<1, dim>& normal = fe_interface_M.normal(q);
                    const double U_dot_n = Ux_face[q] * normal[0] + Uy_face[q] * normal[1];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double phi_i_here = fe_interface_M.shape_value(true, i, q);
                        const double phi_i_there = fe_interface_M.shape_value(false, i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double phi_j_here = fe_interface_M.shape_value(true, j, q);
                            const double phi_j_there = fe_interface_M.shape_value(false, j, q);

                            face_hh(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, phi_i_here, 0.0, phi_j_here, 0.0) * JxW;
                            face_ht(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, phi_i_here, 0.0, 0.0, phi_j_there) * JxW;
                            face_th(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, 0.0, phi_i_there, phi_j_here, 0.0) * JxW;
                            face_tt(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, 0.0, phi_i_there, 0.0, phi_j_there) * JxW;
                        }
                    }
                }

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
            // AMR Case 2: Neighbor at same level
            else if (neighbor_M->is_active())
            {
                // Process each face once
                if (cell_M->index() > neighbor_M->index())
                    continue;

                const unsigned int nf = cell_M->neighbor_of_neighbor(f);

                fe_interface_M.reinit(cell_M, f, dealii::numbers::invalid_unsigned_int,
                                      neighbor_M, nf, dealii::numbers::invalid_unsigned_int);

                // OPTIMIZATION: Reinit pre-allocated FEFaceValues
                fe_face_U.reinit(cell_U, f);
                fe_face_U.get_function_values(Ux, Ux_face);
                fe_face_U.get_function_values(Uy, Uy_face);

                // DoF indices
                cell_M->get_dof_indices(dofs_here);
                neighbor_M->get_dof_indices(dofs_there);

                // OPTIMIZATION: Reset pre-allocated face matrices
                face_hh = 0;
                face_ht = 0;
                face_th = 0;
                face_tt = 0;

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    const double JxW = fe_interface_M.JxW(q);
                    const dealii::Tensor<1, dim>& normal = fe_interface_M.normal(q);
                    const double U_dot_n = Ux_face[q] * normal[0] + Uy_face[q] * normal[1];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double phi_i_here = fe_interface_M.shape_value(true, i, q);
                        const double phi_i_there = fe_interface_M.shape_value(false, i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double phi_j_here = fe_interface_M.shape_value(true, j, q);
                            const double phi_j_there = fe_interface_M.shape_value(false, j, q);

                            face_hh(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, phi_i_here, 0.0, phi_j_here, 0.0) * JxW;
                            face_ht(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, phi_i_here, 0.0, 0.0, phi_j_there) * JxW;
                            face_th(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, 0.0, phi_i_there, phi_j_here, 0.0) * JxW;
                            face_tt(i, j) += -skew_magnetic_face_value_scalar_interface(
                                U_dot_n, 0.0, phi_i_there, 0.0, phi_j_there) * JxW;
                        }
                    }
                }

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
            // AMR Case 3: Neighbor is finer - skip, handled by fine cells
        }
    }

    // PARALLEL: Synchronize contributions
    system_matrix.compress(dealii::VectorOperation::add);
    rhs_x.compress(dealii::VectorOperation::add);
    rhs_y.compress(dealii::VectorOperation::add);
}

// ============================================================================
// RHS-only assembly (PARALLEL) - for Picard iterations where matrix is fixed
//
// FIX: Now uses H = h_a + h_d (was missing h_a)
// NOTE: Added current_time parameter for h_a computation
// ============================================================================
template <int dim>
void MagnetizationAssembler<dim>::assemble_rhs_only(
    dealii::TrilinosWrappers::MPI::Vector& rhs_x,
    dealii::TrilinosWrappers::MPI::Vector& rhs_y,
    const dealii::TrilinosWrappers::MPI::Vector& phi,
    const dealii::TrilinosWrappers::MPI::Vector& theta,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_old,
    const dealii::TrilinosWrappers::MPI::Vector& My_old,
    double dt,
    double current_time) const // NEW: added current_time parameter
{
    const dealii::FiniteElement<dim>& fe_M = M_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_phi = phi_dof_handler_.get_fe();
    const dealii::FiniteElement<dim>& fe_theta = theta_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_M.dofs_per_cell;

    dealii::QGauss<dim> quadrature_cell(fe_M.degree + 2);
    const unsigned int n_q_cell = quadrature_cell.size();

    // FIX: Added update_quadrature_points for h_a computation
    dealii::FEValues<dim> fe_values_M(fe_M, quadrature_cell,
                                      dealii::update_values | dealii::update_quadrature_points |
                                      dealii::update_JxW_values);
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
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 0.0;
    const double old_coeff = 1.0 / tau;

    rhs_x = 0;
    rhs_y = 0;

    auto cell_M = M_dof_handler_.begin_active();
    auto cell_phi = phi_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_M != M_dof_handler_.end(); ++cell_M, ++cell_phi, ++cell_theta)
    {
        if (!cell_M->is_locally_owned())
            continue;

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
            const dealii::Point<dim>& x_q = fe_values_M.quadrature_point(q);

            // FIX: Compute total field H = h_a + h_d
            // h_a = applied field from dipoles
            // h_d = ∇φ (demagnetizing field from Poisson solve)
            dealii::Tensor<1, dim> h_a = compute_applied_field<dim>(x_q, params_, current_time);
            dealii::Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
            {
                // Dome mode: H = h_a only
                H = h_a;
            }
            else
            {
                // Full physics: H = h_a + h_d
                H[0] = h_a[0] + grad_phi[q][0];
                H[1] = h_a[1] + grad_phi[q][1];
            }

            const double chi_theta =
                susceptibility(theta_vals[q],
                               params_.physics.epsilon,
                               params_.physics.chi_0);
            /* // EXTENSION: β M×(M×H) Landau-Lifshitz damping term
             dealii::Tensor<1, dim> beta_term;
             beta_term[0] = 0.0;
             beta_term[1] = 0.0;

             if (params_.physics.enable_beta_term && params_.physics.beta != 0.0)
             {
                 const double Mx = Mx_old_vals[q];
                 const double My = My_old_vals[q];
                 const double Hx = H[0];
                 const double Hy = H[1];

                 const double MxH_z = Mx * Hy - My * Hx;
                 beta_term[0] = params_.physics.beta * (-MxH_z * My);
                 beta_term[1] = params_.physics.beta * ( MxH_z * Mx);
             }
             */

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                local_rhs_x(i) += (relax_coeff * chi_theta * H[0]
                    + old_coeff * Mx_old_vals[q]) * phi_i * JxW;
                local_rhs_y(i) += (relax_coeff * chi_theta * H[1]
                    + old_coeff * My_old_vals[q]) * phi_i * JxW;
            }
        }

        cell_M->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            rhs_x[local_dofs[i]] += local_rhs_x(i);
            rhs_y[local_dofs[i]] += local_rhs_y(i);
        }
    }

    rhs_x.compress(dealii::VectorOperation::add);
    rhs_y.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagnetizationAssembler<2>;
