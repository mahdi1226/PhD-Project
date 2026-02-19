// ============================================================================
// magnetization/magnetization_assemble.cc — DG Cell + Face Assembly
//
// Private methods:
//   assemble_system_internal()  — unified matrix+RHS or RHS-only
//   initialize_preconditioner() — ILU from current matrix
//
// CELL INTEGRALS (Eq. 56-57, first line):
//   LHS:  (1/τ + 1/τ_M)(M^k, Z) + (U·∇M^k)·Z + ½(∇·U)(M^k·Z)
//   RHS:  (1/τ_M)(χ(θ)H^k, Z) + (1/τ)(M^{n-1}, Z)
//         + β[M(M·H) - H|M|²]·Z    (Landau-Lifshitz, explicit)
//         + (F_mms, Z)               (MMS source, testing only)
//
// FACE INTEGRALS (Eq. 57, second line):
//   LHS:  -Σ_F ∫_F (U·n⁻)[[Z]]·{M^k} dS  (upwind flux)
//
// β-TERM (Zhang-He-Yang 2021, 2D identity):
//   β M×(M×H) = β[M(M·H) - H|M|²]
//   Explicit treatment using M^{n-1}, H^k on RHS.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) B167-B193
// ============================================================================

#include "magnetization/magnetization.h"
#include "physics/skew_forms.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/lac/full_matrix.h>

using namespace dealii;

// ============================================================================
// assemble_system_internal() — unified assembly
//
// matrix_and_rhs = true:  build matrix + RHS (call once per timestep)
// matrix_and_rhs = false: build RHS only (Picard iteration, matrix frozen)
//
// When RHS-only:
//   - Skip matrix zeroing and all matrix writes
//   - Skip face loop (face terms are matrix-only)
//   - Skip U evaluation (U unchanged within Picard, already in matrix)
//   - Skip MMS source (MMS tests don't use Picard sub-iterations)
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector& Mx_old_relevant,
    const TrilinosWrappers::MPI::Vector& My_old_relevant,
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    const TrilinosWrappers::MPI::Vector& ux_relevant,
    const TrilinosWrappers::MPI::Vector& uy_relevant,
    const DoFHandler<dim>&               u_dof_handler,
    double dt,
    double current_time,
    bool matrix_and_rhs)
{
    Timer timer;
    timer.start();

    const double t_old = current_time - dt;

    const FiniteElement<dim>& fe_M     = dof_handler_.get_fe();
    const FiniteElement<dim>& fe_phi   = phi_dof_handler.get_fe();
    const FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();

    // Quadrature
    QGauss<dim>     quadrature_cell(fe_M.degree + 2);
    QGauss<dim - 1> quadrature_face(fe_M.degree + 2);

    const unsigned int n_q_cell = quadrature_cell.size();
    const unsigned int n_q_face = quadrature_face.size();

    // ========================================================================
    // FEValues — cell integrals
    // ========================================================================
    FEValues<dim> fe_values_M(fe_M, quadrature_cell,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature_cell,
        update_gradients | update_quadrature_points);
    FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
        update_values);

    // Pre-allocated scratch for cell field values
    std::vector<Tensor<1, dim>> grad_phi_vals(n_q_cell);
    std::vector<double>         theta_vals(n_q_cell);
    std::vector<double>         Mx_old_vals(n_q_cell);
    std::vector<double>         My_old_vals(n_q_cell);

    // Velocity FEValues + scratch (only needed for full assembly)
    std::unique_ptr<FEValues<dim>> fe_values_U_ptr;
    std::vector<double>            Ux_vals(n_q_cell);
    std::vector<double>            Uy_vals(n_q_cell);
    std::vector<Tensor<1, dim>>    grad_Ux(n_q_cell);
    std::vector<Tensor<1, dim>>    grad_Uy(n_q_cell);

    if (matrix_and_rhs)
    {
        const FiniteElement<dim>& fe_U = u_dof_handler.get_fe();
        fe_values_U_ptr = std::make_unique<FEValues<dim>>(
            fe_U, quadrature_cell,
            update_values | update_gradients);
    }

    // ========================================================================
    // FEInterfaceValues + face scratch (only for full assembly)
    // ========================================================================
    std::unique_ptr<FEInterfaceValues<dim>> fe_interface_M_ptr;
    std::unique_ptr<FEFaceValues<dim>>      fe_face_U_ptr;
    std::vector<double>                     Ux_face(n_q_face);
    std::vector<double>                     Uy_face(n_q_face);

    FullMatrix<double> face_hh(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> face_ht(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> face_th(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> face_tt(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> dofs_here(dofs_per_cell);
    std::vector<types::global_dof_index> dofs_there(dofs_per_cell);

    if (matrix_and_rhs)
    {
        fe_interface_M_ptr = std::make_unique<FEInterfaceValues<dim>>(
            fe_M, quadrature_face,
            update_values | update_JxW_values | update_normal_vectors);

        const FiniteElement<dim>& fe_U = u_dof_handler.get_fe();
        fe_face_U_ptr = std::make_unique<FEFaceValues<dim>>(
            fe_U, quadrature_face, update_values);
    }

    // ========================================================================
    // Local contributions
    // ========================================================================
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs_x(dofs_per_cell);
    Vector<double>     local_rhs_y(dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

    // ========================================================================
    // Coefficients
    // ========================================================================
    const double tau       = dt;
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff  = (tau_M_val > 0.0)
        ? (1.0 / tau + 1.0 / tau_M_val) : 1.0 / tau;
    const double relax_coeff = (tau_M_val > 0.0)
        ? 1.0 / tau_M_val : 0.0;
    const double old_coeff   = 1.0 / tau;

    const bool beta_active = params_.physics.enable_beta_term
                          && (std::abs(params_.physics.beta) > 1e-14);
    const double beta_val  = params_.physics.beta;

const bool mms_active = static_cast<bool>(mms_source_);

    // ========================================================================
    // Zero outputs
    // ========================================================================
    if (matrix_and_rhs)
        system_matrix_ = 0;
    Mx_rhs_ = 0;
    My_rhs_ = 0;

    // ========================================================================
    // CELL LOOP — synchronized iteration across DoFHandlers
    //
    // All iterators advance in lockstep.  cell_U always advances but is
    // only dereferenced when matrix_and_rhs == true.  In RHS-only mode
    // u_dof_handler is a dummy (phi_dof_handler) sharing the same
    // triangulation, so the iteration count matches.
    // ========================================================================
    auto cell_M     = dof_handler_.begin_active();
    auto cell_phi   = phi_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();
    auto cell_U     = u_dof_handler.begin_active();

    for (; cell_M != dof_handler_.end();
         ++cell_M, ++cell_phi, ++cell_theta, ++cell_U)
    {
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        if (matrix_and_rhs)
        {
            fe_values_U_ptr->reinit(cell_U);
            fe_values_U_ptr->get_function_values(ux_relevant, Ux_vals);
            fe_values_U_ptr->get_function_values(uy_relevant, Uy_vals);
            fe_values_U_ptr->get_function_gradients(ux_relevant, grad_Ux);
            fe_values_U_ptr->get_function_gradients(uy_relevant, grad_Uy);
        }

        // Always-needed field values
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_vals);
        fe_values_theta.get_function_values(theta_relevant, theta_vals);
        fe_values_M.get_function_values(Mx_old_relevant, Mx_old_vals);
        fe_values_M.get_function_values(My_old_relevant, My_old_vals);

        if (matrix_and_rhs)
            local_matrix = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        // ====================================================================
        // Cell quadrature: mass + transport (LHS) + relaxation + β (RHS)
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);

            // H = h_a + ∇φ (total field)
            Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
            {
                H = h_a;
            }
            else
            {
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = h_a[d] + grad_phi_vals[q][d];
            }

            // χ(θ)
            const double chi_theta = susceptibility(
                theta_vals[q], params_.physics.epsilon, params_.physics.chi_0);

            // Velocity + divergence (only for matrix assembly)
            Tensor<1, dim> U_q;
            double div_U_q = 0.0;
            if (matrix_and_rhs)
            {
                U_q[0]  = Ux_vals[q];
                if constexpr (dim > 1)
                    U_q[1] = Uy_vals[q];
                div_U_q = grad_Ux[q][0];
                if constexpr (dim > 1)
                    div_U_q += grad_Uy[q][1];
            }

            // ================================================================
            // β-term: β M×(M×H) = β[M(M·H) - H|M|²]  (explicit, RHS)
            //
            // Uses M^{n-1} (old time), H^k (current Picard iterate)
            // Reference: Zhang-He-Yang 2021, Eq. 2.8
            // ================================================================
            Tensor<1, dim> beta_term;
            if (beta_active)
            {
                const double Mx = Mx_old_vals[q];
                const double My = (dim > 1) ? My_old_vals[q] : 0.0;
                const double MdotH = Mx * H[0]
                    + ((dim > 1) ? My * H[1] : 0.0);
                const double M_sq = Mx * Mx + My * My;

                // β[M(M·H) - H|M|²]
                beta_term[0] = beta_val * (Mx * MdotH - H[0] * M_sq);
                if constexpr (dim > 1)
                    beta_term[1] = beta_val * (My * MdotH - H[1] * M_sq);
            }

            // ================================================================
            // MMS source (full assembly only, when callback is set)
            // ================================================================
            Tensor<1, dim> F_mms;
            if (mms_active)
            {
                F_mms = mms_source_(x_q, current_time, t_old,
                                     tau_M_val, chi_theta,
                                     H, U_q, div_U_q);
            }

            // ================================================================
            // Assemble cell contributions
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double Z_i = fe_values_M.shape_value(i, q);
                const Tensor<1, dim>& grad_Z_i = fe_values_M.shape_grad(i, q);

                // -- Matrix: mass + DG cell transport --
                if (matrix_and_rhs)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const double M_j = fe_values_M.shape_value(j, q);

                        // Mass: (1/τ + 1/τ_M)(M_j, Z_i)
                        double val = mass_coeff * M_j * Z_i;

                        // Transport cell: -B_h^m cell term
                        // B_h^m(U, M, Z) = (U·∇M)Z + ½(∇·U)(MZ)
                        // The negative sign: equation has +B_h^m on LHS
                        val += -skew_magnetic_cell_value_scalar<dim>(
                            U_q, div_U_q, Z_i, grad_Z_i, M_j);

                        local_matrix(i, j) += val * JxW;
                    }
                }

                // -- RHS: relaxation + old-time + β-term + MMS --
                // (1/τ_M)(χH, Z) + (1/τ)(M^old, Z) + β[...] + F_mms
                double rhs_x_val = relax_coeff * chi_theta * H[0]
                    + old_coeff * Mx_old_vals[q];
                double rhs_y_val = (dim > 1)
                    ? relax_coeff * chi_theta * H[1]
                      + old_coeff * My_old_vals[q]
                    : 0.0;

                if (beta_active)
                {
                    rhs_x_val += beta_term[0];
                    if constexpr (dim > 1)
                        rhs_y_val += beta_term[1];
                }

                if (mms_active)
                {
                    rhs_x_val += F_mms[0];
                    if constexpr (dim > 1)
                        rhs_y_val += F_mms[1];
                }

                local_rhs_x(i) += rhs_x_val * Z_i * JxW;
                local_rhs_y(i) += rhs_y_val * Z_i * JxW;
            }
        }

        // ====================================================================
        // Distribute cell contributions to global vectors/matrix
        // ====================================================================
        cell_M->get_dof_indices(local_dofs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_rhs_[local_dofs[i]] += local_rhs_x(i);
            My_rhs_[local_dofs[i]] += local_rhs_y(i);

            if (matrix_and_rhs)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    system_matrix_.add(local_dofs[i], local_dofs[j],
                                       local_matrix(i, j));
            }
        }

        // ====================================================================
        // FACE CONTRIBUTIONS (interior faces only, matrix_and_rhs only)
        //
        // Eq. 57 second line: -Σ_F ∫_F (U·n⁻)[[Z]]·{M} dS
        //
        // Face terms are matrix-only (bilinear in M^k, Z).
        // When RHS-only, the matrix is frozen → skip entirely.
        // ====================================================================
        if (!matrix_and_rhs)
            continue;  // skip face loop for RHS-only mode

        for (unsigned int f = 0; f < cell_M->n_faces(); ++f)
        {
            if (cell_M->at_boundary(f))
                continue;

            const auto neighbor_M = cell_M->neighbor(f);

            // Parallel: skip if neighbor not locally owned and we're higher index
            if (!neighbor_M->is_locally_owned()
                && cell_M->index() > neighbor_M->index())
                continue;

            // ================================================================
            // AMR Case 1: Neighbor is coarser — handle from fine side
            // ================================================================
            if (cell_M->neighbor_is_coarser(f))
            {
                const auto neighbor_info =
                    cell_M->neighbor_of_coarser_neighbor(f);
                const unsigned int neighbor_face    = neighbor_info.first;
                const unsigned int neighbor_subface = neighbor_info.second;

                fe_interface_M_ptr->reinit(
                    cell_M, f, numbers::invalid_unsigned_int,
                    neighbor_M, neighbor_face, neighbor_subface);

                fe_face_U_ptr->reinit(cell_U, f);
                fe_face_U_ptr->get_function_values(ux_relevant, Ux_face);
                fe_face_U_ptr->get_function_values(uy_relevant, Uy_face);

                cell_M->get_dof_indices(dofs_here);
                neighbor_M->get_dof_indices(dofs_there);

                face_hh = 0; face_ht = 0; face_th = 0; face_tt = 0;

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    const double JxW = fe_interface_M_ptr->JxW(q);
                    const Tensor<1, dim>& normal =
                        fe_interface_M_ptr->normal(q);
                    double U_dot_n = Ux_face[q] * normal[0];
                    if constexpr (dim > 1)
                        U_dot_n += Uy_face[q] * normal[1];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double Z_i_here =
                            fe_interface_M_ptr->shape_value(true, i, q);
                        const double Z_i_there =
                            fe_interface_M_ptr->shape_value(false, i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double M_j_here =
                                fe_interface_M_ptr->shape_value(true, j, q);
                            const double M_j_there =
                                fe_interface_M_ptr->shape_value(false, j, q);

                            face_hh(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, Z_i_here, 0.0,
                                    M_j_here, 0.0) * JxW;
                            face_ht(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, Z_i_here, 0.0,
                                    0.0, M_j_there) * JxW;
                            face_th(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, 0.0, Z_i_there,
                                    M_j_here, 0.0) * JxW;
                            face_tt(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, 0.0, Z_i_there,
                                    0.0, M_j_there) * JxW;
                        }
                    }
                }

                // Distribute 4-block face contributions
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        system_matrix_.add(dofs_here[i],  dofs_here[j],
                                           face_hh(i, j));
                        system_matrix_.add(dofs_here[i],  dofs_there[j],
                                           face_ht(i, j));
                        system_matrix_.add(dofs_there[i], dofs_here[j],
                                           face_th(i, j));
                        system_matrix_.add(dofs_there[i], dofs_there[j],
                                           face_tt(i, j));
                    }
            }
            // ================================================================
            // AMR Case 2: Neighbor at same level
            // ================================================================
            else if (neighbor_M->is_active())
            {
                // Process each face once: lower index handles it
                if (cell_M->index() > neighbor_M->index())
                    continue;

                const unsigned int nf = cell_M->neighbor_of_neighbor(f);

                fe_interface_M_ptr->reinit(
                    cell_M, f, numbers::invalid_unsigned_int,
                    neighbor_M, nf, numbers::invalid_unsigned_int);

                fe_face_U_ptr->reinit(cell_U, f);
                fe_face_U_ptr->get_function_values(ux_relevant, Ux_face);
                fe_face_U_ptr->get_function_values(uy_relevant, Uy_face);

                cell_M->get_dof_indices(dofs_here);
                neighbor_M->get_dof_indices(dofs_there);

                face_hh = 0; face_ht = 0; face_th = 0; face_tt = 0;

                for (unsigned int q = 0; q < n_q_face; ++q)
                {
                    const double JxW = fe_interface_M_ptr->JxW(q);
                    const Tensor<1, dim>& normal =
                        fe_interface_M_ptr->normal(q);
                    double U_dot_n = Ux_face[q] * normal[0];
                    if constexpr (dim > 1)
                        U_dot_n += Uy_face[q] * normal[1];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const double Z_i_here =
                            fe_interface_M_ptr->shape_value(true, i, q);
                        const double Z_i_there =
                            fe_interface_M_ptr->shape_value(false, i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const double M_j_here =
                                fe_interface_M_ptr->shape_value(true, j, q);
                            const double M_j_there =
                                fe_interface_M_ptr->shape_value(false, j, q);

                            face_hh(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, Z_i_here, 0.0,
                                    M_j_here, 0.0) * JxW;
                            face_ht(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, Z_i_here, 0.0,
                                    0.0, M_j_there) * JxW;
                            face_th(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, 0.0, Z_i_there,
                                    M_j_here, 0.0) * JxW;
                            face_tt(i, j) +=
                                -skew_magnetic_face_value_scalar_interface(
                                    U_dot_n, 0.0, Z_i_there,
                                    0.0, M_j_there) * JxW;
                        }
                    }
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        system_matrix_.add(dofs_here[i],  dofs_here[j],
                                           face_hh(i, j));
                        system_matrix_.add(dofs_here[i],  dofs_there[j],
                                           face_ht(i, j));
                        system_matrix_.add(dofs_there[i], dofs_here[j],
                                           face_th(i, j));
                        system_matrix_.add(dofs_there[i], dofs_there[j],
                                           face_tt(i, j));
                    }
            }
            // AMR Case 3: Neighbor is finer — skip, handled by fine cells
        }
    }

    // ========================================================================
    // Compress (synchronize MPI contributions)
    // ========================================================================
    if (matrix_and_rhs)
        system_matrix_.compress(VectorOperation::add);
    Mx_rhs_.compress(VectorOperation::add);
    My_rhs_.compress(VectorOperation::add);

    timer.stop();

    pcout_ << "[Magnetization] Assembly ("
           << (matrix_and_rhs ? "matrix+RHS" : "RHS-only")
           << "): " << timer.wall_time() << " s" << std::endl;
}

// ============================================================================
// initialize_preconditioner() — ILU from current matrix
//
// Called once after full assembly. Reused for both Mx and My solves.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::initialize_preconditioner()
{
    TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
    ilu_data.ilu_fill = 1;      // ILU(1) for DG transport
    ilu_data.ilu_atol = 1e-12;
    ilu_data.ilu_rtol = 1.0;

    ilu_preconditioner_.initialize(system_matrix_, ilu_data);
    preconditioner_initialized_ = true;

    pcout_ << "[Magnetization] ILU preconditioner initialized" << std::endl;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void MagnetizationSubsystem<2>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    double, double, bool);

template void MagnetizationSubsystem<2>::initialize_preconditioner();

template void MagnetizationSubsystem<3>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    double, double, bool);

template void MagnetizationSubsystem<3>::initialize_preconditioner();