// ============================================================================
// mms/magnetic/magnetic_mms.h - MMS for Monolithic Magnetics (PARALLEL)
//
// Reuses existing exact solutions from poisson_mms.h and magnetization_mms.h:
//   phi_exact = t * cos(pi*x) * cos(pi*y/L_y)
//   Mx_exact  = t * sin(pi*x) * sin(pi*y/L_y)
//   My_exact  = t * cos(pi*x) * sin(pi*y/L_y)
//
// Provides:
//   - MagneticExactSolution: combined Function<dim> with dim+1 components
//     for VectorTools::interpolate on the FESystem
//   - compute_magnetic_mms_errors_parallel: error computation for both M and phi
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_MMS_H
#define MAGNETIC_MMS_H

#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>
#include <cmath>

// ============================================================================
// Combined exact solution: (Mx, My, phi) for the FESystem
//
// Component 0: Mx = t * sin(pi*x) * sin(pi*y/L_y)
// Component 1: My = t * cos(pi*x) * sin(pi*y/L_y)
// Component 2: phi = t * cos(pi*x) * cos(pi*y/L_y)
// ============================================================================
template <int dim>
class MagneticExactSolution : public dealii::Function<dim>
{
public:
    MagneticExactSolution(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(dim + 1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component) const override
    {
        const double x = p[0];
        const double y = p[1];

        if (component == 0) // Mx
            return time_ * std::sin(M_PI * x) * std::sin(M_PI * y / L_y_);
        else if (component == 1) // My
            return time_ * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        else // phi
            return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual void vector_value(const dealii::Point<dim>& p,
                              dealii::Vector<double>& values) const override
    {
        for (unsigned int c = 0; c < this->n_components; ++c)
            values(c) = value(p, c);
    }

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Error results for the combined system
// ============================================================================
struct MagneticMMSError
{
    // Magnetization errors
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;
    double M_Linf = 0.0;
    double M_H1 = 0.0;    // H1 seminorm of M (||∇(M-M*)||)

    // Poisson errors
    double phi_L2 = 0.0;
    double phi_Linf = 0.0;
    double phi_H1 = 0.0;
};

// ============================================================================
// Compute errors for the monolithic magnetics system (PARALLEL)
//
// Uses FEValuesExtractors to evaluate M and phi components from the
// combined solution vector.
//
// For phi (Neumann BC): mean-shift correction before L2 error.
// ============================================================================
template <int dim>
MagneticMMSError compute_magnetic_mms_errors_parallel(
    const dealii::DoFHandler<dim>& mag_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& mag_solution,
    double time,
    double L_y,
    MPI_Comm mpi_communicator)
{
    MagneticMMSError errors;

    const auto& fe = mag_dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    const dealii::FEValuesExtractors::Vector M_ext(0);
    const dealii::FEValuesExtractors::Scalar phi_ext(dim);

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<dealii::Tensor<1, dim>> M_values(n_q_points);
    std::vector<dealii::Tensor<2, dim>> M_gradients(n_q_points);
    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    MagExactMx<dim> exact_Mx(time, L_y);
    MagExactMy<dim> exact_My(time, L_y);
    PoissonExactSolution<dim> exact_phi(time, L_y);

    // Pass 1: M errors (L2, H1, Linf) + phi mean shift + phi H1/Linf
    double local_Mx_L2_sq = 0.0;
    double local_My_L2_sq = 0.0;
    double local_M_H1_sq = 0.0;
    double local_M_Linf = 0.0;
    double local_phi_H1_sq = 0.0;
    double local_phi_Linf = 0.0;
    double local_phi_mean_diff = 0.0;
    double local_volume = 0.0;

    for (const auto& cell : mag_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values[M_ext].get_function_values(mag_solution, M_values);
        fe_values[M_ext].get_function_gradients(mag_solution, M_gradients);
        fe_values[phi_ext].get_function_values(mag_solution, phi_values);
        fe_values[phi_ext].get_function_gradients(mag_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // M errors (L2 + Linf)
            const double Mx_err = M_values[q][0] - exact_Mx.value(x_q);
            const double My_err = M_values[q][1] - exact_My.value(x_q);
            local_Mx_L2_sq += Mx_err * Mx_err * JxW;
            local_My_L2_sq += My_err * My_err * JxW;
            local_M_Linf = std::max(local_M_Linf,
                std::max(std::abs(Mx_err), std::abs(My_err)));

            // M H1 seminorm: ||∇(M - M*)||
            const auto grad_Mx_exact = exact_Mx.gradient(x_q);
            const auto grad_My_exact = exact_My.gradient(x_q);
            for (unsigned int d = 0; d < dim; ++d)
            {
                const double grad_Mx_err = M_gradients[q][0][d] - grad_Mx_exact[d];
                const double grad_My_err = M_gradients[q][1][d] - grad_My_exact[d];
                local_M_H1_sq += (grad_Mx_err * grad_Mx_err + grad_My_err * grad_My_err) * JxW;
            }

            // phi mean shift
            const double phi_diff = phi_values[q] - exact_phi.value(x_q);
            local_phi_mean_diff += phi_diff * JxW;
            local_volume += JxW;

            // phi H1 seminorm
            const auto grad_exact = exact_phi.gradient(x_q);
            const auto grad_error = phi_gradients[q] - grad_exact;
            local_phi_H1_sq += (grad_error * grad_error) * JxW;

            // phi Linf (pre-mean-shift, corrected in pass 2)
            local_phi_Linf = std::max(local_phi_Linf, std::abs(phi_diff));
        }
    }

    // Global reductions
    double global_Mx_L2_sq = 0.0, global_My_L2_sq = 0.0;
    double global_M_H1_sq = 0.0, global_M_Linf = 0.0;
    double global_phi_H1_sq = 0.0;
    double global_phi_mean_diff = 0.0, global_volume = 0.0;

    MPI_Allreduce(&local_Mx_L2_sq, &global_Mx_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_My_L2_sq, &global_My_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_M_H1_sq, &global_M_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_M_Linf, &global_M_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
    MPI_Allreduce(&local_phi_H1_sq, &global_phi_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_phi_mean_diff, &global_phi_mean_diff, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    errors.Mx_L2 = std::sqrt(global_Mx_L2_sq);
    errors.My_L2 = std::sqrt(global_My_L2_sq);
    errors.M_L2 = std::sqrt(global_Mx_L2_sq + global_My_L2_sq);
    errors.M_H1 = std::sqrt(global_M_H1_sq);
    errors.M_Linf = global_M_Linf;
    errors.phi_H1 = std::sqrt(global_phi_H1_sq);

    // Pass 2: phi L2 and Linf with mean shift
    const double c_shift = global_phi_mean_diff / global_volume;
    double local_phi_L2_sq = 0.0;
    local_phi_Linf = 0.0;

    for (const auto& cell : mag_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values[phi_ext].get_function_values(mag_solution, phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_err = (phi_values[q] - c_shift) - exact_phi.value(x_q);
            local_phi_L2_sq += phi_err * phi_err * JxW;
            local_phi_Linf = std::max(local_phi_Linf, std::abs(phi_err));
        }
    }

    double global_phi_L2_sq = 0.0;
    double global_phi_Linf = 0.0;
    MPI_Allreduce(&local_phi_L2_sq, &global_phi_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_phi_Linf, &global_phi_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);
    errors.phi_L2 = std::sqrt(global_phi_L2_sq);
    errors.phi_Linf = global_phi_Linf;

    return errors;
}

#endif // MAGNETIC_MMS_H
