// ============================================================================
// diagnostics/ferrofluid_diagnostics.h - Comprehensive Diagnostics
//
// Tracks and reports all critical physics components for Rosensweig instability
// ============================================================================
#ifndef FERROFLUID_DIAGNOSTICS_H
#define FERROFLUID_DIAGNOSTICS_H

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

template <int dim>
class FerrofluidDiagnostics
{
private:
    std::ofstream log_file_;
    bool first_output_;
    MPI_Comm mpi_comm_;
    int rank_;

public:
    FerrofluidDiagnostics(const std::string& filename = "ferrofluid_diagnostics.csv")
        : first_output_(true), mpi_comm_(MPI_COMM_WORLD)
    {
        rank_ = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);
        if (rank_ == 0)
        {
            log_file_.open(filename);
            log_file_ << std::scientific << std::setprecision(6);
        }
    }

    ~FerrofluidDiagnostics()
    {
        if (log_file_.is_open())
            log_file_.close();
    }

    // Main diagnostic function
    void check_physics(
        const dealii::DoFHandler<dim>& theta_dof,
        const dealii::DoFHandler<dim>& phi_dof,
        const dealii::DoFHandler<dim>& M_dof,
        const dealii::DoFHandler<dim>& u_dof,
        const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
        const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
        const dealii::TrilinosWrappers::MPI::Vector& My_solution,
        const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
        const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
        double chi_0,
        double epsilon,
        double mu_0,
        double time)
    {
        dealii::QGauss<dim> quadrature(3);

        // Setup FEValues
        dealii::FEValues<dim> fe_theta(theta_dof.get_fe(), quadrature,
            dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
        dealii::FEValues<dim> fe_phi(phi_dof.get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients | dealii::update_hessians);
        dealii::FEValues<dim> fe_M(M_dof.get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
        dealii::FEValues<dim> fe_u(u_dof.get_fe(), quadrature,
            dealii::update_values);

        const unsigned int n_q = quadrature.size();

        // === DIAGNOSTIC ACCUMULATORS ===
        double max_theta = -1e10, min_theta = 1e10;
        double max_chi_linear = 0, min_chi_linear = 1e10;
        double max_chi_smooth = 0, min_chi_smooth = 1e10;
        double max_M_ferro = 0, max_M_air = 0;  // Max |M| in each phase
        double total_M_ferro = 0, total_M_air = 0;  // Integrated |M| in each phase
        double ferro_volume = 0, air_volume = 0;  // Volume of each phase
        double max_H_magnitude = 0;
        double max_kelvin_up = 0, max_kelvin_down = 0;  // Vertical Kelvin force
        double interface_y_min = 1e10, interface_y_max = -1e10;  // Interface deformation
        double max_uy = -1e10, min_uy = 1e10;  // Vertical velocity

        // Check θ sign convention (detect which function is used)
        double theta_test_values[] = {-1.0, 0.0, 1.0};
        double chi_at_minus1 = 0, chi_at_plus1 = 0;

        // Loop over cells
        auto cell_theta = theta_dof.begin_active();
        auto cell_phi = phi_dof.begin_active();
        auto cell_M = M_dof.begin_active();
        auto cell_u = u_dof.begin_active();

        for (; cell_theta != theta_dof.end(); ++cell_theta, ++cell_phi, ++cell_M, ++cell_u)
        {
            if (!cell_theta->is_locally_owned()) continue;

            fe_theta.reinit(cell_theta);
            fe_phi.reinit(cell_phi);
            fe_M.reinit(cell_M);
            fe_u.reinit(cell_u);

            // Get local solutions
            std::vector<double> theta_vals(n_q);
            std::vector<dealii::Tensor<1,dim>> H_vals(n_q);
            std::vector<dealii::Tensor<2,dim>> hess_phi_vals(n_q);
            std::vector<double> Mx_vals(n_q), My_vals(n_q);
            std::vector<dealii::Tensor<1,dim>> grad_Mx_vals(n_q), grad_My_vals(n_q);
            std::vector<double> ux_vals(n_q), uy_vals(n_q);

            fe_theta.get_function_values(theta_solution, theta_vals);
            fe_phi.get_function_gradients(phi_solution, H_vals);
            fe_phi.get_function_hessians(phi_solution, hess_phi_vals);
            fe_M.get_function_values(Mx_solution, Mx_vals);
            fe_M.get_function_values(My_solution, My_vals);
            fe_M.get_function_gradients(Mx_solution, grad_Mx_vals);
            fe_M.get_function_gradients(My_solution, grad_My_vals);
            fe_u.get_function_values(ux_solution, ux_vals);
            fe_u.get_function_values(uy_solution, uy_vals);

            for (unsigned int q = 0; q < n_q; ++q)
            {
                const double theta = theta_vals[q];
                const auto& H = H_vals[q];
                const auto& pt = fe_theta.quadrature_point(q);
                const double y = pt[1];

                // Track theta range
                max_theta = std::max(max_theta, theta);
                min_theta = std::min(min_theta, theta);

                // Test BOTH susceptibility functions
                const double chi_linear = chi_0 * 0.5 * (1.0 + theta);  // Correct for θ=+1 ferro
                const double chi_smooth = chi_0 / (1.0 + std::exp(-2.0 * theta / epsilon));

                max_chi_linear = std::max(max_chi_linear, chi_linear);
                min_chi_linear = std::min(min_chi_linear, chi_linear);
                max_chi_smooth = std::max(max_chi_smooth, chi_smooth);
                min_chi_smooth = std::min(min_chi_smooth, chi_smooth);

                // Test susceptibility at extreme values
                if (std::abs(theta - 1.0) < 0.1)  // Near θ=+1
                    chi_at_plus1 = chi_linear;
                if (std::abs(theta + 1.0) < 0.1)  // Near θ=-1
                    chi_at_minus1 = chi_linear;

                // Magnetization
                const double Mx = Mx_vals[q];
                const double My = My_vals[q];
                const double M_mag = std::sqrt(Mx*Mx + My*My);

                // H magnitude
                const double H_mag = H.norm();
                max_H_magnitude = std::max(max_H_magnitude, H_mag);

                // Phase-separated quantities
                const double JxW = fe_theta.JxW(q);
                if (theta > 0)  // ferrofluid
                {
                    max_M_ferro = std::max(max_M_ferro, M_mag);
                    total_M_ferro += M_mag * JxW;
                    ferro_volume += JxW;
                }
                else  // air
                {
                    max_M_air = std::max(max_M_air, M_mag);
                    total_M_air += M_mag * JxW;
                    air_volume += JxW;
                }

                // Interface location (θ ≈ 0)
                if (std::abs(theta) < 0.1)
                {
                    interface_y_min = std::min(interface_y_min, y);
                    interface_y_max = std::max(interface_y_max, y);
                }

                // Kelvin force (vertical component)
                dealii::Tensor<1,dim> M;
                M[0] = Mx;
                M[1] = My;

                // (M·∇)H
                dealii::Tensor<1,dim> M_grad_H;
                for (unsigned int i = 0; i < dim; ++i)
                {
                    M_grad_H[i] = 0;
                    for (unsigned int j = 0; j < dim; ++j)
                        M_grad_H[i] += M[j] * hess_phi_vals[q][i][j];
                }

                // div(M)
                const double div_M = grad_Mx_vals[q][0] + grad_My_vals[q][1];

                // Vertical Kelvin force: F_y = μ₀(M_grad_H[1] + 0.5*div_M*H[1])
                const double kelvin_y = mu_0 * (M_grad_H[1] + 0.5 * div_M * H[1]);

                if (kelvin_y > 0)
                    max_kelvin_up = std::max(max_kelvin_up, kelvin_y);
                else
                    max_kelvin_down = std::max(max_kelvin_down, -kelvin_y);

                // Velocity
                max_uy = std::max(max_uy, uy_vals[q]);
                min_uy = std::min(min_uy, uy_vals[q]);
            }
        }

        // MPI reductions
        max_theta = dealii::Utilities::MPI::max(max_theta, mpi_comm_);
        min_theta = dealii::Utilities::MPI::min(min_theta, mpi_comm_);
        max_chi_linear = dealii::Utilities::MPI::max(max_chi_linear, mpi_comm_);
        min_chi_linear = dealii::Utilities::MPI::min(min_chi_linear, mpi_comm_);
        max_chi_smooth = dealii::Utilities::MPI::max(max_chi_smooth, mpi_comm_);
        min_chi_smooth = dealii::Utilities::MPI::min(min_chi_smooth, mpi_comm_);
        max_M_ferro = dealii::Utilities::MPI::max(max_M_ferro, mpi_comm_);
        max_M_air = dealii::Utilities::MPI::max(max_M_air, mpi_comm_);
        total_M_ferro = dealii::Utilities::MPI::sum(total_M_ferro, mpi_comm_);
        total_M_air = dealii::Utilities::MPI::sum(total_M_air, mpi_comm_);
        ferro_volume = dealii::Utilities::MPI::sum(ferro_volume, mpi_comm_);
        air_volume = dealii::Utilities::MPI::sum(air_volume, mpi_comm_);
        max_H_magnitude = dealii::Utilities::MPI::max(max_H_magnitude, mpi_comm_);
        max_kelvin_up = dealii::Utilities::MPI::max(max_kelvin_up, mpi_comm_);
        max_kelvin_down = dealii::Utilities::MPI::max(max_kelvin_down, mpi_comm_);
        interface_y_min = dealii::Utilities::MPI::min(interface_y_min, mpi_comm_);
        interface_y_max = dealii::Utilities::MPI::max(interface_y_max, mpi_comm_);
        max_uy = dealii::Utilities::MPI::max(max_uy, mpi_comm_);
        min_uy = dealii::Utilities::MPI::min(min_uy, mpi_comm_);
        chi_at_plus1 = dealii::Utilities::MPI::max(chi_at_plus1, mpi_comm_);
        chi_at_minus1 = dealii::Utilities::MPI::max(chi_at_minus1, mpi_comm_);

        // Average magnetizations
        const double avg_M_ferro = (ferro_volume > 0) ? total_M_ferro / ferro_volume : 0;
        const double avg_M_air = (air_volume > 0) ? total_M_air / air_volume : 0;

        // Expected magnetization in ferrofluid
        const double expected_M_ferro = chi_0 * max_H_magnitude;

        // Interface deformation
        const double interface_amplitude = interface_y_max - interface_y_min;

        // === OUTPUT TO CONSOLE (rank 0 only) ===
        if (rank_ == 0)
        {
            std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║        FERROFLUID PHYSICS DIAGNOSTICS (t=" << std::setw(8) << time << ")       ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

            // Phase field
            std::cout << "║ PHASE FIELD:                                                 ║\n";
            std::cout << "║   θ range: [" << std::setw(10) << min_theta << ", " << std::setw(10) << max_theta << "]     ║\n";
            std::cout << "║   Interface amplitude: " << std::setw(10) << interface_amplitude << " m                  ║\n";

            // Susceptibility check
            std::cout << "║ SUSCEPTIBILITY CHECK:                                        ║\n";
            std::cout << "║   χ_linear range: [" << std::setw(10) << min_chi_linear << ", " << std::setw(10) << max_chi_linear << "]  ║\n";
            std::cout << "║   χ_smooth range: [" << std::setw(10) << min_chi_smooth << ", " << std::setw(10) << max_chi_smooth << "]  ║\n";
            std::cout << "║   χ(θ=+1): " << std::setw(10) << chi_at_plus1 << " (should be " << chi_0 << ")           ║\n";
            std::cout << "║   χ(θ=-1): " << std::setw(10) << chi_at_minus1 << " (should be 0)               ║\n";

            // Determine which convention is likely being used
            bool correct_convention = (chi_at_plus1 > 0.8 * chi_0) && (chi_at_minus1 < 0.2 * chi_0);
            if (!correct_convention)
            {
                std::cout << "║   ⚠️  WARNING: SUSCEPTIBILITY FUNCTION APPEARS WRONG!         ║\n";
            }

            // Magnetic field
            std::cout << "║ MAGNETIC FIELD:                                              ║\n";
            std::cout << "║   Max |H|: " << std::setw(10) << max_H_magnitude << " A/m                           ║\n";

            // Magnetization
            std::cout << "║ MAGNETIZATION:                                               ║\n";
            std::cout << "║   Ferrofluid: max=" << std::setw(10) << max_M_ferro << ", avg=" << std::setw(10) << avg_M_ferro << " ║\n";
            std::cout << "║   Air:        max=" << std::setw(10) << max_M_air << ", avg=" << std::setw(10) << avg_M_air << " ║\n";
            std::cout << "║   Expected M_ferro: " << std::setw(10) << expected_M_ferro << "                     ║\n";

            // Check magnetization health
            if (max_M_ferro < 0.1 * expected_M_ferro)
            {
                std::cout << "║   ⚠️  WARNING: M in ferrofluid is much too small!             ║\n";
            }
            if (max_M_air > 0.1 * max_M_ferro && max_M_ferro > 1e-10)
            {
                std::cout << "║   ⚠️  WARNING: Significant M in air phase!                    ║\n";
            }

            // Kelvin force
            std::cout << "║ KELVIN FORCE (vertical):                                     ║\n";
            std::cout << "║   Max upward:   " << std::setw(10) << max_kelvin_up << " N/m³                     ║\n";
            std::cout << "║   Max downward: " << std::setw(10) << max_kelvin_down << " N/m³                     ║\n";

            if (max_kelvin_up < 1e-10)
            {
                std::cout << "║   ⚠️  WARNING: No upward Kelvin force - instability blocked!  ║\n";
            }

            // Velocity
            std::cout << "║ VELOCITY:                                                    ║\n";
            std::cout << "║   Vertical (uy): [" << std::setw(10) << min_uy << ", " << std::setw(10) << max_uy << "]     ║\n";

            std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

            // Write to CSV file
            if (first_output_)
            {
                log_file_ << "time,min_theta,max_theta,interface_amp,max_H,max_M_ferro,avg_M_ferro,"
                         << "max_M_air,avg_M_air,max_kelvin_up,max_kelvin_down,max_uy,chi_at_plus1,chi_at_minus1\n";
                first_output_ = false;
            }

            log_file_ << time << ","
                      << min_theta << "," << max_theta << "," << interface_amplitude << ","
                      << max_H_magnitude << "," << max_M_ferro << "," << avg_M_ferro << ","
                      << max_M_air << "," << avg_M_air << ","
                      << max_kelvin_up << "," << max_kelvin_down << "," << max_uy << ","
                      << chi_at_plus1 << "," << chi_at_minus1 << "\n";
            log_file_.flush();
        }
    }
};

#endif // FERROFLUID_DIAGNOSTICS_H