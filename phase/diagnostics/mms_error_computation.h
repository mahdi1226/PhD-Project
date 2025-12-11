// ============================================================================
// utilities/mms_error_computation.h - MMS error computation for NS-CH
// ============================================================================
#ifndef MMS_ERROR_COMPUTATION_H
#define MMS_ERROR_COMPUTATION_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/block_vector.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../utilities/nsch_mms.h"

// ============================================================================
// Structure to hold all MMS errors
// ============================================================================
struct MMSErrors
{
    double u_L2  = 0.0;  // ||u - u_exact||_L2
    double u_H1  = 0.0;  // |u - u_exact|_H1 (seminorm)
    double p_L2  = 0.0;  // ||p - p_exact||_L2 (with mean subtraction)
    double c_L2  = 0.0;  // ||c - c_exact||_L2
    double c_H1  = 0.0;  // |c - c_exact|_H1 (seminorm)
    double mu_L2 = 0.0;  // ||μ - μ_exact||_L2
    
    void print() const
    {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              MMS ERROR ANALYSIS (at final time)              ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Navier-Stokes:                                              ║\n";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "║    ||u - u_exact||_L2     =   " << u_L2 << "                 ║\n";
        std::cout << "║    |u - u_exact|_H1       =   " << u_H1 << "                 ║\n";
        std::cout << "║    ||p - p_exact||_L2     =   " << p_L2 << "   (mean-sub) ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  Cahn-Hilliard:                                              ║\n";
        std::cout << "║    ||c - c_exact||_L2     =   " << c_L2 << "                 ║\n";
        std::cout << "║    |c - c_exact|_H1       =   " << c_H1 << "                 ║\n";
        std::cout << "║    ||μ - μ_exact||_L2     =   " << mu_L2 << "                 ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }
    
    void print_for_convergence(double h) const
    {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << h << "\t" << u_L2 << "\t" << u_H1 << "\t" << p_L2 
                  << "\t" << c_L2 << "\t" << c_H1 << "\t" << mu_L2 << "\n";
    }
};

// ============================================================================
// Compute MMS errors for NS-CH system
// ============================================================================
template <int dim>
MMSErrors compute_mms_errors(
    const dealii::DoFHandler<dim>& ns_dof_handler,
    const dealii::DoFHandler<dim>& ch_dof_handler,
    const dealii::BlockVector<double>& ns_solution,
    const dealii::BlockVector<double>& ch_solution,
    double current_time)
{
    MMSErrors errors;
    
    const unsigned int fe_degree = ns_dof_handler.get_fe().degree;
    dealii::QGauss<dim> quadrature(fe_degree + 2);
    
    // ========================================================================
    // Navier-Stokes errors (velocity and pressure)
    // ========================================================================
    {
        const auto& fe = ns_dof_handler.get_fe();
        dealii::FEValues<dim> fe_values(fe, quadrature,
            dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
        
        const dealii::FEValuesExtractors::Vector velocity(0);
        const dealii::FEValuesExtractors::Scalar pressure(dim);
        
        const unsigned int n_q_points = quadrature.size();
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        // Convert block vector to full vector for FEValues
        dealii::Vector<double> ns_full(ns_solution.block(0).size() + ns_solution.block(1).size());
        for (unsigned int i = 0; i < ns_solution.block(0).size(); ++i)
            ns_full[i] = ns_solution.block(0)[i];
        for (unsigned int i = 0; i < ns_solution.block(1).size(); ++i)
            ns_full[ns_solution.block(0).size() + i] = ns_solution.block(1)[i];
        
        // Exact solutions
        MMSExactVelocity<dim> exact_u;
        MMSExactPressure<dim> exact_p;
        exact_u.set_time(current_time);
        exact_p.set_time(current_time);
        
        double u_L2_sq = 0.0, u_H1_sq = 0.0;
        double p_h_integral = 0.0, p_exact_integral = 0.0;
        double p_L2_sq_raw = 0.0;
        double domain_volume = 0.0;
        
        // First pass: compute means and raw errors
        for (const auto& cell : ns_dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            
            // Get numerical values at quadrature points
            std::vector<dealii::Tensor<1, dim>> u_h_values(n_q_points);
            std::vector<dealii::Tensor<2, dim>> u_h_gradients(n_q_points);
            std::vector<double> p_h_values(n_q_points);
            
            fe_values[velocity].get_function_values(ns_full, u_h_values);
            fe_values[velocity].get_function_gradients(ns_full, u_h_gradients);
            fe_values[pressure].get_function_values(ns_full, p_h_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const auto& x_q = fe_values.quadrature_point(q);
                const double JxW = fe_values.JxW(q);
                
                // Exact values
                dealii::Tensor<1, dim> u_exact;
                for (unsigned int d = 0; d < dim; ++d)
                    u_exact[d] = exact_u.value(x_q, d);
                
                dealii::Tensor<2, dim> grad_u_exact;
                // Compute exact gradient analytically
                const double pi = dealii::numbers::PI;
                const double t4 = std::pow(current_time, 4);
                const double sin_px = std::sin(pi * x_q[0]);
                const double cos_px = std::cos(pi * x_q[0]);
                const double sin_py = std::sin(pi * x_q[1]);
                const double cos_py = std::cos(pi * x_q[1]);
                
                // ∇u_x = [π t^4 cos(πx)cos(πy), -π t^4 sin(πx)sin(πy)]
                grad_u_exact[0][0] = t4 * pi * cos_px * cos_py;
                grad_u_exact[0][1] = -t4 * pi * sin_px * sin_py;
                // ∇u_y = [π t^4 sin(πx)sin(πy), -π t^4 cos(πx)cos(πy)]
                grad_u_exact[1][0] = t4 * pi * sin_px * sin_py;
                grad_u_exact[1][1] = -t4 * pi * cos_px * cos_py;
                
                const double p_exact = exact_p.value(x_q);
                
                // Velocity errors
                dealii::Tensor<1, dim> u_error = u_h_values[q] - u_exact;
                dealii::Tensor<2, dim> grad_u_error = u_h_gradients[q] - grad_u_exact;
                
                u_L2_sq += u_error.norm_square() * JxW;
                u_H1_sq += dealii::scalar_product(grad_u_error, grad_u_error) * JxW;
                
                // Pressure integrals for mean computation
                p_h_integral += p_h_values[q] * JxW;
                p_exact_integral += p_exact * JxW;
                domain_volume += JxW;
            }
        }
        
        // Compute means
        const double p_h_mean = p_h_integral / domain_volume;
        const double p_exact_mean = p_exact_integral / domain_volume;
        
        // Second pass: compute pressure error with mean subtraction
        for (const auto& cell : ns_dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            
            std::vector<double> p_h_values(n_q_points);
            fe_values[pressure].get_function_values(ns_full, p_h_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const auto& x_q = fe_values.quadrature_point(q);
                const double JxW = fe_values.JxW(q);
                
                const double p_exact = exact_p.value(x_q);
                
                // Error with mean subtraction
                const double p_error = (p_h_values[q] - p_h_mean) - (p_exact - p_exact_mean);
                p_L2_sq_raw += p_error * p_error * JxW;
            }
        }
        
        errors.u_L2 = std::sqrt(u_L2_sq);
        errors.u_H1 = std::sqrt(u_H1_sq);
        errors.p_L2 = std::sqrt(p_L2_sq_raw);
    }
    
    // ========================================================================
    // Cahn-Hilliard errors (c and μ)
    // ========================================================================
    {
        const auto& fe = ch_dof_handler.get_fe();
        dealii::FEValues<dim> fe_values(fe, quadrature,
            dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
        
        const dealii::FEValuesExtractors::Scalar c_extract(0);
        const dealii::FEValuesExtractors::Scalar mu_extract(1);
        
        const unsigned int n_q_points = quadrature.size();
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        // Convert block vector to full vector
        dealii::Vector<double> ch_full(ch_solution.block(0).size() + ch_solution.block(1).size());
        for (unsigned int i = 0; i < ch_solution.block(0).size(); ++i)
            ch_full[i] = ch_solution.block(0)[i];
        for (unsigned int i = 0; i < ch_solution.block(1).size(); ++i)
            ch_full[ch_solution.block(0).size() + i] = ch_solution.block(1)[i];
        
        // Exact solutions
        MMSExactPhaseField<dim> exact_c;
        MMSExactChemPotential<dim> exact_mu;
        exact_c.set_time(current_time);
        exact_mu.set_time(current_time);
        
        double c_L2_sq = 0.0, c_H1_sq = 0.0, mu_L2_sq = 0.0;
        
        for (const auto& cell : ch_dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);
            
            std::vector<double> c_h_values(n_q_points);
            std::vector<dealii::Tensor<1, dim>> c_h_gradients(n_q_points);
            std::vector<double> mu_h_values(n_q_points);
            
            fe_values[c_extract].get_function_values(ch_full, c_h_values);
            fe_values[c_extract].get_function_gradients(ch_full, c_h_gradients);
            fe_values[mu_extract].get_function_values(ch_full, mu_h_values);
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const auto& x_q = fe_values.quadrature_point(q);
                const double JxW = fe_values.JxW(q);
                
                // Exact values
                const double c_exact = exact_c.value(x_q);
                const double mu_exact = exact_mu.value(x_q);
                
                // Exact gradient of c
                const double pi = dealii::numbers::PI;
                const double t4 = std::pow(current_time, 4);
                dealii::Tensor<1, dim> grad_c_exact;
                grad_c_exact[0] = -t4 * pi * std::sin(pi * x_q[0]) * std::cos(pi * x_q[1]);
                grad_c_exact[1] = -t4 * pi * std::cos(pi * x_q[0]) * std::sin(pi * x_q[1]);
                
                // Errors
                const double c_error = c_h_values[q] - c_exact;
                dealii::Tensor<1, dim> grad_c_error = c_h_gradients[q] - grad_c_exact;
                const double mu_error = mu_h_values[q] - mu_exact;
                
                c_L2_sq += c_error * c_error * JxW;
                c_H1_sq += grad_c_error.norm_square() * JxW;
                mu_L2_sq += mu_error * mu_error * JxW;
            }
        }
        
        errors.c_L2 = std::sqrt(c_L2_sq);
        errors.c_H1 = std::sqrt(c_H1_sq);
        errors.mu_L2 = std::sqrt(mu_L2_sq);
    }
    
    return errors;
}

// ============================================================================
// Print convergence table footer with expected rates
// ============================================================================
inline void print_convergence_footer()
{
    std::cout << "\n";
    std::cout << "Expected convergence rates (Q2-Q1 Taylor-Hood, Q2 for CH):\n";
    std::cout << "  u_L2:  O(h³)  (velocity L² norm)\n";
    std::cout << "  u_H1:  O(h²)  (velocity H¹ seminorm)\n";
    std::cout << "  p_L2:  O(h²)  (pressure L² norm, with mean subtraction)\n";
    std::cout << "  c_L2:  O(h³)  (phase field L² norm)\n";
    std::cout << "  c_H1:  O(h²)  (phase field H¹ seminorm)\n";
    std::cout << "  mu_L2: O(h²-³) (chemical potential L² norm)\n";
}

#endif // MMS_ERROR_COMPUTATION_H
