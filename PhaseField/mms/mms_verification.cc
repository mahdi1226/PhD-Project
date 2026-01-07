// ============================================================================
// mms/mms_verification.cc - MMS Verification Implementation
//
// This file validates the REAL production assemblers using MMS:
// - CH: uses assemble_ch_system from assembly/ch_assembler.h
// - Poisson: uses assemble_poisson_system_mms_simplified from mms/poisson_mms.h
// - NS: uses assemble_ns_system from assembly/ns_assembler.h
// - Magnetization: uses MagnetizationAssembler from assembly/magnetization_assembler.h
// ============================================================================

#include "mms/mms_verification.h"
#include "mms/ch_mms.h"
#include "mms/poisson_mms.h"
#include "mms/ns_mms.h"
#include "mms/magnetization_mms.h"
#include "assembly/ch_assembler.h"
#include "assembly/ns_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "assembly/poisson_assembler.h"
#include "setup/ch_setup.h"
#include "setup/poisson_setup.h"
#include "setup/ns_setup.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>


#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>

// ============================================================================
// Local helper: Extract individual solutions from coupled NS solution
// (avoids dependency on ns_solver.cc)
// ============================================================================
static void extract_ns_solutions_local(
    const dealii::Vector<double>& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution)
{
    if (ux_solution.size() != ux_to_ns_map.size())
        ux_solution.reinit(ux_to_ns_map.size());
    if (uy_solution.size() != uy_to_ns_map.size())
        uy_solution.reinit(uy_to_ns_map.size());
    if (p_solution.size() != p_to_ns_map.size())
        p_solution.reinit(p_to_ns_map.size());

    for (unsigned int i = 0; i < ux_to_ns_map.size(); ++i)
        ux_solution[i] = ns_solution[ux_to_ns_map[i]];
    for (unsigned int i = 0; i < uy_to_ns_map.size(); ++i)
        uy_solution[i] = ns_solution[uy_to_ns_map[i]];
    for (unsigned int i = 0; i < p_to_ns_map.size(); ++i)
        p_solution[i] = ns_solution[p_to_ns_map[i]];
}

// ============================================================================
// Helper: compute convergence rate
// ============================================================================
static double compute_single_rate(double e_fine, double e_coarse,
                                  double h_fine, double h_coarse)
{
    if (e_coarse < 1e-15 || e_fine < 1e-15) return 0.0;
    return std::log(e_coarse / e_fine) / std::log(h_coarse / h_fine);
}

static void fill_rates(const std::vector<double>& errors,
                       const std::vector<double>& h_values,
                       std::vector<double>& rates)
{
    rates.clear();
    for (size_t i = 1; i < errors.size(); ++i)
        rates.push_back(compute_single_rate(errors[i], errors[i - 1],
                                            h_values[i], h_values[i - 1]));
}

// ============================================================================
// MMSConvergenceResult Implementation
// ============================================================================

void MMSConvergenceResult::compute_rates()
{
    fill_rates(theta_L2, h_values, theta_L2_rate);
    fill_rates(theta_H1, h_values, theta_H1_rate);
    fill_rates(psi_L2, h_values, psi_L2_rate);
    fill_rates(phi_L2, h_values, phi_L2_rate);
    fill_rates(phi_H1, h_values, phi_H1_rate);
    fill_rates(ux_L2, h_values, ux_L2_rate);
    fill_rates(ux_H1, h_values, ux_H1_rate);
    fill_rates(uy_L2, h_values, uy_L2_rate);
    fill_rates(uy_H1, h_values, uy_H1_rate);
    fill_rates(p_L2, h_values, p_L2_rate);
    fill_rates(M_L2, h_values, M_L2_rate);
}

void MMSConvergenceResult::print() const
{
    std::cout << "\n========================================\n";
    std::cout << "MMS Convergence Results: " << to_string(level) << "\n";
    std::cout << "========================================\n";

    std::cout << std::left << std::setw(6) << "Ref"
              << std::setw(12) << "h";

    if (!theta_L2.empty())
        std::cout << std::setw(12) << "theta_L2" << std::setw(8) << "rate"
                  << std::setw(12) << "theta_H1" << std::setw(8) << "rate"
                  << std::setw(12) << "psi_L2" << std::setw(8) << "rate";

    if (!phi_L2.empty())
        std::cout << std::setw(12) << "phi_L2" << std::setw(8) << "rate"
                  << std::setw(12) << "phi_H1" << std::setw(8) << "rate";

    if (!ux_L2.empty())
        std::cout << std::setw(12) << "ux_L2" << std::setw(8) << "rate"
                  << std::setw(12) << "ux_H1" << std::setw(8) << "rate"
                  << std::setw(12) << "p_L2" << std::setw(8) << "rate";

    if (!M_L2.empty())
        std::cout << std::setw(12) << "M_L2" << std::setw(8) << "rate";

    std::cout << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        std::cout << std::left << std::setw(6) << refinements[i]
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << h_values[i];

        if (!theta_L2.empty())
        {
            std::cout << std::setw(12) << theta_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? theta_L2_rate[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << theta_H1[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? theta_H1_rate[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << psi_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? psi_L2_rate[i-1] : 0.0);
        }

        if (!phi_L2.empty())
        {
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << phi_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? phi_L2_rate[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << phi_H1[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? phi_H1_rate[i-1] : 0.0);
        }

        if (!ux_L2.empty())
        {
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << ux_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_L2_rate[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << ux_H1[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? ux_H1_rate[i-1] : 0.0)
                      << std::scientific << std::setprecision(2)
                      << std::setw(12) << p_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? p_L2_rate[i-1] : 0.0);
        }

        if (!M_L2.empty())
        {
            std::cout << std::scientific << std::setprecision(2)
                      << std::setw(12) << M_L2[i]
                      << std::fixed << std::setprecision(2)
                      << std::setw(8) << (i > 0 ? M_L2_rate[i-1] : 0.0);
        }

        std::cout << "\n";
    }

    std::cout << "========================================\n\n";
}

bool MMSConvergenceResult::passes(double tolerance) const
{
    bool pass = true;

    auto check = [&](const std::string& name, const std::vector<double>& rates,
                     double expected) -> bool
    {
        if (rates.empty()) return true;
        // Check last rate (finest mesh pair)
        const double rate = rates.back();
        if (rate < expected - tolerance)
        {
            std::cout << "[FAIL] " << name << " rate = " << rate
                      << " < " << expected << " - " << tolerance << "\n";
            return false;
        }
        return true;
    };

    // CH checks
    if (!theta_L2_rate.empty())
    {
        pass &= check("theta_L2", theta_L2_rate, expected_L2_rate);
        pass &= check("theta_H1", theta_H1_rate, expected_H1_rate);
        pass &= check("psi_L2", psi_L2_rate, expected_L2_rate);
    }

    // Poisson checks
    if (!phi_L2_rate.empty())
    {
        pass &= check("phi_L2", phi_L2_rate, expected_L2_rate);
        pass &= check("phi_H1", phi_H1_rate, expected_H1_rate);
    }

    // NS checks
    if (!ux_L2_rate.empty())
    {
        pass &= check("ux_L2", ux_L2_rate, expected_L2_rate);
        pass &= check("ux_H1", ux_H1_rate, expected_H1_rate);
        pass &= check("uy_L2", uy_L2_rate, expected_L2_rate);
        pass &= check("uy_H1", uy_H1_rate, expected_H1_rate);
        // Pressure: Q1 elements, so L2 ~ O(h²)
        pass &= check("p_L2", p_L2_rate, expected_H1_rate); // p uses Q1, so rate = 2
    }

    // Magnetization checks
    if (!M_L2_rate.empty())
    {
        pass &= check("M_L2", M_L2_rate, expected_L2_rate);
    }

    if (pass && (!theta_L2_rate.empty() || !phi_L2_rate.empty() || !ux_L2_rate.empty() || !M_L2_rate.empty()))
        std::cout << "[PASS] All convergence rates within tolerance!\n";

    return pass;
}

void MMSConvergenceResult::write_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "[MMS] Failed to open " << filename << " for writing\n";
        return;
    }

    // Header
    file << "level,fe_degree,refinement,h,n_dofs,dt,n_time_steps,";
    if (!theta_L2.empty())
        file << "theta_L2,theta_L2_rate,theta_H1,theta_H1_rate,psi_L2,psi_L2_rate,";
    if (!phi_L2.empty())
        file << "phi_L2,phi_L2_rate,phi_H1,phi_H1_rate,";
    if (!ux_L2.empty())
        file << "ux_L2,ux_L2_rate,ux_H1,ux_H1_rate,uy_L2,uy_L2_rate,uy_H1,uy_H1_rate,p_L2,p_L2_rate,div_u_L2,";
    file << "wall_time_s\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        file << to_string(level) << ","
            << fe_degree << ","
            << refinements[i] << ","
            << std::scientific << std::setprecision(6) << h_values[i] << ","
            << (i < n_dofs.size() ? n_dofs[i] : 0) << ","
            << dt << ","
            << n_time_steps << ",";

        // CH errors
        if (!theta_L2.empty())
        {
            file << theta_L2[i] << ","
                << (i > 0 && i - 1 < theta_L2_rate.size() ? theta_L2_rate[i - 1] : 0.0) << ","
                << theta_H1[i] << ","
                << (i > 0 && i - 1 < theta_H1_rate.size() ? theta_H1_rate[i - 1] : 0.0) << ","
                << psi_L2[i] << ","
                << (i > 0 && i - 1 < psi_L2_rate.size() ? psi_L2_rate[i - 1] : 0.0) << ",";
        }

        // Poisson errors
        if (!phi_L2.empty())
        {
            file << phi_L2[i] << ","
                << (i > 0 && i - 1 < phi_L2_rate.size() ? phi_L2_rate[i - 1] : 0.0) << ","
                << phi_H1[i] << ","
                << (i > 0 && i - 1 < phi_H1_rate.size() ? phi_H1_rate[i - 1] : 0.0) << ",";
        }

        // NS errors
        if (!ux_L2.empty())
        {
            file << ux_L2[i] << ","
                << (i > 0 && i - 1 < ux_L2_rate.size() ? ux_L2_rate[i - 1] : 0.0) << ","
                << ux_H1[i] << ","
                << (i > 0 && i - 1 < ux_H1_rate.size() ? ux_H1_rate[i - 1] : 0.0) << ","
                << uy_L2[i] << ","
                << (i > 0 && i - 1 < uy_L2_rate.size() ? uy_L2_rate[i - 1] : 0.0) << ","
                << uy_H1[i] << ","
                << (i > 0 && i - 1 < uy_H1_rate.size() ? uy_H1_rate[i - 1] : 0.0) << ","
                << p_L2[i] << ","
                << (i > 0 && i - 1 < p_L2_rate.size() ? p_L2_rate[i - 1] : 0.0) << ","
                << div_u_L2[i] << ",";
        }

        file << std::fixed << std::setprecision(3)
            << (i < wall_times.size() ? wall_times[i] : 0.0) << "\n";
    }

    file.close();
    std::cout << "[MMS] Results written to " << filename << "\n";
}

// ============================================================================
// CH Standalone Implementation - Uses REAL assembler
// ============================================================================

template <int dim>
static MMSConvergenceResult run_ch_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::CH_STANDALONE;
    result.fe_degree = params.fe.degree_phase;
    result.n_time_steps = n_time_steps;

    // Expected rates based on FE degree
    result.expected_L2_rate = params.fe.degree_phase + 1; // p+1 for L2
    result.expected_H1_rate = params.fe.degree_phase; // p for H1

    const double t_init = 0.1;
    const double t_final = 0.2;
    const double dt = (t_final - t_init) / n_time_steps;
    result.dt = dt;

    params.enable_mms = true;
    params.time.dt = dt;

    std::cout << "\n[CH_STANDALONE] Running convergence study...\n";
    std::cout << "  t in [" << t_init << ", " << t_final << "], dt = " << dt << "\n";
    std::cout << "  epsilon = " << params.physics.epsilon
        << ", gamma = " << params.physics.mobility << "\n";
    std::cout << "  FE degree = Q" << params.fe.degree_phase << "\n";
    std::cout << "  Using REAL ch_assembler\n\n";

    for (unsigned int ref : refinements)
    {
        std::cout << "  Refinement " << ref << "... " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, 1.0));
        triangulation.refine_global(ref);

        // Setup FE
        dealii::FE_Q<dim> fe(params.fe.degree_phase);
        dealii::DoFHandler<dim> theta_dof_handler(triangulation);
        dealii::DoFHandler<dim> psi_dof_handler(triangulation);
        theta_dof_handler.distribute_dofs(fe);
        psi_dof_handler.distribute_dofs(fe);

        const unsigned int n_theta = theta_dof_handler.n_dofs();
        const unsigned int n_psi = psi_dof_handler.n_dofs();
        const unsigned int n_ch = n_theta + n_psi;

        // Solution vectors
        dealii::Vector<double> theta_solution(n_theta);
        dealii::Vector<double> theta_old(n_theta);
        dealii::Vector<double> psi_solution(n_psi);
        dealii::Vector<double> ux_solution, uy_solution;

        // Initialize with exact solution at t_init
        apply_ch_mms_initial_conditions(theta_dof_handler, psi_dof_handler,
                                        theta_solution, psi_solution, t_init);
        theta_old = theta_solution;

        // Time stepping
        double current_time = t_init;

        for (unsigned int step = 0; step < n_time_steps; ++step)
        {
            current_time += dt;
            theta_old = theta_solution;

            // Setup MMS Dirichlet constraints
            dealii::AffineConstraints<double> theta_constraints;
            dealii::AffineConstraints<double> psi_constraints;
            apply_ch_mms_boundary_constraints(theta_dof_handler, psi_dof_handler,
                                              theta_constraints, psi_constraints,
                                              current_time);

            // Use REAL setup function
            std::vector<dealii::types::global_dof_index> theta_to_ch;
            std::vector<dealii::types::global_dof_index> psi_to_ch;
            dealii::AffineConstraints<double> ch_constraints;
            dealii::SparsityPattern ch_sparsity;

            setup_ch_coupled_system<dim>(
                theta_dof_handler, psi_dof_handler,
                theta_constraints, psi_constraints,
                theta_to_ch, psi_to_ch,
                ch_constraints, ch_sparsity,
                false);

            dealii::SparseMatrix<double> ch_matrix(ch_sparsity);
            dealii::Vector<double> ch_rhs(n_ch);
            dealii::Vector<double> ch_solution(n_ch);

            // Assemble using REAL assembler
            assemble_ch_system<dim>(
                theta_dof_handler, psi_dof_handler,
                theta_old, ux_solution, uy_solution,
                params, dt, current_time,
                theta_to_ch, psi_to_ch,
                ch_matrix, ch_rhs);

            // Apply constraints
            ch_constraints.condense(ch_matrix, ch_rhs);

            // Solve
            dealii::SparseDirectUMFPACK solver;
            solver.initialize(ch_matrix);
            solver.vmult(ch_solution, ch_rhs);

            ch_constraints.distribute(ch_solution);

            // Extract solutions
            for (unsigned int i = 0; i < n_theta; ++i)
                theta_solution[i] = ch_solution[theta_to_ch[i]];
            for (unsigned int i = 0; i < n_psi; ++i)
                psi_solution[i] = ch_solution[psi_to_ch[i]];
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // Compute errors
        CHMMSErrors errors = compute_ch_mms_errors(
            theta_dof_handler, psi_dof_handler,
            theta_solution, psi_solution, t_final);

        result.refinements.push_back(ref);
        result.h_values.push_back(errors.h);
        result.theta_L2.push_back(errors.theta_L2);
        result.theta_H1.push_back(errors.theta_H1);
        result.psi_L2.push_back(errors.psi_L2);
        result.n_dofs.push_back(n_ch);
        result.wall_times.push_back(wall_time);

        std::cout << "theta_L2 = " << std::scientific << std::setprecision(3)
            << errors.theta_L2 << ", time = " << std::fixed << std::setprecision(2)
            << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// Poisson Standalone Implementation - Uses REAL assembler with M=0
// ============================================================================

template <int dim>
static MMSConvergenceResult run_poisson_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int /* n_time_steps - not used for steady-state */)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::POISSON_STANDALONE;
    result.fe_degree = params.fe.degree_potential;
    result.n_time_steps = 1; // Steady-state

    // Expected rates based on FE degree
    result.expected_L2_rate = params.fe.degree_potential + 1; // p+1 for L2
    result.expected_H1_rate = params.fe.degree_potential; // p for H1
    result.dt = 0.0; // Steady-state

    // Enable MMS mode
    params.enable_mms = true;

    // Disable dipoles for standalone test (h_a = 0)
    params.dipoles.positions.clear();
    params.dipoles.intensity_max = 0.0;

    // Use t = 1.0 for the MMS test (amplitude of exact solution)
    const double time = 1.0;
    const double L_y = 1.0;  // Unit square domain
    params.domain.y_max = L_y;
    params.domain.y_min = 0.0;

    std::cout << "\n[POISSON_STANDALONE] Running convergence study...\n";
    std::cout << "  Steady-state problem with MMS source\n";
    std::cout << "  φ_exact = t·cos(πx)·cos(πy/L_y), t = " << time << "\n";
    std::cout << "  FE degree = Q" << params.fe.degree_potential << "\n";
    std::cout << "  Using REAL assemble_poisson_system with M=0\n\n";

    for (unsigned int ref : refinements)
    {
        std::cout << "  Refinement " << ref << "... " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh on [0,1]²
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        // Setup FE for φ (CG)
        dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);
        dealii::DoFHandler<dim> phi_dof_handler(triangulation);
        phi_dof_handler.distribute_dofs(fe_phi);

        // Setup DUMMY FE for M (DG) - required by assembler interface
        dealii::FE_DGQ<dim> fe_M(0);
        dealii::DoFHandler<dim> M_dof_handler(triangulation);
        M_dof_handler.distribute_dofs(fe_M);

        const unsigned int n_phi = phi_dof_handler.n_dofs();

        // EMPTY M vectors (standalone: M = 0)
        dealii::Vector<double> mx_solution;  // Empty vector signals standalone
        dealii::Vector<double> my_solution;  // Empty vector signals standalone

        // Setup constraints (with MMS-aware pinning)
        dealii::AffineConstraints<double> phi_constraints;
        setup_poisson_mms_constraints(phi_dof_handler, phi_constraints, time, L_y);

        // Setup sparsity pattern
        dealii::DynamicSparsityPattern dsp(n_phi);
        dealii::DoFTools::make_sparsity_pattern(phi_dof_handler, dsp,
                                                phi_constraints, false);
        dealii::SparsityPattern phi_sparsity;
        phi_sparsity.copy_from(dsp);

        // Allocate system
        dealii::SparseMatrix<double> phi_matrix(phi_sparsity);
        dealii::Vector<double> phi_rhs(n_phi);
        dealii::Vector<double> phi_solution(n_phi);

        // Use REAL assembler with empty M vectors
        assemble_poisson_system<dim>(
            phi_dof_handler,
            M_dof_handler,
            mx_solution,      // Empty - signals standalone mode
            my_solution,      // Empty - signals standalone mode
            params,
            time,
            phi_matrix,
            phi_rhs,
            phi_constraints);

        // Solve
        dealii::SparseDirectUMFPACK solver;
        solver.initialize(phi_matrix);
        solver.vmult(phi_solution, phi_rhs);

        phi_constraints.distribute(phi_solution);

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // Compute errors
        PoissonMMSError errors = compute_poisson_mms_error(
            phi_dof_handler, phi_solution, time, L_y);

        // Compute mesh size
        const double h = phi_dof_handler.begin_active()->diameter();

        result.refinements.push_back(ref);
        result.h_values.push_back(h);
        result.phi_L2.push_back(errors.L2_error);
        result.phi_H1.push_back(errors.H1_error);
        result.n_dofs.push_back(n_phi);
        result.wall_times.push_back(wall_time);

        std::cout << "phi_L2 = " << std::scientific << std::setprecision(3)
            << errors.L2_error << ", phi_H1 = " << errors.H1_error
            << ", time = " << std::fixed << std::setprecision(2)
            << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// NS Standalone Implementation - Uses REAL assembler
// ============================================================================

template <int dim>
static MMSConvergenceResult run_ns_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::NS_STANDALONE;
    result.fe_degree = params.fe.degree_velocity;
    result.n_time_steps = n_time_steps;

    // Expected rates: Q2-Q1 Taylor-Hood
    result.expected_L2_rate = params.fe.degree_velocity + 1; // 3 for Q2
    result.expected_H1_rate = params.fe.degree_velocity;     // 2 for Q2

    // Time integration parameters
    const double t_start = 0.1;
    const double t_end = 0.2;
    result.dt = (t_end - t_start) / n_time_steps;
    const double L_y = 1.0;
    const double nu = 1.0;

    // CRITICAL: Enable MMS mode in parameters
    params.enable_mms = true;
    params.physics.nu_water = nu;  // MMS uses this as constant viscosity
    params.domain.y_max = L_y;
    params.domain.y_min = 0.0;

    // Disable other physics for standalone NS test
    params.enable_magnetic = false;
    params.enable_gravity = false;

    std::cout << "\n[NS_STANDALONE] Running convergence study...\n";
    std::cout << "  t in [" << t_start << ", " << t_end << "], dt = " << result.dt << "\n";
    std::cout << "  nu = " << nu << ", L_y = " << L_y << "\n";
    std::cout << "  Using REAL ns_assembler with enable_mms=true\n";
    std::cout << "  FE: Q" << params.fe.degree_velocity << " velocity, Q"
              << params.fe.degree_pressure << " pressure\n\n";

    for (unsigned int ref : refinements)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        const double h = triangulation.begin_active()->diameter();
        const double dt = 0.01 * h;  // Scale dt with mesh size
        const unsigned int n_steps = static_cast<unsigned int>((t_end - t_start) / dt) + 1;

        std::cout << "  Refinement " << ref << " (h=" << std::scientific << h << ", dt=" << dt
          << ", steps=" << n_steps << ")... " << std::flush;

        // Setup FE (Taylor-Hood: Q2-Q1 for velocity-pressure)
        dealii::FE_Q<dim> fe_velocity(params.fe.degree_velocity);
        dealii::FE_Q<dim> fe_pressure(params.fe.degree_pressure);
        dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);  // For dummy theta/psi

        // Velocity and pressure DoF handlers
        dealii::DoFHandler<dim> ux_dof_handler(triangulation);
        dealii::DoFHandler<dim> uy_dof_handler(triangulation);
        dealii::DoFHandler<dim> p_dof_handler(triangulation);
        ux_dof_handler.distribute_dofs(fe_velocity);
        uy_dof_handler.distribute_dofs(fe_velocity);
        p_dof_handler.distribute_dofs(fe_pressure);

        // DUMMY theta/psi DoF handlers (required by real assembler interface)
        dealii::DoFHandler<dim> theta_dof_handler(triangulation);
        dealii::DoFHandler<dim> psi_dof_handler(triangulation);
        theta_dof_handler.distribute_dofs(fe_phase);
        psi_dof_handler.distribute_dofs(fe_phase);

        const unsigned int n_ux = ux_dof_handler.n_dofs();
        const unsigned int n_uy = uy_dof_handler.n_dofs();
        const unsigned int n_p = p_dof_handler.n_dofs();
        const unsigned int n_theta = theta_dof_handler.n_dofs();
        const unsigned int n_psi = psi_dof_handler.n_dofs();
        const unsigned int n_total = n_ux + n_uy + n_p;

        // Setup velocity constraints (no-slip on all boundaries for MMS)
        dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
        setup_ns_mms_velocity_constraints(ux_dof_handler, uy_dof_handler,
                                          ux_constraints, uy_constraints);

        // Setup pressure constraints - pin DoF 0 to zero
        // (error computation uses mean subtraction, so pinning to zero is fine)
        p_constraints.clear();
        dealii::DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);
        if (p_dof_handler.n_dofs() > 0 && !p_constraints.is_constrained(0))
        {
            p_constraints.add_line(0);
            p_constraints.set_inhomogeneity(0, 0.0);
        }
        p_constraints.close();

        // Setup coupled system
        std::vector<dealii::types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
        dealii::AffineConstraints<double> ns_constraints;
        dealii::SparsityPattern ns_sparsity;

        setup_ns_coupled_system(ux_dof_handler, uy_dof_handler, p_dof_handler,
                                ux_constraints, uy_constraints, p_constraints,
                                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                                ns_constraints, ns_sparsity, false);

        // Allocate system
        dealii::SparseMatrix<double> ns_matrix(ns_sparsity);
        dealii::Vector<double> ns_rhs(n_total);
        dealii::Vector<double> ns_solution(n_total);

        // Individual solution vectors
        dealii::Vector<double> ux_solution(n_ux), uy_solution(n_uy), p_solution(n_p);
        dealii::Vector<double> ux_old(n_ux), uy_old(n_uy);

        // DUMMY theta/psi solutions (constant values, not used in MMS physics)
        dealii::Vector<double> theta_old(n_theta);
        dealii::Vector<double> psi_solution(n_psi);
        theta_old = 1.0;   // Constant theta = 1 (gives nu = nu_water)
        psi_solution = 0.0; // Zero psi (no capillary force)

        // Apply initial conditions at t_start
        apply_ns_mms_initial_conditions(ux_dof_handler, uy_dof_handler, p_dof_handler,
                                        ux_old, uy_old, p_solution, t_start, L_y);
        ux_solution = ux_old;
        uy_solution = uy_old;

        // Time stepping
        double current_time = t_start;
        for (unsigned int step = 0; step < n_steps; ++step)
        {
            current_time += dt;

            // CALL THE REAL ASSEMBLER
            assemble_ns_system<dim>(
                ux_dof_handler,
                uy_dof_handler,
                p_dof_handler,
                theta_dof_handler,   // Dummy for MMS
                psi_dof_handler,     // Dummy for MMS
                nullptr,             // No phi_dof_handler
                nullptr,             // No M_dof_handler
                ux_old,
                uy_old,
                theta_old,           // Dummy constant field
                psi_solution,        // Dummy zero field
                nullptr,             // No phi_solution
                nullptr,             // No mx_solution
                nullptr,             // No my_solution
                params,              // Has enable_mms = true
                dt,
                current_time,
                ux_to_ns_map,
                uy_to_ns_map,
                p_to_ns_map,
                ns_constraints,
                ns_matrix,
                ns_rhs);

            // Solve using direct solver
            dealii::SparseDirectUMFPACK solver;
            solver.initialize(ns_matrix);
            solver.vmult(ns_solution, ns_rhs);

            ns_constraints.distribute(ns_solution);

            // Extract solutions
            extract_ns_solutions_local(ns_solution, ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                                       ux_solution, uy_solution, p_solution);

            // Update old solutions for next step
            ux_old = ux_solution;
            uy_old = uy_solution;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // Compute errors against exact solution
        NSMMSError errors = compute_ns_mms_error(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_solution, uy_solution, p_solution,
            current_time, L_y);

        result.refinements.push_back(ref);
        result.h_values.push_back(h);
        result.ux_L2.push_back(errors.ux_L2);
        result.ux_H1.push_back(errors.ux_H1);
        result.uy_L2.push_back(errors.uy_L2);
        result.uy_H1.push_back(errors.uy_H1);
        result.p_L2.push_back(errors.p_L2);
        result.div_u_L2.push_back(errors.div_U_L2);
        result.n_dofs.push_back(n_total);
        result.wall_times.push_back(wall_time);

        std::cout << "ux_L2 = " << std::scientific << std::setprecision(3) << errors.ux_L2
          << ", ux_H1 = " << errors.ux_H1
          << ", p_L2 = " << errors.p_L2
          << ", div_U = " << errors.div_U_L2
          << ", time = " << std::fixed << std::setprecision(2) << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}


// ============================================================================
// Magnetization Standalone Implementation - Uses REAL assembler
//
// Tests the DG magnetization transport equation with relaxation.
// For standalone: U = 0 (no transport), θ = 1, φ = 0 (no equilibrium)
//
// Equation: (1/τ + 1/τ_M)(M^n, Z) = (1/τ)(M^{n-1}, Z) + (f_MMS, Z)
// ============================================================================

template <int dim>
static MMSConvergenceResult run_magnetization_standalone(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps)
{

    MMSConvergenceResult result;
    result.level = MMSLevel::MAGNETIZATION_STANDALONE;

    // DG-Q1 for magnetization (standard choice from paper)
    const unsigned int mag_degree = 1;
    result.fe_degree = mag_degree;
    result.n_time_steps = n_time_steps;

    // Expected rates: DG-Q1 gives O(h²) in L2
    result.expected_L2_rate = mag_degree + 1;  // 2 for Q1
    result.expected_H1_rate = mag_degree;      // 1 for Q1

    // Time integration parameters
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double L_y = 1.0;

    // CRITICAL: Enable MMS mode in parameters
    params.enable_mms = true;
    params.domain.y_max = L_y;
    params.domain.y_min = 0.0;
    params.physics.tau_M = 1.0;  // Relaxation time
    params.physics.chi_0 = 1.0;  // Susceptibility

    // CRITICAL: Disable dipoles for standalone test (h_a = 0)
    params.dipoles.positions.clear();
    params.dipoles.intensity_max = 0.0;

    std::cout << "\n[MAGNETIZATION_STANDALONE] Running convergence study...\n";
    std::cout << "  t in [" << t_start << ", " << t_end << "]\n";
    std::cout << "  tau_M = " << params.physics.tau_M << ", chi_0 = " << params.physics.chi_0 << "\n";
    std::cout << "  Using REAL MagnetizationAssembler with enable_mms=true\n";
    std::cout << "  FE: DG-Q" << mag_degree << "\n\n";

    std::cout << "\n[MAGNETIZATION_STANDALONE] Running convergence study...\n";
    std::cout << "  t in [" << t_start << ", " << t_end << "]\n";
    std::cout << "  tau_M = " << params.physics.tau_M << ", chi_0 = " << params.physics.chi_0 << "\n";
    std::cout << "  Using REAL MagnetizationAssembler with enable_mms=true\n";
    std::cout << "  FE: DG-Q" << mag_degree << "\n\n";

    for (unsigned int ref : refinements)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        const double h = triangulation.begin_active()->diameter();
        const double dt = 0.1 * h;  // Scale dt with mesh size
        const unsigned int n_steps = static_cast<unsigned int>((t_end - t_start) / dt) + 1;
        const double actual_dt = (t_end - t_start) / n_steps;

        std::cout << "  Refinement " << ref << " (h=" << std::scientific << h
                  << ", dt=" << actual_dt << ", steps=" << n_steps << ")... " << std::flush;

        // Setup FE spaces
        // M uses DG elements
        dealii::FE_DGQ<dim> fe_M(mag_degree);
        // U, phi, theta use CG elements (dummy for standalone test)
        dealii::FE_Q<dim> fe_U(params.fe.degree_velocity);
        dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);
        dealii::FE_Q<dim> fe_theta(params.fe.degree_phase);

        // DoF handlers
        dealii::DoFHandler<dim> M_dof_handler(triangulation);
        dealii::DoFHandler<dim> U_dof_handler(triangulation);
        dealii::DoFHandler<dim> phi_dof_handler(triangulation);
        dealii::DoFHandler<dim> theta_dof_handler(triangulation);

        M_dof_handler.distribute_dofs(fe_M);
        U_dof_handler.distribute_dofs(fe_U);
        phi_dof_handler.distribute_dofs(fe_phi);
        theta_dof_handler.distribute_dofs(fe_theta);

        const unsigned int n_M = M_dof_handler.n_dofs();
        const unsigned int n_U = U_dof_handler.n_dofs();
        const unsigned int n_phi = phi_dof_handler.n_dofs();
        const unsigned int n_theta = theta_dof_handler.n_dofs();

        // Setup sparsity pattern for DG
        dealii::DynamicSparsityPattern dsp(n_M, n_M);
        dealii::DoFTools::make_flux_sparsity_pattern(M_dof_handler, dsp);
        dealii::SparsityPattern M_sparsity;
        M_sparsity.copy_from(dsp);

        // Allocate system
        dealii::SparseMatrix<double> M_matrix(M_sparsity);
        dealii::Vector<double> rhs_x(n_M), rhs_y(n_M);
        dealii::Vector<double> Mx_solution(n_M), My_solution(n_M);
        dealii::Vector<double> Mx_old(n_M), My_old(n_M);

        // Dummy fields (zero velocity, zero phi, constant theta = 1)
        dealii::Vector<double> Ux(n_U), Uy(n_U);
        dealii::Vector<double> phi_solution(n_phi);
        dealii::Vector<double> theta_solution(n_theta);
        Ux = 0.0;
        Uy = 0.0;
        phi_solution = 0.0;
        theta_solution = 1.0;  // χ(1) = χ₀

        // Initialize with exact solution at t_start
        apply_mag_mms_initial_conditions(M_dof_handler, Mx_old, My_old, t_start, L_y);

        // Create assembler
        MagnetizationAssembler<dim> assembler(params, M_dof_handler, U_dof_handler,
                                               phi_dof_handler, theta_dof_handler);

        // Time stepping
        double current_time = t_start;
        for (unsigned int step = 0; step < n_steps; ++step)
        {
            current_time += actual_dt;

            // Assemble
            assembler.assemble(M_matrix, rhs_x, rhs_y,
                               Ux, Uy, phi_solution, theta_solution,
                               Mx_old, My_old, actual_dt, current_time);

            // Solve Mx
            dealii::SparseDirectUMFPACK solver_x;
            solver_x.initialize(M_matrix);
            solver_x.vmult(Mx_solution, rhs_x);

            // Solve My (same matrix)
            dealii::SparseDirectUMFPACK solver_y;
            solver_y.initialize(M_matrix);
            solver_y.vmult(My_solution, rhs_y);

            // Update old solutions
            Mx_old = Mx_solution;
            My_old = My_solution;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // Compute errors
        MagMMSError errors = compute_mag_mms_error(
            M_dof_handler, Mx_solution, My_solution, current_time, L_y);

        result.refinements.push_back(ref);
        result.h_values.push_back(h);
        result.M_L2.push_back(errors.M_L2);
        result.n_dofs.push_back(n_M);
        result.wall_times.push_back(wall_time);

        std::cout << "M_L2 = " << std::scientific << std::setprecision(3) << errors.M_L2
                  << ", Mx_L2 = " << errors.Mx_L2
                  << ", My_L2 = " << errors.My_L2
                  << ", time = " << std::fixed << std::setprecision(2) << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}


// ============================================================================
// Poisson + Magnetization Coupled Test
//
// Tests the coupling between:
// - Poisson: (∇φ, ∇χ) = (h_a - M, ∇χ)
// - Magnetization: (1/τ + 1/τ_M)M - B_h^m(U,Z,M) = (1/τ_M)χ_θ·H + (1/τ)M_old
//
// With U = 0 (no transport) and θ = 1 (constant χ):
// - Poisson: -Δφ = -∇·(h_a - M)
// - Magnetization: (1/τ + 1/τ_M)M = (1/τ_M)χ·∇φ + (1/τ)M_old + f_MMS
//
// The coupling: M → Poisson RHS, φ → M equilibrium via H = ∇φ
// ============================================================================

// NOTE: φ_exact solution is now provided by PoissonExactSolution<dim> from poisson_mms.h
// No local class needed.

template <int dim>
static MMSConvergenceResult run_poisson_magnetization(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::POISSON_MAGNETIZATION;

    const unsigned int mag_degree = 1;  // DG-Q1
    const unsigned int phi_degree = 1;  // CG-Q1
    result.fe_degree = mag_degree;
    result.n_time_steps = n_time_steps;

    // Expected: both O(h²) for Q1 elements
    result.expected_L2_rate = 2.0;
    result.expected_H1_rate = 1.0;

    // Time parameters
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double L_y = 1.0;

    // Physics parameters
    params.enable_mms = true;
    params.domain.y_max = L_y;
    params.domain.y_min = 0.0;
    params.physics.tau_M = 1.0;
    params.physics.chi_0 = 1.0;

    // CRITICAL: Disable dipoles for MMS test (h_a = 0)
    params.dipoles.positions.clear();
    params.dipoles.intensity_max = 0.0;

    std::cout << "\n[POISSON_MAGNETIZATION] Running coupled convergence study...\n";

    std::cout << "\n[POISSON_MAGNETIZATION] Running coupled convergence study...\n";
    std::cout << "  t in [" << t_start << ", " << t_end << "]\n";
    std::cout << "  tau_M = " << params.physics.tau_M << ", chi_0 = " << params.physics.chi_0 << "\n";
    std::cout << "  Testing Poisson ↔ Magnetization coupling (U=0, θ=1)\n";
    std::cout << "  FE: Poisson Q" << phi_degree << ", Magnetization DG-Q" << mag_degree << "\n\n";

    for (unsigned int ref : refinements)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        const double h = triangulation.begin_active()->diameter();
        const double dt = 0.1 * h;
        const unsigned int n_steps = static_cast<unsigned int>((t_end - t_start) / dt) + 1;
        const double actual_dt = (t_end - t_start) / n_steps;

        std::cout << "  Refinement " << ref << " (h=" << std::scientific << h
                  << ", dt=" << actual_dt << ", steps=" << n_steps << ")... " << std::flush;

        // ====== FE Spaces ======
        // Magnetization: DG
        dealii::FE_DGQ<dim> fe_M(mag_degree);
        // Poisson: CG
        dealii::FE_Q<dim> fe_phi(phi_degree);
        // Dummies for assembler interface
        dealii::FE_Q<dim> fe_U(2);
        dealii::FE_Q<dim> fe_theta(1);

        // ====== DoF Handlers ======
        dealii::DoFHandler<dim> M_dof_handler(triangulation);
        dealii::DoFHandler<dim> phi_dof_handler(triangulation);
        dealii::DoFHandler<dim> U_dof_handler(triangulation);
        dealii::DoFHandler<dim> theta_dof_handler(triangulation);

        M_dof_handler.distribute_dofs(fe_M);
        phi_dof_handler.distribute_dofs(fe_phi);
        U_dof_handler.distribute_dofs(fe_U);
        theta_dof_handler.distribute_dofs(fe_theta);

        const unsigned int n_M = M_dof_handler.n_dofs();
        const unsigned int n_phi = phi_dof_handler.n_dofs();
        const unsigned int n_U = U_dof_handler.n_dofs();
        const unsigned int n_theta = theta_dof_handler.n_dofs();

        // ====== Sparsity Patterns ======
        // Magnetization (DG flux pattern)
        dealii::DynamicSparsityPattern dsp_M(n_M, n_M);
        dealii::DoFTools::make_flux_sparsity_pattern(M_dof_handler, dsp_M);
        dealii::SparsityPattern M_sparsity;
        M_sparsity.copy_from(dsp_M);

        // Poisson (CG standard pattern)
        dealii::DynamicSparsityPattern dsp_phi(n_phi, n_phi);
        dealii::DoFTools::make_sparsity_pattern(phi_dof_handler, dsp_phi);
        dealii::SparsityPattern phi_sparsity;
        phi_sparsity.copy_from(dsp_phi);

        // ====== Allocate Vectors ======
        dealii::SparseMatrix<double> M_matrix(M_sparsity);
        dealii::Vector<double> rhs_Mx(n_M), rhs_My(n_M);
        dealii::Vector<double> Mx_solution(n_M), My_solution(n_M);
        dealii::Vector<double> Mx_old(n_M), My_old(n_M);

        dealii::SparseMatrix<double> phi_matrix(phi_sparsity);
        dealii::Vector<double> phi_rhs(n_phi);
        dealii::Vector<double> phi_solution(n_phi);

        // Dummy fields
        dealii::Vector<double> Ux(n_U), Uy(n_U);
        dealii::Vector<double> theta_solution(n_theta);
        Ux = 0.0;
        Uy = 0.0;
        theta_solution = 1.0;  // Constant θ = 1 → χ = χ₀

        // ====== Initial Conditions ======
        apply_mag_mms_initial_conditions(M_dof_handler, Mx_old, My_old, t_start, L_y);

        // Initialize solution vectors from ICs (needed for first Poisson solve)
        Mx_solution = Mx_old;
        My_solution = My_old;

        // Initial φ from exact solution
        PoissonExactSolution<dim> phi_exact_func(t_start, L_y);
        dealii::VectorTools::interpolate(phi_dof_handler, phi_exact_func, phi_solution);

        // ====== Create Assemblers ======
        MagnetizationAssembler<dim> mag_assembler(params, M_dof_handler, U_dof_handler,
                                                   phi_dof_handler, theta_dof_handler);

        // ====== Time Stepping with Coupled Iterations ======
        double current_time = t_start;

        for (unsigned int step = 0; step < n_steps; ++step)
        {
            current_time += actual_dt;

            // Poisson constraints - must update each time step with time-dependent BC!
            dealii::AffineConstraints<double> phi_constraints;
            phi_constraints.clear();
            PoissonExactSolution<dim> phi_bc(current_time, L_y);
            // Apply Dirichlet BC on all boundaries (0 and 1 for hyper_rectangle)
            for (dealii::types::boundary_id bid = 0; bid <= 1; ++bid)
            {
                dealii::VectorTools::interpolate_boundary_values(
                    phi_dof_handler, bid, phi_bc, phi_constraints);
            }
            phi_constraints.close();

            // Coupled iteration (simple fixed-point)
            const unsigned int max_coupled_iters = 3;
            for (unsigned int iter = 0; iter < max_coupled_iters; ++iter)
            {
                // 1. Solve Poisson with current M using REAL assembler
                //    The real assembler uses numerical M in (h_a - M, ∇χ) term
                //    and adds MMS source (f_φ, χ) when params.enable_mms = true
                phi_matrix = 0;
                phi_rhs = 0;

                assemble_poisson_system<dim>(
                    phi_dof_handler, M_dof_handler,
                    Mx_solution, My_solution,
                    params, current_time,
                    phi_matrix, phi_rhs,
                    phi_constraints);

                // Solve Poisson
                dealii::SparseDirectUMFPACK phi_solver;
                phi_solver.initialize(phi_matrix);
                phi_solver.vmult(phi_solution, phi_rhs);
                phi_constraints.distribute(phi_solution);

                // 2. Solve Magnetization with current φ
                mag_assembler.assemble(M_matrix, rhs_Mx, rhs_My,
                                       Ux, Uy, phi_solution, theta_solution,
                                       Mx_old, My_old, actual_dt, current_time);

                dealii::SparseDirectUMFPACK Mx_solver, My_solver;
                Mx_solver.initialize(M_matrix);
                Mx_solver.vmult(Mx_solution, rhs_Mx);
                My_solver.initialize(M_matrix);
                My_solver.vmult(My_solution, rhs_My);
            }

            // Update for next time step
            Mx_old = Mx_solution;
            My_old = My_solution;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // ====== Compute Errors ======
        // Magnetization error
        MagMMSError mag_errors = compute_mag_mms_error(
            M_dof_handler, Mx_solution, My_solution, current_time, L_y);

        // Poisson error (L2)
        PoissonExactSolution<dim> phi_exact_final(current_time, L_y);
        dealii::Vector<float> phi_L2_error_per_cell(triangulation.n_active_cells());
        dealii::VectorTools::integrate_difference(
            phi_dof_handler, phi_solution, phi_exact_final,
            phi_L2_error_per_cell, dealii::QGauss<dim>(phi_degree + 2),
            dealii::VectorTools::L2_norm);
        const double phi_L2_error = dealii::VectorTools::compute_global_error(
            triangulation, phi_L2_error_per_cell, dealii::VectorTools::L2_norm);

        // Poisson error (H1 seminorm)
        dealii::Vector<float> phi_H1_error_per_cell(triangulation.n_active_cells());
        dealii::VectorTools::integrate_difference(
            phi_dof_handler, phi_solution, phi_exact_final,
            phi_H1_error_per_cell, dealii::QGauss<dim>(phi_degree + 2),
            dealii::VectorTools::H1_seminorm);
        const double phi_H1_error = dealii::VectorTools::compute_global_error(
            triangulation, phi_H1_error_per_cell, dealii::VectorTools::H1_seminorm);

        // Store results
        result.refinements.push_back(ref);
        result.h_values.push_back(h);
        result.M_L2.push_back(mag_errors.M_L2);
        result.phi_L2.push_back(phi_L2_error);
        result.phi_H1.push_back(phi_H1_error);
        result.n_dofs.push_back(n_M + n_phi);
        result.wall_times.push_back(wall_time);

        std::cout << "M_L2 = " << std::scientific << std::setprecision(3) << mag_errors.M_L2
                  << ", phi_L2 = " << phi_L2_error
                  << ", phi_H1 = " << phi_H1_error
                  << ", time = " << std::fixed << std::setprecision(2) << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}

// ============================================================================
// CH_NS_CAPILLARY: Coupled Cahn-Hilliard + Navier-Stokes
//
// Tests coupling:
//   - CH advection by velocity U (Eq. 42a)
//   - NS capillary force F_cap = ψ·∇θ (Eq. 42e)
//
// Uses REAL assemblers: ch_assembler.cc, ns_assembler.cc
// Uses REAL setup: ch_setup.cc, ns_setup.cc
// ============================================================================
template <int dim>
static MMSConvergenceResult run_ch_ns_capillary(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps)
{
    MMSConvergenceResult result;
    result.level = MMSLevel::CH_NS_CAPILLARY;
    result.fe_degree = params.fe.degree_phase;
    result.n_time_steps = n_time_steps;
    result.expected_L2_rate = 2.0;
    result.expected_H1_rate = 1.0;

    const double t_start = 0.1;
    const double t_end = 0.2;
    const double L_y = 1.0;

    params.enable_mms = true;
    params.enable_magnetic = false;
    params.enable_gravity = false;
    params.domain.y_max = L_y;
    params.domain.y_min = 0.0;

    std::cout << "  gamma = " << params.physics.mobility << "\n";
    std::cout << "\n[CH_NS_CAPILLARY] Running coupled convergence study...\n";
    std::cout << "  t in [" << t_start << ", " << t_end << "]\n";
    std::cout << "  Testing CH ↔ NS coupling (advection + capillary force)\n";
    std::cout << "  Using REAL assemblers: ch_assembler, ns_assembler\n\n";

    for (unsigned int ref : refinements)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Setup mesh
        dealii::Triangulation<dim> triangulation;
        dealii::GridGenerator::hyper_rectangle(
            triangulation,
            dealii::Point<dim>(0.0, 0.0),
            dealii::Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        const double h = triangulation.begin_active()->diameter();
        const double actual_dt = (t_end - t_start) / n_time_steps;
        const unsigned int n_steps = n_time_steps;

        std::cout << "  Refinement " << ref << " (h=" << std::scientific << h
                  << ", dt=" << actual_dt << ", steps=" << n_steps << ")... " << std::flush;

        // ====== FE Spaces ======
        dealii::FE_Q<dim> fe_phase(params.fe.degree_phase);
        dealii::FE_Q<dim> fe_velocity(params.fe.degree_velocity);
        dealii::FE_Q<dim> fe_pressure(params.fe.degree_pressure);

        // ====== DoF Handlers ======
        dealii::DoFHandler<dim> theta_dof_handler(triangulation);
        dealii::DoFHandler<dim> psi_dof_handler(triangulation);
        dealii::DoFHandler<dim> ux_dof_handler(triangulation);
        dealii::DoFHandler<dim> uy_dof_handler(triangulation);
        dealii::DoFHandler<dim> p_dof_handler(triangulation);

        theta_dof_handler.distribute_dofs(fe_phase);
        psi_dof_handler.distribute_dofs(fe_phase);
        ux_dof_handler.distribute_dofs(fe_velocity);
        uy_dof_handler.distribute_dofs(fe_velocity);
        p_dof_handler.distribute_dofs(fe_pressure);

        const unsigned int n_theta = theta_dof_handler.n_dofs();
        const unsigned int n_psi = psi_dof_handler.n_dofs();
        const unsigned int n_ux = ux_dof_handler.n_dofs();
        const unsigned int n_uy = uy_dof_handler.n_dofs();
        const unsigned int n_p = p_dof_handler.n_dofs();
        const unsigned int n_ch = n_theta + n_psi;
        const unsigned int n_ns = n_ux + n_uy + n_p;

        // ====== Solution Vectors ======
        dealii::Vector<double> theta_solution(n_theta), theta_old(n_theta);
        dealii::Vector<double> psi_solution(n_psi);
        dealii::Vector<double> ux_solution(n_ux), ux_old(n_ux);
        dealii::Vector<double> uy_solution(n_uy), uy_old(n_uy);
        dealii::Vector<double> p_solution(n_p);

        // ====== Initial Conditions ======
        apply_ch_mms_initial_conditions(theta_dof_handler, psi_dof_handler,
                                        theta_solution, psi_solution, t_start);
        theta_old = theta_solution;

        NSExactVelocityX<dim> ux_exact(t_start, L_y);
        NSExactVelocityY<dim> uy_exact(t_start, L_y);
        dealii::VectorTools::interpolate(ux_dof_handler, ux_exact, ux_solution);
        dealii::VectorTools::interpolate(uy_dof_handler, uy_exact, uy_solution);
        ux_old = ux_solution;
        uy_old = uy_solution;

        // ====== Time Stepping ======
        double current_time = t_start;

        for (unsigned int step = 0; step < n_steps; ++step)
        {
            current_time += actual_dt;
            theta_old = theta_solution;
            ux_old = ux_solution;
            uy_old = uy_solution;

            // ------ CH Constraints ------
            dealii::AffineConstraints<double> theta_constraints, psi_constraints;
            apply_ch_mms_boundary_constraints(theta_dof_handler, psi_dof_handler,
                                              theta_constraints, psi_constraints,
                                              current_time);

            // ------ CH Setup ------
            std::vector<dealii::types::global_dof_index> theta_to_ch, psi_to_ch;
            dealii::AffineConstraints<double> ch_constraints;
            dealii::SparsityPattern ch_sparsity;

            setup_ch_coupled_system<dim>(
                theta_dof_handler, psi_dof_handler,
                theta_constraints, psi_constraints,
                theta_to_ch, psi_to_ch,
                ch_constraints, ch_sparsity, false);

            dealii::SparseMatrix<double> ch_matrix(ch_sparsity);
            dealii::Vector<double> ch_rhs(n_ch), ch_solution(n_ch);

            // ------ Solve CH with velocity from NS ------
            assemble_ch_system<dim>(
                theta_dof_handler, psi_dof_handler,
                theta_old, ux_old, uy_old,  // Use lagged velocity
                params, actual_dt, current_time,
                theta_to_ch, psi_to_ch,
                ch_matrix, ch_rhs);

            ch_constraints.condense(ch_matrix, ch_rhs);

            dealii::SparseDirectUMFPACK ch_solver;
            ch_solver.initialize(ch_matrix);
            ch_solver.vmult(ch_solution, ch_rhs);
            ch_constraints.distribute(ch_solution);

            for (unsigned int i = 0; i < n_theta; ++i)
                theta_solution[i] = ch_solution[theta_to_ch[i]];
            for (unsigned int i = 0; i < n_psi; ++i)
                psi_solution[i] = ch_solution[psi_to_ch[i]];

            // ------ NS Constraints ------
            dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
            setup_ns_mms_velocity_constraints(ux_dof_handler, uy_dof_handler,
                                              ux_constraints, uy_constraints);
            p_constraints.clear();
            dealii::DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);
            if (!p_constraints.is_constrained(0))
            {
                p_constraints.add_line(0);
                p_constraints.set_inhomogeneity(0, 0.0);
            }
            p_constraints.close();

            // ------ NS Setup ------
            std::vector<dealii::types::global_dof_index> ux_to_ns, uy_to_ns, p_to_ns;
            dealii::AffineConstraints<double> ns_constraints;
            dealii::SparsityPattern ns_sparsity;

            setup_ns_coupled_system<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                ux_constraints, uy_constraints, p_constraints,
                ux_to_ns, uy_to_ns, p_to_ns,
                ns_constraints, ns_sparsity, false);

            dealii::SparseMatrix<double> ns_matrix(ns_sparsity);
            dealii::Vector<double> ns_rhs(n_ns), ns_solution(n_ns);

            // ------ Solve NS with θ, ψ from CH ------
            assemble_ns_system<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                theta_dof_handler, psi_dof_handler,
                nullptr, nullptr,  // No phi/M for this test
                ux_old, uy_old, theta_old, psi_solution,
                nullptr, nullptr, nullptr,  // No magnetic fields
                params, actual_dt, current_time,
                ux_to_ns, uy_to_ns, p_to_ns,
                ns_constraints, ns_matrix, ns_rhs);

            dealii::SparseDirectUMFPACK ns_solver;
            ns_solver.initialize(ns_matrix);
            ns_solver.vmult(ns_solution, ns_rhs);
            ns_constraints.distribute(ns_solution);

            extract_ns_solutions_local(ns_solution, ux_to_ns, uy_to_ns, p_to_ns,
                                       ux_solution, uy_solution, p_solution);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(end_time - start_time).count();

        // ====== Compute Errors ======
        CHMMSErrors ch_errors = compute_ch_mms_errors(
            theta_dof_handler, psi_dof_handler,
            theta_solution, psi_solution, current_time);

        NSMMSError ns_errors = compute_ns_mms_error(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_solution, uy_solution, p_solution,
            current_time, L_y);

        // Store results
        result.refinements.push_back(ref);
        result.h_values.push_back(h);
        result.theta_L2.push_back(ch_errors.theta_L2);
        result.theta_H1.push_back(ch_errors.theta_H1);
        result.psi_L2.push_back(ch_errors.psi_L2);
        result.ux_L2.push_back(ns_errors.ux_L2);
        result.ux_H1.push_back(ns_errors.ux_H1);
        result.uy_L2.push_back(ns_errors.uy_L2);
        result.p_L2.push_back(ns_errors.p_L2);
        result.n_dofs.push_back(n_ch + n_ns);
        result.wall_times.push_back(wall_time);

        std::cout << "θ_L2=" << std::scientific << std::setprecision(2) << ch_errors.theta_L2
                  << ", ux_L2=" << ns_errors.ux_L2
                  << ", time=" << std::fixed << std::setprecision(1) << wall_time << "s\n";
    }

    result.compute_rates();
    return result;
}


// ============================================================================
// Main Dispatcher
// ============================================================================

MMSConvergenceResult run_mms_convergence_study(
    MMSLevel level,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps)
{
    switch (level)
    {
    case MMSLevel::CH_STANDALONE:
        return run_ch_standalone<2>(refinements, params, n_time_steps);

    case MMSLevel::POISSON_STANDALONE:
        return run_poisson_standalone<2>(refinements, params, n_time_steps);

    case MMSLevel::NS_STANDALONE:
        return run_ns_standalone<2>(refinements, params, n_time_steps);

    case MMSLevel::MAGNETIZATION_STANDALONE:
        return run_magnetization_standalone<2>(refinements, params, n_time_steps);

    case MMSLevel::POISSON_MAGNETIZATION:
        return run_poisson_magnetization<2>(refinements, params, n_time_steps);
    case MMSLevel::CH_NS_CAPILLARY:
        return run_ch_ns_capillary<2>(refinements, params, n_time_steps);

    default:
        {
            std::cerr << "[MMS] Level " << to_string(level)
                << " not yet implemented.\n";
            MMSConvergenceResult empty;
            empty.level = level;
            return empty;
        }
    }
}