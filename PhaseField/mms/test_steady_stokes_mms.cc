// ============================================================================
// mms/test_steady_stokes_mms.cc - NS MMS Verification Test (Phases A-D)
//
// Phase A: Steady Stokes - ν(T(U), T(V)) - (p, ∇·V) + (∇·U, q) = (f, V)
// Phase B: Unsteady Stokes - adds (U/τ, V) time derivative
// Phase C: Steady NS - adds B_h(U_old, U, V) convection
// Phase D: Unsteady NS - full equation matching ns_assembler.cc
//
// Expected convergence rates for Q2-Q1 Taylor-Hood:
//   - Velocity L2: ~3.0 (p+1 for Q2)
//   - Velocity H1: ~2.0 (p for Q2)
//   - Pressure L2: ~2.0 (p+1 for Q1)
//
// Build: cmake --build . --target test_navier_stokes
// Run:   ./test_navier_stokes [phase]
//        phase = A (steady Stokes), B (unsteady Stokes),
//                C (steady NS), D (unsteady NS)
//
// ============================================================================

#include "assembly/ns_mms_assembler.h"

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace dealii;

// ============================================================================
// MMS Exact Solution Classes
// ============================================================================

template <int dim>
class ExactVelocityX : public Function<dim>
{
public:
    ExactVelocityX(double time = 1.0, double L_y = 1.0)
        : Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        const double sin_px = std::sin(M_PI * x);
        return time_ * (M_PI / L_y_) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y_);
    }

    virtual Tensor<1, dim> gradient(const Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        const double sin_px = std::sin(M_PI * x);
        const double sin_2px = std::sin(2.0 * M_PI * x);
        const double sin_2py = std::sin(2.0 * M_PI * y / L_y_);
        const double cos_2py = std::cos(2.0 * M_PI * y / L_y_);

        Tensor<1, dim> grad;
        grad[0] = time_ * (M_PI * M_PI / L_y_) * sin_2px * sin_2py;
        grad[1] = time_ * (2.0 * M_PI * M_PI / (L_y_ * L_y_)) * sin_px * sin_px * cos_2py;
        return grad;
    }

private:
    double time_;
    double L_y_;
};

template <int dim>
class ExactVelocityY : public Function<dim>
{
public:
    ExactVelocityY(double time = 1.0, double L_y = 1.0)
        : Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        const double sin_py = std::sin(M_PI * y / L_y_);
        return -time_ * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;
    }

    virtual Tensor<1, dim> gradient(const Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        const double sin_py = std::sin(M_PI * y / L_y_);
        const double cos_2px = std::cos(2.0 * M_PI * x);
        const double sin_2px = std::sin(2.0 * M_PI * x);
        const double sin_2py = std::sin(2.0 * M_PI * y / L_y_);

        Tensor<1, dim> grad;
        grad[0] = -time_ * 2.0 * M_PI * M_PI * cos_2px * sin_py * sin_py;
        grad[1] = -time_ * (M_PI * M_PI / L_y_) * sin_2px * sin_2py;
        return grad;
    }

private:
    double time_;
    double L_y_;
};

template <int dim>
class ExactPressure : public Function<dim>
{
public:
    ExactPressure(double time = 1.0, double L_y = 1.0)
        : Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const Point<dim>& p, const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Setup coupled NS system (DoF numbering and constraints)
// ============================================================================

template <int dim>
void setup_ns_system(
    const DoFHandler<dim>& ux_dof_handler,
    const DoFHandler<dim>& uy_dof_handler,
    const DoFHandler<dim>& p_dof_handler,
    const AffineConstraints<double>& ux_constraints,
    const AffineConstraints<double>& uy_constraints,
    const AffineConstraints<double>& p_constraints,
    std::vector<types::global_dof_index>& ux_to_ns_map,
    std::vector<types::global_dof_index>& uy_to_ns_map,
    std::vector<types::global_dof_index>& p_to_ns_map,
    AffineConstraints<double>& ns_constraints,
    SparsityPattern& ns_sparsity)
{
    const unsigned int n_ux = ux_dof_handler.n_dofs();
    const unsigned int n_uy = uy_dof_handler.n_dofs();
    const unsigned int n_p = p_dof_handler.n_dofs();
    const unsigned int n_total = n_ux + n_uy + n_p;

    // Create index maps: [ux | uy | p]
    ux_to_ns_map.resize(n_ux);
    uy_to_ns_map.resize(n_uy);
    p_to_ns_map.resize(n_p);

    for (unsigned int i = 0; i < n_ux; ++i)
        ux_to_ns_map[i] = i;
    for (unsigned int i = 0; i < n_uy; ++i)
        uy_to_ns_map[i] = n_ux + i;
    for (unsigned int i = 0; i < n_p; ++i)
        p_to_ns_map[i] = n_ux + n_uy + i;

    // Merge constraints
    ns_constraints.clear();

    // Copy ux constraints
    for (const auto& line : ux_constraints.get_lines())
    {
        const auto global_index = ux_to_ns_map[line.index];
        ns_constraints.add_line(global_index);
        for (const auto& entry : line.entries)
            ns_constraints.add_entry(global_index, ux_to_ns_map[entry.first], entry.second);
        ns_constraints.set_inhomogeneity(global_index, line.inhomogeneity);
    }

    // Copy uy constraints
    for (const auto& line : uy_constraints.get_lines())
    {
        const auto global_index = uy_to_ns_map[line.index];
        ns_constraints.add_line(global_index);
        for (const auto& entry : line.entries)
            ns_constraints.add_entry(global_index, uy_to_ns_map[entry.first], entry.second);
        ns_constraints.set_inhomogeneity(global_index, line.inhomogeneity);
    }

    // Copy p constraints
    for (const auto& line : p_constraints.get_lines())
    {
        const auto global_index = p_to_ns_map[line.index];
        ns_constraints.add_line(global_index);
        for (const auto& entry : line.entries)
            ns_constraints.add_entry(global_index, p_to_ns_map[entry.first], entry.second);
        ns_constraints.set_inhomogeneity(global_index, line.inhomogeneity);
    }

    ns_constraints.close();

    // Build sparsity pattern
    DynamicSparsityPattern dsp(n_total, n_total);

    // Add couplings for each cell
    const auto& fe_Q2 = ux_dof_handler.get_fe();
    const auto& fe_Q1 = p_dof_handler.get_fe();

    std::vector<types::global_dof_index> ux_dofs(fe_Q2.n_dofs_per_cell());
    std::vector<types::global_dof_index> uy_dofs(fe_Q2.n_dofs_per_cell());
    std::vector<types::global_dof_index> p_dofs(fe_Q1.n_dofs_per_cell());

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_cell->get_dof_indices(ux_dofs);
        uy_cell->get_dof_indices(uy_dofs);
        p_cell->get_dof_indices(p_dofs);

        // All-to-all coupling within velocity and pressure blocks
        for (const auto i : ux_dofs)
            for (const auto j : ux_dofs)
                dsp.add(ux_to_ns_map[i], ux_to_ns_map[j]);

        for (const auto i : ux_dofs)
            for (const auto j : uy_dofs)
                dsp.add(ux_to_ns_map[i], uy_to_ns_map[j]);

        for (const auto i : uy_dofs)
            for (const auto j : ux_dofs)
                dsp.add(uy_to_ns_map[i], ux_to_ns_map[j]);

        for (const auto i : uy_dofs)
            for (const auto j : uy_dofs)
                dsp.add(uy_to_ns_map[i], uy_to_ns_map[j]);

        // Velocity-pressure coupling
        for (const auto i : ux_dofs)
            for (const auto j : p_dofs)
            {
                dsp.add(ux_to_ns_map[i], p_to_ns_map[j]);
                dsp.add(p_to_ns_map[j], ux_to_ns_map[i]);
            }

        for (const auto i : uy_dofs)
            for (const auto j : p_dofs)
            {
                dsp.add(uy_to_ns_map[i], p_to_ns_map[j]);
                dsp.add(p_to_ns_map[j], uy_to_ns_map[i]);
            }
    }

    ns_constraints.condense(dsp);
    ns_sparsity.copy_from(dsp);
}

// ============================================================================
// Compute errors
// ============================================================================

struct ErrorResult
{
    double ux_L2, ux_H1;
    double uy_L2, uy_H1;
    double p_L2;
    double div_U_L2;
};

template <int dim>
ErrorResult compute_errors(
    const DoFHandler<dim>& ux_dof_handler,
    const DoFHandler<dim>& uy_dof_handler,
    const DoFHandler<dim>& p_dof_handler,
    const Vector<double>& ux_solution,
    const Vector<double>& uy_solution,
    const Vector<double>& p_solution,
    double time,
    double L_y)
{
    ErrorResult error;

    const auto& fe_ux = ux_dof_handler.get_fe();
    const auto& fe_p = p_dof_handler.get_fe();

    QGauss<dim> quadrature(fe_ux.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> ux_fe_values(fe_ux, quadrature,
        update_values | update_gradients | update_quadrature_points | update_JxW_values);
    FEValues<dim> uy_fe_values(fe_ux, quadrature,
        update_values | update_gradients);
    FEValues<dim> p_fe_values(fe_p, quadrature,
        update_values);

    std::vector<double> ux_vals(n_q_points), uy_vals(n_q_points), p_vals(n_q_points);
    std::vector<Tensor<1, dim>> ux_grads(n_q_points), uy_grads(n_q_points);

    ExactVelocityX<dim> exact_ux(time, L_y);
    ExactVelocityY<dim> exact_uy(time, L_y);
    ExactPressure<dim> exact_p(time, L_y);

    double ux_L2_sq = 0, ux_H1_sq = 0;
    double uy_L2_sq = 0, uy_H1_sq = 0;
    double p_L2_sq = 0;
    double div_U_L2_sq = 0;

    // Compute mean pressures for normalization
    double p_mean_num = 0, p_mean_exact = 0, domain_area = 0;

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    // First pass: compute means
    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        p_fe_values.reinit(p_cell);
        p_fe_values.get_function_values(p_solution, p_vals);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            p_mean_num += p_vals[q] * JxW;
            p_mean_exact += exact_p.value(x_q) * JxW;
            domain_area += JxW;
        }
    }

    p_mean_num /= domain_area;
    p_mean_exact /= domain_area;

    // Second pass: compute errors
    ux_cell = ux_dof_handler.begin_active();
    uy_cell = uy_dof_handler.begin_active();
    p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        ux_fe_values.get_function_values(ux_solution, ux_vals);
        uy_fe_values.get_function_values(uy_solution, uy_vals);
        p_fe_values.get_function_values(p_solution, p_vals);
        ux_fe_values.get_function_gradients(ux_solution, ux_grads);
        uy_fe_values.get_function_gradients(uy_solution, uy_grads);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            const double ux_exact = exact_ux.value(x_q);
            const double uy_exact = exact_uy.value(x_q);
            const Tensor<1, dim> grad_ux_exact = exact_ux.gradient(x_q);
            const Tensor<1, dim> grad_uy_exact = exact_uy.gradient(x_q);

            const double ux_err = ux_vals[q] - ux_exact;
            const double uy_err = uy_vals[q] - uy_exact;
            const Tensor<1, dim> grad_ux_err = ux_grads[q] - grad_ux_exact;
            const Tensor<1, dim> grad_uy_err = uy_grads[q] - grad_uy_exact;

            ux_L2_sq += ux_err * ux_err * JxW;
            uy_L2_sq += uy_err * uy_err * JxW;
            ux_H1_sq += (grad_ux_err * grad_ux_err) * JxW;
            uy_H1_sq += (grad_uy_err * grad_uy_err) * JxW;

            // Pressure error (zero-mean adjusted)
            const double p_exact = exact_p.value(x_q);
            const double p_err = (p_vals[q] - p_mean_num) - (p_exact - p_mean_exact);
            p_L2_sq += p_err * p_err * JxW;

            // Divergence
            const double div_U = ux_grads[q][0] + uy_grads[q][1];
            div_U_L2_sq += div_U * div_U * JxW;
        }
    }

    error.ux_L2 = std::sqrt(ux_L2_sq);
    error.ux_H1 = std::sqrt(ux_H1_sq);
    error.uy_L2 = std::sqrt(uy_L2_sq);
    error.uy_H1 = std::sqrt(uy_H1_sq);
    error.p_L2 = std::sqrt(p_L2_sq);
    error.div_U_L2 = std::sqrt(div_U_L2_sq);

    return error;
}

// ============================================================================
// Main: Run convergence study
// ============================================================================

enum class Phase { A, B, C, D };

std::string phase_name(Phase p)
{
    switch (p) {
        case Phase::A: return "STEADY_STOKES";
        case Phase::B: return "UNSTEADY_STOKES";
        case Phase::C: return "STEADY_NS";
        case Phase::D: return "UNSTEADY_NS";
    }
    return "UNKNOWN";
}

int main(int argc, char* argv[])
{
    constexpr int dim = 2;

    // Parse phase from command line
    Phase phase = Phase::A;  // Default: steady Stokes
    if (argc > 1)
    {
        std::string arg = argv[1];
        if (arg == "A" || arg == "a") phase = Phase::A;
        else if (arg == "B" || arg == "b") phase = Phase::B;
        else if (arg == "C" || arg == "c") phase = Phase::C;
        else if (arg == "D" || arg == "d") phase = Phase::D;
        else
        {
            std::cerr << "Usage: " << argv[0] << " [A|B|C|D]\n";
            std::cerr << "  A = Steady Stokes\n";
            std::cerr << "  B = Unsteady Stokes\n";
            std::cerr << "  C = Steady NS\n";
            std::cerr << "  D = Unsteady NS\n";
            return 1;
        }
    }

    const bool include_time_derivative = (phase == Phase::B || phase == Phase::D);
    const bool include_convection = (phase == Phase::C || phase == Phase::D);

    const double nu = 1.0;
    const double L_y = 1.0;

    // Time parameters
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double time_steady = 1.0;  // For steady cases, evaluate at t=1

    const unsigned int velocity_degree = 2;  // Q2
    const unsigned int pressure_degree = 1;  // Q1

    std::vector<unsigned int> refinements = {3, 4, 5, 6};

    std::cout << "================================================================\n";
    std::cout << "  NS MMS VERIFICATION TEST - Phase " << static_cast<char>('A' + static_cast<int>(phase))
              << ": " << phase_name(phase) << "\n";
    std::cout << "================================================================\n";
    std::cout << "  nu = " << nu << ", L_y = " << L_y << "\n";
    if (include_time_derivative)
        std::cout << "  t in [" << t_start << ", " << t_end << "]\n";
    else
        std::cout << "  t = " << time_steady << " (steady)\n";
    std::cout << "  include_time_derivative = " << (include_time_derivative ? "true" : "false") << "\n";
    std::cout << "  include_convection = " << (include_convection ? "true" : "false") << "\n";
    std::cout << "  FE: Q" << velocity_degree << " velocity, Q" << pressure_degree << " pressure\n";
    std::cout << "================================================================\n\n";

    std::vector<double> h_values;
    std::vector<ErrorResult> errors;

    for (unsigned int ref : refinements)
    {
        std::cout << "  Refinement " << ref << "... " << std::flush;

        // Setup mesh
        Triangulation<dim> triangulation;
        GridGenerator::hyper_rectangle(triangulation,
            Point<dim>(0.0, 0.0), Point<dim>(1.0, L_y));
        triangulation.refine_global(ref);

        const double h = triangulation.begin_active()->diameter();
        h_values.push_back(h);

        // Time step scales with mesh size for stability
        const double dt = include_time_derivative ? 0.1 * h : 1.0;
        const unsigned int n_steps = include_time_derivative
            ? static_cast<unsigned int>(std::ceil((t_end - t_start) / dt))
            : 1;
        const double actual_dt = include_time_derivative ? (t_end - t_start) / n_steps : 1.0;

        // Setup FE
        FE_Q<dim> fe_velocity(velocity_degree);
        FE_Q<dim> fe_pressure(pressure_degree);

        DoFHandler<dim> ux_dof_handler(triangulation);
        DoFHandler<dim> uy_dof_handler(triangulation);
        DoFHandler<dim> p_dof_handler(triangulation);

        ux_dof_handler.distribute_dofs(fe_velocity);
        uy_dof_handler.distribute_dofs(fe_velocity);
        p_dof_handler.distribute_dofs(fe_pressure);

        const unsigned int n_ux = ux_dof_handler.n_dofs();
        const unsigned int n_uy = uy_dof_handler.n_dofs();
        const unsigned int n_p = p_dof_handler.n_dofs();
        const unsigned int n_total = n_ux + n_uy + n_p;

        // Setup constraints
        AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;

        ux_constraints.clear();
        uy_constraints.clear();
        DoFTools::make_hanging_node_constraints(ux_dof_handler, ux_constraints);
        DoFTools::make_hanging_node_constraints(uy_dof_handler, uy_constraints);

        // Apply homogeneous Dirichlet (exact solution is zero on boundaries)
        std::map<types::global_dof_index, double> ux_boundary_values;
        std::map<types::global_dof_index, double> uy_boundary_values;

        VectorTools::interpolate_boundary_values(ux_dof_handler, 0,
            Functions::ZeroFunction<dim>(), ux_boundary_values);
        VectorTools::interpolate_boundary_values(ux_dof_handler, 1,
            Functions::ZeroFunction<dim>(), ux_boundary_values);
        VectorTools::interpolate_boundary_values(uy_dof_handler, 0,
            Functions::ZeroFunction<dim>(), uy_boundary_values);
        VectorTools::interpolate_boundary_values(uy_dof_handler, 1,
            Functions::ZeroFunction<dim>(), uy_boundary_values);

        for (const auto& [dof, value] : ux_boundary_values)
            if (!ux_constraints.is_constrained(dof))
            {
                ux_constraints.add_line(dof);
                ux_constraints.set_inhomogeneity(dof, value);
            }

        for (const auto& [dof, value] : uy_boundary_values)
            if (!uy_constraints.is_constrained(dof))
            {
                uy_constraints.add_line(dof);
                uy_constraints.set_inhomogeneity(dof, value);
            }

        ux_constraints.close();
        uy_constraints.close();

        // Pressure: pin one DoF to zero
        p_constraints.clear();
        DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);
        if (!p_constraints.is_constrained(0))
        {
            p_constraints.add_line(0);
            p_constraints.set_inhomogeneity(0, 0.0);
        }
        p_constraints.close();

        // Setup coupled system
        std::vector<types::global_dof_index> ux_to_ns_map, uy_to_ns_map, p_to_ns_map;
        AffineConstraints<double> ns_constraints;
        SparsityPattern ns_sparsity;

        setup_ns_system(ux_dof_handler, uy_dof_handler, p_dof_handler,
                        ux_constraints, uy_constraints, p_constraints,
                        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                        ns_constraints, ns_sparsity);

        // Allocate system
        SparseMatrix<double> ns_matrix(ns_sparsity);
        Vector<double> ns_rhs(n_total);
        Vector<double> ns_solution(n_total);

        // Solution vectors
        Vector<double> ux_solution(n_ux), uy_solution(n_uy), p_solution(n_p);
        Vector<double> ux_old(n_ux), uy_old(n_uy);

        // Initialize with exact solution at start time
        double current_time = include_time_derivative ? t_start : time_steady;
        ExactVelocityX<dim> exact_ux_init(current_time, L_y);
        ExactVelocityY<dim> exact_uy_init(current_time, L_y);
        VectorTools::interpolate(ux_dof_handler, exact_ux_init, ux_old);
        VectorTools::interpolate(uy_dof_handler, exact_uy_init, uy_old);

        // Time stepping loop
        for (unsigned int step = 0; step < n_steps; ++step)
        {
            if (include_time_derivative)
                current_time += actual_dt;

            // Assemble
            assemble_ns_mms_system<dim>(
                ux_dof_handler, uy_dof_handler, p_dof_handler,
                ux_old, uy_old,
                nu,
                actual_dt,
                current_time,
                L_y,
                include_time_derivative,
                include_convection,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_constraints,
                ns_matrix, ns_rhs);

            // Solve
            SparseDirectUMFPACK solver;
            solver.initialize(ns_matrix);
            solver.vmult(ns_solution, ns_rhs);

            ns_constraints.distribute(ns_solution);

            // Extract individual solutions
            for (unsigned int i = 0; i < n_ux; ++i)
                ux_solution[i] = ns_solution[ux_to_ns_map[i]];
            for (unsigned int i = 0; i < n_uy; ++i)
                uy_solution[i] = ns_solution[uy_to_ns_map[i]];
            for (unsigned int i = 0; i < n_p; ++i)
                p_solution[i] = ns_solution[p_to_ns_map[i]];

            // Update old solution for next step
            ux_old = ux_solution;
            uy_old = uy_solution;
        }

        // Compute errors at final time
        ErrorResult err = compute_errors(
            ux_dof_handler, uy_dof_handler, p_dof_handler,
            ux_solution, uy_solution, p_solution,
            current_time, L_y);

        errors.push_back(err);

        std::cout << "h=" << std::scientific << std::setprecision(2) << h;
        if (include_time_derivative)
            std::cout << ", dt=" << actual_dt << ", steps=" << n_steps;
        std::cout << ", ux_L2=" << err.ux_L2 << ", p_L2=" << err.p_L2
                  << ", div_U=" << err.div_U_L2 << "\n";
    }

    // Print convergence table
    std::cout << "\n========================================\n";
    std::cout << "Convergence Results: " << phase_name(phase) << "\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(6) << "Ref"
              << std::setw(12) << "h"
              << std::setw(12) << "ux_L2" << std::setw(8) << "rate"
              << std::setw(12) << "ux_H1" << std::setw(8) << "rate"
              << std::setw(12) << "p_L2" << std::setw(8) << "rate"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < refinements.size(); ++i)
    {
        double ux_L2_rate = 0, ux_H1_rate = 0, p_L2_rate = 0;
        if (i > 0)
        {
            ux_L2_rate = std::log(errors[i-1].ux_L2 / errors[i].ux_L2) /
                         std::log(h_values[i-1] / h_values[i]);
            ux_H1_rate = std::log(errors[i-1].ux_H1 / errors[i].ux_H1) /
                         std::log(h_values[i-1] / h_values[i]);
            p_L2_rate = std::log(errors[i-1].p_L2 / errors[i].p_L2) /
                        std::log(h_values[i-1] / h_values[i]);
        }

        std::cout << std::left << std::setw(6) << refinements[i]
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << h_values[i]
                  << std::setw(12) << errors[i].ux_L2
                  << std::fixed << std::setprecision(2) << std::setw(8) << ux_L2_rate
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << errors[i].ux_H1
                  << std::fixed << std::setprecision(2) << std::setw(8) << ux_H1_rate
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << errors[i].p_L2
                  << std::fixed << std::setprecision(2) << std::setw(8) << p_L2_rate
                  << "\n";
    }

    std::cout << "========================================\n";

    // Check expected rates
    bool pass = true;
    const double tol = 0.3;

    if (errors.size() >= 2)
    {
        const size_t last = errors.size() - 1;
        double ux_L2_rate = std::log(errors[last-1].ux_L2 / errors[last].ux_L2) /
                            std::log(h_values[last-1] / h_values[last]);
        double ux_H1_rate = std::log(errors[last-1].ux_H1 / errors[last].ux_H1) /
                            std::log(h_values[last-1] / h_values[last]);
        double p_L2_rate = std::log(errors[last-1].p_L2 / errors[last].p_L2) /
                           std::log(h_values[last-1] / h_values[last]);

        // For unsteady problems, we expect first-order in time, so rates may be limited
        double expected_L2 = include_time_derivative ? 2.0 : 3.0;  // First-order time limits spatial
        double expected_H1 = 2.0;
        double expected_p = 2.0;

        if (ux_L2_rate < expected_L2 - tol) { std::cout << "[FAIL] ux_L2 rate = " << ux_L2_rate << " < " << expected_L2 - tol << "\n"; pass = false; }
        if (ux_H1_rate < expected_H1 - tol) { std::cout << "[FAIL] ux_H1 rate = " << ux_H1_rate << " < " << expected_H1 - tol << "\n"; pass = false; }
        if (p_L2_rate < expected_p - tol) { std::cout << "[FAIL] p_L2 rate = " << p_L2_rate << " < " << expected_p - tol << "\n"; pass = false; }
    }

    if (pass)
        std::cout << "\n[PASS] All convergence rates within expected bounds!\n";
    else
        std::cout << "\n[FAIL] Some convergence rates below expected.\n";

    std::cout << "================================================================\n";

    return pass ? 0 : 1;
}