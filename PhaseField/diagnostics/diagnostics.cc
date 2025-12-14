// ============================================================================
// diagnostics/diagnostics.cc - Simulation Diagnostics Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "diagnostics/diagnostics.h"
#include "physics/material_properties.h"
#include "physics/kelvin_force.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <iomanip>

// ============================================================================
// Sigmoid function H(x) = 1/(1 + e^(-x)) [Eq. 18]
// Local definition to avoid header dependency issues
// ============================================================================
namespace {
inline double sigmoid(double x)
{
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}
}  // anonymous namespace

// ============================================================================
// DiagnosticsLogger implementation
// ============================================================================

void DiagnosticsLogger::open(const std::string& filename,
                              bool ns_enabled,
                              bool magnetic_enabled)
{
    ns_enabled_ = ns_enabled;
    magnetic_enabled_ = magnetic_enabled;

    file_.open(filename);
    if (!file_.is_open())
    {
        throw std::runtime_error("Failed to open diagnostics file: " + filename);
    }

    // Write CSV header
    file_ << "step,time,mass,mass_raw,energy_ch,theta_min,theta_max";

    if (ns_enabled_)
    {
        file_ << ",u_max,div_u_L2,cfl";
    }

    file_ << ",F_cap_L2";

    if (magnetic_enabled_)
    {
        file_ << ",F_mag_L2,phi_min,phi_max";
    }

    if (ns_enabled_)
    {
        file_ << ",F_grav_L2";
    }

    file_ << "\n";
    file_.flush();
}

void DiagnosticsLogger::write(const DiagnosticData& data)
{
    if (!file_.is_open())
        return;

    file_ << std::setprecision(8);
    file_ << data.step << ","
          << data.time << ","
          << data.mass << ","
          << data.mass_raw << ","
          << data.energy_ch << ","
          << data.theta_min << ","
          << data.theta_max;

    if (ns_enabled_)
    {
        file_ << "," << data.u_max
              << "," << data.div_u_L2
              << "," << data.cfl;
    }

    file_ << "," << data.F_cap_L2;

    if (magnetic_enabled_)
    {
        file_ << "," << data.F_mag_L2
              << "," << data.phi_min
              << "," << data.phi_max;
    }

    if (ns_enabled_)
    {
        file_ << "," << data.F_grav_L2;
    }

    file_ << "\n";
    file_.flush();
}

void DiagnosticsLogger::close()
{
    if (file_.is_open())
        file_.close();
}

// ============================================================================
// Diagnostic computation functions
// ============================================================================

template <int dim>
double compute_ch_energy(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::FE_Q<dim>& fe,
    double epsilon)
{
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    double energy = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double grad_theta_sq = theta_gradients[q].norm_square();

            // Double-well potential: F(θ) = (1/4)(θ² - 1)²
            const double F_theta = 0.25 * std::pow(theta * theta - 1.0, 2);

            // Energy density: ε/2 |∇θ|² + (1/ε) F(θ)
            const double energy_density =
                0.5 * epsilon * grad_theta_sq +
                (1.0 / epsilon) * F_theta;

            energy += energy_density * fe_values.JxW(q);
        }
    }

    return energy;
}

void compute_theta_bounds(
    const dealii::Vector<double>& theta_solution,
    double& theta_min,
    double& theta_max)
{
    theta_min = std::numeric_limits<double>::max();
    theta_max = std::numeric_limits<double>::lowest();

    for (unsigned int i = 0; i < theta_solution.size(); ++i)
    {
        theta_min = std::min(theta_min, theta_solution[i]);
        theta_max = std::max(theta_max, theta_solution[i]);
    }
}

double compute_u_max(
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution)
{
    double u_max = 0.0;

    for (unsigned int i = 0; i < ux_solution.size(); ++i)
    {
        const double u_mag = std::sqrt(
            ux_solution[i] * ux_solution[i] +
            uy_solution[i] * uy_solution[i]);
        u_max = std::max(u_max, u_mag);
    }

    return u_max;
}

template <int dim>
double compute_div_u_L2(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::FE_Q<dim>& fe)
{
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    double div_u_L2_sq = 0.0;

    for (const auto& cell : ux_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(ux_solution, ux_gradients);
        fe_values.get_function_gradients(uy_solution, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // ∇·u = ∂ux/∂x + ∂uy/∂y
            const double div_u = ux_gradients[q][0] + uy_gradients[q][1];
            div_u_L2_sq += div_u * div_u * fe_values.JxW(q);
        }
    }

    return std::sqrt(div_u_L2_sq);
}

template <int dim>
void compute_force_magnitudes(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params,
    double time,
    double& F_cap_L2,
    double& F_mag_L2,
    double& F_grav_L2)
{
    // Suppress unused parameter warning (we use theta_dof_handler for iteration)
    (void)psi_dof_handler;

    const unsigned int degree = params.fe.degree_velocity;
    dealii::QGauss<dim> quadrature(degree + 1);
    dealii::FEValues<dim> fe_values(
        dealii::FE_Q<dim>(degree), quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    double F_cap_sq = 0.0;
    double F_mag_sq = 0.0;
    double F_grav_sq = 0.0;

    const double lambda = params.ch.lambda;
    const double epsilon = params.ch.epsilon;
    const double chi_0 = params.magnetization.chi_0;
    const double mu_0 = params.ns.mu_0;
    const double r = params.ns.r;

    // Gravity vector
    dealii::Tensor<1, dim> g;
    g[0] = 0.0;
    g[1] = -params.gravity.magnitude;

    // Suppress unused variable warning for r if NS disabled
    (void)r;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (phi_dof_handler != nullptr && phi_solution != nullptr)
        {
            fe_values.get_function_gradients(*phi_solution, phi_gradients);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];

            // Material properties
            const double H_val = sigmoid(theta / epsilon);

            // ================================================================
            // Capillary force: F_cap = (λ/ε) θ ∇ψ  (or just (λ/ε)∇ψ - TBD)
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            for (unsigned int d = 0; d < dim; ++d)
                F_cap[d] = (lambda / epsilon) * theta * psi_gradients[q][d];

            F_cap_sq += F_cap.norm_square() * fe_values.JxW(q);

            // ================================================================
            // Kelvin force: F_mag = μ₀ χ(θ) (H·∇)H where H = -∇φ
            // ================================================================
            if (phi_dof_handler != nullptr && phi_solution != nullptr)
            {
                dealii::Tensor<1, dim> H;
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = -phi_gradients[q][d];

                const double chi = chi_0 * H_val;

                // Simplified: F_mag ≈ μ₀ χ ∇(|H|²/2) = μ₀ χ (H·∇)H
                // For now, compute magnitude estimate: |F_mag| ~ μ₀ χ |H|²/L
                // More accurate would require Hessian of φ
                // Approximation: |F_mag| ~ μ₀ χ |H|² (order of magnitude)
                const double H_mag_sq = H.norm_square();
                const double F_mag_approx = mu_0 * chi * H_mag_sq;

                F_mag_sq += F_mag_approx * F_mag_approx * fe_values.JxW(q);
            }

            // ================================================================
            // Gravity/buoyancy: F_grav = (1 + r H(θ/ε)) g
            // ================================================================
            if (params.gravity.enabled)
            {
                const double density_factor = 1.0 + r * H_val;
                dealii::Tensor<1, dim> F_grav;
                for (unsigned int d = 0; d < dim; ++d)
                    F_grav[d] = density_factor * g[d];

                F_grav_sq += F_grav.norm_square() * fe_values.JxW(q);
            }
        }
    }

    F_cap_L2 = std::sqrt(F_cap_sq);
    F_mag_L2 = std::sqrt(F_mag_sq);
    F_grav_L2 = std::sqrt(F_grav_sq);
}

template <int dim>
DiagnosticData compute_all_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const dealii::Vector<double>* ux_solution,
    const dealii::Vector<double>* uy_solution,
    const dealii::FE_Q<dim>& fe_Q2,
    const Parameters& params,
    unsigned int step,
    double time,
    double dt,
    double h_min)
{
    // Suppress unused parameter warnings
    (void)psi_dof_handler;
    (void)uy_dof_handler;

    DiagnosticData data;
    data.step = step;
    data.time = time;

    const double epsilon = params.ch.epsilon;
    const unsigned int degree = params.fe.degree_velocity;

    // ========================================================================
    // Compute everything in one loop for efficiency
    // ========================================================================
    dealii::QGauss<dim> quadrature(degree + 1);
    dealii::FEValues<dim> fe_values(fe_Q2, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    double mass_raw = 0.0;
    double energy_ch = 0.0;
    double F_cap_sq = 0.0;
    double F_mag_sq = 0.0;
    double F_grav_sq = 0.0;
    double div_u_sq = 0.0;

    const double lambda = params.ch.lambda;
    const double chi_0 = params.magnetization.chi_0;
    const double mu_0 = params.ns.mu_0;
    const double r = params.ns.r;

    dealii::Tensor<1, dim> g;
    g[0] = 0.0;
    g[1] = -params.gravity.magnitude;

    const bool compute_ns = (ux_dof_handler != nullptr && ux_solution != nullptr);
    const bool compute_mag = (phi_dof_handler != nullptr && phi_solution != nullptr);

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);
        fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (compute_mag)
            fe_values.get_function_gradients(*phi_solution, phi_gradients);

        if (compute_ns)
        {
            fe_values.get_function_gradients(*ux_solution, ux_gradients);
            fe_values.get_function_gradients(*uy_solution, uy_gradients);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double JxW = fe_values.JxW(q);

            // Mass
            mass_raw += theta * JxW;

            // CH Energy
            const double grad_theta_sq = theta_gradients[q].norm_square();
            const double F_theta = 0.25 * std::pow(theta * theta - 1.0, 2);
            energy_ch += (0.5 * epsilon * grad_theta_sq +
                          (1.0 / epsilon) * F_theta) * JxW;

            // Material property
            const double H_val = sigmoid(theta / epsilon);

            // Capillary force
            dealii::Tensor<1, dim> F_cap;
            for (unsigned int d = 0; d < dim; ++d)
                F_cap[d] = (lambda / epsilon) * theta * psi_gradients[q][d];
            F_cap_sq += F_cap.norm_square() * JxW;

            // Magnetic force
            if (compute_mag)
            {
                dealii::Tensor<1, dim> H;
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = -phi_gradients[q][d];

                const double chi = chi_0 * H_val;
                const double H_mag_sq = H.norm_square();
                const double F_mag_approx = mu_0 * chi * H_mag_sq;
                F_mag_sq += F_mag_approx * F_mag_approx * JxW;
            }

            // Gravity force
            if (params.gravity.enabled)
            {
                const double density_factor = 1.0 + r * H_val;
                dealii::Tensor<1, dim> F_grav;
                for (unsigned int d = 0; d < dim; ++d)
                    F_grav[d] = density_factor * g[d];
                F_grav_sq += F_grav.norm_square() * JxW;
            }

            // Divergence
            if (compute_ns)
            {
                const double div_u = ux_gradients[q][0] + uy_gradients[q][1];
                div_u_sq += div_u * div_u * JxW;
            }
        }
    }

    // Store computed values
    data.mass_raw = mass_raw;

    // Domain area for normalized mass
    const double domain_area = (params.domain.x_max - params.domain.x_min) *
                               (params.domain.y_max - params.domain.y_min);
    data.mass = (mass_raw + domain_area) / (2.0 * domain_area);  // Normalized to [0,1]

    data.energy_ch = energy_ch;
    data.F_cap_L2 = std::sqrt(F_cap_sq);
    data.F_mag_L2 = std::sqrt(F_mag_sq);
    data.F_grav_L2 = std::sqrt(F_grav_sq);
    data.div_u_L2 = std::sqrt(div_u_sq);

    // Theta bounds (from nodal values)
    compute_theta_bounds(theta_solution, data.theta_min, data.theta_max);

    // Phi bounds
    if (compute_mag)
    {
        data.phi_min = *std::min_element(phi_solution->begin(), phi_solution->end());
        data.phi_max = *std::max_element(phi_solution->begin(), phi_solution->end());
    }

    // Velocity max and CFL
    if (compute_ns)
    {
        data.u_max = compute_u_max(*ux_solution, *uy_solution);
        data.cfl = data.u_max * dt / h_min;
    }

    return data;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template double compute_ch_energy<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::FE_Q<2>&,
    double);

template double compute_div_u_L2<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::FE_Q<2>&);

template void compute_force_magnitudes<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>*,
    const Parameters&,
    double,
    double&,
    double&,
    double&);

template DiagnosticData compute_all_diagnostics<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>*,
    const dealii::DoFHandler<2>*,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const dealii::FE_Q<2>&,
    const Parameters&,
    unsigned int,
    double,
    double,
    double);