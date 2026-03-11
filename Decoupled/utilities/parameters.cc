// ============================================================================
// utilities/parameters.cc - Runtime Configuration
//
// Includes presets and CLI options for all subsystems:
//   Poisson, Magnetization, Cahn-Hilliard, Navier-Stokes
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Zhang, He & Yang, CMAME 371 (2020) — β-term extension
// ============================================================================

#include "utilities/parameters.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>

// ============================================================================
// Rosensweig preset (Zhang, He & Yang, SIAM J. Sci. Comput. 43 (2021),
// Section 4.3, Eq 4.4, Fig 4.7-4.9)
//
// Flat ferrofluid interface, applied H field from 5 dipoles below domain.
// Spikes form via Rosensweig instability.
//
// Domain: [0,1] × [0,0.6]
// IC: Φ = 1 for y ≤ 0.2, Φ = 0 for y > 0.2 (Heaviside at y=0.2)
// ε = 5e-3, M = 2e-4, β = 1, λ = 1, τ = 1e-4
// ν_f = 2, ν_w = 1, μ₀ = 1, χ₀ = 0.5, r = 0.1
// g = (0, -6e4), h = 1e-2, δt = 1e-3
// Applied field: 5 dipoles at y=-15, α ramps 0→8000 over t∈[0,1.6], slope=5000
// ============================================================================
void Parameters::setup_rosensweig()
{
    // Domain — Zhang Section 4.3: Ω = [0,1] × [0,0.6]
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    // Zhang Eq 4.4: h = 1e-2 → 100×60 square cells
    domain.initial_cells_x = 100;
    domain.initial_cells_y = 60;

    // Physics — Zhang Eq 4.4 (SIAM J. Sci. Comput. 43(1), 2021, p.B186)
    physics.epsilon = 5e-3;        // ε = 5e-3
    physics.chi_0 = 0.5;          // χ₀ = 0.5
    physics.mu_0 = 1.0;           // μ₀ = 1

    // Physics — Magnetization (Zhang Eq 4.4: full PDE with β=1, τ=1e-4)
    physics.tau_M = 1e-4;         // Zhang Eq 4.4: τ = 1e-4
    physics.beta = 1.0;           // Zhang Eq 4.4: β = 1 (Landau-Lifshitz damping)
    physics.enable_beta_term = true;

    // Physics — Cahn-Hilliard
    // Zhang Eq 4.4 uses Φ∈{0,1}. Our code uses θ∈{-1,+1} with θ=2Φ-1.
    // double_well F(θ)=(θ²-1)²/16 already has the Φ→θ conversion baked in.
    // λ_θ = λ_Φ/4, γ_θ = 4·M_Φ, ch_reaction_scale = 1.0 (default).
    physics.lambda = 0.25;            // Zhang λ=1 → θ-space: λ_Φ/4
    physics.mobility = 8e-4;          // Zhang M=2e-4 → θ-space: 4×M_Φ
    // ch_reaction_scale = 1.0 (default) — F'(θ) already includes 1/4 factor

    // Physics — Navier-Stokes
    physics.nu_water = 1.0;       // Zhang Eq 4.4: ν_w = 1
    physics.nu_ferro = 2.0;       // Zhang Eq 4.4: ν_f = 2
    physics.r = 0.1;              // Zhang Eq 4.4: r = 0.1 (density ratio)
    physics.gravity_magnitude = 6e4;    // Zhang Eq 4.4: g = (0, -6e4)
    physics.gravity_direction = {0.0, -1.0};

    // Applied field: 5 dipoles far below domain (y = -15)
    uniform_field.enabled = false;
    dipoles.positions.clear();
    dipoles.positions.push_back(dealii::Point<2>(-0.5, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 0.0, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 0.5, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 1.0, -15.0));
    dipoles.positions.push_back(dealii::Point<2>( 1.5, -15.0));
    dipoles.direction    = {0.0, 1.0};
    dipoles.ramp_slope   = 5000.0;   // Zhang Eq 4.4: α ramps 0→8000 over [0,1.6]
    dipoles.ramp_time    = 0.0;
    dipoles.intensity_max = 8000.0;  // Zhang Eq 4.4: cap at α_s = 8000 after t=1.6

    // Time stepping — Zhang Eq 4.4: δt = 1e-3
    time.dt = 1e-3;
    time.t_final = 2.0;
    time.max_steps = 2000;

    // Mesh — Zhang Eq 4.4: h = 1e-2, no global refinement needed
    mesh.initial_refinement = 0;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;
    use_reduced_magnetic_field = false;

    // Zhang's SAV scheme — use FULL magnetization PDE (not algebraic)
    use_algebraic_magnetization = false;  // Zhang solves mag PDE (Eq 3.15-3.16)
    sav.S1 = 0.0;   // auto-computed: S = lambda_theta/(4*epsilon) = 12.5
}

// ============================================================================
// Rosensweig instability under NONUNIFORM applied magnetic field
// (Zhang, He & Yang, SIAM J. Sci. Comput. 43 (2021), Section 4.4,
//  Fig 4.10-4.13)
//
// Same as Section 4.3 but with:
//   - 42 dipoles (3 rows × 14) approximating a bar magnet (0.4 × 0.5)
//   - χ₀ = 0.9, h = 1/120, δt = 2e-4
//   - IC: flat interface at y = 0.1 (not 0.2)
//   - α ramps with slope 1.2 (much slower than Section 4.3's 5000)
//   - Figures shown up to t = 3.5
// ============================================================================
void Parameters::setup_hedgehog()
{
    // Zhang Section 4.4: Rosensweig instability under nonuniformly applied
    // magnetic field. 42 dipoles arranged as bar magnet → hedgehog pattern.
    // Start from uniform Rosensweig as base
    setup_rosensweig();

    // Override parameters per Section 4.4
    physics.chi_0 = 0.9;          // Paper: "with χ₀ = 0.9"

    // IC: flat interface at y = 0.1 (Paper Eq 4.5)
    flat_interface_y = 0.1;

    // Time stepping: δt = 2e-4 (5× smaller than Section 4.3)
    time.dt = 2e-4;
    time.t_final = 4.0;
    time.max_steps = 20000;

    // Mesh: h = 1/120 (Paper: "h = 1/120") → 120×72 square cells
    domain.initial_cells_x = 120;
    domain.initial_cells_y = 72;
    mesh.initial_refinement = 0;

    // 42 dipoles: 3 rows at y = -0.5, -0.75, -1.0
    // 14 per row, equidistributed in x over [0.3, 0.7] (bar magnet width 0.4)
    // Paper: "The intention of this setup of the dipoles is to create a crude
    //         approximation of a bar magnet of 0.4 units width and 0.5 units height."
    dipoles.positions.clear();
    const double y_rows[3] = {-0.5, -0.75, -1.0};
    const int n_per_row = 14;
    const double x_start = 0.3;   // centered on domain, width = 0.4
    const double x_end   = 0.7;
    for (int row = 0; row < 3; ++row)
    {
        for (int j = 0; j < n_per_row; ++j)
        {
            double x = x_start + j * (x_end - x_start) / (n_per_row - 1);
            dipoles.positions.push_back(dealii::Point<2>(x, y_rows[row]));
        }
    }
    dipoles.direction    = {0.0, 1.0};
    dipoles.ramp_slope   = 1.2;       // Paper: "slope of 1.2"
    dipoles.ramp_time    = 0.0;
    dipoles.intensity_max = 0.0;       // no cap mentioned

}

void Parameters::setup_dome()
{
    // Dome test: same dipole setup as hedgehog, but uses H = h_a only
    // (no demagnetizing field from Poisson). Without demagnetizing feedback,
    // the interface forms a smooth dome instead of spikes.
    // Matches Nochetto et al. Fig. 7 / Semi_Coupled dome preset.
    setup_hedgehog();

    // Key difference: H = h_a only (skip Poisson for demagnetizing field)
    use_reduced_magnetic_field = true;
}

// ============================================================================
// Command line parsing
// ============================================================================
Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // ---- Run mode (standalone subsystem drivers only) ----
        if (std::strcmp(argv[i], "--mode") == 0)
        {
            if (++i >= argc) { std::cerr << "--mode requires a value\n"; std::exit(1); }
            params.run.mode = argv[i];
        }
        else if (std::strcmp(argv[i], "--ref") == 0)
        {
            params.run.refs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
                params.run.refs.push_back(std::stoul(argv[++i]));
            if (params.run.refs.empty()) { std::cerr << "--ref requires at least one value\n"; std::exit(1); }
        }
        else if (std::strcmp(argv[i], "--steps") == 0)
        {
            if (++i >= argc) { std::cerr << "--steps requires a value\n"; std::exit(1); }
            params.run.steps = std::stoi(argv[i]);
        }

        // ---- Presets (all physics hardcoded inside) ----
        else if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();
        else if (std::strcmp(argv[i], "--hedgehog") == 0)
            params.setup_hedgehog();
        else if (std::strcmp(argv[i], "--dome") == 0)
            params.setup_dome();

        // ---- Mesh ----
        else if (std::strcmp(argv[i], "--refinement") == 0 ||
                 std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            params.mesh.initial_refinement = std::stoul(argv[i]);
        }

        // ---- Time stepping (override preset) ----
        else if (std::strcmp(argv[i], "--dt") == 0)
        {
            if (++i >= argc) { std::cerr << "--dt requires a value\n"; std::exit(1); }
            params.time.dt = std::stod(argv[i]);
        }

        // ---- Solver overrides ----
        else if (std::strcmp(argv[i], "--all-direct") == 0)
        {
            params.solvers.poisson.use_iterative = false;
            params.solvers.ns.use_iterative = false;
        }
        else if (std::strcmp(argv[i], "--poisson-direct") == 0)
            params.solvers.poisson.use_iterative = false;
        else if (std::strcmp(argv[i], "--ns-direct") == 0)
            params.solvers.ns.use_iterative = false;

        // ---- Subsystem enables ----
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.enable_mms = true;
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.enable_magnetic = false;
        else if (std::strcmp(argv[i], "--reduced_field") == 0)
            params.use_reduced_magnetic_field = true;
        else if (std::strcmp(argv[i], "--no_ns") == 0)
            params.enable_ns = false;
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.enable_gravity = false;
        else if (std::strcmp(argv[i], "--uniform-field") == 0)
        {
            // Override dipole field with truly uniform h_a.
            // Use AFTER a preset (e.g., --rosensweig --uniform-field).
            // Copies ramp/intensity from dipoles config, then disables dipoles.
            params.uniform_field.enabled = true;
            params.uniform_field.direction = {0.0, 1.0};  // vertical
            params.uniform_field.intensity_max = params.dipoles.intensity_max;
            params.uniform_field.ramp_slope    = params.dipoles.ramp_slope;
            params.uniform_field.ramp_time     = params.dipoles.ramp_time;
            params.dipoles.positions.clear();  // disable dipoles
        }

        // ---- Validation presets (Zhang, He & Yang, SIAM J. Sci. Comput. 43 (2021)) ----
        else if (std::strcmp(argv[i], "--validation") == 0)
        {
            if (++i >= argc) { std::cerr << "--validation requires a value (square|droplet)\n"; std::exit(1); }
            params.validation_test = argv[i];

            if (params.validation_test == "square")
            {
                // ============================================================
                // Square test (Zhang, He & Yang, SIAM J. Sci. Comput. 43 (2021),
                // Section 4.2, Fig 4.3):
                //
                // Diamond → circle relaxation. All fields active but zero
                // applied field (h_a=0), zero initial velocity, zero initial M.
                // Same parameters as MMS test (Eq 4.1).
                //
                // Domain: [0, 2π]², h = 2π/64 (r=6 on 1-cell base grid)
                // IC: diamond at (π,π), R=1.75, θ = 0.5-0.5*tanh((d-R)/(1.2ε))
                // ε = M = 0.05, ν_f = 2, ν_w = 1
                // λ = μ = τ = β = χ₀ = 1
                // dt = 1e-3, t_final = 5.0
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 2.0 * M_PI;
                params.domain.y_min = 0.0;
                params.domain.y_max = 2.0 * M_PI;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = true;   // all fields ON
                params.enable_ns       = true;
                params.enable_gravity  = false;  // no gravity term in MMS params

                // No applied field — h_a = 0
                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.intensity_max = 0.0;
                params.dipoles.ramp_slope = 0.0;

                // Zhang Eq 4.1 / Section 4.2: same params as MMS
                // Zhang uses Φ∈{0,1}. Convert: λ_θ = λ_Φ/4, M_θ = 4·M_Φ.
                params.physics.epsilon  = 0.05;    // ε = 0.05
                params.physics.lambda   = 0.25;    // Zhang λ=1 → θ-space: 1/4
                params.physics.mobility = 0.2;     // Zhang M=0.05 → θ-space: 4×
                params.physics.mu_0     = 1.0;     // μ = 1
                params.physics.chi_0    = 1.0;     // χ₀ = 1
                params.physics.tau_M    = 1.0;     // τ = 1
                params.physics.beta     = 1.0;     // β = 1
                params.physics.enable_beta_term = true;
                params.physics.nu_water = 1.0;     // ν_w = 1
                params.physics.nu_ferro = 2.0;     // ν_f = 2
                params.physics.r        = 0.0;     // no density difference
                params.physics.gravity_magnitude = 0.0;

                params.time.dt        = 1e-3;
                params.time.t_final   = 5.0;
                params.time.max_steps = 5000;

                params.mesh.initial_refinement = 6;  // h = 2π/64 ≈ 0.098

                params.use_algebraic_magnetization = false;
                params.sav.S1 = 0.0;   // auto-computed: S = lambda_theta/(4*epsilon) = 1.25
            }
            else if (params.validation_test == "droplet")
            {
                // ============================================================
                // Droplet deformation test (Zhang, He & Yang, SIAM J. Sci.
                // Comput. 43 (2021), Section 4.5, Eq 4.8, Fig 4.14-4.16):
                //
                // Circle in applied field → elongates vertically.
                //
                // Domain: [0,1]², h = 1/128 (r=7)
                // IC: circle at center, R = 0.1
                // ε = 2e-3, M_Φ = 2e-4 (θ-space: 8e-4), β = 1, λ_Φ = 1 (θ-space: 0.25), τ = 1e-4
                // ν_f = ν_w = 1, r = 0 (uniform density), no gravity
                // μ₀ = 0.1, χ₀ = 2
                // Applied field: 5 dipoles at y=-15, ramp slope=1000 (no cap)
                // dt = 1e-3, t_final = 1.5
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 1.0;
                params.domain.y_min = 0.0;
                params.domain.y_max = 1.0;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = true;
                params.enable_ns       = true;
                params.enable_gravity  = false;

                // Applied field: 5 dipoles far below domain (y = -15)
                // This produces a nearly uniform vertical field inside [0,1]²
                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.positions.push_back(dealii::Point<2>(-0.5, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 0.0, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 0.5, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 1.0, -15.0));
                params.dipoles.positions.push_back(dealii::Point<2>( 1.5, -15.0));
                params.dipoles.direction    = {0.0, 1.0};
                params.dipoles.ramp_slope   = 1000.0;   // Zhang Eq 4.8: "slope of 1000"
                params.dipoles.ramp_time    = 0.0;
                params.dipoles.intensity_max = 0.0;      // no cap — Zhang doesn't specify one

                params.physics.epsilon  = 2e-3;    // Zhang Eq 4.8
                params.physics.chi_0    = 2.0;     // Zhang Eq 4.8
                params.physics.mu_0     = 0.1;     // Zhang Eq 4.8: μ = 0.1
                params.physics.tau_M    = 1e-4;    // Zhang Eq 4.8: τ = 1e-4
                params.physics.beta     = 1.0;     // Zhang Eq 4.8: β = 1
                params.physics.enable_beta_term = true;
                // Zhang Eq 4.8 specifies λ=1, M=2e-4 in Φ∈{0,1} convention.
                // Code uses θ∈{-1,+1}: F(θ)=¼(θ²-1)² gives 4× surface energy
                // vs G(Φ)=Φ²(1-Φ)². Convert: λ_θ = λ_Φ/4, M_θ = 4·M_Φ.
                params.physics.lambda   = 0.25;    // Zhang λ=1 → θ-space: 1/4
                params.physics.mobility = 8e-4;    // Zhang M=2e-4 → θ-space: 4×
                params.physics.nu_water = 1.0;     // Zhang Eq 4.8: ν_w = 1
                params.physics.nu_ferro = 1.0;     // Zhang Eq 4.8: ν_f = 1
                params.physics.r        = 0.0;     // uniform density
                params.physics.gravity_magnitude = 0.0;

                params.time.dt        = 1e-3;      // Zhang Eq 4.8: δt = 1e-3
                params.time.t_final   = 1.5;
                params.time.max_steps = 1500;

                params.mesh.initial_refinement = 7;  // h = 1/128

                // Zhang's SAV scheme — use FULL magnetization PDE
                params.use_algebraic_magnetization = false;
                params.sav.S1 = 0.0;   // auto-computed: S = lambda_theta/(4*epsilon) = 31.25
            }
            else if (params.validation_test == "droplet_nofield")
            {
                // ============================================================
                // Droplet baseline test — NO magnetic field.
                // Same as droplet (Zhang Eq 4.8) but magnetic OFF.
                // Circle should remain circular (pure CH + NS relaxation).
                // ============================================================
                params.domain.x_min = 0.0;
                params.domain.x_max = 1.0;
                params.domain.y_min = 0.0;
                params.domain.y_max = 1.0;
                params.domain.initial_cells_x = 1;
                params.domain.initial_cells_y = 1;

                params.enable_magnetic = false;
                params.enable_ns       = true;
                params.enable_gravity  = false;

                params.uniform_field.enabled = false;
                params.dipoles.positions.clear();
                params.dipoles.intensity_max = 0.0;

                params.physics.epsilon  = 2e-3;
                params.physics.chi_0    = 2.0;
                params.physics.mu_0     = 0.1;
                params.physics.tau_M    = 1e-6;
                // Zhang Φ∈{0,1} → θ-space: λ_θ = λ_Φ/4, M_θ = 4·M_Φ
                params.physics.lambda   = 0.25;    // Zhang λ=1 → θ-space: 1/4
                params.physics.mobility = 8e-4;    // Zhang M=2e-4 → θ-space: 4×
                params.physics.nu_water = 1.0;
                params.physics.nu_ferro = 1.0;
                params.physics.r        = 0.0;
                params.physics.gravity_magnitude = 0.0;

                params.time.dt        = 1e-3;
                params.time.t_final   = 1.5;
                params.time.max_steps = 1500;

                params.mesh.initial_refinement = 7;

                params.use_algebraic_magnetization = true;
                params.sav.S1 = 0.0;   // auto-computed: S = lambda_theta/(4*epsilon) = 31.25
            }
            else
            {
                std::cerr << "Unknown validation test: " << params.validation_test
                          << " (use 'square', 'droplet', or 'droplet_nofield')\n";
                std::exit(1);
            }
        }

        // ---- AMR ----
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no-amr") == 0 || std::strcmp(argv[i], "--no_amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr-interval") == 0 || std::strcmp(argv[i], "--amr_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-interval requires a value\n"; std::exit(1); }
            params.mesh.amr_interval = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-max-level") == 0 || std::strcmp(argv[i], "--amr_max_level") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-max-level requires a value\n"; std::exit(1); }
            params.mesh.amr_max_level = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-min-level") == 0 || std::strcmp(argv[i], "--amr_min_level") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-min-level requires a value\n"; std::exit(1); }
            params.mesh.amr_min_level = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-upper-fraction") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-upper-fraction requires a value\n"; std::exit(1); }
            params.mesh.amr_upper_fraction = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-lower-fraction") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-lower-fraction requires a value\n"; std::exit(1); }
            params.mesh.amr_lower_fraction = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-refine-threshold") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-refine-threshold requires a value\n"; std::exit(1); }
            params.mesh.amr_refine_threshold = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--amr-activation-U") == 0 || std::strcmp(argv[i], "--amr_activation_U") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr-activation-U requires a value\n"; std::exit(1); }
            params.mesh.amr_activation_U = std::stod(argv[i]);
        }

        // ---- Output ----
        else if (std::strcmp(argv[i], "--vtk_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--vtk_interval requires a value\n"; std::exit(1); }
            params.output.vtk_interval = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--verbose") == 0 ||
                 std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // ---- Sparsity / Renumbering ----
        else if (std::strcmp(argv[i], "--renumber-dofs") == 0)
            params.renumber_dofs = true;
        else if (std::strcmp(argv[i], "--no-renumber-dofs") == 0)
            params.renumber_dofs = false;
        else if (std::strcmp(argv[i], "--dump-sparsity") == 0)
            params.dump_sparsity = true;

        // ---- Help ----
        else if (std::strcmp(argv[i], "--help") == 0 ||
                 std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Decoupled Ferrofluid Solver (Zhang, He & Yang, SIAM J. Sci. Comput. 43, 2021)\n";
            std::cout << "Algorithm 3.1: CH → NS → Pressure → Velocity → Magnetization+Poisson\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "  Presets (all physics hardcoded per paper section):\n";
            std::cout << "    --rosensweig       Uniform field, 5 dipoles (Section 4.3)\n";
            std::cout << "    --hedgehog         Nonuniform field, 42 dipoles (Section 4.4)\n";
            std::cout << "    --dome             Hedgehog with H=h_a only (no Poisson)\n";
            std::cout << "    --validation MODE  Validation test (square|droplet|droplet_nofield)\n\n";
            std::cout << "  Overrides (applied after preset):\n";
            std::cout << "    -r, --refinement N  Mesh refinement level\n";
            std::cout << "    --dt VALUE          Time step size\n\n";
            std::cout << "  Subsystem toggles:\n";
            std::cout << "    --mms              MMS verification mode\n";
            std::cout << "    --no_magnetic      Disable magnetic subsystem\n";
            std::cout << "    --reduced_field    H = h_a only (dome setup)\n";
            std::cout << "    --no_ns            Disable Navier-Stokes\n";
            std::cout << "    --no_gravity       Disable gravity body force\n\n";
            std::cout << "  Solvers:\n";
            std::cout << "    --all-direct       Force direct solvers for ALL subsystems\n";
            std::cout << "    --poisson-direct   Force direct solver for Poisson\n";
            std::cout << "    --ns-direct        Force direct solver for NS\n\n";
            std::cout << "  AMR:\n";
            std::cout << "    --amr / --no-amr   Enable/disable adaptive mesh refinement\n";
            std::cout << "    --amr-interval N   Refine every N steps (default: 5)\n";
            std::cout << "    --amr-max-level N  Max refinement level\n";
            std::cout << "    --amr-min-level N  Min refinement level\n";
            std::cout << "    --amr-activation-U V  |U| threshold for AMR (default: 1e-3)\n\n";
            std::cout << "  Sparsity / Renumbering:\n";
            std::cout << "    --renumber-dofs    Apply Cuthill-McKee DoF renumbering\n";
            std::cout << "    --no-renumber-dofs Disable DoF renumbering (default)\n";
            std::cout << "    --dump-sparsity    Export sparsity patterns\n\n";
            std::cout << "  Output:\n";
            std::cout << "    --vtk_interval N   VTK output every N steps\n";
            std::cout << "    --verbose          Verbose output\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // ================================================================
    // AMR finalization: auto-compute level limits if not explicitly set
    //
    // Following Semi_Coupled/Nochetto approach:
    //   amr_min_level = max(1, initial_refinement - 2)  [bulk coarsening floor]
    //   amr_max_level = initial_refinement + 2          [interface refinement cap]
    // ================================================================
    if (params.mesh.use_amr)
    {
        if (params.mesh.amr_min_level == 0)
            params.mesh.amr_min_level = std::max(1u,
                params.mesh.initial_refinement >= 2
                    ? params.mesh.initial_refinement - 2 : 0u);

        if (params.mesh.amr_max_level == 0)
            params.mesh.amr_max_level = params.mesh.initial_refinement + 2;
    }

    // Print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  r=" << params.mesh.initial_refinement
                  << ", dt=" << params.time.dt
                  << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
                  << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "  Magnetic: " << (params.enable_magnetic ? "ON" : "OFF")
                  << ", NS: " << (params.enable_ns ? "ON" : "OFF")
                  << ", Gravity: " << (params.enable_gravity ? "ON" : "OFF") << "\n";
        if (params.mesh.use_amr)
            std::cout << "  AMR: ON, interval=" << params.mesh.amr_interval
                      << ", levels=[" << params.mesh.amr_min_level
                      << "," << params.mesh.amr_max_level << "]\n";
        std::cout << "=====================\n";
    }

    return params;
}