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
    // Base mesh: r=4 gives h ≈ 1/160 (finer than paper's h=1/100)
    //   x: 10*16 = 160 cells, h_x = 1/160 = 0.00625
    //   y: 6*16 = 96  cells, h_y = 0.6/96 = 0.00625
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;

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

    // Mesh — Zhang Eq 4.4: h = 1e-2 (paper), we use h ≈ 1/160
    // r=4: 10*16=160 cells in x (h_x=1/160≈0.006), finer than paper
    mesh.initial_refinement = 4;

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;
    use_reduced_magnetic_field = false;

    // Picard sub-iteration for Poisson-Magnetization coupling
    picard_iterations = 7;
    picard_tolerance = 0.01;

    // Zhang's SAV scheme — use FULL magnetization PDE (not algebraic)
    use_algebraic_magnetization = false;  // Zhang solves mag PDE (Eq 3.15-3.16)
    use_sav = true;
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
void Parameters::setup_rosensweig_nonuniform()
{
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

    // Mesh: h = 1/120 (Paper: "h = 1/120")
    // Base grid 15×9 with refinement 3: 15×8=120 cells in x, 9×8=72 in y
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    mesh.initial_refinement = 3;   // 120×72 cells, h = 1/120

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

    // More Picard iterations — nonuniform field is harder to resolve
    picard_iterations = 15;
    picard_tolerance = 1e-4;
}

// ============================================================================
// Command line parsing
// ============================================================================
Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    // Track which flags were explicitly set (for post-processing)
    bool got_alpha_max = false;
    bool got_Lx = false;
    bool got_Ly = false;
    double new_Lx = 0.0;
    double new_Ly = 0.0;

    for (int i = 1; i < argc; ++i)
    {
        // ---- Run mode / multi-ref / steps ----
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

        // ---- Presets ----
        else if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();
        else if (std::strcmp(argv[i], "--rosensweig-nonuniform") == 0)
            params.setup_rosensweig_nonuniform();

        // ---- Mesh ----
        else if (std::strcmp(argv[i], "--refinement") == 0 ||
                 std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            params.mesh.initial_refinement = std::stoul(argv[i]);
        }

        // ---- Time stepping ----
        else if (std::strcmp(argv[i], "--dt") == 0)
        {
            if (++i >= argc) { std::cerr << "--dt requires a value\n"; std::exit(1); }
            params.time.dt = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--t_final") == 0)
        {
            if (++i >= argc) { std::cerr << "--t_final requires a value\n"; std::exit(1); }
            params.time.t_final = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--max_steps") == 0)
        {
            if (++i >= argc) { std::cerr << "--max_steps requires a value\n"; std::exit(1); }
            params.time.max_steps = std::stoul(argv[i]);
        }

        // ---- Uniform applied field ----
        else if (std::strcmp(argv[i], "--uniform_field") == 0)
        {
            if (++i >= argc) { std::cerr << "--uniform_field requires intensity value\n"; std::exit(1); }
            params.uniform_field.enabled = true;
            params.uniform_field.intensity_max = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--field_ramp_time") == 0)
        {
            if (++i >= argc) { std::cerr << "--field_ramp_time requires a value\n"; std::exit(1); }
            params.uniform_field.ramp_time = std::stod(argv[i]);
        }

        // ---- Magnetization physics ----
        else if (std::strcmp(argv[i], "--tau_M") == 0)
        {
            if (++i >= argc) { std::cerr << "--tau_M requires a value\n"; std::exit(1); }
            params.physics.tau_M = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--beta") == 0)
        {
            if (++i >= argc) { std::cerr << "--beta requires a value\n"; std::exit(1); }
            params.physics.beta = std::stod(argv[i]);
            params.physics.enable_beta_term = (params.physics.beta != 0.0);
        }
        else if (std::strcmp(argv[i], "--enable_beta") == 0)
            params.physics.enable_beta_term = true;
        else if (std::strcmp(argv[i], "--no-spin-coupling") == 0)
            params.disable_spin_coupling = true;

        // ---- Cahn-Hilliard physics ----
        else if (std::strcmp(argv[i], "--mobility") == 0)
        {
            if (++i >= argc) { std::cerr << "--mobility requires a value\n"; std::exit(1); }
            params.physics.mobility = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--lambda") == 0)
        {
            if (++i >= argc) { std::cerr << "--lambda requires a value\n"; std::exit(1); }
            params.physics.lambda = std::stod(argv[i]);
        }

        // ---- Susceptibility / field strength ----
        else if (std::strcmp(argv[i], "--chi0") == 0)
        {
            if (++i >= argc) { std::cerr << "--chi0 requires a value\n"; std::exit(1); }
            params.physics.chi_0 = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--alpha_max") == 0)
        {
            if (++i >= argc) { std::cerr << "--alpha_max requires a value\n"; std::exit(1); }
            params.dipoles.intensity_max = std::stod(argv[i]);
            got_alpha_max = true;
        }
        else if (std::strcmp(argv[i], "--epsilon") == 0)
        {
            if (++i >= argc) { std::cerr << "--epsilon requires a value\n"; std::exit(1); }
            params.physics.epsilon = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--y_interface") == 0)
        {
            if (++i >= argc) { std::cerr << "--y_interface requires a value\n"; std::exit(1); }
            params.flat_interface_y = std::stod(argv[i]);
        }

        // ---- Domain geometry ----
        else if (std::strcmp(argv[i], "--Lx") == 0)
        {
            if (++i >= argc) { std::cerr << "--Lx requires a value\n"; std::exit(1); }
            got_Lx = true;
            new_Lx = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--Ly") == 0)
        {
            if (++i >= argc) { std::cerr << "--Ly requires a value\n"; std::exit(1); }
            got_Ly = true;
            new_Ly = std::stod(argv[i]);
        }

        // ---- Navier-Stokes physics ----
        else if (std::strcmp(argv[i], "--nu_water") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu_water requires a value\n"; std::exit(1); }
            params.physics.nu_water = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--nu_ferro") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu_ferro requires a value\n"; std::exit(1); }
            params.physics.nu_ferro = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--density_ratio") == 0)
        {
            if (++i >= argc) { std::cerr << "--density_ratio requires a value\n"; std::exit(1); }
            params.physics.r = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--gravity") == 0)
        {
            if (++i >= argc) { std::cerr << "--gravity requires a value\n"; std::exit(1); }
            params.physics.gravity_magnitude = std::stod(argv[i]);
        }

        // ---- Picard (DEPRECATED: Zhang single-pass scheme, no iteration) ----
        else if (std::strcmp(argv[i], "--picard_iters") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_iters requires a value\n"; std::exit(1); }
            // Deprecated: value parsed but ignored (Zhang single-pass)
            std::cerr << "Warning: --picard_iters deprecated (Zhang single-pass scheme)\n";
        }
        else if (std::strcmp(argv[i], "--picard_omega") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_omega requires a value\n"; std::exit(1); }
            // Deprecated: value parsed but ignored (Zhang single-pass)
            std::cerr << "Warning: --picard_omega deprecated (Zhang single-pass scheme)\n";
        }

        // ---- SAV + Algebraic Magnetization ----
        else if (std::strcmp(argv[i], "--algebraic_M") == 0)
            params.use_algebraic_magnetization = true;
        else if (std::strcmp(argv[i], "--no_algebraic_M") == 0)
            params.use_algebraic_magnetization = false;
        else if (std::strcmp(argv[i], "--sav") == 0)
            params.use_sav = true;
        else if (std::strcmp(argv[i], "--no_sav") == 0)
            params.use_sav = false;
        else if (std::strcmp(argv[i], "--sav_S") == 0 || std::strcmp(argv[i], "--sav_S1") == 0)
        {
            if (++i >= argc) { std::cerr << "--sav_S requires a value\n"; std::exit(1); }
            params.sav.S1 = std::stod(argv[i]);
        }

        // ---- Solver overrides ----
        else if (std::strcmp(argv[i], "--all-direct") == 0)
        {
            // Force direct solvers for ALL subsystems (diagnostic for solver accuracy)
            params.solvers.poisson.use_iterative = false;
            params.solvers.ns.use_iterative = false;
            // Magnetization and CH already default to direct
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

        // ---- Validation presets (Zhang, He & Yang, CMAME 371 (2020)) ----
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
                params.use_sav = true;
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
                params.use_sav = true;
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
                params.use_sav = true;
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
            std::cout << "Ferrofluid Solver — Poisson + Magnetization + CH + NS\n";
            std::cout << "Reference: Nochetto et al. CMAME 309 (2016) Eq. 42a-42f\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "  Run mode:\n";
            std::cout << "    --mode <mms|2d|3d|temporal>  Run mode (default: mms)\n";
            std::cout << "    --ref 2 3 4 5 6    Refinement levels for mms/temporal\n";
            std::cout << "    --steps N          Override number of time steps\n\n";
            std::cout << "  Presets:\n";
            std::cout << "    --rosensweig        Rosensweig preset (Section 4.3, uniform field)\n";
            std::cout << "    --rosensweig-nonuniform  Rosensweig preset (Section 4.4, 42 dipoles)\n";
            std::cout << "    --validation MODE   Validation test (droplet|square)\n\n";
            std::cout << "  Mesh:\n";
            std::cout << "    --refinement N      Mesh refinement level (2d/3d modes)\n";
            std::cout << "    --amr / --no-amr    Enable/disable adaptive mesh refinement\n";
            std::cout << "    --amr-interval N    Refine every N steps (default: 5)\n";
            std::cout << "    --amr-max-level N   Max refinement level (default: 0=no cap)\n";
            std::cout << "    --amr-min-level N   Min refinement level (default: 0)\n";
            std::cout << "    --amr-activation-U V  |U| threshold to activate AMR (default: 1e-3, 0=immediate)\n\n";
            std::cout << "  Time stepping:\n";
            std::cout << "    --dt VALUE          Time step size\n";
            std::cout << "    --t_final VALUE     Final simulation time\n\n";
            std::cout << "  Physics (parametric study):\n";
            std::cout << "    --chi0 VALUE        Susceptibility χ₀ (default: 0.5)\n";
            std::cout << "    --alpha_max VALUE   Max field intensity (default: 8000)\n";
            std::cout << "    --epsilon VALUE     Interface width ε (default: 5e-3)\n";
            std::cout << "    --y_interface VALUE  Flat interface y-position (default: 0.2)\n";
            std::cout << "    --Lx VALUE          Domain width (rescales cells+dipoles)\n";
            std::cout << "    --Ly VALUE          Domain height (rescales cells)\n\n";
            std::cout << "  Cahn-Hilliard:\n";
            std::cout << "    --mobility VALUE    CH mobility γ (default: 0.0002)\n";
            std::cout << "    --lambda VALUE      Surface tension λ (default: 0.05)\n\n";
            std::cout << "  Magnetization:\n";
            std::cout << "    --tau_M VALUE       Relaxation time (default: 1e-6)\n";
            std::cout << "    --beta VALUE        Landau-Lifshitz damping coefficient\n";
            std::cout << "    --enable_beta       Enable β-term even if β=0\n\n";
            std::cout << "  Navier-Stokes:\n";
            std::cout << "    --nu_water VALUE    Non-magnetic viscosity (default: 1.0)\n";
            std::cout << "    --nu_ferro VALUE    Ferrofluid viscosity (default: 2.0)\n";
            std::cout << "    --density_ratio V   Density ratio r (default: 0.1)\n";
            std::cout << "    --gravity VALUE     Gravity magnitude (default: 30000)\n\n";
            std::cout << "  Coupling:\n";
            std::cout << "    Mag-Poisson: Zhang single-pass (no Picard iteration)\n";
            std::cout << "    --picard_iters N    [DEPRECATED] ignored\n";
            std::cout << "    --picard_omega V    [DEPRECATED] ignored\n\n";
            std::cout << "  Subsystems:\n";
            std::cout << "    --mms               MMS verification mode\n";
            std::cout << "    --no_magnetic       Disable applied field\n";
            std::cout << "    --reduced_field     H = h_a only (dome setup)\n";
            std::cout << "    --no_ns             Disable Navier-Stokes\n";
            std::cout << "    --no_gravity        Disable gravity body force\n\n";
            std::cout << "  Solvers:\n";
            std::cout << "    --all-direct         Force direct solvers for ALL subsystems\n";
            std::cout << "    --poisson-direct     Force direct solver for Poisson\n";
            std::cout << "    --ns-direct          Force direct solver for NS (ux, uy, p)\n\n";
            std::cout << "  Sparsity / Renumbering:\n";
            std::cout << "    --renumber-dofs         Apply Cuthill-McKee DoF renumbering (reduces bandwidth)\n";
            std::cout << "    --no-renumber-dofs      Disable DoF renumbering (default)\n";
            std::cout << "    --dump-sparsity         Export sparsity patterns (SVG + gnuplot + bandwidth CSV)\n\n";
            std::cout << "  Output:\n";
            std::cout << "    --verbose           Verbose output\n";
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
    // Post-processing: apply deferred CLI overrides
    // (Must run after all flags are parsed, since presets set defaults
    //  and CLI flags override them. Order: preset → flag → post-process)
    // ================================================================

    // --alpha_max: auto-adjust ramp_slope to keep ramp time = 1.6s
    if (got_alpha_max && params.dipoles.ramp_slope > 0.0)
        params.dipoles.ramp_slope = params.dipoles.intensity_max / 1.6;

    // --Lx: rescale domain, initial_cells_x, and dipole x-positions
    if (got_Lx)
    {
        const double old_Lx = params.domain.x_max - params.domain.x_min;
        params.domain.x_max = params.domain.x_min + new_Lx;

        // Scale initial_cells_x to maintain ~same cell aspect ratio
        // Rosensweig base: 10 cells for Lx=1.0
        params.domain.initial_cells_x =
            static_cast<unsigned int>(std::round(
                params.domain.initial_cells_x * new_Lx / old_Lx));
        if (params.domain.initial_cells_x < 1)
            params.domain.initial_cells_x = 1;

        // Rescale dipole x-positions: maintain same relative overhang
        // Base: 5 dipoles span [-0.5, 1.5] for domain [0,1] → overhang = 0.5 each side
        // For new domain [0, Lx]: span [-0.5, Lx+0.5]
        if (!params.dipoles.positions.empty())
        {
            const double x_center_old = (params.domain.x_min + params.domain.x_min + old_Lx) / 2.0;
            const double x_center_new = (params.domain.x_min + params.domain.x_max) / 2.0;
            const double scale = new_Lx / old_Lx;
            for (auto& pos : params.dipoles.positions)
            {
                double x_rel = pos[0] - x_center_old;  // relative to old center
                pos[0] = x_center_new + x_rel * scale;  // scale and shift
            }
        }
    }

    // --Ly: rescale domain height and initial_cells_y
    if (got_Ly)
    {
        const double old_Ly = params.domain.y_max - params.domain.y_min;
        params.domain.y_max = params.domain.y_min + new_Ly;

        params.domain.initial_cells_y =
            static_cast<unsigned int>(std::round(
                params.domain.initial_cells_y * new_Ly / old_Ly));
        if (params.domain.initial_cells_y < 1)
            params.domain.initial_cells_y = 1;
    }

    // Print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
                  << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "  epsilon=" << params.physics.epsilon << "\n";
        std::cout << "  CH: mobility=" << params.physics.mobility
                  << ", lambda=" << params.physics.lambda << "\n";
        std::cout << "  Mag: chi_0=" << params.physics.chi_0
                  << ", tau_M=" << params.physics.tau_M << "\n";
        std::cout << "  beta=" << params.physics.beta
                  << " (enabled=" << (params.physics.enable_beta_term ? "yes" : "no") << ")\n";
        std::cout << "  NS: nu_w=" << params.physics.nu_water
                  << ", nu_f=" << params.physics.nu_ferro
                  << ", r=" << params.physics.r
                  << ", |g|=" << params.physics.gravity_magnitude << "\n";
        std::cout << "  dt=" << params.time.dt
                  << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Mag-Poisson: Zhang single-pass (no Picard)\n";
        std::cout << "  Magnetic: " << (params.enable_magnetic ? "ON" : "OFF") << "\n";
        std::cout << "  NS: " << (params.enable_ns ? "ON" : "OFF")
                  << ", Gravity: " << (params.enable_gravity ? "ON" : "OFF") << "\n";
        std::cout << "  MMS: " << (params.enable_mms ? "ON" : "OFF") << "\n";
        if (params.uniform_field.enabled)
            std::cout << "  Uniform field: ON, |H_a|=" << params.uniform_field.intensity_max
                      << ", ramp=" << params.uniform_field.ramp_time << "\n";
        std::cout << "=====================\n";
    }

    return params;
}