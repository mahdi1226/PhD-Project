// ============================================================================
// utilities/parameters.cc - Runtime Configuration for FHD
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "utilities/parameters.h"

#include <cstring>
#include <iostream>
#include <mpi.h>

// ============================================================================
// Spinning magnet preset (Nochetto Section 7.1)
//
// Ω = (0,1)², t ∈ [0,4]
// Single dipole orbiting the center of the box
// All constitutive constants set to 1, except 𝒯 = 1e-4
// ============================================================================
void Parameters::setup_spinning_magnet()
{
    experiment_name = "spinning_magnet";

    domain.x_min = 0.0;  domain.x_max = 1.0;
    domain.y_min = 0.0;  domain.y_max = 1.0;
    domain.initial_cells_x = 1;
    domain.initial_cells_y = 1;

    // All constants = 1 (Nochetto Section 7.1)
    physics.nu = 1.0;
    physics.nu_r = 1.0;
    physics.mu_0 = 1.0;
    physics.j_micro = 1.0;
    physics.c_1 = 1.0;
    physics.c_2 = 1.0;
    physics.sigma = 0.0;
    physics.T_relax = 1e-4;  // fast relaxation
    physics.kappa_0 = 1.0;

    // Dipole: initial position below box, d = (0,1)
    dipoles.positions.clear();
    dipoles.positions.push_back(dealii::Point<2>(0.5, -0.4));
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 10.0;
    dipoles.ramp_time = 1.0;

    time.dt = 1e-2;
    time.t_final = 4.0;
    time.max_steps = 400;

    mesh.initial_refinement = 5;  // 32 elements per direction

    picard_iterations = 20;
    picard_tolerance = 1e-4;
    picard_relaxation = 0.5;

    use_simplified_model = false;
}

// ============================================================================
// MMS validation preset (Nochetto Section 6)
//
// Ω = (0,1)², manufactured solutions, τ = h²
// ============================================================================
void Parameters::setup_mms_validation()
{
    domain.x_min = 0.0;  domain.x_max = 1.0;
    domain.y_min = 0.0;  domain.y_max = 1.0;
    domain.initial_cells_x = 1;
    domain.initial_cells_y = 1;

    physics.nu = 1.0;
    physics.nu_r = 1.0;
    physics.mu_0 = 1.0;
    physics.j_micro = 1.0;
    physics.c_1 = 1.0;
    physics.c_2 = 1.0;
    physics.sigma = 0.0;
    physics.T_relax = 1.0;
    physics.kappa_0 = 1.0;

    uniform_field.enabled = false;
    dipoles.positions.clear();

    enable_mms = true;
    use_simplified_model = true;  // h := h_a for MMS

    mesh.initial_refinement = 4;
}

// ============================================================================
// Pumping preset (Nochetto Section 7.2)
//
// Channel Ω = [0,6] × [0,1], 64 dipoles (32 below, 32 above)
// in the middle section x ∈ [2,4].
// α_s(t) = |sin(ωt − κx_s)|^{2q}  (traveling pulses, set per-timestep)
// ============================================================================
void Parameters::setup_pumping()
{
    experiment_name = "pumping";

    // Channel domain: [0,6] × [0,1]
    domain.x_min = 0.0;  domain.x_max = 6.0;
    domain.y_min = 0.0;  domain.y_max = 1.0;
    domain.initial_cells_x = 6;
    domain.initial_cells_y = 1;

    // All constants = 1 (Nochetto Section 7.2)
    physics.nu = 1.0;
    physics.nu_r = 1.0;
    physics.mu_0 = 1.0;
    physics.j_micro = 1.0;
    physics.c_1 = 1.0;
    physics.c_2 = 1.0;
    physics.sigma = 0.0;
    physics.T_relax = 1e-4;
    physics.kappa_0 = 1.0;

    // 64 dipoles in middle section [2,4], all d = (0,1)
    // 32 below (y = -0.1), 32 above (y = 1.1)
    dipoles.positions.clear();
    dipoles.intensities.clear();
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 1.0;
    dipoles.ramp_time = 0.0;

    const double y_below = -0.1;
    const double y_above = 1.1;
    const double x_start = 2.0;
    const double dx = 2.0 / 32.0;  // 32 dipoles over 2 units

    for (int i = 0; i < 32; ++i)
    {
        const double x = x_start + (i + 0.5) * dx;
        dipoles.positions.push_back(dealii::Point<2>(x, y_below));
        dipoles.positions.push_back(dealii::Point<2>(x, y_above));
        dipoles.intensities.push_back(0.0);  // set per-timestep
        dipoles.intensities.push_back(0.0);
    }

    time.dt = 1e-2;
    time.t_final = 2.0;
    time.max_steps = 200;

    // Channel 6×1 with 6 initial cells, ref 4 → 96×16 = 1536 cells
    mesh.initial_refinement = 4;

    picard_iterations = 20;
    picard_tolerance = 1e-4;
    picard_relaxation = 0.5;

    use_simplified_model = false;
}

// ============================================================================
// Stirring preset — common base (Nochetto Section 7.3)
//
// Ω = (0,1)², ν = ν_r = 0.5, passive scalar with α = 0.001
// ============================================================================
void Parameters::setup_stirring()
{
    domain.x_min = 0.0;  domain.x_max = 1.0;
    domain.y_min = 0.0;  domain.y_max = 1.0;
    domain.initial_cells_x = 1;
    domain.initial_cells_y = 1;

    physics.nu = 0.5;
    physics.nu_r = 0.5;
    physics.mu_0 = 1.0;
    physics.j_micro = 1.0;
    physics.c_1 = 1.0;
    physics.c_2 = 1.0;
    physics.sigma = 0.0;
    physics.T_relax = 1e-4;
    physics.kappa_0 = 1.0;

    time.dt = 1e-2;
    time.t_final = 4.0;
    time.max_steps = 400;

    mesh.initial_refinement = 5;

    picard_iterations = 20;
    picard_tolerance = 1e-4;
    picard_relaxation = 0.5;

    enable_passive_scalar = true;
    passive_scalar.alpha = 0.001;

    use_simplified_model = false;
}

// ============================================================================
// Stirring Approach 1 (Nochetto Section 7.3, Eq. 105)
//
// Two dipoles at bottom with alternating polarity (phase mismatch π/2)
// α₁ = α₀ sin(ωt), α₂ = α₀ sin(ωt + π/2), f=20Hz, α₀=5
// ============================================================================
void Parameters::setup_stirring_approach1()
{
    setup_stirring();
    experiment_name = "stirring_approach1";

    // Two dipoles below bottom edge with opposite polarity (Figure 12)
    // y = -0.4 matches spinning magnet distance (Section 7.1)
    dipoles.positions.clear();
    dipoles.positions.push_back(dealii::Point<2>(0.25, -0.4));
    dipoles.positions.push_back(dealii::Point<2>(0.75, -0.4));

    dipoles.directions.clear();
    dipoles.directions.push_back({0.0, 1.0});    // d₁ = (0,1)
    dipoles.directions.push_back({0.0, -1.0});   // d₂ = (0,-1) — opposite

    dipoles.intensities = {0.0, 0.0};  // set per-timestep
    dipoles.intensity_max = 5.0;
    dipoles.ramp_time = 0.0;

    time.t_final = 4.0;
    time.max_steps = 400;
}

// ============================================================================
// Stirring Approach 2 (Nochetto Section 7.3, Eq. 106)
//
// Eight dipoles on lower edge, traveling wave
// α_s = α₀ |sin(ωt − κx_s)|, κ = 2π/λ, λ = 0.8, f=20Hz, α₀=5
// ============================================================================
void Parameters::setup_stirring_approach2()
{
    setup_stirring();
    experiment_name = "stirring_approach2";

    // 8 dipoles evenly spaced on lower edge, all pointing up
    dipoles.positions.clear();
    dipoles.intensities.clear();
    dipoles.direction = {0.0, 1.0};

    const double y_dipole = -0.1;  // "on the lower edge of the box" (paper p.30)
    for (int s = 0; s < 8; ++s)
    {
        double x_s = (s + 0.5) / 8.0;  // centered in each 1/8 segment
        dipoles.positions.push_back(dealii::Point<2>(x_s, y_dipole));
        dipoles.intensities.push_back(0.0);  // set per-timestep
    }

    dipoles.intensity_max = 5.0;
    dipoles.ramp_time = 0.0;

    time.t_final = 1.0;
    time.max_steps = 100;
}

// ============================================================================
// Stirring Approach 2 Enhanced (Nochetto Figure 19)
//
// Same as Approach 2 but with: f=40Hz, ν=ν_r=0.1, t_final=4.0
// "Much more striking results" — higher frequency, lower viscosity, longer time
// ============================================================================
void Parameters::setup_stirring_approach2_enhanced()
{
    setup_stirring_approach2();
    experiment_name = "stirring_approach2_enhanced";

    // Enhanced parameters from Figure 19 caption
    physics.nu = 0.1;
    physics.nu_r = 0.1;
    dipoles.frequency = 40.0;   // f = 40Hz (doubled from baseline)

    time.t_final = 4.0;
    time.max_steps = 400;
}

// ============================================================================
// Command line parsing
// ============================================================================
Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // ---- Run mode ----
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
            if (params.run.refs.empty()) { std::cerr << "--ref requires values\n"; std::exit(1); }
        }
        else if (std::strcmp(argv[i], "--steps") == 0)
        {
            if (++i >= argc) { std::cerr << "--steps requires a value\n"; std::exit(1); }
            params.run.steps = std::stoi(argv[i]);
        }

        // ---- Presets ----
        else if (std::strcmp(argv[i], "--spinning-magnet") == 0)
            params.setup_spinning_magnet();
        else if (std::strcmp(argv[i], "--pumping") == 0)
            params.setup_pumping();
        else if (std::strcmp(argv[i], "--stirring") == 0 ||
                 std::strcmp(argv[i], "--stirring-1") == 0)
            params.setup_stirring_approach1();
        else if (std::strcmp(argv[i], "--stirring-2") == 0)
            params.setup_stirring_approach2();
        else if (std::strcmp(argv[i], "--stirring-2-enhanced") == 0)
            params.setup_stirring_approach2_enhanced();
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.setup_mms_validation();

        // ---- Mesh ----
        else if (std::strcmp(argv[i], "--refinement") == 0 ||
                 std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            params.mesh.initial_refinement = std::stoul(argv[i]);
        }

        // ---- Time ----
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

        // ---- Physics ----
        else if (std::strcmp(argv[i], "--nu") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu requires a value\n"; std::exit(1); }
            params.physics.nu = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--nu_r") == 0)
        {
            if (++i >= argc) { std::cerr << "--nu_r requires a value\n"; std::exit(1); }
            params.physics.nu_r = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--kappa_0") == 0)
        {
            if (++i >= argc) { std::cerr << "--kappa_0 requires a value\n"; std::exit(1); }
            params.physics.kappa_0 = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--T_relax") == 0)
        {
            if (++i >= argc) { std::cerr << "--T_relax requires a value\n"; std::exit(1); }
            params.physics.T_relax = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--sigma") == 0)
        {
            if (++i >= argc) { std::cerr << "--sigma requires a value\n"; std::exit(1); }
            params.physics.sigma = std::stod(argv[i]);
        }

        // ---- Model selection ----
        else if (std::strcmp(argv[i], "--simplified") == 0)
            params.use_simplified_model = true;

        // ---- Picard ----
        else if (std::strcmp(argv[i], "--picard_iters") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_iters requires a value\n"; std::exit(1); }
            params.picard_iterations = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--picard_omega") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_omega requires a value\n"; std::exit(1); }
            params.picard_relaxation = std::stod(argv[i]);
        }

        // ---- DG penalty ----
        else if (std::strcmp(argv[i], "--penalty") == 0)
        {
            if (++i >= argc) { std::cerr << "--penalty requires a value\n"; std::exit(1); }
            params.dg.penalty_parameter = std::stod(argv[i]);
        }

        // ---- Output ----
        else if (std::strcmp(argv[i], "--vtk_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--vtk_interval requires a value\n"; std::exit(1); }
            params.output.vtk_interval = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // ---- Solver ----
        else if (std::strcmp(argv[i], "--block-schur") == 0)
        {
            params.solvers.navier_stokes.use_iterative = true;
            params.solvers.navier_stokes.preconditioner =
                LinearSolverParams::Preconditioner::BlockSchur;
        }

        // ---- Help ----
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "FHD — Ferrohydrodynamics Solver\n";
            std::cout << "Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381\n\n";
            std::cout << "Presets:\n";
            std::cout << "  --spinning-magnet   Section 7.1\n";
            std::cout << "  --pumping           Section 7.2\n";
            std::cout << "  --stirring-1        Section 7.3, Approach 1 (two dipoles)\n";
            std::cout << "  --stirring-2        Section 7.3, Approach 2 (traveling wave)\n";
            std::cout << "  --mms               Section 6 MMS validation\n\n";
            std::cout << "Physics:\n";
            std::cout << "  --nu, --nu_r, --kappa_0, --T_relax, --sigma\n\n";
            std::cout << "Model:\n";
            std::cout << "  --simplified        h := h_a (no Poisson solve)\n\n";
            std::cout << "Run:\n";
            std::cout << "  --mode <mms|2d>     --ref 2 3 4 5 6\n";
            std::cout << "  --dt VALUE          --t_final VALUE\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== FHD Configuration ===\n";
        std::cout << "  Domain: [" << params.domain.x_min << "," << params.domain.x_max
                  << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  ν=" << params.physics.nu << ", ν_r=" << params.physics.nu_r
                  << ", μ₀=" << params.physics.mu_0 << ", κ₀=" << params.physics.kappa_0 << "\n";
        std::cout << "  ȷ=" << params.physics.j_micro
                  << ", c₁=" << params.physics.c_1 << ", c₂=" << params.physics.c_2 << "\n";
        std::cout << "  σ=" << params.physics.sigma << ", 𝒯=" << params.physics.T_relax << "\n";
        std::cout << "  dt=" << params.time.dt << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Model: " << (params.use_simplified_model ? "simplified (h:=h_a)" : "full") << "\n";
        if (params.enable_passive_scalar)
            std::cout << "  Passive scalar: α=" << params.passive_scalar.alpha << "\n";
        std::cout << "=========================\n";
    }

    return params;
}
