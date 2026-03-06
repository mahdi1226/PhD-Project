// ============================================================================
// utilities/parameters.cc - Runtime Configuration
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "utilities/parameters.h"
#include <array>
#include <cstring>
#include <mpi.h>

void Parameters::setup_rosensweig()
{
    preset_name = "rosen";  // For auto-generating run_name

    // Domain (Section 6.2, p.522)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 6;
    ic.pool_depth = 0.2;

    // Physical parameters (Section 6.2, p.520-522)
    physics.epsilon = 0.01;       // interface thickness
    physics.mobility = 0.0002;    // γ
    physics.lambda = 0.05;        // capillary coefficient
    physics.chi_0 = 0.5;          // susceptibility
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;              // density ratio
    physics.gravity = 30000.0;    // non-dimensional gravity

    dipoles.positions = {
        dealii::Point<2>(-0.5, -15.0),
        dealii::Point<2>( 0.0, -15.0),
        dealii::Point<2>( 0.5, -15.0),
        dealii::Point<2>( 1.0, -15.0),
        dealii::Point<2>( 1.5, -15.0)
    };
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 6000.0;
    dipoles.ramp_time = 1.6;

    // Time-stepping
    time.dt = 5e-4;
    time.t_final = 2.0;
    time.max_steps = 4000;
    time.use_adaptive_dt = false;  // PAPER_MATCH: Paper uses fixed dt

    // Mesh (Paper Section 6.2, p.522)
    // Paper: "initial mesh has 10 elements in x, 6 in y, allow maximum refinement of 4,5,6,7 levels"
    // The -r flag sets amr_max_level (target interface resolution), NOT initial_refinement.
    // Start with coarse uniform mesh, let AMR refine up to target level.
    mesh.initial_refinement = 4;      // Paper: 10×6 base + 4 levels → 160×96 = 15360 cells
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    mesh.amr_min_level = 2;           // Coarsen bulk (paper Fig. 2)
    mesh.amr_max_level = 7;           // Target: 7 levels (paper Fig. 3, column 4)
    mesh.amr_refine_threshold = 0.0;  // Let fixed_fraction decide (no artificial barrier)

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Paper Eq. 42d: algebraic M = χ(θ)∇Φ (L2 projection, NOT transport PDE)
    use_dg_transport = false;

    // Output
    output.frequency = 10;
}

void Parameters::setup_hedgehog()
{
    preset_name = "hedge";  // For auto-generating run_name

    // Domain (Section 6.3, p.527)
    // Same rectangular domain as Rosensweig: Ω = (0,1) × (0,0.6)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    ic.pool_depth = 0.11;

    // Physical parameters (Section 6.3, p.527) - DIFFERENT from Rosensweig!
    physics.epsilon = 0.005;      // interface thickness (HALF of Rosensweig)
    physics.mobility = 0.0002;    // γ (same)
    physics.lambda = 0.025;       // capillary coefficient (HALF of Rosensweig)
    physics.chi_0 = 0.9;          // susceptibility (HIGHER than Rosensweig: 0.5)
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;              // density ratio
    physics.gravity = 30000.0;    // non-dimensional gravity

    // Dipoles (Section 6.3, p.527-528)
    // 42 dipoles arranged in 3 rows × 14 columns
    // Approximating a bar magnet of 0.4 width × 0.5 height
    // Rows at y = -0.5, -0.75, -1.0
    // X-positions: equi-distributed from x = 0.3 to x = 0.7 (centered on domain)
    dipoles.positions.clear();
    const double x_min_dipole = 0.3;
    const double x_max_dipole = 0.7;
    const double x_spacing = (x_max_dipole - x_min_dipole) / 13.0;
    const std::array<double, 3> y_rows = {-0.5, -0.75, -1.0};

    for (double y : y_rows)
    {
        for (int j = 0; j < 14; ++j)
        {
            double x = x_min_dipole + j * x_spacing;
            dipoles.positions.push_back(dealii::Point<2>(x, y));
        }
    }

    dipoles.direction = {0.0, 1.0};
    // FIXED: Paper Section 6.3 specifies αs = 4.3 (NOT 6000!)
    // "αs is increased linearly in time from αs = 0 at time t = 0,
    //  to its maximum value αs = 4.3 at time t = 4.2"
    dipoles.intensity_max = 4.3;
    dipoles.ramp_time = 4.2;

    // Time-stepping
    // Paper Section 6.3: dt=1e-4 needed for CFL stability with ε=0.005
    time.dt = 1e-4;
    time.t_final = 6.0;
    time.max_steps = 60000;
    time.use_adaptive_dt = false;  // PAPER_MATCH: Paper uses fixed dt

    // Mesh (Paper Section 6.3)
    // Start coarse, let AMR refine up to target level at interface
    mesh.initial_refinement = 3;      // Start coarse
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    mesh.amr_min_level = 1;           // Coarsen bulk aggressively
    mesh.amr_max_level = 7;           // Default: 7 levels (hedgehog needs finer than rosen due to ε=0.005)

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Paper Eq. 42d: algebraic M = χ(θ)∇Φ (L2 projection, NOT transport PDE)
    use_dg_transport = false;

    // Output
    output.frequency = 10;
}

void Parameters::setup_dome()
{
    preset_name = "dome";  // For auto-generating run_name

    // Same as hedgehog but WITHOUT demagnetizing field
    // Results in dome shape (Fig. 7) instead of spikes (Fig. 6)

    // Domain (same as hedgehog)
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 0.6;
    domain.initial_cells_x = 15;
    domain.initial_cells_y = 9;
    ic.pool_depth = 0.11;
    time.use_adaptive_dt = false;

    // Physical parameters (same as Hedgehog, Section 6.3)
    physics.epsilon = 0.005;
    physics.mobility = 0.0002;
    physics.lambda = 0.025;
    physics.chi_0 = 0.9;
    physics.nu_water = 1.0;
    physics.nu_ferro = 2.0;
    physics.r = 0.1;
    physics.gravity = 30000.0;

    // Same dipoles as hedgehog
    dipoles.positions.clear();
    const double x_min_dipole = 0.3;
    const double x_max_dipole = 0.7;
    const double x_spacing = (x_max_dipole - x_min_dipole) / 13.0;
    const std::array<double, 3> y_rows = {-0.5, -0.75, -1.0};

    for (double y : y_rows)
    {
        for (int j = 0; j < 14; ++j)
        {
            double x = x_min_dipole + j * x_spacing;
            dipoles.positions.push_back(dealii::Point<2>(x, y));
        }
    }

    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 4.3;
    dipoles.ramp_time = 4.2;

    // KEY DIFFERENCE: Use reduced field (h = ha only, skip Poisson for hd)
    use_reduced_magnetic_field = true;  // NEW FLAG

    // Time-stepping: dt=1e-4 for CFL stability with ε=0.005
    time.dt = 1e-4;
    time.t_final = 6.0;
    time.max_steps = 60000;
    time.use_adaptive_dt = false;  // Disable adaptive!

    // Same mesh as hedgehog (Paper Section 6.3)
    mesh.initial_refinement = 3;      // Start coarse
    mesh.use_amr = true;
    mesh.amr_interval = 5;
    mesh.amr_min_level = 1;           // Coarsen bulk aggressively
    mesh.amr_max_level = 7;           // Default: 7 levels

    // Subsystems
    enable_magnetic = true;
    enable_ns = true;
    enable_gravity = true;

    // Paper Eq. 42d: algebraic M = χ(θ)∇Φ (L2 projection, NOT transport PDE)
    use_dg_transport = false;

    // Output
    output.frequency = 10;
}

void Parameters::setup_droplet()
{
    preset_name = "droplet";  // For auto-generating run_name

    // Simple droplet test case - no magnetic, no gravity
    // Verifies surface tension and phase field dynamics

    // Domain: unit square
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 1.0;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 10;

    // Initial condition: circular droplet (type = 1)
    ic.type = 1;  // Circular droplet
    ic.droplet_center_x = 0.5;
    ic.droplet_center_y = 0.5;
    ic.droplet_radius = 0.25;

    // Physical parameters (mild)
    physics.epsilon = 0.02;       // interface thickness
    physics.mobility = 0.001;     // γ
    physics.lambda = 0.1;         // capillary coefficient
    physics.chi_0 = 0.0;          // no magnetic susceptibility
    physics.nu_water = 1.0;
    physics.nu_ferro = 1.0;       // same viscosity
    physics.r = 0.0;              // no density difference
    physics.gravity = 0.0;        // no gravity

    // No dipoles
    dipoles.positions.clear();
    dipoles.intensity_max = 0.0;
    dipoles.ramp_time = 1.0;

    // Time-stepping
    time.dt = 0.001;
    time.t_final = 1.0;
    time.max_steps = 1000;
    time.use_adaptive_dt = false;

    // Mesh
    mesh.initial_refinement = 5;
    mesh.use_amr = true;
    mesh.amr_interval = 10;
    mesh.amr_min_level = mesh.initial_refinement - 2;
    mesh.amr_max_level = mesh.initial_refinement + 2;

    // Subsystems - disable magnetic and gravity
    enable_magnetic = false;
    enable_ns = true;
    enable_gravity = false;

    // Output
    output.frequency = 10;
}

void Parameters::setup_square()
{
    preset_name = "square";  // For auto-generating run_name

    // Square relaxation test case
    // Verifies surface tension driving square -> circle

    // Domain: unit square
    domain.x_min = 0.0;
    domain.x_max = 1.0;
    domain.y_min = 0.0;
    domain.y_max = 1.0;
    domain.initial_cells_x = 10;
    domain.initial_cells_y = 10;

    // Initial condition: diamond droplet (type = 2)
    ic.type = 2;  // Diamond droplet
    ic.droplet_center_x = 0.5;
    ic.droplet_center_y = 0.5;
    ic.droplet_radius = 0.25; // Half-width of square

    // Physical parameters (mild)
    physics.epsilon = 0.02;       // interface thickness
    physics.mobility = 0.001;     // γ
    physics.lambda = 0.1;         // capillary coefficient
    physics.chi_0 = 0.0;          // no magnetic susceptibility
    physics.nu_water = 1.0;
    physics.nu_ferro = 1.0;       // same viscosity
    physics.r = 0.0;              // no density difference
    physics.gravity = 0.0;        // no gravity

    // No dipoles
    dipoles.positions.clear();
    dipoles.intensity_max = 0.0;
    dipoles.ramp_time = 1.0;

    // Time-stepping
    time.dt = 0.001;
    time.t_final = 1.0;
    time.max_steps = 1000;
    time.use_adaptive_dt = false;

    // Mesh
    mesh.initial_refinement = 5;
    mesh.use_amr = true;
    mesh.amr_interval = 10;
    mesh.amr_min_level = mesh.initial_refinement - 2;
    mesh.amr_max_level = mesh.initial_refinement + 2;

    // Subsystems - disable magnetic and gravity
    enable_magnetic = false;
    enable_ns = true;
    enable_gravity = false;

    // Output
    output.frequency = 10;
}

// ============================================================================
// Droplet + Uniform Magnetic Field
// Same as setup_droplet() but with magnetic enabled and constant h_a = (0, 1)
// Tests that uniform field + symmetric droplet → no Kelvin force (no deformation)
// ============================================================================
void Parameters::setup_droplet_uniform_B()
{
    // Start from base droplet
    setup_droplet();
    preset_name = "droplet-uniform-B";

    // Enable magnetic subsystem
    enable_magnetic = true;
    physics.chi_0 = 0.5;
    physics.tau_M = 1e-6;

    // Paper Eq. 42d: algebraic M = χ(θ)∇Φ
    use_dg_transport = false;

    // Uniform applied field h_a = (0, 1) — vertical, instant (no ramp)
    dipoles.use_uniform_field = true;
    dipoles.uniform_field_value[0] = 0.0;
    dipoles.uniform_field_value[1] = 1.0;
    dipoles.ramp_time = 0.0;  // instant
}

// ============================================================================
// Droplet + Non-Uniform Magnetic Field (single dipole)
// Same as setup_droplet() but with a single dipole below center
// Tests that non-uniform field → Kelvin force → droplet elongation
// ============================================================================
void Parameters::setup_droplet_nonuniform_B()
{
    // Start from base droplet
    setup_droplet();
    preset_name = "droplet-nonuniform-B";

    // Enable magnetic subsystem
    enable_magnetic = true;
    physics.chi_0 = 0.5;
    physics.tau_M = 1e-6;

    // Paper Eq. 42d: algebraic M = χ(θ)∇Φ
    use_dg_transport = false;

    // Single dipole below domain center
    dipoles.use_uniform_field = false;
    dipoles.positions.clear();
    dipoles.positions.push_back(dealii::Point<2>(0.5, -0.5));
    dipoles.direction = {0.0, 1.0};
    dipoles.intensity_max = 0.5;
    dipoles.ramp_time = 0.0;
    dipoles.regularization = 0.01;
}

void Parameters::finalize_run_name()
{
    // If user specified --run_name, use it as-is
    if (!output.run_name.empty())
        return;

    // Auto-generate: for AMR use max level, otherwise use initial refinement
    if (mesh.use_amr)
        output.run_name = preset_name + "-L" + std::to_string(mesh.amr_max_level) + "-amr";
    else
        output.run_name = preset_name + "-r" + std::to_string(mesh.initial_refinement);
}

Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Parameters params;

    for (int i = 1; i < argc; ++i)
    {
        // Presets
        if (std::strcmp(argv[i], "--rosensweig") == 0)
            params.setup_rosensweig();
        else if (std::strcmp(argv[i], "--hedgehog") == 0)
            params.setup_hedgehog();
        else if (std::strcmp(argv[i], "--droplet") == 0)
            params.setup_droplet();
        else if (std::strcmp(argv[i], "--droplet_uniform_B") == 0)
            params.setup_droplet_uniform_B();
        else if (std::strcmp(argv[i], "--droplet_nonuniform_B") == 0)
            params.setup_droplet_nonuniform_B();
        else if (std::strcmp(argv[i], "--square") == 0)
            params.setup_square();
        else if (std::strcmp(argv[i], "--dome") == 0)
            params.setup_dome();

        // Overrides
        else if (std::strcmp(argv[i], "--refinement") == 0 || std::strcmp(argv[i], "-r") == 0)
        {
            if (++i >= argc) { std::cerr << "--refinement requires a value\n"; std::exit(1); }
            unsigned int r = std::stoul(argv[i]);
            if (params.mesh.use_amr)
            {
                // For AMR presets: -r sets the max refinement level (target resolution)
                // The initial uniform mesh stays coarse; AMR refines up to this level
                params.mesh.amr_max_level = r;
            }
            else
            {
                // For non-AMR: -r sets uniform global refinement (original behavior)
                params.mesh.initial_refinement = r;
            }
        }
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

        // AMR
        else if (std::strcmp(argv[i], "--amr") == 0)
            params.mesh.use_amr = true;
        else if (std::strcmp(argv[i], "--no_amr") == 0)
            params.mesh.use_amr = false;
        else if (std::strcmp(argv[i], "--amr_interval") == 0)
        {
            if (++i >= argc) { std::cerr << "--amr_interval requires a value\n"; std::exit(1); }
            params.mesh.amr_interval = std::stoul(argv[i]);
        }

        // Adaptive time stepping
        else if (std::strcmp(argv[i], "--adaptive_dt") == 0)
            params.time.use_adaptive_dt = true;
        else if (std::strcmp(argv[i], "--no_adaptive_dt") == 0)
            params.time.use_adaptive_dt = false;

        // Solver
        else if (std::strcmp(argv[i], "--direct") == 0)
        {
            params.solvers.ns.use_iterative = false;
            params.solvers.ch.use_iterative = false;  // Also use direct for CH
        }

        // CH convection treatment
        else if (std::strcmp(argv[i], "--explicit_ch_convection") == 0)
            params.physics.implicit_ch_convection = false;
        else if (std::strcmp(argv[i], "--implicit_ch_convection") == 0)
            params.physics.implicit_ch_convection = true;

        // Block-Gauss-Seidel global iteration
        else if (std::strcmp(argv[i], "--bgs") == 0)
            params.enable_bgs = true;
        else if (std::strcmp(argv[i], "--no_bgs") == 0)
            params.enable_bgs = false;
        else if (std::strcmp(argv[i], "--bgs_iters") == 0)
        {
            if (++i >= argc) { std::cerr << "--bgs_iters requires a value\n"; std::exit(1); }
            params.bgs_max_iterations = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--bgs_tol") == 0)
        {
            if (++i >= argc) { std::cerr << "--bgs_tol requires a value\n"; std::exit(1); }
            params.bgs_tolerance = std::stod(argv[i]);
        }

        // Picard iteration settings (inner Poisson <-> Magnetization loop)
        else if (std::strcmp(argv[i], "--picard_iters") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_iters requires a value\n"; std::exit(1); }
            params.picard_iterations = std::stoul(argv[i]);
        }
        else if (std::strcmp(argv[i], "--picard_tol") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_tol requires a value\n"; std::exit(1); }
            params.picard_tolerance = std::stod(argv[i]);
        }
        else if (std::strcmp(argv[i], "--picard_omega") == 0)
        {
            if (++i >= argc) { std::cerr << "--picard_omega requires a value\n"; std::exit(1); }
            params.picard_omega = std::stod(argv[i]);
        }

        // Dipole intensity override
        else if (std::strcmp(argv[i], "--intensity") == 0)
        {
            if (++i >= argc) { std::cerr << "--intensity requires a value\n"; std::exit(1); }
            params.dipoles.intensity_max = std::stod(argv[i]);
        }

        // Kelvin force options
        else if (std::strcmp(argv[i], "--kelvin_face") == 0)
            params.skip_kelvin_face_terms = false;
        else if (std::strcmp(argv[i], "--no_kelvin_face") == 0)
            params.skip_kelvin_face_terms = true;
        else if (std::strcmp(argv[i], "--gradient_kelvin") == 0)
            params.use_gradient_kelvin_force = true;
        else if (std::strcmp(argv[i], "--skew_kelvin") == 0)
            params.use_gradient_kelvin_force = false;

        else if (std::strcmp(argv[i], "--dg_transport") == 0)
            params.use_dg_transport = true;
        else if (std::strcmp(argv[i], "--no_dg_transport") == 0)
            params.use_dg_transport = false;

        // Run name (NEW)
        else if (std::strcmp(argv[i], "--run_name") == 0)
        {
            if (++i >= argc) { std::cerr << "--run_name requires a value\n"; std::exit(1); }
            params.output.run_name = argv[i];
        }

        // Parallel diagnostics
        else if (std::strcmp(argv[i], "--parallel-diag") == 0)
            params.enable_parallel_diagnostics = true;
        else if (std::strcmp(argv[i], "--parallel-diag-all-ranks") == 0)
        {
            params.enable_parallel_diagnostics = true;
            params.parallel_diag_all_ranks = true;
        }

        // DoF renumbering (Cuthill-McKee)
        else if (std::strcmp(argv[i], "--renumber-dofs") == 0)
            params.renumber_dofs = true;
        else if (std::strcmp(argv[i], "--no-renumber-dofs") == 0)
            params.renumber_dofs = false;

        // Sparsity pattern export
        else if (std::strcmp(argv[i], "--dump-sparsity") == 0)
            params.dump_sparsity = true;

        // Debugging
        else if (std::strcmp(argv[i], "--mms") == 0)
            params.enable_mms = true;
        else if (std::strcmp(argv[i], "--magnetic") == 0)
            params.enable_magnetic = true;
        else if (std::strcmp(argv[i], "--no_magnetic") == 0)
            params.enable_magnetic = false;
        else if (std::strcmp(argv[i], "--no_gravity") == 0)
            params.enable_gravity = false;
        else if (std::strcmp(argv[i], "--no_ns") == 0)
            params.enable_ns = false;

        else if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0)
            params.output.verbose = true;

        // Help
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            std::cout << "Ferrofluid Phase Field Solver\n";
            std::cout << "Reference: Nochetto et al. CMAME 309 (2016) 497-531\n\n";
            std::cout << "Usage: " << argv[0] << " --rosensweig|--hedgehog|--droplet|--square [options]\n";
            std::cout << "  (Presets set defaults; later options override them)\n\n";

            std::cout << "PRESETS (pick one):\n";
            std::cout << "  --rosensweig    Rosensweig instability (Section 6.2)\n";
            std::cout << "  --hedgehog      Hedgehog instability (Section 6.3)\n";
            std::cout << "  --dome          Hedgehog with h=ha only (Fig. 7 - dome)\n";
            std::cout << "  --droplet       Simple droplet (no magnetic, no gravity)\n";
            std::cout << "  --droplet_uniform_B     Droplet + uniform magnetic field\n";
            std::cout << "  --droplet_nonuniform_B  Droplet + single dipole (non-uniform)\n";
            std::cout << "  --square        Square relaxation (no magnetic, no gravity)\n\n";

            std::cout << "OVERRIDES:\n";
            std::cout << "  --refinement N  Mesh refinement level\n";
            std::cout << "  --dt DT         Time step size\n";
            std::cout << "  --t_final T     Final simulation time\n";
            std::cout << "  --max_steps N   Maximum number of steps\n\n";

            std::cout << "AMR:\n";
            std::cout << "  --amr / --no_amr      Enable/disable AMR\n";
            std::cout << "  --amr_interval N      Refine every N steps\n\n";

            std::cout << "TIME STEPPING:\n";
            std::cout << "  --adaptive_dt         Enable adaptive time stepping\n";
            std::cout << "  --no_adaptive_dt      Disable adaptive time stepping (paper default)\n\n";

            std::cout << "SOLVER:\n";
            std::cout << "  --implicit_ch_convection  Implicit CH convection (default, no CFL limit)\n";
            std::cout << "  --explicit_ch_convection  Explicit CH convection (CFL-limited, original scheme)\n";
            std::cout << "  --direct              Use direct solver (recommended)\n";
            std::cout << "  --bgs / --no_bgs      Enable/disable Block-Gauss-Seidel global iteration\n";
            std::cout << "  --bgs_iters N         Max Block-GS iterations (default: 5)\n";
            std::cout << "  --bgs_tol TOL         Block-GS convergence tolerance (default: 1e-2)\n";
            std::cout << "  --picard_iters N      Max Picard iterations for Mag-Poisson (default: 7)\n";
            std::cout << "  --picard_tol TOL      Picard convergence tolerance (default: 0.05)\n";
            std::cout << "  --picard_omega W      Picard under-relaxation for M (default: 0.35)\n";
            std::cout << "  --dg_transport        Enable DG magnetization transport\n";
            std::cout << "  --no_dg_transport     Disable DG transport (quasi-equilibrium)\n";
            std::cout << "  --kelvin_face         Include Kelvin force face terms\n";
            std::cout << "  --no_kelvin_face      Skip Kelvin face terms (default, CG phi)\n";
            std::cout << "  --gradient_kelvin     Use CG gradient Kelvin force: (mu0/2)|H|^2 grad(chi)\n";
            std::cout << "  --skew_kelvin         Use DG skew form Kelvin force (default)\n\n";

            std::cout << "OUTPUT:\n";
            std::cout << "  --run_name NAME       Custom run name (default: auto-generated)\n";
            std::cout << "                        e.g., rosen-r5-amr, dome-r4, hedge-r5\n";
            std::cout << "  Results saved to: ../Results/<run_name>-<timestamp>/\n\n";

            std::cout << "PARALLEL DIAGNOSTICS:\n";
            std::cout << "  --parallel-diag           Record assembly/solve timing, sparsity, load balance\n";
            std::cout << "  --parallel-diag-all-ranks Also write per-rank CSV files\n\n";

            std::cout << "SPARSITY / RENUMBERING:\n";
            std::cout << "  --renumber-dofs           Apply Cuthill-McKee DoF renumbering (reduces bandwidth)\n";
            std::cout << "  --no-renumber-dofs        Disable DoF renumbering (default)\n";
            std::cout << "  --dump-sparsity           Export sparsity patterns (SVG + gnuplot + bandwidth CSV)\n\n";

            std::cout << "DEBUGGING:\n";
            std::cout << "  --mms                 MMS verification mode\n";
            std::cout << "  --magnetic            Enable magnetic forces (for droplet test)\n";
            std::cout << "  --no_magnetic         Disable magnetic forces\n";
            std::cout << "  --no_gravity          Disable gravity\n";
            std::cout << "  --no_ns               Disable Navier-Stokes\n";
            std::cout << "  --verbose             Verbose output\n\n";

            std::cout << "\n";

            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            std::cerr << "Use --help for usage information.\n";
            std::exit(1);
        }
    }

    // Finalize run_name after all parsing is done
    params.finalize_run_name();

    // Only print config from rank 0
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (params.output.verbose && rank == 0)
    {
        std::cout << "=== Configuration ===\n";
        std::cout << "  Run name: " << params.output.run_name << "\n";
        std::cout << "  Refinement: " << params.mesh.initial_refinement << "\n";
        std::cout << "  dt=" << params.time.dt << ", t_final=" << params.time.t_final << "\n";
        std::cout << "  Adaptive dt: " << (params.time.use_adaptive_dt ? "ON" : "OFF") << "\n";
        std::cout << "  AMR: " << (params.mesh.use_amr ? "ON" : "OFF");
        if (params.mesh.use_amr)
            std::cout << " (every " << params.mesh.amr_interval << " steps)";
        std::cout << "\n";
        std::cout << "  Physics: epsilon=" << params.physics.epsilon
                  << ", lambda=" << params.physics.lambda
                  << ", chi_0=" << params.physics.chi_0 << "\n";
        std::cout << "  CH convection: " << (params.physics.implicit_ch_convection ? "IMPLICIT" : "EXPLICIT") << "\n";
        std::cout << "  Solver: " << (params.solvers.ns.use_iterative ? "Iterative" : "Direct") << "\n";
        std::cout << "  Block-GS: " << (params.enable_bgs ? "ON" : "OFF");
        if (params.enable_bgs)
            std::cout << " (max " << params.bgs_max_iterations << " iters, tol=" << params.bgs_tolerance << ")";
        std::cout << "\n";
        std::cout << "  Picard: max " << params.picard_iterations
                  << " iters, tol=" << params.picard_tolerance
                  << ", omega=" << params.picard_omega << "\n";
        std::cout << "  Subsystems: "
                  << (params.enable_magnetic ? "Magnetic " : "")
                  << (params.enable_ns ? "NS " : "")
                  << (params.enable_gravity ? "Gravity " : "")
                  << (params.enable_mms ? "MMS " : "") << "\n";
        if (params.renumber_dofs)
            std::cout << "  DoF renumbering: Cuthill-McKee ON\n";
        if (params.dump_sparsity)
            std::cout << "  Sparsity export: ON (SVG + gnuplot + bandwidth)\n";

        std::cout << "=====================\n";
    }

    return params;
}