// ============================================================================
// utilities/nsch_parameters.cc - Parameters implementation
//
// Based on Nochetto, Salgado & Tomas (2016):
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#include "utilities/nsch_parameters.h"

#include <deal.II/base/patterns.h>
#include <iostream>
#include <algorithm>

void NSCHParameters::declare(dealii::ParameterHandler& prm)
{
    // Navier-Stokes (Nochetto Eq 17: phase-dependent viscosity)
    prm.enter_subsection("Navier-Stokes");
    {
        prm.declare_entry("nu_water",  "1.0", dealii::Patterns::Double(0.0),
                          "Water/air phase viscosity nu_w");
        prm.declare_entry("nu_ferro",  "2.0", dealii::Patterns::Double(0.0),
                          "Ferrofluid phase viscosity nu_f");
        prm.declare_entry("viscosity", "1.0", dealii::Patterns::Double(0.0),
                          "Constant viscosity (MMS mode only)");
        prm.declare_entry("grad_div_gamma", "0.0", dealii::Patterns::Double(0.0),
                          "Grad-div stabilization parameter");
    }
    prm.leave_subsection();

    // Cahn-Hilliard (Nochetto Eq 14a-14b)
    prm.enter_subsection("Cahn-Hilliard");
    {
        prm.declare_entry("epsilon",  "0.01",   dealii::Patterns::Double(0.0),
                          "Interface thickness parameter");
        prm.declare_entry("lambda",   "0.05",   dealii::Patterns::Double(0.0),
                          "Capillary/surface tension coefficient");
        prm.declare_entry("mobility", "0.0002", dealii::Patterns::Double(0.0),
                          "Mobility coefficient gamma");
    }
    prm.leave_subsection();

    // Magnetostatics (Nochetto Eq 14c-14d, Section 6.2)
    prm.enter_subsection("Magnetostatics");
    {
        prm.declare_entry("enable",            "false",  dealii::Patterns::Bool(),
                          "Enable magnetic field effects");
        prm.declare_entry("chi_m",             "0.5",    dealii::Patterns::Double(0.0),
                          "Magnetic susceptibility kappa_0");
        prm.declare_entry("mu_0",              "1.0",    dealii::Patterns::Double(0.0),
                          "Vacuum permeability");
        prm.declare_entry("dipole_intensity",  "6000.0", dealii::Patterns::Double(0.0),
                          "Maximum dipole intensity");
        prm.declare_entry("dipole_ramp_time",  "1.6",    dealii::Patterns::Double(0.0),
                          "Time to ramp dipole from 0 to max");
        prm.declare_entry("dipole_y_position", "-15.0",  dealii::Patterns::Double(),
                          "Y-position of dipole sources");
        prm.declare_entry("dipole_dir_x",      "0.0",    dealii::Patterns::Double(),
                          "Dipole direction x-component");
        prm.declare_entry("dipole_dir_y",      "1.0",    dealii::Patterns::Double(),
                          "Dipole direction y-component");
        prm.declare_entry("uniform_field",     "0.0",    dealii::Patterns::Double(0.0),
                          "Uniform vertical magnetic field strength B_0");
        prm.declare_entry("use_uniform_field", "false",  dealii::Patterns::Bool(),
                          "Use uniform field instead of dipoles");
    }
    prm.leave_subsection();

    // Gravity (Nochetto Eq 19: Boussinesq)
    prm.enter_subsection("Gravity");
    {
        prm.declare_entry("enable",        "true",    dealii::Patterns::Bool(),
                          "Enable gravity");
        prm.declare_entry("magnitude",     "30000.0", dealii::Patterns::Double(0.0),
                          "Gravity magnitude");
        prm.declare_entry("angle",         "-90.0",   dealii::Patterns::Double(),
                          "Gravity angle in degrees (-90 = downward)");
        prm.declare_entry("density_ratio", "0.1",     dealii::Patterns::Double(0.0),
                          "Density ratio r for Boussinesq");
    }
    prm.leave_subsection();

    // Domain (Nochetto Section 6.2)
    prm.enter_subsection("Domain");
    {
        prm.declare_entry("x_min", "0.0", dealii::Patterns::Double(),
                          "Domain left boundary");
        prm.declare_entry("x_max", "1.0", dealii::Patterns::Double(),
                          "Domain right boundary");
        prm.declare_entry("y_min", "0.0", dealii::Patterns::Double(),
                          "Domain bottom boundary");
        prm.declare_entry("y_max", "0.6", dealii::Patterns::Double(),
                          "Domain top boundary");
    }
    prm.leave_subsection();

    // Time stepping
    prm.enter_subsection("Time");
    {
        prm.declare_entry("dt",       "5e-4", dealii::Patterns::Double(0.0),
                          "Time step size");
        prm.declare_entry("t_final",  "2.0",  dealii::Patterns::Double(0.0),
                          "Final simulation time");
        prm.declare_entry("theta",    "1.0",  dealii::Patterns::Double(0.0, 1.0),
                          "Time discretization (1.0=BE, 0.5=CN)");
        prm.declare_entry("adaptive", "false", dealii::Patterns::Bool(),
                          "Enable adaptive time stepping");
    }
    prm.leave_subsection();

    // Coupling (Picard iteration)
    prm.enter_subsection("Coupling");
    {
        prm.declare_entry("use_picard",      "false", dealii::Patterns::Bool(),
                          "Enable Picard iteration for coupling");
        prm.declare_entry("picard_max_iter", "5",     dealii::Patterns::Integer(1),
                          "Maximum Picard iterations per time step");
        prm.declare_entry("picard_tol",      "1e-6",  dealii::Patterns::Double(0.0),
                          "Picard iteration tolerance");
    }
    prm.leave_subsection();

    // Finite elements (Nochetto Section 6: Q2-Q1 Taylor-Hood, l=2 for CH)
    prm.enter_subsection("FEM");
    {
        prm.declare_entry("fe_degree_velocity",  "2", dealii::Patterns::Integer(1),
                          "Velocity FE degree (Taylor-Hood)");
        prm.declare_entry("fe_degree_pressure",  "1", dealii::Patterns::Integer(1),
                          "Pressure FE degree (Taylor-Hood)");
        prm.declare_entry("fe_degree_phase",     "2", dealii::Patterns::Integer(1),
                          "Phase field FE degree");
        prm.declare_entry("fe_degree_potential", "2", dealii::Patterns::Integer(1),
                          "Magnetic potential FE degree");
        prm.declare_entry("refinements",         "4", dealii::Patterns::Integer(0),
                          "Initial uniform mesh refinement level");
    }
    prm.leave_subsection();

    // Initial condition (Nochetto Section 6.2)
    prm.enter_subsection("InitialCondition");
    {
        prm.declare_entry("ic_type",      "1",    dealii::Patterns::Integer(0, 2),
                          "IC type: 0=droplet, 1=flat layer, 2=perturbed layer");
        prm.declare_entry("layer_height", "0.2",  dealii::Patterns::Double(0.0),
                          "Ferrofluid layer height");
        prm.declare_entry("perturbation", "0.01", dealii::Patterns::Double(0.0),
                          "Perturbation amplitude");
        prm.declare_entry("pert_modes",   "5",    dealii::Patterns::Integer(1),
                          "Number of perturbation modes");
    }
    prm.leave_subsection();

    // Output
    prm.enter_subsection("Output");
    {
        prm.declare_entry("output_interval", "10",      dealii::Patterns::Integer(1),
                          "Output every N time steps");
        prm.declare_entry("output_dir",      "results", dealii::Patterns::Anything(),
                          "Output directory");
        prm.declare_entry("verbose",         "true",    dealii::Patterns::Bool(),
                          "Verbose console output");
    }
    prm.leave_subsection();

    // AMR
    prm.enter_subsection("AMR");
    {
        prm.declare_entry("enable",           "false", dealii::Patterns::Bool(),
                          "Enable adaptive mesh refinement");
        prm.declare_entry("interval",         "5",     dealii::Patterns::Integer(1),
                          "Refine every N time steps");
        prm.declare_entry("min_level",        "3",     dealii::Patterns::Integer(0),
                          "Minimum refinement level");
        prm.declare_entry("max_level",        "7",     dealii::Patterns::Integer(0),
                          "Maximum refinement level");
        prm.declare_entry("refine_fraction",  "0.3",   dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of cells to refine");
        prm.declare_entry("coarsen_fraction", "0.1",   dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of cells to coarsen");
        prm.declare_entry("indicator_type",   "0",     dealii::Patterns::Integer(0, 2),
                          "Error indicator: 0=gradient, 1=Kelly, 2=interface");
    }
    prm.leave_subsection();

    // MMS verification
    prm.enter_subsection("MMS");
    {
        prm.declare_entry("enable", "false", dealii::Patterns::Bool(),
                          "Enable MMS verification mode");
        prm.declare_entry("alpha",  "1.0",   dealii::Patterns::Double(),
                          "MMS parameter alpha");
        prm.declare_entry("beta",   "1.0",   dealii::Patterns::Double(),
                          "MMS parameter beta");
        prm.declare_entry("delta",  "1.0",   dealii::Patterns::Double(),
                          "MMS parameter delta");
    }
    prm.leave_subsection();

    // Convergence study
    prm.enter_subsection("Convergence");
    {
        prm.declare_entry("eoc_mode",       "false", dealii::Patterns::Bool(),
                          "Enable EOC convergence study");
        prm.declare_entry("min_refinement", "3",     dealii::Patterns::Integer(0),
                          "Minimum refinement for EOC");
        prm.declare_entry("max_refinement", "6",     dealii::Patterns::Integer(0),
                          "Maximum refinement for EOC");
    }
    prm.leave_subsection();
}


void NSCHParameters::parse(dealii::ParameterHandler& prm)
{
    prm.enter_subsection("Navier-Stokes");
    {
        nu_water      = prm.get_double("nu_water");
        nu_ferro      = prm.get_double("nu_ferro");
        viscosity     = prm.get_double("viscosity");
        grad_div_gamma = prm.get_double("grad_div_gamma");
    }
    prm.leave_subsection();

    prm.enter_subsection("Cahn-Hilliard");
    {
        epsilon  = prm.get_double("epsilon");
        lambda   = prm.get_double("lambda");
        mobility = prm.get_double("mobility");
    }
    prm.leave_subsection();

    prm.enter_subsection("Magnetostatics");
    {
        enable_magnetic        = prm.get_bool("enable");
        chi_m                  = prm.get_double("chi_m");
        mu_0                   = prm.get_double("mu_0");
        dipole_intensity       = prm.get_double("dipole_intensity");
        dipole_ramp_time       = prm.get_double("dipole_ramp_time");
        dipole_y_position      = prm.get_double("dipole_y_position");
        dipole_dir_x           = prm.get_double("dipole_dir_x");
        dipole_dir_y           = prm.get_double("dipole_dir_y");
        uniform_magnetic_field = prm.get_double("uniform_field");
        use_uniform_field      = prm.get_bool("use_uniform_field");
    }
    prm.leave_subsection();

    prm.enter_subsection("Gravity");
    {
        enable_gravity = prm.get_bool("enable");
        gravity        = prm.get_double("magnitude");
        gravity_angle  = prm.get_double("angle");
        density_ratio  = prm.get_double("density_ratio");
    }
    prm.leave_subsection();

    prm.enter_subsection("Domain");
    {
        x_min = prm.get_double("x_min");
        x_max = prm.get_double("x_max");
        y_min = prm.get_double("y_min");
        y_max = prm.get_double("y_max");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time");
    {
        dt              = prm.get_double("dt");
        t_final         = prm.get_double("t_final");
        theta           = prm.get_double("theta");
        use_adaptive_dt = prm.get_bool("adaptive");
    }
    prm.leave_subsection();

    prm.enter_subsection("Coupling");
    {
        use_picard      = prm.get_bool("use_picard");
        picard_max_iter = prm.get_integer("picard_max_iter");
        picard_tol      = prm.get_double("picard_tol");
    }
    prm.leave_subsection();

    prm.enter_subsection("FEM");
    {
        fe_degree_velocity  = prm.get_integer("fe_degree_velocity");
        fe_degree_pressure  = prm.get_integer("fe_degree_pressure");
        fe_degree_phase     = prm.get_integer("fe_degree_phase");
        fe_degree_potential = prm.get_integer("fe_degree_potential");
        n_refinements       = prm.get_integer("refinements");
    }
    prm.leave_subsection();

    prm.enter_subsection("InitialCondition");
    {
        ic_type                       = prm.get_integer("ic_type");
        rosensweig_layer_height       = prm.get_double("layer_height");
        rosensweig_perturbation       = prm.get_double("perturbation");
        rosensweig_perturbation_modes = prm.get_integer("pert_modes");
    }
    prm.leave_subsection();

    prm.enter_subsection("Output");
    {
        output_interval = prm.get_integer("output_interval");
        output_dir      = prm.get("output_dir");
        verbose         = prm.get_bool("verbose");
    }
    prm.leave_subsection();

    prm.enter_subsection("AMR");
    {
        use_amr              = prm.get_bool("enable");
        amr_interval         = prm.get_integer("interval");
        amr_min_level        = prm.get_integer("min_level");
        amr_max_level        = prm.get_integer("max_level");
        amr_refine_fraction  = prm.get_double("refine_fraction");
        amr_coarsen_fraction = prm.get_double("coarsen_fraction");
        amr_indicator_type   = prm.get_integer("indicator_type");
    }
    prm.leave_subsection();

    prm.enter_subsection("MMS");
    {
        mms_mode  = prm.get_bool("enable");
        mms_alpha = prm.get_double("alpha");
        mms_beta  = prm.get_double("beta");
        mms_delta = prm.get_double("delta");
    }
    prm.leave_subsection();

    prm.enter_subsection("Convergence");
    {
        eoc_mode       = prm.get_bool("eoc_mode");
        min_refinement = prm.get_integer("min_refinement");
        max_refinement = prm.get_integer("max_refinement");
    }
    prm.leave_subsection();
}


NSCHParameters parse_command_line(int argc, char* argv[])
{
    NSCHParameters params;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        // ====================================================================
        // Navier-Stokes (Eq 17)
        // ====================================================================
        if (arg == "--nu_water" && i + 1 < argc)
            params.nu_water = std::stod(argv[++i]);
        else if (arg == "--nu_ferro" && i + 1 < argc)
            params.nu_ferro = std::stod(argv[++i]);
        else if (arg == "--viscosity" && i + 1 < argc)
            params.viscosity = std::stod(argv[++i]);

        // ====================================================================
        // Cahn-Hilliard (Eq 14a-14b)
        // ====================================================================
        else if (arg == "--epsilon" && i + 1 < argc)
            params.epsilon = std::stod(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc)
            params.lambda = std::stod(argv[++i]);
        else if (arg == "--mobility" && i + 1 < argc)
            params.mobility = std::stod(argv[++i]);

        // ====================================================================
        // Magnetostatics (Eq 14c-14d, Section 6.2)
        // ====================================================================
        else if (arg == "--magnetic")
            params.enable_magnetic = true;
        else if (arg == "--no_magnetic")
            params.enable_magnetic = false;
        else if ((arg == "--chi_m" || arg == "--chi") && i + 1 < argc)
            params.chi_m = std::stod(argv[++i]);
        else if (arg == "--mu_0" && i + 1 < argc)
            params.mu_0 = std::stod(argv[++i]);
        else if (arg == "--dipole_intensity" && i + 1 < argc)
            params.dipole_intensity = std::stod(argv[++i]);
        else if (arg == "--dipole_ramp" && i + 1 < argc)
            params.dipole_ramp_time = std::stod(argv[++i]);
        else if (arg == "--dipole_y" && i + 1 < argc)
            params.dipole_y_position = std::stod(argv[++i]);
        else if (arg == "--dipole_dir_x" && i + 1 < argc)
            params.dipole_dir_x = std::stod(argv[++i]);
        else if (arg == "--dipole_dir_y" && i + 1 < argc)
            params.dipole_dir_y = std::stod(argv[++i]);
        // Uniform magnetic field (simpler Rosensweig model)
        else if ((arg == "--magnetic_field" || arg == "--B_field" || arg == "--B0") && i + 1 < argc)
        {
            params.uniform_magnetic_field = std::stod(argv[++i]);
            params.use_uniform_field = true;
            params.enable_magnetic = true;  // Auto-enable magnetic when field is set
        }

        // ====================================================================
        // Gravity (Eq 19: Boussinesq)
        // ====================================================================
        else if (arg == "--gravity")
            params.enable_gravity = true;
        else if (arg == "--no_gravity")
            params.enable_gravity = false;
        else if (arg == "--g" && i + 1 < argc)
            params.gravity = std::stod(argv[++i]);
        else if (arg == "--gravity_angle" && i + 1 < argc)
            params.gravity_angle = std::stod(argv[++i]);
        else if (arg == "--density_ratio" && i + 1 < argc)
            params.density_ratio = std::stod(argv[++i]);

        // ====================================================================
        // Numerical stabilization
        // ====================================================================
        else if ((arg == "--grad_div" || arg == "--grad_div_gamma") && i + 1 < argc)
            params.grad_div_gamma = std::stod(argv[++i]);

        // ====================================================================
        // Domain
        // ====================================================================
        else if (arg == "--x_min" && i + 1 < argc)
            params.x_min = std::stod(argv[++i]);
        else if (arg == "--x_max" && i + 1 < argc)
            params.x_max = std::stod(argv[++i]);
        else if (arg == "--y_min" && i + 1 < argc)
            params.y_min = std::stod(argv[++i]);
        else if (arg == "--y_max" && i + 1 < argc)
            params.y_max = std::stod(argv[++i]);

        // ====================================================================
        // Time stepping
        // ====================================================================
        else if (arg == "--dt" && i + 1 < argc)
            params.dt = std::stod(argv[++i]);
        else if (arg == "--t_final" && i + 1 < argc)
            params.t_final = std::stod(argv[++i]);
        else if (arg == "--theta" && i + 1 < argc)
            params.theta = std::stod(argv[++i]);
        else if (arg == "--adaptive")
            params.use_adaptive_dt = true;

        // ====================================================================
        // Coupling (Picard)
        // ====================================================================
        else if (arg == "--picard")
            params.use_picard = true;
        else if (arg == "--no_picard")
            params.use_picard = false;
        else if (arg == "--picard_max_iter" && i + 1 < argc)
            params.picard_max_iter = std::stoi(argv[++i]);
        else if (arg == "--picard_tol" && i + 1 < argc)
            params.picard_tol = std::stod(argv[++i]);

        // ====================================================================
        // Finite elements
        // ====================================================================
        else if (arg == "--refinement" && i + 1 < argc)
            params.n_refinements = std::stoi(argv[++i]);
        else if (arg == "--fe_velocity" && i + 1 < argc)
            params.fe_degree_velocity = std::stoi(argv[++i]);
        else if (arg == "--fe_pressure" && i + 1 < argc)
            params.fe_degree_pressure = std::stoi(argv[++i]);
        else if (arg == "--fe_phase" && i + 1 < argc)
            params.fe_degree_phase = std::stoi(argv[++i]);

        // ====================================================================
        // Initial condition
        // ====================================================================
        else if (arg == "--droplet")
            params.ic_type = 0;
        else if (arg == "--rosensweig_flat")
            params.ic_type = 1;
        else if (arg == "--rosensweig")
            params.ic_type = 2;
        else if (arg == "--ic_type" && i + 1 < argc)
            params.ic_type = std::stoi(argv[++i]);
        else if (arg == "--layer_height" && i + 1 < argc)
            params.rosensweig_layer_height = std::stod(argv[++i]);
        else if (arg == "--perturbation" && i + 1 < argc)
            params.rosensweig_perturbation = std::stod(argv[++i]);
        else if (arg == "--pert_modes" && i + 1 < argc)
            params.rosensweig_perturbation_modes = std::stoi(argv[++i]);

        // ====================================================================
        // Output
        // ====================================================================
        else if (arg == "--output_interval" && i + 1 < argc)
            params.output_interval = std::stoi(argv[++i]);
        else if (arg == "--output_dir" && i + 1 < argc)
            params.output_dir = argv[++i];
        else if (arg == "--quiet")
            params.verbose = false;
        else if (arg == "--verbose")
            params.verbose = true;

        // ====================================================================
        // AMR
        // ====================================================================
        else if (arg == "--amr")
            params.use_amr = true;
        else if (arg == "--no_amr")
            params.use_amr = false;
        else if (arg == "--amr_interval" && i + 1 < argc)
            params.amr_interval = std::stoi(argv[++i]);
        else if (arg == "--amr_min" && i + 1 < argc)
            params.amr_min_level = std::stoi(argv[++i]);
        else if (arg == "--amr_max" && i + 1 < argc)
            params.amr_max_level = std::stoi(argv[++i]);
        else if (arg == "--amr_refine" && i + 1 < argc)
            params.amr_refine_fraction = std::stod(argv[++i]);
        else if (arg == "--amr_coarsen" && i + 1 < argc)
            params.amr_coarsen_fraction = std::stod(argv[++i]);
        else if (arg == "--amr_indicator" && i + 1 < argc)
            params.amr_indicator_type = std::stoi(argv[++i]);

        // ====================================================================
        // MMS / Convergence
        // ====================================================================
        else if (arg == "--mms")
            params.mms_mode = true;
        else if (arg == "--mms_alpha" && i + 1 < argc)
            params.mms_alpha = std::stod(argv[++i]);
        else if (arg == "--mms_beta" && i + 1 < argc)
            params.mms_beta = std::stod(argv[++i]);
        else if (arg == "--mms_delta" && i + 1 < argc)
            params.mms_delta = std::stod(argv[++i]);
        else if (arg == "--eoc")
            params.eoc_mode = true;
        else if (arg == "--min_refinement" && i + 1 < argc)
            params.min_refinement = std::stoi(argv[++i]);
        else if (arg == "--max_refinement" && i + 1 < argc)
            params.max_refinement = std::stoi(argv[++i]);

        // ====================================================================
        // Help
        // ====================================================================
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << R"(
Ferrofluid NS-CH Solver
Based on Nochetto, Salgado & Tomas (2016)

NAVIER-STOKES (Eq 17: phase-dependent viscosity)
  --nu_water <val>       Water/air viscosity nu_w [1.0]
  --nu_ferro <val>       Ferrofluid viscosity nu_f [2.0]
  --viscosity <val>      Constant viscosity (MMS mode) [1.0]
  --grad_div <val>       Grad-div stabilization [0.0]

CAHN-HILLIARD (Eq 14a-14b)
  --epsilon <val>        Interface thickness [0.01]
  --lambda <val>         Capillary coefficient [0.05]
  --mobility <val>       Mobility gamma [0.0002]

MAGNETOSTATICS (Eq 14c-14d, Section 6.2)
  --magnetic / --no_magnetic   Enable/disable magnetic effects
  --chi_m <val>                Susceptibility kappa_0 [0.5]
  --chi <val>                  (alias for --chi_m)
  --mu_0 <val>                 Permeability [1.0]
  --dipole_intensity <val>     Dipole intensity [6000]
  --dipole_ramp <val>          Ramp time [1.6]
  --dipole_y <val>             Dipole y-position [-15]
  --dipole_dir_x <val>         Dipole direction x [0.0]
  --dipole_dir_y <val>         Dipole direction y [1.0]
  --magnetic_field <val>       Uniform vertical B-field (auto-enables magnetic)
  --B_field <val>              (alias for --magnetic_field)
  --B0 <val>                   (alias for --magnetic_field)

GRAVITY (Eq 19: Boussinesq)
  --gravity / --no_gravity     Enable/disable gravity
  --g <val>                    Gravity magnitude [30000]
  --gravity_angle <val>        Angle in degrees [-90]
  --density_ratio <val>        Density ratio r [0.1]

DOMAIN (Section 6.2)
  --x_min/--x_max <val>   Domain x-bounds [0, 1]
  --y_min/--y_max <val>   Domain y-bounds [0, 0.6]

TIME STEPPING
  --dt <val>             Time step tau [5e-4]
  --t_final <val>        Final time t_F [2.0]
  --theta <val>          Time scheme (1=BE, 0.5=CN) [1.0]
  --adaptive             Enable adaptive dt

COUPLING
  --picard / --no_picard       Enable/disable Picard iteration
  --picard_max_iter <n>        Max Picard iterations [5]
  --picard_tol <val>           Picard tolerance [1e-6]

FINITE ELEMENTS
  --refinement <n>       Mesh refinement level [4]
  --fe_velocity <n>      Velocity degree [2]
  --fe_pressure <n>      Pressure degree [1]
  --fe_phase <n>         Phase field degree [2]

INITIAL CONDITION
  --rosensweig           Perturbed layer (ic_type=2)
  --rosensweig_flat      Flat layer (ic_type=1)
  --droplet              Circular droplet (ic_type=0)
  --layer_height <val>   Pool depth [0.2]
  --perturbation <val>   Perturbation amplitude [0.01]
  --pert_modes <n>       Number of modes [5]

OUTPUT
  --output_interval <n>  Output frequency [10]
  --output_dir <path>    Output directory [results]
  --quiet / --verbose    Control output verbosity

AMR (Adaptive Mesh Refinement)
  --amr / --no_amr       Enable/disable AMR
  --amr_interval <n>     Refinement interval [5]
  --amr_min <n>          Minimum refinement level [3]
  --amr_max <n>          Maximum refinement level [7]
  --amr_refine <val>     Refine fraction [0.3]
  --amr_coarsen <val>    Coarsen fraction [0.1]
  --amr_indicator <n>    0=gradient, 1=Kelly, 2=interface [0]

MMS / CONVERGENCE
  --mms                  Enable MMS verification mode
  --mms_alpha/beta/delta MMS parameters [1.0]
  --eoc                  Enable EOC convergence study
  --min/max_refinement   EOC refinement range [3, 6]

EXAMPLE (Rosensweig with dipoles):
  ./nsch_solver --rosensweig \
    --epsilon 0.01 --lambda 0.05 --mobility 0.0002 \
    --nu_water 1.0 --nu_ferro 2.0 \
    --x_min 0 --x_max 1 --y_min 0 --y_max 0.6 \
    --layer_height 0.2 --perturbation 0.03 --pert_modes 5 \
    --magnetic \
    --chi_m 0.5 --dipole_intensity 6000 --dipole_ramp 1.6 --dipole_y -15 \
    --g 30000 --density_ratio 0.1 \
    --dt 5e-4 --t_final 2.0 \
    --refinement 4 --amr \
    --amr_min 3 --amr_max 7 --amr_interval 10 \
    --grad_div 1.0
)";
            std::exit(0);
        }
        else if (arg[0] == '-')
        {
            std::cerr << "Warning: Unknown option '" << arg << "' (ignored)\n";
        }
    }

    return params;
}