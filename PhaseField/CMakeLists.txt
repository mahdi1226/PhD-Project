##
#  CMakeLists.txt - Two-Phase Ferrofluid Solver
#
#  Coupled Navier-Stokes / Cahn-Hilliard / Magnetization / Poisson System
#
#  Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
#  "A diffuse interface model for two-phase ferrofluid flows"
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.5...3.30)

FIND_PACKAGE(deal.II 9.0.0 QUIET
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
        "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
        "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
        "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_UMFPACK)
    MESSAGE(FATAL_ERROR "\n"
        "*** deal.II must be configured with UMFPACK support. ***"
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(ferrofluid)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/assembly
    ${CMAKE_SOURCE_DIR}/solvers
    ${CMAKE_SOURCE_DIR}/physics
    ${CMAKE_SOURCE_DIR}/output
    ${CMAKE_SOURCE_DIR}/diagnostics
    ${CMAKE_SOURCE_DIR}/utilities
)

# Target name
SET(TARGET "ferrofluid")

# Source files
SET(TARGET_SRC
    main.cc
    
    # Core - Orchestration
    core/phase_field.cc
    core/phase_field_setup.cc
    core/phase_field_amr.cc
    
    # Assembly
    assembly/ns_assembler.cc
    assembly/ch_assembler.cc
    assembly/magnetization_assembler.cc
    assembly/poisson_assembler.cc
    
    # Solvers
    solvers/ns_solver.cc
    solvers/ch_solver.cc
    solvers/magnetization_solver.cc
    solvers/poisson_solver.cc
    
    # Physics
    physics/applied_field.cc
    physics/material_properties.cc
    physics/initial_conditions.cc
    physics/boundary_conditions.cc
    
    # Output
    output/vtk_writer.cc
    output/logger.cc
    
    # Diagnostics
    diagnostics/verification.cc
    diagnostics/spikes.cc
    
    # Utilities
    utilities/parameters.cc
    utilities/trilinear_forms.cc
)

DEAL_II_INVOKE_AUTOPILOT()


#./ferrofluid --t-final 0.005