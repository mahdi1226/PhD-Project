##
#  CMakeLists.txt - Coupled Navier-Stokes/Cahn-Hilliard Solver
#
#  REFACTORED VERSION: Separate DoFHandlers for each scalar field
#  Enables proper SolutionTransfer during AMR (fixes deal.II 9.7 issue)
#
#  Based on: Nochetto, Salgado & Tomas (2016)
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

PROJECT(nsch_solver)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/core
        ${CMAKE_SOURCE_DIR}/assembly
        ${CMAKE_SOURCE_DIR}/solvers
        ${CMAKE_SOURCE_DIR}/output
        ${CMAKE_SOURCE_DIR}/diagnostics
        ${CMAKE_SOURCE_DIR}/utilities
)

# Target name
SET(TARGET "nsch_solver")

# Source files - REFACTORED VERSION with split nsch_problem files
SET(TARGET_SRC
        main.cc
        # Core problem class (split into multiple files for maintainability)
        core/nsch_problem.cc
        core/nsch_problem_setup.cc
        core/nsch_problem_solver.cc
        core/nsch_problem_output.cc
        core/nsch_problem_amr.cc
        # Solvers
        solvers/poisson_solver.cc
        # Diagnostics
        diagnostics/nsch_verification.cc
        # Output
        # Utilities
        utilities/nsch_parameters.cc
)

# Note: The following files are NO LONGER used in the refactored version:
#   - assembly/ch_assembler.cc  (replaced by scalar assembler in nsch_problem_solvers.cc)
#   - assembly/ns_assembler.cc  (replaced by scalar assembler in nsch_problem_solvers.cc)
#   - assembly/poisson_assembler.cc (replaced by scalar assembler in nsch_problem_solvers.cc)
#   - solvers/ns_solver.cc  (functionality moved to nsch_problem_solvers.cc)
#   - solvers/ch_solver.cc  (functionality moved to nsch_problem_solvers.cc)
#
# The following headers are still used for type definitions:
#   - utilities/nsch_linear_algebra.h (CHVector, NSVector types for compatibility)
#   - utilities/nsch_block_structure.h (block indices, may be removed in Phase 2)

DEAL_II_INVOKE_AUTOPILOT()