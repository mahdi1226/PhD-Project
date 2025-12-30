## ============================================================================
## CMakeLists.txt - Ferrofluid Phase Field Solver
## ============================================================================

cmake_minimum_required(VERSION 3.13)
project(ferrofluid LANGUAGES CXX)

# -----------------------------
# C++ standard (robust + explicit)
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Find deal.II
# -----------------------------
find_package(deal.II 9.5 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)
deal_ii_initialize_cached_variables()

# -----------------------------
# Source files
# -----------------------------
set(SOURCES
        main.cc

        # Core
        core/phase_field.cc
        core/phase_field_setup.cc
        core/phase_field_amr.cc

        # Assembly
        assembly/ch_assembler.cc
        assembly/poisson_assembler.cc
        assembly/magnetization_assembler.cc
        assembly/ns_assembler.cc

        # Solvers
        solvers/ch_solver.cc
        solvers/poisson_solver.cc
        solvers/magnetization_solver.cc
        solvers/ns_solver.cc
        solvers/ns_block_preconditioner.cc

        # Setup
        setup/ch_setup.cc
        setup/poisson_setup.cc
        setup/magnetization_setup.cc  # ADDED - was dead code, now active
        setup/ns_setup.cc

        # Diagnostics
        diagnostics/ch_diagnostics.cc
        diagnostics/poisson_diagnostics.cc
        diagnostics/ns_diagnostics.cc

        # Method of Manufactured Solution (MMS)
        mms/ch_mms.cc
        mms/poisson_mms.cc
        mms/ns_mms.cc
        mms/mms_runtime.cc

        # Utilities
        utilities/parameters.cc
)

# Header-only modules (IDE indexing only)
set(HEADERS
        physics/kelvin_force.h
        physics/skew_forms.h
        physics/material_properties.h
)

# -----------------------------
# Target
# -----------------------------
add_executable(ferrofluid ${SOURCES} ${HEADERS})

# Project include root (replaces global include_directories)
target_include_directories(ferrofluid PRIVATE
        ${CMAKE_SOURCE_DIR}
)

# Link deal.II
deal_ii_setup_target(ferrofluid)

# -----------------------------
# Warnings + build-type flags (generator-safe)
# -----------------------------
target_compile_options(ferrofluid PRIVATE
        -Wall -Wextra -Wpedantic

        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -march=native>
)

# (Optional) If you want faster rebuilds with Ninja:
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)