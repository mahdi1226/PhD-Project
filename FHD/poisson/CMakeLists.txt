# ============================================================================
# poisson/CMakeLists.txt — Poisson Subsystem Standalone MMS Test
# ============================================================================

add_executable(test_poisson_mms
    tests/poisson_mms_test.cc
)
target_link_libraries(test_poisson_mms poisson_lib utilities_lib)
deal_ii_setup_target(test_poisson_mms)
