# ============================================================================
# navier_stokes/CMakeLists.txt — NS Subsystem Standalone MMS Test
# ============================================================================

add_executable(test_navier_stokes_mms
    tests/navier_stokes_mms_test.cc
)
target_link_libraries(test_navier_stokes_mms navier_stokes_lib utilities_lib)
deal_ii_setup_target(test_navier_stokes_mms)
