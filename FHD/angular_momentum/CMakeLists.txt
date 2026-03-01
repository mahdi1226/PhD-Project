# ============================================================================
# angular_momentum/CMakeLists.txt — Angular Momentum Standalone MMS Test
# ============================================================================

add_executable(test_angular_momentum_mms
    tests/angular_momentum_mms_test.cc
)
target_link_libraries(test_angular_momentum_mms angular_momentum_lib utilities_lib)
deal_ii_setup_target(test_angular_momentum_mms)
