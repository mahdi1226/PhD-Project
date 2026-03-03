# ============================================================================
# magnetization/CMakeLists.txt — Magnetization Subsystem Standalone MMS Test
# ============================================================================

add_executable(test_magnetization_mms
    tests/magnetization_mms_test.cc
)
target_link_libraries(test_magnetization_mms magnetization_lib utilities_lib)
deal_ii_setup_target(test_magnetization_mms)
