# validation/ — orphaned benchmark suite

These files are **NOT in the default CMake build** as of 2026-05-05.

## Contents

- `cavity_benchmark.cc` — Lid-driven cavity NS validation against Ghia,
  Ghia & Shin, J. Comp. Physics 48 (1982) 387–411. Standalone executable.
- `rosensweig_benchmark.cc` — Rosensweig instability validation, also
  standalone.
- `validation.h` — header used by both `.cc` files above.

## Status

The benchmarks were written in March 2026 against an earlier API
shape. They reference `utilities/mpi_tools.h` and may have minor
drift relative to current production code. The benchmarks themselves
are mathematically meaningful and worth preserving — they cover
external-paper validation (cavity / Ghia) that the MMS-only test
suite does not.

## To re-enable

Add to `Semi_Coupled/CMakeLists.txt`:

```cmake
add_executable(cavity_benchmark
        validation/cavity_benchmark.cc
        # plus shared sources used by validation
)
target_include_directories(cavity_benchmark PRIVATE ${CMAKE_SOURCE_DIR})
deal_ii_setup_target(cavity_benchmark)
```

Same pattern for `rosensweig_benchmark`. Expect to fix include paths
and any stale function signatures.

## Decision rationale

Kept in tree (not deleted) because the validation logic is non-trivial
and externally-anchored. Out of default build because it doesn't
actively support the current research workflow (MMS + Rosensweig
production runs cover what we need).
