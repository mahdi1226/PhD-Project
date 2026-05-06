# Stale: NS variable-viscosity MMS test

`ns_variable_nu_mms_test.{cc,h}` and `ns_variable_nu_mms.h` in this
directory are **not in the build** and do not currently compile against
the production API.

## What it tested

Variable viscosity ν(θ) via prescribed-θ MMS — an intermediate step
between standalone NS (constant ν) and full coupled CH→NS. Useful for
isolating the ν(θ) coupling from CH dynamics.

## Why it's broken

Written before NS was parallelized. References the old
`assemble_ns_system()` / `setup_ns_coupled_system()` (no `_parallel`
suffix) — those signatures no longer exist in the current
`assembly/ns_assembler.h` and `setup/ns_setup.h`.

## To revive

1. Replace `assemble_ns_system` calls with `assemble_ns_system_parallel`
   or `assemble_ns_system_unified` (current API in `assembly/ns_assembler.h`).
2. Replace `setup_ns_coupled_system` with the corresponding parallel
   setup function in `setup/ns_setup.h`.
3. Add to `CMakeLists.txt` as a new executable (`parallel_test_ns_variable_nu`?).
4. Verify variable-viscosity MMS source still matches the variable-ν
   path now in `assemble_ns_core` (look for `theta_dof_handler != nullptr`).

## Decision rationale

Kept in tree (not deleted) because the variable-ν MMS may be useful
when investigating the ν(θ) coupling. Marked stale because reviving it
costs ~half a day to update API calls, and that work isn't on the
current critical path.
