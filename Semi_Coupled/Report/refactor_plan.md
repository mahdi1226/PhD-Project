# Semi_Coupled Refactoring Plan

Audit date: 2026-03-20
Total LOC at audit: ~31,150 (before cleanup)
Deleted so far: ~2,200 lines (7 dead/duplicate/stub files)

---

## COMPLETED

### Dead File Deletion
- [x] `core/amr_postprocess.h` (311 lines) — serial vectors, never included
- [x] `physics/initial_conditions.h` (221 lines) — dead, ICs defined inline in phase_field_setup.cc
- [x] `diagnostics/magnetization_diagnostics.h` (363 lines) — exact duplicate of magnetic_diagnostics.h
- [x] `mms/poisson/poisson_mms.h` (~150 lines) — duplicate of mms/magnetic/poisson_mms.h
- [x] `mms/magnetization/magnetization_mms.h` (~150 lines) — duplicate of mms/magnetic/magnetization_mms.h
- [x] `utilities/questions.h` (379 lines) — tracking file disguised as a header
- [x] `validation/bubble_benchmark.cc` (622 lines) — stub, run_simulation() entirely commented out

---

## TODO — HIGH PRIORITY

### 1. Extract shared `try_direct_solver()` utility
Three nearly identical copies of MUMPS -> SuperLU_DIST -> KLU -> GMRES+ILU fallback cascade:
- `solvers/ch_solver.cc` lines 29-81
- `solvers/ns_solver.cc` lines 145-197
- `solvers/magnetic_solver.cc` lines 56-131

**Action:** Create `solvers/solver_utils.h` with a shared template function.

### 2. Split `run()` in `core/phase_field.cc`
~674 lines. Break into:
- `run_time_loop()`
- `output_results_if_needed()`
- `print_step_summary()`
- `ramp_applied_field()`

### 3. Split `refine_mesh()` in `core/phase_field_amr.cc`
~480 lines. Break into:
- `compute_refinement_indicators()`
- `execute_refinement()`
- `transfer_solutions()`
- `rebuild_systems()`

---

## TODO — MEDIUM PRIORITY

### 4. Move large header implementations to .cc files
| Header | Lines | Contents |
|--------|-------|----------|
| `diagnostics/validation_diagnostics.h` | 563 | Template functions, logger class |
| `diagnostics/magnetic_diagnostics.h` | 498 | Structs, logger class, free functions |
| `utilities/tools.h` | 402 | ensure_directory (uses system()!), write_run_info |
| `output/metrics_logger.h` | 380 | Full class |
| `output/parallel_diagnostics_logger.h` | 383 | Full class |
| `utilities/sparsity_export.h` | 315 | Full analysis + SVG export |
| `utilities/run_tracker.h` | 258 | Full class |

### 5. Deduplicate preset setup functions
`setup_dome()` in `parameters.cc` duplicates ~70 lines of `setup_hedgehog()`.
**Action:** Call `setup_hedgehog()` then override differences.

### 6. Remove unused serial overloads
Every diagnostics header has both parallel (Trilinos) and serial (`dealii::Vector<double>`) overloads.
The serial versions appear unused in the fully-parallel codebase.
Files: `ch_diagnostics.h`, `ns_diagnostics.h`, `magnetic_diagnostics.h`, `force_diagnostics.h`, `interface_tracking.h`, `system_diagnostics.h`

### 7. Deduplicate `smooth_heaviside` / `sigmoid`
Same function under different names:
- `diagnostics/magnetic_diagnostics.h` → `detail::smooth_heaviside`
- `diagnostics/force_diagnostics.h` → `force_detail::sigmoid`
- `physics/material_properties.h` → also has similar logic

**Action:** Single shared utility function.

### 8. Duplicate MMS files (already partially cleaned)
Verify `mms/magnetic/poisson_mms.h` and `mms/magnetic/magnetization_mms.h` are the canonical copies used by all remaining MMS code.

---

## TODO — LOW PRIORITY

### 9. Remove testing macros from production code
`utilities/tools.h` lines 33-34: `#define TEST_ID "A"` and `#define TEST_H_FORMULA`

### 10. Hardcoded magic numbers
| File | Value | Description |
|------|-------|-------------|
| `solvers/ns_solver.cc` | `LAPACK_SIZE_LIMIT = 50000` | Threshold for LAPACK vs Trilinos |
| `solvers/ns_solver.cc` | `1e-8`, `1500`, `100` | Tolerances, max iterations, basis size |
| `solvers/magnetic_solver.cc` | `1e-14`, `1e-12`, `2000` | Zero guard, tolerance, max iterations |
| `core/phase_field_amr.cc` | `0.95` | bulk_threshold |

### 11. Inconsistent patterns
- `MagneticAssembler` is a class, but CH/NS assemblers are free functions
- `MagneticSolver` and `BlockSchurPreconditionerParallel` are classes, but CH/NS solvers use free functions
- Cell iteration: some use `begin_active()/end()`, others use range-based for

### 12. Commented-out debug blocks
- `solvers/ns_solver.cc` lines 128-137

### 13. Suppressed error return
- `solvers/ns_block_preconditioner.cc` line 280: `(void)err;` ignoring `InsertGlobalValues` return
