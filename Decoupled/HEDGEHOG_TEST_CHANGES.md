# Hedgehog Test — Temporary Code Changes Log

**Date**: 2026-03-12
**Purpose**: Diagnose hedgehog crash at t≈2.2. All changes below are TEMPORARY
and should be reverted after testing.

## How to Revert Everything
```bash
git checkout -- utilities/parameters.cc navier_stokes/navier_stokes_assemble.cc physics/applied_field.h
```

---

## Change 1: Command Line Overrides (parameters.cc only)

**File**: `utilities/parameters.cc`

**What**: Added NEW command line flags (after line ~229, marked "TEMP"):
- `--chi0 VALUE` — override magnetic susceptibility
- `--mesh NxM` — override mesh (e.g. `--mesh 150x90`)
- `--ramp-slope VALUE` — override dipole ramp slope

Also MODIFIED existing `--dt` handler to auto-recompute `max_steps`
from `t_final / dt` so the run reaches the same end time.

Updated help text to list the new flags.

**Why**: Allows running multiple hedgehog parameter variations from
the same binary without recompiling each time.

**Where in code**: New flags added after `--dt` block in `parse_command_line()`.
Also changed Rosensweig t_final from 2.2 to 3.0 and max_steps from 2200 to 3000
(this was done BEFORE the hedgehog work, for the ongoing Rosensweig run).

---

## Change 2: Kelvin Force Fix (navier_stokes_assemble.cc)

**File**: `navier_stokes/navier_stokes_assemble.cc`

**What**: In the NS velocity assembly, the Kelvin force computation:
- BEFORE: H = ∇φ only, ∇H = Hess(φ) only
- AFTER:  H = h_a + ∇φ,  ∇H = ∇h_a + Hess(φ)

Added calls to `compute_applied_field()` and
`compute_applied_field_gradient()` from `physics/applied_field.h`.

**Why**: For nonuniform applied fields (hedgehog dipoles at y=-0.5 to -1.0),
the applied field h_a and its gradient ∇h_a are significant. Missing them
means the Kelvin force μ₀(M·∇)H is incomplete.

**Note**: This change is ONLY in BIN-1. BIN-0 uses the original code.

---

## Binary Map

| Binary | Location | Contains |
|--------|----------|----------|
| BIN-0  | `build/bin0_hedgehog` | Change 1 only (overrides) |
| BIN-1  | `build/bin1_hedgehog` | Change 1 + Change 2 (overrides + Kelvin fix) |

---

## Test Matrix (12 tests)

| # | Binary | Command | Tests |
|---|--------|---------|-------|
| T1  | BIN-0 | `--hedgehog` | Baseline crash |
| T2  | BIN-0 | `--hedgehog --dt 1e-4` | CFL stability |
| T3  | BIN-0 | `--hedgehog --dt 5e-5` | Aggressive CFL |
| T4  | BIN-0 | `--hedgehog --chi0 0.5` | Susceptibility |
| T5  | BIN-0 | `--hedgehog --mesh 150x90` | Resolution |
| T6  | BIN-0 | `--hedgehog --ramp-slope 0.6` | Slower ramp |
| T7  | BIN-1 | `--hedgehog` | Kelvin fix alone |
| T8  | BIN-1 | `--hedgehog --dt 1e-4` | Kelvin fix + finer dt |
| T9  | BIN-1 | `--hedgehog --chi0 0.5` | Kelvin fix + easy χ₀ |
| T10 | BIN-1 | `--hedgehog --mesh 150x90` | Kelvin fix + finer mesh |
| T11 | BIN-1 | `--hedgehog --dt 1e-4 --mesh 150x90` | Kelvin fix + both |
| T12 | BIN-1 | `--hedgehog --ramp-slope 0.6` | Kelvin fix + slower ramp |

---

## Change 3: Field Shutoff Test (applied_field.h)

**File**: `physics/applied_field.h`

**What**: Added `if (current_time > 2.5) alpha = 0.0;` in 3 places:
- `compute_applied_field()` uniform path (line ~75)
- `compute_applied_field()` dipole path (line ~112)
- `compute_applied_field_gradient()` dipole path (line ~204)

Also changed Rosensweig t_final from 3.0 to 4.0, max_steps 3000 to 4000.

**Why**: Test relaxation dynamics when field is suddenly removed at t=2.5.
Spikes should collapse back to flat under gravity + surface tension.

**Note**: This is in the MAIN build binary, NOT in BIN-0 or BIN-1.
Remove these lines before running hedgehog tests on HPC.
