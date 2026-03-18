# Semi_Coupled HPC Improvement Audit

---

## Part 1: Non-Solver Audit

### 🔴 High Priority

#### 1. BGS temporary vectors allocated every timestep
**File:** `core/phase_field.cc`

`theta_bgs_prev`, `ux_bgs_prev`, `uy_bgs_prev` + convergence diff vectors are created fresh each step. With 4000 steps = 12,000+ Trilinos vector constructions.

**Fix:** Make them persistent class members.

#### 2. h_min recomputed every timestep
**File:** `core/phase_field.cc`

`GridTools::minimal_cell_diameter()` walks all cells + MPI_Allreduce. Only changes after AMR (every 5 steps), wasted on the other 3600+ steps.

**Fix:** Cache and recompute only after AMR.

#### 3. extract_magnetic_components() does cell-by-cell extraction every step
**File:** `core/phase_field_setup.cc`

After every magnetics solve, iterates all cells with 3 synchronized DoFHandlers to extract Mx, My, phi from the monolithic solution.

**Fix:** Build an index map once during setup, then extraction becomes a simple vector copy with index remapping — no cell iteration.

### 🟡 Medium Priority

#### 4. 6 separate diagnostics cell loops every timestep
CH diagnostics, magnetic diagnostics, NS diagnostics, interface tracking, force diagnostics, and validation diagnostics (256-sample interface profiling) — all run every step with their own FEValues.

**Fix:** Run heavy diagnostics (force, validation) every N steps. Gate Rosensweig validation behind a dedicated flag instead of `enable_magnetic`.

#### 5. Parallel diagnostics: ~15 MPI_Allreduce calls
**File:** `diagnostics/parallel_data.h`

When `--parallel-diag` is enabled. The `compute_imbalance` lambda alone does 6 Allreduce calls.

**Fix:** Pack into 3 batched Allreduce calls (SUM, MIN, MAX).

### 🟢 Minor

- Duplicate serial/parallel diagnostics code — every diagnostics header has a full serial overload used only by MMS tests. Dead weight for production.
- Duplicate heaviside/susceptibility in `magnetic_diagnostics.h` — redeclares functions from `material_properties.h`.
- `linfty_norm()` called every CH assembly — global MPI reduction just to check if velocity is nonzero. Should be a cached boolean flag.
- VTK output allocates temporary ghosted vectors each call — should be persistent members.
- Possibly dead `amr_postprocess.h` — serial AMR post-processor, likely unused in production.

### ✅ Well Done

- AMR frequency controlled (every N steps)
- MetricsLogger flushes every 10 steps (not every step)
- Clean BGS coupling structure (CH->Mag->NS)
- Monolithic magnetics assembly is efficient
- NS assembly handles variable viscosity + Kelvin force correctly

### Top 3 Actions by Wall-Time Impact

| # | Action | Saves |
|---|--------|-------|
| 1 | Cache h_min, recompute only after AMR | ~3600 unnecessary global cell walks + Allreduce |
| 2 | Persistent BGS vectors | ~12,000 Trilinos vector allocations |
| 3 | Index-map extraction for magnetics | Cell-by-cell 3-DoFHandler loop -> simple vector copy every step |

---

## Part 2: Solver Audit

### 🔴 Critical (P0)

#### 1. 8 temporary vectors allocated per vmult call
**File:** `ns_block_preconditioner.cc`

vmult is called ~50-100 times per outer FGMRES iteration, every timestep. Each call creates 8 Trilinos MPI vectors (`r_vel`, `r_p`, `z_p`, `z_p_ghosted`, `Bt_zp`, `rhs_vel`, `z_vel`, `p_relevant`). Over 2000 steps that's potentially millions of unnecessary MPI-touching allocations.

**Fix:** Cache as mutable class members — the MagneticBlockPreconditioner already does this correctly (lines 356-365 with `tmp_initialized_` flag). Copy that pattern.

#### 2. MPI_Barrier inside every vmult call
**File:** `ns_block_preconditioner.cc`

A synchronization barrier at the start of every vmult. With ~100K vmult calls over a run, this serializes all ranks for no reason. Likely a debug leftover.

**Fix:** Remove or gate behind `verbose_` flag.

### 🟡 High Priority (P1)

#### 3. Block Schur preconditioner rebuilt from scratch every timestep
**File:** `ns_solver.cc`

The constructor extracts the velocity block (two full Epetra row scans), builds sparsity, fills the matrix, and initializes ILU/AMG. For ~50k velocity DOFs, this is 0.1-0.5s per step x 2000 steps = 200-1000s wasted.

**Fix:** Cache the preconditioner. If sparsity is unchanged, only update values and re-initialize the smoother.

#### 4. Magnetic preconditioner rebuilt every call
**File:** `magnetic_solver.cc`

Same issue — MagneticBlockPreconditioner constructed fresh each step with block extraction + AMG/ILU setup.

#### 5. CH preconditioner rebuilt every call
**File:** `ch_solver.cc`

AMG on the indefinite coupled system is expensive to set up. Could be cached when sparsity unchanged.

### 🟡 Medium Priority (P2)

#### 6. True residual computed after NS solve
**File:** `ns_solver.cc`

Allocates a full-size vector + SpMV just to log the residual. Redundant when the iterative solver already tracks convergence.

**Fix:** Make optional (verbose only).

#### 7. ReadWriteVector allocated per apply_C_M_phi call
**File:** `magnetic_block_preconditioner.cc`

MPI communication + heap allocation every coupling application. Should be cached.

#### 8. Ghosted pressure IndexSet rebuilt per vmult
**File:** `ns_block_preconditioner.cc`

`p_relevant` reconstructed from fixed IndexSets every call. Cache it.

#### 9. const_cast mutates caller's NS matrix
**File:** `ns_solver.cc`

The direct solver path zeros a row for pressure pinning via `const_cast`. If the caller reuses the matrix, it's silently corrupted. Correctness risk.

### 🟢 Minor (P3)

- **Direct solver chain** tries all 5 backends sequentially (`direct_solver_utils.h`). On The Mill only KLU works. Each failed attempt has exception handling overhead. Add a preferred-backend option.
- **Verbose output not rank-gated** in vmult — all ranks print to stdout, flooding output.

### ✅ Well Done

- MagneticBlockPreconditioner correctly caches temp vectors (lazy init pattern)
- Direct solver fallback chain is robust
- CH AMG settings appropriate for indefinite system (`elliptic=false`, Chebyshev smoother)
- `solver_info.h` clean data structure

### Priority Summary

| # | Fix | Impact |
|---|-----|--------|
| 1 | Cache vmult temp vectors (copy MagBlock pattern) | 10-30% NS solve time |
| 2 | Remove MPI_Barrier from vmult | Eliminates ~100K synchronization points |
| 3 | Cache preconditioners across timesteps | 200-1000s over full run |
| 4 | Make post-solve residual optional | ~5% NS solve time |
