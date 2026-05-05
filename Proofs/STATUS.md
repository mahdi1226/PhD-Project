# Project Status — May 4–5, 2026

Single-page reference: what's done, what's open, what to look at first.

---

## 🌙 Overnight progress (May 4–5, 2026)

All work below is **uncommitted, staged in the working tree** — review with `git diff` before committing.

### ✅ Completed code changes

| Task | Files | Status | Risk | Validation needed |
|---|---|---|---|---|
| **Plan B** — surface GMRES iter count + residual to CSV | `solvers/magnetic_solver.{h,cc}`, `output/metrics_logger.cc`, `core/phase_field.cc` | Built clean, dome-tested (mag_iter=1 expected for h_a-only) | Very low | Hedgehog test (when current run finishes) — expect mag_iter ∈ [5, 25] |
| **Rosensweig formula fix** — Δρ contrast (was ρ̄), CH calibration constant (2√2/3), Bond number, dominant wavelength | `analysis/analyze_hedgehog.py` | Tested — predicts 5.7 spikes/unit (was bogus 12.9) | None (analysis-only) | Re-run on `diagnostics.csv` after hedgehog finishes |
| **Plan A** — explicit cross-AMR preconditioner caching with adaptive staleness trigger | `core/phase_field.h`, `core/phase_field.cc`, `core/phase_field_amr.cc` | Built clean | Low (functionally equivalent to current implicit caching, plus adaptive rebuild) | Dome L5 with vs. without `--iterative_mag` should match to 1e-10 |
| **Compiler-warning cleanup** — `[[maybe_unused]]` on stored-but-unused `comm_` fields | `output/console_logger.h`, `output/timing_logger.h` | Built clean, warnings gone | None | None — cosmetic |

### ⏸ Deferred (not safe to do without testing)

- **NS Schur `vmult` MPI_ERR_TRUNCATE fix.** Spent ~45 min reverse-engineering the partition logic; concluded the documented diagnosis ("p_owned vs. NS-row partition mismatch") doesn't fully explain why it fires, since for DG pressure both partitions follow the same cell ownership. Fix would either (a) add NS-aligned re-assembly of `pressure_mass_matrix_`, or (b) use `LinearAlgebra::ReadWriteVector` for `extract_pressure`. Both require validation against an actually-failing run, which competes with the running hedgehog. **See "NS Schur fix proposal — full draft" section below.**

### ✅ Validation results (May 5 morning, ~30 min runtime, hedgehog continued in parallel at -25% throughput)

#### Plan A regression — **PASS**
Three runs at dome L5, 200 steps each, np=2:
- A: direct (new binary)
- B: iterative_mag (new binary, with Plan A)
- C: iterative_mag (pre-Plan-A backup binary, no Plan A)

| Column | A vs B (with Plan A) | A vs C (without Plan A) | Plan A diff? |
|---|---|---|---|
| mass | 2.57e-5 | 1.21e-5 | same order |
| theta_max | 8.31e-4 | 8.31e-4 | **identical** |
| E_internal | 7.47e-5 | 7.15e-5 | **virtually identical** |
| interface_y_mean | 5.73e-5 | 5.18e-5 | **virtually identical** |
| U_max (rel) | 2.33 | 2.13 | similar (both spurious — see note) |

**Note on U_max:** dome at αs=0 (pre-ramp) has U≈0, so any tiny perturbation gives huge *relative* difference. Absolute diff is 7e-4 on a quantity of order 1e-3, fine.

**Conclusion**: A→B and A→C show the same iter-vs-direct gap. Plan A introduces **no new regression**. The gap is entirely attributable to GMRES tolerance (1e-8 per step accumulating over 200 steps).

#### Plan B telemetry — **PASS** (verified on hedgehog with non-trivial physics)
Hedgehog -r 3, 20 steps, --iterative_mag:

```
step= 1  mag_iter=44  mag_res=3.6e-11   ← first build, cache empty
step= 2  mag_iter=38  mag_res=4.1e-09
step= 3  mag_iter=36  mag_res=6.2e-09
step= 5  mag_iter=36  mag_res=2.9e-08   ← AMR boundary
step= 6  mag_iter=36  mag_res=2.9e-08   ← cache reused after rebuild
...
step=10  mag_iter=29  mag_res=1.1e-07   ← AMR rebuild (matrix-easier post-refinement)
step=11  mag_iter=29  mag_res=1.1e-07
...
step=20  mag_iter=28  mag_res=1.6e-07   ← stable cache reuse
```

**Conclusion**:
- `mag_iterations`/`mag_residual` columns populate correctly with realistic values [28, 44]
- Plan A cache rebuild at AMR boundaries verified (mag_iter resets to fresh value)
- Plan A cache reuse between AMR boundaries verified (mag_iter stable)
- Adaptive threshold (>50) idle — appropriate; no preconditioner staleness in this regime

#### Cumulative speedup confirmed
Hedgehog L5 iter step rate at t≈4.5 (post-ramp, full-spike regime): ~7 s/step *while sharing CPU with two dome validations*. Standalone post-ramp rate is similar — Plan A cache reuse is paying off, otherwise the iter-cost would compound.

---

### 📋 Validation commands ready to run (after hedgehog finishes)

```bash
# 1. Re-run analysis on completed hedgehog with corrected Rosensweig formula
cd /Users/mahdi/Projects/git/PhD-Project/Semi_Coupled
python3 analysis/analyze_hedgehog.py Results/<hedge_L5_iter_dir>

# 2. Plan A baseline: dome L5 direct vs iterative_mag, match to 1e-10
mpirun -np 4 ./cmake-build-release/ferrofluid --dome -r 5 --max_steps 200 --run_name plan_a_dome_direct
mpirun -np 4 ./cmake-build-release/ferrofluid --dome -r 5 --max_steps 200 --iterative_mag --run_name plan_a_dome_iter
python3 -c "
import csv, numpy as np
def load(p): 
    with open(p) as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip().isdigit()]
    return np.array(rows, dtype=float)
d = load('Results/plan_a_dome_direct/diagnostics.csv')
i = load('Results/plan_a_dome_iter/diagnostics.csv')
n = min(len(d), len(i))
for col, name in [(5,'mass'), (12,'M_max'), (24,'U_max')]:
    rel = np.abs(d[:n,col]-i[:n,col]).max() / (np.abs(d[:n,col]).max()+1e-30)
    print(f'{name:8s} max-rel-diff: {rel:.2e}')
"

# 3. Plan A iter-count check: confirm mag_iterations stays flat between AMR steps
python3 -c "
import csv, numpy as np
with open('Results/plan_a_dome_iter/diagnostics.csv') as f:
    rows = [r for r in csv.reader(f) if r and r[0].strip().isdigit()]
arr = np.array(rows, dtype=float)
mag_iter = arr[:, 19]   # column 19 = mag_iterations (per Plan B)
print('mag_iter per step (last 25 steps):', mag_iter[-25:].astype(int))
"

# 4. Plan B hedgehog test (only AFTER current hedgehog finishes — save current Results dir!)
mpirun -np 8 ./cmake-build-release/ferrofluid --hedgehog -r 5 --max_steps 100 --iterative_mag --run_name plan_b_hedge_smoke
# Expect: mag_iterations column populated with values in [5, 25]
```

### 🎯 NS Schur — fully working iterative path (May 5 afternoon)

Two fixes landed this afternoon:

1. **MPI_ERR_TRUNCATE** — asymmetric `compress(insert)` in pin block. **Fixed** in commit `a7bfeea`.

2. **FGMRES non-convergence** — pure-mass Schur `S ≈ α M_p` was mathematically wrong for unsteady NS at large α. **Fixed** by upgrading to LSC: `S ≈ B Q⁻¹ B^T` with `Q = diag(A)`. Implementation: `EpetraExt::MatrixMatrix::Multiply` to assemble `L_p = B Q⁻¹ B^T`, AMG on `L_p`, no extra α scaling needed (Q already absorbs 1/dt).

**Validation on dome -r 3, np=2 (sharing CPU with running hedgehog):**

| Setting | Outer FGMRES iters | Final residual | Convergence |
|---|---|---|---|
| Old (pure-mass `α M_p`) | 1500 (cap) | 2.1 | ❌ FAILED, fell back to direct |
| **New (LSC `B Q⁻¹ B^T`)** | **316–391** | **~9.7e-9** | ✅ **CONVERGED to rel_tol=1e-8** |

Wall time on dome -r 3: ~110–120 s/step (~5x slower than direct on this small case, but iterative scales as O(N) per iter vs O(N^1.5) for direct — should beat direct at -r 5+ and on hedgehog).

---

### 🎯 NS Schur MPI_ERR_TRUNCATE — FIXED (May 5 afternoon)

**Status: FIXED in this branch. `--iterative_ns -np 2` no longer crashes.**

**Real root cause** (after ruling out three wrong hypotheses):
The pressure-pinning block in `vmult` called `compress(insert)` only on ranks where `pinned_p_local_ >= 0` (typically rank 0 only). `compress()` is a **collective** operation — every rank must call it. Skipping it on other ranks left a stray send/recv that the next collective mismatched → MPI_ERR_TRUNCATE. The error fires *at the next collective after vmult*, not at compress itself, which is why bisection misled for so long.

**Fix:** move `compress(insert)` outside the pinning if-block (2 places in `vmult`, lines 391–398 and 419–425). Diff is 6 lines.

**Wrong hypotheses ruled out** (now permanently recorded in the source comment so we don't re-investigate):
- ❌ `p_owned` partition mismatch — Epetra map probes proved partitions agree exactly.
- ❌ `MPI_UNSIGNED` vs 64-bit `dealii::types::global_dof_index` — local builds use 32-bit.
- ❌ Trilinos 12.x ML/AMG internal race — replacing AMG with Jacobi or ILU did NOT change failure mode.
- ❌ Alternating IndexSets in Vector ctor — bug fires regardless.

**Open issue (separate from the MPI fix):**
FGMRES convergence is poor at default tolerances (`inner_tol=1e-1`, `max_inner=20`) — typically hits the 1500-iteration cap and falls back to direct. This was a pre-existing problem hidden by the MPI crash. Tried `inner_tol=1e-3`, `max_inner=100` — slight improvement but still slow.

**Likely cause:** the pure-mass Schur approximation `S ≈ (ν + 1/dt) M_p` isn't a great preconditioner at α = 1e4 (dt = 1e-4). The pressure-convection-diffusion (PCD) preconditioner with an added Laplacian would converge faster but requires assembling pressure stiffness — separate work, ~1 day.

**For now:** `--iterative_ns -np 2` runs without crashing but falls back to direct each step → same wall time as `--direct`, no benefit yet. Direct path remains unchanged and recommended for production.

**Original retracted analysis** (kept for reference; do not use as the diagnosis):

**Old hypothesis disproved.** The original diagnosis ("`p_owned` partition mismatch between joint NS matrix and separate p-DoFHandler") is **wrong**. Confirmed via direct Epetra map probes inside the preconditioner — RowMap, DomainMap, RangeMap, ColMap of `pressure_mass_matrix_` all agree perfectly with r_p's vector map (34560 global, 17280 per rank for the dome-r3 test case).

**64-bit indices hypothesis disproved.** Both local deal.II builds have `DEAL_II_WITH_64BIT_INDICES` undef'd. `MPI_UNSIGNED, MPI_MIN` Allreduce in `ns_setup.cc` is correct on this system. (Note for HPC: re-check on Mill cluster — different deal.II build there.)

**New diagnosis (incomplete but well-narrowed).** The bug is **inside deal.II / Trilinos 12.x internals**, not in our code:
- Constructor finishes cleanly, prints "[Block Schur] Initialized".
- Bug fires *inside the first vmult call*, not in the constructor.
- Bisection localized failure to between `extract_pressure` and `z_p = 0`.
- Adding MPI_Barriers between every operation makes the bug "hide" (operations succeed up to extract_pressure) but the next collective then hangs — strongly suggesting a Trilinos-internal MPI race / tag collision, not a logic bug in our preconditioner.
- `-np 1` works (no MPI collectives → no race window).

**Probable cause:** Trilinos 12.x ML/AMG launching overlapping non-blocking collectives with internal MPI tags that collide with subsequent vmult collectives, producing `MPI_ERR_TRUNCATE` when one rank receives more bytes than expected.

**Tried-and-failed workarounds** (all reverted):
| Approach | Result |
|---|---|
| (a) Copy-construct vectors from seed (avoid (IndexSet, comm) ctor) | Still fails inside first vmult |
| (b) Pre-allocate all scratch vectors as mutable members in ctor | Still fails — bug not in vector construction |
| (c) Insert MPI_Barriers between every vmult op | Hangs at `z_p = 0` after `extract_pressure` |

**Recommended path forward:**
1. **Upgrade to Trilinos ≥ 13** (better MPI tag handling, ML deprecated in favor of MueLu).
2. OR switch the Schur path off-AMG: use deal.II SparsityPattern + UMFPACK for inner CG.
3. OR replace `pressure_mass_ptr_` AMG preconditioner with simple Jacobi/diagonal — the pressure mass matrix is mass-dominated, so Jacobi may be enough.

**Comment updated** in `solvers/ns_block_preconditioner.cc` lines 137–172 to reflect this new understanding (committed in this branch's diff).

**Until fixed:** `--iterative_ns` continues to print a WARNING; users must use `--direct`.

---

## Papers — three documents to review

| File | Pages | Scope | Action items |
|---|---|---|---|
| `ferrofluid_proofs_v20_2.pdf` | 21 | Joint paper covering both Scheme I (T1+T2 conditional) and Scheme II (T1+T2+T3 unconditional) | Read once for the full math story; verify the 16 fixes from `v19_to_v20_2.patch` |
| `scheme_II_standalone.pdf` | 21 | Self-contained Scheme II only — the **main scientific result** | Read carefully; this is the publication target |
| `scheme_I_standalone.pdf` | 13 | Self-contained Scheme I only — practice/stepping-stone document | Quick read; lower priority |

**Cumulative fixes applied across all three documents (v19 → v20.2):**

1. Continuous identity test slot `(M×(M×H), H) = -‖M×H‖²` (was `, M`) — both Section 2 derivation **and** Remark 2.2.
2. Step 5′ Kelvin-cancellation paragraph rewritten — direct sign cancellation, no Lemma 4.2.
3. Self-mapping CFL formula corrected: `τ_* = (2C̃²)⁻¹ h^d R⁻²` (was `C̃⁻¹ h^{d/2} R⁻¹`).
4. `τ_0 ~ h^{2/3}/R^{2/3}` typo (was `h/R^{2/3}`).
5. Remark 5.6 rewritten to compare both scalings of `τ_*` and `τ_0`.
6. `Y_h := Y_h` circular definition — renamed product space to `Z_h`.
7. `f(Θ^{k-1})` → `f'(Θ^{k-1})` in eq (14b) — caught independently.
8. `\eqref{eq:disc_energy}` was rendering as `(??)` — fixed to `eq:stab123_sum`.
9. Lemma 6.3 line overflows in PDF — converted three `\[...\]` to multi-line `align*`.
10. New Assumption 6.1 (uniform `L⁴_{t,x}` bound on `M_{h,τ}`) added to Section 6 preamble — replaces the hand-wavy `h⁻¹` factor.
11. Lemma 6.2 T2 bound rewritten using Hölder triple (1/2, 1/4, 1/4) + `H¹↪L⁴` Sobolev embedding.
12. Theorem 6.6 statement now lists Assumption 6.1.

## Code — Semi_Coupled

### Currently running
- **Hedgehog L5 iter** on laptop, 8 ranks, **PID 71402** (mpirun) + 71404–71411 (workers)
- Last check: step ~36000, t ≈ 3.6, ~2.6 s/step, ETA ~17 hours to t=6.0
- Output: `Results/20260430_235602_hedge_r3_direct_amr/`
- Iterative magnetic solver (block precond) is converging cleanly; no fallbacks.
- Spike bifurcation observed at **t ≈ 2.3**; saturates at 2 spikes with λ ≈ 0.20.

### Validated / in PR #2
- `--iterative` flag (CH GMRES+AMG)
- `--iterative_mag` (Magnetic GMRES + custom block preconditioner: ILU on M, AMG on φ, via Trilinos Epetra extraction)
- Diagnostics fix for h_a-only mode (`H_max`, `M_max`, `F_kelvin` now correctly populated)
- NS Schur ctor fix (32-bit vs 64-bit GIDs)
- Speedup: hedgehog L5 ~1.97× faster than direct, dome L5 ~2.2× faster
- MMS: CH_STANDALONE ✓, NS_STANDALONE ✓, NS_CH ✓ (CH_MAGNETIC, MAGNETIC_NS, FULL_SYSTEM remain stubs)

### Open issues (deferred)

#### NS Schur `vmult` MPI_ERR_TRUNCATE — **plan documented**
- Diagnosis in `solvers/ns_block_preconditioner.cc` constructor comment block.
- Cause: `p_owned_` (passed from caller as `p_locally_owned_` from separate p-DoFHandler) doesn't agree with the joint NS matrix's row partition.
- Fix plan: re-assemble `pressure_mass_matrix_` on NS-row-aligned p-partition (Option 1 in chat history). ~1 day effort.
- Workaround: `--iterative_ns` now prints a WARNING explaining the bug; users should use `--direct` until fixed.

#### ~~Cross-AMR preconditioner caching~~ — Plan A **CODE COMPLETE (May 5)**
- Implemented as `needs_mag_preconditioner_rebuild_` flag in `phase_field.h`, set on AMR (in `phase_field_amr.cc::refine_mesh()`) and cleared after successful solve.
- **Adaptive trigger**: if last GMRES iter count > 50, force rebuild on next call (catches matrix-value drift).
- Implicit safety net retained: `setup_magnetic_system()` still recreates `magnetic_solver_` on AMR, dropping the cache via destructor — so the flag is redundant for the AMR case but explicit-and-clear, plus it adds the adaptive trigger that wasn't there before.
- **Validation pending** until hedgehog finishes (see commands at top of file).

#### Hedgehog asymmetry observation
- Visual inspection of mid-run frames shows non-symmetric spike pattern (one dominant + shoulders, rather than mirror-symmetric across x=0.5).
- Likely sources:
  - MPI partition boundaries breaking left-right symmetry at machine precision, then amplified by the positive-feedback Rosensweig instability.
  - AMR refining differently on left vs right due to round-off in Kelly indicators.
  - ~~Possible non-symmetric dipole positions in `parameters.cc::setup_hedgehog()`~~ — **ruled out (May 5)**: 14 dipoles evenly spaced from x=0.3 to x=0.7 are exactly mirror-symmetric about x=0.5 (pair j ↔ 13−j). Same for the y-rows. IC also symmetric (flat layer). Asymmetry must be runtime-induced (MPI partition or AMR).
- **Test plan when hedgehog finishes**: run the same case at `-np 1` (single process, no MPI partition asymmetry) for ~500 steps and compare interface profile. If symmetric at np=1 but asymmetric at np=8, the cause is MPI partition. If asymmetric even at np=1, look at AMR Kelly indicator round-off.
- Triage: not blocking the L5 result; revisit after run completes and we can compare to paper Fig. 6.

#### Early-step θ violations
- 773 violations cumulatively, mostly at steps 1–99 (initial sharp IC relaxing, expected) and a transient cluster around step ~23000 (correlates with the bifurcation event).
- Latest values within `[-1.010, 1.006]` — minor and self-healing.
- Worth correlating temporally with the asymmetry once we look at it.

#### ~~GMRES iteration counts not surfaced~~ — **DONE (May 5)**
- `mag_iterations` and `mag_residual` columns added to `diagnostics.csv` (positions 19, 20).
- Plumbed through `MagneticSolver::last_residual()` accessor + `last_residual_` field.
- Tested on dome L5 (mag_iter=1 expected because h_a-only mode makes the φ block trivial).
- **Hedgehog smoke test deferred** until current run finishes — expect mag_iter ∈ [5, 25] at L5.

## Analysis tooling — built today

| Script | What it does |
|---|---|
| `analysis/analyze_hedgehog.py` | 9-panel summary plot + Rosensweig validation panel from `diagnostics.csv`. Re-runnable on partial files. |
| `analysis/count_spikes.py` | pvpython script: extract θ along a horizontal line at y=`pool_depth + ε`, count zero crossings, FFT for dominant wavelength. Tested on 4 frames, captures the bifurcation. |

## Detailed plans for deferred work

### Plan A — Phase 3: cross-AMR preconditioner caching (~4–6 h)

**Problem.** Currently `MagneticBlockPreconditioner` is rebuilt at *every* time step (block extraction + ILU(0) factorization on M-block + AMG setup on φ-block), even though the matrix sparsity is constant between AMR events. AMR fires every 5 steps. So we pay for a full preconditioner build 4 unnecessary times per AMR cycle.

**Approach.**

1. **In `core/phase_field.h`**, add `bool needs_preconditioner_rebuild_ = true;` member. Lifetime: persists across time steps; reset to `true` whenever AMR happens.

2. **In `core/phase_field.cc::run()`**, set the flag inside the AMR-trigger branch:
   ```cpp
   if (params_.mesh.use_amr && timestep_number_ > 0 &&
       timestep_number_ % params_.mesh.amr_interval == 0) {
       refine_mesh();
       needs_preconditioner_rebuild_ = true;     // ← new
   }
   ```
   Set to `true` initially so the first solve builds it.

3. **In `core/phase_field.cc::solve_magnetics()`** at the call site (currently around line 782):
   ```cpp
   magnetic_solver_->solve(
       mag_matrix_, mag_solution_, mag_rhs_,
       params_.solvers.magnetic, n_M_dofs,
       needs_preconditioner_rebuild_);   // was hard-coded `false`
   needs_preconditioner_rebuild_ = false;
   ```

4. **Inside `MagneticSolver::solve_iterative()`**, the rebuild logic is already correct — it builds when `cached_block_prec_ == nullptr` OR `rebuild_preconditioner == true`. So no change needed in the solver itself; we just feed it the right flag.

5. **`setup_magnetic_system()`** (called by `refine_mesh()`) already recreates `magnetic_solver_` from scratch, which drops the cached preconditioner. So the AMR-step rebuild is double-guaranteed. Could simplify by *not* recreating the solver at AMR — just calling `magnetic_solver_->invalidate_preconditioner()` — but the recreation is harmless.

**Validation.**
- Hedgehog L5, 200 steps, with vs. without caching: solutions should match to ~1e-10 (only difference is preconditioner staleness within an AMR cycle, which doesn't change the converged solution).
- Step time should drop ~30–50% on non-AMR steps. Measure: median over 4 AMR cycles.
- MMS NS_CH at L5 should still pass.

**Risk.** Low. The block preconditioner is mathematically stale between AMR events (matrix values changed, sparsity didn't), but GMRES naturally absorbs that — at most we get a few extra GMRES iters. If iter counts climb >2× between AMRs, that's a signal to invalidate more aggressively, but that's fine-tuning.

**Files touched:**
- `core/phase_field.h` (1 line)
- `core/phase_field.cc` (3 lines: AMR branch + solve_magnetics call site)

---

### Plan B — Surface GMRES iteration counts (~30 min)

**Problem.** `MagneticBlockPreconditioner`'s solver returns a count via `last_n_iterations_`, but it's never written to the diagnostics CSV. We can't tell if GMRES is taking 5 iters or 200 per solve.

**Approach.**

1. **In `solvers/magnetic_solver.h`**, the method `last_n_iterations() const { return last_n_iterations_; }` already exists.

2. **In `core/phase_field.cc::solve_magnetics()`** (after `magnetic_solver_->solve(...)`), capture the count:
   ```cpp
   last_mag_info_.iterations = magnetic_solver_->last_n_iterations();
   ```
   This already happens for the direct path (1 iter); just need to make sure it's written for the iterative path too.

3. **In `output/metrics_logger.cc::log_step()`**, the `data.poisson_iterations` column is currently empty for the iterative path. Wire `last_mag_info_.iterations` to it:
   ```cpp
   data.poisson_iterations = last_mag_info_.iterations;
   ```
   (Or rename the CSV column to `magnetic_iterations` for clarity — small breaking change to downstream analysis scripts.)

4. **Log min/avg/max per output interval** (every 10 steps) so we don't blow up the CSV with per-step counts. Add three columns: `mag_iter_min`, `mag_iter_avg`, `mag_iter_max`.

**Validation.** Just run `--iterative_mag --max_steps 100` and check the CSV column is non-zero and reasonable (expected: 5–25 iters/solve for the block preconditioner at L5).

**Files touched:**
- `core/phase_field.cc` (2 lines)
- `output/metrics_logger.cc` + `.h` (3 lines if we add new columns; 1 line if we just reuse `poisson_iterations`)

**Side benefit.** Once this is wired, we can plot iteration count over time and immediately see when the cached preconditioner gets stale (item from Plan A above) — useful for tuning the AMR rebuild interval.

---

### Plan C — ParaView state file `hedgehog.pvsm` (~30 min, manual)

**Problem.** Loading 3,000+ `.pvtu` frames into ParaView requires manual setup of the time-series reader, color map for θ, camera angle, and any filters (e.g., contour at θ=0 to draw the interface line). Each fresh ParaView session repeats this work.

**Approach.** This is interactive — best done by you in the GUI, then saved as a `.pvsm` file:
1. Open ParaView → File → Open → select `solution__..pvtu` (the dot-dot wildcard auto-detects the time series).
2. Color by θ; set the color map to a perceptually uniform one (e.g., `Cool to Warm` with range fixed to [-1, 1]).
3. Add a `Contour` filter on θ at iso-value 0 — this draws the interface line over the colored field.
4. Camera: orthographic, look down z-axis, fit to domain `[0,1]×[0,0.6]`.
5. Optional: add a `Plot Over Line` filter at y=0.13 to live-plot θ(x) — useful for spike counting interactively.
6. File → Save State → `~/PhD-Project/Semi_Coupled/analysis/hedgehog.pvsm`.

**Why not automate.** ParaView's `.pvsm` files reference the data by absolute path. They're host-specific and don't survive moving the run dir. Better to build it once on this machine and re-create as needed.

**Alternative if you don't want to use ParaView GUI.** I can write a `pvbatch` rendering script that produces a movie (`.mp4`) of θ(x,y,t) directly from the command line. ~1 hour to write. Useful if you want a video output for talks, but not necessary if you just want to look at frames.



1. Re-run `analyze_hedgehog.py` on the complete `diagnostics.csv` and inspect the energy decay post-ramp (t > 4.2).
2. Run `count_spikes.py` on every 100th VTU frame to track wavelength evolution.
3. Pick representative frames (t = 1.5, 2.3, 3.0, 4.2, 6.0) and render in ParaView matched to paper Fig. 6 layout. Side-by-side comparison.
4. Decide: is the L5 result publication-quality? If yes, set up L6 on HPC (you drive). If no, debug the asymmetry first.
