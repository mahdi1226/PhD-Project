# Semi_Coupled HPC Experiment Matrix

**Goal:** Identify why Rosensweig instability (spikes) does not form.
**Paper:** Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
**Binary:** `ferrofluid` (build with `make -j` in `cmake-build-release/`)

## Code Changes (vs previous flat-interface runs)

1. Rosensweig `amr_interval` fixed: 50 → 5 (paper p.520: "refined-coarsened once every 5 time steps")
2. Added `--iterative` CLI flag (GMRES+AMG instead of MUMPS)
3. Auto-computed gravity via Eq.103: g = 31,583 (paper used hardcoded 30,000)

## Hypotheses Being Tested

| Hypothesis | Tests |
|-----------|-------|
| AMR too infrequent (50 vs paper's 5) → no numerical noise to seed instability | T1 vs T8 |
| Direct solver too clean → no perturbation source | T2, T3 vs T1 |
| Gravity 5% too high | T4, T5 vs T1 |
| BGS coupling too weak (1 pass vs iterated) | T6, T7 |
| Mesh resolution | T9, T10 |
| Time step sensitivity | T11 |
| Hedgehog preset | T12 |

## Launch Commands

All tests use 4 MPI ranks. Adjust `-np` for your allocation.
Launch from the project root (so `./Results/` is written correctly).

```bash
# ============================================================
# CRITICAL — Run these first
# ============================================================

# T1: Baseline with AMR fix (amr_interval=5 is now default in preset)
mpirun -np 4 ./ferrofluid --rosensweig --run_name T1-rosen-amr5-direct

# T2: Iterative solver (adds numerical noise)
mpirun -np 4 ./ferrofluid --rosensweig --iterative --run_name T2-rosen-amr5-iterative

# T3: Both iterative + paper gravity
mpirun -np 4 ./ferrofluid --rosensweig --iterative --gravity 30000 --run_name T3-rosen-amr5-iter-g30k

# T4: AMR fix + paper's exact gravity (direct solver)
mpirun -np 4 ./ferrofluid --rosensweig --gravity 30000 --run_name T4-rosen-amr5-g30k

# T5: AMR fix + iterative + gravity + 2 BGS (all fixes combined)
mpirun -np 4 ./ferrofluid --rosensweig --iterative --gravity 30000 --bgs_iters 2 --run_name T5-rosen-all-fixes

# ============================================================
# SERIOUS — Run if T1-T5 all stay flat
# ============================================================

# T6: 2 BGS iterations (stronger coupling per step)
mpirun -np 4 ./ferrofluid --rosensweig --bgs_iters 2 --run_name T6-rosen-amr5-bgs2

# T7: 3 BGS iterations
mpirun -np 4 ./ferrofluid --rosensweig --bgs_iters 3 --run_name T7-rosen-amr5-bgs3

# ============================================================
# MODERATE — Resolution and time step study
# ============================================================

# T8: Control — old AMR interval (should stay flat, confirms AMR is the variable)
mpirun -np 4 ./ferrofluid --rosensweig --amr_interval 50 --run_name T8-rosen-amr50-control

# T9: Coarser mesh (quick ~1h diagnostic)
mpirun -np 4 ./ferrofluid --rosensweig -r 5 --run_name T9-rosen-amr5-r5

# T10: Finer mesh
mpirun -np 4 ./ferrofluid --rosensweig -r 7 --run_name T10-rosen-amr5-r7

# T11: Finer time step
mpirun -np 4 ./ferrofluid --rosensweig --dt 2e-4 --max_steps 10000 --run_name T11-rosen-amr5-dt2e4

# T12: Hedgehog preset (different physics: chi=0.9, eps=0.005, 42 dipoles)
mpirun -np 4 ./ferrofluid --hedgehog -r 6 --run_name T12-hedge-amr5-r6
```

## Expected Runtime (4 ranks)

| Test | Steps | Est. Time |
|------|-------|-----------|
| T1-T4, T8 | 4000 | 2-4h |
| T5-T7 | 4000 | 3-6h (BGS adds overhead) |
| T9 | 4000 | ~1h (coarser mesh) |
| T10 | 4000 | 8-10h (finer mesh) |
| T11 | 10000 | 6-8h |
| T12 | 60000 | 20-30h (smaller dt, finer mesh) |

## How to Interpret Results

Check `Results/<run_name>/energy.csv` — look for **E_kin growing** (fluid motion).
Check `Results/<run_name>/rosensweig_validation.csv` — look for **n_spikes > 0**.

**Key comparisons:**
- **T1 has spikes, T8 flat** → AMR frequency was the fix
- **T1 flat, T2 has spikes** → iterative solver noise needed
- **T1+T2 flat, T5 has spikes** → needed all fixes combined
- **All flat** → deeper code bug (assembly, coupling, or sign error)

## Preset Parameters (for reference)

Rosensweig: eps=0.01, lambda=0.05, chi=0.5, dt=5e-4, t_final=2.0, pool=0.2
Hedgehog: eps=0.005, lambda=0.025, chi=0.9, dt=1e-4, t_final=6.0, pool=0.11
Gravity (auto): g = 4*pi^2*lambda / (ell_c^2 * r * eps) = 31,583 (both presets)
Paper gravity: g = 30,000 (hardcoded)
