# Semi_Coupled HPC Experiment Matrix (v3)

**Goal:** Identify why Rosensweig instability (spikes) does not form.
**Paper:** Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
**Binary:** `ferrofluid` (build with `make -j` in `cmake-build-release/`)
**Date:** 2026-03-18

## Code Changes (vs previous flat-interface runs)

### Audit fixes (baked into current binary)

1. **CH convection: implicit θ^k → explicit θ^{k-1}** (paper Eq 65a). Implicit was over-stabilizing the interface. Now default in `setup_rosensweig()` and `setup_hedgehog()`.
2. **BGS mag_old overwrite fixed:** `mag_old_solution_` saved once before BGS loop (was overwritten each BGS iteration).
3. **Heaviside cutoff consistency:** `susceptibility()` now calls `heaviside()` (cutoff 30, was reimplemented with cutoff 20).
4. **MPI type portability:** `MPI_UNSIGNED` → sizeof-based type selection in `ns_setup.cc`.
5. **`use_ilu` dead code fixed:** Block PC sub-block ILU selection now uses `params.use_ilu` (was checking `preconditioner == ILU` inside `BlockSchur` path → always false).
6. **NS residual reporting:** `info.residual` now always computed (was gated behind `if (verbose)` → returned 0.0).

### Iterative solver status

- **CH:** Block Schur PC works (~273 iters). Enable with `--ch-iterative`.
- **Mag:** Block Schur PC works (~9-11 iters, ~8x faster than direct). Enable with `--mag-iterative`.
- **NS:** BFBt Schur stagnates for DG Q1 pressure. **Keep direct (MUMPS).**
- **HPC shortcut:** `--ilu --ns-direct` enables CH+Mag iterative with ILU inner solves (no AMG/ML needed) while keeping NS direct.

### Preset defaults (no flags needed)

- `amr_interval = 5` (paper p.520)
- `use_explicit_ch_convection = true` (paper Eq 65a)
- Gravity auto-computed: g = 31,583 (Eq 103)

## Hypotheses Being Tested

| # | Hypothesis | Tests |
|---|-----------|-------|
| H1 | Explicit CH convection fix alone produces spikes | T1 (post-audit baseline) |
| H2 | Paper gravity (30k vs auto 31.6k) matters | T2 vs T1 |
| H3 | BGS coupling too weak (1 pass vs iterated) | T3, T4 vs T1 |
| H4 | Iterative solver noise helps seed instability | T5 vs T1 |
| H5 | Mesh resolution insufficient | T6, T7 vs T1 |
| H6 | Time step too coarse | T8 vs T1 |
| H7 | AMR interval matters (5 vs 50) | T9 vs T1 |
| H8 | All favorable parameters combined | T10 |

## Test Matrix

Submit with SLURM array jobs:
```bash
cd ~/Semi_Coupled-HPC/Semi_Coupled
mkdir -p logs Results
sbatch --array=1-10 hpcc_run/submit.sub hpcc_run/rosensweig_v3.dat
```

| Test | Description | Key Flags |
|------|-------------|-----------|
| T1 | **Baseline** — post-audit binary, direct solver, paper defaults | `--rosensweig` |
| T2 | Paper gravity (g=30000 vs auto 31583) | `--rosensweig --gravity 30000` |
| T3 | BGS=2 (tighter coupling) | `--rosensweig --bgs_iters 2` |
| T4 | BGS=3 (even tighter) | `--rosensweig --bgs_iters 3` |
| T5 | CH+Mag iterative, NS direct (solver noise) | `--rosensweig --ch-iterative --mag-iterative` |
| T6 | Coarser mesh L5 (quick diagnostic) | `--rosensweig -r 5` |
| T7 | Finer mesh L7 | `--rosensweig -r 7` |
| T8 | Finer time step dt=2.5e-4 | `--rosensweig --dt 2.5e-4 --max_steps 8000` |
| T9 | Old AMR interval=50 (control) | `--rosensweig --amr_interval 50` |
| T10 | **Kitchen sink** — paper gravity + iterative + BGS=2 | `--rosensweig --ch-iterative --mag-iterative --gravity 30000 --bgs_iters 2` |

## Expected Runtime (8 ranks on The Mill)

| Test | Steps | Est. Time |
|------|-------|-----------|
| T1, T2, T9 | 4000 | 1-2h |
| T3, T4, T10 | 4000 | 2-4h (BGS overhead) |
| T5 | 4000 | 1-2h (iterative CH+Mag faster) |
| T6 | 4000 | ~30min (coarser mesh) |
| T7 | 4000 | 4-6h (finer mesh) |
| T8 | 8000 | 3-5h |

## How to Interpret Results

Check `Results/<run_name>/energy.csv` — look for **E_kin growing** (fluid motion).
Check `Results/<run_name>/rosensweig_validation.csv` — look for **n_spikes > 0**.

**Key comparisons:**
- **T1 has spikes** → explicit CH convection fix was the key (most likely outcome)
- **T1 flat, T5 has spikes** → iterative solver noise needed to seed instability
- **T1 flat, T3/T4 has spikes** → BGS coupling strength matters
- **T1 flat, T2 has spikes** → gravity 5% too high was suppressing instability
- **T1 flat, T7 has spikes** → needed finer mesh to resolve instability wavelength
- **T10 has spikes, T1 flat** → needed combination of fixes
- **All flat** → deeper issue (model limitation, missing physics, or boundary conditions)

## Preset Parameters (for reference)

```
Rosensweig: eps=0.01, lambda=0.05, chi=0.5, dt=5e-4, t_final=2.0, pool=0.2
Hedgehog:   eps=0.005, lambda=0.025, chi=0.9, dt=1e-4, t_final=6.0, pool=0.11
Gravity (auto): g = 4*pi^2*lambda / (ell_c^2 * r * eps) = 31,583
Paper gravity:  g = 30,000 (hardcoded)
```

## HPC Notes (The Mill — Missouri S&T)

```bash
# SSH
ssh mg6f4@mill-login-p2.itrss.mst.edu

# Module
module load dealii/9.7.1_gcc_12.3.0_mvapich

# Build (on compute node, NOT login node)
salloc --partition=hpcc --ntasks=8 --time=01:00:00
cd ~/Semi_Coupled-HPC/Semi_Coupled
mkdir -p cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ..

# Submit
mkdir -p logs Results
sbatch --array=1-10 hpcc_run/submit.sub hpcc_run/rosensweig_v3.dat
```
