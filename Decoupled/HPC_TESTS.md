# Hedgehog HPC Test Matrix

## Binaries

| Binary | Description | Kelvin Force |
|--------|-------------|-------------|
| `bin0_hedgehog` | Current code + CLI overrides | H = ∇φ only (original) |
| `bin1_hedgehog` | Kelvin fix + CLI overrides | H = h_a + ∇φ (fixed) |

### Building

```bash
# Build BIN-0 (no Kelvin fix)
git checkout -- navier_stokes/navier_stokes_assemble.cc
cd build && cmake .. && make -j32
cp drivers/ferrofluid_decoupled bin0_hedgehog

# Apply Kelvin fix: in navier_stokes/navier_stokes_assemble.cc
#   add: #include "physics/applied_field.h"
#   change H_vec and grad_H to include h_a (see HEDGEHOG_TEST_CHANGES.md)
make -j32
cp drivers/ferrofluid_decoupled bin1_hedgehog
```

---

## Test Matrix (12 tests)

### BIN-0 — Current code, no Kelvin fix

| Test | Command | What it tests |
|------|---------|---------------|
| T1 | `mpirun -np 4 ./bin0_hedgehog --hedgehog` | Baseline (expect crash ~t=2.2) |
| T2 | `mpirun -np 4 ./bin0_hedgehog --hedgehog --dt 1e-4` | CFL / timestep stability |
| T3 | `mpirun -np 4 ./bin0_hedgehog --hedgehog --dt 5e-5` | Aggressive CFL test |
| T4 | `mpirun -np 4 ./bin0_hedgehog --hedgehog --chi0 0.5` | Lower susceptibility |
| T5 | `mpirun -np 4 ./bin0_hedgehog --hedgehog --mesh 150x90` | Finer spatial resolution |
| T6 | `mpirun -np 4 ./bin0_hedgehog --hedgehog --ramp-slope 0.6` | Slower magnetic ramp |

### BIN-1 — With Kelvin fix (H = h_a + ∇φ)

| Test | Command | What it tests |
|------|---------|---------------|
| T7 | `mpirun -np 4 ./bin1_hedgehog --hedgehog` | Kelvin fix alone |
| T8 | `mpirun -np 4 ./bin1_hedgehog --hedgehog --dt 1e-4` | Kelvin fix + finer dt |
| T9 | `mpirun -np 4 ./bin1_hedgehog --hedgehog --chi0 0.5` | Kelvin fix + easy chi |
| T10 | `mpirun -np 4 ./bin1_hedgehog --hedgehog --mesh 150x90` | Kelvin fix + finer mesh |
| T11 | `mpirun -np 4 ./bin1_hedgehog --hedgehog --dt 1e-4 --mesh 150x90` | Kelvin fix + dt + mesh |
| T12 | `mpirun -np 4 ./bin1_hedgehog --hedgehog --ramp-slope 0.6` | Kelvin fix + slower ramp |

---

## CLI Flags (added temporarily)

| Flag | What it overrides | Example |
|------|-------------------|---------|
| `--dt VALUE` | Time step (auto-adjusts max_steps) | `--dt 1e-4` |
| `--chi0 VALUE` | Magnetic susceptibility χ₀ | `--chi0 0.5` |
| `--mesh NxM` | Mesh cells (width x height) | `--mesh 150x90` |
| `--ramp-slope VALUE` | Dipole ramp slope | `--ramp-slope 0.6` |

---

## Default Hedgehog Parameters (for reference)

| Parameter | Value | Source |
|-----------|-------|--------|
| χ₀ | 0.9 | Zhang Section 4.4 |
| dt | 2e-4 | Zhang Section 4.4 |
| mesh | 120x72 (h=1/120) | Zhang Section 4.4 |
| ramp_slope | 1.2 | Zhang Section 4.4 |
| intensity_max | 0 (no cap) | Zhang Section 4.4 |
| t_final | 4.0 | Zhang Fig 4.11 shows up to t=3.5 |
| domain | [0,1] x [0,0.6] | Zhang Section 4.3 |
| interface | y = 0.1 | Zhang Eq 4.5 |

---

## How to Interpret Results

- **T1 crashes, T7 survives** → Kelvin fix (missing h_a) was the issue
- **T1 crashes, T2 survives** → timestep too large (CFL)
- **T1 crashes, T4 survives** → χ₀=0.9 too aggressive for current scheme
- **T1 crashes, T5 survives** → mesh too coarse
- **T1 crashes, T6 survives** → ramp too fast
- **All BIN-0 crash, all BIN-1 survive** → definitely the Kelvin force
- **Everything crashes** → multiple issues, check T11 (all fixes combined)

---

## SLURM Submission

```bash
sbatch hpc/hedgehog_array.sub
```

Or individual tests:
```bash
sbatch --partition=hpcc --ntasks=4 --time=28-00:00:00 --wrap="mpirun -np 4 ./bin0_hedgehog --hedgehog"
```
