# Semi_Coupled: HPC Performance Improvements

## Status: Identified — Not Yet Implemented

These optimizations were identified during HPC scaling tests on The Mill (March 2026).
Current scaling shows 8 ranks optimal for L5-L6, with diminishing returns beyond that.
Root cause: communication overhead dominates computation at higher rank counts.

---

## P1: Cache Block Schur Preconditioner Across Timesteps
**File:** `solvers/ns_solver.cc`, `solvers/ns_block_preconditioner.cc`
**Impact:** ~10-15% per step

Currently, `BlockSchurPreconditionerParallel` is rebuilt from scratch every NS solve:
velocity block extraction (2 full Epetra row scans), sparsity construction, AMG/ILU setup.
On non-AMR steps (90% of steps), the sparsity is unchanged — only matrix values change.

**Fix:** Add a `dirty` flag set by AMR. On clean steps, reuse the existing preconditioner
and only update the matrix values + reinitialize the smoother.

## P2: Cache CH and Magnetic Preconditioners
**File:** `solvers/ch_solver.cc`, `solvers/magnetic_solver.cc`
**Impact:** ~3-5% per step

Same pattern — AMG and direct solver factorizations rebuilt every call.
Symbolic factorization (MUMPS) and AMG hierarchy can be reused when sparsity is unchanged.

## P3: Batch MPI Reductions in Diagnostics
**File:** `diagnostics/*.h`
**Impact:** ~1-3% per step

Currently ~15 individual `MPI_Allreduce` calls per timestep across all diagnostic routines.
Each call has latency overhead (~0.1-1ms). At 32+ ranks, latency accumulates.

**Fix:** Pack SUM values into one array, MIN into another, MAX into another.
3 calls instead of 15.

## P4: Eliminate O(N_dofs) Broadcasts in NS Setup
**File:** `setup/ns_setup.cc`
**Impact:** ~2-5% per step (grows with problem size)

NS setup uses `MPI_Allreduce` with arrays of size `n_ux + n_uy + n_p` (full DOF count)
to exchange DOF index maps. This is O(N_dofs) communication per rank.
Also uses `MPI_UNSIGNED` (32-bit) which will silently truncate on large problems.

**Fix:** Replace with local-only index map construction using the Epetra matrix's
row/column maps. Each rank only needs its own locally owned/relevant mappings.

---

## Scaling Data (March 2026)

### L5, dt=1e-3, iterative solver (step 400)
| Ranks | s/step |
|-------|--------|
| 6     | 232    |
| 8     | 229    |
| 10    | 213    |
| 12    | 226    |

### L6, dt=5e-4, iterative solver (step 100)
| Ranks | s/step |
|-------|--------|
| 8     | 191    |
| 16    | 395    |
| 24    | 387    |
| 28    | 404    |

### L5, dt=1e-3, direct solver (step 400)
| Ranks | s/step |
|-------|--------|
| 8     | 274    |
| 16    | 257    |
| 24    | 240    |
| 32    | 221    |

**Key finding:** Direct solver scales modestly with ranks (Amdahl's law limits).
Iterative solver shows *negative* scaling beyond 8 ranks due to communication overhead.
Fixing P1-P4 should improve iterative scaling at higher rank counts.
