# Sparsity Analysis & Cuthill-McKee Renumbering Report

## Semi_Coupled Solver — Methodology & Results

**Date:** 2026-03-05
**Solver:** Semi_Coupled (parallel, p4est + Trilinos, separate DoFHandlers per field)
**Mesh:** Rosensweig preset, initial refinement 4 (15,360 cells, ~62K Q2 DoFs per scalar)

---

## 1. What Was Done

Three capabilities were added to the solver:

1. **`--renumber-dofs`** — Cuthill-McKee (CM) DoF renumbering to reduce matrix bandwidth
2. **`--dump-sparsity`** — Export sparsity patterns (SVG, gnuplot, bandwidth CSV) at startup
3. **Bandwidth tracking** in the parallel diagnostics CSV (per-step, per-matrix)

Additionally, **node-wise velocity interleaving** was implemented as an alternative to block ordering in the monolithic NS system.

---

## 2. How Cuthill-McKee Was Implemented

### 2.1 Location

In the `setup_dof_system()` method (file: `core/phase_field_setup.cc`), immediately **after** `distribute_dofs()` and **before** extracting `locally_owned_dofs()` / `locally_relevant_dofs()`.

### 2.2 Code Pattern

```cpp
#include <deal.II/dofs/dof_renumbering.h>

// In setup_dof_system(), after distribute_dofs():
theta_dof_handler_.distribute_dofs(fe_Q2_);
psi_dof_handler_.distribute_dofs(fe_Q2_);

// Apply CM BEFORE extracting index sets
if (params_.renumber_dofs)
{
    dealii::DoFRenumbering::Cuthill_McKee(theta_dof_handler_);
    dealii::DoFRenumbering::Cuthill_McKee(psi_dof_handler_);
}

// NOW extract index sets (with renumbered DoFs)
theta_locally_owned_ = theta_dof_handler_.locally_owned_dofs();
theta_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler_);
```

### 2.3 Which Handlers Get CM

| Handler | FE Space | CM Applied? | Reason |
|---------|----------|-------------|--------|
| θ (phase field) | CG Q2 | YES | Inter-element coupling, CM reduces bandwidth |
| ψ (chemical potential) | CG Q2 | YES | Same mesh, same benefit |
| φ (Poisson/potential) | CG Q2 | YES | Dramatic bandwidth reduction (88%) |
| M (magnetization) | DG Q2 | NO | Cell-local basis, no inter-element coupling matrix entries |
| ux, uy (velocity) | CG Q2 | NO | See Section 4 — CM on individual components hurts the monolithic NS system |
| p (pressure) | DG P1 | NO | Cell-local basis (DG) |

### 2.4 Critical Ordering Rule

```
distribute_dofs()  →  Cuthill_McKee()  →  locally_owned_dofs()
```

CM must be called **after** `distribute_dofs()` (which creates the initial numbering) and **before** any `locally_owned_dofs()` / `extract_locally_relevant_dofs()` calls (which cache the index sets used everywhere else).

### 2.5 Parameters

In `utilities/parameters.h`:
```cpp
bool renumber_dofs = false;    // --renumber-dofs / --no-renumber-dofs
bool dump_sparsity = false;    // --dump-sparsity
```

In `utilities/parameters.cc`, CLI parsing:
```cpp
else if (std::strcmp(argv[i], "--renumber-dofs") == 0)
    params.renumber_dofs = true;
else if (std::strcmp(argv[i], "--no-renumber-dofs") == 0)
    params.renumber_dofs = false;
else if (std::strcmp(argv[i], "--dump-sparsity") == 0)
    params.dump_sparsity = true;
```

---

## 3. How Sparsity Was Measured

### 3.1 Bandwidth Computation

Bandwidth = max |i - j| over all nonzero entries a_{ij}.

Computed by iterating over locally owned rows of the Trilinos matrix:

```cpp
auto compute_bandwidth = [](const dealii::TrilinosWrappers::SparseMatrix& mat)
    -> unsigned int
{
    unsigned int max_bw = 0;
    const auto range = mat.local_range();
    for (unsigned int i = range.first; i < range.second; ++i)
        for (auto entry = mat.begin(i); entry != mat.end(i); ++entry)
        {
            const unsigned int j = entry->column();
            const unsigned int dist = (i > j) ? (i - j) : (j - i);
            if (dist > max_bw) max_bw = dist;
        }
    return max_bw;
};
```

For parallel runs, global bandwidth = MPI_Reduce(local_bandwidth, MPI_MAX).

### 3.2 O(n × b²) Cost Metric

For banded direct solvers (LU factorization), the dominant cost is:

```
Cost ≈ n × b²
```

where:
- `n` = number of rows (matrix size)
- `b` = bandwidth (max |i-j| for nonzero entries)

This quantifies fill-in during factorization. Reducing `b` by factor `k` reduces cost by `k²`.

### 3.3 Sparsity Export Utility

File: `utilities/sparsity_export.h` (header-only, self-contained)

Three functions:
1. **`analyze_sparsity(matrix, name)`** — Returns `SparsityAnalysis` struct with:
   - n_rows, n_cols, total_nnz
   - bandwidth (local), avg_bandwidth
   - min/max/avg/std nnz_per_row
   - density = nnz / (n_rows × n_cols)
   - Per-row nnz vector (for histograms)

2. **`export_sparsity_pattern(matrix, name, output_dir, comm, pcout)`** — Writes:
   - `{name}_bandwidth.csv` — per-row nnz distribution + summary header
   - `{name}_sparsity.svg` — visual sparsity pattern (matrices < 5000 rows only)
   - `{name}_sparsity.gnuplot` — gnuplot format (matrices < 5000 rows only)

3. **`write_sparsity_summary(analyses, output_dir, renumbered, pcout)`** — Combined CSV:
   ```
   matrix,rows,cols,nnz,bandwidth,avg_bandwidth,min_nnz_row,max_nnz_row,avg_nnz_row,std_nnz_row,density
   ```

### 3.4 Integration Points

**At startup** (after CH/Poisson/Mag matrices are assembled, before time loop):
```cpp
if (params_.dump_sparsity)
{
    std::vector<SparsityAnalysis> all_analyses;
    // Analyze and export CH, Poisson, Magnetization
    auto a = analyze_sparsity(ch_matrix_, "CH");
    unsigned int global_bw = 0;
    MPI_Reduce(&a.bandwidth, &global_bw, 1, MPI_UNSIGNED, MPI_MAX, 0, mpi_communicator_);
    if (is_root) a.bandwidth = global_bw;
    all_analyses.push_back(a);
    export_sparsity_pattern(ch_matrix_, "ch", output_dir, mpi_communicator_, pcout_);
    // ... same for Poisson, Magnetization ...
    write_sparsity_summary(all_analyses, output_dir, params_.renumber_dofs, pcout_);
}
```

**After step 1** (NS matrix is only assembled during `solve_ns()`, not at setup):
```cpp
if (params_.dump_sparsity && timestep_number_ == 1 && params_.enable_ns)
{
    export_sparsity_pattern(ns_matrix_, "ns", output_dir, mpi_communicator_, pcout_);
    // Append NS row to sparsity_summary.csv
}
```

### 3.5 Bandwidth in Parallel Diagnostics

Per-step bandwidth is computed once (cached with `static bool bandwidth_computed`) and written to `parallel_diagnostics.csv` alongside timing and nnz data. Uses `MPI_Allreduce(MPI_MAX)` for global values.

---

## 4. NS Velocity Ordering — Block vs Interleaved

### 4.1 The Problem

The monolithic NS system stacks three separate DoFHandlers (ux, uy, p) into one system:

```
Default block ordering:  [all_ux_DoFs | all_uy_DoFs | all_p_DoFs]
```

With ~62K DoFs per component, the velocity-pressure offset is ~124K, which dominates the bandwidth regardless of CM on individual handlers.

Applying CM to individual ux/uy handlers **before** stacking actually **increased** NS bandwidth from 134,088 to 154,309 (+15%) because CM scrambles the natural mesh ordering that previously aligned well across components.

### 4.2 The Fix — Node-Wise Interleaving

Instead of block ordering, interleave velocity components by mesh node:

```
Interleaved ordering:  [ux_0, uy_0, ux_1, uy_1, ..., ux_N, uy_N, p_0, p_1, ..., p_M]
```

This places velocity DoFs from the same mesh node adjacent (offset = 1 instead of ~62K).

### 4.3 Implementation

In `setup/ns_setup.cc`, the `setup_ns_coupled_system_parallel()` function builds local-to-global DoF maps. Added a `bool interleave_velocity` parameter:

```cpp
if (interleave_velocity)
{
    // Node-wise: [ux_0,uy_0, ux_1,uy_1, ..., p_0,p_1,...]
    auto ux_it = ux_owned.begin();
    auto uy_it = uy_owned.begin();
    for (; ux_it != ux_owned.end(); ++ux_it, ++uy_it)
    {
        ux_to_ns_map[*ux_it] = coupled_idx++;
        uy_to_ns_map[*uy_it] = coupled_idx++;
    }
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        p_to_ns_map[*it] = coupled_idx++;
}
else
{
    // Block: [all_ux | all_uy | all_p]
    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
        ux_to_ns_map[*it] = coupled_idx++;
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
        uy_to_ns_map[*it] = coupled_idx++;
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        p_to_ns_map[*it] = coupled_idx++;
}
```

### 4.4 Function Signature

```cpp
template <int dim>
void setup_ns_coupled_system_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    // ... constraints, maps, index sets, sparsity ...
    MPI_Comm mpi_comm,
    dealii::ConditionalOStream& pcout,
    bool interleave_velocity = false);  // NEW: default false for backward compat
```

The `interleave_velocity` flag is passed from the driver:
```cpp
setup_ns_coupled_system_parallel<dim>(
    ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
    ...,
    mpi_communicator_, pcout_,
    params_.renumber_dofs);  // interleave when CM is enabled
```

### 4.5 Constraint

Both `ux_owned` and `uy_owned` must have the same size (guaranteed when both use the same FE on the same mesh). The Trilinos Epetra requirement for contiguous ownership is satisfied because each MPI rank gets a contiguous block of coupled indices regardless of interleaving.

---

## 5. Results — Rosensweig Preset, Refinement 4

### 5.1 Raw Sparsity Data

**Configuration A: Default (no CM, no interleave)**
```
matrix    rows     cols     nnz       bandwidth   avg_bandwidth
CH        123906   123906   3940356   72520       31125.97
Poisson   61953    61953    985073    10567       298.95
NS        185346   185346   6397956   134088      51225.57
```

**Configuration B: CM on θ,ψ,φ + block NS (CM on ux,uy too — WRONG approach)**
```
matrix    rows     cols     nnz       bandwidth   avg_bandwidth
CH        123906   123906   3940356   63195       31102.62
Poisson   61953    61953    985073    1242        252.25
NS        185346   185346   6397956   154309      51199.89
```

**Configuration C: CM on θ,ψ,φ + interleaved NS (final, correct approach)**
```
matrix    rows     cols     nnz       bandwidth   avg_bandwidth
CH        123906   123906   3940356   63195       31102.62
Poisson   61953    61953    985073    1242        252.25
NS        185346   185346   6397956   133003      32441.77
```

### 5.2 Bandwidth Reduction

| System | Default b | CM b | Reduction | Factor |
|--------|-----------|------|-----------|--------|
| CH (coupled θ+ψ) | 72,520 | 63,195 | 12.9% | 1.15× |
| Poisson (φ) | 10,567 | 1,242 | **88.2%** | **8.5×** |
| NS (block, CM on ux/uy) | 134,088 | 154,309 | -15.1% (WORSE) | 0.87× |
| NS (interleaved) | 134,088 | 133,003 | 0.8% | 1.01× |

### 5.3 O(n × b²) Direct Solver Cost

| Config | CH n×b² | Poisson n×b² | NS n×b² | **Total** | vs Default |
|--------|---------|-------------|---------|-----------|------------|
| Default | 6.52e14 | 6.92e12 | 3.33e15 | **3.98e15** | — |
| CM + block | 4.95e14 | 9.55e7 | 4.41e15 | **4.91e15** | **+23% WORSE** |
| CM + interleave | 4.95e14 | 9.55e7 | 3.28e15 | **3.77e15** | **-5% better** |

### 5.4 Key Findings

1. **Poisson benefits enormously**: 88.2% bandwidth reduction → 72× cheaper factorization. CM is highly effective for single-field CG systems.

2. **CH benefits moderately**: 12.9% bandwidth reduction. The coupled (θ,ψ) system has bandwidth ~ n/2 because θ and ψ are interleaved in the same matrix.

3. **NS is the bottleneck**: NS dominates total cost (~84-87% of total n×b²). The saddle-point structure [A, B^T; B, 0] creates large bandwidth from the velocity-pressure offset, which CM cannot fix.

4. **CM on individual velocity handlers is HARMFUL**: Applying CM to ux and uy separately scrambles the natural ordering. When stacked into the monolithic system, bandwidth increased by 15%.

5. **Node-wise interleaving helps marginally**: Reduces NS avg_bandwidth significantly (51K → 32K) but max bandwidth only drops 0.8% because the v-p offset (~62K) still dominates.

6. **The nnz count is unchanged** across all configurations — CM and interleaving only reorder DoFs, they never add or remove matrix entries.

---

## 6. How to Replicate in Another Solver

### 6.1 Minimum Steps

1. **Add CLI flags** to parameters:
   ```cpp
   bool renumber_dofs = false;
   bool dump_sparsity = false;
   ```

2. **Apply CM after distribute_dofs()**, before extracting index sets:
   ```cpp
   #include <deal.II/dofs/dof_renumbering.h>

   dof_handler.distribute_dofs(fe);
   if (renumber_dofs)
       dealii::DoFRenumbering::Cuthill_McKee(dof_handler);
   locally_owned = dof_handler.locally_owned_dofs();
   ```

3. **Copy `sparsity_export.h`** into your utilities — it is self-contained and works with any `TrilinosWrappers::SparseMatrix`.

4. **Call the export** after matrices are assembled:
   ```cpp
   if (dump_sparsity)
   {
       auto a = analyze_sparsity(my_matrix, "SystemName");
       unsigned int global_bw = 0;
       MPI_Reduce(&a.bandwidth, &global_bw, 1, MPI_UNSIGNED, MPI_MAX, 0, comm);
       if (rank == 0) a.bandwidth = global_bw;
       export_sparsity_pattern(my_matrix, "system", output_dir, comm, pcout);
   }
   ```

5. **Run twice** — once without `--renumber-dofs`, once with — and compare `sparsity_summary.csv`.

6. **Compute O(n×b²)** from the CSV:
   ```
   Cost = rows × bandwidth²
   ```

### 6.2 For the Decoupled (CG) Solver

The Decoupled solver has separate subsystems, each with its own `*_setup.cc`:
- `cahn_hilliard/cahn_hilliard_setup.cc`
- `poisson/poisson_setup.cc`
- `magnetization/magnetization_setup.cc`
- `navier_stokes/navier_stokes_setup.cc`

For each, insert the CM call right after `distribute_dofs()` in the setup function. Since subsystems are solved independently (not monolithic), **CM can be applied to every CG handler** including ux and uy — the problem of monolithic NS stacking does not arise if NS is solved with separate component solves or a block preconditioner.

### 6.3 Expected Results for CG Solver

- **Scalar CG systems** (Poisson, CH components, angular momentum): expect 50-90% bandwidth reduction
- **Coupled CG systems** (monolithic NS): depends on how velocity/pressure are stacked
- **DG systems**: no benefit from CM (skip)
- If using iterative solvers (CG + AMG), bandwidth reduction has less direct impact — but reduced bandwidth improves cache locality and preconditioner quality

---

## 7. Files Created / Modified

### New Files
| File | Purpose |
|------|---------|
| `utilities/sparsity_export.h` | Sparsity analysis & export utility (self-contained, copy to other solvers) |

### Modified Files
| File | Changes |
|------|---------|
| `utilities/parameters.h` | Added `renumber_dofs`, `dump_sparsity` flags |
| `utilities/parameters.cc` | CLI parsing for `--renumber-dofs`, `--dump-sparsity` |
| `core/phase_field_setup.cc` | CM calls after `distribute_dofs()`, interleave flag to NS setup |
| `core/phase_field.cc` | Sparsity export integration, bandwidth in parallel diagnostics |
| `setup/ns_setup.h` | Added `interleave_velocity` parameter |
| `setup/ns_setup.cc` | Node-wise interleaved velocity ordering |
| `diagnostics/parallel_data.h` | Bandwidth fields per matrix |
| `output/parallel_diagnostics_logger.h` | Bandwidth columns in CSV |

### Output Files Generated
| File | Contents |
|------|---------|
| `sparsity_summary.csv` | One-row-per-matrix summary of all metrics |
| `{name}_bandwidth.csv` | Per-row nnz distribution + summary header |
| `{name}_sparsity.svg` | Visual sparsity pattern (small matrices only, < 5000 rows) |
| `{name}_sparsity.gnuplot` | Gnuplot-format sparsity data |

---

## 8. Verification

All 7 MMS tests pass with CM enabled:
- CH_STANDALONE, NS_STANDALONE, POISSON_STANDALONE, MAGNETIZATION_STANDALONE
- CH_NS, POISSON_MAGNETIZATION, FULL_SYSTEM

Convergence rates are identical — CM only reorders DoFs, it does not change the solution.
