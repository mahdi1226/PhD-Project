# Parallel Computation Improvement Plan
## Two-Phase Ferrofluid Solver (deal.II / MPI)

---

## Candidate Titles

> These titles describe the contribution at a technical paper/dissertation section level.

1. **Physics-Informed Sparsity Exploitation for Parallel Multiphysics FEM Solvers**
2. **Interface-Driven Sparsity Reduction in Coupled Two-Phase Flow Simulations**
3. **Phase-Field-Guided Matrix Compression for Parallel Ferrofluid FEM Systems**
4. **Operator-Aware Ghost Layer Minimization in MPI-Parallel Multiphysics Assemblies**
5. **Energy-Stable Parallel Assembly with Physics-Constrained Sparsity Patterns**
6. **Exploiting Phase-Field Structure for Efficient Parallel Matrix Assembly in Two-Phase FEM**

---

## 1. Motivation & Reason

The current parallel implementation uses deal.II libraries as a **black box** — the parallel scheme (assembly, communication, partitioning) is accepted as-is without physics-informed modification. However, our system has a fundamentally **non-uniform medium** (two-phase ferrofluid with a sharp interface governed by the Cahn-Hilliard phase field φ). This means:

- The computational work is **heavily concentrated near the interface** (φ ≈ 0)
- Bulk regions (φ ≈ ±1) are mathematically simple — smooth, slowly varying fields
- Coupling terms (Kelvin force, CH-NS coupling) are **nearly zero in bulk** by physics
- Standard parallel schemes treat all regions equally, wasting computation in bulk

The key insight is that **the sparsity pattern of the assembled matrices reflects the physics** — many nonzero entries in bulk regions are structurally present but numerically zero or negligible. By controlling the sparsity pattern, we can:

1. Reduce the number of active matrix entries
2. Reduce ghost layer communication as a natural byproduct
3. Improve load balance across MPI ranks

---

## 2. Goal

> **Primary Goal:** Control and improve the sparsity pattern of assembled matrices by exploiting the physics of the two-phase system — specifically, the fact that many coupling terms vanish in bulk regions where ∇φ ≈ 0.

> **Secondary Goal (Byproduct):** A sparser matrix pattern naturally reduces the ghost layer data exchanged between MPI ranks, lowering communication overhead without explicit communication tuning.

The contribution is **mathematically motivated**, not a pure computer science optimization — it is grounded in the operator structure of the coupled ferrofluid PDE system.

---

## 3. Mathematical Justification

The coupling terms in the assembled matrices involve products and powers of φ and ∇φ:

- Kelvin force: `f_K ~ μ₀(M·∇)H` — concentrated at interface
- Capillary force: `f_cap ~ σ μ ∇φ` — proportional to ∇φ, zero in bulk
- CH-NS coupling: involves `∇φ`, `φ²`, `∇φ·∇φ`
- Magnetization: `M = χ(θ)∇Φ` — susceptibility χ(θ) interpolates with phase field
- Variable viscosity: `ν(θ)` — varies only across interface

**Key observation:** φ ∈ [-1, 1] throughout the domain. In bulk regions:
- ∇φ ≈ 0 (field is flat)
- Products and powers of ∇φ become even smaller: `|∇φ|² ≪ |∇φ|`
- Any matrix entry involving ∇φ is effectively zero

**Threshold:** Entries below O(ε) (interface thickness) can be safely dropped. This is not a heuristic — it is controlled by the same ε that governs the interface width in the Cahn-Hilliard equation.

**Two cases for dropped entries:**
- **Exactly zero:** Dropped trivially, no justification needed
- **Approximately zero (< O(ε)):** Requires proof that the modified scheme still converges and energy stability is preserved — this is the mathematical content of the contribution

### Critical Distinction: Coupling Terms vs Intra-System Terms

> **WARNING:** Only **inter-system coupling** terms can be dropped in bulk. The **intra-system** operators must NEVER be skipped:
> - CH diffusion (ε∇²θ) — active everywhere, maintains interface
> - NS viscosity (ν∇²u) — active everywhere, required for well-posedness
> - Poisson Laplacian (∇²φ) — active everywhere, elliptic operator
>
> Droppable terms are ONLY the coupling contributions:
> - Kelvin force → NS RHS (zero when ∇φ ≈ 0, so M ≈ 0 or constant)
> - Capillary force → NS RHS (proportional to μ∇φ, zero in bulk)
> - CH convection u·∇θ → CH (velocity-driven, but θ is flat in bulk so ∇θ ≈ 0)
> - Variable viscosity (ν(θ) - ν_constant) → NS matrix (variation only at interface)

---

## 4. Tools

### 4.1 Sparsity Pattern Visualization
- **Reference:** deal.II Step-2 tutorial (https://dealii.org/current/doxygen/deal.II/step_2.html)
- **Method:** Plot sparsity pattern of assembled matrices before and after modification
- **deal.II API:**
  ```cpp
  SparsityPattern::print_svg(std::ofstream("sparsity.svg"));
  SparsityPattern::print_gnuplot(std::ofstream("sparsity.gpl"));
  ```
- **Trilinos note:** Our matrices are `TrilinosWrappers::SparseMatrix`, not native deal.II `SparsityPattern`. Must extract the pattern to a `DynamicSparsityPattern` for SVG/gnuplot export. See Section 10.2 for implementation.
- **What to look for:** Compare sparsity patterns for interface cells vs bulk cells; identify how many nonzeros are in bulk-bulk coupling blocks

### 4.2 Ghost Layer Analysis
- **What it is:** The set of cells owned by neighboring MPI ranks that must be exchanged for assembly
- **Connection to sparsity:** A sparser pattern means fewer DoF dependencies across rank boundaries → fewer ghost entries needed
- **How to inspect:** `DoFHandler::n_locally_relevant_dofs()` vs `DoFHandler::n_locally_owned_dofs()` — the difference is the ghost layer size

### 4.3 DoF Renumbering (Baseline Comparison)
- **deal.II API:** `DoFRenumbering::Cuthill_McKee(dof_handler)`
- **Purpose:** Standard bandwidth reduction — provides a **baseline** sparsity improvement to compare against physics-informed improvement
- **Why it helps:** Direct solvers (UMFPACK) have cost O(n × b²) where b = bandwidth. Reducing b via Cuthill-McKee reduces fill-in during LU factorization → faster solve, less memory
- **Limitation:** CM only rearranges existing entries — it does NOT remove entries. Physics-informed approach actually removes entries, which CM cannot achieve. This is the key differentiator for our contribution.

---

## 5. Approach

### Step 0 — Instrumentation (DONE ✓)
Parallel diagnostics infrastructure implemented:
- `diagnostics/parallel_data.h` — Per-rank timing, sparsity, load balance data struct
- `output/parallel_diagnostics_logger.h` — CSV writer with MPI reductions
- `phase_field.cc` instrumented with assembly vs solve split timers for all 4 subsystems
- CLI flags: `--parallel-diag`, `--parallel-diag-all-ranks`
- Output: `parallel_diagnostics.csv` per run

**Metrics currently recorded per step:**
- Assembly time vs solve time for CH, Poisson, Magnetization, NS (separate)
- Total nnz per matrix (CH, Poisson, Mag, NS — local + global)
- Local cells, ghost cells, DoFs per rank per subsystem
- MPI-reduced imbalance ratios: max/avg step time, per-subsystem imbalance
- Memory per rank (cross-platform: macOS + Linux)
- Picard/BGS iteration counts, solver iteration counts
- AMR timing, diagnostics timing, output timing

### Step 1 — Sparsity Pattern Recording (DONE ✓)
Full sparsity analysis implemented via `utilities/sparsity_export.h`:
- SVG/gnuplot export for matrices < 5000 rows
- Per-row nnz distribution (bandwidth CSV)
- Matrix bandwidth (max |i-j| for nonzero a_{ij}) with MPI reduction
- Summary CSV with all metrics (rows, cols, nnz, bandwidth, avg_bandwidth, density)
- CLI flag: `--dump-sparsity`
- Bandwidth cached in parallel diagnostics CSV (per-step)

### Step 2 — Cuthill-McKee Baseline (DONE ✓)
CM renumbering implemented with O(n×b²) comparison:
- `DoFRenumbering::Cuthill_McKee()` applied to CG handlers (θ, ψ, φ) after `distribute_dofs()`
- Toggle via `--renumber-dofs` / `--no-renumber-dofs`
- DG handlers (M, p) skipped (cell-local, no inter-element coupling)
- NS velocity: individual CM harmful (+15% bandwidth); node-wise interleaving used instead
- Results: Poisson 88% bandwidth reduction (72× cheaper), CH 13%, NS ~1%
- Total O(n×b²): 5% improvement (NS dominates at 84% of total cost)
- See `Report/SPARSITY_ANALYSIS_REPORT.md` for full methodology and results

### Step 3 — Physics-Informed Pattern Identification (TODO)
For each assembled matrix, identify entries that are zero or near-zero by physics:
- Evaluate φ and |∇φ| at cell centers during assembly
- Flag cells where |∇φ| < threshold (ε-based) as "bulk cells"
- Identify which DoF pairs connected through bulk cells have negligible coupling
- **Only coupling terms** — never intra-system operators (see Section 3 warning)

### Step 4 — Modified Assembly (TODO)
During matrix assembly, skip **coupling** contributions from bulk cells where coupling terms are provably below O(ε):
```cpp
// In NS assembly (Kelvin/capillary force contribution):
double grad_phi_norm = grad_phi_at_cell_center.norm();
if (grad_phi_norm < epsilon_threshold) {
    // Skip coupling force terms — bulk cell, contribution is O(ε)
    // Still assemble: viscous term, pressure gradient, time derivative
    continue;  // skip ONLY the coupling RHS contribution
}
```

### Step 5 — Verification (TODO)
- Confirm MMS convergence rates are preserved with modified pattern
- Confirm energy stability is maintained (E_internal should still dissipate)
- Record new nonzero count, ghost layer size, and timing metrics
- Run threshold sensitivity study: how does improvement scale with ε? (ε = 0.01, 0.005, 0.002)

### Step 6 — 4-Configuration Comparison (TODO)
The experimental design for the paper:

| Configuration | Description | Shows |
|---|---|---|
| **A: Default** | No renumbering, no physics skip | Baseline (control) |
| **B: Cuthill-McKee only** | Standard bandwidth reduction | Known CS technique |
| **C: Physics-informed only** | Skip coupling in bulk, no CM | Our novel contribution |
| **D: Both combined** | CM + physics-informed | Best combined result |

Comparison metrics: solve time, assembly time, memory, nnz count, bandwidth, ghost layer size, imbalance ratio.

### Step 7 — Scaling Study (TODO)
Run all 4 configurations at multiple MPI rank counts to show the improvement matters at scale:
- 1, 2, 4, 8, 16, 32 MPI ranks
- Fixed problem size (strong scaling) and increasing problem size (weak scaling)
- Plot: speedup, efficiency, communication volume vs rank count

---

## 6. Metrics to Record

### 6.1 Timing Metrics ✓ (Implemented)
- Wall-time per matrix assembly per subsystem (CH, NS, Poisson, Magnetization)
- Wall-time per solver per subsystem
- Step total time per rank
- Diagnostics computation time
- AMR time, VTK output time
- Preconditioner setup time vs apply time (TODO — not yet split)

### 6.2 Sparsity & Matrix Metrics (Partially Implemented)
- ✓ Number of nonzeros in sparsity pattern (before and after modification) — total nnz recorded
- **TODO:** Number of nonzeros per row (distribution, not just total)
- **TODO:** Sparsity pattern structure (plotted via SVG/gnuplot)
- **TODO:** Matrix bandwidth (max |i-j| for nonzero A(i,j))
- **TODO:** Sparsity pattern before/after Cuthill-McKee comparison
- **TODO:** Identify and count near-zero entries (|A(i,j)| < ε threshold)

### 6.3 Communication Metrics (Partially Implemented)
- ✓ Ghost layer size: `n_locally_relevant_dofs - n_locally_owned_dofs` — recorded as local/global DoFs
- **TODO:** Volume of data exchanged per MPI rank (bytes sent/received)
- **TODO:** Number of MPI messages per timestep
- **TODO:** Idle/waiting time per rank (reveals load imbalance beyond work imbalance)

### 6.4 Solver Metrics ✓ (Implemented)
- ✓ Number of iterations to convergence per solve (all subsystems)
- ✓ Picard iteration count, BGS iteration count
- **TODO:** Residual norm per iteration (convergence history per solve)

### 6.5 Load Balance Metrics ✓ (Implemented)
- ✓ DOF count per rank (per subsystem)
- ✓ Cell count per rank
- ✓ Work imbalance ratio = `max(rank_time) / mean(rank_time)` (overall + per subsystem)
- ✓ Cell/DoF min/max across ranks

### 6.6 Memory Metrics ✓ (Implemented)
- ✓ Memory usage per rank (cross-platform: macOS via mach_task_basic_info, Linux via rusage)
- ✓ Peak memory min/max across ranks

### 6.7 Physics-Correlated Metrics
- **TODO:** All above metrics recorded separately for **interface cells vs bulk cells**
- ✓ Timestep number when measurements are taken (interface evolves over time)

---

## 7. Libraries & Tools

### deal.II (Primary)
| Feature | API | Status |
|---|---|---|
| Wall-time measurement | `CumulativeTimer` (custom, uses `std::chrono`) | ✓ Implemented |
| Nonzero count | `TrilinosWrappers::SparseMatrix::n_nonzero_elements()` | ✓ Implemented |
| Nonzeros per row | `SparsityPattern::row_length(i)` | TODO — need extraction |
| Sparsity plot | `SparsityPattern::print_svg()` / `print_gnuplot()` | TODO — need Trilinos→deal.II extraction |
| DOF count per rank | `DoFHandler::n_locally_owned_dofs()` | ✓ Implemented |
| Cell count per rank | `Triangulation::n_locally_owned_active_cells()` | ✓ Implemented |
| Solver iterations | `SolverControl::last_step()` / `SolverInfo` | ✓ Implemented |
| Residual norm | `SolverControl::last_value()` | ✓ Implemented |
| DoF renumbering | `DoFRenumbering::Cuthill_McKee()` | TODO — Step 2 |
| Ghost layer size | `n_locally_relevant - n_locally_owned` | ✓ Implemented |
| Per-rank imbalance | `MPI_Allreduce` (min/max/sum) | ✓ Implemented |
| Memory usage | `mach_task_basic_info` (macOS) / `rusage` (Linux) | ✓ Implemented |

### MPI (Standard, alongside deal.II)
| Feature | API |
|---|---|
| Bytes sent/received | `MPI_Get_count` |
| Per-rank timing | `MPI_Wtime` / `std::chrono` |
| Rank synchronization check | `MPI_Barrier` + `MPI_Wtime` |

### External Tools (Optional, Non-invasive)
| Tool | Purpose |
|---|---|
| **mpiP** | Lightweight MPI profiling — message counts, volumes, idle time |
| **TAU** | Full performance profiling with MPI and OpenMP support |
| **Valgrind/Callgrind** | Memory and cache profiling |
| **perf** (Linux) | CPU-level performance counters |
| `/proc/self/status` | Memory usage per rank (VmPeak, VmRSS fields) — trivial to read in C++ |

---

## 8. Expected Contribution Statement

> "We identify and exploit the physics-informed sparsity structure of the assembled matrices in a coupled two-phase ferrofluid finite element system. By characterizing which matrix entries are provably negligible (O(ε)) in bulk regions where ∇φ ≈ 0, we construct a reduced sparsity pattern that preserves convergence and energy stability while reducing assembly cost and MPI ghost layer communication. This reduction is mathematically justified by the operator structure of the Cahn-Hilliard/Navier-Stokes/Magnetization system and is not achievable through standard DoF renumbering techniques alone."

---

## 9. References
- deal.II Step-2: Sparsity Pattern — https://dealii.org/current/doxygen/deal.II/step_2.html
- Nochetto, Salgado, Tomas — CMAME 2016 (base ferrofluid model)
- Zhang, He, Yang — 2021 (energy-stable extensions, Shliomis terms)
- Cuthill, McKee — "Reducing the bandwidth of sparse symmetric matrices" (1969)
- Codina — "A stabilized finite element method..." (1998) — SUPG framework for convection-dominated problems

---

## 10. Implementation Status

### 10.1 Completed (2026-03-05)

**Files created:**
- `diagnostics/parallel_data.h` — `ParallelStepData` struct with all per-rank metrics and `compute_mpi_reductions()` method
- `output/parallel_diagnostics_logger.h` — `ParallelDiagnosticsLogger` class with CSV output, per-rank file support, and summary generation

**Files modified:**
- `core/phase_field.h` — Added assembly/solve timing members: `last_ch_assemble_time_`, `last_ch_solve_time_`, `last_poisson_assemble_time_`, `last_poisson_solve_time_`, `last_mag_time_`, `last_ns_assemble_time_`, `last_ns_solve_time_`
- `core/phase_field.cc` — Instrumented `solve_ch()`, `solve_poisson()`, `solve_poisson_magnetization_picard()`, `solve_ns()` with assembly vs solve split timers. Added parallel diagnostics collection block after timing log. Added AMR timing. Added diagnostics computation timing.
- `utilities/parameters.h` — Added `enable_parallel_diagnostics`, `parallel_diag_all_ranks` flags
- `utilities/parameters.cc` — Added `--parallel-diag`, `--parallel-diag-all-ranks` CLI flags with help text

**Verified:** Build succeeds, 3-step Rosensweig test produces correct `parallel_diagnostics.csv` with all metrics populated.

### 10.2 TODO — Sparsity Pattern SVG Export

To export Trilinos matrix sparsity patterns as SVG (matching deal.II Step-2):
```cpp
// Extract Trilinos sparsity to deal.II SparsityPattern for visualization
void export_sparsity_pattern(const TrilinosWrappers::SparseMatrix& matrix,
                             const std::string& filename)
{
    // Only rank 0 writes (gather full pattern or write local portion)
    const auto& trilinos_matrix = matrix.trilinos_matrix();
    const int n_rows = trilinos_matrix.NumGlobalRows();

    DynamicSparsityPattern dsp(n_rows, n_rows);
    // Fill dsp from Trilinos CrsMatrix row structure...

    SparsityPattern sp;
    sp.copy_from(dsp);
    std::ofstream out(filename);
    sp.print_svg(out);
}
```
**Note:** For large matrices (>100K rows), SVG becomes impractical. Use gnuplot format or sample-based visualization instead.

### 10.3 TODO — Cuthill-McKee Implementation

```cpp
// In phase_field_setup.cc, after distribute_dofs():
#include <deal.II/dofs/dof_renumbering.h>

// Apply Cuthill-McKee renumbering (toggle via --renumber-dofs)
if (params_.enable_dof_renumbering)
{
    DoFRenumbering::Cuthill_McKee(theta_dof_handler_);
    DoFRenumbering::Cuthill_McKee(psi_dof_handler_);
    if (params_.enable_magnetic)
    {
        DoFRenumbering::Cuthill_McKee(phi_dof_handler_);
        // Note: M_dof_handler_ uses DG — CM has limited effect on DG
    }
    if (params_.enable_ns)
    {
        DoFRenumbering::Cuthill_McKee(ux_dof_handler_);
        DoFRenumbering::Cuthill_McKee(uy_dof_handler_);
        // Note: p_dof_handler_ uses DG — CM has limited effect on DG
    }
}
```

### 10.4 TODO — Physics-Informed Assembly Skip

```cpp
// In ns_assembler.cc — Kelvin force cell loop:
// Evaluate |∇φ| at cell center to determine if coupling is active
double grad_phi_norm = /* evaluate at cell center */;
if (grad_phi_norm < params.physics.epsilon) {
    // SKIP: Kelvin force, capillary force contributions
    // KEEP: viscous term, pressure, time derivative, convection
}
```

### 10.5 Experimental Matrix

| Step | Description | Output | Status |
|---|---|---|---|
| Step 0 | Instrumentation | `parallel_diagnostics.csv` | ✓ Done |
| Step 1 | Sparsity pattern recording | SVG files, per-row nnz CSV, bandwidth | ✓ Done |
| Step 2 | Cuthill-McKee baseline | Before/after O(n×b²) comparison | ✓ Done |
| Step 3 | Physics-informed identification | Bulk cell map, droppable entry count | TODO |
| Step 4 | Modified assembly | Reduced nnz, faster assembly | TODO |
| Step 5 | Verification | MMS rates, energy stability | TODO |
| Step 6 | 4-configuration comparison | Paper figures and tables | TODO |
| Step 7 | Scaling study | Strong/weak scaling plots | TODO |
