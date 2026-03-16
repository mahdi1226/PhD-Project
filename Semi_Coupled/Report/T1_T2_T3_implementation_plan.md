# Implementation Plan: Add Zhang-He-Yang Terms (T1, T2, T3)

## Context

The Semi_Coupled code currently implements the base Nochetto CMAME 2016 scheme (Eq. 42), which omits three nonlinear terms from the full Shliomis model. The user's proofs paper (`/tmp/Proofs/`) provides complete theoretical analysis (energy stability, existence, convergence) for adding these terms. This plan implements all three as switchable features.

**Terms to add:**
- **T1** (Landau-Lifshitz damping): −β M^{k-1}×(M^{k-1}×H^k) in magnetization eq
- **T2** (Spin-vorticity coupling): +½(∇×U)×M in magnetization eq
- **T3** (Antisymmetric magnetic stress): +½μ₀(M^k×H^k, ∇×V) in NS eq

**Scheme correspondence:**
- Base Nochetto: T1=off, T2=off, T3=off
- Scheme I (conditional stability): T1+T2 on, T3 off, bgs=1
- Scheme II (unconditional stability): T1+T2+T3 on, bgs>1

---

## Step 1: Add Parameters (`parameters.h` + `parameters.cc`)

**File:** `utilities/parameters.h` — Add to `Physics` struct (after line 121):
```cpp
double beta = 0.0;          // Landau-Lifshitz damping coefficient (T1)
bool enable_T1 = false;     // −β M×(M×H) in magnetization
bool enable_T2 = false;     // ½(∇×u)×M in magnetization
bool enable_T3 = false;     // ½μ₀(M×H, ∇×v) in NS
```

**File:** `utilities/parameters.cc` — Add CLI flags in `parse_command_line()` (after line 608):
```cpp
else if (std::strcmp(argv[i], "--T1") == 0) params.physics.enable_T1 = true;
else if (std::strcmp(argv[i], "--T2") == 0) params.physics.enable_T2 = true;
else if (std::strcmp(argv[i], "--T3") == 0) params.physics.enable_T3 = true;
else if (std::strcmp(argv[i], "--zhang") == 0)  // convenience: all three
{ params.physics.enable_T1 = params.physics.enable_T2 = params.physics.enable_T3 = true; }
else if (std::strcmp(argv[i], "--beta") == 0)
{ if (++i >= argc) exit(1); params.physics.beta = std::stod(argv[i]); }
```

Also add `--help` text and verbose config output.

---

## Step 2: Add T1 + T2 to Magnetization Assembler

**File:** `assembly/magnetic_assembler.cc`

### T1: Landau-Lifshitz (LHS matrix contribution)

T1 is −β⟨M^{k-1}×(M^{k-1}×H^k), Z⟩ where H^k = ∇φ^k is a **trial function** in the monolithic system. Using the BAC-CAB identity:

```
M×(M×H) = M(M·H) − |M|²H
```

This gives a **matrix contribution** (linear in ∇φ_j, tested with Z_i):

**Insert in the (i,j) matrix loop** (after line 335, before `cell_matrix(i,j) += val * JxW`):
```cpp
// T1: Landau-Lifshitz damping — LHS matrix (C_M_phi block)
// −β(M^{k-1}×(M^{k-1}×∇φ_j), Z_i)
// = β[|M_old|²(∇φ_j · Z_i) − (M_old · ∇φ_j)(M_old · Z_i)]
if (params_.physics.enable_T1 && params_.physics.beta > 0.0)
{
    const double M_dot_grad_phi = M_old_vals[q] * grad_phi_j;
    const double M_dot_Z       = M_old_vals[q] * Z_i;
    const double M_sq           = M_old_vals[q] * M_old_vals[q];
    val += params_.physics.beta * (M_sq * (grad_phi_j * Z_i)
                                    - M_dot_grad_phi * M_dot_Z);
}
```

**Note:** `grad_phi_j` and `Z_i` are already computed at lines 316 and 307.

### T2: Spin-Vorticity (RHS contribution for Scheme I)

T2 is +½⟨(∇×U^{k-1})×M^{k-1}, Z⟩. Both U and M are lagged → **explicit RHS**.

In 2D: ∇×U = ∂U_y/∂x − ∂U_x/∂y (scalar ω), and ω×M = (−ωM_y, ωM_x).

**Pre-compute before the i-loop** (after line 273, alongside `div_U`):
```cpp
// Curl of velocity (2D scalar vorticity)
const double curl_U = grad_Uy[q][0] - grad_Ux[q][1];
```

**Insert in the RHS loop** (after line 349, before MMS):
```cpp
// T2: Spin-vorticity coupling — RHS (explicit, Scheme I)
// Move +½(∇×U)×M to RHS: subtract from rhs_val
// In 2D: (∇×U)×M = curl_U · (-M_y, M_x)
if (params_.physics.enable_T2)
{
    dealii::Tensor<1, dim> curl_cross_M;
    curl_cross_M[0] = -curl_U * M_old_vals[q][1];
    curl_cross_M[1] =  curl_U * M_old_vals[q][0];
    rhs_val -= 0.5 * (curl_cross_M * Z_i);
}
```

---

## Step 3: Add T3 to NS Assembler

**File:** `assembly/ns_assembler.cc`

T3 in weak form: +½μ₀⟨M^k×H^k, ∇×V⟩ on the NS RHS.

In 2D:
- M×H = M_x·H_y − M_y·H_x (scalar)
- ∇×V = ∂V_y/∂x − ∂V_x/∂y (scalar)

For separated ux/uy test functions:
- V=(φ_ux, 0): ∇×V = −∂φ_ux/∂y → contribution = ½μ₀(MxH)(−grad_ux[1])
- V=(0, φ_uy): ∇×V = +∂φ_uy/∂x → contribution = ½μ₀(MxH)(+grad_uy[0])

### New static function (after `assemble_kelvin_force`, ~line 550):

Create `assemble_antisymmetric_stress()` following the **exact same pattern** as `assemble_kelvin_force()` (same signature, same DoFHandler args, same distribution mechanism). The function:

1. Takes same args as Kelvin force (phi_solution, Mx_solution, My_solution, DoFHandlers, etc.)
2. Loops over locally owned cells
3. At each quadrature point computes:
   ```cpp
   double MxH = Mx_values[q] * H[1] - My_values[q] * H[0];
   double coeff = 0.5 * mu_0 * MxH;
   ```
4. Accumulates RHS:
   ```cpp
   local_rhs_ux(i) += coeff * (-grad_phi_ux_i[1]) * JxW;  // −∂φ_ux/∂y
   local_rhs_uy(i) += coeff * ( grad_phi_uy_i[0]) * JxW;  // +∂φ_uy/∂x
   ```
5. Distributes via `ns_constraints.distribute_local_to_global()`

### Call from unified pipeline

In the main `assemble_ns_system()` function (around line 1001-1024), add after Kelvin force call:
```cpp
if (params.physics.enable_T3)
    assemble_antisymmetric_stress<dim>(...same args as kelvin...);
```

---

## Step 4: MMS Source Terms (Verification)

**Files:** `mms/magnetization/magnetization_mms.h`, `mms/ns/ns_mms.h`

### Magnetization MMS source update
The manufactured magnetization source must include T1 and T2 contributions when enabled:
- T1 source: +β M*×(M*×H*) evaluated from exact solutions
- T2 source: −½(∇×U*)×M* evaluated from exact solutions

### NS MMS source update
- T3 source: −½μ₀∇×(M*×H*) evaluated from exact solutions

These are computed analytically from the existing manufactured solutions (M*, U*, H* = ∇φ*).

---

## Step 5: Rebuild and Test

1. **Build**: `cd cmake-build-release && make -j8 ferrofluid`
2. **Smoke test** (base unchanged): `mpirun -np 4 ferrofluid --rosensweig -r 4 --dt 0.002 --max_steps 10`
3. **T1 only**: `mpirun -np 4 ferrofluid --rosensweig -r 4 --T1 --beta 0.01 --max_steps 100`
4. **T2 only**: `mpirun -np 4 ferrofluid --rosensweig -r 4 --T2 --max_steps 100`
5. **T3 only**: `mpirun -np 4 ferrofluid --rosensweig -r 4 --T3 --max_steps 100`
6. **Full Zhang (Scheme II)**: `mpirun -np 4 ferrofluid --rosensweig -r 4 --zhang --beta 0.01 --bgs_iters 3 --max_steps 100`
7. **MMS convergence**: Run with `--mms --T1 --T2 --T3 --beta 0.01` at refinements 3,4,5 — verify O(h) or O(h²) convergence

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `utilities/parameters.h` | Add beta, enable_T1/T2/T3 to Physics struct |
| `utilities/parameters.cc` | Add --T1, --T2, --T3, --zhang, --beta CLI flags + help |
| `assembly/magnetic_assembler.cc` | T1 matrix block (~8 lines) + T2 RHS (~6 lines) + curl_U computation (~1 line) |
| `assembly/ns_assembler.cc` | New `assemble_antisymmetric_stress()` function (~80 lines) + call from pipeline (~3 lines) |
| `mms/magnetization/magnetization_mms.h` | T1+T2 exact source contributions |
| `mms/ns/ns_mms.h` (or `.cc`) | T3 exact source contribution |

**Total new code**: ~120 lines (assembly) + ~40 lines (parameters) + ~60 lines (MMS) ≈ 220 lines

---

## Key Design Decisions

1. **T1 on LHS (matrix), not RHS**: Since H^k = ∇φ^k is a trial function in the monolithic M+φ system, T1 is linear in φ^k with M^{k-1} lagged. This preserves the monolithic solve structure.

2. **T2 explicit (Scheme I default)**: With bgs=1, T2 uses M^{k-1} and U^{k-1} (both lagged) → RHS only. With bgs>1, subsequent iterations see updated U^k, naturally transitioning to Scheme II behavior.

3. **T3 as separate function**: Follows the established pattern of `assemble_kelvin_force()`, `assemble_capillary_force()`, `assemble_gravity_force()` — clean, testable, parallel-safe.

4. **Switches independent**: Each term can be toggled independently for testing and debugging. `--zhang` is a convenience flag for all three.

5. **2D only**: Cross products use 2D specializations. The code is already 2D-only (template instantiated for dim=2 only).
