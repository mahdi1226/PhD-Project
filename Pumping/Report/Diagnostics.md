# Diagnostics: Issues Encountered and Resolutions

## Inherited from FHD Phase A

See `FHD/Report/Diagnostics.md` for full history of resolved issues from the
single-phase solver development. Key lessons:

1. NS viscous-MMS factor: nu(D(u),D(v)) -> strong form -(nu/2)*Delta(u)
2. MMS 1/tau amplification: sources must use discrete old values
3. DG face velocity evaluation: must use fe_face_U at quadrature points
4. DG face dedup: each face processed once (smaller CellId)
5. H = grad(phi) total field: never add h_a separately (double-counting)

## Phase B Issues

### Issue B1: DataComponentInterpretation include (resolved)
- **Problem**: `#include <deal.II/base/data_component_interpretation.h>` not found
- **Cause**: Not a separate header in deal.II
- **Fix**: Remove include; DataComponentInterpretation is available through `<deal.II/numerics/data_out.h>`

### Issue B2: FEValues get_function_values API (resolved)
- **Problem**: `fe_values[extractor].get_function_values(vec)[0]` doesn't compile
- **Cause**: Return value is not directly subscriptable
- **Fix**: Use buffer vector:
  ```cpp
  std::vector<double> phi_vals(n_q_points);
  fe_values[phi_extractor].get_function_values(ch_solution, phi_vals);
  double phi_q = phi_vals[q];
  ```

### Issue B3: Rosensweig instability -- no spike formation (OPEN)
- **Symptoms**: Multiple runs with H=10 and H=30, refinement levels 3-4, 500+ steps.
  Maximum velocity U_max=0.005-0.165 but no visible interface deformation.
- **Runs attempted**:
  - r=3, H=10, 500 steps (t=0.5): U_max=0.016
  - r=4, H=10, 280 steps (t=0.28): U_max=0.005
  - r=4, H=30, 545 steps (t=0.545): U_max=0.165
- **Interface tracking**: Cell-midpoint method with |phi|<0.9 threshold too coarse (h=0.018)
- **Possible causes**:
  1. Parameters (eps=0.02 too large, nu=2 too high, missing gravity)
  2. Numerical scheme (implicit coupling over-damps instability growth)
  3. Resolution (r=4 may be insufficient for eps=0.02)
- **Next steps**: Check Zhang's exact parameters (Section 4.3), then decide on scheme change
- **Literature context**: User confirms Nochetto's scheme never produced spikes across 3+ projects,
  while Zhang's scheme does. Likely a scheme issue, not just parameters.

### Issue B4: CH chemical potential -- magnetic energy term NOT needed (resolved)
- **Problem**: Initially proposed adding -0.5*mu_0*chi'(phi)*|H|^2 to CH chemical potential
- **Resolution**: After reviewing all 13 reference papers, confirmed NO paper does this.
  The magnetic coupling enters through chi(phi) in magnetization and Kelvin force in NS.
  The variational derivative of the magnetic energy is handled through the coupled
  system's energy estimate, not as an explicit term in W.
- **Impact**: Cancelled the planned modification. Current implementation is correct.

### Issue B5: Capillary force form equivalence (resolved)
- **Question**: Is sigma*mu*grad(phi) (our form) equivalent to (lambda/eps)*phi*grad(W)?
- **Answer**: Yes, modulo a pressure gradient term. Nochetto's Remark 2.1 shows these
  are related by pressure redefinition: p_new = p + (lambda/eps)*W. Both forms yield
  the same physics. Our form sigma*mu*grad(phi) is correct and widely used.
