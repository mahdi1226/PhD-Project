# Diagnostics: Issues Encountered and Resolutions

## Inherited from FHD Phase A

See `FHD/Report/Diagnostics.md` for full history of resolved issues from the
single-phase solver development. Key lessons:

1. NS viscous-MMS factor: ν(D(u),D(v)) → strong form -(ν/2)Δu
2. MMS 1/τ amplification: sources must use discrete old values
3. DG face velocity evaluation: must use fe_face_U at quadrature points
4. DG face dedup: each face processed once (smaller CellId)
5. H = ∇φ total field: never add h_a separately (double-counting)

## Phase B Issues

(To be populated as development progresses)
