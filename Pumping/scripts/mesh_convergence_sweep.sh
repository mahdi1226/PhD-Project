#!/bin/bash
# ============================================================================
# Mesh convergence sweep: D(h) at fixed H0 and epsilon
#
# Purpose: Verify that D converges with mesh refinement at fixed eps.
# Fixes H0 and epsilon, varies refinement level (ref 5, 6, 7).
#
# Expected behavior:
#   - D should converge as h -> 0 (mesh independent)
#   - At eps=0.02: ref 5 (h=1/32=0.031, eps/h=0.64), ref 6 (eps/h=1.28),
#     ref 7 (eps/h=2.56). Only ref 7 truly resolves the interface.
#   - At eps=0.04: ref 5 (eps/h=1.28), ref 6 (eps/h=2.56) — better resolved.
#
# WARNING: ref 7 runs are ~4x more expensive than ref 6 (4x cells, 4x DoFs).
#          Expect ~30-60 min per run at ref 7.
#
# Usage:
#   bash scripts/mesh_convergence_sweep.sh
#   bash scripts/mesh_convergence_sweep.sh --h0 2.0 --eps 0.04
# ============================================================================

DRIVER="/Users/mahdi/Projects/git/PhD-Project/Pumping/build/drivers/fhd_ch_driver"
CHI=1.19
H0=2.0
EPS=0.02
DT=1e-3
T_FINAL=3.0
RAMP_TIME=0.5
VTK_INTERVAL=500
TAG="mesh_conv"

REF_VALUES=(5 6 7)

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --h0)   H0="$2";   shift 2 ;;
        --eps)  EPS="$2";  shift 2 ;;
        --chi)  CHI="$2";  shift 2 ;;
        --dt)   DT="$2";   shift 2 ;;
        --t-final) T_FINAL="$2"; shift 2 ;;
        *)      shift ;;
    esac
done

echo "============================================================"
echo "Mesh Convergence Sweep: D(h) at fixed H0 and epsilon"
echo "  H0=$H0, Chi=$CHI, eps=$EPS, dt=$DT, t_final=$T_FINAL"
echo "  Refinement levels: ${REF_VALUES[*]}"
echo "  Start: $(date)"
echo "============================================================"

# Run sequentially (ref 7 is memory-intensive, don't overlap)
for REF in "${REF_VALUES[@]}"; do
    H_SIZE=$(python3 -c "print(f'{1.0 / 2**$REF:.5f}')")
    EPS_H=$(python3 -c "print(f'{$EPS / (1.0 / 2**$REF):.2f}')")
    echo ""
    echo "[$(date +%H:%M:%S)] Launching ref=$REF (h=$H_SIZE, eps/h=$EPS_H)"

    mpirun -np 1 "$DRIVER" \
        --droplet-deformation \
        -r "$REF" \
        --chi "$CHI" \
        --field-strength "$H0" \
        --ramp-time "$RAMP_TIME" \
        --epsilon "$EPS" \
        --dt "$DT" \
        --t-final "$T_FINAL" \
        --vtk-interval "$VTK_INTERVAL"

    echo "[$(date +%H:%M:%S)] ref=$REF complete"
done

echo ""
echo "[$(date +%H:%M:%S)] All mesh-convergence runs complete!"
echo ""
echo "Run: python3 scripts/analyze_convergence.py --type mesh"
