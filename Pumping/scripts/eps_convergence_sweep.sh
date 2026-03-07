#!/bin/bash
# ============================================================================
# Epsilon convergence sweep: D(eps) at fixed H0 and mesh
#
# Purpose: Verify that D converges as eps -> 0 (sharp-interface limit).
# Fixes H0 and refinement level, varies epsilon.
#
# Expected behavior:
#   - D should converge to the sharp-interface limit as eps -> 0
#   - Need eps/h < ~0.25 for adequate CH resolution (4 cells across interface)
#   - At ref 6 (h=1/64≈0.0156): eps >= 0.04 gives eps/h >= 2.5 (well-resolved)
#     eps=0.02 gives eps/h=1.3 (marginal), eps=0.01 gives eps/h=0.64 (under-resolved)
#
# NOTE: For under-resolved eps, increase refinement. This script keeps ref
#       fixed to isolate the eps effect. Use mesh_convergence_sweep.sh to
#       study mesh effects at fixed eps.
#
# Usage:
#   bash scripts/eps_convergence_sweep.sh
#   bash scripts/eps_convergence_sweep.sh --ref 7 --h0 2.0
# ============================================================================

DRIVER="/Users/mahdi/Projects/git/PhD-Project/Pumping/build/drivers/fhd_ch_driver"
CHI=1.19
REF=6
H0=2.0
DT=1e-3
T_FINAL=3.0
RAMP_TIME=0.5
VTK_INTERVAL=500
MAX_PARALLEL=2
TAG="eps_conv"

EPS_VALUES=(0.04 0.02 0.01 0.005)

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)  REF="$2";  shift 2 ;;
        --h0)   H0="$2";   shift 2 ;;
        --chi)  CHI="$2";  shift 2 ;;
        --dt)   DT="$2";   shift 2 ;;
        --t-final) T_FINAL="$2"; shift 2 ;;
        *)      shift ;;
    esac
done

echo "============================================================"
echo "Epsilon Convergence Sweep: D(eps) at fixed H0 and mesh"
echo "  H0=$H0, Chi=$CHI, Ref=$REF, dt=$DT, t_final=$T_FINAL"
echo "  Epsilon values: ${EPS_VALUES[*]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Start: $(date)"
echo "============================================================"

pids=()
for EPS in "${EPS_VALUES[@]}"; do
    EPS_H=$(python3 -c "print(f'{$EPS / (1.0 / 2**$REF):.2f}')")
    echo "[$(date +%H:%M:%S)] Launching eps=$EPS (eps/h=$EPS_H)"

    mpirun -np 1 "$DRIVER" \
        --droplet-deformation \
        -r "$REF" \
        --chi "$CHI" \
        --field-strength "$H0" \
        --ramp-time "$RAMP_TIME" \
        --epsilon "$EPS" \
        --dt "$DT" \
        --t-final "$T_FINAL" \
        --vtk-interval "$VTK_INTERVAL" \
        > /dev/null 2>&1 &
    pids+=($!)

    sleep 3  # avoid timestamp collision

    # Throttle: wait for oldest if at max parallel
    if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "[$(date +%H:%M:%S)] All eps-convergence runs complete!"
echo ""
echo "Run: python3 scripts/analyze_convergence.py --type eps"
