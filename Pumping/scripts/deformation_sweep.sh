#!/bin/bash
# ============================================================================
# Deformation sweep: D vs Bo_m for ferrofluid droplet under uniform field
# Afkhami et al. (2010) JFM 663 — droplet deformation
# ============================================================================

DRIVER="/Users/mahdi/Projects/git/PhD-Project/Pumping/build/drivers/fhd_ch_driver"
CHI=1.19
REF=6
DT=1e-3
T_FINAL=3.0
RAMP_TIME=0.5
VTK_INTERVAL=500
MAX_PARALLEL=3

H0_VALUES=(1.0 1.5 2.0 3.0 4.0 5.0)

echo "============================================================"
echo "Deformation Sweep: D vs Bo_m (Afkhami 2010)"
echo "  Chi=$CHI, Ref=$REF, dt=$DT, t_final=$T_FINAL"
echo "  H0 = ${H0_VALUES[*]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Start: $(date)"
echo "============================================================"

pids=()
for H0 in "${H0_VALUES[@]}"; do
    BO_M=$(python3 -c "print(f'{0.2 * $CHI * $H0**2:.4f}')")
    echo "[$(date +%H:%M:%S)] Launching H0=$H0 (Bo_m=$BO_M)"

    mpirun -np 1 "$DRIVER" \
        --droplet-deformation \
        -r "$REF" \
        --chi "$CHI" \
        --field-strength "$H0" \
        --ramp-time "$RAMP_TIME" \
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
echo "[$(date +%H:%M:%S)] All runs complete!"
echo ""

# Analyze
python3 /Users/mahdi/Projects/git/PhD-Project/Pumping/scripts/analyze_deformation_sweep.py
