#!/bin/bash
# ============================================================================
# Rosensweig Production Test Matrix — 9 jobs (3 mesh × 3 dt)
#
# All: no clamping, -λ/ε capillary, sharp IC, t_final=2.0
# VTK output every 0.05 time units for direct frame comparison
#
# SET THESE after scaling tests:
NP=16              # <-- optimal core count from scaling test
SOLVER_FLAG="--direct"  # <-- or "" for iterative
# ============================================================================

set -e
cd "$(dirname "$0")/.."

# Mesh levels and timesteps
LEVELS=(5 6 7)
DTS=("2e-3" "5e-4" "2e-4")
STEPS=(1000 4000 10000)
VTK_INT=(25 100 250)

# Wall time estimates (generous)
WALLTIMES=("2-00:00:00" "7-00:00:00" "14-00:00:00")

for i in "${!LEVELS[@]}"; do
    L=${LEVELS[$i]}
    for j in "${!DTS[@]}"; do
        DT=${DTS[$j]}
        NSTEPS=${STEPS[$j]}
        VTKI=${VTK_INT[$j]}
        WTIME=${WALLTIMES[$j]}

        RUN_NAME="rosen_L${L}_dt${DT}"

        echo "Submitting: ${RUN_NAME} (L${L}, dt=${DT}, ${NSTEPS} steps, vtk every ${VTKI})"

        sbatch \
            --job-name="${RUN_NAME}" \
            --partition=hpcc \
            --ntasks=${NP} \
            --time=${WTIME} \
            --output="Results/${RUN_NAME}.out" \
            --error="Results/${RUN_NAME}.err" \
            --wrap="mpirun -np ${NP} ./cmake-build-release/ferrofluid \
                --rosensweig \
                --run_name ${RUN_NAME} \
                --refinement ${L} \
                --amr \
                ${SOLVER_FLAG} \
                --max_steps ${NSTEPS} \
                --dt ${DT} \
                --vtk_interval ${VTKI} \
                > Results/${RUN_NAME}.log 2>&1"
    done
done

echo ""
echo "All 9 production jobs submitted. Check with: squeue -u \$USER"
echo ""
echo "Results comparison at identical frames: t = 0.05, 0.10, ..., 2.00"
