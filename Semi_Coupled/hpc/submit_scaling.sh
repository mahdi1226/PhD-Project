#!/bin/bash
# ============================================================================
# Rosensweig Scaling Tests — submit all 8 jobs
# All: L5, dt=1e-3, 2000 steps, t_final=2.0, vtk every 0.05 (interval=50)
# ============================================================================

set -e
cd "$(dirname "$0")/.."

CORES=(8 16 24 32)
SOLVERS=(direct iterative)

for SOL in "${SOLVERS[@]}"; do
    for NP in "${CORES[@]}"; do
        RUN_NAME="scale_${SOL}_np${NP}"

        if [ "$SOL" = "direct" ]; then
            SOLVER_FLAG="--direct"
        else
            SOLVER_FLAG=""
        fi

        echo "Submitting: ${RUN_NAME} (${NP} ranks, ${SOL})"

        sbatch \
            --job-name="${RUN_NAME}" \
            --partition=hpcc \
            --ntasks=${NP} \
            --time=2-00:00:00 \
            --output="Results/${RUN_NAME}.out" \
            --error="Results/${RUN_NAME}.err" \
            --wrap="mpirun -np ${NP} ./cmake-build-release/ferrofluid \
                --rosensweig \
                --run_name ${RUN_NAME} \
                --refinement 5 \
                --amr \
                ${SOLVER_FLAG} \
                --max_steps 2000 \
                --dt 1e-3 \
                --vtk_interval 50 \
                > Results/${RUN_NAME}.log 2>&1"
    done
done

echo ""
echo "All 8 scaling jobs submitted. Check with: squeue -u \$USER"
