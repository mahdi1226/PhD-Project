#!/bin/bash
# Validation suite: Square → Elongation → Rosensweig
# L4, AMR, 4 MPI ranks, dt=5e-4, t_final=2.0
# Run with: nohup caffeinate -i bash run_validation_suite.sh &

set -e
cd "$(dirname "$0")"

EXE=./cmake-build-release/ferrofluid
NP=4
COMMON="--refinement 4 --amr --dt 5e-4 --t_final 2.0"

echo "=== Validation Suite Started: $(date) ==="

echo ""
echo ">>> [1/3] Square (no magnetic, no gravity) — $(date)"
mpirun -np $NP $EXE --square $COMMON 2>&1 | tee square_L4_amr.log
echo "<<< Square finished: $(date)"

echo ""
echo ">>> [2/3] Elongation — $(date)"
mpirun -np $NP $EXE --elongation $COMMON 2>&1 | tee elongation_L4_amr.log
echo "<<< Elongation finished: $(date)"

echo ""
echo ">>> [3/3] Rosensweig — $(date)"
mpirun -np $NP $EXE --rosensweig $COMMON 2>&1 | tee rosensweig_L4_amr.log
echo "<<< Rosensweig finished: $(date)"

echo ""
echo "=== Validation Suite Complete: $(date) ==="
