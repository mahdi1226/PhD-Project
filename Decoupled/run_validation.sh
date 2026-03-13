#!/bin/bash
# =============================================================================
# run_validation.sh — Square relaxation + Droplet elongation (back-to-back)
#
# Usage: caffeinate -is bash run_validation.sh
# Runs from: Decoupled/ (project root)
# =============================================================================
set -e

NP=4
BUILD=build
EXE="$BUILD/drivers/ferrofluid_decoupled"
TS=$(date +%Y%m%d_%H%M%S)

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

echo "============================================================"
echo "  VALIDATION TEST SUITE"
echo "  Date:    $(date)"
echo "  MPI:     $NP ranks"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# 1. Square relaxation (pure CH+NS, no field, AMR)
#    Domain 2π×2π, r=6, dt=2e-3, 5000 steps
# ---------------------------------------------------------------
echo "============================================================"
echo "  [1/2] SQUARE RELAXATION (AMR)"
echo "============================================================"
T0=$(date +%s)

mpirun -np $NP $EXE --validation square --amr --amr-interval 5 2>&1 | tee "square_${TS}.log"

DT=$(( $(date +%s) - T0 ))
echo ""
echo -e "${GREEN}[DONE]${NC} Square relaxation finished in ${DT}s"
echo ""

# ---------------------------------------------------------------
# 2. Droplet elongation (CH+NS+Mag+Poisson, dipole field, AMR)
#    Domain 1×1, r=7, dt=1e-3, 1500 steps
# ---------------------------------------------------------------
echo "============================================================"
echo "  [2/2] DROPLET ELONGATION (AMR)"
echo "============================================================"
T0=$(date +%s)

mpirun -np $NP $EXE --validation elongation --amr --amr-interval 5 2>&1 | tee "elongation_${TS}.log"

DT=$(( $(date +%s) - T0 ))
echo ""
echo -e "${GREEN}[DONE]${NC} Elongation test finished in ${DT}s"
echo ""

echo "============================================================"
echo "  ALL VALIDATION TESTS COMPLETE at $(date)"
echo "  Results in: Results/"
echo "  Logs: square_${TS}.log, elongation_${TS}.log"
echo "============================================================"
