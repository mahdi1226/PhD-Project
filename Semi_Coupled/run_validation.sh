#!/bin/bash
# ============================================================================
# Semi_Coupled validation suite: square → droplet → elongation
# Run from project root: Semi_Coupled/
#
# Usage:
#   caffeinate -dims ./run_validation.sh 2>&1 | tee validation.log
# ============================================================================

set -e

BIN="./cmake-build-release/ferrofluid"
NP=6
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " Semi_Coupled Validation Suite"
echo " Started: $(date)"
echo " Binary:  $BIN"
echo " Ranks:   $NP"
echo "============================================"
echo ""

# Verify binary exists
if [ ! -x "$BIN" ]; then
    echo "ERROR: Binary not found at $BIN"
    echo "       Run: cd cmake-build-release && make -j"
    exit 1
fi

# Verify we're in the right directory (Results/ will be created here)
if [ ! -d "cmake-build-release" ]; then
    echo "ERROR: Run this script from Semi_Coupled/ project root"
    exit 1
fi

# ------------------------------------------------------------------
# T1: Square relaxation (CH + NS, no magnetic)
# Validates: surface tension, phase field dynamics
# Expected: square → circle, energy monotone decreasing
# ~500 steps, ~10-15 min
# ------------------------------------------------------------------
echo "============================================"
echo " [1/3] Square Relaxation"
echo "       Preset: --square"
echo "       Steps:  500, dt=0.002, t_final=1.0"
echo "       Start:  $(date)"
echo "============================================"

mpirun -np $NP $BIN --square --run_name val-square-${TIMESTAMP}

echo ""
echo " [1/3] Square: DONE at $(date)"
echo ""

# ------------------------------------------------------------------
# T2: Droplet no-field (CH + NS, no magnetic)
# Validates: circular droplet relaxation, NS coupling
# Expected: circle stays circular, energy decreases
# ~1000 steps, ~20-30 min
# ------------------------------------------------------------------
echo "============================================"
echo " [2/3] Droplet (no field)"
echo "       Preset: --droplet"
echo "       Steps:  1000, dt=0.001, t_final=1.0"
echo "       Start:  $(date)"
echo "============================================"

mpirun -np $NP $BIN --droplet --run_name val-droplet-${TIMESTAMP}

echo ""
echo " [2/3] Droplet: DONE at $(date)"
echo ""

# ------------------------------------------------------------------
# T3: Elongation (CH + NS + magnetic)
# Validates: Kelvin force coupling, monolithic M+phi
# Expected: droplet elongates vertically under h_a=(0,45)
# R=0.1, r=7 (16K cells), ~1500 steps, ~2-4h
# ------------------------------------------------------------------
echo "============================================"
echo " [3/3] Elongation (droplet + magnetic field)"
echo "       Preset: --elongation"
echo "       Steps:  1500, dt=0.001, t_final=1.5"
echo "       R=0.1, h=1/128, Bm=20.25"
echo "       Start:  $(date)"
echo "============================================"

mpirun -np $NP $BIN --elongation --run_name val-elongation-${TIMESTAMP}

echo ""
echo " [3/3] Elongation: DONE at $(date)"
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "============================================"
echo " ALL 3 TESTS COMPLETE"
echo " Finished: $(date)"
echo ""
echo " Results:"
echo "   Results/val-square-${TIMESTAMP}/"
echo "   Results/val-droplet-${TIMESTAMP}/"
echo "   Results/val-elongation-${TIMESTAMP}/"
echo ""
echo " Check energy.csv in each for energy monotonicity."
echo "============================================"
