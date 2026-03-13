#!/bin/bash
# ============================================================================
# Hedgehog Test Matrix — 12 simultaneous runs
# Date: 2026-03-12
#
# BIN-0 = current code (no Kelvin fix)
# BIN-1 = with Kelvin fix (h_a + ∇h_a in NS Kelvin force)
#
# Usage: bash launch_hedgehog_tests.sh [NP]
#   NP = number of MPI ranks (default: 4)
#
# All runs use nohup and log to individual files.
# Results go to Results/ with auto-generated timestamped folders.
# ============================================================================

NP=${1:-4}
BIN0="./build/bin0_hedgehog"
BIN1="./build/bin1_hedgehog"

echo "Launching 12 hedgehog tests with $NP MPI ranks each..."
echo ""

# --- BIN-0 tests (current code, no Kelvin fix) ---

echo "T1:  BIN-0 baseline"
nohup mpirun -np $NP $BIN0 --hedgehog > logs/T01_bin0_baseline.log 2>&1 &
echo "  PID: $!"

echo "T2:  BIN-0 dt=1e-4"
nohup mpirun -np $NP $BIN0 --hedgehog --dt 1e-4 > logs/T02_bin0_dt1e-4.log 2>&1 &
echo "  PID: $!"

echo "T3:  BIN-0 dt=5e-5"
nohup mpirun -np $NP $BIN0 --hedgehog --dt 5e-5 > logs/T03_bin0_dt5e-5.log 2>&1 &
echo "  PID: $!"

echo "T4:  BIN-0 chi0=0.5"
nohup mpirun -np $NP $BIN0 --hedgehog --chi0 0.5 > logs/T04_bin0_chi0.5.log 2>&1 &
echo "  PID: $!"

echo "T5:  BIN-0 mesh=150x90"
nohup mpirun -np $NP $BIN0 --hedgehog --mesh 150x90 > logs/T05_bin0_mesh150x90.log 2>&1 &
echo "  PID: $!"

echo "T6:  BIN-0 ramp-slope=0.6"
nohup mpirun -np $NP $BIN0 --hedgehog --ramp-slope 0.6 > logs/T06_bin0_slope0.6.log 2>&1 &
echo "  PID: $!"

# --- BIN-1 tests (with Kelvin fix) ---

echo "T7:  BIN-1 Kelvin fix baseline"
nohup mpirun -np $NP $BIN1 --hedgehog > logs/T07_bin1_baseline.log 2>&1 &
echo "  PID: $!"

echo "T8:  BIN-1 Kelvin fix + dt=1e-4"
nohup mpirun -np $NP $BIN1 --hedgehog --dt 1e-4 > logs/T08_bin1_dt1e-4.log 2>&1 &
echo "  PID: $!"

echo "T9:  BIN-1 Kelvin fix + chi0=0.5"
nohup mpirun -np $NP $BIN1 --hedgehog --chi0 0.5 > logs/T09_bin1_chi0.5.log 2>&1 &
echo "  PID: $!"

echo "T10: BIN-1 Kelvin fix + mesh=150x90"
nohup mpirun -np $NP $BIN1 --hedgehog --mesh 150x90 > logs/T10_bin1_mesh150x90.log 2>&1 &
echo "  PID: $!"

echo "T11: BIN-1 Kelvin fix + dt=1e-4 + mesh=150x90"
nohup mpirun -np $NP $BIN1 --hedgehog --dt 1e-4 --mesh 150x90 > logs/T11_bin1_dt1e-4_mesh150x90.log 2>&1 &
echo "  PID: $!"

echo "T12: BIN-1 Kelvin fix + ramp-slope=0.6"
nohup mpirun -np $NP $BIN1 --hedgehog --ramp-slope 0.6 > logs/T12_bin1_slope0.6.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All 12 tests launched. Monitor with:"
echo "  tail -f logs/T*.log"
echo "  ls -lt Results/ | head -15"
