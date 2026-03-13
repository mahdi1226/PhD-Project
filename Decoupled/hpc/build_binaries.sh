#!/bin/bash
# ============================================================================
# Build both hedgehog binaries on a compute node
#
# Usage (from interactive session on compute node):
#   cd ~/PhD-Project/Decoupled
#   bash hpc/build_binaries.sh
#
# Prerequisites:
#   - Running on a compute node (salloc first!)
#   - module load dealii/9.7.1_umfpack_mumps already done
# ============================================================================

set -e  # exit on error

echo "=== Building Hedgehog Binaries ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

# Ensure we're in Decoupled directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "ERROR: Run this from ~/PhD-Project/Decoupled/"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure
echo "--- Configuring CMake (Release) ---"
cmake -DCMAKE_BUILD_TYPE=Release ..

# Step 1: Build BIN-1 (current code = Kelvin fix included)
echo ""
echo "--- Building BIN-1 (with Kelvin fix) ---"
make -j16
cp drivers/ferrofluid_decoupled bin1_hedgehog
echo "BIN-1 built: build/bin1_hedgehog"

# Step 2: Revert Kelvin fix for BIN-0
echo ""
echo "--- Reverting Kelvin fix for BIN-0 ---"
cd ..
git checkout -- navier_stokes/navier_stokes_assemble.cc
cd build

# Rebuild without Kelvin fix
echo "--- Building BIN-0 (no Kelvin fix) ---"
make -j16
cp drivers/ferrofluid_decoupled bin0_hedgehog
echo "BIN-0 built: build/bin0_hedgehog"

# Step 3: Restore Kelvin fix (leave code in BIN-1 state)
cd ..
git checkout main -- navier_stokes/navier_stokes_assemble.cc
cd build

# Verify
echo ""
echo "=== Build Complete ==="
ls -lh bin0_hedgehog bin1_hedgehog
echo ""
echo "Next: exit interactive session, then run:"
echo "  cd ~/PhD-Project/Decoupled && sbatch hpc/hedgehog_array.sub"
