#!/bin/bash
# ============================================================================
# setup_runs.sh - Set up and submit ferrofluid production runs
#
# Usage: ./setup_runs.sh [setup|submit|status|all]
# ============================================================================

SCRATCH_DIR="/share/ceph/scratch/mg6f4/dealii_runs"
SUB_DIR="$(dirname "$0")"

# List of all runs
RUNS=(
    "rosen_r5"
    "rosen_r5_amr"
    "hedge_r4"
    "hedge_r5"
    "hedge_r4_amr"
    "dome_r4"
    "dome_r4_amr"
    "dome_r5"
)

setup_directories() {
    echo "=== Setting up run directories ==="
    for run in "${RUNS[@]}"; do
        dir="$SCRATCH_DIR/$run"
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created: $dir"
        else
            echo "Exists:  $dir"
        fi
    done
    echo ""
}

submit_jobs() {
    echo "=== Submitting jobs ==="
    cd "$SUB_DIR"
    for run in "${RUNS[@]}"; do
        sub_file="${run}.sub"
        if [ -f "$sub_file" ]; then
            echo -n "Submitting $run... "
            sbatch "$sub_file"
        else
            echo "WARNING: $sub_file not found!"
        fi
    done
    echo ""
}

check_status() {
    echo "=== Job Status ==="
    squeue -u $USER --format="%.10i %.12j %.8T %.12M %.6D %R"
    echo ""
    echo "=== Recent completed jobs ==="
    sacct -u $USER --starttime=$(date -d "3 days ago" +%Y-%m-%d) \
          --format=JobID,JobName%15,State,Elapsed,ExitCode | head -30
}

show_help() {
    echo "Ferrofluid Production Runs Setup"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup   - Create run directories in scratch"
    echo "  submit  - Submit all jobs to SLURM"
    echo "  status  - Check job status"
    echo "  all     - Setup + submit"
    echo ""
    echo "Runs to be submitted:"
    echo "  Paper-matching (fixed dt, no adaptive):"
    for run in "${RUNS[@]}"; do
        echo "    - $run"
    done
    echo ""
    echo "Paper parameters used:"
    echo "  Rosensweig: dt=5e-4, t_final=2.0, 4000 steps"
    echo "  Hedgehog:   dt=1e-3, t_final=6.0, 6000 steps"
    echo "  Dome:       dt=1e-3, t_final=6.0, 6000 steps"
}

case "${1:-help}" in
    setup)
        setup_directories
        ;;
    submit)
        submit_jobs
        ;;
    status)
        check_status
        ;;
    all)
        setup_directories
        submit_jobs
        check_status
        ;;
    *)
        show_help
        ;;
esac