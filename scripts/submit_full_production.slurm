#!/bin/bash
#SBATCH --job-name=hyperstars_full
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=hyperstars_full_%j.out
#SBATCH --error=hyperstars_full_%j.err

# HyperSTARS Full Production Job Script
# Based on extrapolated measurements:
#   - Expected peak memory: ~60 GB (requesting 80 GB for safety)
#   - Expected runtime: ~3 hours (requesting 4 hours)
#   - Workers: configurable (requesting 16 CPUs)
#
# Configuration: 31 days (August 2022), full 1411×2085 extent
# Note: Adjust memory/time based on your specific date range

echo "=========================================="
echo "HyperSTARS Full Production Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Load Julia module (adjust module name for your HPC system)
# Uncomment and modify as needed:
# module load julia/1.11
# module load julia/1.10
# Or use a custom Julia installation:
# export PATH=/path/to/julia/bin:$PATH

# Verify Julia is available
which julia
julia --version

# Set number of threads (optional)
export JULIA_NUM_THREADS=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo ""
echo "Working directory: $(pwd)"
echo "Running kings_canyon_hls_emit_local.jl..."
echo "WARNING: This is the full 31-day production run"
echo ""

# Run the Julia script
julia --project=. scripts/kings_canyon_hls_emit_local.jl

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
