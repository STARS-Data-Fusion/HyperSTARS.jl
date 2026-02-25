#!/bin/bash
#SBATCH --job-name=hyperstars_minimal
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=hyperstars_minimal_%j.out
#SBATCH --error=hyperstars_minimal_%j.err

# HyperSTARS Minimal Example Job Script
# Based on empirical measurements:
#   - Peak memory: 21 GB (requesting 32 GB for safety)
#   - Runtime: 49 minutes (requesting 90 minutes)
#   - Workers: 4 (requesting 8 CPUs)
#
# Configuration: 5 days (Aug 13-17, 2022), full 1411×2085 extent
# Expected output: ~20 GB memory, ~50 minutes runtime

echo "=========================================="
echo "HyperSTARS Minimal Example"
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

# Set number of threads (optional, for threading within Julia)
export JULIA_NUM_THREADS=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo ""
echo "Working directory: $(pwd)"
echo ""

# Generate required CSV files if they don't exist
DATA_DIR="/gpfs/scratch/refl-datafusion-trtd"
META_DIR="/gpfs/scratch/refl-datafusion-trtd/$USER/hyperstars_metadata"
export HYPERSTARS_DATA_DIR="$DATA_DIR"
export HYPERSTARS_METADATA_DIR="$META_DIR"
mkdir -p "$META_DIR"

echo "Checking for required data files..."
if [ ! -f "$META_DIR/EMIT_metadata.csv" ] || [ ! -f "$META_DIR/HLS_L30_srf.csv" ] || [ ! -f "$META_DIR/HLS_S30_srf.csv" ]; then
    echo "Generating missing CSV files..."
    julia --project=. scripts/generate_srf_files.jl
    julia --project=. scripts/generate_emit_metadata.jl
    echo "CSV files generated successfully"
else
    echo "All required CSV files found"
fi

echo ""
echo "Running minimal_example.jl..."
echo ""

# Run the Julia script
julia --project=. scripts/minimal_example.jl

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
