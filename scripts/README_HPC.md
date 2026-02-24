# Running HyperSTARS on HPC with SLURM

This directory contains SLURM job scripts for running HyperSTARS data fusion on HPC clusters.

## Available Job Scripts

### 1. Minimal Example (`submit_minimal_example.slurm`)

**Purpose:** Quick test/validation run with reduced temporal extent.

**Configuration:**
- Date range: 5 days (Aug 13-17, 2022)
- Spatial extent: Full scene (1411×2085 pixels)
- Memory: 32 GB
- Runtime: 90 minutes
- CPUs: 8

**When to use:**
- Testing the pipeline on new data
- Validating code changes
- Quick turnaround for results (~1 hour)

**Submit with:**
```bash
sbatch scripts/submit_minimal_example.slurm
```

### 2. Full Production (`submit_full_production.slurm`)

**Purpose:** Complete 31-day analysis for publication/operations.

**Configuration:**
- Date range: 31 days (August 2022)
- Spatial extent: Full scene (1411×2085 pixels)
- Memory: 80 GB
- Runtime: 4 hours
- CPUs: 16

**When to use:**
- Production runs for research
- Complete temporal coverage needed
- Final results for publication

**Submit with:**
```bash
sbatch scripts/submit_full_production.slurm
```

## Setup Instructions

### 1. Check Your HPC Julia Module

Find the correct Julia module name on your system:
```bash
module avail julia
```

Edit the SLURM scripts to uncomment and update the module load line:
```bash
# Change this line:
# module load julia/1.11

# To match your system, e.g.:
module load julia/1.10.0
```

Or if you have a custom Julia installation:
```bash
export PATH=/custom/path/to/julia/bin:$PATH
```

### 2. Verify Data Paths

Ensure your data is accessible from compute nodes:
```bash
# Data should be at:
~/data/Kings_Canyon_HLS/
~/data/Kings_Canyon_EMIT/
~/data/HLS_L30_srf.csv
~/data/HLS_S30_srf.csv
~/data/EMIT_metadata.csv
```

Or update the `dir_path` variable in the Julia scripts to match your data location.

### 3. Activate Julia Environment

Before submitting, ensure the Julia environment is set up:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Submitting Jobs

### Basic Submission
```bash
# Minimal example (5 days)
sbatch scripts/submit_minimal_example.slurm

# Full production (31 days)
sbatch scripts/submit_full_production.slurm
```

### Check Job Status
```bash
# View queue
squeue -u $USER

# Watch specific job
watch -n 5 squeue -j JOBID

# View output (while running)
tail -f hyperstars_minimal_JOBID.out
```

### Cancel Job
```bash
scancel JOBID
```

## Monitoring Resource Usage

### While Running
```bash
# SSH to compute node (if allowed)
srun --jobid=JOBID --pty bash
top -u $USER

# Or check from login node
sstat -j JOBID --format=JobID,MaxRSS,AveCPU
```

### After Completion
```bash
# Memory and time used
sacct -j JOBID --format=JobID,JobName,MaxRSS,Elapsed,State

# Detailed efficiency report
seff JOBID
```

## Customizing Resource Requests

### For Different Date Ranges

Memory scales approximately linearly with days:
```
Memory ≈ 1.6 GB × n_days + 10.5 GB (fusion overhead)

Examples:
  5 days  → 21 GB  (request 32 GB)
  10 days → 26 GB  (request 40 GB)
  31 days → 60 GB  (request 80 GB)
  60 days → 106 GB (request 128 GB)
```

Update the SLURM script accordingly:
```bash
#SBATCH --mem=128G  # For 60-day runs
#SBATCH --time=6:00:00  # Adjust proportionally
```

### For Spatial Subsets

If using spatial cropping (see MEMORY_ARCHITECTURE.md):
```bash
# 256×256 pixel subset
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
```

### Adjusting Workers

To use more/fewer parallel workers, edit the Julia script:
```julia
# In minimal_example.jl or kings_canyon_hls_emit_local.jl
addprocs(16)  # Change from 4 to 16
```

Then update SLURM CPUs to match:
```bash
#SBATCH --cpus-per-task=20  # workers + overhead
```

## Troubleshooting

### Out of Memory Error
```
Solution: Increase --mem or reduce date range
Check: sacct -j JOBID --format=MaxRSS
```

### Timeout
```
Solution: Increase --time
Check: Elapsed time with seff JOBID
```

### Module Not Found
```
Solution: Update module load command or use custom Julia path
Check: module avail julia
```

### Data Not Found
```
Solution: Verify data paths are accessible from compute nodes
Check: srun --jobid=JOBID ls ~/data/
```

## Output Files

After successful completion:
- Standard output: `hyperstars_minimal_JOBID.out`
- Standard error: `hyperstars_minimal_JOBID.err`
- Results: (location depends on script configuration)

## Best Practices

1. **Test with minimal example first** before submitting expensive full runs
2. **Monitor resource usage** to optimize future requests
3. **Use job arrays** for processing multiple scenes
4. **Save intermediate results** to enable recovery from failures
5. **Document your exact configuration** in job script comments

## Memory and Runtime Reference

Based on empirical measurements (Kings Canyon test case):

| Configuration | Memory | Runtime | CPUs | Recommended Request |
|---------------|--------|---------|------|---------------------|
| 5 days (minimal) | 21 GB | 49 min | 5 | 32 GB, 90 min, 8 CPUs |
| 31 days (full) | ~60 GB | ~3 hrs | 5-16 | 80 GB, 4 hrs, 16 CPUs |
| 256×256 spatial subset | ~15 GB | ~10 min | 5 | 24 GB, 30 min, 8 CPUs |

See [MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md) for detailed memory scaling analysis.

## Additional Resources

- HyperSTARS.jl documentation: [README.md](../README.md)
- Memory architecture: [MEMORY_ARCHITECTURE.md](../MEMORY_ARCHITECTURE.md)
- Example scripts: [examples/](../examples/)

## Support

For issues specific to:
- **HyperSTARS algorithm:** Open issue on GitHub
- **HPC/SLURM configuration:** Contact your HPC support team
- **Julia environment:** Check Julia Discourse or Stack Overflow
