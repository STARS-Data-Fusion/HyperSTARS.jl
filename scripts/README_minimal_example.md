# Minimal Working Example - HyperSTARS Data Fusion

This is a reduced-complexity version of the Kings Canyon HLS+EMIT fusion demo for testing and development.

## Key Reductions

Compared to [kings_canyon_hls_emit_local.jl](kings_canyon_hls_emit_local.jl):

| Parameter | Full Script | Minimal Example | Impact |
|-----------|-------------|-----------------|--------|
| **Date range** | 31 days (Aug 2022) | 3 days | ~10x faster |
| **Window size (scf)** | 4×4 pixels | 16×16 pixels | ~16x fewer windows |
| **Sampling points (nsamp)** | 50 | 20 | ~2.5x faster per window |
| **Window buffer** | 4 pixels | 2 pixels | Smaller spatial context |
| **Target times** | 1:2 | 1:2 | Same (already minimal) |

**Estimated speedup:** ~40-100× faster than full script

## Memory Profiling

The script tracks memory usage at each stage using `Sys.maxrss()`:

```julia
Memory breakdown:
  Start:        145.2    MB
  After load:   892.7    MB  (+747.5 MB)
  After PCA:    905.1    MB  (+12.4 MB)
  After setup:  915.3    MB  (+10.2 MB)
  After fusion: 1247.8   MB  (+332.5 MB)
  PEAK:         1247.8   MB
```

This shows:
- **Data loading** is the largest memory cost (~750 MB for 3 days)
- **Fusion** adds ~330 MB during parallel processing
- **Peak memory** scales linearly with date range and inversely with window size

## Usage

### Basic Run

```bash
julia --project=. scripts/minimal_example.jl
```

### With More Workers

```bash
# Edit line 11 in minimal_example.jl to increase workers:
addprocs(8)  # Instead of addprocs(4)
```

### Customize Parameters

Edit the configuration section (lines 240-260):

```julia
# Reduce even further (1 day, max window size)
date_range = [Date("2022-08-01"), Date("2022-08-01")]
scf = 32  # Even larger windows

# Or scale up (5 days, smaller windows)
date_range = [Date("2022-08-01"), Date("2022-08-05")]
scf = 8
```

## Expected Runtime

On a typical laptop (M1 MacBook with 4 workers):

- **Minimal example (3 days, scf=16):** ~5-10 minutes
- **Medium example (7 days, scf=8):** ~30-60 minutes  
- **Full example (31 days, scf=4):** ~4-8 hours

## Memory Requirements

Approximate peak memory usage by configuration:

| Config | Date Range | scf | Peak Memory |
|--------|------------|-----|-------------|
| Minimal | 3 days | 16 | ~1.2 GB |
| Medium | 7 days | 8 | ~2.5 GB |
| Full | 31 days | 4 | ~8-12 GB |

## Output

The script produces:
- `fused_images`: (y, x, wavelength, time) array of fused hyperspectral data
- `fused_sd_images`: (y, x, wavelength, time) array of uncertainty estimates

These are not saved by default. To save output, add at the end:

```julia
using JLD2
@save "fusion_output_minimal.jld2" fused_images fused_sd_images
```

## Scaling Up

To run the full 1-month example with optimal performance:

1. **Use the production script:** [kings_canyon_hls_emit.jl](kings_canyon_hls_emit.jl) (designed for HPC)
2. **Increase workers:** `addprocs(16)` or more on HPC
3. **Adjust parameters:**
   - `scf = 4` (smaller windows, better spatial resolution)
   - `nsamp = 50` (more sampling points, better accuracy)
   - `window_buffer = 4` (larger spatial context)

## Troubleshooting

### Out of Memory

Reduce memory footprint:
```julia
scf = 32           # Larger windows
target_times = 1:1 # Single time point only
```

### Too Slow

Speed up computation:
```julia
addprocs(8)        # More workers
nsamp = 10         # Fewer sampling points
```

### No Data Found

Ensure data directories exist and contain data for the date range:
```bash
~/data/
  ├── Kings_Canyon_HLS/
  │   ├── L30/
  │   └── S30/
  ├── Kings_Canyon_EMIT/
  ├── HLS_L30_srf.csv
  ├── HLS_S30_srf.csv
  └── EMIT_metadata.csv
```

## See Also

- [Full script documentation](../../README.md)
- [Setup instructions](../../README.md#setup)
- [Main script](kings_canyon_hls_emit_local.jl)
