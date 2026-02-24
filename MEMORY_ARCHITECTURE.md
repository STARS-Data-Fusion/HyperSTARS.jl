# Memory Architecture and Limitations in HyperSTARS.jl

## Executive Summary

The current HyperSTARS data fusion pipeline uses an **eager loading strategy** that loads all temporal data into memory at once. This design enables flexible Kalman filtering across the full time series but creates memory constraints that scale linearly with the temporal extent of the dataset.

**Key Finding:** For a full-resolution Kings Canyon scene (1411×2085 pixels):
- 5-day time series: ~21 GB peak memory
- 31-day time series: ~60 GB peak memory  
- Memory scales at ~1.6 GB per day for data loading

## Current Architecture

### Data Loading Strategy

The pipeline loads data in three stages:

1. **Lazy Initialization** (Good ✅)
   ```julia
   band_rasters = [Raster(x, lazy=true) for x in band_files]
   ```
   Rasters.jl creates lazy disk-backed arrays without loading data.

2. **Immediate Materialization** (Bottleneck ⚠️)
   ```julia
   band_arrays = [Array(r) for r in band_rasters]
   ```
   All rasters are immediately converted to in-memory arrays.

3. **Pre-allocation of Full Temporal Tensor** (Memory-intensive ❌)
   ```julia
   hls_array = zeros(Float32, ny, nx, nbands, ntime)
   for (bi, band_arrays) in enumerate(band_arrays_list)
       for (ti, arr) in enumerate(band_arrays)
           # Copy all time steps into single 4D array
       end
   end
   ```

### Memory Scaling Behavior

Based on empirical measurements (Kings Canyon test case):

| Component | Memory Requirement | Scaling Factor |
|-----------|-------------------|----------------|
| **Data Loading** | 1.6 GB × n_days | Linear with temporal extent |
| **Fusion Processing** | ~10.5 GB | Constant (spatial extent dependent) |
| **Total Peak** | Data + Fusion | Sub-linear overall |

**Examples:**
- 5 days: 8 GB (data) + 10.5 GB (fusion) = 21 GB peak
- 31 days: 50 GB (data) + 10.5 GB (fusion) = 60 GB peak
- 365 days: 584 GB (data) + 10.5 GB (fusion) = 595 GB peak

### Why This Design Exists

The current architecture requires full temporal access because:

1. **Kalman Filter Requirements**
   - Random access to different time steps for state propagation
   - Temporal covariance calculations across non-contiguous dates
   - Backward smoothing requires reverse temporal traversal

2. **Flexible Time Selection**
   - Users can specify arbitrary `target_times` for output
   - Algorithm can interpolate/extrapolate to any date in range
   - No assumption of sequential processing

3. **Spatial Windowing with Temporal Context**
   - Each spatial window processes its full time series independently
   - Parallel processing across windows requires independent temporal access
   - No assumptions about temporal locality

## Limitations

### Current Constraints

1. **Maximum Temporal Extent**
   - Limited by available RAM: ~365 days requires >500 GB for full scenes
   - HPC systems: typically 128-256 GB per node → ~80-160 day maximum
   - Desktop systems: 16-64 GB → ~10-40 day maximum

2. **Spatial vs Temporal Trade-offs**
   - Cannot process arbitrarily large spatial extents AND long time series
   - Must choose: full scene (short time) OR time series (small spatial subset)

3. **No Incremental/Streaming Capability**
   - Cannot process data in temporal chunks
   - Cannot emit partial results during processing
   - All-or-nothing memory commitment

### Workarounds (Current)

1. **Spatial Cropping**
   ```
   256×256 subset: ~16× less memory than full scene
   Effect: 21 GB → 1.3 GB for same temporal extent
   ```

2. **Temporal Chunking (Manual)**
   ```
   Process 30-day windows separately
   Manual stitching of outputs required
   Discontinuities at chunk boundaries
   ```

3. **Reduced Spatial Resolution**
   ```
   Downsampling from 30m to 60m/90m
   4-9× memory reduction
   Loss of spatial detail
   ```

## Future Work Required

To overcome these limitations, several architectural changes would be needed:

### 1. Streaming Temporal Processing Architecture

**Objective:** Process time series incrementally without loading all dates.

**Requirements:**
- **Sequential Kalman Filter Formulation**
  - Forward-only state propagation (no backward smoothing)
  - One-pass algorithm design
  - State persistence between time steps

- **Lazy Temporal Slicing**
  ```julia
  # Instead of: band_arrays = [Array(r) for r in rasters]
  # Use: Lazy temporal iterator that loads on-demand
  for t in temporal_iterator(rasters)
      data_t = Array(t)  # Load only current time step
      process_spatial_windows(data_t)
      # Discard after processing
  end
  ```

- **Chunked Output Writing**
  - Write fused results to disk incrementally
  - NetCDF/Zarr formats with append capability
  - Memory-mapped outputs

**Challenges:**
- ⚠️ Loses ability to do backward smoothing (quality degradation)
- ⚠️ Can only output time steps as they're processed (no random access)
- ⚠️ More complex state management across chunks

### 2. Hybrid Lazy/Eager Loading Strategy

**Objective:** Keep spatial data lazy, materialize temporal on-demand.

**Approach:**
```julia
# Keep rasters lazy-backed
raster_cube = RasterStack(files, lazy=true)

# Process windows spatially, load temporal slices per window
function process_window(window_coords, raster_cube)
    # Extract spatial window (still lazy)
    window_data = raster_cube[window_coords..., :, :]
    
    # Materialize only this window's time series
    window_temporal = Array(window_data)  # Much smaller!
    
    # Process and return result
    fuse_window_timeseries(window_temporal)
end
```

**Benefits:**
- ✓ Memory scales with window size, not full scene
- ✓ Retains full temporal access within windows
- ✓ Parallelizes cleanly across windows

**Challenges:**
- Requires careful coordination of lazy operations
- Disk I/O becomes bottleneck (must read same data multiple times)
- May need caching strategy for overlapping windows

### 3. Distributed Temporal Chunking

**Objective:** Distribute long time series across multiple compute nodes.

**Architecture:**
```
Time Series: [Day 1-30] [Day 31-60] [Day 61-90] ...
             ↓          ↓           ↓
          Node 1      Node 2      Node 3
             ↓          ↓           ↓
          Results 1  Results 2  Results 3
             ↓          ↓           ↓
          Merge/Stitch Results
```

**Requirements:**
- Temporal overlap between chunks for continuity
- Blending/stitching algorithm for chunk boundaries
- Distributed state management (MPI/Dagger.jl)

**Challenges:**
- ⚠️ Complex coordination and communication
- ⚠️ Boundary artifacts from chunking
- ⚠️ Requires distributed computing infrastructure

### 4. Spectral Dimension Reduction (Preprocessing)

**Objective:** Reduce hyperspectral data dimensionality before fusion.

**Approach:**
- Apply PCA/dimensionality reduction to EMIT data *before* loading
- Store reduced representation (3-10 components vs 212 bands)
- Fusion operates on reduced space, reconstruct full spectrum at end

**Memory Impact:**
```
EMIT: 1411 × 2085 × 212 × 31 → 1411 × 2085 × 5 × 31
Reduction: 42× less memory for EMIT component
```

**Challenges:**
- Information loss in dimensionality reduction
- Two-stage processing pipeline
- Optimal component count unclear

### 5. Out-of-Core Computation Framework

**Objective:** Use disk-backed arrays that appear in-memory but page to disk.

**Candidate Libraries:**
- **Dask.jl** (if/when ported from Python)
- **JuliaDB.jl** with memory-mapped arrays
- **Zarr.jl** with chunked on-disk storage

**Approach:**
```julia
# Use memory-mapped or chunked storage
data = Zarr.zarray("data.zarr", (ny, nx, nbands, ntime))

# Operations transparently page data in/out
result = fusion_operations(data)  # Handles memory management
```

**Benefits:**
- ✓ Abstracts memory management away from algorithm
- ✓ Can process datasets larger than RAM
- ✓ Minimal code changes

**Challenges:**
- ⚠️ Significant performance overhead from I/O
- ⚠️ Optimal chunking strategy critical
- ⚠️ Limited Julia ecosystem support currently

## Recommended Next Steps

### Near-Term (High Impact, Low Effort)

1. **Document Memory Requirements**
   - Add memory profiling to all scripts
   - Create lookup table: scene size × days → memory needed
   - Update README with memory planning guide

2. **Implement Spatial Cropping Utilities**
   - Helper functions for extracting spatial subsets
   - Parallel processing of spatial tiles
   - Stitching utilities for tiled outputs

3. **Optimize Current Loading**
   - Profile memory hotspots in loading code
   - Avoid intermediate copies where possible
   - Use `@views` for slicing operations

### Medium-Term (Moderate Effort, High Impact)

4. **Hybrid Lazy Loading**
   - Refactor loading to keep spatial data lazy
   - Materialize per-window during fusion
   - Benchmark I/O vs memory trade-offs

5. **Temporal Chunking with Overlap**
   - Manual temporal chunking utilities
   - Overlap and blending strategies
   - Quality assessment at boundaries

### Long-Term (High Effort, Transformative)

6. **Sequential Kalman Filter Implementation**
   - Research one-pass fusion algorithms
   - Implement streaming variant
   - Validate against current approach

7. **Distributed Computing Support**
   - MPI.jl or Dagger.jl integration
   - Distributed temporal chunks
   - Coordinated state management

## Performance Targets

### Current Capabilities
- Spatial extent: Up to 1411×2085 pixels (full scene)
- Temporal extent: 5-30 days (practical on 64 GB systems)
- Memory efficiency: ~1.6 GB/day of data

### Target Capabilities (Post-Optimization)
- Spatial extent: Arbitrary (tiled processing)
- Temporal extent: 365+ days (streaming/chunking)
- Memory efficiency: Constant memory regardless of temporal extent

## Conclusion

The current eager loading strategy is well-suited for:
- ✓ Short time series (< 30 days)
- ✓ Moderate spatial extents
- ✓ High-memory systems (64+ GB)
- ✓ Algorithms requiring random temporal access

Future work on streaming/lazy architectures would enable:
- ✓ Multi-year time series
- ✓ Continental-scale spatial processing
- ✓ Lower-memory systems
- ✓ Production-scale operational deployment

The path forward depends on use case priorities: quality and flexibility (current) vs scale and efficiency (future streaming approach).

---

**Document Version:** 1.0  
**Date:** February 24, 2026  
**Based on:** HyperSTARS.jl v1.2.0 performance analysis  
**Test Case:** Kings Canyon HLS+EMIT fusion (1411×2085 pixels, 5-31 days)
