# Comparison: HyperSTARS Overleaf Manuscript vs. HyperSTARS.jl Package

**Date**: February 3, 2026

## Overview

The **HyperSTARS Overleaf manuscript** (`main.tex`) is a methodological document describing the theoretical foundations, mathematical formulations, and algorithmic approaches for hyperspectral data fusion. The **HyperSTARS.jl package** is the practical implementation of these methods in Julia, designed for real-world hyperspectral data processing.

---

## 1. Core Methodology Alignment

| **Component** | **Manuscript (Theoretical)** | **Julia Package (Implementation)** | **Status** |
|--------------|------------------------------|-------------------------------------|------------|
| **Dynamic Linear Models (DLMs)** | Eqs. 1-3: State-space formulation with observation/process equations | Implemented via `KSModel` and `HSModel` structs in `src/HyperSTARS.jl` | ✅ **Fully aligned** |
| **Observation Model** | Eq. 4-6: Change-of-support operators ($\bm{F}_t$, $\bm{H}_s$, $\bm{H}_w$) | `unif_weighted_obs_operator_centroid`, `rsr_conv_matrix` in `src/resampling_utils.jl` | ✅ **Implemented** |
| **Spectral Dimensionality Reduction** | Eq. 6: FPCA decomposition $x_t(\omega,\bm{s}) = \mu(\omega) + \sum \phi_j(\omega)\eta_{jt}(\bm{s})$ | PCA via `MultivariateStats` in `examples/hyperstars_example.jl` | ✅ **Using PCA, not FPCA** |
| **Process Model** | Eqs. 7-9: Random walk + spatial GP covariance | `woodbury_filter_kr` with spatial covariance from `src/GP_utils.jl` | ✅ **Fully implemented** |
| **Kalman Filtering** | Woodbury identity for efficient updates | `woodbury_filter_kr` function | ✅ **Optimized implementation** |
| **Kalman Smoothing** | RTS backward smoothing (Section 1) | `smooth_series` function | ✅ **Implemented** |
| **Moving Window Partition** | Section 1.4: Block partitioning with buffers | `scene_fusion_pmap` | ✅ **Parallel implementation** |

---

## 2. Advanced Features

### A. Kronecker Product Structures (Section 1.5.2)

**Manuscript**: Derives computational tricks for efficient matrix operations using Kronecker products:
- Eq. 12-18: $\bm{H} = \bm{H}_w \bm{B} \otimes \bm{H}_s$
- Exploits separability of spectral and spatial operations

**Julia Package**: 
- ✅ **Implemented** in `woodbury_filter_kr` function
- Uses `kronecker()` from `Kronecker.jl`
- See lines 221-269 in `src/HyperSTARS.jl` for Woodbury identity with Kronecker structure

**Alignment**: Strong — The manuscript's derivations directly inform the code's efficient matrix computations.

---

### B. Adaptive Process Noise (Section 2.1: State in Covariance)

**Manuscript**: Proposes including state values in covariance:
- Eq. 10-11: $C(s,s') = \alpha C_s(s,s') + (1-\alpha) C_x(x_{t-1}(s), x_{t-1}(s'))$
- Motivation: Spatial correlation should depend on **similarity of reflectance values**, not just distance

**Julia Package**:
- ✅ **Implemented** as `state_in_cov` flag in `hyperSTARS_fusion_kr_dict`
- `state_cov()` function in `src/GP_utils.jl`
- Weighted combination: `Qf = cov_wt * Q + (1 - cov_wt) * Qss`

**Alignment**: **Excellent** — The manuscript's conceptual innovation is directly translated into code with a tunable `cov_wt` parameter.

---

### C. Parameter Estimation (Section 1.3 & 1.6)

| **Method** | **Manuscript** | **Julia Package** | **Status** |
|-----------|----------------|-------------------|------------|
| **MLE via Score Function** | Section 1.6: Detailed derivations (Eqs. 32-43) | ❌ **Not implemented** | 🔄 **Future work** |
| **Hessian for Newton-Raphson** | Eqs. after 43 | ❌ **Not implemented** | 🔄 **Future work** |
| **Subsampling + Interpolation** | Figure 2: Spatial GP for parameter interpolation | ❌ **Not implemented** | 🔄 **Future work** |
| **Manual Parameter Setting** | Assumed in examples | ✅ **Current approach** | — |

**Key Gap**: The manuscript details sophisticated MLE methods, but the current Julia package requires **manually specified** `model_pars` (spatial variance/range). This is the biggest methodological gap.

---

### D. Ensemble Kalman Filter (Section 2.2)

**Manuscript**: Brief mention of EnKF as alternative to standard Kalman filter

**Julia Package**: ❌ **Not implemented**

**Status**: 🔄 **Planned future work** (mentioned in manuscript as an idea)

---

## 3. Computational Efficiency

| **Feature** | **Manuscript** | **Julia Package** | **Notes** |
|------------|----------------|-------------------|-----------|
| **Block Diagonal Approximation** | Section 1.5.3: $\bm{P}_{t\|t-1}$ as block diagonal | ⚠️ **Partially explored** | Code comments suggest potential optimization |
| **Sequential Instrument Updates** | Section 1.5.1: Independent updates per instrument | ✅ **Implemented** in filtering loops | See lines 600-623 in `src/HyperSTARS.jl` |
| **Parallel Processing** | Not explicitly discussed | ✅ **Core feature** via `pmap` | `scene_fusion_pmap` function |
| **BLAS Optimizations** | Not discussed | ✅ **Implemented** | `LAPACK.potrf!`, BLAS operations throughout |

**Advantage of Code**: The Julia implementation adds **parallel processing** and **low-level BLAS optimizations** not detailed in the manuscript.

---

## 4. Data Handling

### Instrument Integration

**Manuscript**:
- Describes STARS (spatial + temporal) and HyperSTARS (spatial + spectral + temporal)
- Examples: VIIRS-HLS fusion (STARS), EMIT-HLS-PACE fusion (HyperSTARS)

**Julia Package**:
- ✅ **Structured data containers**: `InstrumentData`, `InstrumentGeoData`
- ✅ **Fidelity levels** (0=target, 1=high-res, 2=coarse) guide sampling strategy
- ✅ **Spectral Response Functions** (SRF): Handles discrete (HLS) and continuous (EMIT/PACE)
- ✅ **Real data support**: See `examples/emit_hls_demo.jl` and `notes/EMIT_DATA_WORKFLOW.md`

**Alignment**: **Strong** — The code provides a robust, practical framework for multi-sensor fusion matching the manuscript's vision.

---

## 5. Manuscript-Only Content (Not in Code)

1. **Theoretical Derivations** (Section 1.6):
   - Log-likelihood functions (Eq. 31)
   - Score functions and Hessians (Eqs. 32-43)
   - Parameter transformations (log-space)

2. **Subsampling for Parameter Estimation** (Section 1.3):
   - Figure 2 workflow: Estimate on subset → Interpolate via GP

3. **Figures/Simulations**:
   - Figure 1: Partitioned parameter estimates
   - Figure 3: Synthetic data examples
   - Figure 5: Subsampling artifacts
   - Figure 6: State-in-covariance simulations

4. **Discussion of Kalman Filter Violations** (Section 2.1):
   - Acknowledgment that state-dependent covariance violates linearity assumptions
   - "Unclear of practical implications" — suggests need for empirical validation

---

## 6. Code-Only Features (Not in Manuscript)

1. **Parallel Infrastructure**:
   - `Distributed.jl` integration with `pmap`
   - Worker process management (`addprocs`, `@everywhere`)
   
2. **Production-Ready Tools**:
   - Progress bars (`@showprogress`)
   - Memory-efficient in-place operations (`woodbury_filter_kr!`)
   - Raster I/O via `Rasters.jl`, `ArchGDAL.jl`

3. **Comprehensive Examples**:
   - `examples/hyperstars_example.jl`: Synthetic data demo
   - `examples/emit_hls_demo.jl`: Real data fusion

4. **Documentation**:
   - Extensive README with step-by-step setup
   - Troubleshooting guides
   - Data workflow documentation

---

## 7. Key Differences

| **Aspect** | **Manuscript** | **Julia Package** |
|-----------|----------------|-------------------|
| **Purpose** | Research documentation, methodological exploration | Production tool for data analysis |
| **Scope** | Broad (STARS + HyperSTARS + future ideas) | Focused on HyperSTARS implementation |
| **Parameter Estimation** | Detailed MLE/Bayesian methods | Manual specification (for now) |
| **State-in-Covariance** | Conceptual proposal with simulations | Implemented with `cov_wt` tuning parameter |
| **Validation** | Synthetic examples (Figures) | Real EMIT/HLS/PACE data workflows |

---

## 8. Recommendations for Alignment

### Priority 1: Implement Automated Parameter Estimation
- Add `estimate_model_pars()` function based on Section 1.6 equations
- Implement subsampling + GP interpolation strategy (Figure 2)
- **Benefit**: Eliminate manual tuning, make package accessible to non-experts

### Priority 2: Document State-in-Covariance
- Add manuscript Section 2.1 derivations to code comments
- Provide guidance on choosing `cov_wt` parameter
- **Benefit**: Users understand when/why to use adaptive covariance

### Priority 3: Cross-Reference Equations
- Add comments like `# Implements Eq. 7 from manuscript` in `src/HyperSTARS.jl`
- Create a "Math-to-Code" translation guide
- **Benefit**: Bridge gap between theory (manuscript) and implementation (code)

### Priority 4: Sync Examples
- Add synthetic data generation script matching Figure 3
- Create validation notebooks comparing results to manuscript figures
- **Benefit**: Reproducibility and method verification

---

## 9. Overall Assessment

### Strengths
- ✅ **Core algorithms are faithfully implemented** (Kalman filtering/smoothing, spatial GP, Kronecker products)
- ✅ **Code is production-ready** with parallelization, error handling, and documentation
- ✅ **Innovative features** (adaptive covariance) transition from theory to practice

### Gaps
- ❌ **Parameter estimation is manual** (biggest practical limitation)
- ⚠️ **Some advanced ideas remain conceptual** (EnKF, full block-diagonal exploits)

### Recommendation
The manuscript and package complement each other well. To maximize impact:
1. Add automated parameter estimation to the package
2. Publish the manuscript with code repository references
3. Create a "Reproducing Manuscript Results" vignette in the package

---

## Summary Table

| **Category** | **Manuscript Coverage** | **Package Implementation** | **Alignment** |
|-------------|------------------------|---------------------------|---------------|
| Core DLM Theory | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| Kalman Filtering | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| Spatial Covariance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Very Good |
| Parameter Estimation | ⭐⭐⭐⭐⭐ | ⭐ | ❌ Major Gap |
| Computational Efficiency | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Code exceeds manuscript |
| Real Data Examples | ⭐ | ⭐⭐⭐⭐⭐ | ✅ Code exceeds manuscript |
| Documentation | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Code exceeds manuscript |

**Overall**: The HyperSTARS.jl package is a robust, well-engineered implementation of the methodology described in the manuscript, with the notable exception of automated parameter estimation. The package adds significant value through parallelization, real-world data handling, and production-ready features.

---

## Detailed Component Mapping

### Observation Equation Components

**Manuscript Eq. 5**:
```
y_t^k (A_w^k,A_s^k) = ∑_{A_s^k ∩ D} w^k(A_s^k, s) {∑_{A_w^k ∩ W} λ^k(A_w^k, ω) x_t(ω,s)} + v_t^k(A_w^k,A_s^k)
```

**Code Implementation**:
- `w^k(A_s^k, s)`: Implemented by `unif_weighted_obs_operator_centroid()` in `src/spatial_utils_ll.jl`
- `λ^k(A_w^k, ω)`: Implemented by `rsr_conv_matrix()` in `src/resampling_utils.jl`
- Overall operator: `Hs * x_predr * Hw'` in `woodbury_filter_kr` (line ~218)

### Process Equation Components

**Manuscript Eq. 7-8**:
```
η_{jt} = η_{jt-1} + ω_t,  ω_t ~ N(0, W_{jt})
η_{j0} ~ N(0, C_0)
```

**Code Implementation**:
- Random walk: `F = UniformScaling(ar_phi)` where `ar_phi=1.0` by default
- Spatial covariance `W`: Built in initialization as `Q = Matrix(BlockDiagonal(Qs))`
- Predict step: `x_pred = F * filtering_means[:,t]` and `P_pred = F * filtering_covs[:,:,t] * F' + Qf`

### Kalman Gain via Woodbury Identity

**Manuscript Eqs. (implicit in Section 1.5)**:
The Woodbury matrix identity allows efficient computation of:
```
(P^{-1} + H'R^{-1}H)^{-1}
```

**Code Implementation** (lines 221-269):
```julia
HViH .+= kronecker(HwtV*Hw, HstV*Hs)  # Accumulate H'V^{-1}H
S = P_pred[:,:] + HViH                 # S = P^{-1} + H'V^{-1}H
# Multiple LAPACK calls for efficient inversion
HtRHIi = BLAS.symm('R', 'U', S, HViH) # Key Woodbury term
```

### Spatial Covariance Functions

**Manuscript Eq. 8**:
```
Cov(η_{jt}(s), η_{jt}(s'), ψ_m) = σ²_{j,m} exp(-||s - s'||/φ_{j,m})
```

**Code Implementation** (`src/GP_utils.jl`):
```julia
function exp_corD(D::AbstractArray{<:Real}, pars::AbstractVector{<:Real})
    return exp.(-D ./ pars[1])
end

function mat32_corD(D::AbstractArray{<:Real}, pars::AbstractVector{<:Real})
    r = D ./ pars[1]
    return (1.0 .+ sqrt(3.0) .* r) .* exp.(-sqrt(3.0) .* r)
end
```

Called as: `Qs[i] = x[1] .* spatial_mod(dd, x[2:end])` where `x[1]` is `σ²` and `x[2]` is `φ`.

---

## Implementation Notes

### Memory Management
- In-place operations (`woodbury_filter_kr!`) allocate helper structs (`HSModel_helpers`)
- Pre-allocated arrays: `filtering_means`, `filtering_covs`, `predicted_means`, `predicted_covs`
- Avoids repeated allocations in time loop

### Numerical Stability
- Uses Cholesky decomposition (`LAPACK.potrf!`) for matrix inversions
- Symmetric matrix handling (`BLAS.symm`) avoids full matrix operations
- `copytri!` ensures full symmetric matrices after triangular operations

### Parallel Strategy
- Scene divided into windows via `window_geodata.ndims`
- Each window processed independently in `pmap`
- Results reassembled by Cartesian indexing: `fused_image[result[i][1],:,:] = result[i][2]`

---

## Testing and Validation Needs

### Current Validation
- ✅ Synthetic data example (`examples/hyperstars_example.jl`)
- ✅ Real EMIT+HLS data example (`examples/emit_hls_demo.jl`)

### Missing Validation (from Manuscript)
- ❌ Comparison to Figures 1, 3, 5, 6 in manuscript
- ❌ Quantitative metrics (RMSE, correlation) vs. ground truth
- ❌ Parameter sensitivity analysis
- ❌ Computational benchmarks (time vs. scene size)

### Recommended Tests
1. **Reproduce Figure 5**: Test subsampling effects with varying `nsamp`
2. **Reproduce Figure 6**: Test `state_in_cov` with varying `cov_wt`
3. **Benchmark Suite**: Time vs. (scene size, # bands, # time steps, # workers)
4. **Unit Tests**: Individual functions (covariance, observation operators, etc.)

---

## Future Development Roadmap

### Short Term (1-3 months)
1. Implement basic MLE for spatial parameters
2. Add equation cross-references in code comments
3. Create validation notebooks for manuscript figures

### Medium Term (3-6 months)
1. Full MLE with Hessian-based optimization
2. Parameter interpolation via GP
3. Automated model selection (AR order, covariance function)

### Long Term (6-12 months)
1. Ensemble Kalman Filter implementation
2. Full block-diagonal covariance exploitation
3. GPU acceleration for large scenes
4. Python bindings via PythonCall.jl

---

## Related Files

- **Manuscript**: `../HyperSTARS_Overleaf/main.tex`
- **Main Package**: `src/HyperSTARS.jl`
- **Spatial Utils**: `src/spatial_utils_ll.jl`, `src/spatial_utils.jl`
- **GP Utils**: `src/GP_utils.jl`
- **Resampling**: `src/resampling_utils.jl`
- **Examples**: `examples/hyperstars_example.jl`, `examples/emit_hls_demo.jl`
- **Workflow Guide**: `notes/EMIT_DATA_WORKFLOW.md`
