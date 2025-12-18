# HyperSTARS.jl Consolidation Action Plan

**Repository:** STARS-Data-Fusion/HyperSTARS.jl  
**Date:** December 18, 2025  
**Status:** Ready for Implementation  
**Depends On:** STARSDataFusion.jl consolidation must complete first

---

## Overview

This action plan details the changes needed in **HyperSTARS.jl** to import shared utilities from STARSDataFusion.jl instead of maintaining duplicate copies. After completion, HyperSTARS will focus solely on hyperspectral/spectral fusion algorithms while using STARSDataFusion for utilities.

### Goals for This Repository

1. ✅ Add STARSDataFusion.jl as a dependency
2. ✅ Update HyperSTARS.jl to import utilities from STARSDataFusion
3. ✅ Delete duplicate utility files (GP_utils, resampling_utils, spatial_utils)
4. ✅ Maintain 100% backward compatibility - no API changes
5. ✅ Keep all spectral fusion algorithms in HyperSTARS

### What Stays the Same

- All exported functions remain available
- All function signatures unchanged
- All spectral fusion algorithms stay in HyperSTARS
- User code continues to work without modifications
- Examples run unchanged

### What Changes (Internal Only)

- Utility functions imported from STARSDataFusion instead of local files
- Project.toml gains STARSDataFusion dependency
- Utility files deleted (no longer needed)

---

## Prerequisites

⚠️ **IMPORTANT:** STARSDataFusion.jl consolidation must complete first!

Before starting HyperSTARS changes:
- [ ] STARSDataFusion.jl has enhanced GP_utils.jl
- [ ] STARSDataFusion.jl has merged resampling_utils.jl
- [ ] STARSDataFusion.jl exports new utility functions
- [ ] STARSDataFusion.jl tests pass
- [ ] STARSDataFusion.jl new version tagged and released

---

## Action Items

### Phase 1: Add Dependency (Week 1)

#### Task 1.1: Update Project.toml

**File:** `Project.toml`

**Action:** Add STARSDataFusion as a dependency

**Current dependencies:**
```toml
[deps]
BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GaussianRandomFields = "e4b2fa70-8a2f-543f-b56c-746f5f3b229e"
GeoArrays = "2fb1d81b-e6a0-5fc5-82e6-8e06903437ab"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
KernelFunctions = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
Kronecker = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Rasters = "a3a2b9e3-a471-40c9-b274-f62e5ce4b119"
Sobol = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

**Add:**
```toml
STARSDataFusion = "..."  # UUID will be added automatically
```

**Also update compat bounds:**
```toml
[compat]
STARSDataFusion = "0.X+1"  # Require new version with enhanced utilities
```

**Steps:**
1. ✅ Open `Project.toml`
2. ✅ Add `STARSDataFusion` to `[deps]` section
3. ✅ Add version constraint to `[compat]` section
4. ✅ Run `julia --project -e 'using Pkg; Pkg.resolve()'`
5. ✅ Verify dependencies resolve correctly

**Verification:**
```bash
julia --project -e 'using Pkg; Pkg.status()'
# Should show STARSDataFusion as dependency
```

**Risk:** Low - additive change  
**Mitigation:** Can remove dependency if issues arise

---

#### Task 1.2: Remove Redundant Dependencies (Optional)

**File:** `Project.toml`

**Action:** Remove dependencies now provided by STARSDataFusion

**Can potentially remove** (already in STARSDataFusion):
- `Distances` (used in GP_utils)
- `GaussianRandomFields` (used in GP_utils)
- `Sobol` (used in spatial_utils)
- `Interpolations` (used in resampling_utils)

**Keep** (HyperSTARS-specific):
- `BlockDiagonals` (spectral fusion)
- `Kronecker` (spectral fusion)
- `KernelFunctions` (spectral fusion)
- Other spectral-specific packages

**Steps:**
1. ✅ Identify which dependencies are only used in utility files
2. ✅ Verify HyperSTARS spectral code doesn't use them directly
3. ✅ Remove from `[deps]` if safe
4. ✅ Test that code still works

**Verification:**
```bash
julia --project -e 'using HyperSTARS'
# Should load without errors
```

**Risk:** Medium - could break if assumptions wrong  
**Mitigation:** Can re-add dependencies if needed; start conservative

---

### Phase 2: Update Main Module (Week 1-2)

#### Task 2.1: Modify HyperSTARS.jl to Import Utilities

**File:** `src/HyperSTARS.jl`

**Action:** Add imports from STARSDataFusion for utility functions

**At the top of file, add:**
```julia
# Import shared utilities from STARSDataFusion
using STARSDataFusion
import STARSDataFusion: 
    # GP utilities
    exp_cor, mat32_cor, mat52_cor,
    exp_corD, mat32_corD, mat52_corD,
    matern_cor, matern_cor_nonsym, matern_cor_fast,
    kernel_matrix, state_cov,
    build_gpcov, mat32_1D, mat32_cor2, mat32_cor3,
    # Spatial utilities
    find_nearest_ij, find_nearest_ij_multi, find_nearest_ind,
    find_all_bau_ij, get_sij_from_ij,
    get_origin_raster, get_centroid_origin_raster,
    bbox_from_centroid, cell_size,
    # Resampling utilities
    unif_weighted_obs_operator_centroid,
    rsr_conv_matrix,
    # General utilities
    nanmean
```

**Re-export for backward compatibility:**
```julia
# Re-export utilities so users can still do `using HyperSTARS; exp_cor(...)`
export exp_cor, mat32_cor, mat52_cor
export exp_corD, mat32_corD, mat52_corD
export matern_cor, matern_cor_nonsym, matern_cor_fast
export unif_weighted_obs_operator_centroid
export nanmean, cell_size, get_centroid_origin_raster
export build_gpcov, mat32_1D
export rsr_conv_matrix
```

**Steps:**
1. ✅ Open `src/HyperSTARS.jl`
2. ✅ Add `using STARSDataFusion` near top
3. ✅ Add `import STARSDataFusion: ...` for all needed utilities
4. ✅ Add `export` statements for backward compatibility
5. ✅ Remove any `include()` statements for utility files
6. ✅ Verify module loads without errors

**Current includes to remove:**
```julia
# DELETE these lines:
include("GP_utils.jl")
include("resampling_utils.jl")
include("spatial_utils_ll.jl")
include("spatial_utils.jl")
```

**Keep:**
```julia
# KEEP all spectral fusion code
# All struct definitions
# All spectral fusion functions
```

**Verification:**
```julia
using HyperSTARS

# Verify re-exports work
@test isdefined(HyperSTARS, :exp_cor)
@test isdefined(HyperSTARS, :HSModel)
@test isdefined(HyperSTARS, :hyperSTARS_fusion_kr_dict)
```

**Risk:** Medium - must import all needed functions  
**Mitigation:** Comprehensive testing; can add missing imports if needed

---

### Phase 3: Delete Redundant Files (Week 2)

#### Task 3.1: Remove Utility Files

**Action:** Delete utility files now imported from STARSDataFusion

**Files to delete:**
- `src/GP_utils.jl` (163 lines) - now from STARSDataFusion
- `src/resampling_utils.jl` (94 lines) - now from STARSDataFusion
- `src/spatial_utils_ll.jl` (378 lines) - now from STARSDataFusion
- `src/spatial_utils.jl` (162 lines) - deprecated version

**Steps:**
1. ✅ Ensure Task 2.1 complete (imports working)
2. ✅ Run tests to verify utilities accessible
3. ✅ Delete files:
   ```bash
   cd src/
   git rm GP_utils.jl
   git rm resampling_utils.jl
   git rm spatial_utils_ll.jl
   git rm spatial_utils.jl
   ```
4. ✅ Test again to ensure nothing broken

**Verification:**
```bash
julia --project -e 'using HyperSTARS; @assert isdefined(HyperSTARS, :exp_cor)'
```

**Risk:** Low - already imported from STARSDataFusion  
**Mitigation:** Can restore from git if needed

**Result:** ~800 lines of code removed (all duplicates)

---

#### Task 3.2: Update Directory Structure

**Before:**
```
HyperSTARS.jl/
└── src/
    ├── HyperSTARS.jl (345 lines)
    ├── GP_utils.jl (163 lines) ❌ DELETE
    ├── resampling_utils.jl (94 lines) ❌ DELETE
    ├── spatial_utils_ll.jl (378 lines) ❌ DELETE
    └── spatial_utils.jl (162 lines) ❌ DELETE
```

**After:**
```
HyperSTARS.jl/
└── src/
    └── HyperSTARS.jl (~350 lines, updated with imports)
        ├── Imports from STARSDataFusion
        ├── HyperSTARS-specific exports
        ├── Spectral fusion data structures
        └── Spectral fusion algorithms
```

---

### Phase 4: Testing (Week 2-3)

#### Task 4.1: Update Test Suite

**File:** `test/runtests.jl`

**Action:** Ensure all tests still pass with imported utilities

**Add test:**
```julia
@testset "Utility imports from STARSDataFusion" begin
    using HyperSTARS
    
    # Verify utilities available
    @test isdefined(HyperSTARS, :exp_cor)
    @test isdefined(HyperSTARS, :mat32_cor)
    @test isdefined(HyperSTARS, :unif_weighted_obs_operator_centroid)
    @test isdefined(HyperSTARS, :nanmean)
    @test isdefined(HyperSTARS, :cell_size)
    @test isdefined(HyperSTARS, :get_centroid_origin_raster)
    
    # Verify new utilities available
    @test isdefined(HyperSTARS, :build_gpcov)
    @test isdefined(HyperSTARS, :mat32_1D)
    @test isdefined(HyperSTARS, :rsr_conv_matrix)
end

@testset "Spectral fusion unchanged" begin
    # Verify all HyperSTARS-specific functions still work
    @test isdefined(HyperSTARS, :HSModel)
    @test isdefined(HyperSTARS, :InstrumentData)
    @test isdefined(HyperSTARS, :hyperSTARS_fusion_kr_dict)
    @test isdefined(HyperSTARS, :woodbury_filter_kr)
    @test isdefined(HyperSTARS, :scene_fusion_pmap)
end
```

**Steps:**
1. ✅ Update test file
2. ✅ Run full test suite
3. ✅ Verify all tests pass
4. ✅ Check for any new warnings

**Verification:**
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

**Risk:** Medium - tests might reveal missing imports  
**Mitigation:** Add any missing imports to main module

---

#### Task 4.2: Run Examples

**Action:** Verify all examples work unchanged

**Files to test:**
- `examples/hyperstars_example.jl`

**Steps:**
1. ✅ Run example:
   ```bash
   julia --project examples/hyperstars_example.jl
   ```
2. ✅ Verify output matches baseline
3. ✅ Check for any errors or warnings
4. ✅ Verify results scientifically correct

**Verification:**
- Example completes without errors
- Results within expected range
- No deprecation warnings

**Risk:** Low - examples should use exported functions  
**Mitigation:** Fix any import issues discovered

---

#### Task 4.3: Backward Compatibility Test

**Create:** `test/test_backward_compatibility.jl`

**Action:** Ensure existing user code patterns still work

```julia
@testset "Backward compatibility" begin
    # Pattern 1: Import and use utilities
    using HyperSTARS
    
    x1 = [0.0, 0.0]
    x2 = [1.0, 1.0]
    pars = [1.0, 200.0, 1e-10, 1.5]
    
    @test exp_cor(x1, x2, pars) isa Real
    @test mat32_cor(x1, x2, pars) isa Real
    
    # Pattern 2: Use spectral fusion
    data = InstrumentData(...)
    geodata = InstrumentGeoData(...)
    model = HSModel(...)
    
    @test isa(model, HSModel)
    
    # Pattern 3: Full workflow
    result = hyperSTARS_fusion_kr_dict(...)
    @test isa(result, ...)
end
```

**Steps:**
1. ✅ Create backward compatibility test
2. ✅ Test common usage patterns
3. ✅ Verify no breaking changes
4. ✅ Document any changes needed (should be none)

**Risk:** High - critical for users  
**Mitigation:** Fix any issues before release

---

### Phase 5: Documentation (Week 3)

#### Task 5.1: Update README

**File:** `README.md`

**Action:** Document dependency on STARSDataFusion

**Add section:**
```markdown
## Dependencies

HyperSTARS.jl now uses shared utilities from [STARSDataFusion.jl](https://github.com/STARS-Data-Fusion/STARSDataFusion.jl):
- Gaussian process covariance functions
- Spatial utility functions
- Resampling and observation operators

All utility functions are re-exported by HyperSTARS.jl, so your code works unchanged:

\`\`\`julia
using HyperSTARS

# Utilities available as before
cov = exp_cor(x1, x2, pars)
obs = unif_weighted_obs_operator_centroid(...)

# Spectral fusion functions (HyperSTARS-specific)
model = HSModel(...)
result = hyperSTARS_fusion_kr_dict(...)
\`\`\`

### Installation

\`\`\`julia
using Pkg
Pkg.add("HyperSTARS")  # Automatically installs STARSDataFusion dependency
\`\`\`
```

**Steps:**
1. ✅ Update README.md
2. ✅ Clarify dependency structure
3. ✅ Note backward compatibility
4. ✅ Update installation instructions if needed

---

#### Task 5.2: Update Documentation

**Files:** `docs/src/*.md` (if using Documenter.jl)

**Action:** Update API docs to note utilities from STARSDataFusion

**Add note in API reference:**
```markdown
### Utility Functions

The following utility functions are imported from STARSDataFusion.jl and re-exported for convenience:

- `exp_cor`, `mat32_cor`, `mat52_cor` - Covariance functions
- `unif_weighted_obs_operator_centroid` - Observation operators
- `cell_size`, `get_centroid_origin_raster` - Spatial utilities
- `nanmean` - Statistical utilities

For full documentation, see [STARSDataFusion.jl docs](link).

### HyperSTARS-Specific Functions

The following functions are unique to HyperSTARS.jl for spectral fusion:

- `HSModel`, `InstrumentData`, `InstrumentGeoData` - Data structures
- `hyperSTARS_fusion_kr_dict()` - Main fusion algorithm
- `woodbury_filter_kr()` - Kalman filtering with Woodbury identity
- `scene_fusion_pmap()` - Parallelized scene fusion
```

**Steps:**
1. ✅ Update documentation
2. ✅ Link to STARSDataFusion docs
3. ✅ Clarify what's HyperSTARS-specific

---

### Phase 6: Version Release (Week 3-4)

#### Task 6.1: Update Project.toml Version

**File:** `Project.toml`

**Action:** Bump version number

**Current:** `version = "0.A.B"`  
**New:** `version = "0.(A+1).0"`

**Rationale:** Minor version bump (internal changes, no API changes)

**Steps:**
1. ✅ Update version in Project.toml
2. ✅ Update CHANGELOG.md
3. ✅ Tag release in git

---

#### Task 6.2: Create CHANGELOG Entry

**File:** `CHANGELOG.md`

**Add:**
```markdown
## [0.A+1.0] - 2025-XX-XX

### Changed
- **Internal:** HyperSTARS now imports shared utilities from STARSDataFusion.jl
- Removed duplicate utility files (GP_utils, resampling_utils, spatial_utils)
- All utilities re-exported for backward compatibility

### Added
- Access to enhanced utilities from STARSDataFusion.jl
  - `build_gpcov()` for building GP covariance matrices
  - `mat32_1D()`, `mat32_cor2()`, `mat32_cor3()` additional correlation functions
  - Enhanced `rsr_conv_matrix()` spectral resampling

### Notes
- **No breaking changes** - all existing code works unchanged
- **No user action required** - STARSDataFusion installed automatically as dependency
- All spectral fusion algorithms remain in HyperSTARS.jl
- ~800 lines of duplicate code eliminated through consolidation
```

---

## Testing Checklist

Before release, verify:

- [ ] STARSDataFusion.jl consolidated version released
- [ ] HyperSTARS imports utilities successfully
- [ ] All existing tests pass
- [ ] New tests for imports pass
- [ ] Examples run successfully
- [ ] No deprecation warnings
- [ ] Documentation updated
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] No performance regression (±10%)

---

## Rollback Plan

If issues discovered after release:

1. **Immediate:** Tag current state as `v0.A.B-pre-consolidation`
2. **Revert:** Restore utility files from git history
3. **Remove:** STARSDataFusion dependency from Project.toml
4. **Release:** Tag reverted version as `v0.A.B+1`
5. **Communicate:** Notify users of temporary rollback

**Files to restore if rollback needed:**
- `src/GP_utils.jl` (from git @ commit before deletion)
- `src/resampling_utils.jl` (from git)
- `src/spatial_utils_ll.jl` (from git)
- `src/HyperSTARS.jl` (remove STARSDataFusion imports)
- `Project.toml` (remove STARSDataFusion dependency)

---

## Timeline

| Week | Tasks | Status |
|------|-------|--------|
| 1 | Task 1.1-1.2: Update Project.toml | ⬜ Not Started |
| 1-2 | Task 2.1: Update main module imports | ⬜ Not Started |
| 2 | Task 3.1-3.2: Delete redundant files | ⬜ Not Started |
| 2-3 | Task 4.1-4.3: Testing | ⬜ Not Started |
| 3 | Task 5.1-5.2: Documentation | ⬜ Not Started |
| 3-4 | Task 6.1-6.2: Release | ⬜ Not Started |

**Total Duration:** ~3-4 weeks (after STARSDataFusion completion)

---

## Dependencies

**Prerequisites:**
- STARSDataFusion.jl consolidation complete
- STARSDataFusion.jl new version released and registered

**After completion:**
- HyperSTARS.jl users automatically get STARSDataFusion as dependency
- No user code changes required
- Both packages maintained separately

**No changes needed for:**
- User code importing HyperSTARS
- Examples or tutorials
- Publications citing HyperSTARS

---

## Questions or Issues?

**Contact:**
- GitHub Issues: https://github.com/STARS-Data-Fusion/HyperSTARS.jl/issues
- Maintainer: [contact email]

**Reference:**
- Full consolidation proposal: `Code Consolidation Proposal.md`
- STARSDataFusion action plan: `../STARSDataFusion.jl/CONSOLIDATION_ACTION_PLAN.md`

---

**Last Updated:** December 18, 2025  
**Status:** Waiting for STARSDataFusion.jl consolidation  
**Estimated Start:** After STARSDataFusion.jl release  
**Estimated Completion:** 3-4 weeks after start
