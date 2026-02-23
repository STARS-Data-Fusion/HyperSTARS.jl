# HyperSTARS.jl Unit Test Suite

This directory contains comprehensive unit tests for the HyperSTARS.jl package designed to catch breaking changes in future updates.

## Test Organization

The test suite is organized into the following modules:

### `test_structures.jl`
Tests for HyperSTARS data structures:
- `InstrumentData` creation and validation
- `InstrumentGeoData` creation and validation
- `KSModel` (Kalman State Model) structure
- `HSModel` (Hyperspectral STARS Model) structure
- Data consistency checks across different structure types

**Key Tests:**
- Basic structure creation
- Type preservation
- Dimension consistency
- Support for different parameter combinations

### `test_kalman_filtering.jl`
Tests for Kalman filtering and smoothing algorithms:
- `woodbury_filter_kr()` - Woodbury matrix identity-based Kalman filtering
- `smooth_series()` - Rauch-Tung-Striebel Kalman smoothing
- Filter with single and multiple instruments
- Covariance matrix properties and numerical stability

**Key Tests:**
- Basic filtering functionality
- Multiple instrument fusion
- Covariance reduction during filtering
- Smoothing with AR(1) processes
- Symmetric and positive-definite covariance maintenance

### `test_observation_operators.jl`
Tests for spatial and spectral observation operators:
- `unif_weighted_obs_operator_centroid()` - Uniform-weighted observation mapping
- `gauss_weighted_obs_operator()` - Gaussian-weighted observation mapping
- `rsr_conv_matrix()` - Spectral response function convolution

**Key Tests:**
- Identity matrix cases
- Spatial overlap calculations
- Row normalization
- Sparse matrix structure
- Resolution parameter effects
- Distance-based weight decay

### `test_gp_utils.jl`
Tests for Gaussian Process and covariance utilities:
- `exp_corD()` - Exponential correlation kernel
- `mat32_corD()` - Matérn 3/2 kernel
- `mat52_corD()` - Matérn 5/2 kernel
- `nanmean()`, `nanvar()` - NaN-aware statistics
- `state_cov()` - State-dependent covariance

**Key Tests:**
- Kernel positivity and symmetry
- Distance decay properties
- Length scale parameter effects
- Numerical stability at zero distance
- NaN handling in statistics

### `test_data_organization.jl`
Tests for data preprocessing and organization:
- `organize_data()` - Multi-instrument data organization by resolution fidelity
- Multi-fidelity instrument handling
- Spatial extent calculations
- Data preservation through organization

**Key Tests:**
- Basic data organization
- Multiple instruments at different resolutions
- Fidelity level handling
- Data shape preservation
- Output consistency

### `test_integration.jl`
Integration tests combining multiple components:
- Complete fusion workflows (filtering and smoothing)
- Observation operator integration
- Multiple time step processing
- Numerical stability with extreme values
- Output dimension consistency

**Key Tests:**
- Small scene fusion (complete workflow)
- Kalman filtering and smoothing integration
- Dimension checking
- Multi-time step processing
- Extreme value handling

## Running the Tests

### Run all tests:
```bash
cd /path/to/HyperSTARS.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Run specific test file:
```bash
julia --project=. -e 'include("test/test_structures.jl")'
```

### Run tests with verbose output:
```bash
julia --project=. test/runtests.jl
```

### Run tests in parallel (if applicable):
```bash
julia --project=. -p 4 -e 'using Pkg; Pkg.test()'
```

## Test Coverage

These tests aim to cover:

1. **API Contract**: All exported functions maintain stable interfaces
2. **Data Types**: Structures and custom types behave correctly
3. **Core Algorithms**: Kalman filtering/smoothing operations
4. **Numerical Properties**: Covariance matrices remain symmetric positive-definite
5. **Edge Cases**: NaN handling, extreme values, empty inputs
6. **Integration**: Multiple components work together correctly
7. **Dimensions**: Output shapes match expectations
8. **Reproducibility**: Same inputs produce same outputs

## Breaking Change Detection

These tests are specifically designed to catch:

- Changes to function signatures (parameter additions/removals)
- Changes to return types or dimensions
- Numerical algorithm changes affecting results
- Data structure field changes
- Type system modifications
- Sparse vs. dense matrix representation changes
- NaN and Inf handling changes
- Covariance matrix property violations

## Adding New Tests

When adding new functionality:

1. Create appropriate tests in the relevant test file
2. Verify tests pass before committing: `julia --project=. -e 'using Pkg; Pkg.test()'`
3. For new modules, create a new test file following the naming convention: `test_module_name.jl`
4. Include the new test file in `runtests.jl`
5. Document expected behaviors and edge cases in test comments

## Test Dependencies

The test suite uses:
- `Test` - Julia standard library for testing
- `HyperSTARS` - The package being tested
- `LinearAlgebra` - Matrix operations
- `Statistics` - Statistical functions
- `SparseArrays` - Sparse matrix support
- `Distributions` - For random data generation
- `Distances` - Distance metrics

All dependencies are already included in Project.toml.

## Continuous Integration

For CI/CD pipelines, use:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This will run all tests and report results.
