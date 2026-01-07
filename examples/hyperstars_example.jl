# --- 1. Package Imports and Environment Setup ---
# These are standard Julia packages required for various operations
using ArchGDAL # For geospatial data operations (though not directly used in this snippet's core logic, it's often part of a geospatial workflow)
using Rasters # Essential for working with raster data structures, aggregating, and spatial indexing
using Plots # For visualization, specifically creating a heatmap of the fused image
using Statistics # Provides `mean`, `std`, and other statistical functions
using JLD2 # For loading data stored in JLD2 format (Julia's binary data format)
using Distributed # Crucial for parallel processing using Julia's `pmap`
using MultivariateStats # For performing Principal Component Analysis (PCA)
using LinearAlgebra # Provides `Diagonal`, `I` (identity matrix), and other linear algebra utilities
using HyperSTARS

# Add worker processes for parallel execution.
# This line starts 8 additional Julia processes that can run code concurrently.
# The number of processes typically matches the number of CPU cores available for optimal performance.
addprocs(8)

# The `@everywhere` macro executes the subsequent expression on all available worker processes.
# This ensures that the `HyperSTARS` module and its functions are loaded into memory on all workers,
# which is necessary for `pmap` to execute `HyperSTARS` functions in parallel.

# Ensure LinearAlgebra and BLAS are loaded on all worker processes.
# BLAS.set_num_threads(1) can help prevent over-threading if `pmap` is already using many cores.
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
@everywhere using HyperSTARS

# --- 2. Data Loading ---
# This section loads synthetic hyperspectral data from a JLD2 file.
# The data simulates measurements from different satellite sensors: EMIT, HLS, and PACE.
# It also includes a `target_array` which represents the desired high-resolution grid.
filename = "./data/synthetic_emit_hls_pace_data.jld2"

begin
    # Open the JLD2 file in read mode.
    f = jldopen(filename)
    
    # Load multi-dimensional arrays representing satellite observations.
    # `hls_array`: HLS (Harmonized Landsat Sentinel) data, typically 30m resolution.
    # `emit_array`: EMIT (Earth Surface Mineral Dust Source Investigation) data, higher spectral resolution.
    # `pace_array`: PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) data, often coarser spatial resolution.
    # Dimensions are typically (rows, columns, wavelengths, time).
    hls_array = f["hls_array"]
    emit_array = f["emit_array"]
    pace_array = f["pace_array"] ## 4 day arrays, nx x ny x nw x nt (example dimensions)
    target_array = f["target_array"] # The desired high-resolution grid for fusion output

    # Load time information for each sensor's observations.
    # These are vectors of "times" (e.g., days since an epoch).
    hls_times = f["hls_times"] 
    emit_times = f["emit_times"]
    pace_times = f["pace_times"] ## vectors of nt "times" (or days)

    # Load spatial resolution (cell size) for each sensor.
    # `csize` is typically a 2-element vector `[res_x, res_y]`.
    # A negative value (e.g., for `res_y`) often indicates that the Y-coordinate decreases as row indices increase.
    hls_csize = f["hls_csize"] 
    emit_csize = f["emit_csize"] 
    pace_csize = f["pace_csize"] ## spatial resolution of pixels, negative means coordinate is descending

    # Load the spatial origin (top-left corner) of each sensor's image.
    # Then, convert this top-left origin to the **centroid** of the (1,1) pixel.
    # HyperSTARS functions often work with centroids for spatial referencing.
    hls_origin = f["hls_origin"] .+ hls_csize/2
    emit_origin = f["emit_origin"] .+ emit_csize/2
    pace_origin = f["pace_origin"] .+ pace_csize/2 ## spatial location of image "origin" (now centroid of (1,1) pixel)

    # Load Spectral Response Function (SRF) information.
    # `fwhm_pace`: Full Width at Half Maximum for PACE, used to model its Gaussian spectral response.
    # `hls_srf`: A dictionary containing pre-defined HLS spectral response weights and their corresponding wavelengths, for discrete multi-spectral bands.
    fwhm_pace = f["fwhm_pace"] ## vector of fwhm of gaussian spectral response kernel for each pace wavelength
    hls_srf = f["hls_srf"] ## Dict of HLS wavelengths and weights for discretized multispectral spectral response function

    # Load the center wavelengths for each spectral band of the sensors.
    hls_waves = f["hls_waves"] 
    emit_waves = f["emit_waves"]
    pace_waves = f["pace_waves"] ## wavelengths (center channel) for each spectral band

    #### Add 1km PACE data (simulated by aggregating existing PACE data)
    # This section creates a coarser (1km) version of the PACE data for testing multi-resolution fusion.
    pace1km_csize = [1000.0, -1000.0] # Define 1km cell size
    pace1km_origin = f["pace_origin"] .+ pace1km_csize/2 # Convert to centroid origin

    close(f) # Close the JLD2 file after loading all data.
end

# Adjust an HLS time point. This might be for aligning time series or specific analysis.
hls_times[1] = 1

#### Add 1km PACE data (continued)
# Convert the original `pace_array` into a `Raster` object for easy aggregation.
# `dims` specify the physical dimensions (X, Y, Z for wavelengths, Band for time) for the Raster.
pace_raster = Raster(pace_array, dims = (X(1:size(pace_array,1)), Y(1:size(pace_array,2)), Z(1:size(pace_array,3)), Band(1:size(pace_array,4))))
# Aggregate the `pace_raster` to a coarser resolution (e.g., by a factor of 2 in X and Y).
# `aggregate(mean, ...)` takes the mean of groups of pixels.
# The result `pace1km_array` will have half the spatial resolution of the original `pace_array`.
pace1km_array = Array{Float64}(aggregate(mean, pace_raster, (X(2), Y(2))))


# --- 3. Fusion Setup ---
# This section prepares the data and model parameters required by the HyperSTARS fusion functions.

### Spatial size of each instrument array
# These define the number of rows and columns for each sensor's grid.
hls_ndims = collect(size(hls_array)[1:2])
emit_ndims = collect(size(emit_array)[1:2])
pace_ndims = collect(size(pace_array)[1:2])
pace1km_ndims = collect(size(pace1km_array)[1:2])

### Get spectral basis functions using EMIT (PCA)
# Principal Component Analysis (PCA) is used to reduce the dimensionality of the spectral data.
# The resulting principal components (stored in `B`) will serve as the "latent spectral basis"
# for the fusion model, simplifying the spectral estimation.
emit_rt = reshape(permutedims(emit_array, (1,2,4,3)),(size(emit_array)[1]*size(emit_array)[2]*size(emit_array)[4],size(emit_array)[3]))'
# Reshape the EMIT data: (rows, cols, time, wavelengths) -> (rows*cols*time, wavelengths)
# Then transpose (`'`) to make wavelengths the rows and observations the columns, suitable for PCA.

emit_rt2 = emit_rt[:,.!vec(any(isnan, emit_rt; dims=1))]
# Filter out columns (observations) that contain `NaN` (Not a Number) values, as PCA cannot handle them.

mm = mean(emit_rt2, dims=2)[:] # Calculate the mean spectral signature (per wavelength)
sx = std(emit_rt2, dims=2)[:] # Calculate the standard deviation spectral signature (per wavelength)
Xt = (emit_rt2 .- mm) ./ sx # Normalize the data (mean-center and scale by std dev)

# Fit a PCA model to the normalized EMIT data.
# `pratio=0.995` means components are retained until 99.5% of the variance is explained.
pca = MultivariateStats.fit(PCA, Xt; pratio=0.995)
B = projection(pca) # `B` is the projection matrix (principal components or eigenvectors). This is our spectral basis.
vrs = principalvars(pca) # `vrs` are the variances explained by each principal component.
scs = predict(pca, Xt) # `scs` are the scores, i.e., the original data projected onto the principal components.

### Create geospatial structs for each instrument
# `InstrumentGeoData` stores static geospatial information (origin, cell size, dimensions)
# and also includes a `fidelity` flag:
# 0: highest spatial resolution (e.g., target resolution)
# 1: high resolution, but not the target resolution
# 2: coarse resolution
# The `rsr` (Relative Spectral Response) parameter is crucial for spectral conversion.
hls_geodata = InstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, hls_times, hls_waves, hls_srf)
emit_geodata = InstrumentGeoData(emit_origin, emit_csize, emit_ndims, 1, emit_times, emit_waves, [1.0]) # [1.0] implies a generic SRF for EMIT
pace_geodata = InstrumentGeoData(pace_origin, pace_csize, pace_ndims, 2, pace_times, pace_waves, fwhm_pace)
pace1km_geodata = InstrumentGeoData(pace1km_origin, pace1km_csize, pace1km_ndims, 2, pace_times, pace_waves, fwhm_pace)

## Create a vector of InstrumentGeoData for all instruments that will be used in the fusion.
# This specific list includes EMIT, HLS, and the aggregated 1km PACE data.
inst30m_geodata = [emit_geodata, hls_geodata, pace1km_geodata]

## Define moving window sizes for parallel processing
# The fusion process divides the entire scene into smaller, overlapping "windows" or "partitions".
# This allows for parallel computation and management of memory.
scf = 5 # `scf` (scaling factor) defines the size of each window in terms of target pixels (e.g., 5x5 target pixels per window).
nwindows = Int.(ceil.(hls_ndims./scf)) # Calculate the number of windows needed in X and Y directions for the entire scene.
target_ndims = hls_ndims # The ultimate prediction grid size is set to the HLS dimensions (e.g., 30m).
# `window30m_geodata` defines the *structure* of these moving windows:
# - `origin`: Centroid of the (1,1) window.
# - `cell_size`: The size of each window (e.g., 5 * 30m = 150m for a 5x5 window).
# - `ndims`: The number of windows in the scene.
window30m_geodata = InstrumentGeoData(hls_origin .+ (scf - 1)/2*hls_csize, scf*hls_csize, nwindows, 0, [1], emit_waves, 7.5) # The SRF here (7.5) seems arbitrary for the window struct; might be a placeholder.
# `target30m_geodata` defines the characteristics of the *output* grid:
# - `origin`: Top-left origin of the target grid.
# - `csize`: Cell size of the target grid (e.g., 30m).
# - `ndims`: Total dimensions of the output scene.
target30m_geodata = InstrumentGeoData(hls_origin, hls_csize, target_ndims, 0, [1], emit_waves, 7.5)

# Create `InstrumentData` structs for each sensor.
# These structs contain the actual measurement arrays, along with error biases (`bias`),
# uncertainty quantification (`uq` - here, 1e-5 variance), spatial resolution,
# observation times, a placeholder for coordinates (`[1,1]`), wavelengths, and SRF info.
hls_data = InstrumentData(hls_array, zeros(size(hls_array)[3:4]), 1e-5*ones(size(hls_array)[3:4]), abs.(hls_csize), hls_times, [1,1], hls_waves, hls_srf)
emit_data = InstrumentData(emit_array, zeros(size(emit_array)[3:4]), 1e-5*ones(size(emit_array)[3:4]), abs.(emit_csize), emit_times, [1,1], emit_waves, 1.0) # 1.0 implies simple SRF for EMIT
pace_data = InstrumentData(pace_array, zeros(size(pace_array)[3:4]), 1e-5*ones(size(pace_array)[3:4]), abs.(pace_csize), pace_times, [1,1], pace_waves, fwhm_pace)
pace1km_data = InstrumentData(pace1km_array, zeros(size(pace1km_array)[3:4]), 1e-5*ones(size(pace1km_array)[3:4]), abs.(pace1km_csize), pace_times, [1,1], pace_waves, fwhm_pace)

# Create a list of `InstrumentData` for the fusion.
data30m_list = [emit_data, hls_data, pace1km_data];

# Scale the spectral basis functions `B` by the standard deviation `sx`.
# This brings the basis functions back to the original reflectance scale if `B` was derived from normalized data.
Bs = B .* sx[:]

# Initialize the prior mean and variance arrays for the state vector.
# `pmean`: Prior mean array, initialized to zeros. Dimensions: (rows, cols, num_latent_components).
pmean = zeros((target_ndims...,size(B)[2])) 
# `pvar`: Prior variance array, initialized to ones (representing high initial uncertainty).
pvar = ones((target_ndims...,size(B)[2])) 

# Define spatial model parameters (`model_pars`) for each latent spectral component.
# First and second dimensions are dimensions of window_geodata
# Third dimension corresponds to latent components.
# Fourth dimension is: [amplitude/variance, spatial_range, nugget, smoothness_parameter].
# These parameters define the spatial covariance structure (e.g., Matern 3/2).
model_pars = ones((nwindows...,size(B)[2],4)) .* reshape([0.1,500,1e-10,1.5], (1,1,1,4))

# The third dimension of `model_pars` (amplitude/variance) is set to the principal variances (`vrs`) from PCA.
# This aligns the spatial variance with the variance explained by each latent spectral component.
for (i,x) in enumerate(vrs)
    model_pars[:,:,i,1] .= x
end

# --- 4. Run Data Fusion ---
# Execute the `scene_fusion_pmap` function, which orchestrates the parallel fusion across the entire scene.
# `@time` measures the execution time of this function.
@time fused_images, fused_sd_images = HyperSTARS.scene_fusion_pmap(data30m_list, # List of instrument data (measurements)
            inst30m_geodata, # List of instrument geospatial metadata
            window30m_geodata, # Geospatial definition of moving windows
            target30m_geodata, # Geospatial definition of the target output grid
            mm, ## Spectral mean (from PCA)
            pmean, ## Mean-zero prior mean for the latent state
            pvar, ## Diagonal prior variance for the latent state
            Bs, ## Scaled spectral basis function matrix (PCA projection)
            model_pars; # Spatial model parameters for each latent component
            nsamp=50, # Number of Basic Area Units (BAUs) to subsample within a window for efficiency
            window_buffer = 3, # Number of buffer pixels around each window to avoid edge effects
            target_times = 1:4, # Time steps for which to produce fused output (e.g., day 1 to day 4)
            smooth = false, # Flag: `false` means only Kalman filtering (forward pass); `true` would also apply smoothing (backward pass)
            spatial_mod = HyperSTARS.mat32_corD, # Spatial covariance function (Matern 3/2 with precomputed distances)
            obs_operator = HyperSTARS.unif_weighted_obs_operator_centroid, # Observation operator (uniform weighting based on centroid overlap)
            state_in_cov=false, # Flag: `false` means process noise covariance is static; `true` would make it adaptive
            window_radius=1000.0); # Parameter defining spatial size of neighborhood to sample neighbors

## Remove worker processes after parallel computation is complete.
# This frees up system resources.
rmprocs(workers())

# --- 5. Visualization ---
# Visualize one of the fused images.
k=1 # Select the first latent spectral component.
# `fused_images[:,:,k,4]` selects all rows and columns, for the `k`-th latent component, at the 4th time step.
# `heatmap` creates a 2D plot where colors represent values, useful for visualizing raster data.
heatmap(fused_images[:,:,k,4],title="Fused Image, Day 4)",size=(600,600))
