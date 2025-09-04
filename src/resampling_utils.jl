using Interpolations # For interpolation methods, especially in RSR conversion
using Distances # For calculating pairwise distances (e.g., Euclidean, Squared Euclidean)
using LinearAlgebra # For basic linear algebra operations like identity matrix

"""
    unif_weighted_obs_operator_centroid(sensor, target, sensor_res)

Constructs a uniform-weighted observation operator matrix (`H`) that maps from
target grid cells to sensor observations based on spatial overlap.
If `sensor` and `target` coordinates are identical, it returns an identity matrix.
Otherwise, it assigns a uniform weight (1.0) to target cells whose centroids fall
within the sensor's spatial resolution window, then normalizes these weights.

# Arguments
- `sensor::AbstractArray{T}`: Spatial coordinates of the sensor observations (e.g., `n_sensor x 2`).
- `target::AbstractArray{T}`: Spatial coordinates of the target grid cells (e.g., `n_target x 2`).
- `sensor_res::AbstractVector{T}`: A 2-element vector `[res_x, res_y]` representing the
  sensor's spatial resolution in x and y dimensions.

# Returns
- `H::SparseMatrixCSC{Float64}`: A sparse observation matrix where `H[i, j]` is the
  weight of target cell `j` contributing to sensor observation `i`.

# Logic
1.  **Identity Case**: If sensor and target coordinate arrays are identical, it means
    a 1:1 mapping exists, so a sparse identity matrix is returned.
2.  **Overlap Calculation**:
    - `d1`: A boolean matrix indicating if the x-coordinate difference between
      each sensor observation and each target cell centroid is within half the sensor's x-resolution.
    - `d2`: Similar to `d1`, but for the y-coordinates and y-resolution.
    - `H = d1 .* d2`: The element-wise product identifies target cells whose centroids
      fall within the rectangular footprint defined by `sensor_res` around each sensor observation.
3.  **Normalization**: Each row of `H` is normalized by its sum. This ensures that the
    weights for a given sensor observation sum to 1, meaning all contributing target
    cells collectively represent the full sensor observation.
"""
function unif_weighted_obs_operator_centroid(sensor::AbstractArray{T}, target::AbstractArray{T}, sensor_res::AbstractVector{T}) where T<:Real
    if sensor == target # Check if sensor and target coordinates are identical
        n = size(target,1) # Get the number of target cells
        return sparse(1.0I, n, n) # Return a sparse identity matrix (1:1 mapping)
    else
        # Calculate boolean matrices indicating if target cells are within half the sensor resolution
        # for x and y dimensions independently.
        # pairwise computes distances between `sensor` (rows/columns) and `target` (rows/columns).
        # Here, `sensor[:, 1]'` transposes the x-coordinates to a row vector for pairwise comparison.
        # The comparison `<=` `sensor_res[1] / 2` checks if the absolute difference is within the half-resolution.
        d1 = pairwise(Euclidean(1e-12), sensor[:, 1]', target[:, 1]', dims=2) .<= sensor_res[1] / 2
        d2 = pairwise(Euclidean(1e-12), sensor[:, 2]', target[:, 2]', dims=2) .<= sensor_res[2] / 2

        # The element-wise product `H = d1 .* d2` creates a boolean matrix where `true`
        # indicates that the target cell centroid falls within the sensor pixel's footprint.
        H = d1 .* d2
        
        # Normalize each row of H by its sum. This converts counts of overlapping cells
        # into weights, ensuring that the sum of weights for each sensor observation is 1.
        # `broadcast(/, H, sum(H, dims=2))` performs element-wise division of H by its row sums.
        return sparse(broadcast(/, H, sum(H, dims=2)))
    end
end

"""
    gauss_weighted_obs_operator(sensor, target, res; scale=1.0, p=2.0)

Constructs a Gaussian-weighted observation operator matrix (`H`).
The weights are based on the (scaled) squared Euclidean distance between
sensor observation locations and target grid cell locations, following a Gaussian kernel.
A cutoff threshold (`exp(-0.5 * p^2)`) is applied to set very small weights to zero,
resulting in a sparse matrix.

# Arguments
- `sensor`: Spatial coordinates of the sensor observations.
- `target`: Spatial coordinates of the target grid cells.
- `res`: Sensor's spatial resolution (e.g., cell size).
- `scale::Real = 1.0`: Scaling factor for the resolution, effectively adjusting the kernel's spread.
- `p::Real = 2.0`: Parameter defining the cutoff distance (in units of scaled resolution).
  Weights below `exp(-0.5 * p^2)` are set to zero.

# Returns
- `H::SparseMatrixCSC{Float64}`: A sparse observation matrix with Gaussian weights.

# Logic
1.  **Scaled Distance Calculation**: `pairwise(SqEuclidean(...))` computes the squared
    Euclidean distances. The coordinates are scaled by `res ./ scale` before distance
    calculation, making the kernel's influence dependent on resolution.
2.  **Gaussian Kernel Application**: `exp.(-0.5 * ...)` applies the Gaussian (squared exponential) kernel formula.
3.  **Thresholding**: Any weight in `H` that falls below a certain threshold (defined by `p`)
    is set to zero. This helps in creating a sparse matrix by removing negligible connections.
4.  **Normalization**: Each row of `H` is normalized by its sum.
"""
function gauss_weighted_obs_operator(sensor, target, res; scale=1.0, p = 2.0)
    # Compute the squared Euclidean distances between sensor locations and target locations.
    # `sensor ./ transpose(res ./ scale)` scales the sensor coordinates by `res/scale`.
    # This effectively normalizes distances by the resolution, making the kernel spatially adaptive.
    H = exp.(-0.5 * pairwise(SqEuclidean(1e-12), sensor ./ transpose(res ./ scale), target ./ transpose(res ./ scale), dims=1))

    # Apply a cutoff: values in H below a certain threshold are set to zero.
    # This promotes sparsity by removing connections that are too weak.
    # The threshold is determined by `p`, which represents a multiple of the scaled resolution.
    H[H.<exp(-0.5 * p^2)] .= 0

    # Normalize each row of H by its sum, similar to `unif_weighted_obs_operator_centroid`.
    return sparse(broadcast(/, H, sum(H, dims=2)))
end

"""
    uniform_obs_operator_indices(target, target_cell_size, bau_origin, bau_cell_size, n_dims, bau_sub_inds)

Constructs a uniform observation operator matrix based on grid indices rather than
direct coordinates. It maps target pixels (defined by `target` coordinates and `target_cell_size`)
to Basic Area Units (BAUs) within a larger grid, considering the BAU grid's origin,
cell size, and dimensions.

# Arguments
- `target`: Spatial coordinates of the target grid cells (e.g., `n_target x 2`).
- `target_cell_size`: Size of individual target grid cells.
- `bau_origin`: Origin `[x, y]` of the overall BAU grid.
- `bau_cell_size`: Size `[res_x, res_y]` of individual BAU cells.
- `n_dims`: Dimensions `[n_x, n_y]` of the overall BAU grid.
- `bau_sub_inds`: Linear indices of a subset of BAUs to consider (e.g., from a buffered window).

# Returns
- `H::SparseMatrixCSC{Float64}`: A sparse observation matrix `p x length(bau_sub_inds)`
  where `p` is the number of target pixels. `H[i,j]` indicates the weight of BAU `j`
  contributing to target pixel `i`. If no valid indices are found, returns a zero matrix.

# Logic
1.  **Target Extents**: Calculate the min/max x and y coordinates for each target pixel.
2.  **Map to BAU Indices**: Use `find_nearest_ind` to map the target pixel extents to
    the corresponding column/row indices in the BAU grid.
3.  **Index Clamping**: Ensure that the determined BAU indices are within the valid `n_dims`
    range (clamped to 1 and `n_dims`).
4.  **Sparse Matrix Construction**:
    - `irngs`, `jrngs`: Ranges of BAU indices covered by each target pixel.
    - `II`: Linear indices of the entire BAU grid.
    - `cols`: Collects linear indices of all BAUs covered by *any* target pixel.
    - `rows`: Creates corresponding row indices for the `H` matrix (each target pixel maps to its contributing BAUs).
    - `zs`: Calculates the uniform weight (1 / number of BAUs covered by that target pixel).
    - `sparse(...)`: Constructs the sparse observation matrix `H`.
5.  **Subset to Relevant BAUs**: The final `H` matrix is subsetted to `bau_sub_inds` to
    only include BAUs relevant to the current processing window.
"""
function uniform_obs_operator_indices(target, target_cell_size, bau_origin, bau_cell_size, n_dims, bau_sub_inds)
    oi, oj = bau_origin # Origin (x, y) of the BAU grid
    c, r = bau_cell_size # Cell size (resolution) of the BAU grid

    # Calculate the x- and y-extents (min and max coordinates) for each target pixel.
    # `target[:,1]*[1 1]` creates a matrix where each row is [x, x].
    # `[-0.5+1e-10 0.5-1e-10]*target_cell_size[1]` adds/subtracts half the cell size to get boundaries.
    ext_ti = target[:,1]*[1 1] .+ [-0.5+1e-10 0.5-1e-10]*target_cell_size[1] 
    ext_tj = target[:,2]*[1 1] .+ [-0.5+1e-10 0.5-1e-10]*target_cell_size[2] 

    # Find the nearest integer indices in the BAU grid for the target pixel extents.
    # `find_nearest_ind` is assumed to convert a continuous coordinate to a discrete grid index.
    is = find_nearest_ind.(ext_ti, oi, c) # x-indices in BAU grid
    js = find_nearest_ind.(ext_tj, oj, r) # y-indices in BAU grid
    p = size(target,1) # Number of target pixels

    # Check if any valid indices were found and are within the BAU grid dimensions.
    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        # Clamp indices to be within valid grid bounds [1, n_dims].
        is[is .< 1] .= 1
        js[js .< 1] .= 1
        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        # Create ranges of BAU indices (rows and columns) covered by each target pixel.
        irngs = (:).(is[:,1],is[:,2]) # x-index ranges (start:end)
        jrngs = (:).(js[:,1],js[:,2]) # y-index ranges (start:end)

        # Create a LinearIndices object for easy conversion from (i,j) to linear index.
        II = LinearIndices((1:n_dims[1],1:n_dims[2]))

        # For each target pixel, get the linear indices of all BAUs it covers.
        cols = [II[irngs[i], jrngs[i]][:] for i in eachindex(irngs)] 
        ns = size.(cols,1) # Number of BAUs covered by each target pixel
        
        # Create row indices for the sparse H matrix. `inverse_rle` is assumed to
        # expand `1:p` (target pixel index) by `ns` times, meaning each target pixel
        # will have `ns[i]` entries in the sparse matrix, mapping to its `ns[i]` covered BAUs.
        rows = inverse_rle(1:p,ns)
        
        # Calculate the weight for each target-BAU connection. It's uniform: 1 / (number of BAUs covered).
        zs = inverse_rle(1 ./ns, ns)

        # Construct the sparse observation matrix H.
        # `p` is number of target pixels (rows), `n_dims[1]*n_dims[2]` is total BAUs (columns).
        H = sparse(rows, vcat(cols...), zs, p, n_dims[1]*n_dims[2])
    
        # Return H, subsetted to only the BAUs specified by `bau_sub_inds`.
        return H[:,bau_sub_inds]
    else
        # If no valid indices or overlaps, return a zero sparse matrix.
        return spzeros(p, length(bau_sub_inds))
    end
end

"""
    rsr_conv_matrix(rsr, inst_waves, target_waves)

Constructs a Relative Spectral Response (RSR) convolution matrix when `rsr` is provided as an array
of FWHM (Full Width at Half Maximum) values for each instrument band.
This matrix maps reflectance from the `target_waves` (fine grid) to the `inst_waves` (instrument bands),
effectively simulating how the instrument observes the target spectrum.

# Arguments
- `rsr::AbstractArray{<:Real}`: A vector or array where each element represents the FWHM
  for an instrument's spectral band.
- `inst_waves::AbstractVector{<:Real}`: A vector of wavelengths corresponding to the
  instrument's spectral bands.
- `target_waves::AbstractVector{<:Real}`: A vector of wavelengths for the target (finer) spectral grid.

# Returns
- `Ks::SparseMatrixCSC{Float64}`: A sparse convolution matrix where `Ks[i, j]`
  is the weight of `target_waves[j]` contributing to `inst_waves[i]`.

# Logic
1.  **Squared Euclidean Distances**: Computes `dd`, the squared Euclidean distances
    between all `inst_waves` and `target_waves`.
2.  **Squared Half-Width at Half Maximum**: `s2p` calculates the squared half-width
    for each instrument band, derived from the FWHM assuming a Gaussian shape.
3.  **Gaussian Weighting**: `K = exp.(-dd ./ s2p)` applies a Gaussian kernel
    to these distances, creating a weighting matrix based on spectral proximity.
4.  **Normalization**: Each row of `K` (corresponding to an instrument band) is
    normalized by its sum, ensuring weights for each instrument band sum to 1.
"""
function rsr_conv_matrix(rsr::AbstractArray{<:Real}, inst_waves::AbstractVector{<:Real}, target_waves::AbstractVector{<:Real})

    # Compute squared Euclidean distances between all pairs of instrument wavelengths and target wavelengths.
    dd = Distances.pairwise(SqEuclidean(1e-12), inst_waves, target_waves)

    # Calculate squared half-width from Full Width at Half Maximum (FWHM).
    # Assuming a Gaussian RSR, FWHM = 2 * sqrt(2 * log(2)) * sigma, so sigma = FWHM / (2 * sqrt(2 * log(2))).
    # Here, `s2p` is (sigma^2) which is the variance term in the Gaussian exponent.
    s2p = (rsr ./ (2 * sqrt(2 * log(2)))).^2
    
    # Apply a Gaussian (or squared exponential) kernel to the distances, scaled by `s2p`.
    # This creates the raw convolution weights.
    K = exp.(-dd ./ s2p)
    
    # Normalize each row of K by its sum.
    # This ensures that the sum of weights for each instrument band (row) is 1,
    # representing a proper convolution.
    Ks = K ./sum(K, dims=2)

    return sparse(Ks) # Return as a sparse matrix for efficiency
end

"""
    rsr_conv_matrix(rsr, inst_waves, target_waves)

Constructs a Relative Spectral Response (RSR) convolution matrix when `rsr` is provided as a dictionary
containing pre-defined RSR curves (e.g., discrete weights at specific wavelengths).
If `target_waves` matches the RSR's internal knot points, it uses the RSR directly.
Otherwise, it interpolates the RSR curves to the `target_waves`.

# Arguments
- `rsr::Dict`: A dictionary containing RSR information:
    - `rsr[:w]`: Wavelengths (knot points) at which the RSR values (`rsr[:rsr]`) are defined.
    - `rsr[:rsr]`: An array of RSR values, where each row is an instrument band's RSR curve.
- `inst_waves::AbstractVector{<:Real}`: A vector of wavelengths corresponding to the
  instrument's spectral bands (used for `size(Ki)[1]`, but not directly for interpolation here).
- `target_waves::AbstractVector{<:Real}`: A vector of wavelengths for the target (finer) spectral grid.

# Returns
- `Ks::SparseMatrixCSC{Float64}`: A sparse convolution matrix.

# Logic
1.  **Direct Use Case**: If `target_waves` is identical to `rsr[:w]` (the RSR knot points),
    the pre-defined `rsr[:rsr]` is used directly after row-wise normalization.
2.  **Interpolation Case**: If wavelengths do not match, `Interpolations.jl` is used:
    - For each instrument band's RSR curve in `rsr[:rsr]`, an `interpolate` object is created
      using the `rsr[:w]` knot points and `Gridded(Linear())` interpolation.
    - `extrapolate(..., 0)` extends the interpolation to return 0 for points outside the defined range.
    - This interpolation function is then applied to `target_waves` to get the RSR values
      at the desired target wavelengths.
3.  **Normalization**: The interpolated (or direct) RSR matrix is then normalized row-wise.
"""
function rsr_conv_matrix(rsr::Dict, inst_waves::AbstractVector{<:Real}, target_waves::AbstractVector{<:Real})

    knts = rsr[:w] # Wavelength knot points where RSR is defined
    Ki = rsr[:rsr] # RSR values (each row is an instrument band's RSR curve)

    if target_waves == knts # If target wavelengths match the RSR knot points
        Ks = sparse(Ki ./sum(Ki, dims=2)) # Use RSR directly, normalize and make sparse
    else # Otherwise, interpolate the RSR to the target wavelengths
        K = zeros(size(Ki)[1],size(target_waves)[1]) # Pre-allocate matrix for interpolated RSR
        for i in 1:size(Ki)[1] # Iterate through each instrument band's RSR curve
            # Create an interpolation object for the current RSR curve (Ki[i,:])
            # `Gridded(Linear())` specifies linear interpolation on a grid.
            # `extrapolate(..., 0)` means values outside the defined `knts` range will be 0.
            itp = extrapolate(interpolate((knts,), Ki[i,:], Gridded(Linear())),0)
            
            # Apply the interpolation to the `target_waves` to get RSR values at desired wavelengths.
            K[i,:] = itp(target_waves)
        end
        # Normalize each row of the interpolated RSR matrix and make it sparse.
        Ks = sparse(K ./sum(K, dims=2))

    end
  
    return Ks
end
