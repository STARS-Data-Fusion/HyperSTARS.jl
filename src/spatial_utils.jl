using Rasters # For working with raster data structures and their properties
using Statistics # For statistical operations like mean, median
using StatsBase # For sampling functions
using Random # For random number generation
using LinearAlgebra # For identity matrix (I)

############ Functions for spatial indexing and transformations ############
## Conventions:
## - `i` is row index, `j` is column index, consistent with `rast[i,j,...]`
## - `c` is cell size along rows (often X-dimension), `r` is cell size along columns (often Y-dimension)
## - If `c` or `r` is negative, it indicates that the spatial coordinate decreases as the index increases (e.g., origin is top-left, Y increases downwards).
## - `oi, oj` refer to the origin geolocation (spatial coordinates) at `i=1, j=1`.
## - `si, sj` refer to the spatial centroid for cell `i,j` (assuming `i,j` refers to the upper-left corner of the cell).

"""
    find_nearest_ij(target, origin, cell_size)

Finds the `[i, j]` (row, column) grid indices for a given `target` spatial coordinate,
relative to a grid defined by its `origin` and `cell_size`.

# Arguments
- `target::AbstractVector{T}`: A 2-element vector `[x, y]` representing the spatial coordinate.
- `origin::AbstractVector{T}`: A 2-element vector `[origin_x, origin_y]` representing the
  spatial coordinates of the grid's (1,1) cell's origin.
- `cell_size::AbstractVector{T}`: A 2-element vector `[cell_size_x, cell_size_y]` representing the
  size of each grid cell in x and y dimensions.

# Returns
- `[i, j]::Vector{Int64}`: The integer row and column indices.

# Logic
The formula `floor((coordinate - origin) / cell_size + 1)` calculates the 1-based index
by determining how many `cell_size` units away the `target` coordinate is from the `origin`.
`floor` ensures integer indices.
"""
function find_nearest_ij(target::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}) where T <: Real
    oi, oj = origin    # Origin coordinates (x, y)
    c, r = cell_size  # Cell sizes (x-resolution, y-resolution)
    ti,tj = target    # Target coordinates (x, y)

    # Calculate row index: (target_x - origin_x) / cell_size_x + 1
    i = Int64(floor((ti - oi) / c + 1))
    # Calculate column index: (target_y - origin_y) / cell_size_y + 1
    j = Int64(floor((tj - oj) / r + 1))
    return [i, j]
end

"""
    find_nearest_ij_multi(target, origin, cell_size, n_dims)

Finds the `[i, j]` (row, column) grid indices for multiple `target` spatial coordinates.
It filters out indices that fall outside the defined grid dimensions (`n_dims`).

# Arguments
- `target::AbstractVector{T}`: An `N x 2` matrix where each row is a spatial coordinate `[x, y]`.
- `origin::AbstractVector{T}`: A 2-element vector `[origin_x, origin_y]` of the grid's origin.
- `cell_size::AbstractVector{T}`: A 2-element vector `[cell_size_x, cell_size_y]` of grid cell sizes.
- `n_dims::AbstractVector{<:Real}`: A 2-element vector `[num_rows, num_cols]` defining the
  dimensions of the grid.

# Returns
- `df2::Matrix{Int64}`: A matrix of valid `[i, j]` indices, where each row is a pair of indices.
  Rows corresponding to coordinates outside `n_dims` are excluded.
"""
function find_nearest_ij_multi(target::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}) where T <: Real
    oi, oj = origin   # Origin coordinates
    c, r = cell_size # Cell sizes

    # Calculate row indices for all target points
    i = Int64.(floor.((target[:,1] .- oi) ./ c .+ 1))
    # Calculate column indices for all target points
    j = Int64.(floor.((target[:,2] .- oj) ./ r .+ 1))

    # Combine i and j into a single matrix.
    df = [i j]
    
    # Filter out rows where indices are outside the valid grid dimensions [1, n_dims].
    df2 = df[(1 .<= i .<= n_dims[1]) .& (1 .<= j .<= n_dims[2]),:]
    return df2
end

"""
    find_nearest_ind(target, origin, cell_size)

Finds the nearest 1D grid index for a single `target` coordinate along one dimension.

# Arguments
- `target::T`: The single spatial coordinate.
- `origin::T`: The origin coordinate of the 1D grid.
- `cell_size::T`: The cell size along this dimension.

# Returns
- `ind::Int`: The integer 1-based index.
"""
function find_nearest_ind(target::T, origin::T, cell_size::T) where T <: Real
    # Formula similar to `find_nearest_ij` but for a single dimension.
    ind = Int(floor((target - origin) / cell_size + 1))
    return ind
end

"""
    find_all_ij_ext(ext_ti, ext_tj, origin, cell_size, n_dims; inclusive=false)

Finds all `[i, j]` grid indices (rows and columns) that fall within a given
rectangular spatial extent defined by `ext_ti` (x-extent) and `ext_tj` (y-extent).
It considers the grid's `origin`, `cell_size`, and `n_dims`.

# Arguments
- `ext_ti::AbstractVector{T}`: A 2-element vector `[min_x, max_x]` defining the x-extent.
- `ext_tj::AbstractVector{T}`: A 2-element vector `[min_y, max_y]` defining the y-extent.
- `origin::AbstractVector{T}`: The grid's origin `[origin_x, origin_y]`.
- `cell_size::AbstractVector{T}`: The grid's cell sizes `[cell_size_x, cell_size_y]`.
- `n_dims::AbstractVector{<:Real}`: The grid's dimensions `[num_rows, num_cols]`.
- `inclusive::Bool = false`: If `true`, the extent includes the upper boundary. If `false`,
  a small offset is applied to exclude the upper boundary for open intervals (e.g., `[min, max)`).

# Returns
- `unique(hcat(iss,jss),dims=1)::Matrix{Int64}`: A unique matrix of `[i, j]` pairs
  within the extent and valid grid boundaries. Returns an empty array if no valid indices are found.
"""
function find_all_ij_ext(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}; inclusive=false) where T <: Real
    oi, oj = origin   # Origin coordinates
    c, r = cell_size # Cell sizes

    # Adjust extents if `inclusive` is false, by subtracting a tiny value (1e-9)
    # from the upper bound to ensure an open interval `[min, max)`.
    # `sign(c)` handles cases where cell_size is negative (decreasing coordinate with increasing index).
    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    # Find the nearest grid indices for the extent boundaries.
    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    # Check if any part of the calculated index range is within the grid dimensions.
    # This prevents processing empty or completely out-of-bounds extents.
    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        # Clamp indices to ensure they are within valid grid boundaries [1, n_dims].
        is[is .< 1] .= 1
        js[js .< 1] .= 1
        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        # Generate all integer indices within the clamped ranges.
        # `repeat` creates combinations of i and j indices covering the rectangular area.
        iss = repeat(is[1]:is[2], inner=js[2]-js[1]+1, outer=[1]) # All row indices repeated
        jss = repeat(js[1]:js[2], inner=[1], outer=is[2]-is[1]+1) # All column indices repeated

        # Combine into [i j] pairs and ensure uniqueness.
        return unique(hcat(iss,jss),dims=1)
    else
        # Return an empty array if no valid indices are found.
        return Array{Int64}(undef, 0, 0)
    end
end

"""
    find_all_bau_ij(target, target_cell_size, bau_origin, bau_cell_size)

Finds all Basic Area Unit (BAU) `[i, j]` grid indices that are covered by a single
`target` pixel. This function is likely used when mapping a higher-resolution
`target` pixel to a lower-resolution BAU grid.

# Arguments
- `target`: A 2-element vector `[x, y]` representing the centroid of the target pixel.
- `target_cell_size`: A 2-element vector `[res_x, res_y]` of the target pixel's size.
- `bau_origin`: The origin `[x, y]` of the BAU grid.
- `bau_cell_size`: The cell size `[res_x, res_y]` of the BAU grid.

# Returns
- `unique(hcat(iss,jss),dims=1)::Matrix{Int64}`: A unique matrix of `[i, j]` pairs
  of BAUs covered by the target pixel.
"""
function find_all_bau_ij(target, target_cell_size, bau_origin, bau_cell_size)
    oi, oj = bau_origin # BAU grid origin
    c, r = bau_cell_size # BAU cell sizes

    # Calculate the spatial extent of the target pixel (from its centroid and size).
    # `[-0.5,0.5-1e-9]*target_cell_size[1]` extends half a cell size in both directions
    # with a slight adjustment to ensure open interval on the upper bound.
    ext_ti = target[1] .+ [-0.5,0.5-1e-9]*target_cell_size[1]
    ext_tj = target[2] .+ [-0.5,0.5-1e-9]*target_cell_size[2]

    # Find the nearest BAU grid indices for the target pixel's extent.
    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    # Determine the range of BAU indices. Handle cases where the extent might map
    # to a single index or a range.
    if size(is)[1] == 1
        is2 = is
    else
        is2 = is[1]:is[2]
    end
    if size(js)[1] == 1
        js2 = js
    else
        js2 = js[1]:js[2]
    end

    # Generate all combinations of BAU indices covered by the target pixel.
    iss = repeat(is2, inner=size(js2)[1], outer=[1])
    jss = repeat(js2, inner=[1], outer=size(is2)[1])

    # Return unique [i j] pairs.
    return unique(hcat(iss,jss),dims=1)
end

"""
    find_all_bau_ij_multi(target, target_cell_size, bau_origin, bau_cell_size, n_dims)

Finds all Basic Area Unit (BAU) `[i, j]` grid indices that are covered by
multiple `target` pixels. This is a vectorized version of `find_all_bau_ij`.

# Arguments
- `target`: An `N x 2` matrix of spatial coordinates for multiple target pixels.
- `target_cell_size`: Size of individual target grid cells.
- `bau_origin`: Origin `[x, y]` of the BAU grid.
- `bau_cell_size`: Cell size `[res_x, res_y]` of the BAU grid.
- `n_dims`: Dimensions `[n_x, n_y]` of the overall BAU grid.

# Returns
- `stack(unique(tpls))'::Matrix{Int64}`: A unique matrix of `[i, j]` pairs
  of BAUs covered by the target pixels. Returns an empty array if no valid indices are found.
"""
function find_all_bau_ij_multi(target, target_cell_size, bau_origin, bau_cell_size, n_dims)
    oi, oj = bau_origin # BAU grid origin
    c, r = bau_cell_size # BAU cell sizes

    # Calculate x- and y-extents for each target pixel (similar to `find_all_bau_ij`).
    ext_ti = target[:,1]*[1 1] .+ [-0.5 0.5-1e-6]*target_cell_size[1] 
    ext_tj = target[:,2]*[1 1] .+ [-0.5 0.5-1e-6]*target_cell_size[2] 

    # Find the nearest BAU grid indices for the extent boundaries of all target pixels.
    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    # Check for valid indices within the grid dimensions.
    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        # Clamp indices to valid grid bounds.
        is[is .< 1] .= 1
        js[js .< 1] .= 1
        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        # Create ranges of BAU indices (rows and columns) for each target pixel.
        irngs = (:).(is[:,1],is[:,2])
        jrngs = (:).(js[:,1],js[:,2])

        # Use `Iterators.product` to generate combinations of indices for each target pixel.
        # `Iterators.flatten` then flattens these into a single iterator of (i,j) tuples.
        t1 = Iterators.product.(irngs,jrngs)
        tpls = Iterators.flatten(t1)
        
        # Convert tuples to a matrix and ensure uniqueness. `stack` converts iterator of tuples to array.
        return stack(unique(tpls))'
    else
        # Return an empty array if no valid indices are found.
        return Array{Int64}(undef, 0, 0)
    end
end

"""
    subsample_bau_ij(ext_ti, ext_tj, origin, cell_size, n_dims; nsamp=100, inclusive=false)

Subsamples a specified number of Basic Area Unit (BAU) `[i, j]` grid indices
within a given rectangular extent. If `nsamp` equals the total number of BAUs
in the extent, all are returned. Otherwise, it samples `nsamp` BAUs with replacement.

# Arguments
- `ext_ti`, `ext_tj`: X and Y spatial extents `[min, max]`.
- `origin`, `cell_size`, `n_dims`: Grid definition parameters.
- `nsamp::Int = 100`: The number of BAU samples to generate.
- `inclusive::Bool = false`: If `true`, the extent includes the upper boundary.

# Returns
- `unique(hcat(iss,jss),dims=1)::Matrix{Int64}`: A unique matrix of sampled `[i, j]` pairs.
  Returns an empty array if no valid indices are found.
"""
function subsample_bau_ij(ext_ti, ext_tj, origin, cell_size, n_dims; nsamp=100, inclusive=false)

    oi, oj = origin # Grid origin
    c, r = cell_size # Cell sizes

    # Adjust extents based on `inclusive` flag.
    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    # Find grid indices for the extent boundaries.
    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    # Check for valid indices within grid dimensions.
    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        # Clamp indices to valid grid bounds.
        is[is .< 1] .= 1
        js[js .< 1] .= 1
        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        # Calculate dimensions of the sampled area.
        ni = is[2]-is[1] + 1 # Number of rows in the sampled area
        nj = js[2]-js[1] + 1 # Number of columns in the sampled area

        if nsamp == ni*nj # If requested samples equals total BAUs in extent, return all.
            iss = repeat(is[1]:is[2], inner=[nj], outer=[1])
            jss = repeat(js[1]:js[2], inner=[1], outer=[ni])
        else # Otherwise, randomly sample `nsamp` BAUs.
            # `minimum([nsamp, ni*nj])` ensures `np` doesn't exceed available BAUs.
            np = minimum([nsamp, ni*nj]) 
            
            # Sample row and column indices with replacement.
            iss = sample(is[1]:is[2], np, replace=true)
            jss = sample(js[1]:js[2], np, replace=true)
        end

        # Return unique sampled [i j] pairs.
        return unique(hcat(iss,jss),dims=1)
    else
        # Return empty array if no valid indices.
        return Array{Int64}(undef, 0, 0)
    end
end

"""
    get_sij_from_ij(ij_array, origin, cell_size)

Converts `[i, j]` (row, column) grid indices to `[x, y]` spatial centroid coordinates.

# Arguments
- `ij_array`: An `N x 2` matrix where each row is an `[i, j]` grid index.
- `origin`: The grid's origin `[origin_x, origin_y]`.
- `cell_size`: The grid's cell sizes `[cell_size_x, cell_size_y]`.

# Returns
- `xy_coords::Matrix{Float64}`: An `N x 2` matrix of `[x, y]` centroid coordinates.

# Logic
The centroid is calculated by adding half the cell size to the origin of the cell's top-left corner.
Formula: `origin + cell_size * (index - 1) + cell_size / 2`
"""
function get_sij_from_ij(ij_array, origin, cell_size)
    # Transpose origins and cell sizes for broadcasting with `ij_array`.
    # `ij_array .- [1.,1.]'` converts 1-based indices to 0-based, then centers them.
    # `cell_size' ./ 2` adds half a cell size to get the centroid.
    origin' .+ cell_size' .* (ij_array .- [1.,1.]') .+ cell_size' ./ 2
end

"""
    get_origin_raster(rast)

Retrieves the spatial origin (top-left corner) of a `Raster` object.

# Arguments
- `rast::Raster`: The input Raster object.

# Returns
- `origin::Vector{Float64}`: A 2-element vector `[origin_x, origin_y]`.
"""
function get_origin_raster(rast::Raster)
    # `rast.dims[1].val[1]` is the first coordinate of the first dimension (typically X).
    # `rast.dims[2].val[1]` is the first coordinate of the second dimension (typically Y).
    return [rast.dims[1].val[1],rast.dims[2].val[1]]
end

"""
    bbox_from_ul(target, cell_size)

Calculates the bounding box coordinates given an upper-left corner and cell size.
This assumes the `target` is the upper-left (UL) coordinate of a cell.

# Arguments
- `target`: A 2-element vector `[x_ul, y_ul]` representing the upper-left corner.
- `cell_size`: A 2-element vector `[res_x, res_y]` representing the cell dimensions.

# Returns
- `bbox::Matrix{Float64}`: A `2 x 2` matrix representing the bounding box, where
  `bbox[1,:]` is `[min_x, max_x]` and `bbox[2,:]` is `[min_y, max_y]`.

# Logic
The bounding box spans from the upper-left coordinate to `upper_left + cell_size`.
`hcat(zeros(2),cell_size)` creates a `2 x 2` matrix where the first column is `[0,0]`
and the second column is `[cell_size_x, cell_size_y]`.
Adding `target` shifts this rectangle to the correct position.
"""
function bbox_from_ul(target,cell_size)
    # The bounding box extends from the target (UL) point by the cell size.
    # `hcat(zeros(2),cell_size)` creates a matrix like `[0 res_x; 0 res_y]`.
    # Adding `target` shifts this to the correct absolute coordinates.
    return hcat(zeros(2),cell_size) .+ target
end

"""
    extent_from_xy(xy_dat, cell_size)

Calculates the overall spatial extent (bounding box) of a set of `xy_dat` points,
considering the `cell_size` to infer the full cell coverage.

# Arguments
- `xy_dat`: An `N x 2` matrix of `[x, y]` spatial coordinates.
- `cell_size`: A 2-element vector `[res_x, res_y]` of grid cell sizes.

# Returns
- `es::Matrix{Float64}`: A `2 x 2` matrix `[min_x max_x; min_y max_y]` representing the
  overall spatial extent.
"""
function extent_from_xy(xy_dat, cell_size)
    # Find the minimum and maximum coordinates along each dimension.
    mms = extrema(xy_dat,dims=1)
    # Reshape into a 2x2 matrix: `[min_x max_x; min_y max_y]`
    es = vcat(first.(mms), last.(mms))'

    # Adjust the extent if cell sizes are negative (indicating decreasing coordinate values).
    if sign(cell_size[1]) == -1
        es[1,:] = reverse(es[1,:]) # Reverse x-bounds
    end
    if sign(cell_size[2]) == -1
        es[2,:] = reverse(es[2,:]) # Reverse y-bounds
    end

    # Add/subtract half a cell size to accurately represent the extent of the cells,
    # assuming `xy_dat` are centroids or similar.
    return es .+ [-0.5,0.5].*cell_size'
end

"""
    find_overlapping_ext(ext_ti, ext_tj, origin, cell_size; inclusive=false)

Finds the spatial extent (`[min_x max_x; min_y max_y]`) of the cells in a grid
(defined by `origin` and `cell_size`) that overlap with a specified input extent
(`ext_ti`, `ext_tj`).

# Arguments
- `ext_ti`, `ext_tj`: X and Y spatial extents `[min, max]` to check for overlap.
- `origin`, `cell_size`: Grid definition parameters.
- `inclusive::Bool = false`: If `true`, the input extent includes the upper boundary.

# Returns
- `extent::Matrix{Float64}`: A `2 x 2` matrix representing the overlapping spatial extent.
"""
function find_overlapping_ext(ext_ti, ext_tj, origin, cell_size; inclusive=false)
    oi, oj = origin # Grid origin
    c, r = cell_size # Cell sizes

    # Adjust input extents based on `inclusive` flag.
    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    # Find the nearest grid indices (i,j) for the input extent boundaries.
    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    # Convert these indices back to spatial coordinates (top-left corners of the cells).
    si = oi .+ c * (is .- 1.) # X-coordinates of the overlapping cells' origins
    sj = oj .+ r * (js .- 1.) # Y-coordinates of the overlapping cells' origins

    # Calculate the full extent of the overlapping cells by adding cell_size to their origins.
    ext_ovi = si .+ [0,1] .* cell_size[1] # [min_x, max_x] of overlapping cells
    ext_ovj = sj .+ [0,1] .* cell_size[2] # [min_y, max_y] of overlapping cells

    # Return the combined extent as a 2x2 matrix.
    return hcat(ext_ovi, ext_ovj)'
end

"""
    merge_extents(exts, sgns)

Merges multiple spatial extents (`exts`) into a single, combined bounding box.

# Arguments
- `exts::AbstractVector{<:AbstractMatrix{<:Real}}`: A vector where each element is a `2 x 2`
  matrix representing a spatial extent `[min_x max_x; min_y max_y]`.
- `sgns::AbstractVector{<:Real}`: A 2-element vector `[sign_x, sign_y]` indicating the
  direction of increasing coordinates for each dimension (e.g., from `cell_size`).
  Used to correctly handle reversed extents.

# Returns
- `es::Matrix{Float64}`: A `2 x 2` matrix representing the merged spatial extent.
"""
function merge_extents(exts, sgns)
    # Concatenate all input extents into a single large matrix.
    aa = hcat(exts...)
    
    # Find the overall minimum and maximum coordinate for each dimension.
    mms = extrema(aa, dims=2)
    # Reshape into `[min_x max_x; min_y max_y]` format.
    es = hcat(first.(mms), last.(mms))

    # Adjust the merged extent based on the `sgns` (signs of cell sizes).
    # If a dimension has a negative cell size, its extent boundaries need to be reversed.
    if sign(sgns[1]) == -1
        es[1,:] = reverse(es[1,:])
    end
    if sign(sgns[2]) == -1
        es[2,:] = reverse(es[2,:])
    end

    return es
end

"""
    cell_size(raster)

Calculates the cell size `[width, height]` for a given `Raster` object.
Note that Julia's raster arrays are typically column-major, meaning rows usually
correspond to X-dimension and columns to Y-dimension in geospatial contexts.

# Arguments
- `raster::Raster`: The input `Raster` object.

# Returns
- `(cell_width, cell_height)::Tuple{Float64, Float64}`: A tuple containing the
  cell width (along rows, X) and cell height (along columns, Y).
  The cell height is negative if Y-coordinates decrease as row indices increase (e.g., for top-left origins).
"""
function cell_size(raster::Raster)
    # Get the dimensions of the raster array.
    rows = size(raster)[1]
    cols = size(raster)[2]

    # Get the bounding box of the raster. For 3D rasters, take the first slice.
    if length(size(raster)) == 3
        bbox = Rasters.bounds(raster[:, :, 1])
    else
        bbox = Rasters.bounds(raster)
    end

    # Extract min/max X and Y coordinates from the bounding box.
    xmin, xmax = bbox[1]
    ymin, ymax = bbox[2]
    
    # Calculate total width and height.
    width = xmax - xmin
    height = ymax - ymin
    
    # Calculate cell width and height.
    # `cell_width = width / rows` assumes rows correspond to X.
    # `cell_height = -height / cols` assumes columns correspond to Y, and the negative
    # sign accounts for Y-coordinates typically decreasing with increasing row index in top-left origin rasters.
    cell_width = width / rows
    cell_height = -height / cols
    return cell_width, cell_height
end
