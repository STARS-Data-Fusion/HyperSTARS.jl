
using Rasters
using Statistics
using StatsBase
using Random
using LinearAlgebra
using Sobol

############ figuring out faster raster indexing...
## i is row index, j is col index, i.e. rast[i,j,...]
## c is row spacing, r is col spacing
## if r or c is negative, indicates decreasing si/sj as i/j increases
## oi,oj is origin geolocation at i=1, j=1
## si,sj is spatial centroid for i,j assuming i,j is upper left corner or cell

function find_nearest_ij(target::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}) where T<:Real
    oi, oj = origin
    c, r = cell_size
    ti,tj = target

    i = Int64(floor((ti - oi) / c + 1))
    j = Int64(floor((tj - oj) / r + 1))
    return [i, j]
end

function find_nearest_ij_multi(target::AbstractArray{T}, origin::AbstractVector{T}, csize::AbstractVector{T}, n_dims::AbstractVector{<:Real}) where T <: Real
    oi, oj = origin
    c, r = csize

    i = Int64.(round.((target[:,1] .- oi) ./ c .+ 1))
    j = Int64.(round.((target[:,2] .- oj) ./ r .+ 1))

    df = [i j]
    df2 = df[(1 .<= i .<= n_dims[1]) .& (1 .<= j .<= n_dims[2]),:]
    return df2
end


function find_nearest_ind(target::T, origin::T, cell_size::T) where T <: Real
    ind = Int(round((target - origin) / cell_size + 1))
    return ind
end

### pixels overlapping extent
function find_touching_inds_ext(ext::AbstractVector{T}, origin::T, cellsize::T) where T <: Real
    ind1 = (ext[1] - origin) / cellsize + 1
    ind2 = (ext[2] - origin) / cellsize + 1

    return [Int(round(ind1)), Int(round(ind2))]
end

### find all pixels overlapping extent
function find_all_touching_ij_ext(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cellsize::AbstractVector{T}, n_dims::AbstractVector{<:Real}; inclusive=false) where T<:Real
    oi, oj = origin
    c, r = cellsize

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[-1e-8,1e-8]
        ext_tj = ext_tj .- sign(r)*[-1e-8,1e-8]
    end

    is = find_touching_inds_ext(ext_ti, oi, c)
    js = find_touching_inds_ext(ext_tj, oj, r)

    # is = find_nearest_ind.(ext_ti, oi, c)
    # js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        iss = repeat(is[1]:is[2], inner=js[2]-js[1]+1, outer=[1])
        jss = repeat(js[1]:js[2], inner=[1], outer=is[2]-is[1]+1)

        return unique(hcat(iss,jss),dims=1)
    else
        return Array{Int64}(undef, 0, 0)
    end
end

### pixels with centroid contained in extent
function find_nearest_inds_ext(ext::AbstractVector{T}, origin::T, cellsize::T) where T <: Real
    ind1 = (ext[1] - origin) / cellsize + 1
    ind2 = (ext[2] - origin) / cellsize + 1

    return [Int(ceil(ind1)), Int(floor(ind2))]
end

### find all pixels with centroid contained in extent
function find_all_ij_ext(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cellsize::AbstractVector{T}, n_dims::AbstractVector{<:Real}; inclusive=false) where T<:Real
    oi, oj = origin
    c, r = cellsize

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-8]
        ext_tj = ext_tj .- sign(r)*[0,1e-8]
    end

    is = find_nearest_inds_ext(ext_ti, oi, c)
    js = find_nearest_inds_ext(ext_tj, oj, r)

    # is = find_nearest_ind.(ext_ti, oi, c)
    # js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        iss = repeat(is[1]:is[2], inner=js[2]-js[1]+1, outer=[1])
        jss = repeat(js[1]:js[2], inner=[1], outer=is[2]-is[1]+1)

        return unique(hcat(iss,jss),dims=1)
    else
        return Array{Int64}(undef, 0, 0)
    end
end

function find_all_bau_ij(target::AbstractVector{T}, target_cell_size::AbstractVector{T}, bau_origin::AbstractVector{T}, bau_cell_size::AbstractVector{T}) where T<:Real
    oi, oj = bau_origin
    c, r = bau_cell_size

    ext_ti = target[1] .+ [-0.5,0.5-1e-9]*target_cell_size[1]
    ext_tj = target[2] .+ [-0.5,0.5-1e-9]*target_cell_size[2]

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

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

    iss = repeat(is2, inner=size(js2)[1], outer=[1])
    jss = repeat(js2, inner=[1], outer=size(is2)[1])

    return unique(hcat(iss,jss),dims=1)
end

function find_all_bau_ij_multi(target::AbstractArray{T}, target_cell_size::AbstractVector{T}, bau_origin::AbstractVector{T}, bau_cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}) where T<:Real
    oi, oj = bau_origin
    c, r = bau_cell_size

    ext_ti = target[:,1]*[1 1] .+ [-0.5+1e-6 0.5-1e-6]*target_cell_size[1] ## 1e-6 is used to force julia to round 0.5 up instead of down 
    ext_tj = target[:,2]*[1 1] .+ [-0.5+1e-6 0.5-1e-6]*target_cell_size[2] 

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        irngs = (:).(is[:,1],is[:,2])
        jrngs = (:).(js[:,1],js[:,2])

        t1 = Iterators.product.(irngs,jrngs)
        tpls = Iterators.flatten(t1)
        return stack(unique(tpls))'
    else
        return Array{Int64}(undef, 0, 0)
    end
end

function subsample_bau_ij(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}; nsamp=100, inclusive=false) where T<:Real

    oi, oj = origin
    c, r = cell_size

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        ni = is[2]-is[1] + 1
        nj = js[2]-js[1] + 1

        if nsamp >= ni*nj
            iss = repeat(is[1]:is[2], inner=[nj], outer=[1])
            jss = repeat(js[1]:js[2], inner=[1], outer=[ni])

        else
            # r = ni/nj

            np = minimum([nsamp, ni*nj])
            
            iss = sample(is[1]:is[2], np, replace=true)
            jss = sample(js[1]:js[2], np, replace=true)
        end

        return unique(hcat(iss,jss),dims=1)
    else
        return Array{Int64}(undef, 0, 0)
    end
end

function subsample_bau_ij2(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}; nsamp=100, inclusive=false) where T<:Real

    oi, oj = origin
    c, r = cell_size

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        ni = is[2]-is[1] + 1
        nj = js[2]-js[1] + 1

        iss = repeat(is[1]:is[2], inner=[nj], outer=[1])
        jss = repeat(js[1]:js[2], inner=[1], outer=[ni])

        indss = hcat(iss,jss)

        if nsamp >= ni*nj
            return indss
        else
            is = sample(1:size(indss)[1], nsamp, replace=false)
            return(indss[is,:])
        end
    else
        return Array{Int64}(undef, 0, 0)
    end
end

function sobol_bau_ij(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}, n_dims::AbstractVector{<:Real}; nsamp=100, inclusive=false) where T<:Real

    oi, oj = origin
    c, r = cell_size

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        ni = is[2]-is[1] + 1
        nj = js[2]-js[1] + 1

        if nsamp >= ni*nj
            iss = repeat(is[1]:is[2], inner=[nj], outer=[1])
            jss = repeat(js[1]:js[2], inner=[1], outer=[ni])
            dd = unique(hcat(iss,jss),dims=1)
        else
            s = SobolSeq([is[1],js[1]],[is[2],js[2]])            
            p = reduce(hcat, Sobol.next!(s) for i = 1:nsamp)'
            dd = Int64.(unique(round.(p),dims=1))
        end

        return dd
    else
        return Array{Int64}(undef, 0, 0)
    end
end

function get_sij_from_ij(ij_array::AbstractArray{<:Real}, origin::AbstractVector{T}, cell_size::AbstractVector{T}) where T<:Real
    origin' .+ cell_size' .* (ij_array .- [1.,1.]')
end

function get_origin_raster(rast::Raster)
    return [rast.dims[1].val[1],rast.dims[2].val[1]]
end

function get_centroid_origin_raster(rast::Raster)
    cc = abs.(collect(cell_size(rast)))
    return [rast.dims[1].val[1],rast.dims[2].val[1]] .+ cc./2
end

function bbox_from_ul(target::AbstractVector{T},cell_size::AbstractVector{T}) where T<:Real
    return hcat(zeros(2),cell_size) .+ target
end

function bbox_from_centroid(target::AbstractVector{T},cell_size::AbstractVector{T}) where T<:Real
    return ones(2,2) .* [-0.5 0.5] .* cell_size .+ target
end

function extent_from_xy(xy_dat::AbstractArray{T}, cell_size::AbstractVector{T}) where T<:Real
    mms = extrema(xy_dat,dims=1)
    es = vcat(first.(mms), last.(mms))'

    if sign(cell_size[1]) == -1
        es[1,:] = reverse(es[1,:])
    end
    if sign(cell_size[2]) == -1
        es[2,:] = reverse(es[2,:])
    end

    return es .+ [-0.5,0.5].*cell_size'
end

function find_overlapping_ext(ext_ti::AbstractVector{T}, ext_tj::AbstractVector{T}, origin::AbstractVector{T}, cell_size::AbstractVector{T}; inclusive=false) where T<:Real
    oi, oj = origin
    c, r = cell_size

    if !inclusive
        ext_ti = ext_ti .- sign(c)*[0,1e-9]
        ext_tj = ext_tj .- sign(r)*[0,1e-9]
    end

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)

    si = oi .+ c * (is .- 1.)
    sj = oj .+ r * (js .- 1.)

    ext_ovi = si .+ [-0.5,0.5] .* cell_size[1]
    ext_ovj = sj .+ [-0.5,0.5] .* cell_size[2]

    return hcat(ext_ovi, ext_ovj)'
end

function merge_extents(exts::AbstractVector{<:AbstractArray}, sgns)
    aa = hcat(exts...)
    mms = extrema(aa, dims=2)
    es = hcat(first.(mms), last.(mms))

    if sign(sgns[1]) == -1
        es[1,:] = reverse(es[1,:])
    end
    if sign(sgns[2]) == -1
        es[2,:] = reverse(es[2,:])
    end

    return es
end

function cell_size(raster::Raster)
    # Julia is column-major so raster arrays are transposed from the way they are in Numpy
    # in Julia, rows translate to X, and cols translate to Y
    rows = size(raster)[1]
    cols = size(raster)[2]

    if length(size(raster)) == 3
        bbox = Rasters.bounds(raster[:, :, 1])
    else
        bbox = Rasters.bounds(raster)
    end

    xmin, xmax = bbox[1]
    ymin, ymax = bbox[2]
    width = xmax - xmin
    height = ymax - ymin
    cell_width = width / rows
    cell_height = -height / cols
    cell_width, cell_height
end