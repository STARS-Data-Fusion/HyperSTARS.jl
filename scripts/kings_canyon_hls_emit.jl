using NetCDF
using ArchGDAL
using Rasters
using Plots
using DelimitedFiles
using Statistics
using Dates
using StatsBase
using GeoArrays
using Distributed
using MultivariateStats
using LinearAlgebra
using Plots
using Glob
using Missings
using HyperSTARS ## Pkg.add("https://github.com/STARS-Data-Fusion/HyperSTARS.jl") 

# Rasters.checkmem!(false)
# addprocs(8)
addprocs(Sys.CPU_THREADS - 1) ## workers (using Distributed package), Check how this automatic detection works on gattaca

@everywhere using HyperSTARS

######### Wrapper functions to reduce memory and compiling time
function get_hls_data(dir, bands, date_range)
    raster_list = []
    time_dates = nothing
    band_arrays_list = []
    ref_raster = nothing
    
    for band in bands 
        band_files = glob("*$(band)*.tif", joinpath(dir,band)) ## find all tif files in band subdirectory
        band_dates = [Date(match(r"\d{8}", f).match, "yyyymmdd") for f in band_files] ## extract dates from tif filenames
        kp_dates = date_range[1] .<= band_dates .<= date_range[2] ## find all available dates within target date range
        band_rasters = [Raster(x, lazy=true) for x in band_files[kp_dates]] ## read geotiffs for all dates found in date range, using lazy loading
        if ref_raster === nothing && !isempty(band_rasters)
            ref_raster = band_rasters[1]
        end
        if time_dates === nothing
            time_dates = band_dates[kp_dates]
        end
        # Convert rasters to arrays
        band_arrays = [Array(r) for r in band_rasters]
        push!(band_arrays_list, band_arrays)
    end ## loop over all bands

    # Get dimensions from first raster
    ny, nx = size(band_arrays_list[1][1])
    ntime = length(time_dates)
    nbands = length(findall(x -> !isempty(x), band_arrays_list))  # Only count non-empty bands
    
    # Create output array
    hls_array = zeros(Float32, ny, nx, nbands, ntime)
    
    # Fill array by concatenating times, then bands
    for (bi, band_arrays) in enumerate(band_arrays_list)
        for (ti, arr) in enumerate(band_arrays)
            # Replace missing values with NaN first, then assign
            for y in 1:ny, x in 1:nx
                val = arr[y, x]
                hls_array[y, x, bi, ti] = ismissing(val) ? NaN32 : Float32(val)
            end
        end
    end
    
    # Keep y,x,band,time order to match expected data layout (nx x ny x nw x T)
    return hls_array, time_dates, ref_raster
end

#### loading EMIT data from EMIT directory
function get_emit(dir_path, emit_dir, date_range)
    emit_metadata, _ = readdlm(joinpath(dir_path,"EMIT_metadata.csv"), ',', header=true)
    wavelengths = emit_metadata[emit_metadata[:,2].==1,1]
    fwhm = emit_metadata[emit_metadata[:,2].==1,3]
    good_wavelengths = emit_metadata[:,2]
    good_wavelength_inds = findall(good_wavelengths .== 1)

    emit_files = glob("*.tif", emit_dir)
    emit_dates = [Date(match(r"\d{8}", f).match, "yyyymmdd") for f in emit_files] ## extract dates from tif filenames
    kp_dates = date_range[1] .<= emit_dates .<= date_range[2] ## find all available dates within target date range

    # Use lazy=true to avoid loading entire large files into memory
    emit_rasters = [Raster(x, lazy=true)[:,:,good_wavelength_inds] for x in emit_files[kp_dates]] # read emit rasters for dates in range, only keep good wavelengths
    ref_raster = isempty(emit_rasters) ? nothing : emit_rasters[1]

    # Convert rasters to arrays
    emit_arrays = [Array(r) for r in emit_rasters]
    
    if !isempty(emit_arrays)
        ny, nx, nwaves = size(emit_arrays[1])
        ntime = length(emit_arrays)
        
        # Create output array
        emit_array = zeros(Float32, ny, nx, nwaves, ntime)
        
        # Fill array
        for (ti, arr) in enumerate(emit_arrays)
            for y in 1:ny, x in 1:nx, w in 1:nwaves
                val = arr[y, x, w]
                emit_array[y, x, w, ti] = ismissing(val) ? NaN32 : Float32(val)
            end
        end
    else
        emit_array = zeros(Float32, 0, 0, 0, 0)
    end

    emit_array[.!isfinite.(emit_array) .| (emit_array .== -9999)] .= NaN ## replace missings and -9999 with nans
    return emit_array, emit_dates[kp_dates], fwhm, wavelengths, ref_raster
end

###### spectral response functions for landsat (L30) and sentinel (S30)
function get_srf(dir_path)
    l30_srf_path = joinpath(dir_path, "HLS_L30_srf.csv")
    s30_srf_path = joinpath(dir_path, "HLS_S30_srf.csv")
    
    if !isfile(l30_srf_path)
        error("HLS_L30_srf.csv not found at $(l30_srf_path). " *
              "Please ensure spectral response function files are in your data directory.")
    end
    if !isfile(s30_srf_path)
        error("HLS_S30_srf.csv not found at $(s30_srf_path). " *
              "Please ensure spectral response function files are in your data directory.")
    end
    
    HLS_L30_srf, _ = readdlm(l30_srf_path, ',', header=true)
    HLS_S30_srf, _ = readdlm(s30_srf_path, ',', header=true)

    ## HLS uses the landsat srf for matching bands (all but red edge)
    HLS_S30_srf[HLS_S30_srf[:,1] .∈ Ref(HLS_L30_srf[:,1]),[2,3,4,5,10,13,14]] .= HLS_L30_srf[:,[2,3,4,5,6,8,9]]

    ## normalize to weights
    HLS_S30_srf[:,2:end] .= HLS_S30_srf[:,2:end] ./ sum(HLS_S30_srf[:,2:end],dims=1)
    HLS_L30_srf[:,2:end] .= HLS_L30_srf[:,2:end] ./ sum(HLS_L30_srf[:,2:end],dims=1)

    HLS_L30_srf2 = HLS_L30_srf[.!(sum(HLS_L30_srf[:,2:end],dims=2).==0)[:],:] ## remove 0 entries
    HLS_S30_srf2 = HLS_S30_srf[.!(sum(HLS_S30_srf[:,2:end],dims=2).==0)[:],:] ## remove 0 entries

    hls_l30_waves = [round(mean(HLS_L30_srf2[x .> 0,1])) for x in eachcol(HLS_L30_srf2[:,[2:6...,8,9]])]
    hls_s30_waves = [round(mean(HLS_S30_srf2[x .> 0,1])) for x in eachcol(HLS_S30_srf2[:,[2:10...,13,14]])]

    S30_srf = Dict(:w => HLS_S30_srf2[:,1], :rsr => HLS_S30_srf2[:,[2:10...,13,14]]')
    L30_srf = Dict(:w => HLS_L30_srf2[:,1], :rsr => HLS_L30_srf2[:,[2:6...,8,9]]')
    return S30_srf, L30_srf, hls_l30_waves, hls_s30_waves
end

#### process EMIT and HLS, return data and geodata structs for fusion model
function get_data(dir_path, date_range)
    #### product directory
    hls_l30_dir = joinpath(dir_path,"Kings_Canyon_HLS/L30/")
    hls_s30_dir = joinpath(dir_path,"Kings_Canyon_HLS/S30/")
    emit_dir = joinpath(dir_path,"Kings_Canyon_EMIT")

    #### loading HLS data from tif
    hls_l30_bands = ["coastal_aerosol", "blue", "green", "red", "NIR", "SWIR1", "SWIR2"]
    hls_s30_bands = ["coastal_aerosol", "blue", "green", "red", "rededge1","rededge2","rededge3", "NIR_broad", "NIR", "SWIR1", "SWIR2"]

    ## load and combine all L30 and S30 data into x,y,band,time raster
    hls_l30_raster, hls_l30_dates, hls_l30_ref = get_hls_data(hls_l30_dir, hls_l30_bands, date_range)
    hls_s30_raster, hls_s30_dates, hls_s30_ref = get_hls_data(hls_s30_dir, hls_s30_bands, date_range)

    S30_srf, L30_srf, hls_l30_waves, hls_s30_waves = get_srf(dir_path)

    emit_raster, emit_dates, fwhm_emit, emit_waves, emit_ref = get_emit(dir_path, emit_dir, date_range)

    ###### Fusion setup
    emit_origin = get_centroid_origin_raster(emit_ref)
    hls_s30_origin = get_centroid_origin_raster(hls_s30_ref)
    hls_l30_origin = get_centroid_origin_raster(hls_l30_ref)

    emit_ndims = collect(size(emit_raster)[1:2])
    hls_s30_ndims = collect(size(hls_s30_raster)[1:2])
    hls_l30_ndims = collect(size(hls_l30_raster)[1:2])

    emit_csize = collect(cell_size(emit_ref))
    hls_s30_csize = collect(cell_size(hls_s30_ref))
    hls_l30_csize = collect(cell_size(hls_l30_ref))

    ## sequence of dates from start/end observed
    all_dates = minimum([emit_dates..., hls_l30_dates..., hls_s30_dates...]):maximum([emit_dates..., hls_l30_dates..., hls_s30_dates...])

    ## convert to t=1:T times
    emit_times = findall(all_dates .∈ Ref(emit_dates))
    hls_l30_times = findall(all_dates .∈ Ref(hls_l30_dates))
    hls_s30_times = findall(all_dates .∈ Ref(hls_s30_dates))

    ## rasters to data arrays, nx x ny x nw x T
    hls_l30_array = disallowmissing(hls_l30_raster);
    hls_s30_array = disallowmissing(hls_s30_raster);
    emit_array = disallowmissing(emit_raster);

    ### create geospatial structs for each instrument
    # 4th element is spatial "fidelity": 0 for highest (target) spatial res, 1 for high but not target, 2 for coarse res (like PACE)
    hls_l30_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, hls_l30_ndims, 0, hls_l30_times, hls_l30_waves, L30_srf)
    hls_s30_geodata = InstrumentGeoData(hls_s30_origin, hls_s30_csize, hls_s30_ndims, 0, hls_s30_times, hls_s30_waves, S30_srf)
    emit_geodata = InstrumentGeoData(emit_origin, emit_csize, emit_ndims, 1, emit_times, emit_waves, [0.1])

    ## create vector of InstrumentGeoData for all instruments
    inst30m_geodata = [emit_geodata, hls_l30_geodata, hls_s30_geodata]

    ## create data struct objects, time-varying biases and variances repeat for constants
    hls_s30_data = InstrumentData(hls_s30_array, zeros(size(hls_s30_array)[3:4]), 1e-5*ones(size(hls_s30_array)[3:4]), abs.(hls_s30_csize), hls_s30_times, [1,1], hls_s30_waves, S30_srf)
    hls_l30_data = InstrumentData(hls_l30_array, zeros(size(hls_l30_array)[3:4]), 1e-5*ones(size(hls_l30_array)[3:4]), abs.(hls_l30_csize), hls_l30_times, [1,1], hls_l30_waves, L30_srf)
    emit_data = InstrumentData(emit_array, zeros(size(emit_array)[3:4]), 1e-6*ones(size(emit_array)[3:4]), abs.(emit_csize), emit_times, [1,1], emit_waves, fwhm_emit)

    ## vector of InstrumentData for all instruments, order must match inst30m_geodata order!
    data30m_list = [emit_data, hls_l30_data, hls_s30_data];
    return data30m_list, inst30m_geodata, all_dates
end 

#### run pca, return basis information
function get_pca(data30m_list)
    ##### Estimate PCA basis from available EMIT
    n1,n2,n3,n4 = size(data30m_list[1].data)

    ### get basis functions using EMIT (PCA) 
    emit_rt = reshape(permutedims(data30m_list[1].data, (1,2,4,3)),(n1*n2*n4,n3))'
    emit_rt2 = emit_rt[:,.!vec(any(isnan, emit_rt; dims=1))]
    mm = mean(emit_rt2, dims=2)[:]
    sx = std(emit_rt2, dims=2)[:]
    Xt = (emit_rt2 .- mm) ./ sx
    pca = MultivariateStats.fit(PCA, Xt; pratio=0.995)
    B = projection(pca)
    vrs = principalvars(pca)
    scs = predict(pca, Xt)


    # ## compute scores
    # scs_full = zeros(size(scs,1), size(emit_rt,2))
    # scs_full .= NaN
    # scs_full[:,.!vec(any(isnan, emit_rt; dims=1))] = scs

    # scs_rt = reshape(scs_full',(n1,n2,n4,size(scs,1)))

    # ## compute temporal difference of emit scores
    # scs_diff = diff(scs_rt, dims=3) 

    return B, mm, sx, vrs
end




#### Target fusion date range
date_range = [Date("2022-08-01"), Date("2022-08-31")]

#### parent directory
# dir_path = "/Users/maggiej/Documents/Hyperspectral_DataFusion/Data/KingsCanyon/"
dir_path = "/gpfs/scratch/refl-datafusion-trtd/"

data30m_list, inst30m_geodata, all_dates = get_data(dir_path, date_range)

## pull landsat and emit spatial information for target resolution grids
hls_l30_origin = inst30m_geodata[2].origin
hls_l30_ndims = inst30m_geodata[2].ndims
hls_l30_csize = inst30m_geodata[2].cell_size
emit_waves = inst30m_geodata[1].wavelengths
fwhm_emit = data30m_list[1].rsr

## define moving window sizes 
scf = 4 #scf: number of target pixels to create scf x scf partitions
target_ndims = hls_l30_ndims # define prediction grid size 
nwindows = Int.(ceil.(target_ndims./scf)) # number of windows covering target scene 

### geospatial structs for window grid and target grid
window30m_geodata = InstrumentGeoData(hls_l30_origin .+ (scf - 1)/2*hls_l30_csize, scf*hls_l30_csize, nwindows, 0, [1], emit_waves, fwhm_emit)
target30m_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, target_ndims, 0, [1], emit_waves, [0.1])

##### Estimate PCA basis from available EMIT
B, mm, sx, vrs = get_pca(data30m_list)

##### Set up priors
Bs = B .* sx[:] # rescale pc basis 
pmean = zeros((target_ndims...,size(B)[2])) # prior mean array (nx x ny x np) where np is number of PCs
pvar = ones((target_ndims...,size(B)[2])) .* reshape(vrs ./ 100, (1,1,size(Bs,2))) # prior var array (nx x ny x np)

## setting spatial variance parameters (not estimated in this demo and not spatially varying)
model_pars = ones((nwindows...,size(B)[2],4)) .* reshape([0.1,200,1e-10,1.5], (1,1,1,4)) # covariance parameters (nx x ny x np x 4), last dimension is [spatial var, length scale, nugget, smoothness]
for (i,x) in enumerate(vrs)
    model_pars[:,:,i,1] .= x / 1000
end


### run data fusion
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

@time fused_images, fused_sd_images = scene_fusion_pmap(data30m_list,
            inst30m_geodata,
            window30m_geodata,
            target30m_geodata,
            float.(mm), ## spectral mean
            pmean, ## mean-zero prior mean
            pvar, ## diagonal prior var
            Bs, ## spectral basis function matrix
            model_pars; 
            nsamp=50,
            window_buffer = 4,
            target_times = 1:2, 
            smooth = false,           
            spatial_mod = exp_corD,                                           
            obs_operator = unif_weighted_obs_operator_centroid,
            state_in_cov=true,
            cov_wt=0.2,
            tscov_pars = sqrt.(vrs) ./ 10.0, 
            ar_phi=1.0,
            window_radius=100.0);

# fused_images and fused_sd_images are y,x,wavelength,time arrays, need to write to raster or .nc as desired            
