"""
Minimal Working Example for HyperSTARS Data Fusion

This is a reduced-complexity version of kings_canyon_hls_emit_local.jl for:
- Faster testing (3 days instead of 1 month)
- Larger windows (fewer partitions) 
- Reduced sampling for quicker convergence
- Memory usage profiling

Run with: julia --project=. scripts/minimal_example.jl
"""

using Pkg
Pkg.activate(".")

using Distributed
addprocs(4) ## add compute workers

@everywhere using HyperSTARS
using Rasters, ArchGDAL, GeoArrays
using Dates
using Glob
using DelimitedFiles
using Missings
using MultivariateStats
using Statistics
using LinearAlgebra
@everywhere using SparseArrays

@everywhere include(joinpath(@__DIR__, "../src/spatial_utils_ll.jl"))
@everywhere include(joinpath(@__DIR__, "../src/spatial_utils.jl"))  
@everywhere include(joinpath(@__DIR__, "../src/resampling_utils.jl"))
@everywhere include(joinpath(@__DIR__, "../src/GP_utils.jl"))

#### Data loading functions (same as main script) ####

function get_hls_data(dir, bands, date_range)
    raster_list = []
    time_dates = nothing
    band_arrays_list = []
    ref_raster = nothing
    
    for band in bands 
        band_files = glob("*$(band)*.tif", joinpath(dir,band))
        if isempty(band_files)
            @warn "No files found for band $band in $dir"
            continue
        end
        band_dates = [Date(match(r"\d{8}", f).match, "yyyymmdd") for f in band_files]
        kp_dates = date_range[1] .<= band_dates .<= date_range[2]
        band_rasters = [Raster(x, lazy=true) for x in band_files[kp_dates]]
        if isempty(band_rasters)
            @warn "No rasters found for band $band in specified date range"
            continue
        end
        if ref_raster === nothing
            ref_raster = band_rasters[1]
        end
        if time_dates === nothing
            time_dates = band_dates[kp_dates]
        end
        band_arrays = [Array(r) for r in band_rasters]
        push!(band_arrays_list, band_arrays)
    end

    if isempty(band_arrays_list)
        error("No valid raster data found for any bands")
    end
    
    ny, nx = size(band_arrays_list[1][1])
    ntime = length(time_dates)
    nbands = length(findall(x -> !isempty(x), band_arrays_list))
    
    hls_array = zeros(Float32, ny, nx, nbands, ntime)
    
    for (bi, band_arrays) in enumerate(band_arrays_list)
        for (ti, arr) in enumerate(band_arrays)
            for y in 1:ny, x in 1:nx
                val = arr[y, x]
                hls_array[y, x, bi, ti] = ismissing(val) ? NaN32 : Float32(val)
            end
        end
    end

    hls_array[ismissing.(hls_array)] .= NaN
    return hls_array, time_dates, ref_raster
end

function get_emit(dir_path, emit_dir, date_range)
    emit_metadata, _ = readdlm(joinpath(dir_path,"EMIT_metadata.csv"), ',', header=true)
    wavelengths = emit_metadata[emit_metadata[:,2].==1,1]
    fwhm = emit_metadata[emit_metadata[:,2].==1,3]
    good_wavelengths = emit_metadata[:,2]
    good_wavelength_inds = findall(good_wavelengths .== 1)

    emit_files = glob("*.tif", emit_dir)
    emit_dates = [Date(match(r"\d{8}", f).match, "yyyymmdd") for f in emit_files]
    kp_dates = date_range[1] .<= emit_dates .<= date_range[2]

    # Keep reference raster before slicing
    kept_files = emit_files[kp_dates]
    ref_raster = isempty(kept_files) ? nothing : Raster(kept_files[1], lazy=true)
    
    emit_rasters = [Raster(x, lazy=true)[:,:,good_wavelength_inds] for x in kept_files]
    emit_arrays = [Array(r) for r in emit_rasters]
    
    if !isempty(emit_arrays)
        ny, nx, nwaves = size(emit_arrays[1])
        ntime = length(emit_arrays)
        
        emit_array = zeros(Float32, ny, nx, nwaves, ntime)
        
        for (ti, arr) in enumerate(emit_arrays)
            for y in 1:ny, x in 1:nx, w in 1:nwaves
                val = arr[y, x, w]
                emit_array[y, x, w, ti] = ismissing(val) ? NaN32 : Float32(val)
            end
        end
    else
        emit_array = zeros(Float32, 0, 0, 0, 0)
    end

    emit_array[.!isfinite.(emit_array) .| (emit_array .== -9999)] .= NaN
    return emit_array, emit_dates[kp_dates], fwhm, wavelengths, ref_raster
end

function get_srf(dir_path)
    l30_srf_path = joinpath(dir_path, "HLS_L30_srf.csv")
    s30_srf_path = joinpath(dir_path, "HLS_S30_srf.csv")
    
    if !isfile(l30_srf_path)
        error("HLS_L30_srf.csv not found at $(l30_srf_path)")
    end
    if !isfile(s30_srf_path)
        error("HLS_S30_srf.csv not found at $(s30_srf_path)")
    end
    
    HLS_L30_srf, _ = readdlm(l30_srf_path, ',', header=true)
    HLS_S30_srf, _ = readdlm(s30_srf_path, ',', header=true)

    HLS_S30_srf[HLS_S30_srf[:,1] .∈ Ref(HLS_L30_srf[:,1]),[2,3,4,5,10,13,14]] .= HLS_L30_srf[:,[2,3,4,5,6,8,9]]

    HLS_S30_srf[:,2:end] .= HLS_S30_srf[:,2:end] ./ sum(HLS_S30_srf[:,2:end],dims=1)
    HLS_L30_srf[:,2:end] .= HLS_L30_srf[:,2:end] ./ sum(HLS_L30_srf[:,2:end],dims=1)

    HLS_L30_srf2 = HLS_L30_srf[.!(sum(HLS_L30_srf[:,2:end],dims=2).==0)[:],:] 
    HLS_S30_srf2 = HLS_S30_srf[.!(sum(HLS_S30_srf[:,2:end],dims=2).==0)[:],:] 

    hls_l30_waves = [round(mean(HLS_L30_srf2[x .> 0,1])) for x in eachcol(HLS_L30_srf2[:,[2:6...,8,9]])]
    hls_s30_waves = [round(mean(HLS_S30_srf2[x .> 0,1])) for x in eachcol(HLS_S30_srf2[:,[2:10...,13,14]])]

    S30_srf = Dict(:w => HLS_S30_srf2[:,1], :rsr => HLS_S30_srf2[:,[2:10...,13,14]]')
    L30_srf = Dict(:w => HLS_L30_srf2[:,1], :rsr => HLS_L30_srf2[:,[2:6...,8,9]]')
    return S30_srf, L30_srf, hls_l30_waves, hls_s30_waves
end

function get_data(dir_path, date_range)
    hls_l30_dir = joinpath(dir_path,"Kings_Canyon_HLS/L30/")
    hls_s30_dir = joinpath(dir_path,"Kings_Canyon_HLS/S30/")
    emit_dir = joinpath(dir_path,"Kings_Canyon_EMIT")

    hls_l30_bands = ["coastal_aerosol", "blue", "green", "red", "NIR", "SWIR1", "SWIR2"]
    hls_s30_bands = ["coastal_aerosol", "blue", "green", "red", "rededge1","rededge2","rededge3", "NIR_broad", "NIR", "SWIR1", "SWIR2"]

    hls_l30_raster, hls_l30_dates, hls_l30_ref = get_hls_data(hls_l30_dir, hls_l30_bands, date_range)
    hls_s30_raster, hls_s30_dates, hls_s30_ref = get_hls_data(hls_s30_dir, hls_s30_bands, date_range)

    S30_srf, L30_srf, hls_l30_waves, hls_s30_waves = get_srf(dir_path)

    emit_raster, emit_dates, fwhm_emit, emit_waves, emit_ref = get_emit(dir_path, emit_dir, date_range)

    emit_origin = get_centroid_origin_raster(emit_ref)
    hls_s30_origin = get_centroid_origin_raster(hls_s30_ref)
    hls_l30_origin = get_centroid_origin_raster(hls_l30_ref)

    emit_ndims = collect(size(emit_raster)[1:2])
    hls_s30_ndims = collect(size(hls_s30_raster)[1:2])
    hls_l30_ndims = collect(size(hls_l30_raster)[1:2])

    emit_csize = collect(cell_size(emit_ref))
    hls_s30_csize = collect(cell_size(hls_s30_ref))
    hls_l30_csize = collect(cell_size(hls_l30_ref))

    all_dates = minimum([emit_dates..., hls_l30_dates..., hls_s30_dates...]):maximum([emit_dates..., hls_l30_dates..., hls_s30_dates...])

    emit_times = findall(all_dates .∈ Ref(emit_dates))
    hls_l30_times = findall(all_dates .∈ Ref(hls_l30_dates))
    hls_s30_times = findall(all_dates .∈ Ref(hls_s30_dates))

    hls_l30_array = disallowmissing(hls_l30_raster);
    hls_s30_array = disallowmissing(hls_s30_raster);
    emit_array = disallowmissing(emit_raster);

    hls_l30_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, hls_l30_ndims, 0, hls_l30_times, hls_l30_waves, L30_srf)
    hls_s30_geodata = InstrumentGeoData(hls_s30_origin, hls_s30_csize, hls_s30_ndims, 0, hls_s30_times, hls_s30_waves, S30_srf)
    emit_geodata = InstrumentGeoData(emit_origin, emit_csize, emit_ndims, 1, emit_times, emit_waves, [0.1])

    inst30m_geodata = [emit_geodata, hls_l30_geodata, hls_s30_geodata]

    hls_s30_data = InstrumentData(hls_s30_array, zeros(size(hls_s30_array)[3:4]), 1e-5*ones(size(hls_s30_array)[3:4]), abs.(hls_s30_csize), hls_s30_times, [1,1], hls_s30_waves, S30_srf)
    hls_l30_data = InstrumentData(hls_l30_array, zeros(size(hls_l30_array)[3:4]), 1e-5*ones(size(hls_l30_array)[3:4]), abs.(hls_l30_csize), hls_l30_times, [1,1], hls_l30_waves, L30_srf)
    emit_data = InstrumentData(emit_array, zeros(size(emit_array)[3:4]), 1e-6*ones(size(emit_array)[3:4]), abs.(emit_csize), emit_times, [1,1], emit_waves, fwhm_emit)

    data30m_list = [emit_data, hls_l30_data, hls_s30_data];
    return data30m_list, inst30m_geodata, all_dates
end 

function get_pca(data30m_list)
    n1,n2,n3,n4 = size(data30m_list[1].data)
    emit_rt = reshape(permutedims(data30m_list[1].data, (1,2,4,3)),(n1*n2*n4,n3))'
    emit_rt2 = emit_rt[:,.!vec(any(isnan, emit_rt; dims=1))]
    mm = mean(emit_rt2, dims=2)[:]
    sx = std(emit_rt2, dims=2)[:]
    Xt = (emit_rt2 .- mm) ./ sx
    pca = MultivariateStats.fit(PCA, Xt; pratio=0.995)
    B = projection(pca)
    vrs = principalvars(pca)
    return B, mm, sx, vrs
end

#### MINIMAL EXAMPLE CONFIGURATION ####

println("=" ^ 70)
println("MINIMAL WORKING EXAMPLE - HyperSTARS Data Fusion")
println("=" ^ 70)

# Use only 5 days covering available EMIT data (Aug 13, 17)
date_range = [Date("2022-08-13"), Date("2022-08-17")]
println("Date range: $(date_range[1]) to $(date_range[2])")

dir_path = expanduser("~/data/")

# Measure memory
mem_start = Sys.maxrss() / 1024^2

println("\n[1/5] Loading data...")
@time data30m_list, inst30m_geodata, all_dates = get_data(dir_path, date_range)
mem_after_load = Sys.maxrss() / 1024^2
println("Memory: $(round(mem_after_load, digits=1)) MB (+$(round(mem_after_load - mem_start, digits=1)) MB)")

## Spatial information
hls_l30_origin = inst30m_geodata[2].origin
hls_l30_ndims = inst30m_geodata[2].ndims
hls_l30_csize = inst30m_geodata[2].cell_size
emit_waves = inst30m_geodata[1].wavelengths
fwhm_emit = data30m_list[1].rsr

println("Target grid: $(hls_l30_ndims)")

## Larger windows for faster processing
scf = 16  # Was 4 in full example
target_ndims = hls_l30_ndims
nwindows = Int.(ceil.(target_ndims./scf))
println("Windows: $(scf)x$(scf) partitions, $(nwindows) total = $(prod(nwindows)) windows")

window30m_geodata = InstrumentGeoData(hls_l30_origin .+ (scf - 1)/2*hls_l30_csize, 
                                       scf*hls_l30_csize, nwindows, 0, [1], 
                                       emit_waves, fwhm_emit)
target30m_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, 
                                       target_ndims, 0, [1], 
                                       emit_waves, [0.1])

println("\n[2/5] Computing PCA...")
@time B, mm, sx, vrs = get_pca(data30m_list)
mem_after_pca = Sys.maxrss() / 1024^2
println("Memory: $(round(mem_after_pca, digits=1)) MB (+$(round(mem_after_pca - mem_after_load, digits=1)) MB)")
println("PCA: $(size(B,2)) components")

println("\n[3/5] Setting up priors...")
Bs = B .* sx[:]
pmean = zeros((target_ndims..., size(B)[2]))
pvar = ones((target_ndims..., size(B)[2])) .* reshape(vrs ./ 100, (1,1,size(Bs,2)))

model_pars = ones((nwindows..., size(B)[2], 4)) .* reshape([0.1, 200, 1e-10, 1.5], (1,1,1,4))
for (i,x) in enumerate(vrs)
    model_pars[:,:,i,1] .= x / 1000
end

mem_after_setup = Sys.maxrss() / 1024^2
println("Memory: $(round(mem_after_setup, digits=1)) MB")

println("\n[4/5] Setting up workers...")
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

println("\n[5/5] Running fusion (reduced parameters)...")
println("  nsamp=20 (was 50), window_buffer=2 (was 4)")

mem_before = Sys.maxrss() / 1024^2
fusion_time = @elapsed begin
    fused_images, fused_sd_images = scene_fusion_pmap(
        data30m_list, inst30m_geodata, window30m_geodata, target30m_geodata,
        float.(mm), pmean, pvar, Bs, model_pars; 
        nsamp=20, window_buffer=2, target_times=1:2, smooth=false,           
        spatial_mod=exp_corD, obs_operator=unif_weighted_obs_operator_centroid,
        state_in_cov=true, cov_wt=0.2, tscov_pars=sqrt.(vrs)./10.0, 
        ar_phi=1.0, window_radius=100.0
    )
end
mem_after = Sys.maxrss() / 1024^2

println("\n" * "=" ^ 70)
println("✓ COMPLETE!")
println("=" ^ 70)
println("Time: $(round(fusion_time, digits=1))s")
println("Peak RSS: $(round(mem_after, digits=1)) MB")
println("Fusion memory: +$(round(mem_after - mem_before, digits=1)) MB")
println("Total: +$(round(mem_after - mem_start, digits=1)) MB from start")
println("\nOutput: $(size(fused_images)) (y,x,wavelength,time)")
println("=" ^ 70)

