using ArchGDAL
using Rasters
using Plots
using Statistics
using JLD2
using Distributed
using MultivariateStats
using LinearAlgebra

addprocs(8)

@everywhere using HyperSTARS

#### loading data
filename = "./data/synthetic_emit_hls_pace_data.jld2"

begin
    f = jldopen(filename)
    hls_array = f["hls_array"]
    emit_array = f["emit_array"]
    pace_array = f["pace_array"] ## 4 day arrays, nx x ny x nw x nt
    target_array = f["target_array"]

    hls_times = f["hls_times"] 
    emit_times = f["emit_times"]
    pace_times = f["pace_times"] ## vectors of nt "times" (or days)

    hls_csize = f["hls_csize"] 
    emit_csize = f["emit_csize"] 
    pace_csize = f["pace_csize"] ## spatial resolution of pixels, negative means coordinate is descending

    ## convert top-left origin to centroid
    hls_origin = f["hls_origin"] .+ hls_csize/2
    emit_origin = f["emit_origin"] .+ emit_csize/2
    pace_origin = f["pace_origin"] .+ pace_csize/2 ## spatial location of image "origin"

    fwhm_pace = f["fwhm_pace"] ## vector of fwhm of gaussian spectral response kernel for each pace wavelength
    hls_srf = f["hls_srf"] ## Dict of HLS wavelengths and weights for discretized multispectral spectral response function

    hls_waves = f["hls_waves"] 
    emit_waves = f["emit_waves"]
    pace_waves = f["pace_waves"] ## wavelengths (center channel) for each spectral band

    #### add 1km pace
    pace1km_csize = [1000.0, -1000.0]
    pace1km_origin = f["pace_origin"] .+ pace1km_csize/2

    close(f)
end

hls_times[1] = 1

#### add 1km pace
pace_raster = Raster(pace_array, dims = (X(1:size(pace_array,1)), Y(1:size(pace_array,2)), Z(1:size(pace_array,3)), Band(1:size(pace_array,4))))
pace1km_array = Array{Float64}(aggregate(mean, pace_raster, (X(2), Y(2))))

###### Fusion setup

### spatial size of each instrument array
hls_ndims = collect(size(hls_array)[1:2])
emit_ndims = collect(size(emit_array)[1:2])
pace_ndims = collect(size(pace_array)[1:2])
pace1km_ndims = collect(size(pace1km_array)[1:2])

### get spectral basis functions using EMIT (PCA) 
emit_rt = reshape(permutedims(emit_array, (1,2,4,3)),(size(emit_array)[1]*size(emit_array)[2]*size(emit_array)[4],size(emit_array)[3]))'
emit_rt2 = emit_rt[:,.!vec(any(isnan, emit_rt; dims=1))]
mm = mean(emit_rt2, dims=2)[:]
sx = std(emit_rt2, dims=2)[:]
Xt = (emit_rt2 .- mm) ./ sx
pca = MultivariateStats.fit(PCA, Xt; pratio=0.995)
B = projection(pca)
vrs = principalvars(pca)
scs = predict(pca, Xt)

### create geospatial structs for each instrument
# 4th element is spatial "fidelity": 0 for highest (target) spatial res, 1 for high but not target, 2 for coarse res
hls_geodata = InstrumentGeoData(hls_origin, hls_csize, hls_ndims, 0, hls_times, hls_waves, hls_srf)
emit_geodata = InstrumentGeoData(emit_origin, emit_csize, emit_ndims, 1, emit_times, emit_waves, [1.0])
pace_geodata = InstrumentGeoData(pace_origin, pace_csize, pace_ndims, 2, pace_times, pace_waves, fwhm_pace)
pace1km_geodata = InstrumentGeoData(pace1km_origin, pace1km_csize, pace1km_ndims, 2, pace_times, pace_waves, fwhm_pace)

## create vector of InstrumentGeoData for all instruments
inst30m_geodata = [emit_geodata, hls_geodata, pace1km_geodata]

## define moving window sizes 
scf = 5 #scf: number of target pixels to create scf x scf partitions
nwindows = Int.(ceil.(hls_ndims./scf)) # number of windows nested in target scene (still need to do edge cases)
target_ndims = hls_ndims # define prediction grid size as subset of scene containing all nested partitions (i.e. cut-off incomplete edges)
window30m_geodata = InstrumentGeoData(hls_origin .+ (scf - 1)/2*hls_csize, scf*hls_csize, nwindows, 0, [1], emit_waves, 7.5)
target30m_geodata = InstrumentGeoData(hls_origin, hls_csize, target_ndims, 0, [1], emit_waves, 7.5)

hls_data = InstrumentData(hls_array, zeros(size(hls_array)[3]), 1e-5*ones(size(hls_array)[3]), abs.(hls_csize), hls_times, [1,1], hls_waves, hls_srf)
emit_data = InstrumentData(emit_array, zeros(size(emit_array)[3]), 1e-5*ones(size(emit_array)[3]), abs.(emit_csize), emit_times, [1,1], emit_waves, 1.0)
pace_data = InstrumentData(pace_array, zeros(size(pace_array)[3]), 1e-5*ones(size(pace_array)[3]), abs.(pace_csize), pace_times, [1,1], pace_waves, fwhm_pace)
pace1km_data = InstrumentData(pace1km_array, zeros(size(pace1km_array)[3]), 1e-5*ones(size(pace1km_array)[3]), abs.(pace1km_csize), pace_times, [1,1], pace_waves, fwhm_pace)

data30m_list = [emit_data, hls_data, pace1km_data];

Bs = B .* sx[:]
pmean = zeros((target_ndims...,size(B)[2])) # prior mean array
pvar = ones((target_ndims...,size(B)[2])) # prior var array
model_pars = hcat(10.0*ones(size(B)[2]),500*ones(size(B)[2]),1e-10*ones(size(B)[2]),1.5*ones(size(B)[2])); # p x 4 spatial model parameters (var, range, nug, smoothness)
model_pars[:,1] = vrs

### run data fusion, currently returns fused predictions in pca space (uncertainty code is pending)
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

@time fused_images, fused_sd_images = scene_fusion_pmap(data30m_list,
            inst30m_geodata,
            window30m_geodata,
            target30m_geodata,
            mm, ## spectral mean
            pmean, ## mean-zero prior mean
            pvar, ## diagonal prior var
            Bs, ## spectral basis function matrix
            model_pars; 
            nsamp=50,
            window_buffer = 3,
            target_times = 1:4, 
            smooth = false,           
            spatial_mod = mat32_corD,                                           
            obs_operator = unif_weighted_obs_operator_centroid,
            state_in_cov=false,
            nb_coarse=1.0);

## remove workers
rmprocs(workers())

k=1
heatmap(fused_images[:,:,1,4],title="t=$(k),30m",size=(600,600))


