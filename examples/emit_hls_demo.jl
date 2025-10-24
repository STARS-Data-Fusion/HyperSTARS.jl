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
using HyperSTARS
using LinearAlgebra
using Plots
using Glob
Rasters.checkmem!(false)

addprocs(8) ## workers (using Distributed package)

@everywhere using HyperSTARS

#### loading HLS landsat data from netcdf
filename = "HLSS30.020_30m_aid0001_15N.nc"

B01 = ncread(filename, "B01",) # coastal aerosal
B02 = ncread(filename, "B02",) # blue
B03 = ncread(filename, "B03",) # green
B04 = ncread(filename, "B04",) # red
B05 = ncread(filename, "B05",) # red edge 1
B06 = ncread(filename, "B06",) # red edge 2
B07 = ncread(filename, "B07",) # red edge 3
B08 = ncread(filename, "B08",) # nir
B8A = ncread(filename, "B8A",) # red edge 4
B09 = ncread(filename, "B09",) # water vapor
B10 = ncread(filename, "B10",) # cirrus
B11 = ncread(filename, "B11",) # swir1
B12 = ncread(filename, "B12",) # swir2
fmask = ncread(filename, "Fmask")
xdim = ncread(filename, "xdim")
ydim = ncread(filename, "ydim")
times = Day.(ncread(filename, "time")) .+ Date(2015, 12, 1) #days since 2015-12-01 00:00:00          
hls_crs = ncread(filename, "crs")

## ignore water vapor and cirrus bands for EMIT fusion
hls_s30_dat = float.(cat([B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]..., dims=4));
hls_s30_dat[hls_s30_dat .== -9999] .= NaN;
hls_s30_dat *= 0.0001;

# Create a clean pixel mask (no clouds, shadows, or adjacent)
clean = (fmask .& 14) .== 0;
hls_s30_dat[.!clean,:] .= NaN
hls_s30_datp = permutedims(hls_s30_dat, (1,2,4,3))

hls_s30_raster = Raster(hls_s30_datp, dims=(X(xdim), Y(ydim), Band(["Coastal Aerosal","Blue","Green","Red","RedEdge1","RedEdge2","RedEdge3","NIR","RedEdge4","SWIR1","SWIR2"]),Ti(times)))

#### loading sentinel data from netcdf
filename = "HLSL30.020_30m_aid0001_15N.nc"

B01 = ncread(filename, "B01",) # coastal aerosal
B02 = ncread(filename, "B02",) # blue
B03 = ncread(filename, "B03",) # green
B04 = ncread(filename, "B04",) # red
B05 = ncread(filename, "B05",) # nir
B06 = ncread(filename, "B06",) # swir1
B07 = ncread(filename, "B07",) # swir2

fmask = ncread(filename, "Fmask")
xdim = ncread(filename, "xdim")
ydim = ncread(filename, "ydim")
times = Day.(ncread(filename, "time")) .+ Date(2013, 4, 1) # days since 2013-04-01

hls_l30_dat = float.(cat([B01, B02, B03, B04, B05, B06, B07]..., dims=4));
hls_l30_dat[hls_l30_dat .== -9999] .= NaN;
hls_l30_dat *= 0.0001;

# Create a clean pixel mask (no clouds, shadows, or adjacent)
clean = (fmask .& 14) .== 0;
hls_l30_dat[.!clean,:] .= NaN
hls_l30_datp = permutedims(hls_l30_dat, (1,2,4,3));

hls_l30_raster = Raster(hls_l30_datp, dims=(X(xdim), Y(ydim), Band(["Coastal Aerosal","Blue","Green","Red","NIR","SWIR1","SWIR2"]),Ti(times)))

###### spectral response functions for landsat (L30) and sentinel (S30)
HLS_L30_srf, srf_names_l30 = readdlm("HLS_L30_srf.csv", ',', header=true)
HLS_S30_srf, srf_names_s30 = readdlm("HLS_S30_srf.csv", ',', header=true)

## HLS uses the landsat srf for matching bands (all but red edge)
HLS_S30_srf[HLS_S30_srf[:,1] .∈ Ref(HLS_L30_srf[:,1]),[2,3,4,5,10,13,14]] .= HLS_L30_srf[:,[2,3,4,5,6,8,9]]

## normalize to weights
HLS_S30_srf[:,2:end] .= HLS_S30_srf[:,2:end] ./ sum(HLS_S30_srf[:,2:end],dims=1)
HLS_L30_srf[:,2:end] .= HLS_L30_srf[:,2:end] ./ sum(HLS_L30_srf[:,2:end],dims=1)

HLS_L30_srf2 = HLS_L30_srf[.!(sum(HLS_L30_srf[:,2:end],dims=2).==0)[:],:]
HLS_S30_srf2 = HLS_S30_srf[.!(sum(HLS_S30_srf[:,2:end],dims=2).==0)[:],:]

#### loading EMIT data from EMIT directory
emit_files = glob("*.nc", "EMIT")

function emit_data_organize(filename)
    mask_filename = replace(filename, "_RFL_" => "_MASK_")
    mask_filename = replace(mask_filename, "EMIT/" => "EMIT/masks/")

    datetime_str = match(r"\d{8}", filename).match
    dt = DateTime(datetime_str, "yyyymmdd")
    ddate = Date(dt)

    refl = ncread(filename, "reflectance");
    lat = ncread(filename, "lat");
    lon = ncread(filename, "lon");
    fwhm = ncread(filename, "fwhm");
    wavelengths = ncread(filename, "wavelengths");
    good_wavelengths = ncread(filename, "good_wavelengths");
    refl_mask = ncread(mask_filename, "mask");

    refl[refl .== -9999] .= NaN
    good_wavelengths[143:150] .= 0
    good_wavelengths[275:end] .= 0

    refl[refl_mask[:,:,8].==1,:] .= NaN
    refl[:,:,good_wavelengths.==0] .= NaN

    reg_lon = range(minimum(lon), step=round(diff(lon)[1],digits=12), length=length(lon))
    reg_lat = range(maximum(lat), step=round(diff(lat)[1],digits=12), length=length(lat))

    refl2 = reshape(refl, (size(refl)...,1))
    refl_raster = Raster(refl2, (X(reg_lon), Y(reg_lat), Z(wavelengths), Ti([ddate])),crs = EPSG(4326), missingval=NaN)
    refl_raster_utm = resample(refl_raster, crs = EPSG(32615), missingval=NaN)
    refl_raster_utm[ismissing.(refl_raster_utm)] .= NaN

    return refl_raster_utm, wavelengths, Int.(good_wavelengths), fwhm
end

emit_raster_list = []
fwhm_list = []
wavelengths_list = []
good_wavelengths_list = []
for x in emit_files
    refl_raster_utm, wavelengths, good_wavelengths, fwhm = emit_data_organize(x)
    push!(emit_raster_list, refl_raster_utm)
    push!(fwhm_list, fwhm)
    push!(wavelengths_list, wavelengths)
    push!(good_wavelengths_list, good_wavelengths)
end

emit_raster = cat(emit_raster_list...,dims=4)

###### Fusion setup
emit_origin = get_centroid_origin_raster(emit_raster)
hls_s30_origin = get_centroid_origin_raster(hls_s30_raster)
hls_l30_origin = get_centroid_origin_raster(hls_l30_raster)

emit_ndims = collect(size(emit_raster)[1:2])
hls_s30_ndims = collect(size(hls_s30_raster)[1:2])
hls_l30_ndims = collect(size(hls_l30_raster)[1:2])

emit_csize = collect(cell_size(emit_raster))
hls_s30_csize = collect(cell_size(hls_s30_raster))
hls_l30_csize = collect(cell_size(hls_l30_raster))

good_wavelengths = good_wavelengths_list[1]
emit_waves = wavelengths_list[1][good_wavelengths.==1]

hls_l30_waves = [round(mean(HLS_L30_srf2[x .> 0,1])) for x in eachcol(HLS_L30_srf2[:,[2:6...,8,9]])]
hls_s30_waves = [round(mean(HLS_S30_srf2[x .> 0,1])) for x in eachcol(HLS_S30_srf2[:,[2:10...,13,14]])]

hls_l30_waves_xmin = [round(minimum(HLS_L30_srf2[x .> 0,1])) for x in eachcol(HLS_L30_srf2[:,[2:6...,8,9]])]
hls_l30_waves_xmax = [round(maximum(HLS_L30_srf2[x .> 0,1])) for x in eachcol(HLS_L30_srf2[:,[2:6...,8,9]])]
hls_s30_waves_xmin = [round(minimum(HLS_S30_srf2[x .> 0,1])) for x in eachcol(HLS_S30_srf2[:,[2:10...,13,14]])]
hls_s30_waves_xmax = [round(maximum(HLS_S30_srf2[x .> 0,1])) for x in eachcol(HLS_S30_srf2[:,[2:10...,13,14]])]

### create spectral response component, taken by rsr_conv_matrix()
fwhm_emit = fwhm_list[1][good_wavelengths.==1]
S30_srf = Dict(:w => HLS_S30_srf2[:,1], :rsr => HLS_S30_srf2[:,[2:10...,13,14]]')
L30_srf = Dict(:w => HLS_L30_srf2[:,1], :rsr => HLS_L30_srf2[:,[2:6...,8,9]]')

## observed dates
emit_dates = emit_raster.dims[4].val[:]
hls_l30_dates = hls_l30_raster.dims[4].val[:]
hls_s30_dates = hls_s30_raster.dims[4].val[:]

## sequence of dates from start/end observed
all_dates = minimum([emit_dates..., hls_l30_dates..., hls_s30_dates...]):maximum([emit_dates..., hls_l30_dates..., hls_s30_dates...])

## convert to t=1:T times
emit_times = findall(all_dates .∈ Ref(emit_dates))
hls_l30_times = findall(all_dates .∈ Ref(hls_l30_dates))
hls_s30_times = findall(all_dates .∈ Ref(hls_s30_dates))

## rasters to data arrays, nx x ny x nw x T
hls_l30_array = Array(hls_l30_raster);
hls_s30_array = Array(hls_s30_raster);
emit_array = Array(emit_raster[:,:,good_wavelengths.==1,:]);

### get basis functions using EMIT (PCA) 
emit_rt = reshape(permutedims(emit_array, (1,2,4,3)),(size(emit_array)[1]*size(emit_array)[2]*size(emit_array)[4],size(emit_array)[3]))'
emit_rt2 = emit_rt[:,.!vec(any(isnan, emit_rt; dims=1))]
mm = mean(emit_rt2, dims=2)[:]
sx = std(emit_rt2, dims=2)[:]
Xt = (emit_rt2 .- mm) ./ sx
pca = MultivariateStats.fit(PCA, Xt; pratio=0.999)
B = projection(pca)
vrs = principalvars(pca)
scs = predict(pca, Xt)

## compute scores
scs_full = zeros(size(scs,1), size(emit_rt,2))
scs_full .= NaN
scs_full[:,.!vec(any(isnan, emit_rt; dims=1))] = scs

scs_rt = reshape(scs_full',(size(permutedims(emit_array, (1,2,4,3)))[1:3]...,size(scs,1)))

## compute temporal difference of emit scores
scs_diff = diff(scs_rt, dims=3)

## function to compute pairwise temporal differences
function g(x)
    n = length(x)
    dd = []
    for i in 1:(n-1)
        for j in (i+1):n
            push!(dd, x[j] - x[i])
        end
    end
    return dd
end

## compute variance of differences for each score (used to get ballpark value for innovation variance)
tdiffs = g(emit_times)
temp_vars = []
for i in 1:size(scs_rt,4)
    xx = mapslices(g,scs_rt[:,:,:,i],dims=3)
    all_diffs = xx ./ sqrt.(reshape(tdiffs, (1,1,size(xx,3))))
    push!(temp_vars, var(filter(!isnan,all_diffs)))    
end

### create geospatial structs for each instrument
# 4th element is spatial "fidelity": 0 for highest (target) spatial res, 1 for high but not target, 2 for coarse res (like PACE)
hls_l30_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, hls_l30_ndims, 0, hls_l30_times, hls_l30_waves, L30_srf)
hls_s30_geodata = InstrumentGeoData(hls_s30_origin, hls_s30_csize, hls_s30_ndims, 0, hls_s30_times, hls_s30_waves, S30_srf)
emit_geodata = InstrumentGeoData(emit_origin, emit_csize, emit_ndims, 1, emit_times, emit_waves, [0.1])

## create vector of InstrumentGeoData for all instruments
inst30m_geodata = [emit_geodata, hls_l30_geodata, hls_s30_geodata]

## define moving window sizes 
scf = 4 #scf: number of target pixels to create scf x scf partitions
target_ndims = hls_l30_ndims # define prediction grid size 

target_ndims = [50,50] # subset for testing
nwindows = Int.(ceil.(target_ndims./scf)) # number of windows covering target scene 

### geospatial structs for window grid and target grid
window30m_geodata = InstrumentGeoData(hls_l30_origin .+ (scf - 1)/2*hls_l30_csize, scf*hls_l30_csize, nwindows, 0, [1], emit_waves, fwhm_emit)
target30m_geodata = InstrumentGeoData(hls_l30_origin, hls_l30_csize, target_ndims, 0, [1], emit_waves, [0.1])

## create data struct objects
hls_s30_data = InstrumentData(hls_s30_array, zeros(size(hls_s30_array)[3]), 1e-6*ones(size(hls_s30_array)[3]), abs.(hls_s30_csize), hls_s30_times, [1,1], hls_s30_waves, S30_srf)
hls_l30_data = InstrumentData(hls_l30_array, zeros(size(hls_l30_array)[3]), 1e-6*ones(size(hls_l30_array)[3]), abs.(hls_l30_csize), hls_l30_times, [1,1], hls_l30_waves, L30_srf)
emit_data = InstrumentData(emit_array, zeros(size(emit_array)[3]), 1e-6*ones(size(emit_array)[3]), abs.(emit_csize), emit_times, [1,1], emit_waves, fwhm_emit)

## vector of InstrumentData for all instruments, order must match inst30m_geodata order!
data30m_list = [emit_data, hls_l30_data, hls_s30_data];


Bs = B .* sx[:] # rescale pc basis 
pmean = zeros((target_ndims...,size(B)[2])) # prior mean array (nx x ny x np) where np is number of PCs
pvar = ones((target_ndims...,size(B)[2])) .* reshape(2*temp_vars, (1,1,size(Bs,2))) # prior var array (nx x ny x np)

## setting spatial variance parameters (not estimated in this demo and not spatially varying)
model_pars = ones((nwindows...,size(B)[2],4)) .* reshape([0.1,100,1e-10,1.5], (1,1,1,4)) # covariance parameters (nx x ny x np x 4), last dimension is [spatial var, length scale, nugget, smoothness]

## set spatial var for each PC component as estimated by temp_var (variance of differences of EMIT scores)
for (i,x) in enumerate(temp_vars)
    model_pars[:,:,i,1] .= x
end

### run data fusion, currently returns fused predictions in pca space
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)

## spatial only variance
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
            target_times = 1:63, 
            smooth = false,           
            spatial_mod = exp_corD,                                           
            obs_operator = unif_weighted_obs_operator_centroid,
            state_in_cov=false,
            cov_wt=0.7,
            tscov_pars = sqrt.(vrs) ./ 10.0, 
            ar_phi=1.0,
            window_radius=1000.0);

## setting state_in_cov = true computes innovation covariance as Q = alpha * C_s(s,s') + (1-alpha)*C_x(x_{1:t-1}(s), x_{1:t-1}(s'))
@time fused_tscov_images, fused_tscov_sd_images = scene_fusion_pmap(data30m_list,
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
            target_times = 1:63, 
            smooth = false,           
            spatial_mod = exp_corD,                                           
            obs_operator = unif_weighted_obs_operator_centroid,
            state_in_cov=true,
            cov_wt=0.2,
            tscov_pars = sqrt.(vrs) ./ 10.0, 
            ar_phi=1.0,
            window_radius=1000.0);


## some quick plotting            
heatmap(fused_images[:,:,1,12])
heatmap(fused_tscov_images[:,:,1,12])


plot(emit_waves,Bs*fused_images[20,30,:,3] .+ mm); ## backtransform PC to get full fused spectra: y = Bs * fused + mean, where mean is mm
plot!(emit_waves,Bs*fused_tscov_images[20,30,:,3] .+ mm);
scatter!(hls_l30_waves, hls_l30_array[20,30,:,2])
scatter!(hls_s30_waves, hls_s30_array[20,30,:,2])

plot(emit_waves,Bs*fused_images[20,30,:,9] .+ mm, linewidth=2)
scatter!(hls_s30_waves, hls_s30_array[20,30,:,4])


plot(emit_waves, Bs*fused_images[20,20,:,:] .+ mm, legend=false)
scatter!(hls_l30_waves, hls_l30_array[20,20,:,2:end], legend=false)
scatter!(hls_s30_waves, hls_s30_array[20,20,:,2:end], legend=false)

ii=20;jj=20
## find emit overlapping ii,jj target cell
target_x = hls_l30_raster.dims[1][ii]
target_y = hls_l30_raster.dims[2][jj]
nearest_x = emit_raster.dims[1][Near(target_x)]
nearest_y = emit_raster.dims[2][Near(target_y)]
ie = findfirst(==(nearest_x), collect(dims(emit_raster, X)))
je = findfirst(==(nearest_y), collect(dims(emit_raster, Y)))


scatter(hls_l30_waves, hls_l30_array[ii,jj,:,:], color=:red, alpha=0.7)
scatter!(hls_s30_waves, hls_s30_array[ii,jj,:,:], color=:green, alpha=0.7)
scatter!(emit_waves, emit_array[ie,je,:,:], markersize=2, color=:blue, alpha=0.7)
plot!(emit_waves, Bs*fused_images[ii,jj,:,:] .+ mm, color=:magenta, linewidth=2, legend=false)


data_dates = findall(all_dates .∈ Ref([hls_l30_dates..., hls_s30_dates..., emit_dates[1:end-1]...]))
anim = @animate for i in data_dates
    plot(emit_waves, Bs*fused_images[ii,jj,:,i] .+ mm, title="t=$(i)", linewidth=2, label="Fused",ylim=(0,0.65))
    kpl = findall(hls_l30_times .== i)
    if length(kpl) > 0
        plot!([hls_l30_waves_xmin[1], hls_l30_waves_xmax[1]], [hls_l30_array[ii,jj,1,kpl][1], hls_l30_array[ii,jj,1,kpl][1]],
            linewidth=5,
            color=:red, label="HLS L30", alpha=0.7)
        for j in 2:length(hls_l30_waves_xmin)
            plot!([hls_l30_waves_xmin[j], hls_l30_waves_xmax[j]], [hls_l30_array[ii,jj,j,kpl][1], hls_l30_array[ii,jj,j,kpl][1]],
                linewidth=5, color=:red,label=false, alpha=0.7)
        end
    end
    kps = findall(hls_s30_times .== i)
    if length(kps) > 0
        plot!([hls_s30_waves_xmin[1], hls_s30_waves_xmax[1]], [hls_s30_array[ii,jj,1,kps][1], hls_s30_array[ii,jj,1,kps][1]],
            linewidth=5,
            color=:green, label="HLS S30", alpha=0.7)
        for j in 2:length(hls_s30_waves_xmin)
            plot!([hls_s30_waves_xmin[j], hls_s30_waves_xmax[j]], [hls_s30_array[ii,jj,j,kps][1], hls_s30_array[ii,jj,j,kps][1]],
                linewidth=5, color=:green,label=false, alpha=0.7)
        end
    end
    scatter!(emit_waves,emit_array[ie,je,:,emit_times .== i], markersize=2, label="EMIT")
end
# scatter!(hls_l30_waves,hls_l30_array[ii,jj,:,hls_l30_times .== i], markersize=3, label="HLS L30")
# scatter!(hls_s30_waves,hls_s30_array[ii,jj,:,hls_s30_times .== i], markersize=3, label="HLS S30")

gif(anim, "animation.gif", fps=2)