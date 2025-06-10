module HyperSTARS

export organize_data
export scene_fusion_pmap
export create_data_dicts
export hyperSTARS_fusion_kr_dict
export woodbury_filter_kr
export smooth_series

export InstrumentData
export InstrumentGeoData

export cell_size
export get_centroid_origin_raster

export nanmean
export nanvar

export exp_cor
export mat32_cor
export mat52_cor
export exp_corD
export mat32_corD
export mat52_corD
export state_cov

export unif_weighted_obs_operator_centroid


# Write your package code here.
using Dates
using Rasters
using LinearAlgebra
using Distributions
using Statistics
using StatsBase
using SparseArrays
using BlockDiagonals
using Distances
import GaussianRandomFields.CovarianceFunction
import GaussianRandomFields.Matern
import GaussianRandomFields.apply
using GeoArrays
using MultivariateStats
using Kronecker
using ProgressMeter
using Random
using Interpolations
using KernelFunctions
using Distributed
#BLAS.set_num_threads(1)

include("resampling_utils.jl")
include("spatial_utils_ll.jl")
include("GP_utils.jl")

# T = Float64

struct KSModel{Float64}
    H::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end

struct HSModel{Float64}
    Hw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Hs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Vw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Vs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

### struct for instrument measurements and metadata
struct InstrumentData
    data::AbstractArray # n x w x T arrays of measurements
    bias::Union{Any,AbstractArray} # W, n x W, or n x W x T array of error biases (currently only implemented for W)
    uq::Union{Any,AbstractArray} # W, n x W, or n x W x T array of error variances (currently only implemented for W)
    spatial_resolution::AbstractVector # [rx,ry] vector of spatial resolution 
    dates::AbstractVector # vector of dates
    coords::AbstractArray # n x 2 array of spatial coordinates
    wavelengths::AbstractVector # w vector of spectral wavelengths
    rsr::Union{Dict,Any,AbstractArray} # Dict{w: wavelengths, rsr:rsr values} for discrete weights and wavelengths, or scalar common fwhm, or w length vector of fwhms
end

### struct for instrument geospatial data
struct InstrumentGeoData
    origin::AbstractVector # [rx,ry] vector of raster origin
    cell_size::AbstractVector # [rx,ry] vector of spatial resolution 
    ndims::AbstractVector # [nx,ny] vector of size of instrument grid
    fidelity::Int64 # [0,1,2] indicating 0: highest spatial res, 1: high spatial res, 2: coarse res
    dates::AbstractVector # vector of dates
    wavelengths::AbstractVector # w vector of spectral wavelengths
    rsr::Union{Dict,Any,AbstractArray} # Dict{w: wavelengths, rsr:rsr values} for discrete weights and wavelengths, or scalar common fwhm, or w length vector of fwhms
end

"kalman filter one-step recursion"
function woodbury_filter_kr(Ms::AbstractVector{<:HSModel}, 
                            ys::AbstractVector{<:AbstractArray{Float64}}, 
                            x_pred::AbstractVector{Float64}, 
                            P_pred::AbstractArray{Float64}) 

    
    K = length(Ms)
    N = size(x_pred,1)
    ns = size(Ms[1].Hs,2)
    nw = size(Ms[1].Hw,2)

    HViH = zeros(N,N)
    HVie = zeros(N)
    x_new = ones(N)
    P_new = zeros(N,N)
    F = zeros(N,N)

    x_predr = reshape(x_pred, (ns,nw))

    for i in 1:K

        Hw = @views Ms[i].Hw 
        HwtV = @views Hw'*inv(Ms[i].Vw)
        Hs = @views Ms[i].Hs
        HstV = @views Hs'*inv(Ms[i].Vs) 

        res_pred = @views ys[i] - Hs * x_predr * Hw' # innovation

        HViH .+= kronecker(HwtV*Hw, HstV*Hs)
        HVie .+= vec(HstV * res_pred * HwtV')
    end

    ## Woodbury Si 
    # @time begin
    #     S2 = inv(cholesky(P_pred)) + Symmetric(HViH);
    #     S3 = HViH*inv(cholesky(S2));
    # end

    S = P_pred[:,:]
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        LinearAlgebra.copytri!(S,'U')
    end
    S .+= HViH;

    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    HtRHIi = BLAS.symm('R', 'U', S, HViH)

    # @time F2 = P_pred*(I - HtRHIi);
    
    F .= P_pred;
    mul!(F, P_pred, HtRHIi, -1.0, 1.0);

    # @time x_new2 = x_pred .+ F*HVie; # filtering distribution mean

    x_new .= x_pred
    mul!(x_new, F, HVie, 1.0, 1.0)
    
    # @time P_new2 = (I - F*HViH)*P_pred; # filtering distribution covariance

    F .= F * HViH;
    P_new .= P_pred
    mul!(P_new, F, P_pred, -1.0, 1.0)

    return x_new, P_new
end

function smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs) 

    # These arrays start at the final smoothed (= filtered) state
    smoothed_means = [filtering_means[end]]
    smoothed_covs = [filtering_covs[end]]

    # First step that we are interested here in is i = nsteps - 1
    nsteps = length(predicted_means) # This was T previously
    t0 = 1 #FIXME ADDED BLINDLY

    for i ∈ nsteps:-1:(t0+1)
        # NB. filtering_covs[i] is P_{i-1|i-1}, predicted_covs[i] is P_{i|i-1}
        begin # C = filtering_covs[i] * F * inv(predicted_covs[i])
            CC = predicted_covs[i][:,:]
            LAPACK.potrf!('U', CC)
            LAPACK.potri!('U', CC)
            C = zeros(size(predicted_covs[i]))
            BLAS.symm!('R', 'U', 1., CC, filtering_covs[i]*F', 0., C)
        end
        x_smooth = filtering_means[i] .+ C * (smoothed_means[nsteps - i + 1] .- predicted_means[i])

        # P_smooth = filtering_covs[i] + C * (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        # Compute P_smooth = filtering_covs[i] + C *
        # (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        begin
            CC .= smoothed_covs[nsteps - i + 1] .- predicted_covs[i]
            D = BLAS.symm('R', 'U', CC, C) # D = C * CC
            CC .= filtering_covs[i]
            P_smooth = BLAS.gemm!('N', 'T', 1., D, C, 1., CC)
        end

        push!(smoothed_means, x_smooth)
        push!(smoothed_covs, P_smooth)
    end

    return reverse(smoothed_means), reverse(smoothed_covs)
end

function organize_data(full_ext::AbstractArray{<:Real}, 
                        inst_geodata::AbstractVector{InstrumentGeoData}, 
                        inst_data::AbstractVector{InstrumentData}, 
                        target_geodata::InstrumentGeoData, 
                        ss_xy::AbstractArray{<:Real}, 
                        ss_ij::AbstractArray{<:Real}, 
                        res_flag::AbstractVector{<:Real})

    ### Find measurements:
    measurements = Vector{InstrumentData}(undef, length(inst_geodata))

    for i in sortperm(res_flag, rev=true)
        if res_flag[i] == 2
            ins_ij = @views find_all_ij_ext(full_ext[1,:], full_ext[2,:], inst_geodata[i].origin, inst_geodata[i].cell_size, inst_geodata[i].ndims; inclusive=false)
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            ys = @views inst_data[i].data[CartesianIndex.(ins_ij[:,1],ins_ij[:,2]),:,:]
            measurements[i] = @views InstrumentData(ys,
                                inst_data[i].bias, 
                                inst_data[i].uq, 
                                abs.(inst_geodata[i].cell_size),
                                inst_geodata[i].dates,
                                ins_xy,
                                inst_geodata[i].wavelengths,
                                inst_geodata[i].rsr)
        elseif res_flag[i] == 1
            ins_ij = @views unique(find_nearest_ij_multi(ss_xy, inst_geodata[i].origin, inst_geodata[i].cell_size, inst_geodata[i].ndims),dims=1)
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            ys = @views inst_data[i].data[CartesianIndex.(ins_ij[:,1],ins_ij[:,2]),:,:]

            # @time ss_ij = unique(vcat(ss_ij,unique(vcat(map(x -> find_all_bau_ij(x, inst_geodata[i].cell_size, target_origin, target_csize), eachrow(ins_xy))...),dims=1)),dims=1)
            ss_ij = @views unique(vcat(ss_ij,find_all_bau_ij_multi(ins_xy, inst_geodata[i].cell_size, target_geodata.origin, target_geodata.cell_size, target_geodata.ndims)),dims=1)
            # @time blah = find_all_bau_ij_multi(ins_xy, inst_geodata[i].cell_size, target_origin, target_csize)
            # @time blah = find_all_bau_ij_multi(ins_xy, inst_geodata[i].cell_size, target_origin, target_csize)
            ss_xy = get_sij_from_ij(ss_ij, target_geodata.origin, target_geodata.cell_size)
            measurements[i] = @views InstrumentData(ys,
                                inst_data[i].bias, 
                                inst_data[i].uq, 
                                abs.(inst_geodata[i].cell_size),
                                inst_geodata[i].dates,
                                ins_xy,
                                inst_geodata[i].wavelengths,
                                inst_geodata[i].rsr)
        else
            ins_ij = @views unique(find_nearest_ij_multi(ss_xy, inst_geodata[i].origin, inst_geodata[i].cell_size,inst_geodata[i].ndims),dims=1)
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            ys = @views inst_data[i].data[CartesianIndex.(ins_ij[:,1],ins_ij[:,2]),:,:]
            measurements[i] = @views InstrumentData(ys,
                                    inst_data[i].bias, 
                                    inst_data[i].uq, 
                                    abs.(inst_geodata[i].cell_size),
                                    inst_geodata[i].dates,
                                    ins_xy,
                                    inst_geodata[i].wavelengths,
                                    inst_geodata[i].rsr)
        end
    end
    return measurements, ss_ij
end

"fusion model in single window"
#### TODOS
## only filtering capable so far, easy to add smoothing
## decide how to deal with uncertainties, inclined to return uncertainties through simulation
## integrate conditional sim methods from STARS
## right now returns estimates of latent states (PCA) not back transformed reflectances
## add smoothing step

function hyperSTARS_fusion_kr_dict(d,  
                target_wavelengths::AbstractVector{<:Real},
                spectral_mean::AbstractVector{<:Real},   
                B::AbstractArray{<:Real},
                target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1],
                smooth::Bool = false,
                spatial_mod::Function = mat32_corD,                                         
                obs_operator::Function = unif_weighted_obs_operator,
                state_in_cov::Bool = true,
                cov_wt::Real = 0.3,
                ar_phi = 1.0) 

    measurements = @views d[:measurements]
    target_coords = @views d[:target_coords]
    kp_ij = @views d[:kp_ij]
    prior_mean = @views d[:prior_mean]
    prior_var = @views d[:prior_var]
    model_pars = @views d[:model_pars]

    nbau = size(kp_ij,1)
    ni = size(measurements)[1] 
    # println(ni)
    nf = size(target_coords)[1] # number of target resolution grid cells

    p = size(B)[2]
    nnobs = Vector{Int64}(undef, ni)
    nwobs = Vector{Int64}(undef, ni)
    t0v = Vector{Int64}(undef, ni)
    ttv = Vector{Int64}(undef, ni)

    for i in 1:ni
        nnobs[i] = size(measurements[i].data)[1]
        nwobs[i] = size(measurements[i].data)[2]
        t0v[i] = measurements[i].dates[1]
        ttv[i] = measurements[i].dates[end]
    end

    t0 = minimum(t0v)
    tt = maximum(ttv)
    tp = maximum(target_times);
    tpl = minimum(target_times)

    if smooth
        times = minimum([t0,tpl]):maximum([tt,tp])
    else
        times = minimum([t0,tpl]):tp
    end

    nsteps = size(times)[1]
    
    data_kp = falses(ni,nsteps)
   
    ## build observation operator, stack observations and variances 
    Hws = Vector(undef,ni)
    Hms = Vector(undef,ni)
    Hss = Vector(undef,ni)

    for (i,x) in enumerate(measurements)
        Hss[i] = obs_operator(x.coords, target_coords, x.spatial_resolution) # kwargs for uniform needs :target_resolution, # kwargs for gaussian needs :scale, :p
        Hw = rsr_conv_matrix(x.rsr, x.wavelengths, target_wavelengths)
        Hws[i] = Hw*B
        Hms[i] = Hw*spectral_mean  
        data_kp[i,in(measurements[i].dates).(times)] .= true
    end

    Qs = Vector{Matrix{Float64}}(undef,size(model_pars)[1])
    dd = pairwise(Euclidean(1e-12), target_coords', dims=2)
    for (i,x) in enumerate(eachrow(model_pars))
        Qs[i] = x[1] .* spatial_mod(dd, x[2:end]) 
    end

    Q = Matrix(BlockDiagonal(Qs))

    ## Diagonal transition matrices
    F = UniformScaling(ar_phi)

    x0 = prior_mean[:] # don't need this but here to help with synergizing code later
    P0 = Diagonal(prior_var[:]) # just assuming diagonal C0

    filtering_means = Vector{Vector{Float64}}(undef, 0)
    predicted_means = Vector{Vector{Float64}}(undef, 0)
    filtering_covs = Vector{Matrix{Float64}}(undef, 0)
    predicted_covs = Vector{Matrix{Float64}}(undef, 0)
    push!(filtering_means, x0)
    push!(filtering_covs, P0)

    fused_image = zeros(nbau,p,size(target_times,1))
    fused_sd_image = zeros(nbau,p,size(target_times,1))

    kp_times = findall(times .∈ Ref(target_times))

    ## bunch of reusable memory in here...
    for (t,t2) in enumerate(times)
        if state_in_cov ## update this to weighted past
            # Xtt = reshape(filtering_means[t],(nf,p))
            Xtt = cat([reshape(x, (nf,p)) for x in filtering_means[1:t]]..., dims=3)
            Wt = cat([reshape(1.0 ./ sqrt.(diag(x)), (nf,p)) for x in filtering_covs[1:t]]...,dims=3)
            Wtn = sum(Wt, dims=3)

            Wt ./= Wtn
            Xtt .*= Wt

            ### ad hoc
            Qst = Vector{Matrix{Float64}}(undef,p)
            for (i,x) in enumerate(eachrow(model_pars))
                # dd = pairwise(Euclidean(1e-12),Xtt[:,i]',Xtt[:,i]',dims=2) + UniformScaling(1e-10)
                # phi = maximum([0.01,median(dd[:])])
                # Qst[i] = x[1] .* exp.(-dd./phi)
                Qst[i] = state_cov(Xtt[:,i,:]',x)
            end
        
            Qss = Matrix(BlockDiagonal(Qst))

            Qf = cov_wt .* Q .+ (1-cov_wt) .* Qss
        else
            Qf = Q
        end

        Ms = HSModel[]

        ys = Vector{Array{Float64}}()
        for x in findall(data_kp[:,t])
            yss = @views measurements[x].data[:,:,measurements[x].dates .== t2]
            ym = .!vec(any(isnan, yss; dims=2))
            Hs2 = Hss[x][ym,:]

            push!(Ms, HSModel(Hws[x], Hs2, Diagonal(measurements[x].uq[:]), 1.0*I(size(Hs2,1)), Qf, F))
            push!(ys,yss[ym,:] .- Hms[x]');
        end

        # Predictive mean and covariance here
        x_pred = F * filtering_means[t] # filtering_means[1], covs[1] is prior mean
        P_pred = F * filtering_covs[t] * F' + Qf
        push!(predicted_means, x_pred)
        push!(predicted_covs, P_pred)

        # Filtering is done here
        if length(ys) == 0
            push!(filtering_means, x_pred)
            push!(filtering_covs, P_pred)
        else
            x_new, P_new = woodbury_filter_kr(Ms, ys, x_pred, P_pred)
            push!(filtering_means, x_new)
            push!(filtering_covs, P_new)
        end
    end
    if smooth
        st = minimum(kp_times)
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means[st:end], predicted_covs[st:end], filtering_means[st:end], filtering_covs[st:end])
        for (ti,t2) in enumerate(kp_times .- st .+ 1)
            fused_image[:,:,ti] = @views reshape(smoothed_means[t2],(nf,p))[1:nbau,:] 
            fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(smoothed_covs[t2])),(nf,p))[1:nbau,:]
        end
    else
        for (ti,t2) in enumerate(kp_times)
            fused_image[:,:,ti] = @views reshape(filtering_means[t2+1],(nf,p))[1:nbau,:] 
            fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(filtering_covs[t2+1])),(nf,p))[1:nbau,:]
        end
    end    
    return kp_ij, fused_image, fused_sd_image           
end

function create_data_dicts( ii )
    k,l = inds[ii,:]

    ### find target partition given origin and (k,l)th partition coordinate
    window_bbox = bbox_from_centroid(window_origin .+ [k-1, l-1].*window_csize, window_csize)
    
    ### add buffer of window_buffer target pixels around target partition extent
    buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'

    ### find extent of overlapping instruments for each instrument
    all_exts = [Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size)) for x in inst_geodata]
    res_flag = [x.fidelity for x in inst_geodata] #0 highest res, 1 high res, 2 coarse res

    for i in findall(res_flag .== 2)
        exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[i].cell_size'
        push!(all_exts, exx)
    end

    ### finf full extent combining all instrument extents
    full_ext = merge_extents(all_exts, sign.(target_csize))

    ### Find all BAUs within target
    target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
    # t_xy = get_sij_from_ij(target_ij, target_origin, target_csize)

    ### Find all BAUs within target + buffer
    ss_ij = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
    # tb_xy = get_sij_from_ij(target_buffer_ij, target_origin, target_csize)

    ### instrument fidelity
    if any(res_flag .== 2)
        ### subsample BAUs within full extent of coarse pixels
        ss_ij = unique(vcat(ss_ij, sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)),dims=1)
    end

    ss_xy = get_sij_from_ij(ss_ij, target_origin, target_csize)

    ### Find measurements:
    measurements, ss_ij = organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)
    
    ### stack to ensure target partition coords are first in list for easy subsetting later
    bau_ij = unique(vcat(target_ij, ss_ij),dims=1)

    ### x,y coords for all baus
    bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)
    
    ### i,j indices for all baus
    bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])
    
    ### subset prior mean and var arrays to bau pixels
    prior_mean_sub = @views prior_mean[bau_ci,:][:]
    prior_var_sub = @views prior_var[bau_ci,:][:]
    
    tind = CartesianIndex.(target_ij[:,1], target_ij[:,2])

    d = Dict()
    d[:measurements] = measurements
    d[:target_coords] = bau_coords
    d[:kp_ij] = tind
    d[:prior_mean] = prior_mean_sub
    d[:prior_var] = prior_var_sub
    d[:model_pars] = model_pars

    return d
end

function scene_fusion_pmap(inst_data::AbstractVector{InstrumentData},
                      inst_geodata::AbstractVector{InstrumentGeoData},
                      window_geodata::InstrumentGeoData,
                      target_geodata::InstrumentGeoData,
                      spectral_mean::AbstractVector{<:Real},
                      prior_mean::AbstractArray{<:Real},
                      prior_var::AbstractArray{<:Real},
                      B::AbstractArray{<:Real},
                      model_pars::AbstractArray{<:Real};
                      nsamp::Integer = 100,
                      window_buffer::Integer = 2,
                      target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1], 
                      smooth::Bool = false,           
                      spatial_mod::Function = mat32_corD,                                           
                      obs_operator::Function = unif_weighted_obs_operator,
                      state_in_cov::Bool = true,
                      cov_wt::Real = 0.7,
                      ar_phi::Real = 1.0,
                      nb_coarse::Real = 2.0)

    ### define target extent and target + buffer extent
    window_csize = @views window_geodata.cell_size
    target_csize = @views target_geodata.cell_size
    window_origin = @views window_geodata.origin
    nwindows = @views window_geodata.ndims
    target_origin = @views target_geodata.origin
    target_waves = @views target_geodata.wavelengths
    target_ndims = @views target_geodata.ndims

    ni = length(inst_geodata)
    nsteps = size(target_times)[1]
    fused_image = zeros(target_ndims[1], target_ndims[2], size(B)[2], nsteps);
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], size(B)[2], nsteps);
    
    inds = hcat(repeat(1:nwindows[1], inner=nwindows[2]), repeat(1:nwindows[2], outer=nwindows[1]))
    # @showprogress for ii in 1:size(inds)[1]
    # @showprogress Threads.@threads for ii in 1:size(inds)[1]
    # Threads.@threads for ii in 1:size(inds)[1]
    
    nr = size(inds,1)

    # T = [ create_data_dicts( ii ) for ii in 1:nr ];

    T = []
    for ii in 1:nr
        k,l = inds[ii,:]

        ### find target partition given origin and (k,l)th partition coordinate
        window_bbox = bbox_from_centroid(window_origin .+ [k-1, l-1].*window_csize, window_csize)
        
        ### add buffer of window_buffer target pixels around target partition extent
        buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'
    
        ### find extent of overlapping instruments for each instrument
        all_exts = [Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size)) for x in inst_geodata]
        res_flag = [x.fidelity for x in inst_geodata] #0 highest res, 1 high res, 2 coarse res
    
        for i in findall(res_flag .== 2)
            exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[i].cell_size'
            push!(all_exts, exx)
        end
    
        ### finf full extent combining all instrument extents
        full_ext = merge_extents(all_exts, sign.(target_csize))
    
        ### Find all BAUs within target
        target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
        # t_xy = get_sij_from_ij(target_ij, target_origin, target_csize)
    
        ### Find all BAUs within target + buffer
        ss_ij = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
        # tb_xy = get_sij_from_ij(target_buffer_ij, target_origin, target_csize)
    
        ### instrument fidelity
        if any(res_flag .== 2)
            ### subsample BAUs within full extent of coarse pixels
            ss_ij = unique(vcat(ss_ij, sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)),dims=1)
        end
    
        ss_xy = get_sij_from_ij(ss_ij, target_origin, target_csize)
    
        ### Find measurements:
        measurements, ss_ij = organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)
        
        ### stack to ensure target partition coords are first in list for easy subsetting later
        bau_ij = unique(vcat(target_ij, ss_ij),dims=1)
    
        ### x,y coords for all baus
        bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)
        
        ### i,j indices for all baus
        bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])
        
        ### subset prior mean and var arrays to bau pixels
        prior_mean_sub = @views prior_mean[bau_ci,:][:]
        prior_var_sub = @views prior_var[bau_ci,:][:]
        
        tind = CartesianIndex.(target_ij[:,1], target_ij[:,2])
    
        d = Dict()
        d[:measurements] = measurements
        d[:target_coords] = bau_coords
        d[:kp_ij] = tind
        d[:prior_mean] = prior_mean_sub
        d[:prior_var] = prior_var_sub
        d[:model_pars] = model_pars

        push!(T,d)
    end

    result = @showprogress pmap(x -> hyperSTARS_fusion_kr_dict(x,  
                    target_waves, spectral_mean, B,
                    target_times, smooth, spatial_mod, 
                    obs_operator, state_in_cov, cov_wt, ar_phi) , T );
    
    for i in 1:nr
        @views fused_image[result[i][1],:,:] = result[i][2]
        @views fused_sd_image[result[i][1],:,:] = result[i][3]
    end
    
    return fused_image, fused_sd_image
end

end