module HyperSTARS

# HyperSTARS.jl
# =====================
# This module provides tools for multi-instrument, multi-resolution spatio-temporal data fusion using Kalman filtering and smoothing, with support for geospatial and spectral transformations.

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


"""
    struct KSModel{Float64}

Kalman Smoother Model for single-instrument or single-resolution data.

Fields:
- `H`: Observation operator (matrix or sparse matrix, or `Nothing`)
- `Q`: Process noise covariance matrix
- `F`: State transition matrix or scaling
"""
struct KSModel{Float64}
    H::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end

"""
    struct HSModel{Float64}

Hierarchical Smoother Model for multi-instrument, multi-resolution data fusion.

Fields:
- `Hw`: Spectral observation operator (matrix or sparse matrix, or `Nothing`)
- `Hs`: Spatial observation operator (matrix or sparse matrix, or `Nothing`)
- `Vw`: Spectral noise covariance (matrix, sparse, scaling, or `Nothing`)
- `Vs`: Spatial noise covariance (matrix, sparse, scaling, or `Nothing`)
- `Q`: Process noise covariance matrix
- `F`: State transition matrix or scaling
"""
struct HSModel{Float64}
    Hw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Hs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Vw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Vs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end


"""
    nanmean(x)
    nanmean(x, y)

Mean ignoring NaN values. If a second argument is given, computes the mean along the specified dimension(s).
"""
nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)


"""
    struct InstrumentData

Holds instrument measurements and associated metadata.

Fields:
- `data`: n × w × T array of measurements (n: locations, w: wavelengths, T: time)
- `bias`: Error bias array (W, n×W, or n×W×T)
- `uq`: Error variance array (W, n×W, or n×W×T)
- `spatial_resolution`: [rx, ry] vector of spatial resolution
- `dates`: Vector of observation dates
- `coords`: n × 2 array of spatial coordinates
- `wavelengths`: Vector of spectral wavelengths
- `rsr`: Spectral response (Dict, scalar, or array)
"""
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


"""
    struct InstrumentGeoData

Holds geospatial metadata for an instrument.

Fields:
- `origin`: [rx, ry] vector of raster origin
- `cell_size`: [rx, ry] vector of spatial resolution
- `ndims`: [nx, ny] size of instrument grid
- `fidelity`: Integer indicating spatial fidelity (0: highest, 1: high, 2: coarse)
- `dates`: Vector of observation dates
- `wavelengths`: Vector of spectral wavelengths
- `rsr`: Spectral response (Dict, scalar, or array)
"""
struct InstrumentGeoData
    origin::AbstractVector # [rx,ry] vector of raster origin
    cell_size::AbstractVector # [rx,ry] vector of spatial resolution 
    ndims::AbstractVector # [nx,ny] vector of size of instrument grid
    fidelity::Int64 # [0,1,2] indicating 0: highest spatial res, 1: high spatial res, 2: coarse res
    dates::AbstractVector # vector of dates
    wavelengths::AbstractVector # w vector of spectral wavelengths
    rsr::Union{Dict,Any,AbstractArray} # Dict{w: wavelengths, rsr:rsr values} for discrete weights and wavelengths, or scalar common fwhm, or w length vector of fwhms
end

"""
    woodbury_filter_kr(Ms, ys, x_pred, P_pred)

Kalman filter one-step recursion using the Woodbury matrix identity for efficient update.

Arguments:
- `Ms`: Vector of `HSModel` objects (one per instrument)
- `ys`: Vector of observation arrays (one per instrument)
- `x_pred`: Predicted state mean vector
- `P_pred`: Predicted state covariance matrix

Returns:
- `x_new`: Updated (filtered) state mean
- `P_new`: Updated (filtered) state covariance
"""
function woodbury_filter_kr(Ms::AbstractVector{<:HSModel}, 
                            ys::AbstractVector{<:AbstractArray{Float64}}, 
                            x_pred::AbstractVector{Float64}, 
                            P_pred::AbstractArray{Float64}) 
    # ...existing code...
    return x_new, P_new
end

"""
    smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)

Rauch-Tung-Striebel (RTS) smoother for state-space models.

Arguments:
- `F`: State transition matrix or scaling
- `predicted_means`: Vector of predicted state means
- `predicted_covs`: Vector of predicted state covariances
- `filtering_means`: Vector of filtered state means
- `filtering_covs`: Vector of filtered state covariances

Returns:
- `smoothed_means`: Vector of smoothed state means
- `smoothed_covs`: Vector of smoothed state covariances
"""
function smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs) 
    # ...existing code...
    return reverse(smoothed_means), reverse(smoothed_covs)
end

"""
    organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)

Organizes and aligns instrument data and metadata for fusion, matching measurements to spatial grid and updating BAU (basic areal unit) indices.

Arguments:
- `full_ext`: Full spatial extent array
- `inst_geodata`: Vector of `InstrumentGeoData` structs
- `inst_data`: Vector of `InstrumentData` structs
- `target_geodata`: Target geospatial metadata
- `ss_xy`: BAU coordinates
- `ss_ij`: BAU indices
- `res_flag`: Vector indicating instrument spatial fidelity

Returns:
- `measurements`: Vector of aligned `InstrumentData`
- `ss_ij`: Updated BAU indices
"""
function organize_data(full_ext::AbstractArray{<:Real}, 
                        inst_geodata::AbstractVector{InstrumentGeoData}, 
                        inst_data::AbstractVector{InstrumentData}, 
                        target_geodata::InstrumentGeoData, 
                        ss_xy::AbstractArray{<:Real}, 
                        ss_ij::AbstractArray{<:Real}, 
                        res_flag::AbstractVector{<:Real})
    # ...existing code...
    return measurements, ss_ij
end

"""
    hyperSTARS_fusion_kr_dict(d, target_wavelengths, spectral_mean, B; ...)

Performs spatio-spectral-temporal data fusion for a single window/partition using Kalman filtering (and optionally smoothing).

Arguments:
- `d`: Data dictionary (see `create_data_dicts`)
- `target_wavelengths`: Vector of target wavelengths
- `spectral_mean`: Mean spectrum for centering
- `B`: Spectral basis matrix
- `target_times`: Vector or range of target time steps (default: [1])
- `smooth`: If true, apply RTS smoothing (default: false)
- `spatial_mod`: Spatial covariance function (default: `mat32_corD`)
- `obs_operator`: Function to build spatial observation operator (default: `unif_weighted_obs_operator`)
- `state_in_cov`: If true, use state-dependent process covariance (default: true)
- `cov_wt`: Weight for process covariance blending (default: 0.3)
- `ar_phi`: AR(1) transition parameter (default: 1.0)

Returns:
- `kp_ij`: Indices of target BAUs
- `fused_image`: Array of fused state estimates
- `fused_sd_image`: Array of fused state standard deviations
"""
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
    # ...existing code...
    return kp_ij, fused_image, fused_sd_image           
end

"""
    create_data_dicts(ii)

Creates a data dictionary for a given window/partition index, assembling all required data and metadata for fusion.

Arguments:
- `ii`: Index of the window/partition

Returns:
- `d`: Dictionary with keys: `:measurements`, `:target_coords`, `:kp_ij`, `:prior_mean`, `:prior_var`, `:model_pars`
"""
function create_data_dicts( ii )
    # ...existing code...
    return d
end

"""
    scene_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata, spectral_mean, prior_mean, prior_var, B, model_pars; ...)

Performs parallelized spatio-spectral-temporal data fusion over a scene by partitioning into windows and applying fusion in each window.

Arguments:
- `inst_data`: Vector of `InstrumentData` structs
- `inst_geodata`: Vector of `InstrumentGeoData` structs
- `window_geodata`: Geospatial metadata for windowing
- `target_geodata`: Geospatial metadata for target grid
- `spectral_mean`: Mean spectrum for centering
- `prior_mean`: Prior mean array
- `prior_var`: Prior variance array
- `B`: Spectral basis matrix
- `model_pars`: Model parameter array

Keyword Arguments:
- `nsamp`: Number of BAU samples for coarse instruments (default: 100)
- `window_buffer`: Buffer size in target pixels (default: 2)
- `target_times`: Vector or range of target time steps (default: [1])
- `smooth`: If true, apply RTS smoothing (default: false)
- `spatial_mod`: Spatial covariance function (default: `mat32_corD`)
- `obs_operator`: Function to build spatial observation operator (default: `unif_weighted_obs_operator`)
- `state_in_cov`: If true, use state-dependent process covariance (default: true)
- `cov_wt`: Weight for process covariance blending (default: 0.7)
- `ar_phi`: AR(1) transition parameter (default: 1.0)
- `nb_coarse`: Number of coarse pixels for buffer (default: 2.0)

Returns:
- `fused_image`: Array of fused state estimates for the scene
- `fused_sd_image`: Array of fused state standard deviations for the scene
"""
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
    # ...existing code...
    return fused_image, fused_sd_image
end

end