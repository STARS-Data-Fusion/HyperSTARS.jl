"""
Module HyperSTARS

This module provides a framework for hyperspectral spatio-temporal adaptive resolution system (HyperSTARS)
data fusion. It combines measurements from multiple instruments with varying spatial and spectral
resolutions into a single, more refined, high-resolution product. The fusion process is based on
Kalman filtering and smoothing techniques, leveraging Kronecker products and the Woodbury Matrix
Identity for computational efficiency, especially in high-dimensional spatio-spectral spaces.

Key functionalities include:
- Organization and preprocessing of diverse instrument measurement and geospatial data.
- Construction of state-space models for underlying physical processes.
- Implementation of Kalman filtering and smoothing for state estimation.
- Parallel processing capabilities for large scene fusion.
"""
module HyperSTARS

export organize_data
export scene_fusion_pmap
export create_data_dicts
export hyperSTARS_fusion_kr_dict
export woodbury_filter_kr
export woodbury_filter_kr!
export smooth_series

export InstrumentData
export InstrumentGeoData

export cell_size # Assuming this is defined in spatial_utils_ll.jl and re-exported
export get_centroid_origin_raster # Assuming this is defined in spatial_utils_ll.jl and re-exported

export nanmean
export nanvar # nanvar was exported but not defined in the original code, so adding a basic one here

export exp_cor
export mat32_cor
export mat52_cor
export exp_corD
export mat32_corD
export mat52_corD
export state_cov

export unif_weighted_obs_operator_centroid # Assuming this is defined in spatial_utils_ll.jl and re-exported


# --- Package Imports ---
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
using Kronecker # For efficient handling of spatio-spectral products
using ProgressMeter # For displaying progress bars
using Random
using Interpolations
using KernelFunctions
using Distributed # For parallel processing with pmap

# BLAS.set_num_threads(1) # This line is commented out, potentially for performance tuning


# --- Include Helper Files ---
# These files are assumed to contain helper functions related to resampling,
# spatial utilities (like coordinate conversions, grid operations), and
# Gaussian Process (GP) related functions (like covariance kernels).
include("resampling_utils.jl")
include("spatial_utils_ll.jl") # Contains functions like find_all_ij_ext, get_sij_from_ij, find_nearest_ij_multi, find_all_bau_ij_multi, bbox_from_centroid, find_overlapping_ext, merge_extents, sobol_bau_ij, unif_weighted_obs_operator
include("GP_utils.jl") # Contains functions like exp_corD, mat32_corD, mat52_corD, state_cov, rsr_conv_matrix

# --- Custom Statistical Functions ---

"""
    nanmean(x)

Computes the mean of an array `x`, excluding `NaN` (Not a Number) values.
"""
nanmean(x) = mean(filter(!isnan,x))

"""
    nanmean(x, y)

Computes the mean along a specified dimension `y` of an array `x`, excluding `NaN` values.
"""
nanmean(x,y) = mapslices(nanmean,x,dims=y)

"""
    nanvar(x)

Computes the variance of an array `x`, excluding `NaN` (Not a Number) values.
"""
nanvar(x) = var(filter(!isnan, x)) # Added nanvar as it was exported but not defined


# --- State-Space Model Structures ---

"""
    KSModel{Float64}

Represents a standard Kalman filter state-space model.

# Fields
- `H`: Observation matrix. Maps the state space to the observation space.
- `Q`: Process noise covariance matrix. Describes the uncertainty in the state transition.
- `F`: State transition matrix. Describes how the state evolves from one time step to the next.
"""
struct KSModel{Float64}
    H::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end

"""
    HSModel{Float64}

Represents a Hyperspectral STARS (Spatio-Temporal Adaptive Resolution System) model,
an extension of the Kalman filter model tailored for spatio-spectral data fusion.
It explicitly separates observation matrices and noise covariances for
wavelength (spectral) and spatial components, often leveraging Kronecker product structures.

# Fields
- `Hw`: Wavelength-specific observation matrix. Maps latent spectral components to observed wavelengths.
- `Hs`: Spatial-specific observation matrix. Maps latent spatial components to observed spatial locations.
- `Vw`: Wavelength-specific observation noise covariance (e.g., diagonal, or `UniformScaling`).
- `Vs`: Spatial-specific observation noise covariance (e.g., diagonal, or `UniformScaling`).
- `Q`: Process noise covariance matrix for the latent state.
- `F`: State transition matrix for the latent state.
"""
struct HSModel{Float64}
    Hw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Hs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},Nothing}
    Vw::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Vs::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64},Nothing}
    Q::AbstractMatrix{Float64}
    F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}
end


# --- Instrument Data Structures ---

"""
    InstrumentData

A structure to hold measurements and metadata for a single instrument.

# Fields
- `data::AbstractArray`: An `n x w x T` array of measurements, where `n` is spatial samples, `w` is wavelengths, and `T` is time steps.
- `bias::Union{Any,AbstractArray}`: Error biases, can be a scalar, `W` (wavelengths), `n x W`, or `n x W x T` array. Currently, only implemented for `W`.
- `uq::Union{Any,AbstractArray}`: Uncertainty quantification (e.g., error variances). Can be a scalar, `W`, `n x W`, or `n x W x T` array. Currently, only implemented for `W`.
- `spatial_resolution::AbstractVector`: A `[rx, ry]` vector representing the spatial resolution.
- `dates::AbstractVector`: A vector of dates corresponding to the time steps of the measurements.
- `coords::AbstractArray`: An `n x 2` array of spatial coordinates for each measurement.
- `wavelengths::AbstractVector`: A `w` vector of spectral wavelengths.
- `rsr::Union{Dict,Any,AbstractArray}`: Relative Spectral Response (RSR) information. Can be a dictionary `{w: wavelengths, rsr: rsr_values}` for discrete weights, a scalar common FWHM (Full Width at Half Maximum), or a `w`-length vector of FWHMs.
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
    InstrumentGeoData

A structure to hold geospatial metadata for an instrument's grid.

# Fields
- `origin::AbstractVector`: A `[rx, ry]` vector representing the origin (e.g., top-left corner) of the raster grid.
- `cell_size::AbstractVector`: A `[rx, ry]` vector representing the spatial resolution (cell size) of the grid.
- `ndims::AbstractVector`: A `[nx, ny]` vector representing the number of cells in each spatial dimension of the grid.
- `fidelity::Int64`: An integer indicating the spatial resolution fidelity:
    - `0`: highest spatial resolution.
    - `1`: high spatial resolution.
    - `2`: coarse spatial resolution.
- `dates::AbstractVector`: A vector of dates associated with the data.
- `wavelengths::AbstractVector`: A `w` vector of spectral wavelengths.
- `rsr::Union{Dict,Any,AbstractArray}`: Relative Spectral Response (RSR) information, similar to `InstrumentData`.
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

# --- Kalman Filter and Smoothing Functions ---

"""
    woodbury_filter_kr(Ms, ys, x_pred, P_pred)

Performs the update step (filtering) of a Kalman filter using the Woodbury Matrix Identity.
This function is optimized for models leveraging Kronecker product structures, typical
in spatio-spectral applications where the state is a vector of (spatial x spectral) components.

# Arguments
- `Ms::AbstractVector{<:HSModel}`: A vector of `HSModel` instances, one for each instrument
  or observation source, defining their respective observation operators and noise.
- `ys::AbstractVector{<:AbstractArray{Float64}}`: A vector of observation arrays (`y`), one for each instrument,
  containing the actual measurements.
- `x_pred::AbstractVector{Float64}`: The predicted state mean (prior mean) at the current time step.
- `P_pred::AbstractArray{Float64}`: The predicted state covariance (prior covariance) at the current time step.

# Returns
- `x_new::AbstractVector{Float64}`: The updated (filtered) state mean.
- `P_new::AbstractArray{Float64}`: The updated (filtered) state covariance.

# Algorithm
The function implements the Kalman filter update equations:
1.  **Innovation calculation:** `res_pred = ys[i] - Hs * x_predr * Hw'`
2.  **Kalman Gain calculation (implicitly through Woodbury Identity):**
    The Woodbury identity is used to efficiently compute `(P_pred_inv + H'R_inv H)_inv`.
    Here, `HViH` accumulates `H' * V_inv * H` terms for all instruments, where `V` is observation noise.
    `HVie` accumulates `H' * V_inv * innovation` terms.
    The inversion `inv(P_pred)` is effectively handled by Cholesky decomposition and `potri!`.
    Then, `S = P_pred_inv + HViH` is formed, and its inverse is computed.
    `HtRHIi = S_inv * HViH`.
    The Kalman gain `K = P_pred * H' * (H * P_pred * H' + R)_inv` is implicitly incorporated into the `F` and `P_new` calculations using the Woodbury form.
3.  **State Update:** `x_new = x_pred + K * innovation`
    This is calculated as `x_pred .+ F*HVie` where `F` implicitly contains the Kalman gain.
4.  **Covariance Update:** `P_new = (I - K*H) * P_pred`
    This is calculated as `P_new = (I - F*HViH) * P_pred`.
"""
function woodbury_filter_kr(Ms::AbstractVector{<:HSModel}, 
                            ys::AbstractVector{<:AbstractArray{Float64}}, 
                            x_pred::AbstractVector{Float64}, 
                            P_pred::AbstractArray{Float64}) 

    K = length(Ms) # Number of instruments/observation sources
    N = size(x_pred,1) # Total dimension of the state vector
    ns = size(Ms[1].Hs,2) # Number of spatial latent states
    nw = size(Ms[1].Hw,2) # Number of wavelength latent states (p in the calling function)

    # Pre-allocate matrices for accumulating terms
    HViH = zeros(N,N) # Accumulates H' * V_inv * H terms
    HVie = zeros(N) # Accumulates H' * V_inv * innovation terms
    x_new = ones(N) # Filtered state mean
    P_new = zeros(N,N) # Filtered state covariance
    F = zeros(N,N) # Intermediate matrix for Woodbury identity

    # Reshape the predicted state mean for easier matrix multiplication with H_spatial and H_wavelength
    x_predr = reshape(x_pred, (ns,nw))

    # Iterate through each instrument to accumulate terms for the Kalman update
    for i in 1:K
        Hw = @views Ms[i].Hw # Wavelength observation matrix
        # Compute Hw' * Vw_inv
        HwtV = @views Hw'*inv(Ms[i].Vw) # Vw is typically diagonal or uniform scaling for simplicity
        Hs = @views Ms[i].Hs # Spatial observation matrix
        # Compute Hs' * Vs_inv
        HstV = @views Hs'*inv(Ms[i].Vs) # Vs is typically diagonal or uniform scaling

        # Innovation: difference between observed measurements and predicted measurements
        # (Hs * x_predr * Hw') is the predicted measurement based on the current state.
        res_pred = @views ys[i] - Hs * x_predr * Hw' 

        # Accumulate the H' * V_inv * H term using Kronecker product structure
        # This term is (H_w' V_w_inv H_w) kron (H_s' V_s_inv H_s)
        # where V_w and V_s are spectral and spatial observation noise covariances.
        # Here, `inv(Ms[i].Vw)` and `inv(Ms[i].Vs)` are used, assuming they are pre-inverted or easily invertible.
        HViH .+= kronecker(HwtV*Hw, HstV*Hs)

        # Accumulate the H' * V_inv * innovation term.
        # The structure is (H_s' V_s_inv res_pred V_w_inv H_w)'
        HVie .+= vec(HstV * res_pred * HwtV') # vec converts matrix to vector for accumulation
    end

    ## Woodbury Matrix Identity application for efficient covariance update
    # P_new = (P_pred_inv + H'R_inv H)_inv = P_pred - P_pred * H' * (H * P_pred * H' + R)_inv * H * P_pred
    # The term (P_pred_inv + HViH) needs to be inverted.
    # The initial S is P_pred_inv implicitly, then HViH is added.
    S = P_pred[:,:] # Make a mutable copy of P_pred
    
    # Efficiently compute inverse of P_pred using Cholesky decomposition
    # LAPACK.potrf! performs Cholesky factorization (A = U'U or L L')
    # LAPACK.potri! computes the inverse from the Cholesky factor
    # LinearAlgebra.copytri! ensures the full symmetric matrix is formed from the upper/lower triangle
    begin
        LAPACK.potrf!('U', S) # Cholesky factorization of S (P_pred)
        LAPACK.potri!('U', S) # Invert S from its Cholesky factor
        LinearAlgebra.copytri!(S,'U') # Copy upper triangle to lower to complete the symmetric inverse
    end
    S .+= HViH; # S now holds (P_pred_inv + HViH)

    # Invert S again (which is (P_pred_inv + HViH))
    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    
    # Calculate HtRHIi = (P_pred_inv + HViH)_inv * HViH
    # This term is equivalent to K * H, where K is the Kalman gain.
    HtRHIi = BLAS.symm('R', 'U', S, HViH)

    # Compute F = P_pred * (I - HtRHIi) implicitly for Kalman Gain part
    # F here is equivalent to P_pred * (P_pred_inv + HViH)_inv * P_pred_inv
    # It essentially forms P_pred * (I - K*H) where K is the Kalman Gain
    F .= P_pred; # Initialize F with P_pred
    mul!(F, P_pred, HtRHIi, -1.0, 1.0); # F = P_pred - P_pred * HtRHIi (i.e., F = P_pred * (I - HtRHIi))

    # Update state mean: x_new = x_pred + K * innovation
    # Where K * innovation is implicitly F * HVie
    x_new .= x_pred
    mul!(x_new, F, HVie, 1.0, 1.0) # x_new = x_pred + F * HVie
    
    # Update state covariance: P_new = (I - K*H) * P_pred
    # Where K*H is implicitly F*HViH
    F .= F * HViH; # F now holds F_prev * HViH = P_pred * (I - HtRHIi) * HViH
    P_new .= P_pred # Initialize P_new with P_pred
    mul!(P_new, F, P_pred, -1.0, 1.0) # P_new = P_pred - F * P_pred = (I - F_prev*HViH) * P_pred

    return x_new, P_new
end

## Struct to hold memory allocations for woodbury_filter_kr!()
struct HSModel_helpers{Float64}
    HViH::AbstractMatrix{Float64}
    HVie::AbstractVector{Float64}
    F::AbstractMatrix{Float64}
end

"""
    woodbury_filter_kr!(x_new, P_new, Ms, ys, x_pred, P_pred)

In-place implementation of woodbury_filter_kr(), modifies x_new, P_new in-place.
"""
function woodbury_filter_kr!(x_new::AbstractVector{Float64}, 
        P_new::AbstractArray{Float64}, 
        Ms::AbstractVector{<:HyperSTARS.HSModel}, 
        M_helpers::HSModel_helpers,
        ys::AbstractVector{<:AbstractArray{Float64}}, 
        x_pred::AbstractVector{Float64}, 
        P_pred::AbstractArray{Float64}) 

    K = length(Ms)
    N = size(x_pred,1)
    ns = size(Ms[1].Hs,2)
    nw = size(Ms[1].Hw,2)

    HViH = @views M_helpers.HViH #zeros(N,N)
    HVie = @views M_helpers.HVie #zeros(N)
    # x_new = ones(N)
    # P_new = zeros(N,N)
    F = @views M_helpers.F #zeros(N,N)

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

    # F .= F * HViH;
    P_new .= P_pred
    mul!(P_new, F * HViH, P_pred, -1.0, 1.0)
end


"""
    smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)

Performs Kalman smoothing (Rauch-Tung-Striebel type) to improve state estimates
by incorporating future observations. It runs backwards through the time series,
using both predicted and filtered states.

# Arguments
- `F::Union{AbstractMatrix{Float64}, UniformScaling{Float64}}`: The state transition matrix.
- `predicted_means::AbstractArray{Float64}`: Array of predicted state means from the filtering step.
- `predicted_covs::AbstractArray{Float64}`: Array of predicted state covariances from the filtering step.
- `filtering_means::AbstractArray{Float64}`: Array of filtered state means from the filtering step.
- `filtering_covs::AbstractArray{Float64}`: Array of filtered state covariances from the filtering step.

# Returns
- `smoothed_means::AbstractVector{<:AbstractVector{Float64}}`: A vector of smoothed state means.
- `smoothed_covs::AbstractVector{<:AbstractMatrix{Float64}}`: A vector of smoothed state covariances.

# Algorithm (Rauch-Tung-Striebel)
For each time step `i` from `nsteps-1` down to `t0`:
1.  **Smoother Gain (`C`):**
    `C = filtering_covs[i] * F' * inv(predicted_covs[i])`
    This gain tells us how much to adjust the filtered estimate based on the smoothed estimate from the next time step.
2.  **Smoothed Mean (`x_smooth`):**
    `x_smooth = filtering_means[i] + C * (smoothed_means[i+1] - predicted_means[i+1])`
    The smoothed mean is the filtered mean plus a correction term based on the difference between the next smoothed mean and the next predicted mean.
3.  **Smoothed Covariance (`P_smooth`):**
    `P_smooth = filtering_covs[i] + C * (smoothed_covs[i+1] - predicted_covs[i+1]) * C'`
    The smoothed covariance is the filtered covariance plus a correction term based on the covariance difference and the smoother gain.
"""


function smooth_series(F::Union{AbstractSparseMatrix{Float64},AbstractMatrix{Float64},UniformScaling{Float64}}, 
    predicted_means::AbstractArray{Float64}, 
    predicted_covs::AbstractArray{Float64}, 
    filtering_means::AbstractArray{Float64}, 
    filtering_covs::AbstractArray{Float64}) 

    # These arrays start at the final smoothed (= filtered) state
    nsteps = size(predicted_means,2) 
    smoothed_means = similar(predicted_means)
    smoothed_covs = similar(predicted_covs)

    @views smoothed_means[:,nsteps] = filtering_means[:,nsteps+1]
    @views smoothed_covs[:,:,nsteps] = filtering_covs[:,:,nsteps+1]

    # First step that we are interested here in is i = nsteps - 1
    for i ∈ nsteps:-1:2
        # NB. filtering_covs[i] is P_{i-1|i-1}, predicted_covs[i] is P_{i|i-1}
        begin # C = filtering_covs[i] * F * inv(predicted_covs[i])
            CC = predicted_covs[:,:,i]
            LAPACK.potrf!('U', CC)
            LAPACK.potri!('U', CC)
            C = zeros(size(CC))
            BLAS.symm!('R', 'U', 1., CC, filtering_covs[:,:,i]*F', 0., C)
        end
        x_smooth = filtering_means[:,i] .+ C * (smoothed_means[:,i] .- predicted_means[:,i])

        # P_smooth = filtering_covs[i] + C * (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        # Compute P_smooth = filtering_covs[i] + C *
        # (smoothed_covs[nsteps - i + 1] - predicted_covs[i]) * C'
        begin
            CC .= smoothed_covs[:,:,i] .- predicted_covs[:,:,i]
            D = BLAS.symm('R', 'U', CC, C) # D = C * CC
            CC .= filtering_covs[:,:,i]
            P_smooth = BLAS.gemm!('N', 'T', 1., D, C, 1., CC)
        end

        smoothed_means[:,i-1] = x_smooth
        smoothed_covs[:,:,i-1] = P_smooth
    end

    return smoothed_means, smoothed_covs
end

# function smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs) 

#     # These arrays start at the final smoothed (= filtered) state
#     # The last filtered state is the same as the last smoothed state
#     smoothed_means = [filtering_means[end]]
#     smoothed_covs = [filtering_covs[end]]

#     nsteps = length(predicted_means) # Total number of time steps (T in some notations)
#     t0 = 1 # Start index for processing (typically 1 for the first prediction)

#     # Iterate backwards from the second-to-last time step down to the first
#     for i ∈ nsteps:-1:(t0+1)
#         # NB. filtering_covs[i] is P_{i-1|i-1}, predicted_covs[i] is P_{i|i-1}
#         # Compute the smoother gain C = P_{i-1|i-1} * F' * P_{i|i-1}_inv
#         begin # C = filtering_covs[i] * F * inv(predicted_covs[i])
#             CC = predicted_covs[i][:,:]
#             LAPACK.potrf!('U', CC)
#             LAPACK.potri!('U', CC)
#             C = zeros(size(predicted_covs[i]))
#             BLAS.symm!('R', 'U', 1., CC, filtering_covs[i]*F', 0., C)
#         end

#         # begin 
#         #     CC = predicted_covs[i][:,:] # Make a mutable copy for in-place inversion
#         #     LAPACK.potrf!('U', CC) # Cholesky factorization of predicted_covs[i]
#         #     LAPACK.potri!('U', CC) # Invert CC from its Cholesky factor (now holds inv(predicted_covs[i]))
#         #     C = zeros(size(predicted_covs[i])) # Pre-allocate C
#         #     # C = P_{i-1|i-1} * F' * inv(P_{i|i-1})
#         #     # BLAS.symm!('R', 'U', 1., CC, filtering_covs[i]*F', 0., C) performs C = alpha * A * B + beta * C
#         #     # Here, A = CC (inv(predicted_covs[i])), B = filtering_covs[i]*F', alpha=1, beta=0.
#         #     # Note: BLAS.symm! expects the first argument to be symmetric. CC is symmetric.
#         #     # But the order is (Left/Right, Upper/Lower, alpha, A, B, beta, C)
#         #     # 'R' means B is multiplied from the right, so C = A * B.
#         #     # It appears the BLAS call here calculates: C = inv(predicted_covs[i]) * (filtering_covs[i] * F')
#         #     # This is equivalent to C = P_{i-1|i-1} * F' * inv(P_{i|i-1}) if matrix multiplication order is adjusted.
#         #     # Assuming `BLAS.symm!('R', 'U', 1., CC, filtering_covs[i]*F', 0., C)` is intended to compute (filtering_covs[i]*F') * CC
#         #     # which would be (filtering_covs[i]*F') * inv(predicted_covs[i]). This is the standard RTS gain.
#         #     mul!(C, filtering_covs[i], F', 1.0, 0.0) # C = filtering_covs[i] * F'
#         #     mul!(C, C, CC) # C = (filtering_covs[i] * F') * inv(predicted_covs[i])
#         # end
        
#         # Smoothed mean: x_{i-1|T} = x_{i-1|i-1} + C * (x_{i|T} - x_{i|i-1})
#         # Note: `smoothed_means[nsteps - i + 1]` corresponds to `x_{i|T}` (the smoothed mean from the next time step, already computed)
#         # and `predicted_means[i]` corresponds to `x_{i|i-1}` (the predicted mean for the current step).
#         x_smooth = filtering_means[i] .+ C * (smoothed_means[nsteps - i + 1] .- predicted_means[i])

#         # Smoothed covariance: P_{i-1|T} = P_{i-1|i-1} + C * (P_{i|T} - P_{i|i-1}) * C'
#         # Note: `smoothed_covs[nsteps - i + 1]` is `P_{i|T}`.
#         # `predicted_covs[i]` is `P_{i|i-1}`.
#         begin
#             CC .= smoothed_covs[nsteps - i + 1] .- predicted_covs[i]
#             D = BLAS.symm('R', 'U', CC, C) # D = C * CC
#             CC .= filtering_covs[i]
#             P_smooth = BLAS.gemm!('N', 'T', 1., D, C, 1., CC)

#             # CC .= smoothed_covs[nsteps - i + 1] .- predicted_covs[i] # CC now holds (P_{i|T} - P_{i|i-1})
#             # D = BLAS.symm('R', 'U', CC, C) # D = C * CC (if 'R' means B is multiplied from the right, i.e., D = CC * C).
#             #                              # The standard formula is C * (P_{i|T} - P_{i|i-1}) * C'
#             #                              # So this would be `mul!(D, C, CC)` and then `mul!(P_smooth, D, C')`.
#             #                              # Let's use explicit `mul!` for clarity based on formula:
#             # mul!(P_smooth, C, CC, 1.0, 0.0) # P_smooth = C * (P_{i|T} - P_{i|i-1})
#             # mul!(P_smooth, P_smooth, C', 1.0, 0.0) # P_smooth = (C * (P_{i|T} - P_{i|i-1})) * C'
#             # P_smooth .+= filtering_covs[i] # Add P_{i-1|i-1} to complete the formula
#         end

#         push!(smoothed_means, x_smooth)
#         push!(smoothed_covs, P_smooth)
#     end

#     # Reverse the arrays as they were populated in reverse chronological order
#     return reverse(smoothed_means), reverse(smoothed_covs)
# end



"""
    organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)

Organizes and preprocesses instrument data based on spatial overlap and resolution fidelity.
It identifies relevant measurements, subsets data, and adjusts spatial indices for fusion.

# Arguments
- `full_ext::AbstractArray{<:Real}`: The overall geographic extent of interest.
- `inst_geodata::AbstractVector{InstrumentGeoData}`: A vector of `InstrumentGeoData` for all instruments.
- `inst_data::AbstractVector{InstrumentData}`: A vector of `InstrumentData` for all instruments.
- `target_geodata::InstrumentGeoData`: Geospatial data for the target (output) grid.
- `ss_xy::AbstractArray{<:Real}`: Initial subsampled spatial coordinates (x,y).
- `ss_ij::AbstractArray{<:Real}`: Initial subsampled spatial indices (i,j).
- `res_flag::AbstractVector{<:Real}`: A vector indicating the fidelity (resolution) of each instrument (0: highest, 1: high, 2: coarse).

# Returns
- `measurements::Vector{InstrumentData}`: A vector of `InstrumentData` instances, now subsetted and
  containing only the data relevant to the current processing window.
- `ss_ij::AbstractArray{<:Real}`: The updated subsampled spatial indices, potentially expanded
  to include more BAUs if coarse resolution instruments are involved.
"""
function organize_data(full_ext::AbstractArray{<:Real}, 
                        inst_geodata::AbstractVector{InstrumentGeoData}, 
                        inst_data::AbstractVector{InstrumentData}, 
                        target_geodata::InstrumentGeoData, 
                        ss_xy::AbstractArray{<:Real}, 
                        ss_ij::AbstractArray{<:Real}, 
                        res_flag::AbstractVector{<:Real})

    ### Find measurements:
    measurements = Vector{InstrumentData}(undef, length(inst_geodata))

    # Iterate through instruments, sorted by resolution fidelity in reverse (coarse first)
    # This ordering ensures that `ss_ij` is progressively updated with BAUs relevant to coarser instruments.
    for i in sortperm(res_flag, rev=true)
        if res_flag[i] == 2 # Coarse resolution instrument
            # Find all (i,j) indices within the full extent that correspond to the current instrument's grid.
            ins_ij = @views find_all_ij_ext(full_ext[1,:], full_ext[2,:], inst_geodata[i].origin, inst_geodata[i].cell_size, inst_geodata[i].ndims; inclusive=false)
            # Convert these (i,j) indices to (x,y) spatial coordinates.
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            # Subset the instrument's data using the identified indices.
            ys = @views inst_data[i].data[CartesianIndex.(ins_ij[:,1],ins_ij[:,2]),:,:]
            # Store the subsetted data in a new InstrumentData struct
            measurements[i] = @views InstrumentData(ys,
                                inst_data[i].bias, 
                                inst_data[i].uq, 
                                abs.(inst_geodata[i].cell_size), # Absolute cell size for resolution
                                inst_geodata[i].dates,
                                ins_xy,
                                inst_geodata[i].wavelengths,
                                inst_geodata[i].rsr)
        elseif res_flag[i] == 1 # High resolution instrument (but not the highest)
            # Find the nearest (i,j) indices in the instrument's grid for the subsampled target BAUs.
            ins_ij = @views unique(find_nearest_ij_multi(ss_xy, inst_geodata[i].origin, inst_geodata[i].cell_size, inst_geodata[i].ndims),dims=1)
            # Convert to (x,y) coordinates.
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            # Subset the instrument's data.
            ys = @views inst_data[i].data[CartesianIndex.(ins_ij[:,1],ins_ij[:,2]),:,:]

            # Expand `ss_ij` to include all Basic Area Units (BAUs) in the target grid that
            # overlap with the current instrument's observations.
            ss_ij = @views unique(vcat(ss_ij,find_all_bau_ij_multi(ins_xy, inst_geodata[i].cell_size, target_geodata.origin, target_geodata.cell_size, target_geodata.ndims)),dims=1)
            # Update `ss_xy` based on the expanded `ss_ij`.
            ss_xy = get_sij_from_ij(ss_ij, target_geodata.origin, target_geodata.cell_size)
            
            measurements[i] = @views InstrumentData(ys,
                                inst_data[i].bias, 
                                inst_data[i].uq, 
                                abs.(inst_geodata[i].cell_size),
                                inst_geodata[i].dates,
                                ins_xy,
                                inst_geodata[i].wavelengths,
                                inst_geodata[i].rsr)
        else # res_flag[i] == 0: Highest spatial resolution instrument
            # For the highest resolution instrument, find nearest (i,j) in its grid
            # for the already established subsampled target BAUs (`ss_xy`).
            ins_ij = @views unique(find_nearest_ij_multi(ss_xy, inst_geodata[i].origin, inst_geodata[i].cell_size,inst_geodata[i].ndims),dims=1)
            # Convert to (x,y) coordinates.
            ins_xy = @views get_sij_from_ij(ins_ij, inst_geodata[i].origin, inst_geodata[i].cell_size)
            # Subset the instrument's data.
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

"""
    hyperSTARS_fusion_kr_dict(d, target_wavelengths, spectral_mean, B, target_times, smooth, spatial_mod, obs_operator, state_in_cov, cov_wt, ar_phi)

Performs the core hyperspectral data fusion for a single spatial window (partition).
It implements a Kalman filter and optionally a Kalman smoother, incorporating
spatial and spectral correlations and handling multiple instrument measurements.

# Arguments
- `d::Dict`: A dictionary containing pre-organized data for the current window,
  typically created by `create_data_dicts`. It includes:
    - `:measurements`: Subsetted `InstrumentData` for relevant instruments.
    - `:target_coords`: Spatial coordinates of the BAUs in the current window.
    - `:kp_ij`: Cartesian indices of the target partition BAUs.
    - `:prior_mean`: Prior mean of the state vector.
    - `:prior_var`: Prior variance of the state vector.
    - `:model_pars`: Parameters for the spatial covariance model.
- `target_wavelengths::AbstractVector{<:Real}`: Wavelengths of the desired fused product.
- `spectral_mean::AbstractVector{<:Real}`: Mean spectral signature for basis transformation.
- `B::AbstractArray{<:Real}`: Spectral basis matrix (e.g., PCA loadings) used for transforming
  reflectance to latent state and vice versa.
- `target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1]`: The time steps for which to produce fused output.
- `smooth::Bool = false`: If `true`, perform Kalman smoothing; otherwise, only filtering.
- `spatial_mod::Function = mat32_corD`: Function to compute spatial covariance (e.g., Matern 3/2).
- `obs_operator::Function = unif_weighted_obs_operator`: Function to compute the spatial
  observation operator matrix `Hs`.
- `state_in_cov::Bool = true`: If `true`, allow the process noise covariance `Qf` to be
  adaptive and incorporate state information.
- `cov_wt::Real = 0.3`: Weighting factor for mixing the fixed and state-dependent process noise covariance.
- `ar_phi::Real = 1.0`: Autoregressive parameter for the state transition (temporal correlation).

# Returns
- `kp_ij::AbstractArray{<:Real}`: Cartesian indices of the target partition BAUs within the overall target grid.
- `fused_image::AbstractArray{Float64}`: The fused mean image for the target times,
  reshaped to `nbau x p x size(target_times,1)`, where `p` is the number of latent spectral components.
- `fused_sd_image::AbstractArray{Float64}`: The fused standard deviation image (uncertainty)
  for the target times, in the same format as `fused_image`.
"""
# function hyperSTARS_fusion_kr_dict(d,  
#                 target_wavelengths::AbstractVector{<:Real},
#                 spectral_mean::AbstractVector{<:Real},   
#                 B::AbstractArray{<:Real}, # Spectral basis matrix (e.g., PCA loadings)
#                 target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1],
#                 smooth::Bool = false, # Flag to enable/disable smoothing
#                 spatial_mod::Function = mat32_corD, # Function for spatial covariance (e.g., Matern)                                         
#                 obs_operator::Function = unif_weighted_obs_operator, # Function to create spatial observation operator
#                 state_in_cov::Bool = true, # Flag to adaptively update process noise covariance based on state
#                 cov_wt::Real = 0.3, # Weight for combining fixed and state-dependent Q
#                 ar_phi = 1.0) # Autoregressive parameter for temporal transition

#     # Unpack data from the input dictionary `d`
#     measurements = @views d[:measurements] # Subsetted instrument data
#     target_coords = @views d[:target_coords] # Spatial coordinates of BAUs in the current window
#     kp_ij = @views d[:kp_ij] # Cartesian indices of the target partition BAUs
#     prior_mean = @views d[:prior_mean] # Initial prior mean of the state
#     prior_var = @views d[:prior_var] # Initial prior variance of the state
#     model_pars = @views d[:model_pars] # Parameters for spatial covariance models

#     nbau = size(kp_ij,1) # Number of Basic Area Units (BAUs) in the target partition
#     ni = size(measurements)[1] # Number of instruments/measurement types
#     nf = size(target_coords)[1] # Number of BAUs in the *full* window (target + buffer)

#     p = size(B)[2] # Number of latent spectral components (e.g., PCA components)
    
#     # Pre-allocate arrays to store instrument-specific information
#     nnobs = Vector{Int64}(undef, ni) # Number of spatial observations for each instrument
#     nwobs = Vector{Int64}(undef, ni) # Number of wavelength observations for each instrument
#     t0v = Vector{Int64}(undef, ni) # Start date of observations for each instrument
#     ttv = Vector{Int64}(undef, ni) # End date of observations for each instrument

#     # Populate instrument-specific observation dimensions and time ranges
#     for i in 1:ni
#         nnobs[i] = size(measurements[i].data)[1]
#         nwobs[i] = size(measurements[i].data)[2]
#         t0v[i] = measurements[i].dates[1]
#         ttv[i] = measurements[i].dates[end]
#     end

#     # Determine the overall temporal range for the fusion
#     t0 = minimum(t0v) # Earliest observation date
#     tt = maximum(ttv) # Latest observation date
#     tp = maximum(target_times) # Latest target time for output
#     tpl = minimum(target_times) # Earliest target time for output

#     # Define the full sequence of time steps for filtering/smoothing
#     if smooth
#         times = minimum([t0,tpl]):maximum([tt,tp])
#     else
#         times = minimum([t0,tpl]):tp
#     end
#     nsteps = size(times)[1] # Total number of time steps in the fusion process
    
#     data_kp = falses(ni,nsteps) # Boolean matrix: data_kp[i,t] is true if instrument i has data at time t
   
#     ## Build observation operators (Hw, Hs) and spectral mean adjustments (Hm)
#     Hws = Vector(undef,ni) # Hw: Spectral observation operator for each instrument
#     Hms = Vector(undef,ni) # Hm: Spectral mean adjustment for each instrument
#     Hss = Vector(undef,ni) # Hs: Spatial observation operator for each instrument

#     for (i,x) in enumerate(measurements)
#         # Hss[i]: Spatial observation operator, mapping latent spatial states to observed locations.
#         # This function (e.g., `unif_weighted_obs_operator`) accounts for differences
#         # between instrument observation footprints and target grid cell locations.
#         Hss[i] = obs_operator(x.coords, target_coords, x.spatial_resolution) 
        
#         # Hw: Spectral response function convolution matrix.
#         # It maps the target latent spectral components (B) to the instrument's observed wavelengths.
#         # `rsr_conv_matrix` applies the instrument's RSR.
#         Hw = rsr_conv_matrix(x.rsr, x.wavelengths, target_wavelengths)
#         Hws[i] = Hw*B # Hw * B transforms the latent spectral states to instrument's spectral space.
#         Hms[i] = Hw*spectral_mean # Hw * spectral_mean for bias correction in observation space.
        
#         # Mark time steps where each instrument has data
#         data_kp[i,in(measurements[i].dates).(times)] .= true
#     end

#     # --- Process Noise Covariance (Q) ---
#     # Q describes the uncertainty in the state evolution between time steps.
#     # Here, Q is built from spatial covariance models for each latent spectral component.
#     Qs = Vector{Matrix{Float64}}(undef,size(model_pars)[1]) # Qs for each latent component
#     # `pairwise` calculates Euclidean distances between target coordinates, which is input to spatial covariance functions.
#     dd = pairwise(Euclidean(1e-12), target_coords', dims=2) 
#     for (i,x) in enumerate(eachrow(model_pars))
#         # x[1] is the variance (amplitude) and x[2:end] are parameters (e.g., length scales) for `spatial_mod`.
#         Qs[i] = x[1] .* spatial_mod(dd, x[2:end]) 
#     end
#     # Form a block-diagonal matrix Q, assuming independence between latent spectral components (or transformed components).
#     Q = Matrix(BlockDiagonal(Qs))

#     ## Diagonal transition matrices (F)
#     # F describes how the state propagates from one time step to the next.
#     # UniformScaling(ar_phi) implies a simple autoregressive (AR(1)) model for each state component.
#     F = UniformScaling(ar_phi)

#     # --- Initial State and Covariance ---
#     x0 = prior_mean[:] # Initial state mean (vectorized)
#     P0 = Diagonal(prior_var[:]) # Initial state covariance (assumed diagonal)

#     # --- Arrays to store filtering and prediction results ---
#     filtering_means = Vector{Vector{Float64}}(undef, 0)
#     predicted_means = Vector{Vector{Float64}}(undef, 0)
#     filtering_covs = Vector{Matrix{Float64}}(undef, 0)
#     predicted_covs = Vector{Matrix{Float64}}(undef, 0)
    
#     # Initialize with the prior at t=0
#     push!(filtering_means, x0)
#     push!(filtering_covs, P0)

#     # Pre-allocate output arrays for fused images
#     fused_image = zeros(nbau,p,size(target_times,1))
#     fused_sd_image = zeros(nbau,p,size(target_times,1))

#     # Find indices of `times` that correspond to `target_times` (for outputting results)
#     kp_times = findall(times .∈ Ref(target_times))

#     ## Main Kalman Filtering Loop
#     # Iterates through each time step defined by `times`
#     for (t,t2) in enumerate(times)
#         # --- Process Noise Covariance Update (Qf) ---
#         # Optionally make the process noise covariance adaptive based on previous state estimates.
#         if state_in_cov ## update this to weighted past
#             # Xtt gathers past filtered means, reshaped to (spatial_dim, spectral_dim, time_dim)
#             Xtt = cat([reshape(x, (nf,p)) for x in filtering_means[1:t]]..., dims=3)
#             # Wt are weights based on inverse square root of diagonal of filtering covariances (uncertainty).
#             # Lower uncertainty gives higher weight.
#             Wt = cat([reshape(1.0 ./ sqrt.(diag(x)), (nf,p)) for x in filtering_covs[1:t]]...,dims=3)
#             Wtn = sum(Wt, dims=3) # Sum of weights for normalization
#             Wt ./= Wtn # Normalize weights
#             Xtt .*= Wt # Apply weights to state estimates

#             # Compute state-dependent spatial covariance (Qss)
#             # `state_cov` likely computes a covariance matrix for each latent component
#             # based on the (weighted) past states of that component.
#             Qst = Vector{Matrix{Float64}}(undef,p)
#             for (i,x) in enumerate(eachrow(model_pars))
#                 Qst[i] = state_cov(Xtt[:,i,:]',x)
#             end
#             Qss = Matrix(BlockDiagonal(Qst)) # Form block-diagonal state-dependent Qss

#             # Combine the fixed Q and state-dependent Qss using `cov_wt`
#             Qf = cov_wt .* Q .+ (1-cov_wt) .* Qss
#         else
#             Qf = Q # If not adaptive, use the fixed Q
#         end

#         # --- Prepare Observations for Current Time Step ---
#         Ms = HSModel[] # Vector to hold HSModel instances for instruments with data at this time
#         ys = Vector{Array{Float64}}() # Vector to hold corresponding observation arrays

#         # Identify instruments that have data at the current time step `t2`
#         for x in findall(data_kp[:,t])
#             yss = @views measurements[x].data[:,:,measurements[x].dates .== t2] # Extract measurements for current time
#             ym = .!vec(any(isnan, yss; dims=2)) # Find rows (spatial samples) without NaNs
#             Hs2 = Hss[x][ym,:] # Subset spatial observation operator for valid samples

#             # Construct HSModel for the current instrument at this time step
#             push!(Ms, HSModel(Hws[x], Hs2, Diagonal(measurements[x].uq[:]), 1.0*I(size(Hs2,1)), Qf, F))
#             # Push the observed data, adjusted by the spectral mean bias, for valid samples
#             push!(ys,yss[ym,:] .- Hms[x]');
#         end

#         # --- Kalman Prediction Step ---
#         # Predict the next state mean: x_pred = F * x_{t-1|t-1}
#         x_pred = F * filtering_means[t] 
#         # Predict the next state covariance: P_pred = F * P_{t-1|t-1} * F' + Qf
#         P_pred = F * filtering_covs[t] * F' + Qf
#         push!(predicted_means, x_pred)
#         push!(predicted_covs, P_pred)

#         # --- Kalman Filtering (Update) Step ---
#         if length(ys) == 0 # No measurements available at this time step
#             push!(filtering_means, x_pred) # Filtered state is just the predicted state
#             push!(filtering_covs, P_pred) # Filtered covariance is just the predicted covariance
#         else # Measurements are available
#             # Call the `woodbury_filter_kr` function to perform the update
#             x_new, P_new = woodbury_filter_kr(Ms, ys, x_pred, P_pred)
#             push!(filtering_means, x_new)
#             push!(filtering_covs, P_new)
#         end
#     end

#     # --- Kalman Smoothing Step (if `smooth` is true) ---
#     if smooth
#         # Determine the starting index for smoothing.
#         # `st` is the minimum index in `times` that corresponds to a `target_time`.
#         # This ensures smoothing only occurs over the relevant output time range.
#         st = minimum(kp_times) 
#         # Call `smooth_series` using the relevant subsets of predicted and filtered results
#         smoothed_means, smoothed_covs = smooth_series(F, predicted_means[st:end], predicted_covs[st:end], filtering_means[st:end], filtering_covs[st:end])
        
#         # Populate the `fused_image` and `fused_sd_image` with smoothed results
#         for (ti,t2) in enumerate(kp_times .- st .+ 1) # Adjust index for `smoothed_means/covs` array
#             # Reshape the smoothed mean from vector to (spatial_dim, spectral_dim) and subset to `nbau`
#             fused_image[:,:,ti] = @views reshape(smoothed_means[t2],(nf,p))[1:nbau,:] 
#             # Reshape the square root of diagonal elements of smoothed covariance (standard deviation)
#             fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(smoothed_covs[t2])),(nf,p))[1:nbau,:]
#         end
#     else # No smoothing, use filtering results directly
#         # Populate `fused_image` and `fused_sd_image` with filtering results
#         for (ti,t2) in enumerate(kp_times)
#             # `t2+1` because `filtering_means` is indexed from t=0 (prior) to t=nsteps (final filtered)
#             fused_image[:,:,ti] = @views reshape(filtering_means[t2+1],(nf,p))[1:nbau,:] 
#             fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(filtering_covs[t2+1])),(nf,p))[1:nbau,:]
#         end
#     end    
#     return kp_ij, fused_image, fused_sd_image           
# end


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
                tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
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

    n = nf*p 

    Q = zeros(n,n)

    # Qs = Vector{Matrix{Float64}}(undef,size(model_pars)[1])
    dd = pairwise(Euclidean(1e-12), target_coords, dims=1)
    for (i,x) in enumerate(eachrow(model_pars))
        ids = ((i-1)*nf+1):i*nf
        @views Q[ids,ids] = x[1] .* spatial_mod(dd, x[2:end]) 
    end

    ## Diagonal transition matrices
    F = UniformScaling(ar_phi)

    # x0 = prior_mean[:] # don't need this but here to help with synergizing code later
    # P0 = Diagonal(prior_var[:]) # just assuming diagonal C0

    filtering_means = zeros(n,nsteps+1)
    filtering_covs = zeros(n,n,nsteps+1)
    filtering_prec = zeros(n,nsteps+1)

    predicted_means = zeros(n,nsteps)
    predicted_covs = zeros(n,n,nsteps)

    @views filtering_means[:,1] = prior_mean
    @views filtering_covs[:,:,1] = Diagonal(prior_var)
    @views filtering_prec[:,1] = 1.0 ./ sqrt.(prior_var)

    # filtering_means = Vector{Vector{Float64}}(undef, 0)
    # predicted_means = Vector{Vector{Float64}}(undef, 0)
    # filtering_covs = Vector{Matrix{Float64}}(undef, 0)
    # predicted_covs = Vector{Matrix{Float64}}(undef, 0)
    # push!(filtering_means, prior_mean[:])
    # push!(filtering_covs, Diagonal(prior_var[:]))

    fused_image = zeros(nbau,p,size(target_times,1))
    fused_sd_image = zeros(nbau,p,size(target_times,1))

    kp_times = findall(times .∈ Ref(target_times))

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    x_new = zeros(n)
    P_new = zeros(n,n)

    FPpred = similar(P_pred)

    Qf = zeros(n,n)

    HS_helpers = HSModel_helpers(zeros(n,n), zeros(n), zeros(n,n))

    ## bunch of reusable memory in here...
    for (t,t2) in enumerate(times)
        Qf .= Q

        if state_in_cov 
            Qf .*= cov_wt

            # Xtt = cat([reshape(x, (nf,p)) for x in filtering_means[:,1:t]]..., dims=3)
            # Wt = cat([reshape(1.0 ./ sqrt.(diag(x)), (nf,p)) for x in filtering_covs[1:t]]...,dims=3)

            Xtt = reshape(filtering_means[:,1:t], (nf,p,t))[:,:,:]
            Wt = @views reshape(filtering_prec[:,1:t], (nf,p,t))

            Xtt .*= Wt
            Xtt ./= sum(Wt, dims=3)

            # Qst = Vector{Matrix{Float64}}(undef,p)
            for (i,x) in enumerate(tscov_pars)
                ids = ((i-1)*nf+1):i*nf
                x2 = [model_pars[i,1], x]
                @views Qf[ids,ids] .+= state_cov(Xtt[:,i,:]',x2) .* (1.0 .- cov_wt)
                # Qst[i] = state_cov(Xtt[:,i,:]',x)
            end
        
            # Qss = Matrix(BlockDiagonal(Qst)*(1-cov_wt))

            # Qf = cov_wt .* Q .+ (1-cov_wt) .* Qss
            # Qf *= cov_wt
            # Qf .+= Qss

        end

        Ms = HyperSTARS.HSModel[]

        ys = Vector{Array{Float64}}()
        for x in findall(data_kp[:,t])
            yss = @views measurements[x].data[:,:,measurements[x].dates .== t2]
            ym = .!vec(any(isnan, yss; dims=2))
            Hs2 = Hss[x][ym,:]

            push!(Ms, HyperSTARS.HSModel(Hws[x], Hs2, Diagonal(measurements[x].uq[:]), 1.0*I(size(Hs2,1)), Qf, F))
            push!(ys,yss[ym,:] .- Hms[x]');
        end

        # Predictive mean and covariance here
        # x_pred = F * filtering_means[:,t] # filtering_means[1], covs[1] is prior mean
        # P_pred = F * filtering_covs[:,:,t] * F' + Qf
        P_pred .= Qf
        mul!(x_pred, F, @view(filtering_means[:,t]))
        mul!(FPpred, F, @view(filtering_covs[:,:,t]))
        mul!(P_pred, FPpred, F', 1.0, 1.0)

        predicted_means[:,t] = x_pred
        predicted_covs[:,:,t] = P_pred

        # Filtering is done here
        if length(ys) == 0
            filtering_means[:,t+1] = x_pred
            filtering_covs[:,:,t+1] = P_pred
            filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P_pred))
        else
            woodbury_filter_kr!(x_new, P_new, Ms, HS_helpers, ys, x_pred, P_pred)
            filtering_means[:,t+1] = x_new
            filtering_covs[:,:,t+1] = P_new
            filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P_new))
        end
    end

    if smooth
        st = minimum(kp_times)
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means[:,st:end], predicted_covs[:,:,st:end], filtering_means[:,st:end], filtering_covs[:,:,st:end])
        for (ti,t2) in enumerate(kp_times .- st .+ 1)
            fused_image[:,:,ti] = @views reshape(smoothed_means[:,t2],(nf,p))[1:nbau,:] 
            fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(smoothed_covs[:,:,t2])),(nf,p))[1:nbau,:]
        end
    else
        for (ti,t2) in enumerate(kp_times)
            fused_image[:,:,ti] = @views reshape(filtering_means[:,t2+1],(nf,p))[1:nbau,:] 
            fused_sd_image[:,:,ti] = @views reshape(sqrt.(diag(filtering_covs[:,:,t2+1])),(nf,p))[1:nbau,:]
        end
    end  
    return kp_ij, fused_image, fused_sd_image   
end

"""
    create_data_dicts(ii)

Helper function within `scene_fusion_pmap` (originally separated, but now inlined for context)
to create a dictionary of all necessary data for the `hyperSTARS_fusion_kr_dict` function
for a single spatial window (`k,l`).

# Arguments
- `ii::Int`: Index corresponding to the current spatial window (partition).

# Returns
- `d::Dict`: A dictionary populated with data relevant to the specific window,
  ready to be passed to `hyperSTARS_fusion_kr_dict`.

# Global Variables (Implicitly used or passed via closure in `scene_fusion_pmap`'s `pmap`)
- `inds`: Array of (k,l) indices for each window.
- `window_origin`, `window_csize`, `window_buffer`, `nb_coarse`: Window-specific parameters.
- `target_origin`, `target_csize`, `target_ndims`: Target grid parameters.
- `inst_geodata`, `inst_data`: All instrument geospatial and measurement data.
- `prior_mean`, `prior_var`, `model_pars`: Prior and model parameters.
- `nsamp`: Number of samples for subsampling BAUs.
"""
function create_data_dicts( ii )
    # This function is designed to be called within a parallel map (pmap)
    # The variables like `inds`, `window_origin`, etc., are either global
    # within the module or captured from the scope of the calling function (`scene_fusion_pmap`).
    # For clarity and strict encapsulation, these would ideally be passed as arguments,
    # but for `pmap` it's common to capture them.

    k,l = inds[ii,:] # Get the (k,l) grid coordinates for the current window

    ### Find target partition given origin and (k,l)th partition coordinate
    # Calculate the bounding box for the current window based on its origin and cell size.
    window_bbox = bbox_from_centroid(window_origin .+ [k-1, l-1].*window_csize, window_csize)
    
    ### Add buffer of `window_buffer` target pixels around target partition extent
    # Expand the window bounding box by a buffer to include surrounding BAUs.
    buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'

    ### Find extent of overlapping instruments for each instrument
    # Determine the overlapping geographic extent for each instrument within the buffered window.
    all_exts = [Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size)) for x in inst_geodata]
    res_flag = [x.fidelity for x in inst_geodata] # Get resolution fidelity for each instrument

    # For coarse resolution instruments (fidelity == 2), further expand their extent
    # to account for `nb_coarse` additional pixels, ensuring enough context for fusion.
    for i in findall(res_flag .== 2)
        exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[i].cell_size'
        push!(all_exts, exx)
    end

    ### Find full extent combining all instrument extents
    # Merge all individual instrument extents to get a single, overall extent for the current window.
    full_ext = merge_extents(all_exts, sign.(target_csize))

    ### Find all BAUs within target (unbuffered) window
    # Identify all (i,j) Basic Area Units (BAUs) that fall within the original, unbuffered target window.
    target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)

    ### Find all BAUs within target + buffer
    # Identify all (i,j) BAUs that fall within the buffered target window.
    ss_ij = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)

    ### Instrument fidelity based subsampling of BAUs
    # If any coarse resolution instruments are present, add additional subsampled BAUs
    # within the full extent to ensure adequate spatial coverage.
    if any(res_flag .== 2)
        ss_ij = unique(vcat(ss_ij, sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)),dims=1)
    end

    # Convert subsampled (i,j) indices to (x,y) coordinates.
    ss_xy = get_sij_from_ij(ss_ij, target_origin, target_csize)

    ### Find measurements:
    # Call `organize_data` to subset and prepare instrument measurements based on the calculated extents and BAUs.
    measurements, ss_ij = organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)
    
    ### Stack to ensure target partition coords are first in list for easy subsetting later
    # Combine the original target partition BAUs with the (potentially expanded) subsampled BAUs.
    # This ensures that the BAUs relevant to the immediate output window are always at the beginning.
    bau_ij = unique(vcat(target_ij, ss_ij),dims=1)

    ### x,y coords for all baus
    # Get spatial coordinates for all BAUs in `bau_ij`.
    bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)
    
    ### i,j indices for all baus
    # Convert (i,j) indices to `CartesianIndex` for easy array indexing.
    bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])
    
    ### Subset prior mean and var arrays to bau pixels
    # Extract the prior mean and variance corresponding to the `bau_ci` pixels.
    prior_mean_sub = @views prior_mean[bau_ci,:][:]
    prior_var_sub = @views prior_var[bau_ci,:][:]
    
    # Convert `target_ij` to `CartesianIndex` for the final output mapping.
    tind = CartesianIndex.(target_ij[:,1], target_ij[:,2])

    # Populate the dictionary with all prepared data for the current window
    d = Dict()
    d[:measurements] = measurements
    d[:target_coords] = bau_coords
    d[:kp_ij] = tind
    d[:prior_mean] = prior_mean_sub
    d[:prior_var] = prior_var_sub
    d[:model_pars] = @views model_pars[k,l,:,:]

    return d
end

"""
    scene_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata, spectral_mean, prior_mean, prior_var, B, model_pars; kwargs...)

Orchestrates the hyperspectral data fusion process across an entire scene or large area.
It divides the scene into spatial windows and processes each window in parallel using `pmap`.

# Arguments
- `inst_data::AbstractVector{InstrumentData}`: A vector of `InstrumentData` for all available instruments.
- `inst_geodata::AbstractVector{InstrumentGeoData}`: A vector of `InstrumentGeoData` for all instruments.
- `window_geodata::InstrumentGeoData`: Geospatial data defining the structure of the spatial windows.
- `target_geodata::InstrumentGeoData`: Geospatial data defining the final target output grid.
- `spectral_mean::AbstractVector{<:Real}`: Mean spectral signature for basis transformation.
- `prior_mean::AbstractArray{<:Real}`: Global prior mean array for the entire scene.
- `prior_var::AbstractArray{<:Real}`: Global prior variance array for the entire scene.
- `B::AbstractArray{<:Real}`: Spectral basis matrix.
- `model_pars::AbstractArray{<:Real}`: Parameters for the spatial covariance model.

# Keyword Arguments
- `nsamp::Integer = 100`: Number of samples for subsampling Basic Area Units (BAUs).
- `window_buffer::Integer = 2`: Number of buffer pixels to add around each window.
- `target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1]`: The time steps for which to produce fused output.
- `smooth::Bool = false`: If `true`, perform Kalman smoothing; otherwise, only filtering.
- `spatial_mod::Function = mat32_corD`: Function to compute spatial covariance.
- `obs_operator::Function = unif_weighted_obs_operator`: Function to compute the spatial observation operator.
- `state_in_cov::Bool = true`: If `true`, allow the process noise covariance `Qf` to be adaptive.
- `cov_wt::Real = 0.7`: Weighting factor for mixing fixed and state-dependent process noise covariance.
- `ar_phi::Real = 1.0`: Autoregressive parameter for the state transition.
- `nb_coarse::Real = 2.0`: Number of coarse pixels to extend the window bounding box for coarse instruments.

# Returns
- `fused_image::AbstractArray{Float64}`: The complete fused mean image for the entire scene,
  reshaped to `target_ndims[1] x target_ndims[2] x size(B)[2] x nsteps`.
- `fused_sd_image::AbstractArray{Float64}`: The complete fused standard deviation image (uncertainty)
  for the entire scene, in the same format.
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
                      cov_wt::Real = 0.3,
                      tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                      ar_phi::Real = 1.0,
                      nb_coarse::Real = 2.0)

    ### Define target extent and target + buffer extent
    # Extract geospatial parameters for windows and target grid.
    window_csize = @views window_geodata.cell_size
    target_csize = @views target_geodata.cell_size
    window_origin = @views window_geodata.origin
    nwindows = @views window_geodata.ndims # Number of windows in x and y dimensions
    target_origin = @views target_geodata.origin
    target_waves = @views target_geodata.wavelengths
    target_ndims = @views target_geodata.ndims # Dimensions of the overall target grid

    ni = length(inst_geodata) # Total number of instruments
    nsteps = size(target_times)[1] # Number of target time steps for output
    
    # Pre-allocate arrays for the final fused image and its standard deviation across the entire scene.
    # Dimensions: (target_x, target_y, latent_spectral_components, time_steps)
    fused_image = zeros(target_ndims[1], target_ndims[2], size(B)[2], nsteps);
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], size(B)[2], nsteps);
    
    # Generate all (k,l) indices for each window in the scene.
    inds = hcat(repeat(1:nwindows[1], inner=nwindows[2]), repeat(1:nwindows[2], outer=nwindows[1]))
    
    nr = size(inds,1) # Total number of spatial windows

    # Prepare input dictionaries for each spatial window.
    # This loop is essentially the content of the `create_data_dicts` function,
    # inlined here to explicitly show how `T` is constructed before `pmap`.
    T = []
    for ii in 1:nr
        k,l = inds[ii,:]

        window_bbox = bbox_from_centroid(window_origin .+ [k-1, l-1].*window_csize, window_csize)
        buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'
        all_exts = [Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size)) for x in inst_geodata]
        res_flag = [x.fidelity for x in inst_geodata] 
        for i in findall(res_flag .== 2)
            exx = window_bbox .+ [-nb_coarse - 0.01,nb_coarse + 0.01]*inst_geodata[i].cell_size'
            push!(all_exts, exx)
        end
        full_ext = merge_extents(all_exts, sign.(target_csize))
        target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
        ss_ij = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
        if any(res_flag .== 2)
            ss_ij = unique(vcat(ss_ij, sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)),dims=1)
        end
        ss_xy = get_sij_from_ij(ss_ij, target_origin, target_csize)
        measurements, ss_ij = organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)
        bau_ij = unique(vcat(target_ij, ss_ij),dims=1)
        bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)
        bau_ci = CartesianIndex.(bau_ij[:,1],bau_ij[:,2])
        prior_mean_sub = @views prior_mean[bau_ci,:][:]
        prior_var_sub = @views prior_var[bau_ci,:][:]
        tind = CartesianIndex.(target_ij[:,1], target_ij[:,2])

        d = Dict()
        d[:measurements] = measurements
        d[:target_coords] = bau_coords
        d[:kp_ij] = tind
        d[:prior_mean] = prior_mean_sub
        d[:prior_var] = prior_var_sub
        d[:model_pars] = @views model_pars[k,l,:,:]
        push!(T,d)
    end

    # Perform parallel fusion using `pmap`. Each element of `T` (a dictionary for one window)
    # is processed by `hyperSTARS_fusion_kr_dict`. `pmap` distributes these tasks.
    result = @showprogress pmap(x -> hyperSTARS_fusion_kr_dict(x,  
                    target_waves, spectral_mean, B,
                    target_times, smooth, spatial_mod, 
                    obs_operator, state_in_cov, cov_wt, tscov_pars, ar_phi) , T );
    
    # Reconstruct the full fused image and standard deviation image from the results
    # obtained from each individual window.
    for i in 1:nr
        # `result[i][1]` contains `kp_ij` (Cartesian indices of the target BAUs for window `i`).
        # `result[i][2]` contains `fused_image` for window `i`.
        # `result[i][3]` contains `fused_sd_image` for window `i`.
        @views fused_image[result[i][1],:,:] = result[i][2]
        @views fused_sd_image[result[i][1],:,:] = result[i][3]
    end
    
    return fused_image, fused_sd_image

end


end # end of module
