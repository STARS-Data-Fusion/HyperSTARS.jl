"""
Module HyperSTARS

This module provides a framework for hyperspectral spatio-temporal adaptive resolution system (HyperSTARS)
data fusion. It combines measurements from multiple instruments with varying spatial and spectral
resolutions into a single, more refined, high-resolution product. The fusion process is based on
Kalman filtering and smoothing techniques, leveraging Kronecker products and the information Matrix
Identity for computational efficiency, especially in high-dimensional spatio-spectral spaces.

Key functionalities include:
- Organization and preprocessing of diverse instrument measurement and geospatial data.
- Construction of state-space models for underlying physical processes.
- Implementation of Kalman filtering and smoothing for state estimation.
- Parallel processing capabilities for large scene fusion.
"""
module HyperSTARS

export organize_data
export scene_inds_fusion_pmap
export scene_fusion_pmap
export swath_fusion_pmap
export gridded_fusion_pmap
export hyperSTARS_swath_filter
export hyperSTARS_swath_smooth
export hyperSTARS_grid_filter
export hyperSTARS_grid_smooth
export information_filter_kr
export information_filter_kr!

export InstrumentData
export InstrumentGeoData
export SwathData

export cell_size 
export get_centroid_origin_raster 

export nanmean
export nanvar # nanvar was exported but not defined in the original code, so adding a basic one here

export exp_cor
export mat32_cor
export mat52_cor
export exp_corD
export mat32_corD
export mat52_corD
export state_cov

export unif_weighted_obs_operator_centroid 


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

# --- Include Helper Files ---
# These files are assumed to contain helper functions related to resampling,
# spatial utilities (like coordinate conversions, grid operations), and
# Gaussian Process (GP) related functions (like covariance kernels).
include("resampling_utils.jl")
include("spatial_utils_ll.jl") # Contains functions like find_all_ij_ext, get_sij_from_ij, find_nearest_ij_multi, find_all_bau_ij_multi, bbox_from_centroid, find_overlapping_ext, merge_extents, sobol_bau_ij, unif_weighted_obs_operator
include("GP_utils.jl") # Contains functions like exp_corD, mat32_corD, mat52_corD, state_cov, rsr_conv_matrix
include("kalman_core.jl") # kalman filter update/predict functionality

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

"""
    diag_cov_XB(Σ, B, nf, p, r)

Computes the diag((B kron I)*Sigma*(B kron I)') 
"""
function diag_cov_XB(Σ, B, nf, p, r)
    result = zeros(nf, r)
    
    # Precompute all relevant covariance slices
    # For each i, extract the nf-strided slice of Σ we need
    @inbounds for j in 1:r
        Bj = @view B[:, j]
        for i in 1:nf
            acc = zero(eltype(result))
            
            # Diagonal terms (k == l)
            for k in 1:p
                idx = (k-1)*nf + i
                acc += Bj[k]^2 * Σ[idx, idx]
            end
            
            # Off-diagonal terms (k < l, exploit symmetry)
            for k in 1:p-1
                row_idx = (k-1)*nf + i
                @simd for l in k+1:p
                    col_idx = (l-1)*nf + i
                    acc += 2 * Bj[k] * Bj[l] * Σ[row_idx, col_idx]
                end
            end
            
            result[i, j] = acc
        end
    end
    return result
end


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
    bias::Union{Any,AbstractArray} # W x T array of error biases 
    uq::Union{Any,AbstractArray} # W x T array of error variances
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
    information_filter_kr(Ms, ys, x_pred, P_pred)

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
function information_filter_kr(Ms::AbstractVector{<:HSModel}, 
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
    # F = zeros(N,N) # Intermediate matrix for Woodbury identity

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

    ## Information-form update
    # S = P_pred⁻¹ + H' V_inv H ;  P_new = S⁻¹ ;  x_new = x_pred + P_new * HVie
    S = P_pred[:,:] # Make a mutable copy of P_pred
    begin
        LAPACK.potrf!('U', S) # Cholesky factorization of P_pred
        LAPACK.potri!('U', S) # S = P_pred⁻¹ (upper triangle)
        LinearAlgebra.copytri!(S,'U') # complete the symmetric inverse
    end
    S .+= HViH; # S = P_pred⁻¹ + H' V_inv H

    # Invert in place: S now holds (P_pred⁻¹ + H' V_inv H)⁻¹ = P_new
    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    LinearAlgebra.copytri!(S,'U') ## ADDED: potri! fills only the upper triangle; S is now used as a full matrix
   
    # Update state mean: x_new = x_pred + P_new * HVie
    x_new .= x_pred
    mul!(x_new, S, HVie, 1.0, 1.0) ## ADDED: x_new = x_pred + S * HVie
    
    P_new .= S ## ADDED: P_new = (P_pred⁻¹ + H' V_inv H)⁻¹

    # Explicitly symmetrize to avoid numerical precision issues
    # P_new .= (P_new + P_new') / 2
    
    # Add small regularization to maintain numerical stability and positive-definiteness
    # max_diag = maximum(abs, diag(P_new))
    # n_dim = size(P_new, 1)
    # reg_amount = max_diag * 1e-3 + 1e-8
    # for i in 1:n_dim
    #     P_new[i, i] += reg_amount
    # end

    return x_new, P_new
end

## Struct to hold memory allocations for information_filter_kr!()
struct HSModel_helpers{Float64}
    HViH::AbstractMatrix{Float64}
    HVie::AbstractVector{Float64}
    F::AbstractMatrix{Float64}
end

"""
    information_filter_kr!(x_new, P_new, Ms, ys, x_pred, P_pred)

In-place implementation of information_filter_kr(), modifies x_new, P_new in-place.
"""
function information_filter_kr!(x_new::AbstractVector{Float64}, 
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
    
    # Explicitly symmetrize to avoid numerical precision issues
    # P_new .= (P_new + P_new') / 2
    
    # Add small regularization to maintain numerical stability and positive-definiteness
    # max_diag = maximum(abs, diag(P_new))
    # n_dim = size(P_new, 1)
    # reg_amount = max_diag * 1e-3 + 1e-8
    # for i in 1:n_dim
    #     P_new[i, i] += reg_amount
    # end
end

is_scalar_identity(M) = size(M) == (1,1) && M[1,1] == 1.0

"""
    kalman_filter_kr(Ms, x_pred, P_pred)

Implementation of standard filter update (inverse in observation space)
"""
function kalman_filter_kr(Ms::AbstractVector{<:HyperSTARS.HSModel}, 
        y::AbstractVector{<:AbstractArray{Float64}}, 
        x_pred::AbstractVector{Float64}, 
        P_pred::AbstractArray{Float64}) 

    ###############
    # H = vcat([kron(Ms[i].Hw, Ms[i].Hs) for i in eachindex(Ms)]...)
    if is_scalar_identity(Ms[1].Hw) 
        Ht = vcat([Ms[i].Hs for i in eachindex(Ms)]...)
        R = Matrix(BlockDiagonal([Ms[i].Vw[1] .* Ms[i].Vs for i in eachindex(Ms)]))
    else 
        Ht = vcat([kron(Ms[i].Hw, Ms[i].Hs) for i in eachindex(Ms)]...)
        R = Matrix(BlockDiagonal([kron(Ms[i].Vw, Ms[i].Vs) for i in eachindex(Ms)]))
    end
    ys = vcat([x[:] for x in y]...)

    N = size(x_pred,1)
    x_new = ones(N) # Filtered state mean
    P_new = zeros(N,N) # Filtered state covariance

    res_pred = ys - Ht * x_pred # innovation

    HPpT = P_pred * Ht'

    S = Ht * HPpT + R # innovation covariance

    # Kalman gain; K = P_pred * H' * inv(S)
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        K = BLAS.symm('R', 'U', S, HPpT)
    end

    # x_new = x_pred + K * res_pred
    x_new .= x_pred
    mul!(x_new, K, res_pred, 1.0, 1.0)


    # P_new = P_pred - K * (Ht * P_pred) # filtering distribution covariance
    P_new .= P_pred
    mul!(P_new, K, HPpT', -1.0, 1.0)

    ################################
    # Explicitly symmetrize to avoid numerical precision issues
    # P_new .= (P_new + P_new') ./ 2

    return x_new, P_new
end

"""
    kalman_filter_kr!(x_new, P_new, Ms, x_pred, P_pred)

In-place implementation of standard filter update (inverse in observation space), modifies x_new, P_new in-place.
"""
function kalman_filter_kr!(x_new::AbstractVector{Float64}, 
        P_new::AbstractArray{Float64}, 
        Ms::AbstractVector{<:HyperSTARS.HSModel}, 
        y::AbstractVector{<:AbstractArray{Float64}}, 
        x_pred::AbstractVector{Float64}, 
        P_pred::AbstractArray{Float64}) 

    ###############
    if is_scalar_identity(Ms[i].Hw) 
        Ht = vcat([Ms[i].Hs for i in eachindex(Ms)]...)
        R = BlockDiagonal([Ms[i].Vw .* Ms[i].Vs for i in eachindex(Ms)])
    else 
        Ht = vcat([kron(Ms[i].Hw, Ms[i].Hs) for i in eachindex(Ms)]...)
        R = BlockDiagonal([kron(Ms[i].Vw, Ms[i].Vs) for i in eachindex(Ms)])
    end
    ys = vcat([x[:] for x in y]...)

    res_pred = ys - Ht * x_pred # innovation

    HPpT = P_pred * Ht'

    S = Ht * HPpT + R # innovation covariance

    # Kalman gain; K = P_pred * H' * inv(S)
    begin
        LAPACK.potrf!('U', S)
        LAPACK.potri!('U', S)
        K = BLAS.symm('R', 'U', S, HPpT)
    end

    # x_new = x_pred + K * res_pred
    x_new .= x_pred
    mul!(x_new, K, res_pred, 1.0, 1.0)


    # P_new = P_pred - K * (Ht * P_pred) # filtering distribution covariance
    P_new .= P_pred
    mul!(P_new, K, HPpT', -1.0, 1.0)

    ################################
    # Explicitly symmetrize to avoid numerical precision issues
    # P_new .= (P_new + P_new') / 2
end

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
            new_bau_ij = find_all_bau_ij_multi(ins_xy, inst_geodata[i].cell_size, target_geodata.origin, target_geodata.cell_size, target_geodata.ndims)
            if size(new_bau_ij, 2) == 0
                new_bau_ij = Matrix{Int64}(undef, 0, 2)
            end
            ss_ij = @views unique(vcat(ss_ij, new_bau_ij), dims=1)
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

###### Scene Fusion Functions

# --- Deprecated v1 interface ----------------------------------------------------
"""
    scene_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata,
                      spectral_mean, prior_mean, prior_var, B, model_pars; kwargs...)

Deprecated v1 interface. Calls `gridded_fusion_pmap` with the settings that
reproduce v1 gridded behavior:

- `G = ar_phi * I(p)` (v1 used `F = UniformScaling(ar_phi)`)
- `tscov_lag = nothing` (full-history process covariance)
- `nodata_reg = 0.0` (no regularization on steps without data)
- `uncertainty = :smoothed` when `smooth = true` (v1 ran the full RTS smoother)

Keyword arguments and defaults match v1. Returns `(fused_image, fused_sd_image)`.
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
                      obs_operator::Function = unif_weighted_obs_operator_centroid,
                      state_in_cov::Bool = true,
                      cov_wt::Real = 0.3,
                      tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                      ar_phi::Real = 1.0,
                      window_radius::Real = 0.0,
                      use_progress_bar::Bool = true)

    @warn """`scene_fusion_pmap` is deprecated; use `gridded_fusion_pmap`. To reproduce v1 results:
        gridded_fusion_pmap(..., B, ar_phi*I(size(B,2)), model_pars;
                            tscov_lag = nothing, nodata_reg = 0.0,
                            uncertainty = smooth ? :smoothed : :filtered, ...)""" maxlog=1

    p = size(B, 2)
    G = Matrix{Float64}(ar_phi * I, p, p)

    return gridded_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata,
                               spectral_mean, prior_mean, prior_var, B, G, model_pars;
                               nsamp, window_buffer, target_times, smooth,
                               uncertainty = smooth ? :smoothed : :filtered,
                               spatial_mod, obs_operator, state_in_cov, cov_wt, tscov_pars,
                               tscov_lag = nothing,
                               nodata_reg = 0.0,
                               window_radius, use_progress_bar)
end

"""
    scene_inds_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata, inds,
                           spectral_mean, prior_mean, prior_var, B, model_pars; kwargs...)

Deprecated v1 interface. Like `scene_fusion_pmap`, but processes only the windows in
`inds` (an nr×2 matrix of (k,l) window indices). Calls `gridded_fusion_pmap` with
`inds` and the v1 settings listed for `scene_fusion_pmap`.

Unlike v1, the Sobol BAU subsample is added only when a coarse (fidelity 2)
instrument is present, as in `scene_fusion_pmap`.
"""
function scene_inds_fusion_pmap(inst_data::AbstractVector{InstrumentData},
                      inst_geodata::AbstractVector{InstrumentGeoData},
                      window_geodata::InstrumentGeoData,
                      target_geodata::InstrumentGeoData,
                      inds::AbstractArray{<:Real},
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
                      obs_operator::Function = unif_weighted_obs_operator_centroid,
                      state_in_cov::Bool = true,
                      cov_wt::Real = 0.3,
                      tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                      ar_phi::Real = 1.0,
                      window_radius::Real = 0.0,
                      use_progress_bar::Bool = true)

    @warn """`scene_inds_fusion_pmap` is deprecated; use `gridded_fusion_pmap` with `inds`. To reproduce v1 results:
        gridded_fusion_pmap(..., B, ar_phi*I(size(B,2)), model_pars; inds,
                            tscov_lag = nothing, nodata_reg = 0.0,
                            uncertainty = smooth ? :smoothed : :filtered, ...)""" maxlog=1

    p = size(B, 2)
    G = Matrix{Float64}(ar_phi * I, p, p)

    return gridded_fusion_pmap(inst_data, inst_geodata, window_geodata, target_geodata,
                               spectral_mean, prior_mean, prior_var, B, G, model_pars;
                               nsamp, window_buffer, target_times, smooth,
                               uncertainty = smooth ? :smoothed : :filtered,
                               spatial_mod, obs_operator, state_in_cov, cov_wt, tscov_pars,
                               tscov_lag = nothing,
                               nodata_reg = 0.0,
                               window_radius,
                               inds = Int.(inds),
                               use_progress_bar)
end

###### Swath Fusion Functions
struct SwathData
    data::AbstractVector{AbstractArray} # length T vector of (n x w)
    bias::AbstractVector{Union{Any,AbstractArray}} # length T vector of W error biases 
    uq::AbstractVector{Union{Any,AbstractArray}} # length T vector of W error variances 
    fwhm::AbstractVector{AbstractVector} # length T vector of [rx,ry] vector of fwhm of psf
    dates::AbstractVector # vector of dates
    coords::AbstractVector{AbstractArray} # length T vector of n x 2 array of spatial coordinates
    wavelengths::AbstractVector # w vector of spectral wavelengths
    psf::Function
    psf_samp::Bool
    rsr::Union{Dict,Any,AbstractArray} # Dict{w: wavelengths, rsr:rsr values} for discrete weights and wavelengths, or scalar common fwhm, or w length vector of fwhms
end

### refactoring...
# --- Shared pieces for the swath filter / smoother ---------------------------
function _swath_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod)
    if size(G,2) != size(B,2)
        throw(DimensionMismatch("size(G,2)=$(size(G,2)) does not match size(B,2)=$(size(B,2))"))
    end

    measurements  = @views d[:measurements]
    target_coords = @views d[:target_coords]
    bau_ci        = @views d[:bau_ci]
    kp_ij         = @views d[:kp_ij]
    model_pars    = @views d[:model_pars]
    A_bau = prod(abs.(d[:bau_csize]))

    bau_kp = findall(c -> c[1] ∈ kp_ij[1] && c[2] ∈ kp_ij[2], bau_ci)

    ni = length(measurements)
    nf = size(target_coords, 1)   # target-resolution cells in window + buffer
    p  = size(B, 2)               # latent spectral components
    n  = nf*p
    nw = size(B, 1)               # target wavelengths

    t0 = minimum(m.dates[1]   for m in measurements)
    tt = maximum(m.dates[end] for m in measurements)

    # spectral observation operators
    Hws = Vector(undef, ni)
    Hms = Vector(undef, ni)
    for (i,x) in enumerate(measurements)
        Hw = rsr_conv_matrix(x.rsr, x.wavelengths, target_wavelengths)
        Hws[i] = Hw*B
        Hms[i] = Hw*spectral_mean
    end

    # static spatial process covariance
    Q = zeros(n,n)
    dd = pairwise(Distances.Euclidean(1e-12), target_coords, dims=1)
    for (i,x) in enumerate(eachrow(model_pars))
        ids = ((i-1)*nf+1):i*nf
        @views Q[ids,ids] = x[1] .* spatial_mod(dd, x[2:end])
    end

    # transition matrix kron(G, I)
    F = kron(sparse(G), sparse(I, nf, nf))
    F = isdiag(F) ? Diagonal(diag(F)) : F

    return (; measurements, target_coords, kp_ij, bau_kp, model_pars, A_bau,
              nf, p, n, nw, t0, tt, Hws, Hms, Q, F,
              prior_mean = d[:prior_mean], prior_var = d[:prior_var])
end

function _swath_obs_models(measurements, t2, Hws, Hms, target_coords, A_bau, mcvar, Qf, F)
    Ms = HyperSTARS.HSModel[]
    ys = Vector{Array{Float64}}()
    for (i,x) in enumerate(measurements)
        for kt in findall(x.dates .== t2)
            yss = @views x.data[kt]
            o1 = x.psf(x.coords[kt], target_coords, x.fwhm[kt])

            # drop observations that don't cover any BAUs
            valid_rows = vec(mean(isnan.(o1), dims=2) .== 0) .& vec(sum(abs.(o1), dims=2) .!= 0)
            o2 = o1[valid_rows,:]
            yss_subset = yss[valid_rows, :]

            if x.psf_samp
                norm_constant = 2π * prod(x.fwhm[kt][1:2]) / 2.3548^2
                for (row_idx,rr) in enumerate(eachrow(o2))
                    o_row = @views o2[row_idx:row_idx, :]
                    c = sum(o_row) * A_bau/norm_constant
                    o_row_norm = o_row ./ sum(o_row)
                    push!(Ms, HyperSTARS.HSModel(Hws[i], o_row_norm, Diagonal(x.uq[kt][:] .+ ((1-c)^2 * mcvar)), 1.0*I(size(o_row,1)), Qf, F))
                    push!(ys, yss_subset[row_idx:row_idx,:] .- Hms[i]' .- x.bias[kt]')
                end
            else
                o2_norm = if !all(isapprox.(sum(o2, dims=2), 1.0, atol=1e-6))
                    o2 ./ sum(o2, dims=2)
                else
                    o2
                end
                push!(Ms, HyperSTARS.HSModel(Hws[i], o2_norm, Diagonal(x.uq[kt][:]), 1.0*I(size(o2,1)), Qf, F))
                push!(ys, yss_subset .- Hms[i]' .- x.bias[kt]')
            end
        end
    end
    return Ms, ys
end

function hyperSTARS_swath_filter(d,
                target_wavelengths::AbstractVector{<:Real},
                spectral_mean::AbstractVector{<:Real},
                B::AbstractArray{<:Real},
                G::AbstractArray{<:Real},
                target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1];
                spatial_mod::Function = mat32_corD,
                state_in_cov::Bool = true,
                cov_wt::Real = 0.3,
                tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                tscov_lag = 7,
                mcvar = 1e-6,
                nodata_reg::Real = 1e-10)

    s = _swath_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod)
    (; nf, p, n, nw, Q, F) = s

    times  = minimum([s.t0, minimum(target_times)]):maximum(target_times)
    nsteps = length(times)
    kp_times  = findall(times .∈ Ref(target_times))
    out_index = Dict(t => ti for (ti,t) in enumerate(kp_times))

    filtering_means = zeros(n, nsteps+1)
    filtering_prec  = zeros(n, nsteps+1)
    @views filtering_means[:,1] = s.prior_mean
    @views filtering_prec[:,1]  = 1.0 ./ sqrt.(s.prior_var)

    P = Matrix{Float64}(Diagonal(s.prior_var))   # current filtered covariance only

    nbau = length(s.bau_kp)
    fused_image    = zeros(nbau, nw, length(target_times))
    fused_sd_image = zeros(nbau, nw, length(target_times))

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    FPpred = similar(P_pred)
    Qf     = zeros(n,n)

    for (t,t2) in enumerate(times)
        _process_cov!(Qf, Q, filtering_means, filtering_prec, t, nf, p,
                      s.model_pars, state_in_cov, cov_wt, tscov_pars, tscov_lag)
        Ms, ys = _swath_obs_models(s.measurements, t2, s.Hws, s.Hms,
                                   s.target_coords, s.A_bau, mcvar, Qf, F)

        _predict!(x_pred, P_pred, FPpred, F, @view(filtering_means[:,t]), P, Qf)

        x = @view filtering_means[:,t+1]
        _update!(x, P, Ms, ys, x_pred, P_pred; reg = nodata_reg)
        filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P))

        if haskey(out_index, t)
            _backproject!(fused_image, fused_sd_image, out_index[t], x, P,
                          B, spectral_mean, nf, p, nw, s.bau_kp)
        end
    end

    return s.kp_ij, fused_image, fused_sd_image
end

"""
    hyperSTARS_swath_smooth(d, target_wavelengths, spectral_mean, B, G, target_times; kwargs...)

Forward Kalman filter + backward RTS smoother for one swath window. Smoothed means
are always returned; `uncertainty` selects the reported uncertainty:

- `:filtered` (default): filtered covariances, means-only backward pass (≈ n³/3 per step)
- `:smoothed`: smoothed covariances, full RTS backward pass (≈ 7n³ per step)

Filtered and predicted covariances are stored from the first target time onward.
Returns `(kp_ij, fused_image, fused_sd_image)`.
"""
function hyperSTARS_swath_smooth(d,
                target_wavelengths::AbstractVector{<:Real},
                spectral_mean::AbstractVector{<:Real},
                B::AbstractArray{<:Real},
                G::AbstractArray{<:Real},
                target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1];
                uncertainty::Symbol = :filtered,
                spatial_mod::Function = mat32_corD,
                state_in_cov::Bool = true,
                cov_wt::Real = 0.3,
                tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                tscov_lag = 7,
                mcvar = 1e-6,
                nodata_reg::Real = 1e-10)

    uncertainty in (:filtered, :smoothed) ||
        throw(ArgumentError("uncertainty must be :filtered or :smoothed, got :$uncertainty"))

    s = _swath_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod)
    (; nf, p, n, nw, Q, F) = s

    times  = minimum([s.t0, minimum(target_times)]):maximum([s.tt, maximum(target_times)])
    nsteps = length(times)
    kp_times  = findall(times .∈ Ref(target_times))
    out_index = Dict(t => ti for (ti,t) in enumerate(kp_times))
    st = minimum(kp_times)
    nk = nsteps - st + 1

    filtering_means = zeros(n, nsteps+1)
    filtering_prec  = zeros(n, nsteps+1)
    @views filtering_means[:,1] = s.prior_mean
    @views filtering_prec[:,1]  = 1.0 ./ sqrt.(s.prior_var)

    P = Matrix{Float64}(Diagonal(s.prior_var))

    filt_covs = zeros(n, n, nk)
    pred_covs = zeros(n, n, nk-1)

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    FPpred = similar(P_pred)
    Qf     = zeros(n,n)

    # ---- forward pass ----
    for (t,t2) in enumerate(times)
        _process_cov!(Qf, Q, filtering_means, filtering_prec, t, nf, p,
                      s.model_pars, state_in_cov, cov_wt, tscov_pars, tscov_lag)
        Ms, ys = _swath_obs_models(s.measurements, t2, s.Hws, s.Hms,
                                   s.target_coords, s.A_bau, mcvar, Qf, F)

        _predict!(x_pred, P_pred, FPpred, F, @view(filtering_means[:,t]), P, Qf)
        if t > st
            pred_covs[:,:,t-st] .= P_pred
        end

        x = @view filtering_means[:,t+1]
        _update!(x, P, Ms, ys, x_pred, P_pred; reg = nodata_reg)
        filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P))

        if t >= st
            filt_covs[:,:,t-st+1] .= P
        end
    end

    # ---- backward pass, writing outputs as target times are reached ----
    nbau = length(s.bau_kp)
    fused_image    = zeros(nbau, nw, length(target_times))
    fused_sd_image = zeros(nbau, nw, length(target_times))

    function write_output(t, x, Σ)
        if haskey(out_index, t)
            _backproject!(fused_image, fused_sd_image, out_index[t], x, Σ,
                          B, spectral_mean, nf, p, nw, s.bau_kp)
        end
        return nothing
    end

    if uncertainty === :filtered
        _rts_means!(write_output, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    else
        _rts_full!(write_output, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    end

    return s.kp_ij, fused_image, fused_sd_image
end

function _swath_window_dicts(inst_data::AbstractVector{SwathData},
                             target_geodata::InstrumentGeoData,
                             target_coords::AbstractArray{<:Real},
                             target_bau_ids::AbstractVector,
                             window_inst_ids::AbstractVector,
                             window_bau_ids::AbstractVector,
                             prior_mean::AbstractArray{<:Real},
                             prior_var::AbstractArray{<:Real},
                             model_pars)

    ni = length(inst_data)
    nr = length(window_bau_ids)
    ci = CartesianIndices(Tuple(target_geodata.ndims))
    bau_csize = target_geodata.cell_size

    T = Vector{Dict{Symbol,Any}}(undef, nr)
    for ii in 1:nr
        measurements = Vector{SwathData}(undef, ni)
        for i in 1:ni
            ys        = [inst_data[i].data[t][x[ii],:]   for (t,x) in enumerate(window_inst_ids[i])]
            sp_coords = [inst_data[i].coords[t][x[ii],:] for (t,x) in enumerate(window_inst_ids[i])]
            measurements[i] = SwathData(ys,
                                inst_data[i].bias,
                                inst_data[i].uq,
                                inst_data[i].fwhm,
                                inst_data[i].dates,
                                sp_coords,
                                inst_data[i].wavelengths,
                                inst_data[i].psf,
                                inst_data[i].psf_samp,
                                inst_data[i].rsr)
        end

        bau_ids = sort(window_bau_ids[ii])
        bau_ci  = ci[bau_ids]

        d = Dict{Symbol,Any}()
        d[:measurements]  = measurements
        d[:target_coords] = target_coords[bau_ids,:]
        d[:bau_ci]        = bau_ci
        d[:kp_ij]         = target_bau_ids[ii]
        d[:prior_mean]    = @views prior_mean[bau_ci,:][:]
        d[:prior_var]     = @views prior_var[bau_ci,:][:]
        d[:model_pars]    = @views model_pars[ii]
        d[:bau_csize]     = bau_csize
        T[ii] = d
    end
    return T
end

function swath_fusion_pmap(inst_data::AbstractVector{SwathData},
                      target_geodata::InstrumentGeoData,
                      target_coords::AbstractArray{<:Real},
                      target_bau_ids::AbstractVector,
                      window_inst_ids::AbstractVector,
                      window_bau_ids::AbstractVector,
                      spectral_mean::AbstractVector{<:Real},
                      prior_mean::AbstractArray{<:Real},
                      prior_var::AbstractArray{<:Real},
                      B::AbstractArray{<:Real},
                      G::AbstractArray{<:Real},
                      model_pars;
                      target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1],
                      smooth::Bool = false,
                      uncertainty::Symbol = :filtered,
                      spatial_mod::Function = mat32_corD,
                      state_in_cov::Bool = true,
                      cov_wt::Real = 0.3,
                      tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                      tscov_lag = 7,
                      mcvar::Real = 1e-6,
                      nodata_reg::Real = 1e-10)

    if size(G,2) != size(B,2)
        throw(DimensionMismatch("size(G,2)=$(size(G,2)) does not match size(B,2)=$(size(B,2))"))
    end

    uncertainty in (:filtered, :smoothed) ||
        throw(ArgumentError("uncertainty must be :filtered or :smoothed, got :$uncertainty"))
    !smooth && uncertainty === :smoothed &&
        throw(ArgumentError("uncertainty = :smoothed requires smooth = true"))

    target_waves = target_geodata.wavelengths
    target_ndims = target_geodata.ndims
    nw = length(target_waves)
    nt = length(target_times)        # number of output time steps

    fused_image    = zeros(target_ndims[1], target_ndims[2], nw, nt)
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], nw, nt)

    T = _swath_window_dicts(inst_data, target_geodata, target_coords, target_bau_ids,
                            window_inst_ids, window_bau_ids, prior_mean, prior_var, model_pars)

    common_kw = (; spatial_mod, state_in_cov, cov_wt, tscov_pars, tscov_lag, mcvar, nodata_reg)
    run_window(d) = smooth ?
        hyperSTARS_swath_smooth(d, target_waves, spectral_mean, B, G, target_times;
                                uncertainty, common_kw...) :
        hyperSTARS_swath_filter(d, target_waves, spectral_mean, B, G, target_times;
                                common_kw...)

    result = @showprogress pmap(run_window, T)
    
    # mosaic the kept cells of each window
    for (kp_ij, img, sd_img) in result
        nx, ny = length(kp_ij[1]), length(kp_ij[2])
        @views fused_image[kp_ij...,:,:]    = reshape(img,    nx, ny, nw, nt)
        @views fused_sd_image[kp_ij...,:,:] = reshape(sd_img, nx, ny, nw, nt)
    end

    return fused_image, fused_sd_image
end

# --- Grid (fixed-footprint) window helpers ------------------------------------
function _grid_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod, obs_operator)
    if size(G,2) != size(B,2)
        throw(DimensionMismatch("size(G,2)=$(size(G,2)) does not match size(B,2)=$(size(B,2))"))
    end

    measurements  = @views d[:measurements]
    target_coords = @views d[:target_coords]
    kp_ij         = @views d[:kp_ij]
    model_pars    = @views d[:model_pars]

    nbau = size(kp_ij, 1)
    ni = length(measurements)
    nf = size(target_coords, 1)   # target-resolution cells in window + buffer
    p  = size(B, 2)               # latent spectral components
    n  = nf*p
    nw = size(B, 1)               # target wavelengths

    t0 = minimum(m.dates[1]   for m in measurements)
    tt = maximum(m.dates[end] for m in measurements)

    Hss = Vector(undef, ni)        # spatial operators (fixed per instrument)
    Hs_valid = Vector(undef, ni)   # rows of Hss with no NaNs (fixed, so computed once)
    Hws = Vector(undef, ni)
    Hms = Vector(undef, ni)
    for (i,x) in enumerate(measurements)
        Hss[i] = obs_operator(x.coords, target_coords, x.spatial_resolution)
        Hs_valid[i] = mean(isnan.(Hss[i]), dims=2)[:] .== 0
        Hw = rsr_conv_matrix(x.rsr, x.wavelengths, target_wavelengths)
        Hws[i] = Hw*B
        Hms[i] = Hw*spectral_mean
    end

    # static spatial process covariance
    Q = zeros(n,n)
    dd = pairwise(Euclidean(1e-12), target_coords, dims=1)
    for (i,x) in enumerate(eachrow(model_pars))
        ids = ((i-1)*nf+1):i*nf
        @views Q[ids,ids] = x[1] .* spatial_mod(dd, x[2:end])
    end

    # transition matrix kron(G, I), as in the swath workflow
    F = kron(sparse(G), sparse(I, nf, nf))
    F = isdiag(F) ? Diagonal(diag(F)) : F

    return (; measurements, target_coords, kp_ij, rows = 1:nbau, model_pars,
              nf, p, n, nw, t0, tt, Hss, Hs_valid, Hws, Hms, Q, F,
              prior_mean = d[:prior_mean], prior_var = d[:prior_var])
end

function _grid_obs_models(measurements, t2, Hss, Hs_valid, Hws, Hms, Qf, F)
    Ms = HyperSTARS.HSModel[]
    ys = Vector{Array{Float64}}()
    for x in eachindex(measurements)
        kt = measurements[x].dates .== t2
        any(kt) || continue

        yss  = @views measurements[x].data[:,:,kt]
        keep = .!vec(any(isnan, yss; dims=2)) .& Hs_valid[x]   # data present and operator row valid
        any(keep) || continue

        Hs2 = Hss[x][keep,:]
        push!(Ms, HyperSTARS.HSModel(Hws[x], Hs2, Diagonal(measurements[x].uq[:,kt][:]), 1.0*I(size(Hs2,1)), Qf, F))
        push!(ys, yss[keep,:] .- Hms[x]' .- measurements[x].bias[:,kt]')
    end
    return Ms, ys
end

# --- Grid filter ----------------------------------------------------------------
function hyperSTARS_grid_filter(d,
                target_wavelengths::AbstractVector{<:Real},
                spectral_mean::AbstractVector{<:Real},
                B::AbstractArray{<:Real},
                G::AbstractArray{<:Real},
                target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1];
                spatial_mod::Function = mat32_corD,
                obs_operator::Function = unif_weighted_obs_operator_centroid,
                state_in_cov::Bool = true,
                cov_wt::Real = 0.3,
                tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                tscov_lag = 7,
                nodata_reg::Real = 1e-10)

    s = _grid_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod, obs_operator)
    (; nf, p, n, nw, Q, F) = s

    times  = minimum([s.t0, minimum(target_times)]):maximum(target_times)
    nsteps = length(times)
    kp_times  = findall(times .∈ Ref(target_times))
    out_index = Dict(t => ti for (ti,t) in enumerate(kp_times))

    # Mean/precision history (needed for Qf when state_in_cov).
    # Column 1 = prior, column t+1 = filtered at times[t].
    filtering_means = zeros(n, nsteps+1)
    filtering_prec  = zeros(n, nsteps+1)
    @views filtering_means[:,1] = s.prior_mean
    @views filtering_prec[:,1]  = 1.0 ./ sqrt.(s.prior_var)

    P = Matrix{Float64}(Diagonal(s.prior_var))   # current filtered covariance only

    nbau = length(s.rows)
    fused_image    = zeros(nbau, nw, length(target_times))
    fused_sd_image = zeros(nbau, nw, length(target_times))

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    FPpred = similar(P_pred)
    Qf     = zeros(n,n)

    for (t,t2) in enumerate(times)
        _process_cov!(Qf, Q, filtering_means, filtering_prec, t, nf, p,
                      s.model_pars, state_in_cov, cov_wt, tscov_pars, tscov_lag)
        Ms, ys = _grid_obs_models(s.measurements, t2, s.Hss, s.Hs_valid, s.Hws, s.Hms, Qf, F)

        _predict!(x_pred, P_pred, FPpred, F, @view(filtering_means[:,t]), P, Qf)

        x = @view filtering_means[:,t+1]
        _update!(x, P, Ms, ys, x_pred, P_pred; reg = nodata_reg)
        filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P))

        if haskey(out_index, t)
            _backproject!(fused_image, fused_sd_image, out_index[t], x, P,
                          B, spectral_mean, nf, p, nw, s.rows)
        end
    end

    return s.kp_ij, fused_image, fused_sd_image
end

# --- Grid smoother (means only) -------------------------------------------------
"""
    hyperSTARS_grid_smooth(d, target_wavelengths, spectral_mean, B, G, target_times; kwargs...)

Forward Kalman filter + backward RTS smoother for one grid window. Smoothed means
are always returned; `uncertainty` selects the reported uncertainty:

- `:filtered` (default): filtered covariances, means-only backward pass (≈ n³/3 per step)
- `:smoothed`: smoothed covariances, full RTS backward pass (≈ 7n³ per step)

Filtered and predicted covariances are stored from the first target time onward.
Returns `(kp_ij, fused_image, fused_sd_image)`.
"""
function hyperSTARS_grid_smooth(d,
                target_wavelengths::AbstractVector{<:Real},
                spectral_mean::AbstractVector{<:Real},
                B::AbstractArray{<:Real},
                G::AbstractArray{<:Real},
                target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1];
                uncertainty::Symbol = :filtered,
                spatial_mod::Function = mat32_corD,
                obs_operator::Function = unif_weighted_obs_operator_centroid,
                state_in_cov::Bool = true,
                cov_wt::Real = 0.3,
                tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                tscov_lag = 7,
                nodata_reg::Real = 1e-10)

    uncertainty in (:filtered, :smoothed) ||
        throw(ArgumentError("uncertainty must be :filtered or :smoothed, got :$uncertainty"))

    s = _grid_setup(d, target_wavelengths, spectral_mean, B, G, spatial_mod, obs_operator)
    (; nf, p, n, nw, Q, F) = s

    # Smoother runs to the end of the data, not just the last target time
    times  = minimum([s.t0, minimum(target_times)]):maximum([s.tt, maximum(target_times)])
    nsteps = length(times)
    kp_times  = findall(times .∈ Ref(target_times))
    out_index = Dict(t => ti for (ti,t) in enumerate(kp_times))
    st = minimum(kp_times)
    nk = nsteps - st + 1

    filtering_means = zeros(n, nsteps+1)
    filtering_prec  = zeros(n, nsteps+1)
    @views filtering_means[:,1] = s.prior_mean
    @views filtering_prec[:,1]  = 1.0 ./ sqrt.(s.prior_var)

    P = Matrix{Float64}(Diagonal(s.prior_var))   # current filtered covariance

    # filt_covs[:,:,k]   = P_{t|t}   at times[st+k-1],  k = 1..nk
    # pred_covs[:,:,k-1] = P_{t|t-1} at times[st+k-1],  k = 2..nk
    filt_covs = zeros(n, n, nk)
    pred_covs = zeros(n, n, nk-1)

    x_pred = zeros(n)
    P_pred = zeros(n,n)
    FPpred = similar(P_pred)
    Qf     = zeros(n,n)

    # ---- forward pass ----
    for (t,t2) in enumerate(times)
        _process_cov!(Qf, Q, filtering_means, filtering_prec, t, nf, p,
                      s.model_pars, state_in_cov, cov_wt, tscov_pars, tscov_lag)
        Ms, ys = _grid_obs_models(s.measurements, t2, s.Hss, s.Hs_valid, s.Hws, s.Hms, Qf, F)

        _predict!(x_pred, P_pred, FPpred, F, @view(filtering_means[:,t]), P, Qf)
        if t > st
            pred_covs[:,:,t-st] .= P_pred
        end

        x = @view filtering_means[:,t+1]
        _update!(x, P, Ms, ys, x_pred, P_pred; reg = nodata_reg)
        filtering_prec[:,t+1] = 1.0 ./ sqrt.(diag(P))

        if t >= st
            filt_covs[:,:,t-st+1] .= P
        end
    end

    # ---- backward pass, writing outputs as target times are reached ----
    nbau = length(s.rows)
    fused_image    = zeros(nbau, nw, length(target_times))
    fused_sd_image = zeros(nbau, nw, length(target_times))

    function write_output(t, x, Σ)
        if haskey(out_index, t)
            _backproject!(fused_image, fused_sd_image, out_index[t], x, Σ,
                          B, spectral_mean, nf, p, nw, s.rows)
        end
        return nothing
    end

    if uncertainty === :filtered
        _rts_means!(write_output, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    else
        _rts_full!(write_output, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    end

    return s.kp_ij, fused_image, fused_sd_image
end

# --- Grid window construction and scene driver ---------------------------------
function _grid_window_dicts(inst_data::AbstractVector{InstrumentData},
                            inst_geodata::AbstractVector{InstrumentGeoData},
                            window_geodata::InstrumentGeoData,
                            target_geodata::InstrumentGeoData,
                            prior_mean::AbstractArray{<:Real},
                            prior_var::AbstractArray{<:Real},
                            model_pars::AbstractArray{<:Real};
                            nsamp::Integer = 100,
                            window_buffer::Integer = 2,
                            window_radius::Real = 0.0,
                            inds::Union{Nothing,AbstractMatrix{<:Integer}} = nothing)

    window_csize  = window_geodata.cell_size
    target_csize  = target_geodata.cell_size
    window_origin = window_geodata.origin
    nwindows      = window_geodata.ndims
    target_origin = target_geodata.origin
    target_ndims  = target_geodata.ndims

    # Windows to process, one (k,l) per row. Default: every window in the grid.
    if isnothing(inds)
        inds = hcat(repeat(1:nwindows[1], inner=nwindows[2]), repeat(1:nwindows[2], outer=nwindows[1]))
    else
        size(inds, 2) == 2 ||
            throw(DimensionMismatch("inds must be an nr×2 matrix of (k,l) window indices, got size $(size(inds))"))
        (all(1 .<= inds[:,1] .<= nwindows[1]) && all(1 .<= inds[:,2] .<= nwindows[2])) ||
            throw(ArgumentError("inds contains window indices outside 1:$(nwindows[1]) × 1:$(nwindows[2])"))
        allunique(eachrow(inds)) ||
            throw(ArgumentError("inds contains duplicate windows"))
    end
    nr = size(inds,1)

    T = Vector{Dict{Symbol,Any}}(undef, nr)
    for ii in 1:nr
        k,l = inds[ii,:]

        window_bbox = bbox_from_centroid(window_origin .+ [k-1, l-1].*window_csize, window_csize)
        buffer_ext = window_bbox .+ window_buffer*[-1.01,1.01]*target_csize'
        all_exts = [Matrix(find_overlapping_ext(buffer_ext[1,:], buffer_ext[2,:], x.origin, x.cell_size)) for x in inst_geodata]
        res_flag = [x.fidelity for x in inst_geodata]

        exx = window_bbox .+ [-window_radius - 0.01, window_radius + 0.01]
        push!(all_exts, exx)

        full_ext = merge_extents(all_exts, sign.(target_csize))
        target_ij = find_all_ij_ext(window_bbox[1,:], window_bbox[2,:], target_origin, target_csize, target_ndims; inclusive=false)
        ss_ij = find_all_ij_ext(buffer_ext[1,:], buffer_ext[2,:], target_origin, target_csize, target_ndims)
        if any(res_flag .== 2)
            ss_ij = unique(vcat(ss_ij, sobol_bau_ij(full_ext[1,:], full_ext[2,:], target_origin, target_csize, target_ndims; nsamp=nsamp)), dims=1)
        end

        ss_xy = get_sij_from_ij(ss_ij, target_origin, target_csize)
        measurements, ss_ij = organize_data(full_ext, inst_geodata, inst_data, target_geodata, ss_xy, ss_ij, res_flag)
        bau_ij = unique(vcat(target_ij, ss_ij), dims=1)
        bau_coords = get_sij_from_ij(bau_ij, target_origin, target_csize)
        bau_ci = CartesianIndex.(bau_ij[:,1], bau_ij[:,2])

        d = Dict{Symbol,Any}()
        d[:measurements]  = measurements
        d[:target_coords] = bau_coords
        d[:kp_ij]         = CartesianIndex.(target_ij[:,1], target_ij[:,2])
        d[:prior_mean]    = @views prior_mean[bau_ci,:][:]
        d[:prior_var]     = @views prior_var[bau_ci,:][:]
        d[:model_pars]    = @views model_pars[k,l,:,:]
        T[ii] = d
    end
    return T, inds
end

function gridded_fusion_pmap(inst_data::AbstractVector{InstrumentData},
                      inst_geodata::AbstractVector{InstrumentGeoData},
                      window_geodata::InstrumentGeoData,
                      target_geodata::InstrumentGeoData,
                      spectral_mean::AbstractVector{<:Real},
                      prior_mean::AbstractArray{<:Real},
                      prior_var::AbstractArray{<:Real},
                      B::AbstractArray{<:Real},
                      G::AbstractArray{<:Real},
                      model_pars::AbstractArray{<:Real};
                      nsamp::Integer = 100,
                      window_buffer::Integer = 2,
                      target_times::Union{AbstractVector{<:Real}, UnitRange{<:Real}} = [1],
                      smooth::Bool = false,
                      uncertainty::Symbol = :filtered,
                      spatial_mod::Function = mat32_corD,
                      obs_operator::Function = unif_weighted_obs_operator_centroid,
                      state_in_cov::Bool = true,
                      cov_wt::Real = 0.3,
                      tscov_pars::Union{Nothing,AbstractVector{<:Real}} = nothing,
                      tscov_lag = 7,
                      nodata_reg::Real = 1e-10,
                      window_radius::Real = 0.0,
                      inds::Union{Nothing,AbstractMatrix{<:Integer}} = nothing,
                      use_progress_bar::Bool = true)

    if size(G,2) != size(B,2)
        throw(DimensionMismatch("size(G,2)=$(size(G,2)) does not match size(B,2)=$(size(B,2))"))
    end

    uncertainty in (:filtered, :smoothed) ||
        throw(ArgumentError("uncertainty must be :filtered or :smoothed, got :$uncertainty"))
    !smooth && uncertainty === :smoothed &&
        throw(ArgumentError("uncertainty = :smoothed requires smooth = true"))

    target_waves = target_geodata.wavelengths
    target_ndims = target_geodata.ndims
    nw = length(target_waves)
    nt = length(target_times)        # number of output time steps

    fused_image    = zeros(target_ndims[1], target_ndims[2], nw, nt)
    fused_sd_image = zeros(target_ndims[1], target_ndims[2], nw, nt)

    T, inds = _grid_window_dicts(inst_data, inst_geodata, window_geodata, target_geodata,
                                 prior_mean, prior_var, model_pars;
                                 nsamp, window_buffer, window_radius, inds)
    nr = length(T)

    common_kw = (; spatial_mod, obs_operator, state_in_cov, cov_wt,
                   tscov_pars, tscov_lag, nodata_reg)
    run_window(d) = smooth ?
        hyperSTARS_grid_smooth(d, target_waves, spectral_mean, B, G, target_times;
                               uncertainty, common_kw...) :
        hyperSTARS_grid_filter(d, target_waves, spectral_mean, B, G, target_times;
                               common_kw...)

    format_duration(seconds::Real) = begin
        total_seconds = max(round(Int, seconds), 0)
        hours   = div(total_seconds, 3600)
        minutes = div(mod(total_seconds, 3600), 60)
        secs    = mod(total_seconds, 60)
        lpad(hours, 2, '0') * ":" * lpad(minutes, 2, '0') * ":" * lpad(secs, 2, '0')
    end

    if use_progress_bar
        result = @showprogress pmap(run_window, T)
    else
        progress_updates = RemoteChannel(() -> Channel{NamedTuple{(:ii, :window, :window_elapsed), Tuple{Int, NTuple{2, Int}, Float64}}}(nr))
        progress_start = time()

        progress_logger = @async begin
            completed = 0
            while completed < nr
                update = take!(progress_updates)
                completed += 1
                elapsed = time() - progress_start
                avg_per_window = elapsed / completed
                eta = avg_per_window * (nr - completed)
                @info "Completed window fusion iteration $(completed)/$(nr)" window_index=update.ii window=update.window elapsed=format_duration(elapsed) window_time=format_duration(update.window_elapsed) avg_per_window=format_duration(avg_per_window) eta=format_duration(eta)
            end
        end

        result = pmap(enumerate(T)) do (ii, d)
            k, l = inds[ii, :]
            window_start = time()
            out = run_window(d)
            put!(progress_updates, (ii=ii, window=(k, l), window_elapsed=time() - window_start))
            out
        end

        wait(progress_logger)
    end

    # mosaic the kept cells of each window
    for (kp_ij, img, sd_img) in result
        @views fused_image[kp_ij,:,:]    = img
        @views fused_sd_image[kp_ij,:,:] = sd_img
    end

    return fused_image, fused_sd_image
end

end # end of module