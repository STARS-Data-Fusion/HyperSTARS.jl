using LinearAlgebra
using Statistics
using StatsBase
using Distances
import GaussianRandomFields.CovarianceFunction # Used for defining covariance functions
import GaussianRandomFields.Matern # Matern covariance function from GaussianRandomFields
import GaussianRandomFields.apply # Applying covariance functions from GaussianRandomFields
using KernelFunctions # Another package for various kernel functions and kernel matrices

"""
    kernel_matrix(X; reg=1e-10, σ=1.0)

Computes a squared exponential (Gaussian/Radial Basis Function) kernel matrix
for input data `X`. A small regularization (`reg`) is added to the diagonal
to ensure numerical stability (nugget effect).

# Arguments
- `X::AbstractArray{T}`: Input data matrix. For pairwise distances with `dims=1`,
  rows are observations and columns are features (e.g., spatial coordinates).
- `reg::Real`: Regularization parameter (nugget effect) added to the diagonal.
- `σ::Real`: Length-scale parameter of the squared exponential kernel.

# Returns
- `A::AbstractMatrix{T}`: The computed kernel (covariance) matrix.

# Formula
\$ K(x_i, x_j) = \\exp\\left( -\\frac{1}{2} \\frac{\\|x_i - x_j\\|^2}{\\sigma^2} \\right) \$
The matrix includes a nugget effect: \$ K_{ii} = K(x_i, x_i) + \\text{reg} \$
"""
function kernel_matrix(X::AbstractArray{T}; reg=1e-10, σ=1.0) where {T<:Real}
    # Add a regularization (nugget) term to the diagonal for numerical stability.
    # This represents uncorrelated measurement noise or very small-scale variations.
    # Compute squared Euclidean distances between all pairs of rows in X.
    # pairwise(SqEuclidean(1e-12), X, dims=1) computes ||X_i - X_j||^2.
    # The 1e-12 is a small epsilon to prevent issues with zero distances.
    # exp.(-0.5 * ... ./ σ^2) applies the squared exponential kernel formula element-wise.
    Diagonal(reg * ones(size(X)[1])) + exp.(-0.5 * pairwise(SqEuclidean(1e-12), X, dims=1) ./ σ^2)
end

"""
    matern_cor(X, pars)

Computes a Matern covariance matrix for input data `X` using the `GaussianRandomFields` package.

# Arguments
- `X::AbstractArray{T}`: Input data matrix. For `apply(cc, X, X)`, `X` typically
  represents spatial coordinates where rows are points and columns are dimensions.
- `pars::AbstractVector{T}``: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).
    - `pars[3]`: Smoothness parameter (ν).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern covariance matrix.
"""
function matern_cor(X::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale parameter
    reg = pars[2] # Nugget effect
    ν = pars[3] # Smoothness parameter (e.g., 1.5 for Matern 3/2, 2.5 for Matern 5/2)
    
    # Create a Matern covariance function object.
    # `CovarianceFunction(2, Matern(σ, ν))` indicates 2-dimensional data and Matern kernel.
    cc = CovarianceFunction(2, Matern(σ, ν))
    
    # Add a nugget effect to the diagonal (I for identity matrix).
    # `apply(cc, X, X)` computes the full Matern covariance matrix for points in X.
    Diagonal(reg * ones(size(X)[2])) + apply(cc, X, X)
end

"""
    matern_cor_nonsym(X1, X2, pars)

Computes a non-symmetric Matern kernel matrix between two sets of input data `X1` and `X2`,
using the `KernelFunctions` package. This is useful for cross-covariance between different sets of points.

# Arguments
- `X1::AbstractArray{T}`: First input data matrix.
- `X2::AbstractArray{T}`: Second input data matrix.
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[3]`: Smoothness parameter (ν). (Note: pars[2] is unused for `matern_cor_nonsym` based on typical Matern parameterization in KernelFunctions)

# Returns
- `K::AbstractMatrix{T}`: The computed Matern kernel (cross-covariance) matrix.
"""
function matern_cor_nonsym(X1::AbstractArray{T}, X2::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale parameter
    ν = pars[3] # Smoothness parameter
    
    # Create a Matern kernel object with a specified smoothness (ν) and length-scale (σ).
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    
    # Compute the kernel matrix between X1 and X2.
    # `obsdim=2` indicates that observations are along the second dimension (columns).
    kernelmatrix(k,X1,X2, obsdim=2)
end

"""
    matern_cor_fast(X1, pars)

Computes a symmetric Matern kernel matrix for a single set of input data `X1`,
using the `KernelFunctions` package. This is a faster alternative for symmetric cases
compared to `matern_cor_nonsym(X1, X1, pars)`.

# Arguments
- `X1::AbstractArray{T}`: Input data matrix.
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[3]`: Smoothness parameter (ν).

# Returns
- `K::AbstractMatrix{T}`: The computed symmetric Matern kernel matrix.
"""
function matern_cor_fast(X1::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale parameter
    ν = pars[3] # Smoothness parameter
    
    # Create a Matern kernel object.
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    
    # Compute the symmetric kernel matrix for X1.
    kernelmatrix(k,X1, obsdim=2)
end

"""
    build_GP_var(locs, sigma, phi, nugget=1e-10)

Constructs a Gaussian Process covariance matrix based on a squared exponential kernel,
given locations, amplitude, and length-scale.

# Arguments
- `locs`: Spatial locations of the points.
- `sigma::Real`: The overall amplitude (variance) of the GP.
- `phi::Real`: The length-scale parameter of the kernel.
- `nugget::Real = 1e-10`: A small value added to the diagonal for numerical stability (nugget effect).

# Returns
- `A::AbstractMatrix`: The computed GP covariance matrix.
"""
function build_GP_var(locs, sigma, phi, nugget=1e-10)
    # The `kernel_matrix` function computes the correlation part.
    # `sigma` scales this correlation to produce the covariance.
    A = sigma .* kernel_matrix(locs, reg=nugget, σ=phi)
    return A
end

"""
    exp_cor(X, pars)

Computes an Exponential covariance matrix from spatial coordinates `X`.

# Arguments
- `X::AbstractArray{T}`: Input data matrix (spatial coordinates).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Exponential covariance matrix.

# Formula
\$ K(h) = \\exp\\left( -\\frac{\\|h\\|}{\\sigma} \\right) + \\text{reg} \\cdot I \$
where `h` is the distance between points.
"""
function exp_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Compute Euclidean distances between all pairs of columns (dims=2).
    # Then apply the exponential function element-wise.
    # Add a nugget effect (identity matrix scaled by `reg`).
    Diagonal(reg * ones(size(X)[2])) + exp.(-pairwise(Euclidean(1e-12), X, dims=2) ./ σ)
end

"""
    mat32_cor(X, pars)

Computes a Matern 3/2 covariance matrix from spatial coordinates `X`.

# Arguments
- `X::AbstractArray{T}`: Input data matrix (spatial coordinates).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 3/2 covariance matrix.

# Formula
\$ K(h) = \\left( 1 + \\frac{\\sqrt{3}\\|h\\|}{\\sigma} \\right) \\exp\\left( -\\frac{\\sqrt{3}\\|h\\|}{\\sigma} \\right) + \\text{reg} \\cdot I \$
where `h` is the distance between points.
"""
function mat32_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Calculate scaled Euclidean distances: sqrt(3) * ||h|| / σ
    dd = sqrt(3) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    
    # Apply the Matern 3/2 formula element-wise and add nugget.
    reg * I + exp.(-dd).*(1.0 .+ dd)
end

"""
    mat52_cor(X, pars)

Computes a Matern 5/2 covariance matrix from spatial coordinates `X`.

# Arguments
- `X::AbstractArray{T}`: Input data matrix (spatial coordinates).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 5/2 covariance matrix.

# Formula
\$ K(h) = \\left( 1 + \\frac{\\sqrt{5}\\|h\\|}{\\sigma} + \\frac{5\\|h\\|^2}{3\\sigma^2} \\right) \\exp\\left( -\\frac{\\sqrt{5}\\|h\\|}{\\sigma} \\right) + \\text{reg} \\cdot I \$
where `h` is the distance between points.
"""
function mat52_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Calculate scaled Euclidean distances: sqrt(5) * ||h|| / σ
    dd = sqrt(5) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    
    # Apply the Matern 5/2 formula element-wise and add nugget.
    Diagonal(reg * ones(size(X)[2])) + exp.(-dd).*(1.0 .+ dd .+ dd.^2 ./ 3.0) # Corrected formula term
end


"""
    exp_corD(dd, pars)

Computes an Exponential covariance matrix given a precomputed distance matrix `dd`.
This is useful when distances are already calculated to avoid redundant computations.

# Arguments
- `dd::AbstractArray{T}`: Precomputed distance matrix (e.g., Euclidean distances).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Exponential covariance matrix.
"""
function exp_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Apply the Exponential kernel formula directly to the distances `dd`.
    # Add a nugget effect (identity matrix scaled by `reg`).
    reg * I + exp.(-dd ./ σ)
end

"""
    mat32_corD(dd, pars)

Computes a Matern 3/2 covariance matrix given a precomputed distance matrix `dd`.

# Arguments
- `dd::AbstractArray{T}`: Precomputed distance matrix.
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 3/2 covariance matrix.
"""
function mat32_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Scale the distances as required by the Matern 3/2 formula.
    dd_scaled = sqrt(3) .* dd ./ σ
    
    # Apply the Matern 3/2 formula element-wise and add nugget.
    reg * I + exp.(-dd_scaled).*(1.0 .+ dd_scaled)
end

"""
    mat52_corD(dd, pars)

Computes a Matern 5/2 covariance matrix given a precomputed distance matrix `dd`.

# Arguments
- `dd::AbstractArray{T}`: Precomputed distance matrix.
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 5/2 covariance matrix.
"""
function mat52_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1] # Length-scale
    reg = pars[2] # Nugget effect
    
    # Scale the distances as required by the Matern 5/2 formula.
    dd_scaled = sqrt(5) .* dd ./ σ
    
    # Apply the Matern 5/2 formula element-wise and add nugget.
    reg * I + exp.(-dd_scaled).*(1.0 .+ dd_scaled .+ dd_scaled.^2 ./ 3.0) # Corrected formula term
end

"""
    state_cov(Xtt, pars)

Computes a state-dependent covariance matrix, typically used as an adaptive
process noise covariance in a Kalman filter. It uses an Exponential kernel
whose length-scale is adaptively set based on the median distance of the input data.

# Arguments
- `Xtt::AbstractArray{T}`: Input data, likely representing past state estimates
  (e.g., spatial coordinates of latent states over time).
- `pars::AbstractVector{T}`: Parameters, where `pars[1]` is the amplitude/variance.

# Returns
- `Qst::AbstractMatrix{T}`: The computed state-dependent covariance matrix.

# Behavior
The length-scale (`phi`) for the Exponential kernel is determined by the maximum of
a small epsilon (0.00001) and the median of all pairwise Euclidean distances in `Xtt`.
This makes the covariance structure adaptive to the spread of the data.
A small nugget effect (1e-10) is always added for numerical stability.
"""
function state_cov(Xtt::AbstractArray{T}, pars::AbstractVector{T}) where T<:Real
    # Compute pairwise Euclidean distances between columns of Xtt.
    # Xtt typically has rows as features (e.g., latent spatial dimension)
    # and columns as samples (e.g., time steps).
    dd = pairwise(Euclidean(1e-12), Xtt, dims=2) 
    
    # Set the length-scale (`phi`) adaptively.
    # It's the maximum of a small value and the median of all distances.
    # This prevents `phi` from becoming too small (leading to numerical issues or a very spiky kernel).
    phi = maximum([0.00001,median(dd[:])])
    
    # Compute the Exponential kernel based on `pars[1]` (amplitude) and the adaptive `phi`.
    # Add a small nugget effect for numerical stability.
    Qst = pars[1] .* exp.(-dd./phi) + UniformScaling(1e-10)
    return Qst
end

"""
    build_gpcov(Xt, model_pars, kernel_fun)

Constructs a block-diagonal Gaussian Process covariance matrix. Each block
corresponds to a different latent component, with its own set of model parameters.

# Arguments
- `Xt::AbstractArray{<:Real}`: Input data (e.g., spatial coordinates) for which to
  compute the covariance.
- `model_pars::AbstractArray{<:Real,2}`: A 2D array where each row contains
  the parameters for one latent component's kernel function.
- `kernel_fun::Function`: The kernel function to use (e.g., `mat32_corD`, `exp_corD`).

# Returns
- `Q::AbstractMatrix{Float64}`: The block-diagonal GP covariance matrix.
"""
function build_gpcov(Xt::AbstractArray{<:Real}, model_pars::AbstractArray{<:Real,2}, kernel_fun::Function)
    Qs = Vector{Matrix{Float64}}(undef,size(model_pars)[1]) # Vector to hold covariance matrix for each block/latent component

    # Iterate through each row of `model_pars` (each row corresponds to parameters for one latent component)
    for (i,x) in enumerate(eachrow(model_pars))
        # Compute the kernel matrix for the current latent component.
        # `x[1]` is the amplitude, `x[2:end]` are other kernel parameters (e.g., length-scale, smoothness).
        Qs[i] = x[1] .* kernel_fun(Xt, x[2:end]) 
    end

    # Combine the individual covariance matrices into a single block-diagonal matrix.
    # This assumes independence between the different latent components.
    Q = Matrix(BlockDiagonal(Qs))
    return Q
end

"""
    build_gpcov(Xt, model_pars, kernel_fun)

Constructs a single Gaussian Process covariance matrix when there is only one set
of model parameters (e.g., for a single latent component).

# Arguments
- `Xt::AbstractArray{<:Real}`: Input data (e.g., spatial coordinates).
- `model_pars::AbstractVector{<:Real}`: A vector containing parameters for the kernel function.
- `kernel_fun::Function`: The kernel function to use.

# Returns
- `Q::AbstractMatrix{Float64}`: The computed GP covariance matrix.
"""
function build_gpcov(Xt::AbstractArray{<:Real}, model_pars::AbstractVector{<:Real}, kernel_fun::Function)
    # Compute the kernel matrix using the provided kernel function and parameters.
    # `model_pars[1]` is the amplitude, `model_pars[2:end]` are other kernel parameters.
    Q = model_pars[1] .* kernel_fun(Xt, model_pars[2:end]) 

    return Q
end


"""
    mat32_1D(d, σ)

Helper function to compute the Matern 3/2 correlation for a single 1D distance `d`.

# Arguments
- `d::Float64`: The distance.
- `σ::Float64`: The length-scale parameter.

# Returns
- `correlation::Float64`: The Matern 3/2 correlation value.
"""
function mat32_1D(d::Float64, σ)
    d_scaled = d * sqrt(3.0) / σ # Scale the distance
    exp(-d_scaled) * (1.0 + d_scaled) # Apply Matern 3/2 formula
end

"""
    mat32_cor2(X, pars)

Computes a Matern 3/2 covariance matrix from spatial coordinates `X`,
using an element-wise application of `mat32_1D` to pairwise distances.

# Arguments
- `X::AbstractArray{T}`: Input data matrix (spatial coordinates).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 3/2 covariance matrix.
"""
function mat32_cor2(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = @views pars[1] # Length-scale
    reg = @views pars[2] # Nugget effect
    
    # Compute pairwise Euclidean distances between rows (dims=1).
    dd = pairwise(Euclidean(1e-12),X,dims=1)
    
    # Apply `mat32_1D` function element-wise to the distance matrix `dd`.
    # Add nugget effect.
    mat32_1D.(dd, σ) + reg*I
end

"""
    mat32_cor3(X, pars)

Another optimized implementation to compute a Matern 3/2 covariance matrix
from spatial coordinates `X`.

# Arguments
- `X::AbstractArray{T}`: Input data matrix (spatial coordinates).
- `pars::AbstractVector{T}`: Vector of parameters:
    - `pars[1]`: Length-scale parameter (σ).
    - `pars[2]`: Regularization parameter (nugget).

# Returns
- `K::AbstractMatrix{T}`: The computed Matern 3/2 covariance matrix.
"""
function mat32_cor3(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    # Compute pairwise Euclidean distances between columns (dims=2).
    dd = pairwise(Euclidean(1e-12), X, dims=2)
    σ = @views pars[1] # Length-scale
    reg = @views pars[2] # Nugget effect
    
    # Scale distances in-place: dd = sqrt(3) * dd / σ
    dd .*= sqrt(3.0)/σ   
    
    # Calculate exp(-dd) term
    d2 = exp.(-dd)

    # Calculate (1.0 + dd) term
    dd .+= 1.0  
    
    # Multiply terms: (1.0 + dd) * exp(-dd)
    dd .*= d2
    
    # Add nugget effect and return.
    return reg * I + dd
end
