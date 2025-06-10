
using LinearAlgebra
using Statistics
using StatsBase
using Distances
import GaussianRandomFields.CovarianceFunction
import GaussianRandomFields.Matern
import GaussianRandomFields.apply
using KernelFunctions

function kernel_matrix(X::AbstractArray{T}; reg=1e-10, σ=1.0) where {T<:Real}
    Diagonal(reg * ones(size(X)[1])) + exp.(-0.5 * pairwise(SqEuclidean(1e-12), X, dims=1) ./ σ^2)
end

function matern_cor(X::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    ν = pars[3]
    cc = CovarianceFunction(2, Matern(σ, ν))
    Diagonal(reg * ones(size(X)[2])) + apply(cc, X, X)
end

function matern_cor_nonsym(X1::AbstractArray{T}, X2::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    ν = pars[3]
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    kernelmatrix(k,X1,X2, obsdim=2)
end

function matern_cor_fast(X1::AbstractArray{T}, pars=AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    ν = pars[3]
    k = with_lengthscale(MaternKernel(;ν=ν), σ)
    kernelmatrix(k,X1, obsdim=2)
end

function build_GP_var(locs, sigma, phi, nugget=1e-10)
    A = sigma .* kernel_matrix(locs, reg=nugget, σ=phi)
    return A
end

function exp_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    Diagonal(reg * ones(size(X)[2])) + exp.(-pairwise(Euclidean(1e-12), X, dims=2) ./ σ)
end

function mat32_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(3) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd)
end

function mat52_cor(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(5) .* pairwise(Euclidean(1e-12), X, dims=2) ./ σ
    Diagonal(reg * ones(size(X)[2])) + exp.(-dd).*(1.0 .+ dd .+ dd.^2)
end

function exp_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    reg * I + exp.(-dd ./ σ)
end

function mat32_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(3) .* dd ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd)
end

function mat52_corD(dd::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = pars[1]
    reg = pars[2]
    dd = sqrt(5) .* dd ./ σ
    reg * I + exp.(-dd).*(1.0 .+ dd .+ dd.^2)
end

function state_cov(Xtt::AbstractArray{T}, pars::AbstractVector{T}) where T<:Real
    dd = pairwise(Euclidean(1e-12), Xtt, dims=2) 
    phi = maximum([0.00001,median(dd[:])])
    Qst = pars[1] .* exp.(-dd./phi) + UniformScaling(1e-10)
    return Qst
end

function build_gpcov(Xt::AbstractArray{<:Real}, model_pars::AbstractArray{<:Real,2}, kernel_fun::Function)
    Qs = Vector{Matrix{Float64}}(undef,size(model_pars)[1])

    for (i,x) in enumerate(eachrow(model_pars))
        Qs[i] = x[1] .* kernel_fun(Xt, x[2:end]) 
    end

    Q = Matrix(BlockDiagonal(Qs))
    return Q
end

function build_gpcov(Xt::AbstractArray{<:Real}, model_pars::AbstractVector{<:Real}, kernel_fun::Function)

    Q = model_pars[1] .* kernel_fun(Xt, model_pars[2:end]) 

    return Q
end




function mat32_1D(d::Float64, σ)
    d *= sqrt(3.0)/σ
    exp(-d)*(1.0+d)
end

function mat32_cor2(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    σ = @views pars[1]
    reg = @views pars[2]
    dd = pairwise(Euclidean(1e-12),X,dims=1)
    mat32_1D.(dd, σ) + reg*I
end

function mat32_cor3(X::AbstractArray{T}, pars::AbstractVector{T}) where {T<:Real}
    dd = pairwise(Euclidean(1e-12), X, dims=2)
    σ = @views pars[1]
    reg = @views pars[2]
    dd .*= sqrt(3.0)/σ   
    d2 = exp.(-dd)

    dd .+= 1.0  
    dd .*= d2
    return reg * I + dd
end