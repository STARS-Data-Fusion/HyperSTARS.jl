
using Interpolations
using Distances
using LinearAlgebra

function unif_weighted_obs_operator_centroid(sensor::AbstractArray{T}, target::AbstractArray{T}, sensor_res::AbstractVector{T}) where T<:Real
    if sensor == target
        n = size(target,1)
        return sparse(1.0I, n, n)
    else
        d1 = pairwise(Euclidean(1e-12), sensor[:, 1]', target[:, 1]', dims=2) .<= sensor_res[1] / 2
        d2 = pairwise(Euclidean(1e-12), sensor[:, 2]', target[:, 2]', dims=2) .<= sensor_res[2] / 2

        H = d1 .* d2
        return sparse(broadcast(/, H, sum(H, dims=2)))
    end
end

## need to update for corner coordinates
function gauss_weighted_obs_operator(sensor, target, res; scale=1.0, p = 2.0)
    H = exp.(-0.5 * pairwise(SqEuclidean(1e-12), sensor ./ transpose(res ./ scale), target ./ transpose(res ./ scale), dims=1))

    H[H.<exp(-0.5 * p^2)] .= 0

    return sparse(broadcast(/, H, sum(H, dims=2)))
end

function uniform_obs_operator_indices(target, target_cell_size, bau_origin, bau_cell_size, n_dims, bau_sub_inds)
    oi, oj = bau_origin
    c, r = bau_cell_size

    ext_ti = target[:,1]*[1 1] .+ [-0.5+1e-10 0.5-1e-10]*target_cell_size[1] 
    ext_tj = target[:,2]*[1 1] .+ [-0.5+1e-10 0.5-1e-10]*target_cell_size[2] 

    is = find_nearest_ind.(ext_ti, oi, c)
    js = find_nearest_ind.(ext_tj, oj, r)
    p = size(target,1)
    if any(is .> 0) & any(js .> 0) & any(is .<= n_dims[1]) & any(js .<= n_dims[2])
        is[is .< 1] .= 1
        js[js .< 1] .= 1

        is[is .> n_dims[1]] .= n_dims[1]
        js[js .> n_dims[2]] .= n_dims[2]

        irngs = (:).(is[:,1],is[:,2])
        jrngs = (:).(js[:,1],js[:,2])

        II = LinearIndices((1:n_dims[1],1:n_dims[2]))

        cols = [II[irngs[i], jrngs[i]][:] for i in eachindex(irngs)] 
        ns = size.(cols,1)
        rows = inverse_rle(1:p,ns)
        zs = inverse_rle(1 ./ns, ns)

        # H = sparse(rows, vcat(cols...), zs, p, n_dims[1]*n_dims[2])
        H = sparse(rows, vcat(cols...), zs, p, n_dims[1]*n_dims[2])
    
        return H[:,bau_sub_inds]
    else
        return spzeros(p, length(bau_sub_inds))
    end
end


function rsr_conv_matrix(rsr::AbstractArray{<:Real}, inst_waves::AbstractVector{<:Real}, target_waves::AbstractVector{<:Real})

    dd = Distances.pairwise(SqEuclidean(1e-12), inst_waves, target_waves)

    s2p = (rsr ./ (2 * sqrt(2 * log(2)))).^2
    K = exp.(-dd ./ s2p)
    Ks = K ./sum(K, dims=2)

    return sparse(Ks)
end

function rsr_conv_matrix(rsr::Dict, inst_waves::AbstractVector{<:Real}, target_waves::AbstractVector{<:Real})

    knts = rsr[:w]
    Ki = rsr[:rsr]

    if target_waves == knts
        Ks = sparse(Ki ./sum(Ki, dims=2))
    else
        K = zeros(size(Ki)[1],size(target_waves)[1])
        for i in 1:size(Ki)[1]
            itp = extrapolate(interpolate((knts,), Ki[i,:], Gridded(Linear())),0)
            K[i,:] = itp(target_waves)
        end
        Ks = sparse(K ./sum(K, dims=2))

    end
  
    return Ks
end

