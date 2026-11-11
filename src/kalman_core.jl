# kalman_core.jl
#
# Kalman filter pieces shared by the scene and swath workflows.
# Nothing in here depends on InstrumentData or SwathData.
#
# Indexing convention for the mean/precision history:
#   means[:, 1]   / prec[:, 1]    -> prior
#   means[:, t+1] / prec[:, t+1]  -> filtered state at times[t]

"""
    _process_cov!(Qf, Q, means, prec, t, nf, p, model_pars,
                  state_in_cov, cov_wt, tscov_pars, tscov_lag)

Build the process covariance for loop step `t`, in place.

Without `state_in_cov`, `Qf = Q`. With it, `Qf = cov_wt*Q + (1-cov_wt)*Qs`, where
`Qs` is estimated from the precision-weighted filtered means of the last
`tscov_lag` steps. `tscov_lag = nothing` uses the full history (the v1 scene
behavior).
"""
function _process_cov!(Qf, Q, means, prec, t, nf, p, model_pars,
                       state_in_cov, cov_wt, tscov_pars, tscov_lag)
    Qf .= Q
    if state_in_cov
        h = isnothing(tscov_lag) ? t-1 : min(t-1, tscov_lag)
        Qf .*= cov_wt

        Xtt = reshape(means[:,t-h:t], (nf,p,h+1))[:,:,:]
        Wt  = @views reshape(prec[:,t-h:t], (nf,p,h+1))
        Xtt .*= Wt
        Xtt ./= sum(Wt, dims=3)

        for (i,x) in enumerate(tscov_pars)
            ids = ((i-1)*nf+1):i*nf
            x2 = [model_pars[i,1], x]
            @views Qf[ids,ids] .+= state_cov(Xtt[:,i,:]', x2) .* (1.0 .- cov_wt)
        end
    end
    return Qf
end

"""
    _predict!(x_pred, P_pred, FPpred, F, x, P, Qf)

Prediction step, in place: `x_pred = F x`, `P_pred = F P F' + Qf` (symmetrized).
`FPpred` is an n×n workspace. `F` may be a `UniformScaling`, `Diagonal`, or
sparse matrix.
"""
function _predict!(x_pred, P_pred, FPpred, F, x, P, Qf)
    mul!(x_pred, F, x)
    mul!(FPpred, F, P)          # FP = F*P
    mul!(P_pred, F, FPpred')    # F*(FP)' = F*P*F' since P symmetric
    P_pred .+= Qf
    P_pred .= (P_pred .+ P_pred') ./ 2
    return x_pred, P_pred
end

"""
    _update!(x, P, Ms, ys, x_pred, P_pred; reg = 1e-10)

Measurement update. Writes the filtered mean and covariance into `x` and `P`.

Uses the Woodbury (information) form when there are more observations than
states, the standard Kalman form otherwise. With no observations, the filtered
state is the prediction, with `reg` added to the covariance diagonal
(`reg = 0.0` reproduces v1 scene behavior).
"""
function _update!(x, P, Ms, ys, x_pred, P_pred; reg::Real = 1e-10)
    Nobs = isempty(ys) ? 0 : sum(size(M.Hs,1)*size(M.Hw,1) for M in Ms)
    if Nobs == 0
        x .= x_pred
        P .= P_pred
        if !iszero(reg)
            for i in axes(P,1)
                P[i,i] += reg
            end
        end
    else
        x_new, P_new = Nobs > size(P_pred,1) ?
            information_filter_kr(Ms, ys, x_pred, P_pred) :
            kalman_filter_kr(Ms, ys, x_pred, P_pred)
        x .= x_new
        P .= P_new
    end
    return x, P
end

"""
    _backproject!(fused_image, fused_sd_image, ti, x, Σ, B, spectral_mean, nf, p, nw, rows)

Back-project latent state `x` (covariance `Σ`) to the spectral domain and write
output slot `ti`. `rows` selects the target cells kept from the window
(`bau_kp` for swath, `1:nbau` for scene).
"""
function _backproject!(fused_image, fused_sd_image, ti, x, Σ,
                       B, spectral_mean, nf, p, nw, rows)
    fused_image[:,:,ti]    = @views reshape(x,(nf,p))[rows,:] * B' .+ spectral_mean'
    fused_sd_image[:,:,ti] = sqrt.(diag_cov_XB(Σ, B', nf, p, nw)[rows,:])
    return nothing
end

# --- Backward (RTS) passes ----------------------------------------------------
#
# Storage convention from the smoother's forward pass:
#   filtering_means[:, t+1] = filtered mean       at times[t]
#   filt_covs[:,:,t-st+1]   = filtered covariance  at times[t],  t ≥ st
#   pred_covs[:,:,t-st]     = predicted covariance at times[t],  t ≥ st+1
#
# Both passes call emit(t, x_smooth, Σ) for t = nsteps down to st. `Σ` may be a
# buffer that is overwritten on the next step, so `emit` must use it immediately.

"""
    _rts_means!(emit, F, filtering_means, filt_covs, pred_covs, st, nsteps)

Backward RTS pass for the means only. `Σ` passed to `emit` is the filtered
covariance at that time. Overwrites `pred_covs` (factored in place).
Cost per step ≈ n³/3.
"""
function _rts_means!(emit, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    n  = size(filtering_means, 1)
    nk = nsteps - st + 1

    smoothed_means = zeros(n, nk)
    @views smoothed_means[:,nk] .= filtering_means[:,nsteps+1]   # last smoothed = last filtered
    emit(nsteps, @view(smoothed_means[:,nk]), @view(filt_covs[:,:,nk]))

    x_pred = zeros(n)
    dvec   = zeros(n)
    for t in nsteps:-1:st+1
        k = t - st + 1

        # x_{t|t-1} = F x_{t-1|t-1}
        mul!(x_pred, F, @view(filtering_means[:,t]))

        # x_{t-1|T} = x_{t-1|t-1} + P_{t-1|t-1} F' P_{t|t-1}⁻¹ (x_{t|T} - x_{t|t-1})
        @views dvec .= smoothed_means[:,k] .- x_pred
        Ppk = @view pred_covs[:,:,k-1]
        LAPACK.potrf!('U', Ppk)          # factor in place; each P_{t|t-1} is used once
        LAPACK.potrs!('U', Ppk, dvec)
        w = F' * dvec
        @views mul!(smoothed_means[:,k-1], filt_covs[:,:,k-1], w)
        @views smoothed_means[:,k-1] .+= filtering_means[:,t]

        emit(t-1, @view(smoothed_means[:,k-1]), @view(filt_covs[:,:,k-1]))
    end
    return smoothed_means
end

"""
    _rts_full!(emit, F, filtering_means, filt_covs, pred_covs, st, nsteps)

Full backward RTS pass: smoothed means and covariances. `Σ` passed to `emit` is
the smoothed covariance at that time (a buffer, overwritten on the next step).
Only the current smoothed covariance is held in memory. Does not modify
`pred_covs`. Cost per step ≈ 7n³.
"""
function _rts_full!(emit, F, filtering_means, filt_covs, pred_covs, st, nsteps)
    n  = size(filtering_means, 1)
    nk = nsteps - st + 1

    smoothed_means = zeros(n, nk)
    @views smoothed_means[:,nk] .= filtering_means[:,nsteps+1]
    P_s = filt_covs[:,:,nk]                      # current smoothed covariance (copy)
    emit(nsteps, @view(smoothed_means[:,nk]), P_s)

    x_pred = zeros(n)
    CC = zeros(n,n)
    Cg = zeros(n,n)    # smoother gain
    for t in nsteps:-1:st+1
        k = t - st + 1

        mul!(x_pred, F, @view(filtering_means[:,t]))

        # gain  C = P_{t-1|t-1} F' P_{t|t-1}⁻¹
        CC .= @view pred_covs[:,:,k-1]
        LAPACK.potrf!('U', CC)
        LAPACK.potri!('U', CC)
        BLAS.symm!('R', 'U', 1., CC, filt_covs[:,:,k-1]*F', 0., Cg)

        # x_{t-1|T} = x_{t-1|t-1} + C (x_{t|T} - x_{t|t-1})
        x_smooth = filtering_means[:,t] .+ Cg * (smoothed_means[:,k] .- x_pred)

        # P_{t-1|T} = P_{t-1|t-1} + C (P_{t|T} - P_{t|t-1}) C'
        CC .= P_s .- @view(pred_covs[:,:,k-1])
        D = BLAS.symm('R', 'U', CC, Cg)
        CC .= @view filt_covs[:,:,k-1]
        BLAS.gemm!('N', 'T', 1., D, Cg, 1., CC)

        P_s .= (CC .+ CC') ./ 2
        reg_amount = maximum(abs, diag(P_s)) * 1e-3 + 1e-8    # as in v1 smooth_series
        for j in axes(P_s,1)
            P_s[j,j] += reg_amount
        end

        smoothed_means[:,k-1] = x_smooth
        emit(t-1, @view(smoothed_means[:,k-1]), P_s)
    end
    return smoothed_means
end