using LinearAlgebra, Random
Random.seed!(1)

N = 600          # state dimension (nf * p)
m = 900          # number of observations (> N, so the Woodbury branch applies)

# predicted state and covariance (symmetric positive definite)
A      = randn(N, N)
P_pred = Matrix(Symmetric(A * A' / N + 0.5I))
x_pred = randn(N)

# observation operator, error covariance, observations
H = randn(m, N) ./ sqrt(N)
V = Diagonal(0.05 .+ rand(m))
y = H * randn(N) .+ sqrt.(diag(V)) .* randn(m)

# the two accumulated terms the update lines use
HViH = H' * (V \ H)                    # H' V⁻¹ H
HVie = H' * (V \ (y - H * x_pred))     # H' V⁻¹ (y - H x_pred)

# outputs the update lines write into
x_new = zeros(N)
x_new2 = zeros(N)
P_new = zeros(N, N)
P_new2 = zeros(N, N)
F     = zeros(N, N)

### old code
t_old = @elapsed begin
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
end

### new code
t_new = @elapsed begin
    S = P_pred[:,:]
    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    LinearAlgebra.copytri!(S, 'U')

    S .+= HViH
    LAPACK.potrf!('U', S)
    LAPACK.potri!('U', S)
    LinearAlgebra.copytri!(S, 'U')

    x_new2 .= x_pred
    mul!(x_new2, S, HVie, 1.0, 1.0)

    P_new2 .= S
    P_new2 .= (P_new2 + P_new2') / 2
end

using LinearAlgebra, Random, BenchmarkTools
BLAS.set_num_threads(1)

function update_old!(x_new, P_new, F, P_pred, x_pred, HViH, HVie)
    S = P_pred[:,:]
    LAPACK.potrf!('U', S); LAPACK.potri!('U', S); LinearAlgebra.copytri!(S,'U')
    S .+= HViH
    LAPACK.potrf!('U', S); LAPACK.potri!('U', S)
    HtRHIi = BLAS.symm('R', 'U', S, HViH)
    F .= P_pred
    mul!(F, P_pred, HtRHIi, -1.0, 1.0)
    x_new .= x_pred
    mul!(x_new, F, HVie, 1.0, 1.0)
    F .= F * HViH
    P_new .= P_pred
    mul!(P_new, F, P_pred, -1.0, 1.0)
    return x_new, P_new
end

function update_new!(x_new, P_new, P_pred, x_pred, HViH, HVie)
    S = P_pred[:,:]
    LAPACK.potrf!('U', S); LAPACK.potri!('U', S); LinearAlgebra.copytri!(S,'U')
    S .+= HViH
    LAPACK.potrf!('U', S); LAPACK.potri!('U', S); LinearAlgebra.copytri!(S,'U')
    x_new .= x_pred
    mul!(x_new, S, HVie, 1.0, 1.0)
    P_new .= S
    return x_new, P_new
end

# --- synthetic inputs (same as before; choose a realistic size)
Random.seed!(1)
N, m = 1800, 2500
A = randn(N, N); P_pred = Matrix(Symmetric(A*A'/N + 0.5I)); x_pred = randn(N)
H = randn(m, N) ./ sqrt(N); V = Diagonal(0.05 .+ rand(m))
y = H*randn(N) .+ sqrt.(diag(V)) .* randn(m)
HViH = H' * (V \ H); HVie = H' * (V \ (y - H*x_pred))

# --- correctness first
xo, Po = update_old!(zeros(N), zeros(N,N), zeros(N,N), P_pred, x_pred, HViH, HVie)
xn, Pn = update_new!(zeros(N), zeros(N,N), P_pred, x_pred, HViH, HVie)
@show maximum(abs, xo .- xn) / maximum(abs, xn)
@show maximum(abs, Po .- Pn) / maximum(abs, Pn)

# --- timing: fresh output buffers each sample, one evaluation per sample
b_old = @benchmark update_old!(xn, Pn, F, $P_pred, $x_pred, $HViH, $HVie) setup=(xn=zeros($N); Pn=zeros($N,$N); F=zeros($N,$N)) evals=1 seconds=60
b_new = @benchmark update_new!(xn, Pn, $P_pred, $x_pred, $HViH, $HVie) setup=(xn=zeros($N); Pn=zeros($N,$N)) evals=1 seconds=60

display(b_old); display(b_new)
println("speed-up (median): ", round(median(b_old).time / median(b_new).time, digits=2), "x")