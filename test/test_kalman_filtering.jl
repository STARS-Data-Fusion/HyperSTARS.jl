"""
Test file for Kalman filtering and smoothing functions
"""

@testset "Kalman Filtering and Smoothing" begin
    
    @testset "Woodbury Filter Basic Functionality" begin
        Random.seed!(12345)  # Set fixed seed for reproducibility
        # Create a simple test case
        ns = 5  # spatial dimension
        nw = 3  # wavelength dimension
        n = ns * nw  # total state dimension
        m = 10  # measurement dimension
        
        # Create simple HSModel
        Hw = ones(nw, nw)
        Hs = ones(m, ns)
        Vw = Diagonal(ones(nw))
        Vs = Diagonal(ones(m))
        Q = Matrix(Diagonal(ones(n)))
        F = UniformScaling(1.0)
        
        model = HyperSTARS.HSModel(Hw, Hs, Vw, Vs, Q, F)
        
        # Prior state estimate
        x_pred = randn(n)
        P_pred = Matrix(Diagonal(ones(n)))
        
        # Measurement
        y_obs = randn(m, nw)
        
        # Update step
        x_new, P_new = woodbury_filter_kr([model], [y_obs], x_pred, P_pred)
        
        @test size(x_new) == (n,)
        @test size(P_new) == (n, n)
        @test all(isfinite.(x_new))
        @test all(isfinite.(P_new))
        @test isapprox(P_new, P_new', rtol=1e-10, atol=1e-10)  # Check symmetry with tolerance
    end
    
    @testset "Woodbury Filter Multiple Instruments" begin
        Random.seed!(12351)  # Set fixed seed for reproducibility
        # Test with multiple instruments
        ns = 5
        nw = 3
        n = ns * nw
        
        # Create two instruments
        Hw1 = ones(nw, nw)
        Hs1 = ones(8, ns)
        Vw1 = Diagonal(ones(nw))
        Vs1 = Diagonal(ones(8))
        
        Hw2 = ones(nw, nw)
        Hs2 = ones(6, ns)
        Vw2 = Diagonal(ones(nw))
        Vs2 = Diagonal(ones(6))
        
        Q = Matrix(Diagonal(ones(n)))
        F = UniformScaling(1.0)
        
        models = [
            HyperSTARS.HSModel(Hw1, Hs1, Vw1, Vs1, Q, F),
            HyperSTARS.HSModel(Hw2, Hs2, Vw2, Vs2, Q, F)
        ]
        
        x_pred = randn(n)
        P_pred = Matrix(Diagonal(ones(n)))
        
        observations = [randn(8, nw), randn(6, nw)]
        
        x_new, P_new = woodbury_filter_kr(models, observations, x_pred, P_pred)
        
        @test size(x_new) == (n,)
        @test size(P_new) == (n, n)
        @test isapprox(P_new, P_new', rtol=1e-10, atol=1e-10)  # Check symmetry with tolerance
    end
    
    @testset "Woodbury Filter Output Covariance Reduction" begin
        Random.seed!(12352)  # Set fixed seed for reproducibility
        # Verify that filtering reduces uncertainty (P_new <= P_pred in some norm)
        ns = 5
        nw = 3
        n = ns * nw
        m = 10
        
        Hw = ones(nw, nw)
        Hs = ones(m, ns)
        Vw = Diagonal(ones(nw))
        Vs = Diagonal(ones(m))
        Q = Matrix(Diagonal(ones(n)))
        F = UniformScaling(1.0)
        
        model = HyperSTARS.HSModel(Hw, Hs, Vw, Vs, Q, F)
        
        x_pred = randn(n)
        P_pred = Matrix(Diagonal(ones(n)))
        
        y_obs = randn(m, nw)
        
        x_new, P_new = woodbury_filter_kr([model], [y_obs], x_pred, P_pred)
        
        # Trace of covariance should typically decrease with measurement
        trace_before = tr(P_pred)
        trace_after = tr(P_new)
        
        # With positive observations, uncertainty should generally decrease
        @test trace_after <= trace_before + 1e-10  # Small tolerance for numerical error
    end
    
    @testset "Smooth Series Basic Functionality" begin
        Random.seed!(12346)  # Set fixed seed for reproducibility
        # Create simple filtering results
        n = 20  # state dimension
        nsteps = 5  # number of time steps
        
        # Create predicted and filtered means/covariances
        predicted_means = randn(n, nsteps)
        predicted_covs = zeros(n, n, nsteps)
        filtering_means = randn(n, nsteps + 1)
        filtering_covs = zeros(n, n, nsteps + 1)
        
        for i in 1:(nsteps + 1)
            filtering_covs[:, :, i] = Matrix(Diagonal(ones(n)))
        end
        
        for i in 1:nsteps
            predicted_covs[:, :, i] = Matrix(Diagonal(ones(n)))
        end
        
        F = UniformScaling(1.0)
        
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)
        
        @test size(smoothed_means) == (n, nsteps)
        @test size(smoothed_covs) == (n, n, nsteps)
        @test all(isfinite.(smoothed_means))
        @test all(isfinite.(smoothed_covs))
    end
    
    @testset "Smooth Series Symmetry" begin
        Random.seed!(12347)  # Set fixed seed for reproducibility
        # Verify that smoothed covariances remain symmetric
        n = 15
        nsteps = 4
        
        predicted_means = randn(n, nsteps)
        predicted_covs = zeros(n, n, nsteps)
        filtering_means = randn(n, nsteps + 1)
        filtering_covs = zeros(n, n, nsteps + 1)
        
        for i in 1:(nsteps + 1)
            S = randn(n, n)
            filtering_covs[:, :, i] = S' * S + 0.1 * I  # Symmetric positive definite
        end
        
        for i in 1:nsteps
            S = randn(n, n)
            predicted_covs[:, :, i] = S' * S + 0.1 * I
        end
        
        F = UniformScaling(1.0)
        
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)
        
        # Check all smoothed covariances are symmetric (with numerical tolerance)
        for i in 1:nsteps
            @test isapprox(smoothed_covs[:, :, i], smoothed_covs[:, :, i]', rtol=1e-9, atol=1e-9)
        end
    end
    
    @testset "Smooth Series With AR(1) Process" begin
        Random.seed!(12348)  # Set fixed seed for reproducibility
        # Test smoothing with autoregressive state transition
        n = 10
        nsteps = 5
        phi = 0.9  # AR(1) parameter
        
        predicted_means = randn(n, nsteps)
        predicted_covs = zeros(n, n, nsteps)
        filtering_means = randn(n, nsteps + 1)
        filtering_covs = zeros(n, n, nsteps + 1)
        
        for i in 1:(nsteps + 1)
            filtering_covs[:, :, i] = Matrix(Diagonal(ones(n)))
        end
        
        for i in 1:nsteps
            predicted_covs[:, :, i] = Matrix(Diagonal(ones(n)))
        end
        
        F = UniformScaling(phi)
        
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)
        
        @test size(smoothed_means) == (n, nsteps)
        @test all(isfinite.(smoothed_means))
    end
    
    @testset "Smooth Series Positive Definiteness" begin
        Random.seed!(12349)  # Set fixed seed for reproducibility
        # Verify smoothed covariances remain positive definite
        n = 12
        nsteps = 4
        
        predicted_means = randn(n, nsteps)
        predicted_covs = zeros(n, n, nsteps)
        filtering_means = randn(n, nsteps + 1)
        filtering_covs = zeros(n, n, nsteps + 1)
        
        for i in 1:(nsteps + 1)
            S = randn(n, n)
            filtering_covs[:, :, i] = S'S + I
        end
        
        for i in 1:nsteps
            S = randn(n, n)
            predicted_covs[:, :, i] = S'S + I
        end
        
        F = UniformScaling(1.0)
        
        smoothed_means, smoothed_covs = smooth_series(F, predicted_means, predicted_covs, filtering_means, filtering_covs)
        
        # Check positive definiteness via eigenvalues
        # With regularization, all eigenvalues should be positive
        for i in 1:nsteps
            eigs = eigvals(smoothed_covs[:, :, i])
            @test all(eigs .> 0)  # All eigenvalues should be positive with regularization
        end
    end
    
    @testset "Filtering Covariance Properties" begin
        Random.seed!(12350)  # Set fixed seed for reproducibility
        # Test that covariance remains symmetric and positive definite
        ns = 8
        nw = 4
        n = ns * nw
        m = 12
        
        # Create model
        Hw = randn(nw, nw)
        Hw = Hw' * Hw  # Make positive definite
        Hs = randn(m, ns)
        Vw = Diagonal(ones(nw))
        Vs = Diagonal(ones(m))
        Q = randn(n, n)
        Q = Q' * Q  # Make positive definite
        F = UniformScaling(1.0)
        
        model = HyperSTARS.HSModel(Hw, Hs, Vw, Vs, Q, F)
        
        x_pred = randn(n)
        P_pred = randn(n, n)
        P_pred = P_pred' * P_pred + I  # Ensure positive definite
        
        y_obs = randn(m, nw)
        
        x_new, P_new = woodbury_filter_kr([model], [y_obs], x_pred, P_pred)
        
        # Check properties
        @test isapprox(P_new, P_new', rtol=1e-7, atol=1e-7)  # Check symmetry with relaxed tolerance
        eigs = eigvals(P_new)
        @test all(eigs .> 0)  # All eigenvalues should be positive with regularization
    end
    
end
