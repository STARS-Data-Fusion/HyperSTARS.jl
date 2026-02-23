"""
Test file for Gaussian Process utility functions and covariance kernels
"""

@testset "Gaussian Process Utilities" begin
    
    @testset "Exponential Correlation - Basic" begin
        # Test basic exponential correlation
        distances = [0.0, 1.0, 2.0, 5.0, 10.0]
        length_scale = 1.0
        
        C = exp_corD(distances, [length_scale])
        
        @test size(C) == (5, 5)
        @test issymmetric(C)
        @test all(C .>= 0.0)  # Covariance should be non-negative
        @test C[1, 1] ≈ 1.0  # At distance 0, correlation should be 1
    end
    
    @testset "Exponential Correlation - Distance Decay" begin
        # Test that correlation decays with distance
        D = [0.0 2.0 4.0;
             2.0 0.0 2.0;
             4.0 2.0 0.0]
        length_scale = 1.0
        
        C = exp_corD(D, [length_scale])
        
        # Diagonal should be 1
        @test diag(C) ≈ ones(3)
        # Off-diagonal should decrease with distance
        @test C[1, 2] > C[1, 3]
    end
    
    @testset "Matern 3/2 Correlation" begin
        # Test Matern 3/2 kernel
        distances = [0.0, 1.0, 2.0, 5.0]
        params = [1.0, 0.1]  # length_scale, ridge (regularization)
        
        C = mat32_corD(distances, params)
        
        @test size(C) == (4, 4)
        @test issymmetric(C)
        @test all(C .>= 0.0)
        @test C[1, 1] ≈ 1.0 atol=0.15  # Diagonal with ridge may not be exactly 1
    end
    
    @testset "Matern 5/2 Correlation" begin
        # Test Matern 5/2 kernel
        distances = [0.0, 1.0, 2.0, 5.0]
        params = [1.0, 0.1]
        
        C = mat52_corD(distances, params)
        
        @test size(C) == (4, 4)
        @test issymmetric(C)
        @test all(C .>= 0.0)
    end
    
    @testset "Kernel Positivity" begin
        # Test that covariance matrices are positive semi-definite
        D = [0.0 1.5 3.0;
             1.5 0.0 2.0;
             3.0 2.0 0.0]
        
        for kernel_func in [exp_corD, mat32_corD, mat52_corD]
            C = kernel_func(D, [1.0, 0.01])
            eigvals = eigvals(C)
            
            # All eigenvalues should be non-negative
            @test all(eigvals .>= -1e-10)
        end
    end
    
    @testset "Covariance Symmetry" begin
        # Test that all kernels produce symmetric matrices
        distances = randn(5, 5)
        distances = (distances + distances') / 2  # Ensure symmetry
        distances = abs.(distances)  # Ensure non-negative
        
        for kernel_func in [exp_corD, mat32_corD, mat52_corD]
            C = kernel_func(distances, [1.0, 0.01])
            
            @test issymmetric(C)
        end
    end
    
    @testset "Length Scale Parameter Effect" begin
        # Test that length scale affects correlation decay
        distances = [0.0, 1.0, 2.0, 3.0, 4.0]
        length_scale_small = 0.5
        length_scale_large = 2.0
        
        C_small = exp_corD(distances, [length_scale_small])
        C_large = exp_corD(distances, [length_scale_large])
        
        # Larger length scale should give higher correlations at same distance
        for i in 2:5
            if i > 1
                @test C_large[1, i] > C_small[1, i] || C_large[1, i] ≈ C_small[1, i]
            end
        end
    end
    
    @testset "nanmean Function" begin
        # Test nanmean with NaN values
        data = [1.0, 2.0, NaN, 4.0, NaN, 5.0]
        
        result = nanmean(data)
        
        @test result ≈ 3.0  # mean of [1, 2, 4, 5]
        @test !isnan(result)
    end
    
    @testset "nanmean 2D Function" begin
        # Test nanmean on 2D array
        data = [1.0 2.0 3.0;
                4.0 NaN 6.0;
                7.0 8.0 NaN]
        
        result = nanmean(data, 1)
        
        @test size(result) == (1, 3)
        @test result[1, 1] ≈ (1.0 + 4.0 + 7.0) / 3
        @test result[1, 2] ≈ (2.0 + 8.0) / 2
    end
    
    @testset "nanvar Function" begin
        # Test nanvar with NaN values
        data = [1.0, 2.0, NaN, 4.0, NaN, 5.0]
        
        result = nanvar(data)
        
        @test !isnan(result)
        @test result >= 0  # Variance should be non-negative
    end
    
    @testset "State Covariance Function" begin
        # Test state_cov for adaptive process noise
        X = randn(3, 10)  # 3 latent states, 10 time steps
        params = [0.5, 1.0]
        
        C = state_cov(X, params)
        
        @test size(C) == (10, 10)
        @test issymmetric(C)
        @test all(C .>= 0.0)
    end
    
    @testset "Kernel Function with Zero Distances" begin
        # Test numerical stability at zero distance
        distances = [0.0, 1e-15, 1e-10, 0.1]
        
        for kernel_func in [exp_corD, mat32_corD, mat52_corD]
            C = kernel_func(distances, [1.0, 0.01])
            
            @test all(isfinite.(C))
            @test C[1, 1] ≈ 1.0 atol=0.2  # Should be close to 1 at origin
        end
    end
    
    @testset "Kernel Distance Matrix Format" begin
        # Test that kernels work with pre-computed distance matrices
        points = randn(10, 2)
        D = pairwise(Euclidean(), points, dims=1)
        
        # Test each kernel function
        for (kernel_func, params) in [
            (exp_corD, [1.0, 0.01]),
            (mat32_corD, [1.0, 0.01]),
            (mat52_corD, [1.0, 0.01])
        ]
            C = kernel_func(D, params)
            
            @test size(C) == (10, 10)
            @test issymmetric(C)
        end
    end
    
end
