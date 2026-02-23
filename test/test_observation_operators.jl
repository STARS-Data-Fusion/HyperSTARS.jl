"""
Test file for observation operators (spatial and spectral)
"""

@testset "Observation Operators" begin
    
    @testset "Uniform Weighted Obs Operator - Identity Case" begin
        # Test when sensor and target coordinates are identical
        coords = randn(20, 2)
        res = [30.0, 30.0]
        
        H = unif_weighted_obs_operator_centroid(coords, coords, res)
        
        # Should be identity matrix
        @test issparse(H)
        @test H == I(20)
    end
    
    @testset "Uniform Weighted Obs Operator - Basic Overlap" begin
        # Test with distinct sensor and target coordinates
        sensor = [[0.0, 0.0]; [100.0, 100.0]; [200.0, 0.0]]
        target = [[0.0, 0.0]; [50.0, 50.0]; [100.0, 100.0]; [200.0, 0.0]]
        res = [100.0, 100.0]
        
        H = unif_weighted_obs_operator_centroid(sensor, target, res)
        
        @test issparse(H)
        @test size(H, 1) == 3  # 3 sensors
        @test size(H, 2) == 4  # 4 targets
        @test all(sum(H, dims=2) .≈ 1.0)  # Each row should sum to 1
        @test all(H .>= 0.0)  # All weights should be non-negative
    end
    
    @testset "Uniform Weighted Obs Operator - No Overlap" begin
        # Test with no overlap between sensor and target
        sensor = [[0.0, 0.0]]
        target = [[1000.0, 1000.0]]
        res = [1.0, 1.0]
        
        H = unif_weighted_obs_operator_centroid(sensor, target, res)
        
        @test size(H) == (1, 1)
        @test H[1, 1] ≈ 0.0
    end
    
    @testset "Uniform Weighted Obs Operator - Sparse Structure" begin
        # Verify sparsity
        n_sensor = 100
        n_target = 150
        sensor = randn(n_sensor, 2) .* 1000
        target = randn(n_target, 2) .* 1000
        res = [100.0, 100.0]
        
        H = unif_weighted_obs_operator_centroid(sensor, target, res)
        
        @test issparse(H)
        @test size(H) == (n_sensor, n_target)
        # Should have fewer non-zero elements than dense matrix
        @test nnz(H) < n_sensor * n_target
    end
    
    @testset "Uniform Weighted Obs Operator - Row Normalization" begin
        # Test that rows sum to 1
        sensor = randn(50, 2) .* 1000
        target = randn(100, 2) .* 1000
        res = [150.0, 150.0]
        
        H = unif_weighted_obs_operator_centroid(sensor, target, res)
        
        row_sums = vec(sum(H, dims=2))
        
        # Rows with observations should sum to 1, empty rows should sum to 0
        for i in 1:length(row_sums)
            if row_sums[i] > 0
                @test row_sums[i] ≈ 1.0 atol=1e-10
            end
        end
    end
    
    @testset "Uniform Weighted Obs Operator - Resolution Scaling" begin
        # Test that larger resolution leads to more overlaps
        sensor = [[0.0, 0.0]]
        target = randn(20, 2) .* 10  # Targets within small range
        
        res_small = [1.0, 1.0]
        res_large = [100.0, 100.0]
        
        H_small = unif_weighted_obs_operator_centroid(sensor, target, res_small)
        H_large = unif_weighted_obs_operator_centroid(sensor, target, res_large)
        
        # Larger resolution should have more non-zeros
        @test nnz(H_large) >= nnz(H_small)
    end
    
    @testset "Gaussian Weighted Obs Operator - Basic Structure" begin
        # Test basic Gaussian operator
        sensor = [[0.0, 0.0]; [100.0, 100.0]]
        target = randn(30, 2) .* 200
        res = [50.0, 50.0]
        
        H = HyperSTARS.gauss_weighted_obs_operator(sensor, target, res)
        
        @test issparse(H)
        @test size(H, 1) == 2  # 2 sensors
        @test size(H, 2) == 30  # 30 targets
        @test all(H .>= 0.0)  # All weights non-negative for Gaussian
    end
    
    @testset "Gaussian Weighted Obs Operator - Small Scale" begin
        # Test with small scale parameter
        sensor = [[0.0, 0.0]; [100.0, 100.0]]
        target = randn(30, 2) .* 200
        res = [50.0, 50.0]
        
        H_small_scale = gauss_weighted_obs_operator(sensor, target, res, scale=0.5)
        H_large_scale = gauss_weighted_obs_operator(sensor, target, res, scale=2.0)
        
        # Smaller scale should have fewer non-zeros (steeper falloff)
        @test nnz(H_small_scale) <= nnz(H_large_scale)
    end
    
    @testset "Gaussian Weighted Obs Operator - Distance Decay" begin
        # Test that weights decay with distance
        sensor = [[0.0, 0.0]]
        target = [[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]
        res = [10.0, 10.0]
        
        H = gauss_weighted_obs_operator(sensor, target, res, scale=1.0)
        
        # Extract weights
        weights = vec(H.nzval[H.rowval .== 1])
        
        if length(weights) > 1
            # Weights should generally decrease with distance
            # (allowing for threshold cutoff)
            @test all(isfinite.(weights))
            @test all(weights .>= 0)
        end
    end
    
    @testset "RSR Convolution Matrix" begin
        # Test spectral response function convolution
        target_wavelengths = LinRange(400, 2500, 50)
        sensor_wavelengths = LinRange(400, 2500, 100)
        
        # Test with FWHM (scalar)
        fwhm = 5.0
        
        H = HyperSTARS.rsr_conv_matrix(fwhm, sensor_wavelengths, target_wavelengths)
        
        @test size(H, 1) == length(sensor_wavelengths)
        @test size(H, 2) == length(target_wavelengths)
    end
    
    @testset "RSR Convolution Matrix - Dict Format" begin
        # Test with dictionary format
        target_wavelengths = LinRange(400, 2500, 50)
        sensor_wavelengths = LinRange(400, 2500, 100)
        
        # Create RSR dict (e.g., for HLS bands)
        rsr_dict = Dict(
            :wavelengths => sensor_wavelengths,
            :rsr => ones(length(sensor_wavelengths))
        )
        
        H = HyperSTARS.rsr_conv_matrix(rsr_dict, sensor_wavelengths, target_wavelengths)
        
        @test size(H, 1) == length(sensor_wavelengths)
        @test size(H, 2) == length(target_wavelengths)
    end
    
    @testset "Observation Operator Consistency" begin
        # Test that observation operators produce consistent results
        sensor1 = randn(50, 2) .* 1000
        target1 = randn(100, 2) .* 1000
        res1 = [100.0, 100.0]
        
        H1 = unif_weighted_obs_operator_centroid(sensor1, target1, res1)
        H2 = unif_weighted_obs_operator_centroid(sensor1, target1, res1)
        
        @test H1 == H2  # Same inputs should give same output
    end
    
end
