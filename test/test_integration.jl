"""
Integration tests for HyperSTARS.jl - testing combined functionality
"""

@testset "Integration Tests" begin
    
    @testset "Small Scene Fusion - Filtering Only" begin
        # Test a simple complete fusion workflow with filtering (no smoothing)
        
        # Set up dimensions
        n_spatial_target = 10
        n_spatial_samples = 25
        n_wavelengths = 8
        n_time = 3
        n_latent = 2  # PCA components
        
        # Create target geometry
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        # Create instrument data
        sensor_coords = randn(n_spatial_samples, 2) .* 150 .+ 500000
        inst_data_array = randn(n_spatial_samples, n_wavelengths, n_time)
        
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_wavelengths),
            0.1 * ones(n_wavelengths),
            [30.0, 30.0],
            1:n_time,
            sensor_coords,
            LinRange(400, 2500, n_wavelengths),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial_target, n_spatial_target],
            0,
            1:n_time,
            LinRange(400, 2500, n_wavelengths),
            5.0
        )
        
        target_geo = inst_geo
        
        # Create prior statistics
        prior_mean = randn(n_spatial_target * n_spatial_target, n_latent)
        prior_var = ones(n_spatial_target * n_spatial_target, n_latent)
        
        # Create basis and spectral mean
        B = randn(n_wavelengths, n_latent)
        spectral_mean = randn(n_wavelengths)
        
        # Create model parameters
        model_pars = zeros(1, 1, n_latent, 3)  # (1x1 window, nlatent, 3 params for Matern)
        for i in 1:n_latent
            model_pars[1, 1, i, :] = [0.1, 0.01, 1.5]  # variance, nugget, smoothness
        end
        
        # Create data dictionary for fusion
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = randn(n_spatial_target * n_spatial_target, 2) .* 300 .+ 500000
        d[:kp_ij] = 1:n_spatial_target * n_spatial_target
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        # Run fusion
        kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
            d, 
            LinRange(400, 2500, n_wavelengths),
            spectral_mean,
            B,
            target_times = 1:n_time,
            smooth = false,
            spatial_mod = mat32_corD,
            state_in_cov = false
        )
        
        @test size(fused_image, 3) == n_latent
        @test size(fused_image, 2) >= 1
        @test size(fused_sd) == size(fused_image)
        @test all(isfinite.(fused_image))
        @test all(isfinite.(fused_sd))
        @test all(fused_sd .>= 0)
    end
    
    @testset "Small Scene Fusion - With Smoothing" begin
        # Test fusion with Kalman smoothing
        
        n_spatial_target = 8
        n_spatial_samples = 20
        n_wavelengths = 6
        n_time = 4
        n_latent = 2
        
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        sensor_coords = randn(n_spatial_samples, 2) .* 120 .+ 500000
        inst_data_array = randn(n_spatial_samples, n_wavelengths, n_time)
        
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_wavelengths),
            0.1 * ones(n_wavelengths),
            [30.0, 30.0],
            1:n_time,
            sensor_coords,
            LinRange(400, 2500, n_wavelengths),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial_target, n_spatial_target],
            0,
            1:n_time,
            LinRange(400, 2500, n_wavelengths),
            5.0
        )
        
        target_geo = inst_geo
        
        prior_mean = randn(n_spatial_target * n_spatial_target, n_latent)
        prior_var = ones(n_spatial_target * n_spatial_target, n_latent)
        
        B = randn(n_wavelengths, n_latent)
        spectral_mean = randn(n_wavelengths)
        
        model_pars = zeros(1, 1, n_latent, 3)
        for i in 1:n_latent
            model_pars[1, 1, i, :] = [0.1, 0.01, 1.5]
        end
        
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = randn(n_spatial_target * n_spatial_target, 2) .* 250 .+ 500000
        d[:kp_ij] = 1:n_spatial_target * n_spatial_target
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        # Run with smoothing
        kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
            d,
            LinRange(400, 2500, n_wavelengths),
            spectral_mean,
            B,
            target_times = 1:n_time,
            smooth = true,
            spatial_mod = mat32_corD,
            state_in_cov = false
        )
        
        @test size(fused_image, 3) == n_latent
        @test all(isfinite.(fused_image))
        @test all(isfinite.(fused_sd))
    end
    
    @testset "Filter Dimensions Match Expectations" begin
        # Test that filtering output dimensions are correct
        
        n_spatial = 5
        n_spatial_samples = 12
        n_waves = 4
        n_time = 2
        n_latent = 1
        
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        inst_data_array = randn(n_spatial_samples, n_waves, n_time)
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial_samples, 2) .* 100 .+ 500000,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial, n_spatial],
            0,
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        target_geo = inst_geo
        prior_mean = randn(n_spatial * n_spatial, n_latent)
        prior_var = ones(n_spatial * n_spatial, n_latent)
        B = randn(n_waves, n_latent)
        spectral_mean = randn(n_waves)
        
        model_pars = zeros(1, 1, n_latent, 3)
        model_pars[1, 1, 1, :] = [0.1, 0.01, 1.5]
        
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = randn(n_spatial * n_spatial, 2) .* 150 .+ 500000
        d[:kp_ij] = 1:n_spatial * n_spatial
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
            d,
            LinRange(400, 2500, n_waves),
            spectral_mean,
            B,
            target_times = 1:n_time,
            smooth = false
        )
        
        @test size(fused_image, 1) == length(kp_ij)
        @test size(fused_image, 2) == n_latent
        @test size(fused_image, 3) == n_time
        @test size(fused_sd) == size(fused_image)
    end
    
    @testset "Observation Operator Integration" begin
        # Test that observation operators work correctly in fusion context
        
        n_spatial = 6
        n_spatial_samples = 18
        n_waves = 5
        n_time = 2
        n_latent = 1
        
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        # Create aligned coordinates for testing observation operator
        target_coords_base = randn(n_spatial * n_spatial, 2) .* 100 .+ 500000
        sensor_coords = target_coords_base[1:n_spatial_samples, :] .+ randn(n_spatial_samples, 2) .* 5
        
        inst_data_array = randn(n_spatial_samples, n_waves, n_time)
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            sensor_coords,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial, n_spatial],
            0,
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        target_geo = inst_geo
        prior_mean = randn(n_spatial * n_spatial, n_latent)
        prior_var = ones(n_spatial * n_spatial, n_latent)
        B = randn(n_waves, n_latent)
        spectral_mean = randn(n_waves)
        
        model_pars = zeros(1, 1, n_latent, 3)
        model_pars[1, 1, 1, :] = [0.1, 0.01, 1.5]
        
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = target_coords_base
        d[:kp_ij] = 1:n_spatial * n_spatial
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
            d,
            LinRange(400, 2500, n_waves),
            spectral_mean,
            B,
            target_times = 1:n_time,
            smooth = false,
            obs_operator = unif_weighted_obs_operator_centroid
        )
        
        @test all(isfinite.(fused_image))
        @test size(fused_image, 3) == n_latent
    end
    
    @testset "Multiple Time Steps" begin
        # Test fusion with multiple time steps
        n_spatial = 4
        n_spatial_samples = 8
        n_waves = 3
        n_time = 5
        n_latent = 1
        
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        inst_data_array = randn(n_spatial_samples, n_waves, n_time)
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial_samples, 2) .* 80 .+ 500000,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial, n_spatial],
            0,
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        target_geo = inst_geo
        prior_mean = randn(n_spatial * n_spatial, n_latent)
        prior_var = ones(n_spatial * n_spatial, n_latent)
        B = randn(n_waves, n_latent)
        spectral_mean = randn(n_waves)
        
        model_pars = zeros(1, 1, n_latent, 3)
        model_pars[1, 1, 1, :] = [0.1, 0.01, 1.5]
        
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = randn(n_spatial * n_spatial, 2) .* 120 .+ 500000
        d[:kp_ij] = 1:n_spatial * n_spatial
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        # Test with different target times
        for target_times in [1:n_time, [1, 3, 5], [n_time]]
            kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
                d,
                LinRange(400, 2500, n_waves),
                spectral_mean,
                B,
                target_times = target_times,
                smooth = false
            )
            
            @test size(fused_image, 3) == length(target_times)
            @test size(fused_sd, 3) == length(target_times)
        end
    end
    
    @testset "Numerical Stability" begin
        # Test that outputs remain numerically stable
        n_spatial = 5
        n_spatial_samples = 10
        n_waves = 4
        n_time = 2
        n_latent = 2
        
        target_origin = [500000.0, 4000000.0]
        target_csize = [30.0, -30.0]
        
        # Use data with extreme values
        inst_data_array = randn(n_spatial_samples, n_waves, n_time) .* 1e6 .+ 1e8
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_waves) .* 1e5,
            (1e4 * ones(n_waves)),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial_samples, 2) .* 1000 .+ 500000,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            target_origin,
            target_csize,
            [n_spatial, n_spatial],
            0,
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        target_geo = inst_geo
        prior_mean = randn(n_spatial * n_spatial, n_latent) .* 1e8 .+ 1e8
        prior_var = (1e-3 * ones(n_spatial * n_spatial, n_latent))
        B = randn(n_waves, n_latent)
        spectral_mean = randn(n_waves) .* 1e6
        
        model_pars = zeros(1, 1, n_latent, 3)
        for i in 1:n_latent
            model_pars[1, 1, i, :] = [1e12, 1e10, 1.5]
        end
        
        d = Dict()
        d[:measurements] = [inst_data]
        d[:target_coords] = randn(n_spatial * n_spatial, 2) .* 1000 .+ 500000
        d[:kp_ij] = 1:n_spatial * n_spatial
        d[:prior_mean] = prior_mean[:]
        d[:prior_var] = prior_var[:]
        d[:model_pars] = model_pars[1, 1, :, :]
        
        kp_ij, fused_image, fused_sd = hyperSTARS_fusion_kr_dict(
            d,
            LinRange(400, 2500, n_waves),
            spectral_mean,
            B,
            target_times = 1:n_time,
            smooth = false,
            state_in_cov = false
        )
        
        # Check no Infs or NaNs
        @test all(isfinite.(fused_image))
        @test all(isfinite.(fused_sd))
    end
    
end
