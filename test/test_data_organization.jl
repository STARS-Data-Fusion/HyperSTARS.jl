"""
Test file for data organization functions
"""

@testset "Data Organization" begin
    
    @testset "Basic Data Organization" begin
        # Create test data structures
        n_spatial = 50
        n_waves = 30
        n_time = 5
        
        # Create instrument data
        inst_data_array = randn(n_spatial, n_waves, n_time)
        inst_data = InstrumentData(
            inst_data_array,
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial, 2) .* 1000,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        # Create geospatial data
        inst_geo = InstrumentGeoData(
            [500000.0, 4000000.0],
            [30.0, -30.0],
            [100, 100],
            0,  # Highest fidelity
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        target_geo = InstrumentGeoData(
            [500000.0, 4000000.0],
            [30.0, -30.0],
            [100, 100],
            0,
            1:n_time,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        # Test organize_data function
        full_extent = [500000.0, 4001500.0; 4000000.0, 3997000.0]
        ss_xy = randn(10, 2) .* 100 .+ 500000
        ss_ij = ones(Int, 10, 2)
        res_flag = [0]  # Highest fidelity
        
        measurements, out_ss_ij = organize_data(
            full_extent,
            [inst_geo],
            [inst_data],
            target_geo,
            ss_xy,
            ss_ij,
            res_flag
        )
        
        @test length(measurements) == 1
        @test isa(measurements[1], InstrumentData)
    end
    
    @testset "Multiple Instruments Organization" begin
        # Test organize_data with multiple instruments at different fidelities
        n_spatial = 40
        n_waves = 25
        n_time = 4
        
        # Create two instruments with different resolutions
        inst_data1 = InstrumentData(
            randn(n_spatial, n_waves, n_time),
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial, 2) .* 1000,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_data2 = InstrumentData(
            randn(30, n_waves, n_time),
            randn(n_waves),
            0.1 * ones(n_waves),
            [100.0, 100.0],
            1:n_time,
            randn(30, 2) .* 2000,
            LinRange(400, 2500, n_waves),
            10.0
        )
        
        inst_geo1 = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [100, 100],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        inst_geo2 = InstrumentGeoData(
            [499000.0, 4001000.0], [100.0, -100.0], [50, 50],
            2, 1:n_time, LinRange(400, 2500, n_waves), 10.0
        )
        
        target_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [100, 100],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        full_extent = [499000.0, 4002000.0; 4000000.0, 3996500.0]
        ss_xy = randn(15, 2) .* 500 .+ 500000
        ss_ij = ones(Int, 15, 2)
        res_flag = [0, 2]  # Different fidelities
        
        measurements, out_ss_ij = organize_data(
            full_extent,
            [inst_geo1, inst_geo2],
            [inst_data1, inst_data2],
            target_geo,
            ss_xy,
            ss_ij,
            res_flag
        )
        
        @test length(measurements) == 2
        @test all(isa.(measurements, InstrumentData))
    end
    
    @testset "Organization Preserves Data Shapes" begin
        # Verify that organize_data preserves key data dimensions
        n_spatial = 30
        n_waves = 20
        n_time = 3
        
        inst_data = InstrumentData(
            randn(n_spatial, n_waves, n_time),
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial, 2) .* 500,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [80, 80],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        target_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [80, 80],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        full_extent = [500000.0, 4001500.0; 4000000.0, 3997700.0]
        ss_xy = randn(10, 2) .* 200 .+ 500000
        ss_ij = ones(Int, 10, 2)
        res_flag = [0]
        
        measurements, _ = organize_data(
            full_extent, [inst_geo], [inst_data], target_geo, ss_xy, ss_ij, res_flag
        )
        
        # Wavelengths should be preserved
        @test length(measurements[1].wavelengths) == n_waves
        @test length(measurements[1].dates) == n_time
    end
    
    @testset "Organization with Fidelity Levels" begin
        # Test that resolution flag is respected
        n_spatial = 35
        n_waves = 20
        n_time = 3
        
        # Create three instruments with different fidelities
        inst_datas = []
        inst_geos = []
        
        for (fidelity, cell_size) in [(0, [30.0, -30.0]), (1, [60.0, -60.0]), (2, [100.0, -100.0])]
            push!(inst_datas, InstrumentData(
                randn(n_spatial, n_waves, n_time),
                randn(n_waves),
                0.1 * ones(n_waves),
                abs.(cell_size),
                1:n_time,
                randn(n_spatial, 2) .* 500,
                LinRange(400, 2500, n_waves),
                5.0
            ))
            
            push!(inst_geos, InstrumentGeoData(
                [500000.0, 4000000.0],
                cell_size,
                [100, 100],
                fidelity,
                1:n_time,
                LinRange(400, 2500, n_waves),
                5.0
            ))
        end
        
        target_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [100, 100],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        full_extent = [500000.0, 4002000.0; 4000000.0, 3997000.0]
        ss_xy = randn(15, 2) .* 300 .+ 500000
        ss_ij = ones(Int, 15, 2)
        res_flag = [0, 1, 2]
        
        measurements, out_ss_ij = organize_data(
            full_extent, inst_geos, inst_datas, target_geo, ss_xy, ss_ij, res_flag
        )
        
        @test length(measurements) == 3
        @test all(isa.(measurements, InstrumentData))
    end
    
    @testset "Organization Output Consistency" begin
        # Verify reproducibility of organize_data
        n_spatial = 30
        n_waves = 20
        n_time = 3
        
        inst_data = InstrumentData(
            randn(n_spatial, n_waves, n_time),
            randn(n_waves),
            0.1 * ones(n_waves),
            [30.0, 30.0],
            1:n_time,
            randn(n_spatial, 2) .* 500,
            LinRange(400, 2500, n_waves),
            5.0
        )
        
        inst_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [80, 80],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        target_geo = InstrumentGeoData(
            [500000.0, 4000000.0], [30.0, -30.0], [80, 80],
            0, 1:n_time, LinRange(400, 2500, n_waves), 5.0
        )
        
        full_extent = [500000.0, 4001500.0; 4000000.0, 3997700.0]
        ss_xy = randn(10, 2) .* 200 .+ 500000
        ss_ij = ones(Int, 10, 2)
        res_flag = [0]
        
        # Call organize_data twice with same inputs
        m1, s1 = organize_data(full_extent, [inst_geo], [inst_data], target_geo, ss_xy, ss_ij, res_flag)
        m2, s2 = organize_data(full_extent, [inst_geo], [inst_data], target_geo, ss_xy, ss_ij, res_flag)
        
        # Results should be identical
        @test size(m1[1].data) == size(m2[1].data)
        @test maximum(abs.(m1[1].data .- m2[1].data)) < 1e-10
    end
    
end
