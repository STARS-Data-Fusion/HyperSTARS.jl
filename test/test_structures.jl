"""
Test file for HyperSTARS data structures: InstrumentData, InstrumentGeoData, KSModel, HSModel
"""

@testset "Data Structures" begin
    
    @testset "InstrumentData Creation" begin
        # Test basic InstrumentData creation
        data = randn(10, 50, 5)  # 10 spatial samples, 50 wavelengths, 5 time steps
        bias = ones(50)
        uq = 0.1 * ones(50)
        spatial_resolution = [30.0, 30.0]
        dates = 1:5
        coords = randn(10, 2)
        wavelengths = LinRange(400, 2500, 50)
        rsr = 5.0  # FWHM
        
        inst_data = InstrumentData(data, bias, uq, spatial_resolution, dates, coords, wavelengths, rsr)
        
        @test inst_data.data == data
        @test inst_data.spatial_resolution == spatial_resolution
        @test size(inst_data.data, 1) == 10
        @test size(inst_data.data, 2) == 50
        @test size(inst_data.data, 3) == 5
        @test length(inst_data.dates) == 5
        @test length(inst_data.wavelengths) == 50
    end
    
    @testset "InstrumentGeoData Creation" begin
        # Test basic InstrumentGeoData creation
        origin = [500000.0, 4000000.0]
        cell_size = [30.0, -30.0]
        ndims = [100, 100]
        fidelity = 0
        dates = 1:10
        wavelengths = LinRange(400, 2500, 50)
        rsr = 5.0
        
        geo_data = InstrumentGeoData(origin, cell_size, ndims, fidelity, dates, wavelengths, rsr)
        
        @test geo_data.origin == origin
        @test geo_data.cell_size == cell_size
        @test geo_data.ndims == ndims
        @test geo_data.fidelity == 0
        @test size(geo_data.wavelengths, 1) == 50
    end
    
    @testset "InstrumentGeoData Fidelity Values" begin
        # Test all three fidelity levels
        base_params = (
            origin = [500000.0, 4000000.0],
            cell_size = [30.0, -30.0],
            ndims = [100, 100],
            dates = 1:10,
            wavelengths = LinRange(400, 2500, 50),
            rsr = 5.0
        )
        
        for fidelity in 0:2
            geo_data = InstrumentGeoData(
                base_params.origin,
                base_params.cell_size,
                base_params.ndims,
                fidelity,
                base_params.dates,
                base_params.wavelengths,
                base_params.rsr
            )
            @test geo_data.fidelity == fidelity
        end
    end
    
    @testset "KSModel Structure" begin
        # Test KSModel creation
        H = randn(20, 50)
        Q = Matrix(Diagonal(ones(50)))
        F = Matrix(Diagonal(ones(50)))
        
        model = HyperSTARS.KSModel(H, Q, F)
        
        @test model.H == H
        @test model.Q == Q
        @test model.F == F
        @test size(model.H) == (20, 50)
        @test size(model.Q) == (50, 50)
    end
    
    @testset "KSModel with Identity Scaling" begin
        # Test KSModel with UniformScaling state transition
        H = sparse(randn(20, 50))
        Q = Matrix(Diagonal(ones(50)))
        F = UniformScaling(1.0)  # Identity scaling
        
        model = HyperSTARS.KSModel(H, Q, F)
        
        @test model.F == I
        @test isa(model.F, UniformScaling)
    end
    
    @testset "HSModel Structure" begin
        # Test HSModel creation
        Hw = randn(50, 10)  # Wavelength observation matrix
        Hs = randn(15, 20)  # Spatial observation matrix
        Vw = Diagonal(ones(50))  # Wavelength noise covariance
        Vs = Diagonal(ones(15))  # Spatial noise covariance
        Q = Matrix(Diagonal(ones(200)))
        F = UniformScaling(1.0)
        
        model = HyperSTARS.HSModel(Hw, Hs, Vw, Vs, Q, F)
        
        @test model.Hw == Hw
        @test model.Hs == Hs
        @test model.Vw == Vw
        @test model.Vs == Vs
        @test size(model.Q) == (200, 200)
    end
    
    @testset "HSModel with Sparse Matrices" begin
        # Test HSModel with sparse matrices
        Hw = sparse(randn(50, 10))
        Hs = sparse(randn(15, 20))
        Vw = Diagonal(ones(50))
        Vs = UniformScaling(1.0)
        Q = Matrix(Diagonal(ones(200)))
        F = UniformScaling(1.0)
        
        model = HyperSTARS.HSModel(Hw, Hs, Vw, Vs, Q, F)
        
        @test issparse(model.Hw)
        @test issparse(model.Hs)
        @test isa(model.Vs, UniformScaling)
    end
    
    @testset "Data Consistency" begin
        # Test that InstrumentData maintains data consistency
        n_spatial = 20
        n_waves = 100
        n_time = 10
        
        data = randn(n_spatial, n_waves, n_time)
        bias = randn(n_waves)
        uq = abs.(randn(n_waves))
        spatial_resolution = [30.0, 30.0]
        dates = 1:n_time
        coords = randn(n_spatial, 2)
        wavelengths = LinRange(400, 2500, n_waves)
        rsr = 10.0
        
        inst_data = InstrumentData(data, bias, uq, spatial_resolution, dates, coords, wavelengths, rsr)
        
        @test size(inst_data.data) == (n_spatial, n_waves, n_time)
        @test length(inst_data.dates) == n_time
        @test size(inst_data.coords) == (n_spatial, 2)
        @test length(inst_data.wavelengths) == n_waves
    end
    
end
