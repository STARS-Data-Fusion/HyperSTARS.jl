"""
    generate_srf_files.jl

Generate spectral response function (SRF) CSV files for Landsat-8 (L30) and Sentinel-2 (S30)
based on published sensor characteristics.

These files are required by the HyperSTARS data fusion pipeline.
Output files:
  - HLS_L30_srf.csv: Landsat-8 spectral response functions
  - HLS_S30_srf.csv: Sentinel-2 spectral response functions
"""

using DelimitedFiles

# Define wavelength range (400-2500 nm with 1 nm resolution)
wavelengths = Float64.(400:2500)
nwaves = length(wavelengths)

# ============================================================================
# Landsat-8 (L30) - 7 bands + 2 thermal (we use 7 reflected)
# ============================================================================
# Band names: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2

# L30 band center wavelengths and FWHM (Full Width at Half Maximum)
l30_bands = [
    (name="Coastal",    center=433,   fwhm=20),    # Band 1
    (name="Blue",       center=483,   fwhm=60),    # Band 2
    (name="Green",      center=561,   fwhm=57),    # Band 3
    (name="Red",        center=655,   fwhm=37),    # Band 4
    (name="NIR",        center=865,   fwhm=28),    # Band 5
    (name="Skip",       center=0,     fwhm=0),     # Band 6 (thermal) - placeholder
    (name="SWIR1",      center=1609,  fwhm=89),    # Band 7
    (name="SWIR2",      center=2201,  fwhm=187)    # Band 8
]

# Create L30 SRF matrix (wavelengths × 9 columns: wavelength + 8 bands)
l30_srf = zeros(Float64, nwaves, 9)
l30_srf[:, 1] = wavelengths

# Generate Gaussian-like spectral response curves for each band
for (i, band) in enumerate(l30_bands)
    col_idx = i + 1
    if band.center > 0  # Skip thermal bands
        # Gaussian response centered at band center wavelength
        sigma = band.fwhm / (2 * sqrt(2 * log(2)))  # Convert FWHM to sigma
        response = exp.(-0.5 .* ((wavelengths .- band.center) ./ sigma) .^ 2)
        # Keep only responses above 1% of peak
        response[response .< 0.01] .= 0
        l30_srf[:, col_idx] = response
    end
end

# ============================================================================
# Sentinel-2 (S30) - 11 bands (matching HLS specification)
# ============================================================================
# Uses the same bands as L30 where they match, plus red edge and narrow NIR

s30_bands = [
    (name="Coastal",        center=443,   fwhm=20),    # Band 1
    (name="Blue",           center=490,   fwhm=65),    # Band 2
    (name="Green",          center=560,   fwhm=35),    # Band 3
    (name="Red",            center=665,   fwhm=30),    # Band 4
    (name="RedEdge1",       center=705,   fwhm=15),    # Band 5
    (name="RedEdge2",       center=740,   fwhm=15),    # Band 6
    (name="RedEdge3",       center=783,   fwhm=20),    # Band 7
    (name="NIR_Broad",      center=842,   fwhm=115),   # Band 8A
    (name="NIR",            center=865,   fwhm=106),   # Band 8
    (name="WaterVapor",     center=945,   fwhm=20),    # Band 9 (skip)
    (name="SWIR1",          center=1610,  fwhm=90),    # Band 11
    (name="Skip",           center=0,     fwhm=0),     # Placeholder
    (name="SWIR2",          center=2190,  fwhm=180)    # Band 12
]

# Create S30 SRF matrix (wavelengths × 14 columns: wavelength + 13 bands)
s30_srf = zeros(Float64, nwaves, 14)
s30_srf[:, 1] = wavelengths

for (i, band) in enumerate(s30_bands)
    col_idx = i + 1
    if band.center > 0  # Skip placeholder bands
        sigma = band.fwhm / (2 * sqrt(2 * log(2)))
        response = exp.(-0.5 .* ((wavelengths .- band.center) ./ sigma) .^ 2)
        response[response .< 0.01] .= 0
        s30_srf[:, col_idx] = response
    end
end

# ============================================================================
# Normalize responses to sum to 1.0 per wavelength
# ============================================================================
l30_srf[:, 2:end] .= l30_srf[:, 2:end] ./ (sum(l30_srf[:, 2:end], dims=2) .+ 1e-10)
s30_srf[:, 2:end] .= s30_srf[:, 2:end] ./ (sum(s30_srf[:, 2:end], dims=2) .+ 1e-10)

# ============================================================================
# Create header rows
# ============================================================================
l30_header = ["Wavelength", "Coastal", "Blue", "Green", "Red", "NIR", "Thermal", "SWIR1", "SWIR2"]
s30_header = ["Wavelength", "Coastal", "Blue", "Green", "Red", "RedEdge1", "RedEdge2", 
              "RedEdge3", "NIR_Broad", "NIR", "WaterVapor", "Placeholder", "SWIR1", "SWIR2"]

# ============================================================================
# Write CSV files to script directory
# ============================================================================
script_dir = dirname(@__FILE__)
data_dir = expanduser("~/data")

# Create data directory if it doesn't exist
mkpath(data_dir)

l30_path = joinpath(data_dir, "HLS_L30_srf.csv")
s30_path = joinpath(data_dir, "HLS_S30_srf.csv")

# Write L30 SRF file
open(l30_path, "w") do io
    writedlm(io, reshape(l30_header, 1, :), ',')
    writedlm(io, l30_srf, ',')
end
println("✓ Created: $l30_path")

# Write S30 SRF file
open(s30_path, "w") do io
    writedlm(io, reshape(s30_header, 1, :), ',')
    writedlm(io, s30_srf, ',')
end
println("✓ Created: $s30_path")

println("\nSpectral Response Function files generated successfully!")
println("Files are ready for use by the HyperSTARS data fusion pipeline.")
