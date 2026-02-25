"""
    generate_emit_metadata.jl

Generate EMIT metadata CSV file with wavelength information and quality flags.

The EMIT (Earth Surface Mineral and Temperature) sensor has 285 spectral bands
covering the visible to shortwave infrared (400-2500 nm).
"""

using DelimitedFiles

# EMIT sensor specifications
# 285 bands spanning 400-2500 nm with ~7.3 nm resolution
emit_wavelengths = collect(Float64, 400:2500/285:2500)
n_bands = length(emit_wavelengths)

# Create metadata matrix: wavelength, good_wavelengths flag, fwhm
emit_metadata = zeros(Float64, n_bands, 3)
emit_metadata[:, 1] = emit_wavelengths

# Column 2: Quality flags (1 = good, 0 = bad/noisy)
# Mark some bands as bad: atmospheric water absorption bands and noisy regions
bad_band_regions = [
    (1375, 1425),   # Water absorption band
    (1800, 1950),   # Water/CO2 absorption
    (2350, 2400),   # CO2 absorption
    (2500, 2600),   # Beyond sensor range
]

for i in 1:n_bands
    wl = emit_wavelengths[i]
    is_bad = false
    for (lo, hi) in bad_band_regions
        if lo <= wl <= hi
            is_bad = true
            break
        end
    end
    emit_metadata[i, 2] = is_bad ? 0 : 1
end

# Column 3: FWHM (Full Width at Half Maximum) in nm
# EMIT has approximately 7.3 nm spectral resolution
emit_metadata[:, 3] .= 7.3

# Create header
header = ["Wavelength_nm", "GoodWavelength", "FWHM_nm"]

# Write to file
# Auto-detect environment and use appropriate path
if haskey(ENV, "HYPERSTARS_METADATA_DIR")
    data_dir = ENV["HYPERSTARS_METADATA_DIR"]
    println("Using data directory from HYPERSTARS_METADATA_DIR: $data_dir")
elseif haskey(ENV, "HYPERSTARS_DATA_DIR")
    data_dir = ENV["HYPERSTARS_DATA_DIR"]
    println("Using data directory from HYPERSTARS_DATA_DIR: $data_dir")
elseif isdir("/gpfs/scratch/refl-datafusion-trtd/")
    # HPC environment (fallback)
    data_dir = "/gpfs/scratch/refl-datafusion-trtd"
    println("Using HPC data directory: $data_dir")
else
    # Local environment
    data_dir = expanduser("~/data")
    println("Using local data directory: $data_dir")
end

mkpath(data_dir)
metadata_path = joinpath(data_dir, "EMIT_metadata.csv")

open(metadata_path, "w") do io
    writedlm(io, reshape(header, 1, :), ',')
    writedlm(io, emit_metadata, ',')
end

println("✓ Created: $metadata_path")
println("  - $(n_bands) bands spanning 400-2500 nm")
println("  - $(sum(emit_metadata[:, 2])) good wavelengths")
println("  - $(sum(emit_metadata[:, 2] .== 0)) bands flagged as bad (atmospheric absorption)")
println("\nEMIT metadata file generated successfully!")
