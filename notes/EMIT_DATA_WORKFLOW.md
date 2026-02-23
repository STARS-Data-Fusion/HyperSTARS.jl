# EMIT Data Workflow: From Python to Julia

This guide explains how to use the EMIT-L2A-RFL Python package to prepare EMIT data for use with HyperSTARS.jl.

## Overview

The workflow consists of two main steps:

1. **Python (EMIT-L2A-RFL)**: Download and preprocess EMIT L2A reflectance data
2. **Julia (HyperSTARS.jl)**: Perform hyperspectral data fusion with multiple sensors

## Step 1: Setting Up EMIT-L2A-RFL (Python)

### Installation

```bash
cd /Users/halverso/Projects/EMIT-L2A-RFL

# Create conda environment with compatible HDF5
mamba create -n EMITL2ARFL -c conda-forge python=3.10 hdf5 h5py netcdf4
mamba activate EMITL2ARFL

# Install the package
pip install -e .
```

### NASA Earthdata Login

Before downloading EMIT data, you need NASA Earthdata credentials:

1. Register at: https://urs.earthdata.nasa.gov/
2. Create a `.netrc` file in your home directory:

```bash
cat > ~/.netrc << EOF
machine urs.earthdata.nasa.gov
login YOUR_USERNAME
password YOUR_PASSWORD
EOF

chmod 600 ~/.netrc
```

### Download EMIT Data

Use the provided script as a template:

```python
import earthaccess
import geopandas as gpd
import rasters as rt
from EMITL2ARFL import generate_EMIT_L2A_RFL_timeseries

# Configuration
start_date_UTC = "2023-06-01"
end_date_UTC = "2023-09-01"
download_directory = "/tmp/EMIT_download"
output_directory = "~/data/EMIT_timeseries"

# Define area of interest (or load from KML/shapefile)
# Example: Load from KML
gdf = gpd.read_file("area_of_interest.kml")
bbox_UTM = rt.Polygon(gdf.unary_union).UTM.bbox
grid = rt.RasterGrid.from_bbox(bbox_UTM, cell_size=60, crs=bbox_UTM.crs)

# Login to earthaccess
earthaccess.login(strategy="netrc", persist=True)

# Download and process EMIT data
filenames = generate_EMIT_L2A_RFL_timeseries(
    start_date_UTC=start_date_UTC,
    end_date_UTC=end_date_UTC,
    geometry=grid,
    download_directory=download_directory,
    output_directory=output_directory
)

print(f"Generated {len(filenames)} EMIT reflectance files")
```

This will produce GeoTIFF files with filenames like:
- `EMIT_L2A_RFL_20230601.tif`
- `EMIT_L2A_RFL_20230602.tif`
- etc.

## Step 2: Using EMIT Data with HyperSTARS.jl

### Load EMIT Data in Julia

Once you have the EMIT GeoTIFF files, you can load them in Julia:

```julia
using Rasters
using ArchGDAL
using Dates
using HyperSTARS

# Load EMIT raster files
emit_files = readdir("~/data/EMIT_timeseries", join=true)
emit_files = filter(f -> endswith(f, ".tif"), emit_files)

# Load all EMIT data into a single array
emit_rasters = [Raster(f) for f in emit_files]
emit_array = cat([Array(r) for r in emit_rasters]..., dims=4)

# Extract metadata
emit_dates = [Date(match(r"(\d{8})", basename(f)).match, "yyyymmdd") for f in emit_files]
emit_waves = # wavelengths from EMIT (you'll need to extract these from metadata)
```

### Integration with HyperSTARS.jl

The EMIT data can then be combined with other sensors (HLS, PACE, etc.) as shown in the examples:

```julia
# Create EMIT geodata structure
emit_geodata = InstrumentGeoData(
    emit_origin,      # from get_centroid_origin_raster(emit_rasters[1])
    emit_csize,       # from collect(cell_size(emit_rasters[1]))
    emit_ndims,       # from collect(size(emit_array)[1:2])
    1,                # fidelity (1 = high resolution, not target)
    emit_times,       # converted dates to time indices
    emit_waves,       # wavelengths
    [0.1]            # default SRF for hyperspectral
)

# Create EMIT data structure
emit_data = InstrumentData(
    emit_array,
    zeros(size(emit_array)[3:4]),      # bias
    1e-6*ones(size(emit_array)[3:4]),  # uncertainty
    abs.(emit_csize),
    emit_times,
    [1,1],
    emit_waves,
    fwhm_emit  # full width at half maximum
)
```

## Sample Files and Data

### Example Scripts

1. **Python data preparation**: `/Users/halverso/Projects/EMIT-L2A-RFL/generate_kings_canyon_timeseries.py`
2. **Julia fusion example**: `/Users/halverso/Projects/HyperSTARS.jl/examples/emit_hls_demo.jl`

### Test Areas

- Kings Canyon example: `/Users/halverso/Projects/EMIT-L2A-RFL/upper_kings.kml`

## Data Format Summary

### EMIT-L2A-RFL Output
- **Format**: GeoTIFF (one file per date)
- **Bands**: ~285 spectral bands (hyperspectral)
- **Resolution**: 60m
- **Wavelength range**: ~380-2500 nm

### HyperSTARS.jl Input
- **Format**: 4D arrays (rows × cols × wavelengths × time)
- **Coordinate system**: UTM or lat/lon
- **Metadata**: Origin, cell size, dates, wavelengths

## Complete Workflow Example

### 1. Download EMIT Data (Python)

```bash
cd /Users/halverso/Projects/EMIT-L2A-RFL
mamba activate EMITL2ARFL
python generate_kings_canyon_timeseries.py
```

### 2. Download HLS Data

HLS data can be downloaded from: https://lpdaac.usgs.gov/products/hlsl30v002/

Or use NASA's AppEEARS tool: https://appeears.earthdatacloud.nasa.gov/

### 3. Run HyperSTARS Fusion (Julia)

```bash
cd /Users/halverso/Projects/HyperSTARS.jl
julia --project=. examples/emit_hls_demo.jl
```

## Troubleshooting

### Python/HDF5 Issues

If you encounter HDF5 errors:
```bash
set -Ux HDF5_USE_FILE_LOCKING FALSE  # fish shell
# or
export HDF5_USE_FILE_LOCKING=FALSE   # bash shell
```

See `/Users/halverso/Projects/EMIT-L2A-RFL/diagnostics/` for detailed troubleshooting.

### Julia Package Issues

If HyperSTARS.jl doesn't load:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.precompile()'
```

## References

- EMIT Data Product: [doi:10.5067/EMIT/EMITL2ARFL.001](https://doi.org/10.5067/EMIT/EMITL2ARFL.001)
- HLS Documentation: https://lpdaac.usgs.gov/products/hlsl30v002/
- HyperSTARS.jl Documentation: See [README.md](README.md)
