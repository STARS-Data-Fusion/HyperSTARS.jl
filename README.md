# HyperSTARS.jl

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.11%2B-purple.svg)](https://julialang.org/downloads/)
[![EMIT](https://img.shields.io/badge/Data-EMIT%20L2A%20RFL-green.svg)](https://doi.org/10.5067/EMIT/EMITL2ARFL.001)

Hyperspectral Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (HyperSTARS) Julia Package

This Julia package, `HyperSTARS.jl`, is designed for advanced hyperspectral data fusion. It combines data from multiple instruments with varying spatial, spectral, and temporal resolutions into a single, high-resolution, fused product. The core methodology leverages state-space models and advanced statistical filtering and smoothing techniques (specifically, Kalman filtering and smoothing variants), making it robust for integrating diverse remote sensing datasets.

---

## Table of Contents

- [Quick Start](#quick-start-1-minute)
<!-- 
- [Team](#team)
-->
- [Getting Started: Complete Guide](#getting-started-complete-step-by-step-guide)
  - [Step 1: Install Julia Package](#step-1-install-julia-package)
  - [Step 2: NASA Earthdata Credentials](#step-2-set-up-nasa-earthdata-credentials)
  - [Step 3: Install EMIT Data Downloader](#step-3-install-emit-data-downloader-python)
  - [Step 4: Download Example Data](#step-4-download-example-data)
  - [Step 5: Download HLS Data (Optional)](#step-5-download-hls-data-optional)
  - [Step 6: Run the Examples](#step-6-run-the-examples)
  - [Step 7: Customize for Your Data](#step-7-customize-for-your-data)
  - [Troubleshooting](#troubleshooting)
- [Key Features](#key-features)
- [Core Components](#core-components)

---

<!-- 
## Team

Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
Principal investigator: lead of data fusion methodological development and Julia code implementations.<br>
NASA Jet Propulsion Laboratory 

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer for data processing pipelines, code organization and management.<br>
NASA Jet Propulsion Laboratory 

Nimrod Carmon (he/him)<br>
[nimrod.carmon@jpl.nasa.gov](mailto:nimrod.carmon@jpl.nasa.gov)<br>
Technical contributor for data processing, validation/verification, and hyperspectral resampling<br>
NASA Jet Propulsion Laboratory 

Jouni I. Susiluoto<br>
[jouni.i.susiluoto@jpl.nasa.gov](mailto:jouni.i.susiluoto@jpl.nasa.gov)<br>
Technical contributor for methodology development.<br>
NASA Jet Propulsion Laboratory 

Amy Braverman (she/her)<br>
[amy.j.braverman@jpl.nasa.gov](mailto:amy.j.braverman@jpl.nasa.gov)<br>
Technical contributor for methodology development.<br>
NASA Jet Propulsion Laboratory 

Philip Brodrick (he/him) <br>
[philip.brodrick@jpl.nasa.gov](mailto:philip.brodrick@jpl.nasa.gov)<br>
Science and applications discussions, EMIT data considerations.<br>
NASA Jet Propulsion Laboratory 

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Science and applications discussions.<br>
NASA Jet Propulsion Laboratory
-->

## Getting Started: Complete Step-by-Step Guide

This guide will walk you through everything needed to run the HyperSTARS fusion examples, from installation to execution.

### Prerequisites

Before starting, ensure you have:

- ✅ **Julia 1.11.0 or later** - [Download here](https://julialang.org/downloads/)
- ✅ **Python 3.10+ with conda/mamba** - [Get Miniforge](https://github.com/conda-forge/miniforge)
- ✅ **NASA Earthdata account** - [Register free](https://urs.earthdata.nasa.gov/)
- ✅ **~10 GB disk space** - For data and dependencies
- ✅ **Basic command line knowledge** - Running bash/shell commands

### Step 1: Install Julia Package

Clone this repository and install dependencies:

**julia**
```julia
using Pkg
Pkg.add(url = "https://github.com/STARS-Data-Fusion/HyperSTARS.jl", rev = "v2.0.0")
```

### Step 2: Download Data

#### Option 1: Download using supplied python helper packages:

NOTE: In this option, unless you have access to AWS the full EMIT granules must be downloaded locally before spatial subsetting. Each granule (REFL, MASKS, UNCERT together) can be several GBs. The supplied download script deletes full downloaded files on a monthly cadence to avoid exploding data volumes. To avoid downloading full granules, use AppEEARS (Option 2).

1. Set Up NASA Earthdata Credentials

EMIT and HLS data require NASA Earthdata authentication:

**bash**
```bash
# Create .netrc file with your NASA Earthdata credentials
cat > ~/.netrc << EOF
machine urs.earthdata.nasa.gov
login YOUR_USERNAME
password YOUR_PASSWORD
EOF

# Secure the file
chmod 600 ~/.netrc
```

> **Note**: Replace `YOUR_USERNAME` and `YOUR_PASSWORD` with your actual NASA Earthdata credentials from https://urs.earthdata.nasa.gov/

2. Install EMIT and HLS Data Downloaders (Python)

The companion Python packages download and preprocess EMIT and HLS data from PyPI:

**bash**
```bash
# Create conda environment with compatible HDF5 libraries
mamba create -n hyperstars_example -c conda-forge python=3.12 hdf5 h5py netcdf4 scikit-learn pykdtree scikit-image
mamba activate hyperstars_example

# Install the package from PyPI
pip install EMITL2ARFL
pip install harmonized-landsat-sentinel

# Verify installation
python -c "import EMITL2ARFL; print('✅ EMITL2ARFL installed successfully')"
python -c "import harmonized_landsat_sentinel; print('✅ harmonized-landsat-sentinel installed successfully')"
```

3. Download Example Data

Download EMIT data using download_emit_data.py. Update for your date range and AOI.

**python**
```python
# download_emit_data.py
import os
import shutil
from glob import glob

import earthaccess
import geopandas as gpd
import pandas as pd
import rasters as rt
from EMITL2ARFL import generate_EMIT_L2A_RFL_timeseries

START_DATE   = "2023-06-01"
END_DATE     = "2023-08-31"
AOI_FILE     = "iowa_ag.geojson"
DOWNLOAD_DIR = os.path.expanduser("./EMIT_download")
OUTPUT_DIR   = os.path.expanduser("./EMIT_data")


def month_chunks(start, end):
    """(first day, last day) of each calendar month in [start, end], clipped to the range."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for month in pd.period_range(start, end, freq="M"):
        yield max(start, month.start_time.normalize()), min(end, month.end_time.normalize())


def clean_download_dir(directory):
    """Delete downloaded granule folders only (EMIT_L2A_RFL_001_*), nothing else in the directory."""
    for path in glob(os.path.join(directory, "EMIT_L2A_RFL_001_*")):
        if os.path.isdir(path):
            shutil.rmtree(path)


def main():
    # Area of interest → 60 m UTM grid
    geometry = gpd.read_file(AOI_FILE).union_all()
    bbox_UTM = rt.Polygon(geometry).UTM.bbox
    grid = rt.RasterGrid.from_bbox(bbox_UTM, cell_size=60, crs=bbox_UTM.crs)

    earthaccess.login(strategy="netrc", persist=True)

    filenames = []
    for start, end in month_chunks(START_DATE, END_DATE):
        filenames += generate_EMIT_L2A_RFL_timeseries(
            start_date_UTC=start.date(),
            end_date_UTC=end.date(),
            geometry=grid,
            download_directory=DOWNLOAD_DIR,
            output_directory=OUTPUT_DIR,
        )
        clean_download_dir(DOWNLOAD_DIR)    # only reached if the month finished

    print(f"Wrote {len(filenames)} EMIT GeoTIFFs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

Run the script:

**bash**
```bash
python download_emit_data.py
```

Download HLS data using download_hls_data.py. Update for your date range and AOI.

```python
from os.path import join
import matplotlib.pyplot as plt
import earthaccess
import logging

import geopandas as gpd
import rasters as rt

from harmonized_landsat_sentinel import harmonized_landsat_sentinel as HLS
from harmonized_landsat_sentinel import generate_HLS_timeseries

# Configure logging to see info messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def main():
    # Date range
    start_date_UTC = "2023-06-01"
    end_date_UTC = "2023-08-31"

    # Download directory
    download_directory = "./HLS_download"

    # Output directory
    output_directory = "./HLS_data"

    # Arrow Peak area of interest
    gdf = gpd.read_file("iowa_ag.geojson")

    gdf.geometry[0]

    bbox_UTM = rt.Polygon(gdf.union_all()).UTM.bbox

    # Log into earthaccess using netrc credentials
    earthaccess.login(strategy="netrc", persist=True)

    filenames = generate_HLS_timeseries(
        start_date_UTC=start_date_UTC,
        end_date_UTC=end_date_UTC,
        geometry=bbox_UTM,
        download_directory=download_directory,
        output_directory=output_directory,
        source="both"
    )

if __name__ == "__main__":
    main()
```
Run the script.
```bash
python download_hls_data.py
```

### Option 2: Download data from AppEEARS
Visit [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/)

Go to Extract --> Area --> Start a New Request

For HLS
1. Select HLS products: HLSL30.002 and HLSS30.002
2. Choose your area of interest and date range
3. Download as NetCDF format 

For EMIT
1. Select EMIT products: EMIT_L2A_RFL.001, EMIT_L2A_MASK.001, EMIT_L2A_RFLUNCERT.001
2. Choose your area of interest and date range
3. Download as NetCDF format 

Note, AppEEARS has data size limits on individual requests so you may need to make several smaller requests.

### Step 6: Run the example fusion: emit_hls_demo.jl

This example assumes data have been downloaded with the supplied downloader scripts. If you've downloaded data from AppEEARS, you will need to modify how your data are loaded. An updated example assuming AppEEARS inputs is forthcoming...


#### Example 2: Real EMIT + HLS Fusion 🛰️

**What it does:** Processes real satellite data with cloud masking and temporal fusion

First, ensure you have both EMIT and HLS data, then modify the paths in the example:

```bash
julia --project=. examples/emit_hls_demo.jl
```

**Features demonstrated:**
- ✓ Loading real EMIT and HLS NetCDF data
- ✓ Spectral response function handling
- ✓ Cloud masking with Fmask
- ✓ Adaptive process noise covariance
- ✓ Animated time series output

**⏱️ Runtime:** ~30-60 minutes (depending on scene size and CPU cores)

**📊 Output:** 
- Fused images combining EMIT's spectral resolution (285 bands) with HLS's temporal coverage
- Animated GIF showing temporal evolution
- Full hyperspectral reconstructions at each time step
- Plots comparing observations with fused estimates


### Step 7: Customize for Your Data

To process your own data, modify the example scripts:

```julia
# 1. Update file paths
emit_files = glob("*.nc", "your_EMIT_directory")
hls_filename = "your_HLS_file.nc"

# 2. Define your target area
target_origin = [UTM_easting, UTM_northing]
target_csize = [30.0, -30.0]  # 30m pixel size
target_ndims = [rows, cols]

# 3. Set processing parameters
scf = 5  # Window size (5x5 target pixels)
nsamp = 50  # Number of spatial samples per window
window_buffer = 3  # Buffer pixels around windows

# 4. Run fusion
fused_images, fused_sd_images = gridded_fusion_pmap(
    data_list,
    inst_geodata,
    window_geodata,
    target_geodata,
    spectral_mean,
    prior_mean,
    prior_var,
    basis_functions,
    model_pars;
    nsamp=nsamp,
    window_buffer=window_buffer,
    target_times=1:num_timesteps,
    smooth=false
)
```

### Troubleshooting

#### 🐍 Python/HDF5 Errors

**Problem:** "HDF Error -101" or "Unable to open file"
```bash
# Solution: Set environment variable (especially on HPC systems)
set -Ux HDF5_USE_FILE_LOCKING FALSE  # fish shell
# or
export HDF5_USE_FILE_LOCKING=FALSE   # bash shell

# Reinstall with conda-forge HDF5 (recommended)
mamba install -c conda-forge h5py netcdf4 --force-reinstall
```

**Problem:** "Module not found: rasters"
```bash
# Solution: Ensure you're in the correct conda environment
mamba activate hyperstars_example
pip install --upgrade hyperstars_example
```

#### 💾 Memory Issues

**Symptom:** Julia crashes with "Out of Memory" errors

**Solutions:**
- ⬇️ Reduce window size: `scf = 3` (instead of 5)
- ⬇️ Reduce spatial samples: `nsamp = 30` (instead of 50)
- ⬇️ Process fewer time steps: `target_times = 1:10` (instead of 1:63)
- ⬇️ Use fewer parallel workers: `addprocs(4)` (instead of 8)
- 📊 Process smaller spatial subsets: reduce `target_ndims`

#### 🐌 Slow Performance

**Symptom:** Processing takes hours or appears stuck

**Solutions:**
- ⬆️ Increase parallel workers: `addprocs(16)` or `addprocs(Sys.CPU_THREADS)`
- ⚙️ Optimize BLAS threads: `BLAS.set_num_threads(1)` on each worker (already in examples)
- ⬇️ Reduce spatial buffer: `window_buffer = 2` (instead of 3 or 4)
- ✂️ Process subset first: Set `target_ndims = [50, 50]` for testing
- 🔍 Check CPU usage: Ensure workers are actually running in parallel

#### 🔑 NASA Earthdata Issues

**Problem:** "Authentication failed" or "403 Forbidden"

**Solution:** Check your .netrc file
```bash
# Verify .netrc exists and is properly formatted
cat ~/.netrc

# Should contain:
# machine urs.earthdata.nasa.gov
# login YOUR_USERNAME
# password YOUR_PASSWORD

# Fix permissions if needed
chmod 600 ~/.netrc
```

#### 🌐 Data Download Issues

**Problem:** "No granules found" or download hangs

**Solutions:**
- ✅ Verify date range has EMIT coverage: Check [EMIT Orbit Calculator](https://earth.jpl.nasa.gov/emit/)
- ✅ Confirm your area is covered: EMIT has global but not continuous coverage
- ✅ Try smaller date range: Start with 1-2 weeks
- ✅ Check internet connection and NASA server status

---

### Frequently Asked Questions (FAQ)

**Q: Do I need both Julia and Python?**  
A: Yes. Julia runs the fusion algorithm (HyperSTARS.jl), while Python downloads and preprocesses EMIT data (EMIT-L2A-RFL). They work together in the workflow.

**Q: Can I use my own area of interest?**  
A: Absolutely! Define your area using coordinates or KML files when downloading EMIT data. See Step 4, Option B for examples.

**Q: How much data do I need to download?**  
A: Each EMIT granule is ~500 MB. For a small area over 1 month, expect 2-5 GB. HLS data adds another 1-2 GB per month.

**Q: What sensors does HyperSTARS support?**  
A: Currently optimized for EMIT (hyperspectral), HLS (multispectral), and PACE (ocean color). The framework can be adapted for other sensors with appropriate observation operators.

**Q: How accurate are the fused results?**  
A: Accuracy depends on input data quality, coverage, and fusion parameters. The package provides uncertainty estimates alongside fused values. Validation against ground truth is recommended for your application.

**Q: Can I run this on HPC/cluster systems?**  
A: Yes! Julia's parallel processing works well on HPC. Just ensure HDF5 libraries are compatible (see troubleshooting) and increase `addprocs()` to match available cores.

**Q: What's the minimum area size I can process?**  
A: No technical minimum, but small areas (< 10×10 km) may have limited EMIT coverage. Larger areas (> 50×50 km) provide better statistics for fusion.

**Q: How long does processing take?**  
A: For a 50×50 pixel scene with 4 time steps: ~5-10 minutes on a modern laptop. Larger scenes scale roughly linearly with pixel count.

---

### Additional Resources

- **Detailed workflow**: [EMIT_DATA_WORKFLOW.md](EMIT_DATA_WORKFLOW.md)
- **Installation verification**: Run `julia check_setup.jl`
- **Example notebooks**: See `notebooks/` directory in EMIT-L2A-RFL
- **API documentation**: Coming soon

### Support

For questions or issues:
- **HyperSTARS.jl**: [maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)
- **EMIT data**: [gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)
- **GitHub Issues**: [Open an issue](https://github.com/STARS-Data-Fusion/HyperSTARS.jl/issues)

## Key Features

* **Multi-Sensor Data Fusion**: Integrates observations from various instruments with different characteristics (e.g., spatial resolution, spectral bands, temporal coverage).

* **Spatio-Spectral-Temporal Modeling**: Accounts for correlations and dependencies across spatial, spectral, and temporal dimensions.

* **Kalman Filtering and Smoothing**: Employs an efficient, recursive Bayesian estimation framework to produce optimal (minimum mean squared error) estimates of the underlying unobserved processes.

* **Kronecker Product Structures**: Utilizes Kronecker products for efficient handling of high-dimensional spatio-spectral covariance matrices, enhancing computational performance.

* **Adaptive Process Noise**: Allows for dynamically adjusting the model's process noise covariance based on the estimated state and its uncertainty, improving adaptability to changing environmental conditions.

* **Parallel Processing**: Designed to distribute computations across multiple spatial windows using Julia's `pmap` functionality, enabling scalable processing of large scenes.

* **Uncertainty Quantification**: Provides estimates of both the fused product and its associated uncertainty (e.g., standard deviation), crucial for downstream applications and decision-making.

## Core Components

The package is structured into several Julia files, each focusing on specific functionalities:

* **`HyperSTARS.jl` (Main Module)**:

    * Defines the overall module structure and exports key functions and data types.

    * Implements the main `hyperSTARS_fusion_kr_dict` (core fusion algorithm for a single window) and `scene_fusion_pmap` (orchestrates parallel fusion across a scene) functions.

    * Includes definitions for `KSModel` (standard Kalman state-space model) and `HSModel` (Hyperspectral STARS specific model with separated spatio-spectral components).

    * Defines `InstrumentData` and `InstrumentGeoData` structs for organizing diverse input data.

    * Contains the `woodbury_filter_kr` (Kalman filter update using Woodbury identity) and `smooth_series` (Kalman smoother) implementations.

    * Manages data organization (`organize_data`, `create_data_dicts`) for efficient processing.

* **`GP_utils.jl`**:

    * Provides various Gaussian Process (GP) related utility functions.

    * Includes implementations of common covariance functions such as `kernel_matrix` (Squared Exponential), `matern_cor`, `exp_cor`, `mat32_cor`, and `mat52_cor` (Matern family kernels).

    * Offers versions (`_D`) that take precomputed distance matrices for efficiency.

    * Implements `state_cov` for adaptive process noise covariance calculation.

    * Functions for building block-diagonal GP covariance matrices (`build_gpcov`).

* **`resampling_utils.jl`**:

    * Contains functions for handling resampling and creating observation operators.

    * `unif_weighted_obs_operator_centroid` and `gauss_weighted_obs_operator` construct observation matrices based on uniform or Gaussian weighting of target cells to sensor observations.

    * Includes `rsr_conv_matrix` for converting Relative Spectral Response (RSR) information into spectral convolution matrices, handling both FWHM and discrete RSR curve inputs.

* **`spatial_utils.jl` (and `spatial_utils_ll.jl`)**:

    * These files provide a suite of utility functions for spatial indexing, coordinate transformations, and grid operations.

    * Functions like `find_nearest_ij`, `find_all_ij_ext` (for centroid containment), `find_all_touching_ij_ext` (for cell overlap), `get_sij_from_ij` (index to coordinate conversion), and `bbox_from_centroid` are crucial for managing spatial data.

    * Includes methods for subsampling Basic Area Units (BAUs), notably `sobol_bau_ij` for quasi-random sampling using Sobol sequences.

    * Functions for determining raster origins and cell sizes (`get_origin_raster`, `cell_size`).

    * Utilities for finding and merging overlapping spatial extents (`find_overlapping_ext`, `merge_extents`).

## Citations and Acknowledgments

If you use HyperSTARS.jl in your research, please cite:

**EMIT Data Product:**
> Green, R. O., et al. (2023). Earth Surface Mineral Dust Source Investigation (EMIT) L2A Estimated Surface Reflectance and Uncertainty and Masks, Version 1. [Data set]. NASA EOSDIS Land Processes DAAC. [doi:10.5067/EMIT/EMITL2ARFL.001](https://doi.org/10.5067/EMIT/EMITL2ARFL.001)

**EMIT Mission:**
> Green, R. O., et al. (2024). The Earth Surface Mineral Dust Source Investigation (EMIT) on the International Space Station: In-flight instrument performance and first results. *Remote Sensing of Environment*, 282, 113277. [doi:10.1016/j.rse.2023.113277](https://doi.org/10.1016/j.rse.2023.113277)

**HLS Data Product:**
> Claverie, M., et al. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data set. *Remote Sensing of Environment*, 219, 145-161.

This work was supported by NASA Jet Propulsion Laboratory, California Institute of Technology.

## License

See [LICENSE](LICENSE) file for details.

---

**Ready to get started?** Head back to [Step 1](#step-1-install-julia-package) and follow the guide!

