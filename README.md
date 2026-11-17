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
AOI_FILE     = "iowa_small.geojson"
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

**python**
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

**bash**
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

### Step 3: Run the example fusion: emit_hls_demo_v2.jl

This example assumes data have been downloaded with the supplied downloader scripts. If you've downloaded data from AppEEARS, you will need to modify how your data are loaded. An updated example assuming AppEEARS inputs is forthcoming...

First, ensure you have both EMIT and HLS data, then modify the paths in the example:

```bash
julia --project=. examples/emit_hls_demo_v2.jl
```

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


