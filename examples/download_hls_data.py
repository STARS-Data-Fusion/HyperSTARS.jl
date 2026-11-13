# mamba create -n hyperstars_example -c conda-forge python=3.12 hdf5 h5py netcdf4 scikit-learn pykdtree scikit-image
# mamba activate hyperstars_example
# pip install EMITL2ARFL
# pip install harmonized-landsat-sentinel
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
    gdf = gpd.read_file("iowa_small.geojson")

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