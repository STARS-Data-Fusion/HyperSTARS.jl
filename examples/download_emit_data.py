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