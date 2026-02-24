# Spectral Response Function Files for HyperSTARS

## Overview

The HyperSTARS data fusion pipeline requires spectral calibration data to properly combine observations from different sensors. This document describes the spectral response function (SRF) and metadata files that have been generated.

## Files Generated

### 1. `HLS_L30_srf.csv` - Landsat-8 Spectral Response Functions
- **Location:** `~/data/HLS_L30_srf.csv`
- **Size:** ~98 KB
- **Content:** Spectral response curves for Landsat-8 (OLI sensor) 7 reflective bands
- **Bands Included:**
  - Coastal Aerosol (433 nm)
  - Blue (483 nm)
  - Green (561 nm)
  - Red (655 nm)
  - Near-Infrared (865 nm)
  - SWIR1 (1609 nm)
  - SWIR2 (2201 nm)
- **Format:** CSV with 2,101 rows (wavelength 400-2500 nm at 1 nm resolution) + 1 header row
- **Columns:** Wavelength (nm) + 8 band columns (7 usable + 1 thermal placeholder)

### 2. `HLS_S30_srf.csv` - Sentinel-2 Spectral Response Functions
- **Location:** `~/data/HLS_S30_srf.csv`
- **Size:** ~148 KB
- **Content:** Spectral response curves for Sentinel-2 (MSI sensor) 11 bands
- **Bands Included:**
  - Coastal (443 nm)
  - Blue (490 nm)
  - Green (560 nm)
  - Red (665 nm)
  - Red Edge 1 (705 nm)
  - Red Edge 2 (740 nm)
  - Red Edge 3 (783 nm)
  - NIR Broad (842 nm)
  - NIR (865 nm)
  - SWIR1 (1610 nm)
  - SWIR2 (2190 nm)
- **Format:** CSV with 2,101 rows (wavelength 400-2500 nm at 1 nm resolution) + 1 header row
- **Columns:** Wavelength (nm) + 13 band columns (11 usable + 2 placeholders)
- **Special Features:** For matching bands with Landsat-8, this file uses the same SRF as L30 (ensuring consistency in overlapping wavelengths)

### 3. `EMIT_metadata.csv` - EMIT Sensor Metadata
- **Location:** `~/data/EMIT_metadata.csv`
- **Size:** ~6.2 KB
- **Content:** Wavelength and quality information for EMIT (Earth Surface Mineral and Temperature) hyperspectral sensor
- **Band Information:**
  - 240 total bands spanning 400-2500 nm
  - 212 good wavelengths (flagged as usable)
  - 28 bands flagged as bad (atmospheric absorption regions)
- **Format:** CSV with 240 rows + 1 header row
- **Columns:**
  1. Wavelength_nm: Center wavelength of each band (nm)
  2. GoodWavelength: Quality flag (1=good, 0=bad/noisy)
  3. FWHM_nm: Full Width at Half Maximum (~7.3 nm for all EMIT bands)
- **Bad Band Regions Flagged:**
  - 1375-1425 nm: Water absorption band
  - 1800-1950 nm: Water/CO2 absorption
  - 2350-2400 nm: CO2 absorption

## Usage in HyperSTARS

These files are automatically loaded by the fusion scripts:
- `kings_canyon_hls_emit_local.jl` - Local development version
- `kings_canyon_hls_emit.jl` - HPC production version

The fusion pipeline uses these files to:
1. **Normalize sensor responses** to common spectral grid
2. **Account for spectral differences** between Landsat-8, Sentinel-2, and EMIT
3. **Filter bad wavelengths** (atmospheric absorption regions) before fusion
4. **Calculate observation operators** that map sensor measurements to physical quantities

## CSV Format Details

All CSV files use standard comma-delimited format with:
- **Header row:** First line contains column names
- **Data rows:** Subsequent lines contain numerical values
- **Delimiter:** Comma (`,`)
- **Decimal separator:** Period (`.`)
- **Line endings:** Unix-style (LF)

These can be read in Julia using:
```julia
using DelimitedFiles
data, header = readdlm(filename, ',', header=true)
```

## Next Steps

The HyperSTARS scripts can now proceed to the next phase:
1. **Data Loading:** Load actual HLS (Landsat-8, Sentinel-2) GeoTIFF raster files
2. **EMIT Loading:** Load EMIT hyperspectral data from GeoTIFF files
3. **Data Fusion:** Apply Kalman filter-based fusion to combine sensor observations
4. **Output:** Generate fused hyperspectral product with uncertainty estimates

## Notes

- These SRF files are derived from published sensor specifications and represent idealized responses
- Real sensor calibration data should be used for production work with actual satellite data
- The atmospheric absorption band flags in the EMIT metadata prevent artifacts from water and CO2 absorption regions
- All wavelengths are in nanometers (nm) and normalized to sum to 1.0 at each wavelength for proper spectral matching

## References

- **Landsat-8:** USGS Earth Resources Observation and Science (EROS) Center
- **Sentinel-2:** ESA Copernicus Program
- **EMIT:** NASA Jet Propulsion Laboratory (JPL)
