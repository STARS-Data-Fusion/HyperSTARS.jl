# HyperSTARS.jl

Hyperspectral Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) Julia Package

This Julia package, `HyperSTARS.jl`, is designed for advanced hyperspectral data fusion. It combines data from multiple instruments with varying spatial, spectral, and temporal resolutions into a single, high-resolution, fused product. The core methodology leverages state-space models and advanced statistical filtering and smoothing techniques (specifically, Kalman filtering and smoothing variants), making it robust for integrating diverse remote sensing datasets.

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
Technical contributor for methodology development.
NASA Jet Propulsion Laboratory 

Amy Braverman (she/her)<br>
[amy.j.braverman@jpl.nasa.gov](mailto:amy.j.braverman@jpl.nasa.gov)<br>
Technical contributor for methodology development.
NASA Jet Propulsion Laboratory 

Philip Brodrick (he/him) <br>
[philip.brodrick@jpl.nasa.gov](mailto:philip.brodrick@jpl.nasa.gov)<br>
Science and applications discussions, EMIT data considerations.<br>
NASA Jet Propulsion Laboratory 

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Science and applications discussions.<br>
NASA Jet Propulsion Laboratory

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
