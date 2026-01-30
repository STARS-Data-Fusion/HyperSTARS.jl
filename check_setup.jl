#!/usr/bin/env julia

"""
Check HyperSTARS.jl Installation and Setup

This script verifies that HyperSTARS.jl and its dependencies are properly installed
and provides information about the companion EMIT-L2A-RFL Python package.
"""

using Pkg

println("=" ^ 70)
println("HyperSTARS.jl Installation Check")
println("=" ^ 70)
println()

# Check Julia version
println("✓ Julia Version: ", VERSION)
if VERSION < v"1.11.0"
    println("  ⚠️  Warning: HyperSTARS.jl requires Julia 1.11.0 or later")
else
    println("  ✓ Julia version is compatible")
end
println()

# Check if HyperSTARS can be loaded
println("Testing HyperSTARS.jl package...")
try
    using HyperSTARS
    println("  ✓ HyperSTARS.jl loaded successfully")
    
    # Check for key functions
    functions_to_check = [
        "scene_fusion_pmap",
        "hyperSTARS_fusion_kr_dict",
        "InstrumentGeoData",
        "InstrumentData",
        "unif_weighted_obs_operator_centroid",
        "mat32_corD"
    ]
    
    println("\n  Checking key functions:")
    for func in functions_to_check
        if isdefined(HyperSTARS, Symbol(func))
            println("    ✓ ", func)
        else
            println("    ✗ ", func, " not found")
        end
    end
catch e
    println("  ✗ Error loading HyperSTARS.jl:")
    println("    ", e)
    println("\n  Try running: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
end
println()

# Check critical dependencies
println("Checking critical dependencies...")
critical_deps = [
    "Rasters",
    "ArchGDAL",
    "Distributed",
    "MultivariateStats",
    "LinearAlgebra",
    "Plots",
    "JLD2"
]

for dep in critical_deps
    try
        eval(Meta.parse("using $dep"))
        println("  ✓ ", dep)
    catch e
        println("  ✗ ", dep, " - Error: ", e)
    end
end
println()

# Check for parallel processing capability
println("Checking parallel processing...")
using Distributed
n_workers = nworkers()
println("  Number of worker processes: ", n_workers)
if n_workers == 1
    println("  ℹ️  Note: Add workers with addprocs(N) for parallel processing")
else
    println("  ✓ Multiple workers available for parallel processing")
end
println()

# Check for example data
println("Checking for example data...")
data_dir = joinpath(@__DIR__, "data")
if isdir(data_dir)
    println("  ✓ data/ directory exists")
    files = readdir(data_dir)
    if length(files) > 0
        println("    Files found:")
        for file in files
            println("      - ", file)
        end
    else
        println("    ⚠️  data/ directory is empty")
    end
else
    println("  ℹ️  data/ directory not found")
    println("    You'll need to prepare data using EMIT-L2A-RFL Python package")
end
println()

# Information about EMIT-L2A-RFL
println("=" ^ 70)
println("EMIT Data Preparation")
println("=" ^ 70)
println()
println("To prepare EMIT data for HyperSTARS.jl, use the companion Python package:")
println()
println("  Repository: https://github.com/STARS-Data-Fusion/EMIT-L2A-RFL")
println()

emit_path = joinpath(dirname(@__DIR__), "EMIT-L2A-RFL")
if isdir(emit_path)
    println("  ✓ EMIT-L2A-RFL found at: ", emit_path)
    println()
    println("  To activate the Python environment:")
    println("    cd ", emit_path)
    println("    mamba activate EMITL2ARFL")
    println()
    println("  To test the installation:")
    println("    python -c 'import EMITL2ARFL; print(\"EMITL2ARFL version:\", EMITL2ARFL.__version__)'")
else
    println("  ℹ️  EMIT-L2A-RFL not found in parent directory")
    println()
    println("  To install:")
    println("    git clone https://github.com/STARS-Data-Fusion/EMIT-L2A-RFL.git")
    println("    cd EMIT-L2A-RFL")
    println("    mamba create -n EMITL2ARFL -c conda-forge python=3.10 hdf5 h5py netcdf4")
    println("    mamba activate EMITL2ARFL")
    println("    pip install -e .")
end
println()

# Summary
println("=" ^ 70)
println("Next Steps")
println("=" ^ 70)
println()
println("1. If you haven't already, set up NASA Earthdata login:")
println("   - Register at: https://urs.earthdata.nasa.gov/")
println("   - Create ~/.netrc with your credentials")
println()
println("2. Download EMIT data using EMIT-L2A-RFL Python package")
println()
println("3. Run the examples:")
println("   julia --project=. examples/hyperstars_example.jl")
println("   julia --project=. examples/emit_hls_demo.jl")
println()
println("4. For detailed workflow, see: EMIT_DATA_WORKFLOW.md")
println()
println("=" ^ 70)
