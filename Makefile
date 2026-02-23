.PHONY: help test install update clean format check examples docs

# Default target
help:
	@awk 'BEGIN{p=0} /^## Setup Instructions/{p=1} p && /^## / && !/^## Setup Instructions/{exit} p{print}' README.md \
	| sed -E '/^```/d; s/`//g; s/^[[:space:]]*#+[[:space:]]*//; s/\*\*([^*]+)\*\*/\1/g; s/\*([^*]+)\*/\1/g; s/\[([^]]+)\]\([^)]+\)/\1/g'

# Install package dependencies
install:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run all tests using Pkg.test()
test:
	julia --project=. -e 'using Pkg; Pkg.test()'

# Run tests directly (faster, with verbose output)
test-fast:
	julia --project=. test/runtests.jl

# Update dependencies to latest compatible versions
update:
	julia --project=. -e 'using Pkg; Pkg.update()'

# Clean build artifacts and caches
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf Manifest.toml
	@find . -type f -name "*.ji" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete. Run 'make install' to reinstall dependencies."

# Check package status
check:
	julia --project=. -e 'using Pkg; Pkg.status(); println("\n--- Environment Info ---"); versioninfo()'

# Run example scripts
examples:
	@echo "Running hyperstars_example.jl..."
	julia --project=. examples/hyperstars_example.jl
	@echo "\nTo run EMIT/HLS demo (requires data download):"
	@echo "  julia --project=. examples/emit_hls_demo.jl"
