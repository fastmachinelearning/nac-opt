#!/bin/bash
# Wrapper script to run global search with proper library paths

# Get conda environment path
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: CONDA_PREFIX not set. Please activate your conda environment first."
    exit 1
fi

# Set library path to use conda's libraries
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run the Python script with all arguments passed through
python "$(dirname "$0")/run_global_search.py" "$@"
