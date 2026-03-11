#!/usr/bin/env bash
set -e

# make like interactive shell
source "$HOME/.bashrc"

source "/home/users/jdweitz/miniforge3/etc/profile.d/conda.sh"
conda activate rule4ml_update

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" # need this or will get that version error

cd "/home/users/jdweitz/nac-opt_mcp/nac-opt"
fastmcp run mcp/server.py:mcp

