#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRANSPORT="${NAC_OPT_MCP_TRANSPORT:-stdio}"

if [[ -n "${NAC_OPT_CONDA_SH:-}" && -n "${NAC_OPT_CONDA_ENV:-}" ]]; then
  source "${NAC_OPT_CONDA_SH}"
  conda activate "${NAC_OPT_CONDA_ENV}"
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${REPO_ROOT}"
exec fastmcp run "mcp/server.py:mcp" --transport "${TRANSPORT}" "$@"
