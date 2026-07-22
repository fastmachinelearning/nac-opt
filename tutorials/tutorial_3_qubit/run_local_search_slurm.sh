#!/bin/bash
#SBATCH --job-name=qubit_local
#SBATCH --account=amsc011              # Project/account name
#SBATCH --nodes=1                      # Local search is single-node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --constraint=gpu               # Perlmutter GPU nodes
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=qubit_local_%j.out
#SBATCH --error=qubit_local_%j.err
#SBATCH --qos=express_amsc             # use regular if Slurm rejects QoS for this partition
#
# CPU-only variant: job-name=qubit_local_cpu, constraint=cpu, remove --gpus-per-node,
#   sbatch --export=ALL,USE_TF_GPU=0,...  (or leave USE_TF_GPU unset after forcing CPU SBATCH)

set -euo pipefail
trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} command failed: ${BASH_COMMAND}" >&2' ERR

# Local search (combined QAT + pruning) for Tutorial 3 (qubit).
#
# Usage:
#   cd tutorials/tutorial_3_qubit
#   # point at a completed global-search results dir that contains best_model_for_local_search.yaml
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<GLOBAL_JOBID> run_local_search_slurm.sh
#
# Unique output folder (parallel runs): the batch script can see SLURM_JOB_ID; your login shell cannot.
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=...,LOCAL_SEARCH_USE_SLURM_JOB_ID=1 run_local_search_slurm.sh
#   -> ${GLOBAL_RESULTS_DIR}/local_search_combined_<SLURM_JOB_ID>/
# Or set a tag (e.g. Optuna trial id) explicitly:
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=...,LOCAL_SEARCH_TRIAL=42 run_local_search_slurm.sh
#   -> ${GLOBAL_RESULTS_DIR}/local_search_combined_42/
# Override fully: LOCAL_RESULTS_DIR=/path/to/out
#
# Custom architecture (e.g. Pareto pick ``best_low_lut_min_acc.yaml`` instead of default
# ``best_model_for_local_search.yaml``):
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<ID>,\
#     ARCH_YAML=./results/qubit_optuna_job_<ID>/best_low_lut_min_acc.yaml,\
#     LOCAL_RESULTS_DIR=./results/qubit_optuna_job_<ID>/local_search_lowlut_8b3i,\
#     LOCAL_SEARCH_TOTAL_BITS=8,LOCAL_SEARCH_INT_BITS=3 run_local_search_slurm.sh
#
# Single precision only (omit both to use t3_config.yaml local_search.precision_pairs list):
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=...,ARCH_YAML=...,LOCAL_RESULTS_DIR=...,\
#     LOCAL_SEARCH_TOTAL_BITS=8,LOCAL_SEARCH_INT_BITS=3 run_local_search_slurm.sh
#
# Outputs:
#   ${GLOBAL_RESULTS_DIR}/local_search_combined[_<trial or jobid>]/  (CSV logs + checkpoints)

# Ensure we run from the submit directory so relative paths work.
_submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${_submit_dir}"

# SLURM runs a *copy* of this script under /var/spool/slurmd/job*/slurm_script, so
# dirname(BASH_SOURCE) is NOT your repo — never use it for REPO_ROOT.
# Find nac-opt root by walking up from the directory where you ran sbatch until we see utils/.
REPO_ROOT=""
_d="$(cd "${_submit_dir}" && pwd)"
while [[ "${_d}" != "/" ]]; do
  if [[ -d "${_d}/utils" && -f "${_d}/utils/tf_local_search_combined.py" ]]; then
    REPO_ROOT="${_d}"
    break
  fi
  _d="$(dirname "${_d}")"
done
if [[ -z "${REPO_ROOT}" ]]; then
  echo "[ERROR] Could not locate nac-opt repo root (no utils/tf_local_search_combined.py). Submit from tutorials/tutorial_3_qubit." >&2
  exit 2
fi
export REPO_ROOT
echo "REPO_ROOT=${REPO_ROOT}"

module load python/3.10
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12

# Ensure conda functions are available in non-interactive batch shells.
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
elif [[ -f "$HOME/.bash_profile" ]]; then
  source "$HOME/.bash_profile"
elif [[ -f "$HOME/.bashrc.ext" ]]; then
  source "$HOME/.bashrc.ext"
elif [[ -f "$HOME/.bash_profile.ext" ]]; then
  source "$HOME/.bash_profile.ext"
fi

conda activate rule4ml_update

# Training device: default USE_TF_GPU=1 uses the allocated GPU. USE_TF_GPU=0 forces CPU
# (still allocates a GPU node unless you switch SBATCH to constraint=cpu and drop --gpus-per-node).
USE_TF_GPU="${USE_TF_GPU:-1}"
if [[ "${USE_TF_GPU}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
  echo "TF will use GPU(s) on this node (USE_TF_GPU=1)."
else
  export CUDA_VISIBLE_DEVICES=""
  echo "TF forced to CPU (USE_TF_GPU=0)."
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# XLA/libdevice helper (mirrors global search script)
_cuda_xla_roots=(
  "${CUDA_HOME:-}"
  "${CUDA_ROOT:-}"
  "${CUDA_PATH:-}"
  "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_nvcc"
  "${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvcc"
  "${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_nvcc"
  "/usr/local/cuda-12.2"
  "/usr/local/cuda"
)
_xla_cuda_data_dir=""
for _root in "${_cuda_xla_roots[@]}"; do
  [[ -z "${_root}" || ! -d "${_root}" ]] && continue
  if compgen -G "${_root}/nvvm/libdevice/libdevice"*.bc > /dev/null; then
    _xla_cuda_data_dir="${_root}"
    break
  fi
done
if [[ -z "${_xla_cuda_data_dir}" ]] && command -v find >/dev/null; then
  _found="$(find "${CONDA_PREFIX}/lib" -path '*/nvvm/libdevice/libdevice*.bc' 2>/dev/null | head -n 1)"
  if [[ -n "${_found}" ]]; then
    _xla_cuda_data_dir="$(dirname "$(dirname "$(dirname "${_found}")")")"
  fi
fi
if [[ -n "${_xla_cuda_data_dir}" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${_xla_cuda_data_dir} ${XLA_FLAGS:-}"
  echo "Set XLA_FLAGS for CUDA libdevice under: ${_xla_cuda_data_dir}"
else
  echo "WARN: Could not locate nvvm/libdevice/libdevice*.bc; GPU XLA may fail. Try: conda/pip install nvidia-cuda-nvcc-cu12" >&2
fi
unset _root _cuda_xla_roots _xla_cuda_data_dir _found

# Locate global-search results directory and best-model yaml
GLOBAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR:-}"
if [[ -z "${GLOBAL_RESULTS_DIR}" ]]; then
  echo "[ERROR] GLOBAL_RESULTS_DIR is not set. Example:" >&2
  echo "  sbatch --export=ALL,GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<GLOBAL_JOBID> run_local_search_slurm.sh" >&2
  exit 2
fi

ARCH_YAML="${ARCH_YAML:-${GLOBAL_RESULTS_DIR}/best_model_for_local_search.yaml}"
if [[ ! -f "${ARCH_YAML}" ]]; then
  echo "[ERROR] Could not find architecture yaml at: ${ARCH_YAML}" >&2
  exit 2
fi

SCRATCH_DIR="${SCRATCH:-$PWD}"
DATA_DIR="${DATA_DIR:-${SCRATCH_DIR}/qubit_data}"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] DATA_DIR does not exist: ${DATA_DIR}" >&2
  exit 2
fi

# Default output dir unless LOCAL_RESULTS_DIR is set explicitly.
if [[ -z "${LOCAL_RESULTS_DIR:-}" ]]; then
  if [[ "${LOCAL_SEARCH_USE_SLURM_JOB_ID:-0}" == "1" ]]; then
    LOCAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR}/local_search_combined_${SLURM_JOB_ID:-unknown}"
  elif [[ -n "${LOCAL_SEARCH_TRIAL:-}" ]]; then
    LOCAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR}/local_search_combined_${LOCAL_SEARCH_TRIAL}"
  else
    LOCAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR}/local_search_combined"
  fi
fi
mkdir -p "${LOCAL_RESULTS_DIR}"

# Python reads os.environ; bash-only variables are invisible unless exported.
export GLOBAL_RESULTS_DIR ARCH_YAML DATA_DIR LOCAL_RESULTS_DIR
export LOCAL_SEARCH_TOTAL_BITS="${LOCAL_SEARCH_TOTAL_BITS:-}"
export LOCAL_SEARCH_INT_BITS="${LOCAL_SEARCH_INT_BITS:-}"

echo "=========================================="
echo "Local search configuration"
echo "=========================================="
echo "GLOBAL_RESULTS_DIR: ${GLOBAL_RESULTS_DIR}"
echo "ARCH_YAML:          ${ARCH_YAML}"
echo "DATA_DIR:           ${DATA_DIR}"
echo "LOCAL_RESULTS_DIR:  ${LOCAL_RESULTS_DIR}"
echo "SLURM_JOB_ID:       ${SLURM_JOB_ID:-N/A}"
if [[ -n "${LOCAL_SEARCH_TOTAL_BITS:-}" && -n "${LOCAL_SEARCH_INT_BITS:-}" ]]; then
  echo "Precision override: LOCAL_SEARCH_TOTAL_BITS=${LOCAL_SEARCH_TOTAL_BITS} LOCAL_SEARCH_INT_BITS=${LOCAL_SEARCH_INT_BITS} (single pair; ignores t3_config precision_pairs list)"
else
  echo "Precision:          from t3_config.yaml local_search.precision_pairs"
fi
echo "=========================================="

# Run combined local search using the same settings as t3_config.yaml
# (local_search: qat_epochs, pruning_iterations, pruning_epochs, pruning_rate, precision_pairs)
python -c "
import os
import sys
from pathlib import Path
import yaml
import numpy as np

# Ensure repo root is importable (do NOT rely on cwd under SLURM).
repo_root = Path(os.environ['REPO_ROOT']).resolve()
sys.path.insert(0, str(repo_root))

from utils.tf_data_preprocessing import load_and_preprocess_qubit
from utils.tf_local_search_combined import combined_local_search_entrypoint

arch_yaml = Path(os.environ['ARCH_YAML']).resolve()
global_results_dir = Path(os.environ['GLOBAL_RESULTS_DIR']).resolve()
local_results_dir = Path(os.environ['LOCAL_RESULTS_DIR']).resolve()
data_dir = os.environ['DATA_DIR']

cfg = yaml.safe_load(open(Path('t3_config.yaml'), 'r'))
ds_cfg = cfg['dataset']
s_cfg = cfg['search']
ls_cfg = cfg['local_search']
n_folds = int(s_cfg.get('n_folds', 1))

_tb = os.environ.get('LOCAL_SEARCH_TOTAL_BITS', '').strip()
_ib = os.environ.get('LOCAL_SEARCH_INT_BITS', '').strip()
if _tb and _ib:
    precision_pairs = [{'total_bits': int(_tb), 'int_bits': int(_ib)}]
    print(f'Using single precision pair from env: {precision_pairs}')
else:
    precision_pairs = ls_cfg['precision_pairs']

# Prefer dataset window stored in the selected model yaml so local search matches global-search slice.
arch = yaml.safe_load(open(arch_yaml, 'r')) or {}
window = (arch.get('metadata', {}) or {}).get('dataset_window', {}) or {}
start_location = int(window.get('start_location', ds_cfg.get('start_location', 0)))
window_size = int(window.get('window_size', ds_cfg.get('window_size', 400)))

# Keep config next to this run's outputs (avoids overwrites when multiple precisions / dirs share one GLOBAL_RESULTS_DIR).
local_config_path = local_results_dir / 'local_search_config.yaml'
local_search_settings = {
  'pruning_settings': {
    'iterations': int(ls_cfg['pruning_iterations']),
    'epochs_per_iteration': int(ls_cfg['pruning_epochs']),
    'pruning_rate': float(ls_cfg['pruning_rate']),
  },
  'qat_settings': {
    'epochs': int(ls_cfg['qat_epochs']),
    'precision_pairs': precision_pairs,
  },
}
yaml.safe_dump(local_search_settings, open(local_config_path, 'w'), sort_keys=False)

x_train, y_train, x_test, y_test = load_and_preprocess_qubit(
  data_dir=data_dir,
  start_location=start_location,
  window_size=window_size,
  subset_size=ds_cfg.get('subset_size'),
  normalize=bool(ds_cfg.get('normalize', True)),
  flatten=bool(ds_cfg.get('flatten', True)),
  one_hot=bool(ds_cfg.get('one_hot', True)),
  num_classes=int(ds_cfg.get('num_classes', 2)),
)

# combined_local_search_entrypoint uses k-fold CV internally; pass empty val arrays.
x_val_empty = np.empty((0, *x_train.shape[1:]), dtype=x_train.dtype)
y_val_empty = np.empty((0, *y_train.shape[1:]), dtype=y_train.dtype)

print(f'Local search dataset window: start={start_location} window={window_size}')
print(f'Writing local_search_config.yaml to: {local_config_path}')
print(f'Writing local search outputs to: {local_results_dir}')

combined_local_search_entrypoint(
  architecture_yaml_path=str(arch_yaml),
  local_search_config_path=str(local_config_path),
  dataset=(x_train, y_train, x_val_empty, y_val_empty),
  results_dir=str(local_results_dir),
  n_folds=n_folds,
)
"

echo "=========================================="
echo "Local search job completed"
echo "=========================================="

