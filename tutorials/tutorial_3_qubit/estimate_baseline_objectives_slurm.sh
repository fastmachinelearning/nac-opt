#!/bin/bash
#SBATCH --job-name=qubit_baseline_obj
#SBATCH --account=amsc011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=qubit_baseline_obj_%j.out
#SBATCH --error=qubit_baseline_obj_%j.err
#SBATCH --qos=express_amsc
#
# CPU-only: use constraint=cpu, remove --gpus-per-node, set USE_TF_GPU=0.
#
# Trains the FP32 baseline head for EPOCHS (default 5) with stratified k-fold
# (default N_FOLDS=3) on train+test pool (same convention as global-search block objective),
# then writes BOPs + rule4ml metrics to JSON.
#
# Qubit preprocessing matches ``run_local_search_slurm.sh``: ``t3_config.yaml`` (or
# ``--config`` / sibling ``t3_config.yaml`` beside the script) supplies ``dataset`` fields;
# optional ``ARCH_YAML`` / ``GLOBAL_RESULTS_DIR`` sets ``metadata.dataset_window`` like local search.
#
# Usage:
#   cd tutorials/tutorial_3_qubit
#   export DATA_DIR=/path/to/dir_with_npy
#   # optional: same window as a global-search result
#   export GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<JOBID>
#   sbatch estimate_baseline_objectives_slurm.sh
#
# Env (optional overrides):
#   DATA_DIR, SCRATCH_DIR (default $SCRATCH or $PWD) for default DATA_DIR
#   EPOCHS=5  N_FOLDS=3  BATCH_SIZE=128
#   GLOBAL_RESULTS_DIR  -> default ARCH_YAML=$GLOBAL_RESULTS_DIR/best_model_for_local_search.yaml
#   ARCH_YAML  (explicit path to best_model_for_local_search.yaml or similar)
#   BOP_BIT_WIDTH=32  HLS_JSON / VALID_JSON  JSON_OUT  CONFIG_YAML=t3_config.yaml
#   USE_TF_GPU=1  EXTRA_ARGS
#   To force window / flags on the CLI instead: EXTRA_ARGS='--start_location 0 --window_size 400'

set -euo pipefail
trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} command failed: ${BASH_COMMAND}" >&2' ERR

_submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${_submit_dir}"

REPO_ROOT=""
_d="$(cd "${_submit_dir}" && pwd)"
while [[ "${_d}" != "/" ]]; do
  if [[ -d "${_d}/utils" && -f "${_d}/utils/tf_global_search.py" ]]; then
    REPO_ROOT="${_d}"
    break
  fi
  _d="$(dirname "${_d}")"
done
if [[ -z "${REPO_ROOT}" ]]; then
  echo "[ERROR] Could not locate nac-opt root (utils/tf_global_search.py). Submit from tutorials/tutorial_3_qubit." >&2
  exit 2
fi
export REPO_ROOT
echo "REPO_ROOT=${REPO_ROOT}"

TUTOR="${REPO_ROOT}/tutorials/tutorial_3_qubit"
SCRIPT_PY="${TUTOR}/estimate_baseline_objectives.py"
if [[ ! -f "${SCRIPT_PY}" ]]; then
  echo "[ERROR] Missing ${SCRIPT_PY}" >&2
  exit 2
fi

module load python/3.10
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12

if [[ -f "${HOME:-}/.bashrc" ]]; then
  source "$HOME/.bashrc"
elif [[ -f "${HOME:-}/.bash_profile" ]]; then
  source "$HOME/.bash_profile"
elif [[ -f "${HOME:-}/.bashrc.ext" ]]; then
  source "$HOME/.bashrc.ext"
elif [[ -f "${HOME:-}/.bash_profile.ext" ]]; then
  source "$HOME/.bash_profile.ext"
fi

conda activate rule4ml_update

USE_TF_GPU="${USE_TF_GPU:-1}"
if [[ "${USE_TF_GPU}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
  echo "TF will use GPU(s) on this node (USE_TF_GPU=1)."
else
  export CUDA_VISIBLE_DEVICES=""
  echo "TF forced to CPU (USE_TF_GPU=0)."
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

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
  echo "WARN: Could not locate nvvm/libdevice/libdevice*.bc; GPU XLA may fail." >&2
fi
unset _root _cuda_xla_roots _xla_cuda_data_dir _found

python -c "import tensorflow as tf; print('TF backends:', tf.config.list_logical_devices())"

SCRATCH_DIR="${SCRATCH:-$PWD}"
DATA_DIR="${DATA_DIR:-${SCRATCH_DIR}/qubit_data}"
GLOBAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR:-}"
ARCH_YAML="${ARCH_YAML:-}"
if [[ -n "${GLOBAL_RESULTS_DIR}" && -z "${ARCH_YAML}" && -f "${GLOBAL_RESULTS_DIR}/best_model_for_local_search.yaml" ]]; then
  ARCH_YAML="${GLOBAL_RESULTS_DIR}/best_model_for_local_search.yaml"
fi
EPOCHS="${EPOCHS:-5}"
N_FOLDS="${N_FOLDS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BOP_BIT_WIDTH="${BOP_BIT_WIDTH:-32}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HLS_JSON="${HLS_JSON:-${VALID_JSON:-}}"
JSON_OUT="${JSON_OUT:-./results/baseline_objectives_${SLURM_JOB_ID:-local}.json}"
CONFIG_YAML="${CONFIG_YAML:-t3_config.yaml}"

for data_file in \
  0528_X_train_0_770.npy \
  0528_y_train_0_770.npy \
  0528_X_test_0_770.npy \
  0528_y_test_0_770.npy
do
  if [[ ! -f "${DATA_DIR}/${data_file}" ]]; then
    echo "[ERROR] Missing data file: ${DATA_DIR}/${data_file}" >&2
    exit 1
  fi
done

echo "=========================================="
echo "Baseline train + BOPs + rule4ml"
echo "=========================================="
echo "JOB ID: ${SLURM_JOB_ID:-N/A}"
echo "DATA_DIR: ${DATA_DIR}"
echo "EPOCHS: ${EPOCHS}  N_FOLDS: ${N_FOLDS}  BATCH_SIZE: ${BATCH_SIZE}"
if [[ -n "${GLOBAL_RESULTS_DIR}" ]]; then
  echo "GLOBAL_RESULTS_DIR: ${GLOBAL_RESULTS_DIR}"
fi
if [[ -n "${ARCH_YAML}" ]]; then
  echo "ARCH_YAML: ${ARCH_YAML}"
fi
echo "Dataset/window: from t3_config (+ arch yaml if set); see Python log / JSON t3_config_resolved / arch_yaml_resolved"
echo "=========================================="

_PY_ARGS=(
  --data_dir "${DATA_DIR}"
  --epochs "${EPOCHS}"
  --n_folds "${N_FOLDS}"
  --batch_size "${BATCH_SIZE}"
  --bop_bit_width "${BOP_BIT_WIDTH}"
  --quiet
)

if [[ -f "${CONFIG_YAML}" ]]; then
  _PY_ARGS+=(--config "${CONFIG_YAML}")
fi

if [[ -n "${ARCH_YAML}" && -f "${ARCH_YAML}" ]]; then
  _PY_ARGS+=(--arch_yaml "${ARCH_YAML}")
fi

if [[ -n "${SUBSET_SIZE:-}" ]]; then
  _PY_ARGS+=(--subset_size "${SUBSET_SIZE}")
fi

if [[ -n "${HLS_JSON}" && -f "${HLS_JSON}" ]]; then
  _PY_ARGS+=(--hls_json "${HLS_JSON}")
fi

if [[ -n "${JSON_OUT}" ]]; then
  mkdir -p "$(dirname "${JSON_OUT}")"
  _PY_ARGS+=(--json_out "${JSON_OUT}")
fi

# shellcheck disable=SC2086
set -x
python "${SCRIPT_PY}" "${_PY_ARGS[@]}" ${EXTRA_ARGS}
set +x

echo "Done."
