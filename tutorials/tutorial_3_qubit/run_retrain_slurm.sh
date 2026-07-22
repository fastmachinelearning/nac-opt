#!/bin/bash
#SBATCH --job-name=qubit_retrain
#SBATCH --account=amsc011
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=qubit_retrain_%j.out
#SBATCH --error=qubit_retrain_%j.err
#SBATCH --qos=express_amsc

set -euo pipefail
trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} command failed: ${BASH_COMMAND}" >&2' ERR

# Usage:
#   cd tutorials/tutorial_3_qubit
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<ID>,ARCH_YAML_DIR=./results/qubit_optuna_job_<ID>/models_to_train,EPOCHS=20 run_retrain_slurm.sh
#
# Or single YAML:
#   sbatch --export=ALL,GLOBAL_RESULTS_DIR=./results/qubit_optuna_job_<ID>,ARCH_YAML=./results/.../best_low_lut_min_acc.yaml,EPOCHS=20 run_retrain_slurm.sh

_submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${_submit_dir}"

# Find nac-opt repo root
REPO_ROOT=""
_d="$(pwd)"
while [[ "${_d}" != "/" ]]; do
  if [[ -d "${_d}/utils" && -f "${_d}/utils/tf_global_search.py" ]]; then
    REPO_ROOT="${_d}"
    break
  fi
  _d="$(dirname "${_d}")"
done
if [[ -z "${REPO_ROOT}" ]]; then
  echo "[ERROR] Could not locate nac-opt repo root. Submit from tutorials/tutorial_3_qubit." >&2
  exit 2
fi
export REPO_ROOT

module load python/3.10
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12

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

USE_TF_GPU="${USE_TF_GPU:-1}"
if [[ "${USE_TF_GPU}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
else
  export CUDA_VISIBLE_DEVICES=""
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Help XLA find CUDA libdevice bitcode (mirrors global/local search scripts).
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
  # Fallback: disable XLA auto-JIT to avoid libdevice failures.
  export TF_XLA_FLAGS="--tf_xla_auto_jit=0 ${TF_XLA_FLAGS:-}"
  echo "WARN: Could not locate nvvm/libdevice/libdevice*.bc; disabling XLA auto-JIT (TF_XLA_FLAGS=${TF_XLA_FLAGS})." >&2
fi
unset _root _cuda_xla_roots _xla_cuda_data_dir _found

GLOBAL_RESULTS_DIR="${GLOBAL_RESULTS_DIR:-}"
if [[ -z "${GLOBAL_RESULTS_DIR}" ]]; then
  echo "[ERROR] GLOBAL_RESULTS_DIR is not set." >&2
  exit 2
fi

ARCH_YAML="${ARCH_YAML:-}"
ARCH_YAML_DIR="${ARCH_YAML_DIR:-}"
if [[ -z "${ARCH_YAML}" && -z "${ARCH_YAML_DIR}" ]]; then
  echo "[ERROR] Set ARCH_YAML or ARCH_YAML_DIR." >&2
  exit 2
fi

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

SCRATCH_DIR="${SCRATCH:-$PWD}"
DATA_DIR="${DATA_DIR:-${SCRATCH_DIR}/qubit_data}"

OUT_ROOT="${OUT_ROOT:-${GLOBAL_RESULTS_DIR}/models_and_weights}"
mkdir -p "${OUT_ROOT}"

echo "GLOBAL_RESULTS_DIR=${GLOBAL_RESULTS_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE} LEARNING_RATE=${LEARNING_RATE}"

_run_one () {
  local yaml_path="$1"
  local stem
  stem="$(basename "${yaml_path}")"
  stem="${stem%.yaml}"
  local out_dir="${OUT_ROOT}/${stem}"
  mkdir -p "${out_dir}"
  echo "=== Training ${yaml_path} -> ${out_dir}"
  python "${REPO_ROOT}/tutorials/tutorial_3_qubit/retrain_from_arch_yaml.py" \
    --arch-yaml "${yaml_path}" \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --results-dir "${out_dir}"
}

if [[ -n "${ARCH_YAML}" ]]; then
  _run_one "${ARCH_YAML}"
else
  shopt -s nullglob
  yamls=( "${ARCH_YAML_DIR}"/*.yaml )
  if [[ ${#yamls[@]} -eq 0 ]]; then
    echo "[ERROR] No YAML files found in ARCH_YAML_DIR=${ARCH_YAML_DIR}" >&2
    exit 2
  fi
  for y in "${yamls[@]}"; do
    _run_one "${y}"
  done
fi

echo "Done."

