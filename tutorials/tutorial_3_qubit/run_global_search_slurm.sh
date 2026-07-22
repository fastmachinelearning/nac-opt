#!/bin/bash
#SBATCH --job-name=qubit_optuna
#SBATCH --account=amsc011            # Project/account name
#SBATCH --nodes=10                  # Number of nodes being requested for the job
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=8            # CPUs per task (same charge; speeds up each trial)
#SBATCH --time=6:00:00               # Time limit (raise if workers hit timeout during synthesis)
#SBATCH --constraint=gpu             # Perlmutter GPU nodes
#SBATCH --gpus-per-node=1            # 1 GPU per node (omit if using CPU-only constraint below)
#SBATCH --mem=64G                    # Memory per node (increase if OOM)
#SBATCH --output=qubit_optuna_%j.out # Output file
#SBATCH --error=qubit_optuna_%j.err  # Error file
#SBATCH --qos=express_amsc           # OK for CPU on this account/partition; use regular if Slurm rejects QoS
#
# CPU-only variant: constraint=cpu, remove --gpus-per-node, sbatch --export=ALL,USE_TF_GPU=0
# GPU full run (default): keep GPU SBATCH lines; leave USE_TF_GPU unset or set to 1 (see after conda activate).

set -euo pipefail
trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} command failed: ${BASH_COMMAND}" >&2' ERR


# Optuna global search (SQLite + NSGA-II). Match #SBATCH + srun lines below.
#   Smoke:    nodes=1  time=1:00:00  |  --n_trials 3  --epochs 2
#   Full run: nodes=10 time=6:00:00 |  --n_trials 50 --epochs 5  → 500 trials NSGA-II 4 objs (below)
#
# Usage:
#   cd tutorials/tutorial_3_qubit && sbatch run_global_search_slurm.sh
#
# Multi-node: each node runs --n_trials; total trials ≈ nodes × n_trials.

# Load modules (adjust for your system)
# module load python/3.10
# module load cuda/11.8  # If using GPUs
module load python/3.10
module load cudatoolkit/12.2
module load cudnn/8.9.3_cuda12

# Ensure conda functions are available in non-interactive batch shells.
# On NERSC, you may not have ~/.bashrc; older accounts sometimes use *.ext files.
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

# Training device: default USE_TF_GPU=1 uses the allocated GPU(s). Set USE_TF_GPU=0 to force CPU
# (e.g. smoke tests); on GPU nodes that still allocates GPUs you do not use — prefer CPU SBATCH instead.
USE_TF_GPU="${USE_TF_GPU:-1}"
if [[ "${USE_TF_GPU}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
  echo "TF will use GPU(s) on this node (USE_TF_GPU=1)."
else
  export CUDA_VISIBLE_DEVICES=""
  echo "TF forced to CPU (USE_TF_GPU=0)."
fi

# Help TensorFlow find CUDA/cuDNN shared libraries from the active env.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Some TF ops may trigger XLA GPU compilation; XLA requires CUDA "libdevice" bitcode under
# <cuda_root>/nvvm/libdevice/libdevice.<major>.bc. Module stacks and pip wheels vary (10 vs 11…),
# so accept any libdevice.*.bc and prefer module-provided CUDA roots when set.
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
    # .../nvvm/libdevice/libdevice.N.bc -> cuda root is two levels up from libdevice
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

# Optional: GPU smoke checks (only on GPU nodes; breaks CPU jobs due to set -e):
# nvidia-smi
# python -c "import ctypes; ctypes.CDLL('libcudnn.so.8'); print('cudnn8 ok')"
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print('TF backends:', tf.config.list_logical_devices())"

# Activate conda environment
# source activate myenv
# Or: conda activate myenv

# Configure Optuna storage backend
# Option 1: PostgreSQL (recommended for production) - DEFAULT
# Update the connection string with your database credentials
# export OPTUNA_STORAGE="postgresql://user:password@db-host:5432/optuna_db" #change
#
# Option 2: SQLite on shared filesystem (works if all nodes can access it)
# Uncomment the line below and comment out PostgreSQL if you prefer SQLite:
SCRATCH_DIR="${SCRATCH:-$PWD}"
DATA_DIR="${SCRATCH_DIR}/qubit_data"
export OPTUNA_STORAGE="sqlite:///${SCRATCH_DIR}/optuna/qubit_search_${SLURM_JOB_ID}.db"

# Set study name (all nodes use the same name to coordinate)
export OPTUNA_STUDY_NAME="qubit_search_job_${SLURM_JOB_ID}" #change

# Create storage directory if using SQLite (only needed if using SQLite option)
if [[ "$OPTUNA_STORAGE" == sqlite* ]]; then
    STORAGE_DIR=$(echo "$OPTUNA_STORAGE" | sed 's|sqlite:///||' | xargs dirname)
    mkdir -p "$STORAGE_DIR"
fi

for data_file in \
    0528_X_train_0_770.npy \
    0528_y_train_0_770.npy \
    0528_X_test_0_770.npy \
    0528_y_test_0_770.npy
do
    if [[ ! -f "${DATA_DIR}/${data_file}" ]]; then
        echo "Missing data file: ${DATA_DIR}/${data_file}" >&2
        exit 1
    fi
done

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo "Optuna Configuration"
echo "=========================================="
echo "Storage: $OPTUNA_STORAGE"
echo "Study Name: $OPTUNA_STUDY_NAME"
echo "Data Dir: $DATA_DIR"
echo "=========================================="

# Keep all run artifacts under a single job-specific directory.
RESULTS_DIR="./results/qubit_optuna_job_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Objectives (must match Optuna bootstrap dirs + run_global_search.py --objective_names / --maximize_flags)
# NSGA-II: maximize accuracy, minimize average FPGA resource %, minimize clock cycles, minimize BOPs.
# Names are code identifiers: accuracy → performance_metric; avg LUT/FF/BRAM/DSP → avg_resource.
OBJECTIVE_NAMES="performance_metric,avg_resource,clock_cycles,bops"
MAXIMIZE_FLAGS="true,false,false,false"

# Bootstrap: create DB and study once so all workers can connect without "table already exists" race
# Directions must match the number/order of objectives for this run.
python -c "
import optuna
_url = '${OPTUNA_STORAGE}'
_ekw = {'connect_args': {'timeout': 300}} if _url.startswith('sqlite') else None
storage = optuna.storages.RDBStorage(url=_url, engine_kwargs=_ekw)
names = '${OBJECTIVE_NAMES}'.split(',')
flags = [s.strip().lower() for s in '${MAXIMIZE_FLAGS}'.split(',')]
if len(names) != len(flags):
    raise SystemExit(f'objective_names ({len(names)}) and maximize_flags ({len(flags)}) must match')
dirs = [
    optuna.study.StudyDirection.MAXIMIZE if flag in ('true','1','yes','y') else optuna.study.StudyDirection.MINIMIZE
    for flag in flags
]
try:
    optuna.create_study(directions=dirs, storage=storage, study_name='${OPTUNA_STUDY_NAME}', sampler=optuna.samplers.NSGAIISampler())
    print('Created Optuna study for multi-node run.')
except optuna.exceptions.DuplicatedStudyError:
    print('Study already exists (e.g. from prior run); workers will use it.')
"

# Run the search - each srun task will load the same study and contribute trials
# Pass storage and study name on the command line so all workers use the shared DB
# --output=...%t sends each task's stdout to a separate file so you can verify both nodes ran
# Ensure worker processes inherit XLA_FLAGS / LD_LIBRARY_PATH (some sites use minimal exported env).
srun --export=ALL --output="${RESULTS_DIR}/qubit_optuna_%j_task_%t.out" python run_global_search.py \
    --n_trials 50 \
    --epochs 5 \
    --n_folds 3 \
    --subset_size 1000000 \
    --model_type block \
    --use_hardware_metrics \
    --data_dir "${DATA_DIR}" \
    --optuna_storage "${OPTUNA_STORAGE}" \
    --optuna_study_name "${OPTUNA_STUDY_NAME}" \
    --objective_names "${OBJECTIVE_NAMES}" \
    --maximize_flags "${MAXIMIZE_FLAGS}" \
    --num_classes 2 \
    --results_dir "./results/qubit_optuna_job_${SLURM_JOB_ID}" \
    --search_space_path qubit_search_space.yaml \
    --verbose

# Merge per-worker CSVs into block_search_results.csv and copy best_model_for_local_search.yaml.
# With 4 objectives, NSGA-II yields a Pareto set; copied best model is still max performance_metric only.
if [[ -d "$RESULTS_DIR" ]]; then
  python -c "
import pandas as pd
import glob
import os
import shutil
results_dir = '${RESULTS_DIR}'
pattern = os.path.join(results_dir, 'block_search_results_rank*.csv')
files = sorted(glob.glob(pattern))
if files:
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        print('No trial rows to merge yet')
    elif 'trial' not in df.columns:
        print(\"Skip merge: rank CSV missing 'trial' column (likely all trials failed)\")
    else:
        df = df.sort_values('trial').drop_duplicates(subset=['trial'], keep='first')
        out = os.path.join(results_dir, 'block_search_results.csv')
        df.to_csv(out, index=False)
        print(f'Merged {len(files)} rank CSVs -> {out} ({len(df)} trials)')
        if not df.empty and 'performance_metric' in df.columns:
            best = df.loc[df['performance_metric'].idxmax()]
            src = best.get('yaml_path')
            dst = os.path.join(results_dir, 'best_model_for_local_search.yaml')
            if src and os.path.exists(src):
                shutil.copy(src, dst)
                print(f'Best model (trial {best[\"trial\"]}) -> {dst}')
else:
    print('No rank CSVs to merge')
"
fi

# Copy SLURM top-level logs into the same results directory for convenience.
for main_log in "qubit_optuna_${SLURM_JOB_ID}.out" "qubit_optuna_${SLURM_JOB_ID}.err"; do
    if [[ -f "$main_log" ]]; then
        cp -f "$main_log" "${RESULTS_DIR}/"
    fi
done

echo "=========================================="
echo "Job completed"
echo "=========================================="
