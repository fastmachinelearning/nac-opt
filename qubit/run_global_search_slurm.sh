#!/bin/bash
#SBATCH --job-name=qubit_optuna
#SBATCH --account=amsc011            # Project/account name
#SBATCH --nodes=8                    # Number of nodes being requested for the job
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=1            # CPUs per task
#SBATCH --time=30:00                 # Time limit
#SBATCH --constraint=cpu             # Perlmutter CPU nodes (required at NERSC)
#SBATCH --mem=16G                    # Memory per node (increase if OOM)
#SBATCH --output=qubit_optuna_%j.out # Output file
#SBATCH --error=qubit_optuna_%j.err  # Error file
#SBATCH --qos=debug


# Example SLURM script for multi-node Optuna global search
# 
# Usage:
#   sbatch run_global_search_slurm.sh
#
# This script runs the global search across multiple nodes, with each node
# contributing trials to the same Optuna study via shared storage.

# Load modules (adjust for your system)
# module load python/3.10
# module load cuda/11.8  # If using GPUs
module load python/3.10
conda activate rule4ml_update

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
export OPTUNA_STORAGE="sqlite:///${SCRATCH:-./}/optuna/qubit_search_${SLURM_JOB_ID}.db"

# Set study name (all nodes use the same name to coordinate)
export OPTUNA_STUDY_NAME="qubit_search_job_${SLURM_JOB_ID}" #change

# Create storage directory if using SQLite (only needed if using SQLite option)
if [[ "$OPTUNA_STORAGE" == sqlite* ]]; then
    STORAGE_DIR=$(echo "$OPTUNA_STORAGE" | sed 's|sqlite:///||' | xargs dirname)
    mkdir -p "$STORAGE_DIR"
fi

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
echo "=========================================="

# Bootstrap: create DB and study once so all workers can connect without "table already exists" race
# Directions must match run_global_search (performance_metric=maximize, bops/avg_resource/cycles=minimize)
python -c "
import optuna
storage = optuna.storages.RDBStorage(url='${OPTUNA_STORAGE}')
dirs = [optuna.study.StudyDirection.MAXIMIZE] + [optuna.study.StudyDirection.MINIMIZE]*3
try:
    optuna.create_study(directions=dirs, storage=storage, study_name='${OPTUNA_STUDY_NAME}', sampler=optuna.samplers.NSGAIISampler())
    print('Created Optuna study for multi-node run.')
except optuna.exceptions.DuplicatedStudyError:
    print('Study already exists (e.g. from prior run); workers will use it.')
"

# Run the search - each srun task will load the same study and contribute trials
# Pass storage and study name on the command line so all workers use the shared DB
# --output=...%t sends each task's stdout to a separate file so you can verify both nodes ran
srun --output=qubit_optuna_%j_task_%t.out python run_global_search.py \
    --n_trials 3 \
    --epochs 2 \
    --n_folds 1 \
    --subset_size 1000000 \
    --model_type block \
    --use_hardware_metrics \
    --data_dir "${SCRATCH}/qubit_data" \
    --optuna_storage "${OPTUNA_STORAGE}" \
    --optuna_study_name "${OPTUNA_STUDY_NAME}" \
    --start_location 100 \
    --window_size 400 \
    --num_classes 2 \
    --results_dir "./results/qubit_optuna_job_${SLURM_JOB_ID}" \
    --search_space_path qubit_search_space.yaml \
    --verbose

# Merge per-worker CSVs into block_search_results.csv and create best_model_for_local_search.yaml
RESULTS_DIR="./results/qubit_optuna_job_${SLURM_JOB_ID}"
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
    df = pd.concat(dfs, ignore_index=True).sort_values('trial').drop_duplicates(subset=['trial'], keep='first')
    out = os.path.join(results_dir, 'block_search_results.csv')
    df.to_csv(out, index=False)
    print(f'Merged {len(files)} rank CSVs -> {out} ({len(df)} trials)')
    if not df.empty:
        best = df.loc[df['performance_metric'].idxmax()]
        src = best['yaml_path']
        dst = os.path.join(results_dir, 'best_model_for_local_search.yaml')
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f'Best model (trial {best[\"trial\"]}) -> {dst}')
else:
    print('No rank CSVs to merge')
"
fi

echo "=========================================="
echo "Job completed"
echo "=========================================="
