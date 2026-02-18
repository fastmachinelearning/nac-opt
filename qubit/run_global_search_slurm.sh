#!/bin/bash
#SBATCH --job-name=qubit_optuna
#SBATCH --nodes=4                    # Number of nodes being requested for the job
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=8            # CPUs per task
#SBATCH --gres=gpu:nvidia_l40s:1 # check
#SBATCH --partition=gpu # check
#SBATCH --time=48:00:00              # Time limit
#SBATCH --mem=32G                    # Memory per node
#SBATCH --output=qubit_optuna_%j.out  # Output file
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

# Activate conda environment
# source activate myenv
# Or: conda activate myenv

# Configure Optuna storage backend
# Option 1: PostgreSQL (recommended for production) - DEFAULT
# Update the connection string with your database credentials
export OPTUNA_STORAGE="postgresql://user:password@db-host:5432/optuna_db" #change
#
# Option 2: SQLite on shared filesystem (works if all nodes can access it)
# Uncomment the line below and comment out PostgreSQL if you prefer SQLite:
# export OPTUNA_STORAGE="sqlite:///${SCRATCH:-./}/optuna/qubit_search_${SLURM_JOB_ID}.db"

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

# Run the search - each srun task will load the same study and contribute trials
# The script automatically detects SLURM environment and configures accordingly
srun python run_global_search.py \
    --n_trials 1000 \
    --epochs 10 \
    --n_folds 3 \
    --subset_size 1000000 \
    --model_type block \
    --use_hardware_metrics \
    --data_dir ../qubit/data \
    --start_location 100 \
    --window_size 400 \
    --num_classes 2 \
    --results_dir "./results/qubit_optuna_job_${SLURM_JOB_ID}" \
    --search_space_path qubit_search_space.yaml \
    --verbose

echo "=========================================="
echo "Job completed"
echo "=========================================="
