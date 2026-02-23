#!/usr/bin/env python3
"""
Command-line script to run global search for qubit readout model.

Usage (Single-node):
    python run_global_search.py --n_trials 10 --epochs 5
    python run_global_search.py --config config.yaml
    python run_global_search.py --help

Usage (Multi-node with Optuna):
    # Set storage backend (PostgreSQL recommended)
    export OPTUNA_STORAGE="postgresql://user:pass@host:5432/optuna_db"
    export OPTUNA_STUDY_NAME="qubit_experiment_v1"
    
    # Run on each node (they'll coordinate via the shared study)
    python run_global_search.py --n_trials 1000 --epochs 10
    
    # Or use command-line args
    python run_global_search.py \\
        --n_trials 1000 \\
        --optuna_storage "sqlite:///./optuna.db" \\
        --optuna_study_name "qubit_search"
    
    # For SLURM, use the provided script:
    sbatch run_global_search_slurm.sh

If you see a libstdc++/CXXABI import error, the script will try to re-run itself
with LD_LIBRARY_PATH set; you can also run: ./run_global_search.sh [args]
"""

import os
import sys

# Re-exec with conda lib in LD_LIBRARY_PATH so the new process loads the right libs
# (must happen before any imports that touch sqlite/optuna)
if os.environ.get("RUN_GLOBAL_SEARCH_LD_FIX") != "1" and os.environ.get("CONDA_PREFIX"):
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    if os.path.isdir(conda_lib):
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if conda_lib not in current.split(os.pathsep):
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = conda_lib + (os.pathsep + current if current else "")
            env["RUN_GLOBAL_SEARCH_LD_FIX"] = "1"
            script_path = os.path.abspath(os.path.realpath(__file__))
            os.execve(sys.executable, [sys.executable, script_path] + sys.argv[1:], env)

from pathlib import Path
import argparse
import yaml
import json

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Try to import GlobalSearchTF with helpful error message
try:
    from utils.tf_global_search import GlobalSearchTF
except ImportError as e:
    if 'sqlite3' in str(e) or 'libstdc++' in str(e) or 'CXXABI' in str(e):
        print("\n" + "=" * 70)
        print("ERROR: Library compatibility issue detected")
        print("=" * 70)
        print("This appears to be a system library compatibility issue.")
        print("\nTry one of these solutions:")
        print("\n1. Update libstdc++ in your conda environment:")
        print("   conda install -c conda-forge libstdcxx-ng")
        print("\n2. Use the wrapper script:")
        print("   ./run_global_search.sh [arguments]")
        print("\n3. Set LD_LIBRARY_PATH manually and rerun:")
        if 'CONDA_PREFIX' in os.environ:
            print(f"   export LD_LIBRARY_PATH={os.environ['CONDA_PREFIX']}/lib:$LD_LIBRARY_PATH")
        print("   python run_global_search.py")
        print("\n4. Or use a different conda environment with compatible libraries")
        print("\nNote: The script re-ran with LD_LIBRARY_PATH set but the error persisted.")
        print("=" * 70 + "\n")
    raise


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run global search for qubit readout model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Search parameters
    parser.add_argument('--n_trials', type=int, default=4,
                        help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs per trial')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='Number of folds for cross-validation (1 = no CV)')
    
    # Dataset parameters
    parser.add_argument('--subset_size', type=int, default=1000000,
                        help='Subset size for training data')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.environ.get("SCRATCH", ""), "qubit_data") if os.environ.get("SCRATCH") else "../qubit/data",
                        help='Directory containing qubit data files')
    parser.add_argument('--start_location', type=int, default=100,
                        help='Start location for data windowing')
    parser.add_argument('--window_size', type=int, default=400,
                        help='Window size for data')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    
    # Paths
    parser.add_argument('--search_space_path', type=str, default='qubit_search_space.yaml',
                        help='Path to search space YAML file')
    parser.add_argument('--results_dir', type=str, default='./results/node_test_6',
                        help='Directory to save results')
    
    # Objectives
    parser.add_argument('--use_hardware_metrics', action='store_true', default=True,
                        help='Use hardware-aware metrics')
    parser.add_argument('--no_hardware_metrics', dest='use_hardware_metrics', action='store_false',
                        help='Disable hardware-aware metrics')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='block', choices=['block', 'mlp'],
                        help='Model type to search')
    parser.add_argument('--one_hot', action='store_true', default=True,
                        help='Use one-hot encoding')
    parser.add_argument('--no_one_hot', dest='one_hot', action='store_false',
                        help='Disable one-hot encoding')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalize input data')
    parser.add_argument('--flatten', action='store_true', default=True,
                        help='Flatten input data')
    parser.add_argument('--no_flatten', dest='flatten', action='store_false',
                        help='Do not flatten input data')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        help='Quiet mode')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides command-line args)')
    
    # Optuna multi-node support
    parser.add_argument('--optuna_storage', type=str, default=None,
                        help='Optuna storage URL (e.g., postgresql://user:pass@host:5432/db or sqlite:///path/to/db.db). '
                             'If not provided, uses in-memory storage (single-node). '
                             'Can also be set via OPTUNA_STORAGE environment variable.')
    parser.add_argument('--optuna_study_name', type=str, default=None,
                        help='Optuna study name for multi-node coordination. '
                             'If not provided, auto-generated from job ID or timestamp. '
                             'Can also be set via OPTUNA_STUDY_NAME environment variable.')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_optuna_storage(args):
    """Set up Optuna storage backend for multi-node support."""
    import optuna
    
    # Get storage URL from args or environment
    storage_url = args.optuna_storage or os.environ.get('OPTUNA_STORAGE')
    
    if storage_url is None:
        # No storage configured - single-node mode
        return None, None
    
    # Create storage backend
    try:
        storage = optuna.storages.RDBStorage(url=storage_url)
    except Exception as e:
        print(f"Warning: Failed to create Optuna storage backend: {e}")
        print("Falling back to in-memory storage (single-node mode)")
        return None, None
    
    # Get study name from args, environment, or generate from SLURM job ID
    study_name = args.optuna_study_name or os.environ.get('OPTUNA_STUDY_NAME')
    
    if study_name is None:
        # Auto-generate study name
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            study_name = f"qubit_search_job_{job_id}"
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"qubit_search_{timestamp}"
    
    return storage, study_name


def run_search(args):
    """Run the global search with given arguments."""
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up Optuna storage for multi-node support
    storage, study_name = setup_optuna_storage(args)
    storage_url = args.optuna_storage or os.environ.get('OPTUNA_STORAGE', 'N/A')
    
    # Set up objectives
    if args.use_hardware_metrics:
        objective_names = ["performance_metric", "bops", "avg_resource", "clock_cycles"]
        maximize_flags = [True, False, False, False]
    else:
        objective_names = ["performance_metric", "bops"]
        maximize_flags = [True, False]
    
    # Detect SLURM environment
    is_slurm = 'SLURM_JOB_ID' in os.environ
    node_info = ""
    if is_slurm:
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
        node_list = os.environ.get('SLURM_NODELIST', 'unknown')
        node_info = f" (SLURM Job: {job_id}"
        if array_task_id:
            node_info += f", Task: {array_task_id}"
        node_info += f", Node: {node_list})"
    
    # Print configuration
    print("\n" + "=" * 70)
    print("Global Search Configuration")
    print("=" * 70)
    print(f"  Model Type: {args.model_type}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Subset Size: {args.subset_size}")
    print(f"  Hardware Metrics: {args.use_hardware_metrics}")
    print(f"  Objectives: {objective_names}")
    print(f"  Results Dir: {args.results_dir}")
    print(f"  Search Space: {args.search_space_path}")
    if storage is not None:
        print(f"  Optuna Storage: {storage_url}")
        print(f"  Optuna Study: {study_name}")
        print(f"  Mode: Multi-node (distributed)")
    else:
        print(f"  Mode: Single-node (in-memory)")
    if is_slurm:
        print(f"  SLURM Info: {node_info}")
        # Per-worker line: confirms this process is on one of the allocated nodes
        import socket
        proc_id = os.environ.get("SLURM_PROCID", os.environ.get("SLURM_LOCALID", "?"))
        print(f"  This worker: hostname={socket.gethostname()}, task_id={proc_id}")
    print("=" * 70 + "\n")
    
    # Initialize searcher
    searcher = GlobalSearchTF(
        search_space_path=args.search_space_path,
        results_dir=args.results_dir
    )
    
    # Run search
    print("Starting global search...")
    study = searcher.run_search(
        model_type=args.model_type,
        n_trials=args.n_trials,
        epochs=args.epochs,
        dataset='qubit',
        subset_size=args.subset_size,
        objectives=objective_names,
        maximize_flags=maximize_flags,
        use_hardware_metrics=args.use_hardware_metrics,
        one_hot=args.one_hot,
        n_folds=args.n_folds,
        verbose=args.verbose,
        data_dir=args.data_dir,
        start_location=args.start_location,
        window_size=args.window_size,
        num_classes=args.num_classes,
        normalize=args.normalize,
        flatten=args.flatten,
        storage=storage,  # Pass storage backend
        study_name=study_name,  # Pass study name
    )
    
    print("\n" + "=" * 70)
    print("Global Search Complete!")
    print("=" * 70)
    
    # Print summary
    import pandas as pd
    results_df = pd.DataFrame(searcher.results)
    
    if not results_df.empty:
        print(f"\nTotal trials completed: {len(results_df)}")
        print(f"\nBest Performance: {results_df['performance_metric'].max():.4f}")
        print(f"Best Trial: {results_df.loc[results_df['performance_metric'].idxmax(), 'trial']}")
        
        if args.use_hardware_metrics:
            print(f"\nBest BOPs: {results_df['bops'].min():,.0f}")
            print(f"Best Avg Resource: {results_df['avg_resource'].min():.2f}%")
            print(f"Best Clock Cycles: {results_df['clock_cycles'].min():.2f}")
        
        print(f"\nResults saved to: {args.results_dir}")
        print(f"  - CSV: {args.results_dir}/{args.model_type}_search_results.csv")
        print(f"  - Best model: {args.results_dir}/best_model_for_local_search.yaml")
    else:
        print("\nNo successful trials completed.")
    
    return study, searcher


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config file if provided (overrides command-line args)
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Validate paths
    if not os.path.exists(args.search_space_path):
        print(f"Warning: Search space file not found: {args.search_space_path}")
        print("Using default search space.")
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run search
    try:
        study, searcher = run_search(args)
        print("\n✓ Search completed successfully!")
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during search: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
