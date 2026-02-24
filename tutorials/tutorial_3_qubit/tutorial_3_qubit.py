"""
Tutorial 3: Qubit Readout Architecture Search
==============================================
Demonstrates global (NSGA-II) + local (combined QAT + pruning) search for
qubit readout classification targeting FPGA deployment.

Run from repo root:
    /opt/miniconda3/envs/snac-pack-refactor/bin/python tutorials/tutorial_3_qubit/tutorial_3_qubit.py

Multi-node SLURM:
    cd tutorials/tutorial_3_qubit
    sbatch run_global_search_slurm.sh

Multi-node CLI (with shared storage):
    python tutorials/tutorial_3_qubit/run_global_search.py \\
        --n_trials 1000 \\
        --optuna_storage "sqlite:///./optuna.db" \\
        --optuna_study_name "qubit_search"
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from utils.tf_global_search import GlobalSearchTF
from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_qubit

tf.get_logger().setLevel('ERROR')

# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(Path(__file__).parent / "t3_config.yaml"))

ds_cfg = cfg["dataset"]
s_cfg = cfg["search"]
ss_cfg = cfg["search_space"]
ls_cfg = cfg["local_search"]
out_cfg = cfg["output"]

# Resolve data_dir relative to this tutorial's directory
_tutorial_dir = Path(__file__).resolve().parent
data_dir = str((_tutorial_dir / ds_cfg["data_dir"]).resolve())

RESULTS_DIR = out_cfg["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading qubit dataset for visualization...")
x_viz, y_viz, _, _ = load_and_preprocess_qubit(
    data_dir=data_dir,
    start_location=ds_cfg["start_location"],
    window_size=ds_cfg["window_size"],
    subset_size=min(ds_cfg.get("subset_size", 1000), 200),
    normalize=ds_cfg["normalize"],
    flatten=True,
    one_hot=False,
    num_classes=ds_cfg["num_classes"],
)

plt.figure(figsize=(12, 4))
for cls in range(ds_cfg["num_classes"]):
    idx = np.where(y_viz == cls)[0]
    if len(idx) > 0:
        plt.plot(x_viz[idx[0]], alpha=0.7, label=f"Class {cls}")
plt.title("Sample Qubit IQ Waveforms")
plt.xlabel("Time sample (I + Q concatenated)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "qubit_samples.png"), dpi=100)
plt.close()

# ── Global Search ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Running Tutorial 3: Qubit Readout Global Search")
print(f"  Trials: {s_cfg['n_trials']}  Epochs: {s_cfg['epochs']}  Folds: {s_cfg.get('n_folds', 1)}")
print("="*50 + "\n")

# Single-node mode (storage=None). For multi-node, use run_global_search.py instead.
searcher = GlobalSearchTF(
    search_space_path=ss_cfg,
    results_dir=RESULTS_DIR,
)

obj_names = s_cfg["objective_names"]
max_flags = s_cfg["maximize_flags"]
n_folds = s_cfg.get("n_folds", 1)

study = searcher.run_search(
    model_type=s_cfg["model_type"],
    n_trials=s_cfg["n_trials"],
    epochs=s_cfg["epochs"],
    dataset=ds_cfg["name"],
    subset_size=ds_cfg.get("subset_size"),
    objectives=obj_names,
    maximize_flags=max_flags,
    use_hardware_metrics=s_cfg["use_hardware_metrics"],
    one_hot=ds_cfg["one_hot"],
    n_folds=n_folds,
    data_dir=data_dir,
    start_location=ds_cfg["start_location"],
    window_size=ds_cfg["window_size"],
    num_classes=ds_cfg["num_classes"],
    normalize=ds_cfg["normalize"],
    flatten=ds_cfg["flatten"],
    storage=None,  # single-node; pass storage backend here for multi-node
)

print("\nGlobal Search Complete!")

# ── Local Search (Combined QAT + Pruning) ─────────────────────────────────────
LOCAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "local_search_combined")
LOCAL_CONFIG_PATH = os.path.join(RESULTS_DIR, "local_search_config.yaml")
ARCH_YAML_PATH = os.path.join(RESULTS_DIR, "best_model_for_local_search.yaml")

local_search_settings = {
    "pruning_settings": {
        "iterations": ls_cfg["pruning_iterations"],
        "epochs_per_iteration": ls_cfg["pruning_epochs"],
        "pruning_rate": ls_cfg["pruning_rate"],
    },
    "qat_settings": {
        "epochs": ls_cfg["qat_epochs"],
        "precision_pairs": ls_cfg["precision_pairs"],
    },
}
with open(LOCAL_CONFIG_PATH, "w") as f:
    yaml.dump(local_search_settings, f)

x_train, y_train, x_test, y_test = load_and_preprocess_qubit(
    data_dir=data_dir,
    start_location=ds_cfg["start_location"],
    window_size=ds_cfg["window_size"],
    subset_size=ds_cfg.get("subset_size"),
    normalize=ds_cfg["normalize"],
    flatten=ds_cfg["flatten"],
    one_hot=True,
    num_classes=ds_cfg["num_classes"],
)

if not os.path.exists(ARCH_YAML_PATH):
    raise FileNotFoundError(
        f"Could not find best architecture YAML: {ARCH_YAML_PATH}. Run global search first."
    )

# Use empty val arrays; combined_local_search_entrypoint uses k-fold CV internally
x_val_empty = np.empty((0, *x_train.shape[1:]), dtype=x_train.dtype)
y_val_empty = np.empty((0, *y_train.shape[1:]), dtype=y_train.dtype)

combined_df = combined_local_search_entrypoint(
    architecture_yaml_path=ARCH_YAML_PATH,
    local_search_config_path=LOCAL_CONFIG_PATH,
    dataset=(x_train, y_train, x_val_empty, y_val_empty),
    results_dir=LOCAL_RESULTS_DIR,
    n_folds=n_folds,
)

# ── Plot Local Search Results ─────────────────────────────────────────────────
if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for prec in combined_df["Precision"].unique():
        sub = combined_df[combined_df["Precision"] == prec]
        axes[0].plot(sub["EffectiveBOPs"], sub["Accuracy"], marker="o", linewidth=2, label=prec)
        axes[1].plot(sub["Sparsity"], sub["Accuracy"], marker="o", linewidth=2, label=prec)
    axes[0].set(xlabel="Effective BOPs", ylabel="Accuracy", title="Accuracy vs Effective BOPs")
    axes[0].set_xscale("log")
    axes[0].legend(title="Precision")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].set(xlabel="Sparsity", ylabel="Accuracy", title="Accuracy vs Sparsity")
    axes[1].legend(title="Precision")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_RESULTS_DIR, "combined_local_search.png"), dpi=100)
    plt.close()
else:
    print("No combined local search results to plot.")

print("\nTutorial 3 complete.")
