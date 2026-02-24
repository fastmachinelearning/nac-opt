"""
Tutorial 2: Block-Based Architecture Discovery on Fashion-MNIST
================================================================
Demonstrates global (NSGA-II) + local (combined QAT + pruning) search using
Conv, MLP, and ConvAttn blocks.

Run from repo root:
    /opt/miniconda3/envs/snac-pack-refactor/bin/python tutorials/tutorial_2_block/tutorial_2_block.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from utils.tf_global_search import GlobalSearchTF
from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_fashion_mnist

tf.get_logger().setLevel('ERROR')

# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(Path(__file__).parent / "t2_config.yaml"))

ds_cfg = cfg["dataset"]
s_cfg = cfg["search"]
ss_cfg = cfg["search_space"]
ls_cfg = cfg["local_search"]
out_cfg = cfg["output"]

RESULTS_DIR = out_cfg["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading Fashion-MNIST dataset for visualization...")
x_train_viz, y_train_viz, _, _ = load_and_preprocess_fashion_mnist(
    resize_val=ds_cfg["resize_val"],
    subset_size=ds_cfg["subset_size"],
    flatten=False,
    one_hot=False,
)

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_viz[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train_viz[i]}")
    plt.axis('off')
plt.suptitle("Sample Fashion-MNIST Images")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fashion_mnist_samples.png"), dpi=100)
plt.close()

# ── Global Search ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("Running Tutorial 2: Block-Based Global Search on Fashion-MNIST")
print(f"  Trials: {s_cfg['n_trials']}  Epochs: {s_cfg['epochs']}  Folds: {s_cfg.get('n_folds', 1)}")
print("="*50 + "\n")

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
    subset_size=ds_cfg["subset_size"],
    resize_val=ds_cfg["resize_val"],
    objectives=obj_names,
    maximize_flags=max_flags,
    use_hardware_metrics=s_cfg["use_hardware_metrics"],
    one_hot=ds_cfg["one_hot"],
    n_folds=n_folds,
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

x_train, y_train, x_val, y_val = load_and_preprocess_fashion_mnist(
    resize_val=ds_cfg["resize_val"],
    subset_size=ds_cfg["subset_size"],
    flatten=False,
    one_hot=True,
)

if not os.path.exists(ARCH_YAML_PATH):
    raise FileNotFoundError(
        f"Could not find best architecture YAML: {ARCH_YAML_PATH}. Run global search first."
    )

combined_df = combined_local_search_entrypoint(
    architecture_yaml_path=ARCH_YAML_PATH,
    local_search_config_path=LOCAL_CONFIG_PATH,
    dataset=(x_train, y_train, x_val, y_val),
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

print("\nTutorial 2 complete.")
