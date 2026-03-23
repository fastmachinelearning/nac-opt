"""
Tutorial 1: MLP Search on MNIST
================================
Demonstrates global (NSGA-II) + local (QAT + pruning) search for MLP architectures
targeting FPGA deployment.

Run from repo root:
    /opt/miniconda3/envs/snac-pack-refactor/bin/python tutorials/tutorial_1_mlp/tutorial_1_mlp.py
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
from utils.tf_local_search_separated import local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_mnist

tf.get_logger().setLevel('ERROR')

# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(Path(__file__).parent / "t1_config.yaml"))

ds_cfg = cfg["dataset"]
s_cfg = cfg["search"]
ss_cfg = cfg["search_space"]
ls_cfg = cfg["local_search"]
out_cfg = cfg["output"]

RESULTS_DIR = out_cfg["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading MNIST dataset for visualization...")
x_train_viz, y_train_viz, _, _ = load_and_preprocess_mnist(
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
plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mnist_samples.png"), dpi=100)
plt.close()

# ── Global Search ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"Running Tutorial 1: MLP Global Search on MNIST")
print(f"  Trials: {s_cfg['n_trials']}  Epochs: {s_cfg['epochs']}")
print("="*50 + "\n")

searcher = GlobalSearchTF(
    search_space_path=ss_cfg,
    results_dir=RESULTS_DIR,
)

obj_names = s_cfg["objective_names"]
max_flags = s_cfg["maximize_flags"]

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
)

print("\nGlobal Search Complete!")

# ── Local Search ──────────────────────────────────────────────────────────────
LOCAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "local_search_separated")
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

x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
    resize_val=ds_cfg["resize_val"],
    subset_size=ds_cfg["subset_size"],
    flatten=True,
    one_hot=True,
)

if os.path.exists(ARCH_YAML_PATH):
    pruning_df, qat_df = local_search_entrypoint(
        architecture_yaml_path=ARCH_YAML_PATH,
        local_search_config_path=LOCAL_CONFIG_PATH,
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=LOCAL_RESULTS_DIR,
    )
else:
    print(f"ERROR: Architecture YAML not found: {ARCH_YAML_PATH}")
    pruning_df, qat_df = pd.DataFrame(), pd.DataFrame()

# ── Plot Local Search Results ─────────────────────────────────────────────────
if not pruning_df.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_df['Sparsity'], pruning_df['Accuracy'], marker='o', linewidth=2)
    plt.title('Pruning: Accuracy vs. Sparsity')
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_RESULTS_DIR, "pruning_curve.png"), dpi=100)
    plt.close()

if not qat_df.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(qat_df)), qat_df['Accuracy'], tick_label=qat_df['Precision'])
    plt.title('QAT: Accuracy vs. Precision')
    plt.xlabel('Precision')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(LOCAL_RESULTS_DIR, "qat_results.png"), dpi=100)
    plt.close()

print("\nTutorial 1 complete.")
