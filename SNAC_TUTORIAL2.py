# %% [markdown]
# # Tutorial 2: Architectural Discovery with Block-Based Search
# 
# In the previous tutorial, we fine-tuned a known architecture (MLP). But what if the best architecture for our problem is a combination of different diverse layer types?
# 
# In this notebook, we'll use SNAC-pack to **discover new architectures** by combining building blocks like **Convolutional layers**, **Attention**, and **MLPs**.
# 
# ## Hardware-Aware Block Search with K-Fold Cross-Validation
#
# We now use `rule4ml` to estimate FPGA resource usage alongside accuracy and BOPs. Optional **k-fold cross-validation** (`N_FOLDS > 1`) averages performance across multiple train/val splits, reducing sensitivity to random weight initialization. Hardware metrics are estimated once per trial since the architecture is identical across folds.
#
# Objectives:
# 1.  **Accuracy** (Maximize) — averaged across folds when `N_FOLDS > 1`
# 2.  **Computational Cost (BOPs)** (Minimize)
# 3.  **Average FPGA Resource Usage** (Minimize)
# 4.  **Clock Cycles / Latency** (Minimize)

# %%

# imports
import os
import yaml

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.tf_global_search import GlobalSearchTF
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap
from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_mnist
from utils.tf_data_preprocessing import load_and_preprocess_fashion_mnist
import seaborn as sns



# config
N_TRIALS_HYBRID = 5 # 20 # can increase this for better results
EPOCHS_HYBRID = 5 # 10
SUBSET_SIZE_HYBRID = 20000
N_FOLDS = 3  # k-fold cross-validation (1 = no CV, >1 = stratified k-fold)
RESULTS_DIR_HYBRID = "./results/tutorial2_Hybrid_Discovery"
SEARCH_SPACE_PATH = 'hybrid_search_space.yaml'
RESIZE_VAL = 16

os.makedirs(RESULTS_DIR_HYBRID, exist_ok=True)

# %%
# data loading
x_train_viz, y_train_viz, _, _ = load_and_preprocess_fashion_mnist(
    resize_val=RESIZE_VAL,
    subset_size=SUBSET_SIZE_HYBRID, 
    flatten=False, 
    one_hot=False
)

# visualize images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_viz[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train_viz[i]}")
    plt.axis('off')
plt.suptitle("Sample Fashion MNIST Images")
plt.show()

# %% [markdown]
# ## Global Search: Building with Blocks
# 
# We first define a `search_space` in a YAML file. This file provides SNAC-pack with the "Lego bricks" it can use to build and test new architectures. We'll allow it to choose between `Conv`, `MLP`, and `ConvAttn` blocks.
# 

# %%

# yaml config
search_space_yaml = """
channel_space: [4, 8, 16]
mlp_width_space: [5, 16, 42]
kernel_space: [1, 3, 5]
act_space: ["ReLU", "GELU", "LeakyRelu"]
norm_space: [null, "batch"]
block_types: ["Conv", "MLP", "None"]
num_blocks: 3
initial_img_size: 16
output_dim: 10
output_activation: "softmax"
"""

with open(SEARCH_SPACE_PATH, 'w') as f:
    f.write(search_space_yaml)
print(f"Created search space configuration file: {SEARCH_SPACE_PATH}")

# objectives (4 objectives: accuracy up, bops/resource/latency down)
OBJECTIVE_NAMES_HYBRID = ['performance_metric', 'bops', 'avg_resource', 'clock_cycles']
MAXIMIZE_FLAGS_HYBRID = [True, False, False, False]

# run the search
print("\n" + "="*50)
print("Running Part 2: Hybrid Architecture Global Search...")
print("This will take several minutes...")
print("="*50)

searcher_hybrid = GlobalSearchTF(search_space_path=SEARCH_SPACE_PATH, results_dir=RESULTS_DIR_HYBRID)

study_hybrid = searcher_hybrid.run_search(
    model_type='block',
    n_trials=N_TRIALS_HYBRID,
    epochs=EPOCHS_HYBRID,
    dataset='fashion_mnist',
    subset_size=SUBSET_SIZE_HYBRID,
    resize_val=searcher_hybrid.search_space.get('initial_img_size', 11),
    objectives=OBJECTIVE_NAMES_HYBRID,
    maximize_flags=MAXIMIZE_FLAGS_HYBRID,
    use_hardware_metrics=True,
    one_hot=True,
    n_folds=N_FOLDS,
)

print("\nGlobal Search Complete!")

# %% [markdown]
# ## Analyzing the Global Search Results
#
# The search has finished exploring different architectural combinations. With 4 objectives, we can visualize:
# 1. All **pairwise 2D Pareto fronts** (6 plots for 4 objectives)
# 2. An interactive **3D Pareto front** with the 4th objective as a color heatmap

# %%

results_df_hybrid = pd.DataFrame(searcher_hybrid.results)

if not results_df_hybrid.empty:
    # Inspect best architecture
    print("--- Best Discovered Architecture (by Accuracy) ---")
    best_trial_row = results_df_hybrid.loc[results_df_hybrid['performance_metric'].idxmax()]
    print(f"Trial Number: {best_trial_row['trial']}")
    print(f"Accuracy: {best_trial_row['performance_metric']:.4f}")
    print(f"BOPs: {best_trial_row['bops']:.2e}")
    print(f"Avg Resource: {best_trial_row['avg_resource']:.2f}%")
    print(f"Clock Cycles: {best_trial_row['clock_cycles']:.0f}")

    with open(best_trial_row['yaml_path'], 'r') as f:
        best_arch_yaml = yaml.safe_load(f)

    print("\nArchitecture components:")
    for component in best_arch_yaml['architecture']['components']:
        print(f"- Type: {component['block_type']}, Name: {component['name']}")

    # Build objective info list for plotting
    OBJECTIVE_INFO_HYBRID = list(zip(OBJECTIVE_NAMES_HYBRID, MAXIMIZE_FLAGS_HYBRID))

    # Pairwise 2D Pareto fronts (6 combinations)
    print("\n--- Visualizing Pairwise 2D Pareto Fronts ---")
    plot_pareto_fronts(results_df_hybrid, OBJECTIVE_INFO_HYBRID, save_dir=searcher_hybrid.results_dir)

    # Interactive 3D Pareto front with clock_cycles as heatmap color
    print("\n--- Generating 3D Pareto Front Heatmap ---")
    plot_3d_pareto_front_heatmap(results_df_hybrid, OBJECTIVE_INFO_HYBRID, save_dir=searcher_hybrid.results_dir)

    print(f"\nPlots saved to: {searcher_hybrid.results_dir}")
else:
    print("Hybrid search did not yield any results to analyze.")

# %% [markdown]
# ## Local Search: Combined QAT + Pruning
#
# The combined local search applies iterative magnitude pruning to each quantized
# model, revealing the full compression landscape: accuracy vs. (sparsity x bit-width).

# %%
LOCAL_SEARCH_RESULTS_DIR = os.path.join(RESULTS_DIR_HYBRID, "local_search_combined")
LOCAL_SEARCH_CONFIG_PATH = os.path.join(RESULTS_DIR_HYBRID, "local_search_settings.yaml")
ARCHITECTURE_YAML_PATH = os.path.join(RESULTS_DIR_HYBRID, "best_model_for_local_search.yaml")

local_search_settings = {
    "pruning_settings": {
        "iterations": 5,
        "epochs_per_iteration": 2,
        "pruning_rate": 0.8,
    },
    "qat_settings": {
        "epochs": 2,
        "precision_pairs": [
            {"total_bits": 16, "int_bits": 6},
            {"total_bits": 8, "int_bits": 3},
            {"total_bits": 6, "int_bits": 2},
            {"total_bits": 4, "int_bits": 1},
        ],
    },
}

with open(LOCAL_SEARCH_CONFIG_PATH, "w") as f:
    yaml.dump(local_search_settings, f)

# Load dataset (one-hot = True for classification)
x_train, y_train, x_val, y_val = load_and_preprocess_fashion_mnist(
    resize_val=RESIZE_VAL,
    subset_size=SUBSET_SIZE_HYBRID,
    flatten=False,
    one_hot=True,
)

if not os.path.exists(ARCHITECTURE_YAML_PATH):
    raise FileNotFoundError(
        f"Could not find best architecture YAML: {ARCHITECTURE_YAML_PATH}. "
        "Run global search first."
    )

combined_results_df = combined_local_search_entrypoint(
    architecture_yaml_path=ARCHITECTURE_YAML_PATH,
    local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
    dataset=(x_train, y_train, x_val, y_val),
    results_dir=LOCAL_SEARCH_RESULTS_DIR,
    n_folds=N_FOLDS,
)

# %%
if isinstance(combined_results_df, pd.DataFrame) and not combined_results_df.empty:
    # Accuracy vs Effective BOPs (one curve per precision)
    plt.figure(figsize=(12, 7))
    for prec in combined_results_df["Precision"].unique():
        subset = combined_results_df[combined_results_df["Precision"] == prec]
        plt.plot(subset["EffectiveBOPs"], subset["Accuracy"], marker="o", linewidth=2, label=prec)
    plt.xlabel("Effective BOPs")
    plt.ylabel("Accuracy")
    plt.title("Combined QAT + Pruning: Accuracy vs Effective BOPs")
    plt.legend(title="Precision")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Accuracy vs Sparsity (one curve per precision)
    plt.figure(figsize=(12, 7))
    for prec in combined_results_df["Precision"].unique():
        subset = combined_results_df[combined_results_df["Precision"] == prec]
        plt.plot(subset["Sparsity"], subset["Accuracy"], marker="o", linewidth=2, label=prec)
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy")
    plt.title("Combined QAT + Pruning: Accuracy vs Sparsity")
    plt.legend(title="Precision")
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("No combined local search results to plot.")
