# %% [markdown]
# # SNAC-Pack: Surrogate Neural Architecture Codesign Package
# 
# In this tutorial, we will create a search space, explore this space and **discover new architectures**, using resource estimation.

# %%
# !pip install -q seaborn

# %%
# imports
import os
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SNAC-pack utilities
from utils.tf_global_search import run_mlp_search
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap
from utils.tf_local_search_separated import local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_mnist
import seaborn as sns


# plotting settings and logging
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
tf.get_logger().setLevel('ERROR')
print("TensorFlow Version:", tf.__version__)

# config
N_TRIALS_MLP = 3 # number of trials # 25
EPOCHS_MLP = 2 # number of epochs # 10
SUBSET_SIZE_MLP = 10000
RESULTS_DIR_MLP = "./results/tutorial1_MLP_Hardware_Aware"

os.makedirs(RESULTS_DIR_MLP, exist_ok=True)

# %% [markdown]
# ## Dataset: MNIST
# 
# We'll use the MNIST dataset of handwritten digits. We'll begin by loading it and visualize a few examples.
# 

# %%
# load the data for visualization
x_train_viz, y_train_viz, _, _ = load_and_preprocess_mnist(
    subset_size=SUBSET_SIZE_MLP, 
    flatten=False, 
    one_hot=False
)

# first 10 images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_viz[i].squeeze(), cmap='gray')
    plt.title(f"Label: {y_train_viz[i]}")
    plt.axis('off')
plt.suptitle("Sample MNIST Images")
plt.show()

# %% [markdown]
# ## Global Search: Finding the Best Architectural Trade-offs
# 
# Now, we run the global search. We are creating the **Pareto front** of models that provide the best balance across four competing objectives:
# 
# 1.  **Accuracy** (Maximize)
# 2.  **BOPs** (Minimize)
# 3.  **FPGA Resources** (Avg. Resource %) (Minimize)
# 4.  **Latency** (Clock Cycles) (Minimize)
# 
# We enable this by setting `use_hardware_metrics=True`, which activates the `rule4ml` integration.
# 

# %%

# objectives for hardware-aware search
OBJECTIVE_NAMES_HW = ['performance_metric', 'bops', 'avg_resource', 'clock_cycles']
MAXIMIZE_FLAGS_HW = [True, False, False, False] # True = maximize, False = minimize
OBJECTIVE_INFO_HW = list(zip(OBJECTIVE_NAMES_HW, MAXIMIZE_FLAGS_HW))


# run the search
print("\n" + "="*50)
print("Running Part 1: MLP Hardware-Aware Global Search...")
print("This may take a few minutes...")
print("="*50)

study_mlp, searcher_mlp = run_mlp_search(
    results_dir=RESULTS_DIR_MLP,
    n_trials=N_TRIALS_MLP,
    epochs=EPOCHS_MLP,
    subset_size=SUBSET_SIZE_MLP,
    use_hardware_metrics=True # set to true for hardware objectives
)

print("\nGlobal Search Complete!")

# %% [markdown]
# ## Analyzing the Global Search Results
# The search is complete! The results contain the performance of every architecture tried. Let's visualize the Pareto fronts to understand the trade-offs.
# 

# %%

results_df_mlp = pd.DataFrame(searcher_mlp.results)

if not results_df_mlp.empty:
    print("\n--- Visualizing Hardware-Aware Pareto Fronts ---")
    results_df_mlp.rename(columns={'performance_metric': 'accuracy'}, inplace=True)
    OBJECTIVE_INFO_HW[0] = ('accuracy', True)
    
    plot_pareto_fronts(results_df_mlp, OBJECTIVE_INFO_HW, save_dir=searcher_mlp.results_dir)
    
    print("\n--- Generating 3D Pareto Front Heatmap ---")
    plot_3d_pareto_front_heatmap(results_df_mlp, OBJECTIVE_INFO_HW, save_dir=searcher_mlp.results_dir)
    
    print(f"\nMLP search plots saved to: {searcher_mlp.results_dir}")
else:
    print("MLP search did not yield any results.")

# %% [markdown]
# ## Local Search: Compressing the Best Model
# 
# The global search automatically saved the configuration of the best-performing model (`best_model_for_local_search.yaml`). Now, we'll take this champion architecture and make it even more efficient through **Quantization-Aware Training (QAT)** and **iterative pruning**.
# 

# %%
# config for local search
LOCAL_SEARCH_RESULTS_DIR = os.path.join(RESULTS_DIR_MLP, "local_search_separated")
LOCAL_SEARCH_CONFIG_PATH = os.path.join(RESULTS_DIR_MLP, 'local_search_settings_separated.yaml')

# setting for pruning and QAT
local_search_settings = {
    'pruning_settings': {
        'iterations': 5, # 8
        'epochs_per_iteration': 2, # 5
        'pruning_rate': 0.8, # 0.8 to prune 20% of remaining weights each iteration
    },
    'qat_settings': {
        'epochs': 2, # epochs to fine-tune quantized model # 15
        'precision_pairs': [
            {'total_bits': 16, 'int_bits': 6},
            {'total_bits': 8, 'int_bits': 3},
            {'total_bits': 6, 'int_bits': 2},
            {'total_bits': 4, 'int_bits': 1},
        ]
    }
}

# yaml settings
with open(LOCAL_SEARCH_CONFIG_PATH, 'w') as f:
    yaml.dump(local_search_settings, f)
print(f"Created separated local search configuration file: {LOCAL_SEARCH_CONFIG_PATH}")

# path to best model
ARCHITECTURE_YAML_PATH = os.path.join(RESULTS_DIR_MLP, "best_model_for_local_search.yaml")

# load dataset
x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
    resize_val=8, subset_size=SUBSET_SIZE_MLP, flatten=True, one_hot=True
) # changed flatten to True to fix dimension issue

# run local search
if os.path.exists(ARCHITECTURE_YAML_PATH):
    pruning_results_df, qat_results_df = local_search_entrypoint(
        architecture_yaml_path=ARCHITECTURE_YAML_PATH,
        local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=LOCAL_SEARCH_RESULTS_DIR
    )
else:
    print(f"ERROR: Could not find the architecture file: {ARCHITECTURE_YAML_PATH}")
    pruning_results_df, qat_results_df = pd.DataFrame(), pd.DataFrame()

# %%
# Pruning Results: accuracy vs. sparsity
if 'pruning_results_df' in locals() and not pruning_results_df.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(pruning_results_df['Sparsity'], pruning_results_df['Accuracy'], 
             marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    plt.title('Pruning Experiment: Accuracy vs. Model Sparsity', fontsize=16)
    plt.xlabel('Model Sparsity (Fraction of Zero-Value Weights)', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=max(0, pruning_results_df['Accuracy'].min() - 0.05))
    plt.tight_layout()
    plt.show()
else:
    print("Pruning experiment did not produce results to analyze.")

# QAT Results accuracy at each precision
if 'qat_results_df' in locals() and not qat_results_df.empty:
    plt.figure(figsize=(10, 6))
    
    # baseline accuracy
    baseline_acc = pruning_results_df['Accuracy'].iloc[0] if not pruning_results_df.empty else None

    # plot
    palette = sns.color_palette("viridis", n_colors=len(qat_results_df))
    sns.barplot(x='Precision', y='Accuracy', data=qat_results_df, palette=palette)
    
    if baseline_acc:
        plt.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, label=f'FP32 Baseline Acc ({baseline_acc:.4f})')
        plt.legend()

    plt.title('QAT Experiment: Final Accuracy vs. Precision', fontsize=16)
    plt.xlabel('Quantization Precision (<Total Bits, Integer Bits>)', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.ylim(bottom=max(0, qat_results_df['Accuracy'].min() - 0.05))
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("QAT experiment did not produce results to analyze.")

# %% [markdown]
# ## Synthesis
# Then taking a chosen quantized model you can perform synthesis on it with hls4ml. 


