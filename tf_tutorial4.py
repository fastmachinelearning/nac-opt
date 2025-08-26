# %% [markdown]
# **Part 1: Hardware-Aware MLP Search:** We will first run a search on a single, well-defined architecture type (MLP). This allows us to use accurate, model-based hardware performance estimators (`rule4ml`) and visualize the complex trade-offs between accuracy, BOPs, and hardware resources.
# 

# %%

import os
import yaml
import tensorflow as tf
import pandas as pd

# Import the necessary search functions and visualization tools from the library
# Note: We import from two different modules to showcase both search types
from utils.tf_global_search import run_mlp_search
from utils.tf_global_search4 import GlobalSearchTF
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap

# Use matplotlib for inline plotting in notebooks
%matplotlib inline

# Suppress TensorFlow logging for cleaner output
tf.get_logger().setLevel('ERROR')

# %%
# ## Part 1: MLP-Only Hardware-Aware Search
#
# In this section, we'll perform a search over standard MLP architectures. Because the structure is well-defined, we can use hardware-aware estimators to get realistic performance metrics for FPGAs. This allows for a 4-objective optimization.

# %%
# --- Configuration for Part 1 ---
N_TRIALS_MLP = 15
EPOCHS_MLP = 20
SUBSET_SIZE_MLP = 10000
RESULTS_DIR_MLP = "./results/mlp_hw_search_tutorial"
USE_HARDWARE_METRICS = True # Enable hardware-aware metrics

# --- Objectives for Hardware-Aware Search ---
OBJECTIVE_NAMES_HW = ['accuracy', 'bops', 'avg_resource', 'clock_cycles']
MAXIMIZE_FLAGS_HW = [True, False, False, False] # True = maximize, False = minimize
OBJECTIVE_INFO_HW = list(zip(OBJECTIVE_NAMES_HW, MAXIMIZE_FLAGS_HW))

os.makedirs(RESULTS_DIR_MLP, exist_ok=True)


# %%
# --- Run the MLP Hardware-Aware Search ---
print("\n" + "="*50)
print("Running Part 1: MLP Hardware-Aware Search...")
print("="*50)

# Use the convenience function for a standard MLP search
study_mlp, searcher_mlp = run_mlp_search(
    results_dir=RESULTS_DIR_MLP,
    n_trials=N_TRIALS_MLP,
    epochs=EPOCHS_MLP,
    subset_size=SUBSET_SIZE_MLP,
    use_hardware_metrics=USE_HARDWARE_METRICS
)

# %%
# --- Analyze Results for Part 1 ---
print("\n" + "="*50)
print("ANALYZING MLP HARDWARE-AWARE RESULTS")
print("="*50)

results_df_mlp = searcher_mlp.get_results_dataframe()

if not results_df_mlp.empty:
    print("\nMLP Search Summary:")
    print(f"Total trials: {len(results_df_mlp)}")
    print(f"Best Accuracy: {results_df_mlp['accuracy'].max():.4f}")
    
    print("\n--- Visualizing Hardware-Aware Pareto Fronts ---")
    plot_pareto_fronts(results_df_mlp, OBJECTIVE_INFO_HW, save_dir=searcher_mlp.results_dir)
    
    if len(OBJECTIVE_NAMES_HW) >= 4:
        print("\n--- Generating 3D Pareto Front Heatmap ---")
        plot_3d_pareto_front_heatmap(results_df_mlp, OBJECTIVE_INFO_HW, save_dir=searcher_mlp.results_dir)
    
    print(f"\nMLP search plots saved to: {searcher_mlp.results_dir}")
else:
    print("MLP search did not yield any results.")



# %%
# ## Part 2: Hybrid Architecture Search
#
# Now we'll use the block-based search to find novel hybrid architectures. Since hardware estimators may not be accurate for these arbitrary structures, we will focus our optimization on two key objectives: **performance (accuracy)** and **computational cost (BOPs)**.

# %%
# --- Configuration for Part 2 ---
N_TRIALS_HYBRID = 30
EPOCHS_HYBRID = 15
SUBSET_SIZE_HYBRID = 10000
RESULTS_DIR_HYBRID = "./results/hybrid_search_tutorial"
SEARCH_SPACE_PATH = 'hybrid_search_space.yaml'

# --- Objectives for Hybrid Search (Performance vs. Cost) ---
OBJECTIVE_NAMES_HYBRID = ['accuracy', 'bops']
MAXIMIZE_FLAGS_HYBRID = [True, False]
OBJECTIVE_INFO_HYBRID = list(zip(OBJECTIVE_NAMES_HYBRID, MAXIMIZE_FLAGS_HYBRID))

os.makedirs(RESULTS_DIR_HYBRID, exist_ok=True)

# %%
# --- Create the YAML configuration file for the hybrid search ---
search_space_yaml = """
search_spaces:
  channel_space: [8, 16, 32]
  mlp_width_space: [32, 64, 128]
  kernel_space: [1, 3, 5]
  act_space: ["ReLU", "GELU"]
  norm_space: [null, "batch"]
  block_types: ["Conv", "MLP", "None", "ConvAttn"]
  # block_types: ["Conv", "MLP", "None"]
hyperparameters:
  num_blocks: 4
  initial_img_size: 11
  output_dim: 10
"""
with open(SEARCH_SPACE_PATH, 'w') as f:
    f.write(search_space_yaml)
print(f"Created search space configuration file: {SEARCH_SPACE_PATH}")


# %%
# --- Run the Hybrid Search (Simple Method) ---
print("\n" + "="*50)
print("Running Part 2: Hybrid Architecture Search...")
print("="*50)

with open(SEARCH_SPACE_PATH, 'r') as f:
    config = yaml.safe_load(f)
search_space_simple = config.get('search_spaces', {})
search_space_simple.update(config.get('hyperparameters', {}))

searcher_simple = GlobalSearchTF(
    results_dir=RESULTS_DIR_HYBRID + "_simple"
)
searcher_simple.search_space = search_space_simple

study_simple = searcher_simple.run_search(
    model_type='block',
    n_trials=N_TRIALS_HYBRID,
    epochs=EPOCHS_HYBRID,
    dataset='mnist',
    subset_size=SUBSET_SIZE_HYBRID,
    resize_val=searcher_simple.search_space.get('initial_img_size', 11),
    objectives=OBJECTIVE_NAMES_HYBRID,
    maximize_flags=MAXIMIZE_FLAGS_HYBRID,
    verbose=True
)

# %%
# ## Analyzing the Hybrid Search Results
#
# For the hybrid search, we analyze the direct trade-off between model performance and its computational cost (BOPs). The Pareto front here will show us which architectures give the best accuracy for a given computational budget.

# %%
print("\n" + "="*50)
print("ANALYZING AND VISUALIZING HYBRID RESULTS")
print("="*50)

results_df_hybrid = pd.DataFrame(searcher_simple.results)
print(results_df_hybrid.head())

if not results_df_hybrid.empty:
    # Display basic statistics
    print("\nHybrid Search Results Summary:")
    print(f"Total trials completed: {len(results_df_hybrid)}")
    print(f"Best Accuracy: {results_df_hybrid['accuracy'].max():.4f}")
    print(f"Lowest BOPs: {results_df_hybrid['bops'].min()}")

    print("\nTop 5 Hybrid Architectures by Performance:")
    print(results_df_hybrid.sort_values('accuracy', ascending=False).head())

    # --- Visualize the Pareto Fronts (Accuracy vs BOPs) ---
    print("\n--- Generating Pareto Front Plots for Hybrid Search ---")

    # Generate and display the 2D Pareto front plots
    plot_pareto_fronts(results_df_hybrid, OBJECTIVE_INFO_HYBRID, save_dir=searcher_simple.results_dir)

    print(f"\nAll plots and results saved to: {searcher_simple.results_dir}")
else:
    print("Hybrid search did not yield any results to analyze.")



# %%



