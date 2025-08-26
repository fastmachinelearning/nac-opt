
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
# %matplotlib inline

# Suppress TensorFlow logging for cleaner output
tf.get_logger().setLevel('ERROR')



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

