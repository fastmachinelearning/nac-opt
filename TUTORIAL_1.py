# %%
# Basic imports and setup
import os
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import SNAC-pack utilities
from utils.tf_global_search5 import run_mlp_search
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap
from utils.tf_local_search1 import local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_mnist

np.random.seed(42)
tf.random.set_seed(42)

# Plotting settings and TF logging
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
tf.get_logger().setLevel('ERROR')
print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
N_TRIALS_MLP = 15 # Note: Increase for a real search, 15 is for a quick demo
EPOCHS_MLP = 10
SUBSET_SIZE_MLP = 10000
RESULTS_DIR_MLP = "./results/tutorial1_MLP_Hardware_Aware"

os.makedirs(RESULTS_DIR_MLP, exist_ok=True)

# %% [markdown]
# ## The Dataset: MNIST
# 
# We'll use the classic MNIST dataset of handwritten digits. First, let's load it and visualize a few examples to see what we're working with.
# 

# %%

# Load the data for visualization (un-flattened)
x_train_viz, y_train_viz, _, _ = load_and_preprocess_mnist(
    subset_size=SUBSET_SIZE_MLP, 
    flatten=False, 
    one_hot=False
)

# Visualize the first 10 images
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
# Now, we run the global search. We are not looking for a single "best" model, but the **Pareto optimal front** of models that offer the best balance across four competing objectives:
# 
# 1.  **Accuracy** (Maximize)
# 2.  **BOPs** (Minimize)
# 3.  **FPGA Resources** (Avg. Resource %) (Minimize)
# 4.  **Latency** (Clock Cycles) (Minimize)
# 
# We enable this by setting `use_hardware_metrics=True`, which activates the `rule4ml` integration.
# 

# %%

# --- Objectives for Hardware-Aware Search ---
OBJECTIVE_NAMES_HW = ['performance_metric', 'bops', 'avg_resource', 'clock_cycles']
MAXIMIZE_FLAGS_HW = [True, False, False, False] # True = maximize, False = minimize
OBJECTIVE_INFO_HW = list(zip(OBJECTIVE_NAMES_HW, MAXIMIZE_FLAGS_HW))


# --- Run the MLP Hardware-Aware Search ---
print("\n" + "="*50)
print("Running Part 1: MLP Hardware-Aware Global Search...")
print("This may take a few minutes...")
print("="*50)

study_mlp, searcher_mlp = run_mlp_search(
    results_dir=RESULTS_DIR_MLP,
    n_trials=N_TRIALS_MLP,
    epochs=EPOCHS_MLP,
    subset_size=SUBSET_SIZE_MLP,
    use_hardware_metrics=True # This is the key for hardware co-design!
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
    # Rename 'performance_metric' to 'accuracy' for clearer plot labels
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

# --- Configuration for Local Search ---
LOCAL_SEARCH_RESULTS_DIR = os.path.join(RESULTS_DIR_MLP, "local_search")
LOCAL_SEARCH_CONFIG_PATH = os.path.join(RESULTS_DIR_MLP, 'local_search_settings.yaml')

# Define settings for QAT (precisions) and pruning
local_search_settings = {
    'precision_pairs': [
        {'total_bits': 16, 'int_bits': 6},
        {'total_bits': 8, 'int_bits': 3},
        {'total_bits': 4, 'int_bits': 1},
    ],
    'pruning_iterations': 5,
    'epochs_per_iteration': 8,
    'pruning_rate': 0.8, # Prune 20% of remaining weights each iteration (1 - 0.8)
}

# Write the settings to a YAML file
with open(LOCAL_SEARCH_CONFIG_PATH, 'w') as f:
    yaml.dump(local_search_settings, f)
print(f"Created local search configuration file: {LOCAL_SEARCH_CONFIG_PATH}")

# Path to the best model found by the global search
ARCHITECTURE_YAML_PATH = os.path.join(RESULTS_DIR_MLP, "best_model_for_local_search.yaml")


# --- Load Dataset for Local Search ---
resize_val = 8 
x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
    resize_val=resize_val, 
    subset_size=SUBSET_SIZE_MLP, 
    flatten=False,
    one_hot=True
)

# --- Run the Local Search ---
if os.path.exists(ARCHITECTURE_YAML_PATH):
    local_search_df = local_search_entrypoint(
        architecture_yaml_path=ARCHITECTURE_YAML_PATH,
        local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=LOCAL_SEARCH_RESULTS_DIR
    )
else:
    print(f"ERROR: Could not find the architecture file: {ARCHITECTURE_YAML_PATH}")
    local_search_df = pd.DataFrame()

# %% [markdown]
# ## Analyzing the Local Search Results
# 
# The local search has finished. Let's plot the results from the pruning log to see how accuracy holds up as we increase sparsity for different quantization levels.
# 

# %%
# if not local_search_df.empty:
#     plt.figure(figsize=(10, 6))
    
#     # Define distinct colors and markers
#     colors = ['blue', 'red']
#     markers = ['o', 's']  # circle and square
    
#     precisions = local_search_df['Precision'].unique()
#     # print(f"Found precisions: {precisions}")
#     # print(f"Data shape: {local_search_df.shape}")
    
#     for i, prec in enumerate(precisions):
#         subset = local_search_df[local_search_df['Precision'] == prec]
#         print(f"Precision {prec}: {len(subset)} data points")
#         print(subset[['Iteration', 'Sparsity', 'Accuracy']].to_string())
        
#         plt.plot(subset['Sparsity'], subset['Accuracy'], 
#                 marker=markers[i], linestyle='-', 
#                 color=colors[i], linewidth=2,
#                 markersize=8, label=f'Precision {prec}')
    
#     plt.title('Accuracy vs. Sparsity during Local Search')
#     plt.xlabel('Model Sparsity')
#     plt.ylabel('Validation Accuracy')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
# else:
#     print("Local search did not produce results to analyze.")



if 'local_search_df' in locals() and not local_search_df.empty:
    plt.figure(figsize=(12, 7))
    
    # Define distinct colors and markers for better readability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    # Get the unique precision levels from the results
    precisions = local_search_df['Precision'].unique()
    
    # Plot a separate, styled line for each precision
    for i, prec in enumerate(precisions):
        subset = local_search_df[local_search_df['Precision'] == prec]
        
        # Sort by sparsity to ensure the line is drawn correctly
        subset = subset.sort_values(by='Sparsity')
        
        plt.plot(subset['Sparsity'], subset['Accuracy'], 
                 marker=markers[i % len(markers)],  # Cycle through markers
                 linestyle='-', 
                 color=colors[i % len(colors)],    # Cycle through colors
                 linewidth=2,
                 markersize=8, 
                 label=f'Precision {prec}')

    plt.title('Accuracy vs. Sparsity Across Different Precisions', fontsize=16)
    plt.xlabel('Model Sparsity', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(title="Quantization Level", fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=max(0, local_search_df['Accuracy'].min() - 0.05)) # Adjust y-axis to focus on results
    plt.tight_layout()
    plt.show()
else:
    print("Local search did not produce results to analyze.")


# %% [markdown]
# The h5 file is saved HERE and use that to then synthesize with hls4ml


