# %% [markdown]
# # Tutorial 2: Architectural Discovery with Block-Based Search
# 
# In the previous tutorial, we fine-tuned a known architecture (MLP). But what if the best architecture for our problem is a combination of different diverse layer types?
# 
# In this notebook, we'll unleash SNAC-pack to **discover new architectures** by combining building blocks like **Convolutional layers**, **Attention**, and **MLPs**.
# 
# ## The New Challenge
# 
# Since we are building arbitrary structures, our hardware estimator may not be accurate. We will instead optimize for two hardware-agnostic objectives:
# 1.  **Accuracy** (Maximize)
# 2.  **Computational Cost (BOPs)** (Minimize)

# %%

# Basic imports and setup
import os
import yaml

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import SNAC-pack utilities
from utils.tf_global_search5 import GlobalSearchTF
from utils.tf_visualization import plot_interactive_2d_pareto
from utils.tf_local_search1 import local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_mnist
from utils.tf_data_preprocessing import load_and_preprocess_fashion_mnist

np.random.seed(42)
tf.random.set_seed(42)


# --- Configuration ---
N_TRIALS_HYBRID = 15 # Note: Increase for a real search
EPOCHS_HYBRID = 15
SUBSET_SIZE_HYBRID = 30000
RESULTS_DIR_HYBRID = "./results/tutorial2_Hybrid_Discovery"
SEARCH_SPACE_PATH = 'hybrid_search_space.yaml'
RESIZE_VAL = 16

os.makedirs(RESULTS_DIR_HYBRID, exist_ok=True)

# %%

# Load the data for visualization (un-flattened)
x_train_viz, y_train_viz, _, _ = load_and_preprocess_fashion_mnist(
    resize_val=RESIZE_VAL,
    subset_size=SUBSET_SIZE_HYBRID, 
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
plt.suptitle("Sample Fashion MNIST Images")
plt.show()

# %% [markdown]
# ## Global Search: Building with Blocks
# 
# We first define a `search_space` in a YAML file. This file provides SNAC-pack with the "Lego bricks" it can use to build and test new architectures. We'll allow it to choose between `Conv`, `MLP`, and `ConvAttn` blocks.
# 

# %%

# --- Create the YAML configuration file for the hybrid search ---
search_space_yaml = """
channel_space: [8, 16, 32]
mlp_width_space: [32, 64, 128]
kernel_space: [1, 3, 5]
act_space: ["ReLU", "GELU", "LeakyRelu"]
norm_space: [null, "batch", "layer"]
block_types: ["Conv", "MLP", "None"]
num_blocks: 8
initial_img_size: 16
output_dim: 10
"""

with open(SEARCH_SPACE_PATH, 'w') as f:
    f.write(search_space_yaml)
print(f"Created search space configuration file: {SEARCH_SPACE_PATH}")

# --- Objectives for Hybrid Search (Performance vs. Cost) ---
OBJECTIVE_NAMES_HYBRID = ['performance_metric', 'bops']
MAXIMIZE_FLAGS_HYBRID = [True, False]

# --- Run the Hybrid Search ---
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
    one_hot=True,
)

print("\nGlobal Search Complete!")

# %% [markdown]
# ## Analyzing the Global Search Results
# 
# The search has finished exploring different architectural combinations. Let's see what it found!

# %%


results_df_hybrid = pd.DataFrame(searcher_hybrid.results)

if not results_df_hybrid.empty:
    # --- Inspect the Best Discovered Architecture ---
    print("--- Best Discovered Architecture (by Accuracy) ---")
    best_trial_row = results_df_hybrid.loc[results_df_hybrid['performance_metric'].idxmax()]
    print(f"Trial Number: {best_trial_row['trial']}")
    print(f"Accuracy: {best_trial_row['performance_metric']:.4f}")
    print(f"BOPs: {best_trial_row['bops']:.2e}")

    # Load and print the architecture from its YAML file
    with open(best_trial_row['yaml_path'], 'r') as f:
        best_arch_yaml = yaml.safe_load(f)

    print("\nArchitecture components:")
    for component in best_arch_yaml['architecture']['components']:
        print(f"- Type: {component['block_type']}, Name: {component['name']}")

    # --- Visualize the Pareto Front (Accuracy vs BOPs) ---
    print("\n--- Generating Interactive Pareto Front Plot for Hybrid Search ---")
    # REPLACE THE OLD PLOTTING FUNCTION WITH THE NEW ONE
    plot_interactive_2d_pareto(
        results_df_hybrid,
        list(zip(OBJECTIVE_NAMES_HYBRID, MAXIMIZE_FLAGS_HYBRID)),
        save_dir=searcher_hybrid.results_dir
    )
else:
    print("Hybrid search did not yield any results to analyze.")

# %% [markdown]
# ## Local Search: Compressing Our New Discovery
# 
# Just like before, the global search saved the best model. We will now apply the same powerful QAT and pruning techniques to this newly discovered hybrid model, demonstrating the consistent workflow of SNAC-pack.
# 

# %%

# --- Configuration for Local Search ---
LOCAL_SEARCH_RESULTS_DIR = os.path.join(RESULTS_DIR_HYBRID, "local_search")
LOCAL_SEARCH_CONFIG_PATH = os.path.join(RESULTS_DIR_HYBRID, 'local_search_settings.yaml')

# Define settings for QAT and pruning
local_search_settings = {
    'precision_pairs': [
        {'total_bits': 16, 'int_bits': 6},
        {'total_bits': 8, 'int_bits': 3},
        {'total_bits': 4, 'int_bits': 1},
    ],
    'pruning_iterations': 5,
    'epochs_per_iteration': 5,
    'pruning_rate': 0.8,
}

# Write the settings to a YAML file
with open(LOCAL_SEARCH_CONFIG_PATH, 'w') as f:
    yaml.dump(local_search_settings, f)
print(f"Created local search configuration file: {LOCAL_SEARCH_CONFIG_PATH}")

# Path to the best model found by the global search
ARCHITECTURE_YAML_PATH = os.path.join(RESULTS_DIR_HYBRID, "best_model_for_local_search.yaml")

# --- Load Dataset for Local Search ---
resize_val = searcher_hybrid.search_space.get('initial_img_size', 11)
x_train, y_train, x_val, y_val = load_and_preprocess_fashion_mnist(
    resize_val=resize_val, 
    subset_size=SUBSET_SIZE_HYBRID, 
    flatten=False,
    one_hot=True
)

# --- Run the Local Search ---
if os.path.exists(ARCHITECTURE_YAML_PATH):
    local_search_df_hybrid = local_search_entrypoint(
        architecture_yaml_path=ARCHITECTURE_YAML_PATH,
        local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=LOCAL_SEARCH_RESULTS_DIR
    )
else:
    print(f"ERROR: Could not find the architecture file: {ARCHITECTURE_YAML_PATH}")
    local_search_df_hybrid = pd.DataFrame()

# %% [markdown]
# ## Analyzing the Local Search Results
# 
# Finally, let's visualize the accuracy/sparsity trade-off for our discovered and compressed model.
# 

# %%
if not local_search_df_hybrid.empty:
    plt.figure(figsize=(10, 6))
    
    # Define distinct colors and markers
    colors = ['blue', 'red']
    markers = ['o', 's']  # circle and square
    
    precisions = local_search_df_hybrid['Precision'].unique()
    # print(f"Found precisions: {precisions}")
    # print(f"Data shape: {local_search_df.shape}")
    
    for i, prec in enumerate(precisions):
        subset = local_search_df_hybrid[local_search_df_hybrid['Precision'] == prec]
        print(f"Precision {prec}: {len(subset)} data points")
        print(subset[['Iteration', 'Sparsity', 'Accuracy']].to_string())
        
        plt.plot(subset['Sparsity'], subset['Accuracy'], 
                marker=markers[i], linestyle='-', 
                color=colors[i], linewidth=2,
                markersize=8, label=f'Precision {prec}')
    
    plt.title('Accuracy vs. Sparsity during Local Search')
    plt.xlabel('Model Sparsity')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("Local search did not produce results to analyze.")


# %%
if 'local_search_df_hybrid' in locals() and not local_search_df_hybrid.empty:
    plt.figure(figsize=(12, 7))
    
    # Define distinct colors and markers for better readability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    # Get the unique precision levels from the results
    precisions = local_search_df_hybrid['Precision'].unique()
    
    # Plot a separate, styled line for each precision
    for i, prec in enumerate(precisions):
        subset = local_search_df_hybrid[local_search_df_hybrid['Precision'] == prec]
        
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
    # plt.ylim(bottom=max(0, local_search_df_hybrid['Accuracy'].min() - 0.05)) # Adjust y-axis to focus on results
    plt.tight_layout()
    plt.show()
else:
    print("Local search did not produce results to analyze.")


