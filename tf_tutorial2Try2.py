# %%
# Standard imports
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np

# Import from your library
from utils.tf_global_search import GlobalSearchTF, run_mlp_search, run_deepsets_search
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap

%matplotlib inline



# %%

# --- Experiment Configuration ---
MODEL_TYPE = "mlp"              # "mlp" or "deepsets"
N_TRIALS = 50                   # Number of optimization trials
EPOCHS = 20                     # Training epochs for each trial
SUBSET_SIZE = 5000              # Use a smaller subset for faster runs
RESIZE_VAL = 8                  # Resize images to 8x8
RESULTS_DIR = "./results_tf_clean"
USE_HARDWARE_METRICS = False    # Set to True if you have rule4ml installed

# --- Objectives Configuration ---
OBJECTIVE_NAMES = ['accuracy', 'bops', 'avg_resource', 'clock_cycles']
MAXIMIZE_FLAGS = [True, False, False, False]  # True = maximize, False = minimize
OBJECTIVE_INFO = list(zip(OBJECTIVE_NAMES, MAXIMIZE_FLAGS))

# --- Search Space Configuration ---
SEARCH_SPACE_PATH = 'examples/mnist_search_spaces.yaml'  # Optional: use custom search space


# %% [markdown]
# ## Method 1 using convenience function

# %%
print("Method 1: Using convenience function")
print("\n" + "="*50)


# Run MLP search with convenience function
study, searcher = run_mlp_search(
    search_space_path=SEARCH_SPACE_PATH,
    results_dir=RESULTS_DIR,
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    subset_size=SUBSET_SIZE,
    resize_val=RESIZE_VAL,
    use_hardware_metrics=USE_HARDWARE_METRICS
)

# %% [markdown]
# ## Method 2: Using the GlobalSearchTF class directly (More control)

# %%
print("\n" + "="*50)
print("Method 2: Using GlobalSearchTF class directly")
print("="*50)

# Initialize the searcher
searcher_manual = GlobalSearchTF(
    search_space_path=SEARCH_SPACE_PATH,
    results_dir=RESULTS_DIR + "_manual"
)

# Run the search with custom parameters
study_manual = searcher_manual.run_search(
    model_type=MODEL_TYPE,
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    dataset='mnist',
    subset_size=SUBSET_SIZE,
    resize_val=RESIZE_VAL,
    objectives=OBJECTIVE_NAMES,
    maximize_flags=MAXIMIZE_FLAGS,
    use_hardware_metrics=USE_HARDWARE_METRICS,
    verbose=True
)


# %% [markdown]
# ## Analyze the results

# %%

print("\n" + "="*50)
print("ANALYZING RESULTS")
print("="*50)

# Get results as DataFrame
results_df = searcher.get_results_dataframe()

# Display basic statistics
print("\nResults Summary:")
print(f"Total trials completed: {len(results_df)}")
print(f"Best accuracy achieved: {results_df['accuracy'].max():.4f}")
print(f"Lowest BOPs: {results_df['bops'].min()}")

print("\nResults DataFrame:")
print(results_df.head())


print("\n" + "="*50)
print("VISUALIZING RESULTS")
print("="*50)

if not results_df.empty:
    # Generate and display the 2D Pareto front plots
    print("\n--- Generating 2D Pareto Fronts ---")
    plot_pareto_fronts(results_df, OBJECTIVE_INFO, save_dir=searcher.results_dir)

    # Generate and display the 3D Pareto front plot if enough objectives exist
    if len(OBJECTIVE_NAMES) >= 4:
        print("\n--- Generating 3D Pareto Front Heatmap ---")
        plot_3d_pareto_front_heatmap(results_df, OBJECTIVE_INFO[:4], save_dir=searcher.results_dir)
    else:
        print("\nSkipping 3D plot: At least 4 objectives are required.")
        
    print(f"\nAll plots saved to: {searcher.results_dir}")
else:
    print("No results to plot.")

# %% [markdown]
# ## Advanced Usage: Custom Objective Function

# %%
def create_custom_objective_example():
    """
    Example of how to create a custom objective function for specific needs.
    This is useful when you need more control over the optimization process.
    """
    # Load data manually
    from utils.tf_data_preprocessing import load_and_preprocess_mnist
    x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
        resize_val=RESIZE_VAL, subset_size=SUBSET_SIZE, flatten=True, one_hot=True
    )
    
    # Initialize searcher
    custom_searcher = GlobalSearchTF(results_dir=RESULTS_DIR + "_custom")
    
    # Create custom objective with specific parameters
    objective = custom_searcher.create_mlp_objective(
        x_train, y_train, x_val, y_val,
        epochs=EPOCHS,
        use_hardware_metrics=USE_HARDWARE_METRICS,
        verbose=True
    )
    
    # You can now use this objective with Optuna directly for maximum control
    import optuna
    directions = ["maximize" if flag else "minimize" for flag in MAXIMIZE_FLAGS]
    study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
    
    # Run a few trials as example
    study.optimize(objective, n_trials=5)
    
    print("Custom objective function example completed!")
    return study, custom_searcher

# Run the custom example
# custom_study, custom_searcher = create_custom_objective_example()


# %% [markdown]
# ## DeepSets Example (if you want to try a different architecture)

# %%
print("\n" + "="*50)
print("BONUS: DeepSets Architecture Example")
print("="*50)

def run_deepsets_example():
    """
    Example of running DeepSets architecture search.
    Note: This requires different data preprocessing (no flattening).
    """
    print("Running DeepSets search example...")
    
    # For DeepSets, we need 3D input data (batch, particles, features)
    # This is a simplified example - you'd normally load proper set data
    deepsets_study, deepsets_searcher = run_deepsets_search(
        results_dir=RESULTS_DIR + "_deepsets",
        n_trials=5,  # Fewer trials for demo
        epochs=3,
        subset_size=1000,
        use_hardware_metrics=USE_HARDWARE_METRICS
    )
    
    print("DeepSets example completed!")
    return deepsets_study, deepsets_searcher

# Uncomment to run DeepSets example
deepsets_study, deepsets_searcher = run_deepsets_example()


# %%



