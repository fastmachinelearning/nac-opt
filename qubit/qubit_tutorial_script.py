# %% [markdown]
# # SNAC-Pack for qubit readout model
#
# Uses block search with k-fold cross-validation and rule4ml hardware estimation.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

# imports
import os
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SNAC-pack utilities
from utils.tf_global_search import GlobalSearchTF
from utils.tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap
from utils.tf_local_search_separated import local_search_entrypoint
from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_and_preprocess_qubit
import seaborn as sns


# plotting settings and logging
# %matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
tf.get_logger().setLevel('ERROR')
print("TensorFlow Version:", tf.__version__)

# config
N_TRIALS = 3  # 25
EPOCHS = 5  # 10
SUBSET_SIZE = 10000
N_FOLDS = 3
RESULTS_DIR = "./results/tutorial4_qubit_script"
SEARCH_SPACE_PATH = "qubit_search_space.yaml"

os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
# qubit dataset configbut w
QUBIT_DATA_DIR = "../qubit/data"
START_LOCATION = 100
WINDOW_SIZE = 400
NUM_CLASSES = 2

# objectives (hardware-aware: 4 objectives)
OBJECTIVE_NAMES = ["performance_metric", "bops", "avg_resource", "clock_cycles"]
MAXIMIZE_FLAGS = [True, False, False, False]
OBJECTIVE_INFO = list(zip(OBJECTIVE_NAMES, MAXIMIZE_FLAGS))

# %% [markdown]
# ## Qubit Dataset
#
# Loading in 4 numpy arrays:
# - X_train_val: (900000, T)
# - y_train_val: (900000,)
# - X_test shape: (100000, T)
# - y_test shape: (100000,)
#
# The data takes the form of (N, T) where T (770) is a long 1D sequence per N.
#
# The readout window is the subset of time samples from the qubit readout waveform
# that is chosen to keep and fed to the model.
#
# Input size: 2 * readout_window.

# %%
# Load a subset for quick inspection
x_train_viz, y_train_viz, x_test_viz, y_test_viz = load_and_preprocess_qubit(
    data_dir=QUBIT_DATA_DIR,
    start_location=START_LOCATION,
    window_size=WINDOW_SIZE,
    subset_size=SUBSET_SIZE,
    normalize=False,
    flatten=True,
    one_hot=False,
    num_classes=NUM_CLASSES,
)

# # %%
# # plot two samples of the data

# DATA_DIR = "../qubit/data"
# X_TRAIN_FILE = "0528_X_train_0_770.npy"
# Y_TRAIN_FILE = "0528_y_train_0_770.npy"

# # Load raw data + labels
# X_raw = np.load(os.path.join(DATA_DIR, X_TRAIN_FILE)).astype(np.float32)
# y_raw = np.load(os.path.join(DATA_DIR, Y_TRAIN_FILE)).astype(np.int64)

# # Pick one example from each class
# idx0 = int(np.where(y_raw == 0)[0][0])
# idx1 = int(np.where(y_raw == 1)[0][0])

# x_state0 = X_raw[idx0]
# x_state1 = X_raw[idx1]

# print(f"Picked idx0={idx0} (y=0), idx1={idx1} (y=1)")
# print("Raw sample length:", x_state0.shape[0])

# def slice_window_1d(x, start_location, window_size):
#     start = start_location * 2
#     end = (start_location + window_size) * 2
#     if start < 0 or end > x.shape[0]:
#         return None
#     return x[start:end]

# def plot_grid_for_sample(x, label_text):
#     start_locations = [0, 100, 200, 300]
#     window_sizes = [100, 200, 400]
#     n_rows = len(start_locations)
#     n_cols = len(window_sizes)
#     fig, axes = plt.subplots(
#         n_rows, n_cols, figsize=(4.8 * n_cols, 2.6 * n_rows), sharey=True
#     )

#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = axes.reshape(1, -1)
#     elif n_cols == 1:
#         axes = axes.reshape(-1, 1)

#     for i, s in enumerate(start_locations):
#         for j, w in enumerate(window_sizes):
#             ax = axes[i, j]
#             xw = slice_window_1d(x, s, w)
#             ax.set_title(f"{label_text}\nstart={s}, win={w}")

#             if xw is None:
#                 ax.text(0.5, 0.5, "OUT OF RANGE", ha="center", va="center", transform=ax.transAxes)
#                 ax.set_axis_off()
#                 continue

#             # interleaved
#             ax.plot(xw, linewidth=1.0, label="interleaved")

#             # even/odd overlay (interpretable as 2 interleaved channels)
#             even = xw[0::2]
#             odd = xw[1::2]
#             ax.plot(np.arange(0, len(xw), 2), even, linewidth=1.0, alpha=0.8, label="even (0::2)")
#             ax.plot(np.arange(1, len(xw), 2), odd, linewidth=1.0, alpha=0.8, label="odd (1::2)")

#             if i == 0 and j == 0:
#                 ax.legend(fontsize=8)

#     plt.tight_layout()
#     plt.show()

# # Plot one grid for state 0 and one for state 1
# plot_grid_for_sample(x_state0, f"State 0 (idx={idx0})")
# plot_grid_for_sample(x_state1, f"State 1 (idx={idx1})")

# %% [markdown]
# ## Global Search

# %%
print("\n" + "=" * 50)
print("Running Qubit Block-Based Hardware-Aware Global Search...")
print("This may take a few minutes...")
print("=" * 50)

searcher = GlobalSearchTF(search_space_path=SEARCH_SPACE_PATH, results_dir=RESULTS_DIR)

study = searcher.run_search(
    model_type='block',
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    dataset='qubit',
    subset_size=SUBSET_SIZE,
    objectives=OBJECTIVE_NAMES,
    maximize_flags=MAXIMIZE_FLAGS,
    use_hardware_metrics=True,
    one_hot=True,
    n_folds=N_FOLDS,
    data_dir=QUBIT_DATA_DIR,
    start_location=START_LOCATION,
    window_size=WINDOW_SIZE,
    num_classes=NUM_CLASSES,
    normalize=False,
    flatten=True,
)

print("\nGlobal Search Complete!")

# %% [markdown]
# ## Analyze Global Search Results

# # %%
# results_df = pd.DataFrame(searcher.results)

# if not results_df.empty:
#     # Inspect best architecture
#     print("--- Best Discovered Architecture (by Accuracy) ---")
#     best_trial_row = results_df.loc[results_df['performance_metric'].idxmax()]
#     print(f"Trial Number: {best_trial_row['trial']}")
#     print(f"Accuracy: {best_trial_row['performance_metric']:.4f}")
#     print(f"BOPs: {best_trial_row['bops']:.2e}")
#     print(f"Avg Resource: {best_trial_row['avg_resource']:.2f}%")
#     print(f"Clock Cycles: {best_trial_row['clock_cycles']:.0f}")

#     with open(best_trial_row['yaml_path'], 'r') as f:
#         best_arch_yaml = yaml.safe_load(f)

#     print("\nArchitecture components:")
#     for component in best_arch_yaml['architecture']['components']:
#         print(f"- Type: {component['block_type']}, Name: {component['name']}")

#     # Pairwise 2D Pareto fronts
#     print("\n--- Visualizing Pairwise 2D Pareto Fronts ---")
#     plot_pareto_fronts(results_df, OBJECTIVE_INFO, save_dir=searcher.results_dir)

#     # 3D Pareto front heatmap
#     print("\n--- Generating 3D Pareto Front Heatmap ---")
#     plot_3d_pareto_front_heatmap(results_df, OBJECTIVE_INFO, save_dir=searcher.results_dir)

#     print(f"\nPlots saved to: {searcher.results_dir}")
# else:
#     print("No results found (all trials may have failed).")

# # %% [markdown]
# # ## Local Search

# # %%
# LOCAL_SEARCH_RESULTS_DIR = os.path.join(RESULTS_DIR, "local_search_separated")
# LOCAL_SEARCH_CONFIG_PATH = os.path.join(RESULTS_DIR, "local_search_settings_separated.yaml")
# ARCHITECTURE_YAML_PATH = os.path.join(RESULTS_DIR, "best_model_for_local_search.yaml")

# local_search_settings = {
#     "pruning_settings": {
#         "iterations": 5,
#         "epochs_per_iteration": 2,
#         "pruning_rate": 0.8,
#     },
#     "qat_settings": {
#         "epochs": 2,  # 5
#         "precision_pairs": [
#             {"total_bits": 16, "int_bits": 6},
#             {"total_bits": 8, "int_bits": 3},
#             {"total_bits": 6, "int_bits": 2},
#             {"total_bits": 4, "int_bits": 1},
#         ],
#     },
# }

# with open(LOCAL_SEARCH_CONFIG_PATH, "w") as f:
#     yaml.dump(local_search_settings, f)
# print(f"Created local search configuration file: {LOCAL_SEARCH_CONFIG_PATH}")

# # Load dataset (one-hot = True to match classification head expectations)
# x_train, y_train, x_test, y_test = load_and_preprocess_qubit(
#     data_dir=QUBIT_DATA_DIR,
#     start_location=START_LOCATION,
#     window_size=WINDOW_SIZE,
#     subset_size=SUBSET_SIZE,
#     normalize=False,
#     flatten=True,
#     one_hot=True,
#     num_classes=NUM_CLASSES,
# )

# if not os.path.exists(ARCHITECTURE_YAML_PATH):
#     raise FileNotFoundError(
#         f"Could not find best architecture YAML: {ARCHITECTURE_YAML_PATH}. "
#         "Run global search first or check RESULTS_DIR."
#     )

# pruning_results_df, qat_results_df = local_search_entrypoint(
#     architecture_yaml_path=ARCHITECTURE_YAML_PATH,
#     local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
#     dataset=(x_train, y_train, x_test, y_test),
#     results_dir=LOCAL_SEARCH_RESULTS_DIR,
# )

# # %%
# if isinstance(pruning_results_df, pd.DataFrame) and not pruning_results_df.empty:
#     if "Sparsity" in pruning_results_df.columns and "Accuracy" in pruning_results_df.columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(pruning_results_df["Sparsity"], pruning_results_df["Accuracy"], marker="o")
#         plt.title("Pruning: Accuracy vs. Sparsity")
#         plt.xlabel("Sparsity")
#         plt.ylabel("Accuracy")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Unexpected pruning_results_df columns:", list(pruning_results_df.columns))
# else:
#     print("No pruning results to plot.")

# if isinstance(qat_results_df, pd.DataFrame) and not qat_results_df.empty:
#     if "Precision" in qat_results_df.columns and "Accuracy" in qat_results_df.columns:
#         plt.figure(figsize=(10, 6))
#         palette = sns.color_palette("viridis", n_colors=len(qat_results_df))
#         sns.barplot(x="Precision", y="Accuracy", data=qat_results_df, palette=palette)
#         plt.title("QAT: Accuracy vs. Precision")
#         plt.xlabel("Precision")
#         plt.ylabel("Accuracy")
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Unexpected qat_results_df columns:", list(qat_results_df.columns))
# else:
#     print("No QAT results to plot.")

# # %% [markdown]
# # ## Combined Local Search (QAT + Pruning)
# #
# # Instead of running QAT and pruning independently, the combined search applies
# # iterative magnitude pruning to each quantized model. This reveals the full
# # compression landscape: accuracy vs. (sparsity x bit-width).

# # %%
# COMBINED_RESULTS_DIR = os.path.join(RESULTS_DIR, "local_search_combined")

# combined_results_df = combined_local_search_entrypoint(
#     architecture_yaml_path=ARCHITECTURE_YAML_PATH,
#     local_search_config_path=LOCAL_SEARCH_CONFIG_PATH,
#     dataset=(x_train, y_train, x_test, y_test),
#     results_dir=COMBINED_RESULTS_DIR,
#     n_folds=N_FOLDS,
# )

# # %%
# if isinstance(combined_results_df, pd.DataFrame) and not combined_results_df.empty:
#     # Accuracy vs Effective BOPs (one curve per precision)
#     plt.figure(figsize=(12, 7))
#     for prec in combined_results_df["Precision"].unique():
#         subset = combined_results_df[combined_results_df["Precision"] == prec]
#         plt.plot(subset["EffectiveBOPs"], subset["Accuracy"], marker="o", linewidth=2, label=prec)
#     plt.xlabel("Effective BOPs")
#     plt.ylabel("Accuracy")
#     plt.title("Combined QAT + Pruning: Accuracy vs Effective BOPs")
#     plt.legend(title="Precision")
#     plt.xscale("log")
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()

#     # Accuracy vs Sparsity (one curve per precision)
#     plt.figure(figsize=(12, 7))
#     for prec in combined_results_df["Precision"].unique():
#         subset = combined_results_df[combined_results_df["Precision"] == prec]
#         plt.plot(subset["Sparsity"], subset["Accuracy"], marker="o", linewidth=2, label=prec)
#     plt.xlabel("Sparsity")
#     plt.ylabel("Accuracy")
#     plt.title("Combined QAT + Pruning: Accuracy vs Sparsity")
#     plt.legend(title="Precision")
#     plt.gca().invert_xaxis()
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No combined local search results to plot.")

# %%
