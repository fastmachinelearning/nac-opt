# %%


# %%
"""
TensorFlow Tutorial for Neural Architecture Codesign (NAC)
This tutorial demonstrates how to use the main codebase for:
1. Global search on MNIST dataset with custom architectures
2. Training and evaluating models
3. Estimating hardware metrics
4. Visualizing the results of the search
"""

import os
import sys

import argparse
import yaml
import optuna
import tensorflow as tf
import pandas as pd
import time
from rule4ml.models.estimators import MultiModelEstimator

# Import utilities from main codebase
from utils.tf_data_preprocessing import load_and_preprocess_mnist
from utils.tf_model_builder import (
    build_mlp_from_config,
    build_deepsets_model,
    load_yaml_config
)

from utils.tf_processor import train_model, evaluate_model, get_model_metrics
from utils.tf_bops import get_MLP_bops_tf
from models.tf_blocks import DeepSetsArchitecture_tf
from utils.tf_visualization import *


import os
import sys
import yaml
import optuna
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import re
import itertools
import math
import ast
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# To display plots directly in the notebook
%matplotlib inline

# %%


# %%


class GlobalSearchTF:
    """Main class for conducting global search with TensorFlow models."""
    
    def __init__(self, search_space_path=None, hls_config=None, results_dir="./results_tf"):
        """
        Initialize global search.
        
        Parameters:
            search_space_path: Path to YAML file with search space
            hls_config: Hardware configuration for rule4ml
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        if search_space_path and os.path.exists(search_space_path):
             with open(search_space_path, 'r') as f:
                self.search_space = yaml.safe_load(f)
        else:
            self.search_space = self.get_default_search_space()
        
        self.hls_config = hls_config or self.get_default_hls_config()
        self.results = []
    
    def get_default_search_space(self):
        """Returns default search space for MLP on MNIST."""
        return {
            "num_layers": [2, 3],
            "hidden_units1": [8, 16, 32, 64, 128],
            "activation1": ["relu", "tanh", "sigmoid"],
            "batchnorm1": [True, False],
            "hidden_units2": [8, 16, 32, 64],
            "activation2": ["relu", "tanh", "sigmoid"],
            "batchnorm2": [True, False],
        }
    
    def get_default_hls_config(self):
        """Returns default HLS configuration."""
        return {
            "model": {"precision": "ap_fixed<8,3>", "reuse_factor": 1, "strategy": "Latency"},
            "board": "zcu102"
        }
    
    def create_mlp_objective(self, x_train, y_train, x_val, y_val, epochs=10):
        """Creates objective function for MLP optimization."""
        def objective(trial):
            num_layers = trial.suggest_categorical("num_layers", self.search_space["num_layers"])
            config = {
                "num_layers": num_layers,
                "hidden_units1": trial.suggest_categorical("hidden_units1", self.search_space["hidden_units1"]),
                "activation1": trial.suggest_categorical("activation1", self.search_space["activation1"]),
                "batchnorm1": trial.suggest_categorical("batchnorm1", self.search_space["batchnorm1"]),
            }
            if num_layers >= 3:
                config["hidden_units2"] = trial.suggest_categorical("hidden_units2", self.search_space["hidden_units2"])
                config["activation2"] = trial.suggest_categorical("activation2", self.search_space["activation2"])
                config["batchnorm2"] = trial.suggest_categorical("batchnorm2", self.search_space["batchnorm2"])
            
            input_size = x_train.shape[1]
            num_classes = y_train.shape[1]
            model = build_mlp_from_config(config, input_size=input_size, num_classes=num_classes)
            
            train_model(model, (x_train, y_train), (x_val, y_val), epochs=epochs, batch_size=128, patience=3, verbose=0)
            val_metrics = evaluate_model(model, (x_val, y_val))
            val_accuracy = val_metrics['accuracy']
            
            bops = self.calculate_mlp_bops(config, input_size, num_classes)
            
            # Simplified hardware metrics (as rule4ml might not be installed)
            avg_resource = np.random.rand() * 10 # Dummy value
            clock_cycles = np.random.randint(1000, 5000) # Dummy value
            
            print(f"Trial {trial.number}: Accuracy={val_accuracy:.4f}, BOPs={bops}, "
                  f"Avg Resource={avg_resource:.2f}, Clock Cycles={clock_cycles}")
            
            self.results.append({
                'trial': trial.number,
                'accuracy': val_accuracy,
                'bops': bops,
                'avg_resource': avg_resource,
                'clock_cycles': clock_cycles,
                'params': trial.params
            })
            
            return val_accuracy, bops, avg_resource, clock_cycles
        
        return objective

    def calculate_mlp_bops(self, config, input_size, num_classes, bit_width=32):
        """Calculate BOPs for MLP architecture."""
        bops = 0
        bops += input_size * config["hidden_units1"] * bit_width**2
        if config["num_layers"] >= 3:
            bops += config["hidden_units1"] * config["hidden_units2"] * bit_width**2
            bops += config["hidden_units2"] * num_classes * bit_width**2
        else:
            bops += config["hidden_units1"] * num_classes * bit_width**2
        return bops

    def run_search(self, model_type='mlp', n_trials=10, epochs=10, dataset='mnist',
                  subset_size=10000, resize_val=8, objectives=None, maximize_flags=None):
        """Run global search."""
        print(f"\n{'='*50}\nStarting {model_type.upper()} Global Search on {dataset.upper()}\n{'='*50}\n")
        self.objective_names = objectives or ['accuracy', 'bops', 'avg_resource', 'clock_cycles']
        self.maximize_flags = maximize_flags or [True, False, False, False]

        x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
            resize_val=resize_val, subset_size=subset_size, flatten=(model_type == 'mlp'), one_hot=True
        )

        objective = self.create_mlp_objective(x_train, y_train, x_val, y_val, epochs)
        directions = ["maximize" if flag else "minimize" for flag in self.maximize_flags]

        study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
        study.optimize(objective, n_trials=n_trials)

        self.save_results(model_type)
        return study
    
    def save_results(self, model_type):
        """Save search results to a CSV file."""
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, f"{model_type}_search_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nCSV results saved to {csv_file}")

# %%
# --- Configuration ---
MODEL_TYPE = "mlp"
N_TRIALS = 20  # Number of optimization trials
EPOCHS = 5      # Training epochs for each trial
SUBSET_SIZE = 5000 # Use a smaller subset for faster runs
RESIZE_VAL = 8     # Resize images to 8x8
RESULTS_DIR = "./results_tf_notebook"

# --- Objectives Configuration ---
# Names of the objectives you are optimizing
OBJECTIVE_NAMES = ['accuracy', 'bops', 'avg_resource', 'clock_cycles']
# For each objective, specify if you want to maximize (True) or minimize (False)
MAXIMIZE_FLAGS = [True, False, False, False]
OBJECTIVE_INFO = list(zip(OBJECTIVE_NAMES, MAXIMIZE_FLAGS))


# Initialize and run the global search
search_space_path = 'examples/mnist_search_spaces.yaml'
searcher = GlobalSearchTF(results_dir=RESULTS_DIR, search_space_path=search_space_path)

study = searcher.run_search(
    model_type=MODEL_TYPE,
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    subset_size=SUBSET_SIZE,
    resize_val=RESIZE_VAL,
    objectives=OBJECTIVE_NAMES,
    maximize_flags=MAXIMIZE_FLAGS
)

# %%
# --- Display Results ---
print("\n" + "="*50)
print("BEST TRIALS FOUND BY OPTUNA")
print("="*50)

# The 'best_trials' attribute of a study contains the Pareto optimal solutions
for i, trial in enumerate(study.best_trials):
    print(f"\nRank {i+1} (Trial {trial.number}):")
    # Create a clean dictionary of objective values for printing
    values_dict = {name: val for name, val in zip(OBJECTIVE_NAMES, trial.values)}
    print(f"  Values: {values_dict}")
    print(f"  Params: {trial.params}")

# %%
# --- Plotting Results ---
print("\n" + "="*50)
print("PLOTTING RESULTS")
print("="*50)

# Load the results from the CSV file into a pandas DataFrame
results_df = pd.DataFrame(searcher.results)

if not results_df.empty:
    # Sanitize column names for consistency
    results_df.columns = [col.lower().replace(" ", "_") for col in results_df.columns]
    
    # Generate and display the 2D Pareto front plots
    print("\n--- 2D Pareto Fronts ---")
    plot_pareto_fronts(results_df, OBJECTIVE_INFO, save_dir=searcher.results_dir)

    # Generate and display the 3D Pareto front plot if enough objectives exist
    if len(OBJECTIVE_NAMES) >= 4:
        print("\n--- 3D Pareto Front Heatmap ---")
        # Plot using the first 4 objectives
        plot_3d_pareto_front_heatmap(results_df, OBJECTIVE_INFO[:4], save_dir=searcher.results_dir)
    else:
        print("\nSkipping 3D plot: At least 4 objectives are required.")
else:
    print("No results to plot.")

# %%



