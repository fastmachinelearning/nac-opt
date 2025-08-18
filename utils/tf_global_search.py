"""
TensorFlow Global Search utilities for Neural Architecture Codesign (NAC).
This module provides the main GlobalSearchTF class and objective functions
for multi-objective optimization of neural architectures.
"""

import os
import yaml
import optuna
import tensorflow as tf
import pandas as pd
import numpy as np
import time

from .tf_data_preprocessing import load_and_preprocess_mnist
from .tf_model_builder import build_mlp_from_config, build_deepsets_model
from .tf_processor import train_model, evaluate_model, get_model_metrics
from .tf_bops import get_MLP_bops_tf


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
        self.objective_names = []
        self.maximize_flags = []
    
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
    
    def create_mlp_objective(self, x_train, y_train, x_val, y_val, epochs=10, 
                           use_hardware_metrics=False, verbose=True):
        """
        Creates objective function for MLP optimization.
        
        Parameters:
            x_train, y_train: Training data
            x_val, y_val: Validation data  
            epochs: Number of training epochs
            use_hardware_metrics: Whether to calculate real hardware metrics (requires rule4ml)
            verbose: Whether to print trial results
            
        Returns:
            Objective function for Optuna optimization
        """
        def objective(trial):
            # Sample architecture configuration
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
            
            # Build and train model
            input_size = x_train.shape[1]
            num_classes = y_train.shape[1]
            model = build_mlp_from_config(config, input_size=input_size, num_classes=num_classes)
            
            # Train model
            train_model(model, (x_train, y_train), (x_val, y_val), 
                       epochs=epochs, batch_size=128, patience=3, verbose=0)
            
            # Evaluate performance
            val_metrics = evaluate_model(model, (x_val, y_val))
            val_accuracy = val_metrics['accuracy']
            
            # Calculate BOPs
            bops = self.calculate_mlp_bops_tf(model, input_size)
            
            # Calculate hardware metrics
            if use_hardware_metrics:
                try:
                    # This would use rule4ml if available
                    # avg_resource, clock_cycles = self.calculate_hardware_metrics(model)
                    avg_resource, clock_cycles = self.calculate_hardware_metrics(model, input_size)
                except ImportError:
                    if verbose:
                        print("Warning: rule4ml not available, using dummy hardware metrics")
                    avg_resource = np.random.rand() * 10
                    clock_cycles = np.random.randint(1000, 5000)
            else:
                # Use dummy values for demonstration
                avg_resource = np.random.rand() * 10
                clock_cycles = np.random.randint(1000, 5000)
            
            if verbose:
                print(f"Trial {trial.number}: Accuracy={val_accuracy:.4f}, BOPs={bops}, "
                      f"Avg Resource={avg_resource:.2f}, Clock Cycles={clock_cycles}")
            
            # Store results
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
    
    def create_deepsets_objective(self, x_train, y_train, x_val, y_val, epochs=10,
                                use_hardware_metrics=False, verbose=True):
        """
        Creates objective function for DeepSets optimization.
        
        Parameters:
            x_train, y_train: Training data (should be 3D for DeepSets)
            x_val, y_val: Validation data
            epochs: Number of training epochs
            use_hardware_metrics: Whether to calculate real hardware metrics
            verbose: Whether to print trial results
            
        Returns:
            Objective function for Optuna optimization
        """
        def objective(trial):
            # Sample DeepSets architecture
            phi_config = {
                'num_layers': trial.suggest_int('phi_num_layers', 1, 3),
                'units': trial.suggest_categorical('phi_units', [16, 32, 64]),
                'activation': trial.suggest_categorical('phi_activation', ['relu', 'tanh']),
                'batchnorm': trial.suggest_categorical('phi_batchnorm', [True, False]),
                'bottleneck_dim': 2 ** trial.suggest_int('bottleneck_dim', 3, 6),
            }
            
            rho_config = {
                'num_layers': trial.suggest_int('rho_num_layers', 1, 3),
                'units': trial.suggest_categorical('rho_units', [16, 32, 64]),
                'activation': trial.suggest_categorical('rho_activation', ['relu', 'tanh']),
                'batchnorm': trial.suggest_categorical('rho_batchnorm', [True, False]),
            }
            
            aggregator_type = trial.suggest_categorical('aggregator_type', ['mean', 'max'])
            
            # Build model
            num_classes = y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
            input_shape = x_train.shape[1:]  # Remove batch dimension
            
            model = build_deepsets_model(
                phi_config, rho_config, aggregator_type,
                input_shape=(None, *input_shape), num_classes=num_classes
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if len(y_train.shape) == 1 
                     else tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
            
            # Train model
            train_model(model, (x_train, y_train), (x_val, y_val),
                       epochs=epochs, batch_size=128, patience=3, verbose=0)
            
            # Evaluate performance
            val_metrics = evaluate_model(model, (x_val, y_val))
            val_accuracy = val_metrics['accuracy']
            
            # Calculate BOPs (simplified for DeepSets)
            bops = model.count_params() * 32  # Rough approximation
            
            # Calculate hardware metrics
            if use_hardware_metrics:
                try:
                    avg_resource, clock_cycles = self.calculate_hardware_metrics(model)
                except ImportError:
                    if verbose:
                        print("Warning: rule4ml not available, using dummy hardware metrics")
                    avg_resource = np.random.rand() * 10
                    clock_cycles = np.random.randint(1000, 5000)
            else:
                avg_resource = np.random.rand() * 10
                clock_cycles = np.random.randint(1000, 5000)
            
            if verbose:
                print(f"Trial {trial.number}: Accuracy={val_accuracy:.4f}, BOPs={bops}, "
                      f"Avg Resource={avg_resource:.2f}, Clock Cycles={clock_cycles}")
            
            # Store results
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

    def calculate_mlp_bops_tf(self, model, input_size, bit_width=32):
        """
        Calculate BOPs for MLP architecture using tf_bops utilities.
        
        Parameters:
            model: TensorFlow model
            input_size: Input dimension
            bit_width: Bit width for calculations
            
        Returns:
            Total BOPs for the model
        """
        return get_MLP_bops_tf(model, input_shape=(1, input_size), bit_width=bit_width)


    def calculate_hardware_metrics(self, model, input_size):
        """
        Calculate hardware metrics using rule4ml.
        Parameters:
            model: TensorFlow model
            input_size: The input dimension for the model, required for patching.
            
        Returns:
            tuple: (avg_resource, clock_cycles)
        """
        try:
            from rule4ml.models.estimators import MultiModelEstimator
            
            # Patch model layers with shape info, which is required by rule4ml.
            # This logic is taken from your working example script.
            for layer in model.layers:
                if hasattr(layer, "input_spec") and layer.input_spec and layer.input_spec.shape:
                    layer._build_shapes_dict = {"input": layer.input_spec.shape}
                elif hasattr(layer, "input_shape") and layer.input_shape is not None:
                    layer._build_shapes_dict = {"input": layer.input_shape}
                else:
                    # Fallback for the first layer if shape is not yet inferred
                    layer._build_shapes_dict = {"input": (None, input_size)}

            estimator = MultiModelEstimator()
            estimator.load_default_models()
            
            pred_df = estimator.predict([model], [self.hls_config])
            
            if not pred_df.empty:
                results = pred_df.iloc[0]
                lut = results.get("LUT (%)", 0)
                ff = results.get("FF (%)", 0)
                bram = results.get("BRAM (%)", 0)
                dsp = results.get("DSP (%)", 0)
                avg_resource = np.mean([lut, ff, bram, dsp])
                clock_cycles = results.get('CYCLES', 1000)
            else:
                print("Warning: Hardware estimation failed to return results. Returning default values.")
                avg_resource = 0.0
                clock_cycles = 1000

            return avg_resource, clock_cycles
            
        except ImportError:
            raise ImportError("rule4ml package is required for hardware metrics calculation")
        except Exception as e:
            print(f"An error occurred during hardware estimation: {e}. Returning dummy values.")
            return np.random.rand() * 10, np.random.randint(1000, 5000)
            
        except ImportError:
            raise ImportError("rule4ml package is required for hardware metrics calculation")
        except Exception as e:
            # Catch other potential errors from the estimator
            print(f"An error occurred during hardware estimation: {e}. Returning dummy values.")
            return np.random.rand() * 10, np.random.randint(1000, 5000)
        

    def run_search(self, model_type='mlp', n_trials=10, epochs=10, dataset='mnist',
                  subset_size=10000, resize_val=8, objectives=None, maximize_flags=None,
                  use_hardware_metrics=False, verbose=True):
        """
        Run global search.
        
        Parameters:
            model_type: Type of model ('mlp' or 'deepsets')
            n_trials: Number of optimization trials
            epochs: Training epochs per trial
            dataset: Dataset to use
            subset_size: Size of dataset subset
            resize_val: Image resize dimension
            objectives: List of objective names
            maximize_flags: Boolean flags for maximization per objective
            use_hardware_metrics: Whether to use real hardware metrics
            verbose: Whether to print progress
            
        Returns:
            Optuna study object
        """
        if verbose:
            print(f"\n{'='*50}\nStarting {model_type.upper()} Global Search on {dataset.upper()}\n{'='*50}\n")
        
        self.objective_names = objectives or ['accuracy', 'bops', 'avg_resource', 'clock_cycles']
        self.maximize_flags = maximize_flags or [True, False, False, False]

        # Load data
        if dataset == 'mnist':
            x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
                resize_val=resize_val, 
                subset_size=subset_size, 
                flatten=(model_type == 'mlp'), 
                one_hot=(model_type == 'mlp')
            )
        else:
            raise ValueError(f"Dataset {dataset} not supported")

        # Create objective function
        if model_type == 'mlp':
            objective = self.create_mlp_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
            )
        elif model_type == 'deepsets':
            objective = self.create_deepsets_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")

        # Set up optimization directions
        directions = ["maximize" if flag else "minimize" for flag in self.maximize_flags]

        # Run optimization
        study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
        study.optimize(objective, n_trials=n_trials)

        # Save results
        self.save_results(model_type)
        
        if verbose:
            self.print_best_trials(study)
        
        return study
    
    def save_results(self, model_type):
        """Save search results to a CSV file."""
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, f"{model_type}_search_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nCSV results saved to {csv_file}")
        return csv_file
    
    def print_best_trials(self, study):
        """Print the best trials found by Optuna."""
        print("\n" + "="*50)
        print("BEST TRIALS FOUND BY OPTUNA")
        print("="*50)

        for i, trial in enumerate(study.best_trials):
            print(f"\nRank {i+1} (Trial {trial.number}):")
            values_dict = {name: val for name, val in zip(self.objective_names, trial.values)}
            print(f"  Values: {values_dict}")
            print(f"  Params: {trial.params}")
    
    def get_results_dataframe(self):
        """Return results as a pandas DataFrame."""
        df = pd.DataFrame(self.results)
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        return df


def run_mlp_search(search_space_path=None, results_dir="./results_tf", n_trials=20, 
                  epochs=5, subset_size=5000, resize_val=8, use_hardware_metrics=False):
    """
    Convenience function to run MLP search with common parameters.
    
    Parameters:
        search_space_path: Path to search space YAML
        results_dir: Directory for results
        n_trials: Number of trials
        epochs: Training epochs
        subset_size: Dataset subset size
        resize_val: Image resize dimension
        use_hardware_metrics: Whether to use real hardware metrics
        
    Returns:
        tuple: (study, searcher) for further analysis
    """
    searcher = GlobalSearchTF(search_space_path=search_space_path, results_dir=results_dir)
    
    study = searcher.run_search(
        model_type='mlp',
        n_trials=n_trials,
        epochs=epochs,
        subset_size=subset_size,
        resize_val=resize_val,
        use_hardware_metrics=use_hardware_metrics
    )
    
    return study, searcher


def run_deepsets_search(search_space_path=None, results_dir="./results_tf", n_trials=20,
                       epochs=5, subset_size=5000, use_hardware_metrics=False):
    """
    Convenience function to run DeepSets search with common parameters.
    
    Parameters:
        search_space_path: Path to search space YAML
        results_dir: Directory for results  
        n_trials: Number of trials
        epochs: Training epochs
        subset_size: Dataset subset size
        use_hardware_metrics: Whether to use real hardware metrics
        
    Returns:
        tuple: (study, searcher) for further analysis
    """
    searcher = GlobalSearchTF(search_space_path=search_space_path, results_dir=results_dir)
    
    study = searcher.run_search(
        model_type='deepsets',
        n_trials=n_trials,
        epochs=epochs,
        subset_size=subset_size,
        use_hardware_metrics=use_hardware_metrics
    )
    
    return study, searcher