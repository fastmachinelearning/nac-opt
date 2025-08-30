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
from .tf_bops import get_MLP_bops_tf, estimate_conv_bops, estimate_attention_bops, estimate_mlp_bops
from .tf_blocks import ConvAttentionBlock

# Helper function to save architecture to YAML
def _save_architecture_to_yaml(model_details, file_path):
    """Saves the model architecture to a YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(model_details, f, default_flow_style=False, sort_keys=False)


class BlockArchitectureTF:
    """
    A flexible container for a sequence of TensorFlow layers.
    It now conditionally adds a Flatten layer before the final MLP head,
    only if the feature extractor blocks did not already flatten the input.
    """
    def __init__(self, blocks, mlp, input_shape, needs_flattening):
        self.input_shape = input_shape
        self.blocks = blocks
        self.mlp = mlp
        self.needs_flattening = needs_flattening
        self._build_model()

    def _build_model(self):
        """Builds the Keras model from the provided blocks and MLP."""
        self.inputs = tf.keras.Input(shape=self.input_shape)
        x = self.inputs

        for block in self.blocks:
            x = block(x)

        # Only add a Flatten layer if the preceding blocks were all 2D.
        if self.needs_flattening:
            x = tf.keras.layers.Flatten()(x)
        
        x = self.mlp(x)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=x, name='BlockArchitecture')

    def __call__(self, inputs, training=None):
        return self.model(inputs, training=training)

    def compile(self, **kwargs):
        return self.model.compile(**kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def count_params(self):
        return self.model.count_params()

def create_conv_block_tf(channels, kernels, activations, normalizations, name='conv_block'):
    """Creates a sequential block of convolutional layers."""
    layers = []
    for i in range(len(kernels)):
        layers.append(tf.keras.layers.Conv2D(
            channels[i+1], kernel_size=kernels[i], strides=1,
            padding='valid' if kernels[i] > 1 else 'same', name=f'{name}_conv_{i}'
        ))
        if normalizations[i] == 'batch':
            layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        if activations[i] is not None:
            layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)


def create_conv_attention_block_tf(in_channels, hidden_channels, activation=None, name='conv_attention'):
    return ConvAttentionBlock(in_channels, hidden_channels, activation=activation, name=name)


def get_activation_tf(act_name):
    """
    Returns a Keras activation layer instance from a string name.
    Names are NOT set here to prevent duplicates in Sequential models.
    Keras will auto-assign unique names.
    """
    if act_name is None or act_name.lower() == "identity" or act_name.lower() == "linear":
        return tf.keras.layers.Activation('linear')
    elif act_name.lower() == "relu":
        return tf.keras.layers.ReLU()
    elif act_name.lower() == "leakyrelu":
        return tf.keras.layers.LeakyReLU(alpha=0.01)
    elif act_name.lower() == "gelu":
        return tf.keras.layers.Activation('gelu')
    else:
        # Fallback for other keras supported activations
        return tf.keras.layers.Activation(act_name.lower())

def sample_conv_block_tf(trial, prefix, in_channels, search_space, num_layers=2):
    channel_space = search_space["channel_space"]
    kernel_space = search_space["kernel_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]
    channels = [int(in_channels)]
    for i in range(num_layers):
        next_channel_idx = trial.suggest_categorical(f"{prefix}_channels_{i}", list(range(len(channel_space))))
        channels.append(channel_space[next_channel_idx])
    kernels = [trial.suggest_categorical(f"{prefix}_kernels_{i}", kernel_space) for i in range(num_layers)]
    act_names = [trial.suggest_categorical(f"{prefix}_acts_{i}", act_space) for i in range(num_layers)]
    norms = [trial.suggest_categorical(f"{prefix}_norms_{i}", norm_space) for i in range(num_layers)]
    return channels, kernels, act_names, norms


def sample_conv_attention_tf(trial, prefix, search_space):
    hidden_channel_space = search_space["conv_attn"]["hidden_channel_space"]
    act_space = search_space["act_space"]
    hidden_channels_idx = trial.suggest_categorical(f"{prefix}_hiddenchannel", list(range(len(hidden_channel_space))))
    hidden_channels = hidden_channel_space[hidden_channels_idx]
    act_name = trial.suggest_categorical(f"{prefix}_act", act_space)
    return hidden_channels, act_name


def sample_mlp_tf(trial, in_dim, out_dim, prefix, search_space, num_layers=3):
    mlp_width_space = search_space["mlp_width_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]
    widths = [in_dim]
    for i in range(num_layers - 1):
        width_idx = trial.suggest_categorical(f"{prefix}_width_{i}", list(range(len(mlp_width_space))))
        widths.append(mlp_width_space[width_idx])
    widths.append(out_dim)
    act_names = [trial.suggest_categorical(f"{prefix}_acts_{i}", act_space) for i in range(num_layers)]
    norms = [trial.suggest_categorical(f"{prefix}_norms_{i}", norm_space) for i in range(num_layers)]
    return widths, act_names, norms


def build_mlp_from_config_classifier(widths, activations, normalizations, name='mlp'):
    layers = []
    for i in range(len(activations)):
        layers.append(tf.keras.layers.Dense(widths[i+1], name=f'{name}_dense_{i}'))
        if normalizations[i] == 'batch':
            layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        elif normalizations[i] == 'layer':
            layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
        if activations[i] is not None:
            layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)


class GlobalSearchTF:
    def __init__(self, search_space_path=None, hls_config=None, results_dir="./results_tf"):
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
        return {
            "channel_space": [4, 8, 16, 32, 64],
            "mlp_width_space": [4, 8, 16, 32, 64],
            "kernel_space": [1, 3, 5],
            "act_space": ["ReLU", "LeakyReLU", "GELU", "Identity"],
            "norm_space": [None, "batch", "layer"],
            "block_types": ["Conv", "ConvAttn", "MLP", "None"],
            "conv_attn": {"hidden_channel_space": [1, 2, 4, 8, 16, 32]},
            "num_blocks": 3,
            "initial_img_size": 9,
            "output_dim": 2
        }

    def get_default_hls_config(self):
        return {
            "model": {"precision": "ap_fixed<8,3>", "reuse_factor": 1, "strategy": "Latency"},
            "board": "zcu102"
        }

    def create_block_objective(self, x_train, y_train, x_val, y_val, epochs=10, use_hardware_metrics=False, verbose=True):
        """Creates the objective function for Optuna to optimize."""
        def objective(trial):
            try:
                spaces = self.search_space
                num_blocks = spaces.get("num_blocks", 3)
                img_size = spaces.get("initial_img_size", 11)
                output_dim = spaces.get("output_dim", 10)
                
                bops = 0
                
                current_img_size = img_size
                current_channels = x_train.shape[-1]
                is_flattened = False
                last_layer_units = 0
                
                model_components = []
                feature_extractor_blocks = []

                block_types = [trial.suggest_categorical(f"b{i}", spaces["block_types"]) for i in range(num_blocks)]

                for i, block_type in enumerate(block_types):
                    if is_flattened and block_type in ["Conv", "ConvAttn"]:
                        continue
                    if not is_flattened and current_img_size <= 0:
                        break

                    if block_type == "Conv":
                        channels, kernels, act_names, norms = sample_conv_block_tf(trial, f"b{i}_Conv", current_channels, spaces)
                        size_reduction = sum((k - 1) for k in kernels if k > 1)
                        if current_img_size - size_reduction <= 0:
                            kernels = [1] * len(kernels)
                            size_reduction = 0
                        
                        # Convert activation names to layer objects
                        acts = [get_activation_tf(name) for name in act_names]
                        conv_block = create_conv_block_tf(channels, kernels, acts, norms, name=f'conv_block_{i}')
                        feature_extractor_blocks.append(conv_block)
                        model_components.append({
                            'block_type': 'Conv', 'name': f'conv_block_{i}',
                            'params': {'channels': channels, 'kernels': kernels, 'activations': act_names, 'normalizations': norms}
                        })
                        
                        current_img_size -= size_reduction
                        current_channels = channels[-1]
                        bops += estimate_conv_bops(channels, kernels, current_img_size)

                    elif block_type == "MLP":
                        if not is_flattened:
                            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=f'initial_flatten'))
                            model_components.append({'block_type': 'Flatten', 'name': f'initial_flatten', 'params': {}})
                            last_layer_units = current_channels * (current_img_size ** 2)
                            is_flattened = True
                        
                        units, act_name, norm = sample_dense_block_tf(trial, f"b{i}_MLP", spaces)
                        
                        dense_layer = tf.keras.layers.Dense(units, name=f'mlp_block_{i}_dense')
                        feature_extractor_blocks.append(dense_layer)
                        
                        if norm == 'batch':
                            feature_extractor_blocks.append(tf.keras.layers.BatchNormalization(name=f'mlp_block_{i}_bn'))
                        if act_name:
                            feature_extractor_blocks.append(get_activation_tf(act_name))
                        
                        model_components.append({
                            'block_type': 'MLP', 'name': f'mlp_block_{i}',
                            'params': {'widths': [last_layer_units, units], 'activations': [act_name], 'normalizations': [norm]}
                        })
                        bops += estimate_mlp_bops([last_layer_units, units])
                        last_layer_units = units

                if not is_flattened:
                    if current_img_size <= 0:
                        raise optuna.exceptions.TrialPruned("Image size became non-positive.")
                    in_dim = current_channels * (current_img_size ** 2)
                else:
                    in_dim = last_layer_units

                mlp_widths, mlp_act_names, mlp_norms = sample_mlp_tf(trial, in_dim, output_dim, "MLP_Head", spaces)
                
                # FIX: Convert activation names to layer objects before passing to the build function.
                mlp_acts = [get_activation_tf(act) for act in mlp_act_names]
                classifier_head = build_mlp_from_config_classifier(mlp_widths, mlp_acts, mlp_norms, name='classifier_head')
                
                model_components.append({
                    'block_type': 'MLP', 'name': 'classifier_head',
                    'params': {'widths': mlp_widths, 'activations': mlp_act_names, 'normalizations': mlp_norms}
                })
                bops += estimate_mlp_bops(mlp_widths)

                input_shape = (img_size, img_size, x_train.shape[-1])
                model = BlockArchitectureTF(feature_extractor_blocks, classifier_head, input_shape, needs_flattening=(not is_flattened))
                
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                train_model(model, (x_train, y_train), (x_val, y_val), epochs=epochs, batch_size=128, verbose=0)
                
                val_metrics = evaluate_model(model, (x_val, y_val))
                performance_metric = val_metrics['accuracy']
                
                model_details = {
                    'metadata': {'trial_id': trial.number, 'global_search_accuracy': float(performance_metric), 'global_search_bops': float(bops)},
                    'architecture': {
                        'model_type': 'BlockBased',
                        'input_shape': list(input_shape),
                        'output_dim': output_dim,
                        'components': model_components
                    }
                }
                trial_yaml_path = os.path.join(self.results_dir, f"trial_{trial.number}_arch.yaml")
                _save_architecture_to_yaml(model_details, trial_yaml_path)
                
                if verbose:
                    print(f"Trial {trial.number}: Accuracy={performance_metric:.4f}, BOPs={bops}")

                self.results.append({
                    'trial': trial.number, 'performance_metric': performance_metric, 'bops': bops, 
                    'params': trial.params, 'yaml_path': trial_yaml_path
                })
                return performance_metric, bops
            
            except Exception as e:
                print(f"Trial {trial.number} failed with error: {e}")
                return 0.0, 1e12

        return objective

    def calculate_mlp_bops_tf(self, model, input_size, bit_width=32):
        """
        Calculate BOPs for MLP architecture using tf_bops utilities.
        """
        # Note: You need to import get_MLP_bops_tf from .tf_bops at the top of the file
        return get_MLP_bops_tf(model, input_shape=(1, input_size), bit_width=bit_width)

    def create_mlp_objective(self, x_train, y_train, x_val, y_val, epochs=10, 
                           use_hardware_metrics=False, verbose=True):
        """
        Creates objective function for MLP optimization.
        NOW MODIFIED to save a block-compatible architecture to YAML.
        """
        def objective(trial):
            # This search space is specific to this simple MLP objective
            mlp_search_space = {
                "num_layers": [2, 3], "hidden_units1": [8, 16, 32, 64, 128],
                "activation1": ["relu", "tanh", "sigmoid"], "batchnorm1": [True, False],
                "hidden_units2": [8, 16, 32, 64], "activation2": ["relu", "tanh", "sigmoid"],
                "batchnorm2": [True, False],
            }

            # Sample architecture configuration
            num_layers = trial.suggest_categorical("num_layers", mlp_search_space["num_layers"])
            config = {
                "num_layers": num_layers,
                "hidden_units1": trial.suggest_categorical("hidden_units1", mlp_search_space["hidden_units1"]),
                "activation1": trial.suggest_categorical("activation1", mlp_search_space["activation1"]),
                "batchnorm1": trial.suggest_categorical("batchnorm1", mlp_search_space["batchnorm1"]),
            }
            if num_layers >= 3:
                config["hidden_units2"] = trial.suggest_categorical("hidden_units2", mlp_search_space["hidden_units2"])
                config["activation2"] = trial.suggest_categorical("activation2", mlp_search_space["activation2"])
                config["batchnorm2"] = trial.suggest_categorical("batchnorm2", mlp_search_space["batchnorm2"])
            
            input_size = x_train.shape[1]
            num_classes = y_train.shape[1]
            from .tf_model_builder import build_mlp_from_config
            model = build_mlp_from_config(config, input_size=input_size, num_classes=num_classes)
            
            train_model(model, (x_train, y_train), (x_val, y_val), 
                        epochs=epochs, batch_size=128, patience=3, verbose=0)
            
            val_metrics = evaluate_model(model, (x_val, y_val))
            performance_metric = val_metrics['accuracy']
            bops = self.calculate_mlp_bops_tf(model, input_size)
            
            # --- NEW: Logic to create and save YAML architecture ---
            # 1. Reconstruct the 2D image shape from the flattened input_size
            img_dim = int(np.sqrt(input_size))
            image_input_shape = [img_dim, img_dim, 1] # Assuming 1 channel for MNIST

            # 2. Translate the simple config into the component-based format
            widths = [input_size, config["hidden_units1"]] # MLP widths start after flattening
            activations = [config["activation1"]]
            normalizations = ['batch' if config["batchnorm1"] else None]
            if num_layers >= 3:
                widths.append(config["hidden_units2"])
                activations.append(config["activation2"])
                normalizations.append('batch' if config["batchnorm2"] else None)
            widths.append(num_classes)
            activations.append('softmax')
            normalizations.append(None)
            
            # 3. Define components, starting with an explicit Flatten layer
            model_components = [
                {'block_type': 'Flatten', 'name': 'initial_flatten', 'params': {}},
                {
                    'block_type': 'MLP', 'name': 'classifier_head',
                    'params': {'widths': widths, 'activations': activations, 'normalizations': normalizations}
                }
            ]
            
            model_details = {
                'metadata': {'trial_id': trial.number, 'global_search_accuracy': float(performance_metric), 'global_search_bops': float(bops)},
                'architecture': {
                    'model_type': 'BlockBased',
                    'input_shape': image_input_shape, # Use the 2D shape here
                    'output_dim': num_classes, 'components': model_components
                }
            }
            
            trial_yaml_path = os.path.join(self.results_dir, f"trial_{trial.number}_arch.yaml")
            _save_architecture_to_yaml(model_details, trial_yaml_path)

            if use_hardware_metrics:
                avg_resource, clock_cycles = self.calculate_hardware_metrics(model, input_size)
            else:
                avg_resource, clock_cycles = 0.0, 0.0
            
            if verbose:
                print(f"Trial {trial.number}: Accuracy={performance_metric:.4f}, BOPs={bops}")
            
            result_data = {
                'trial': trial.number, 'performance_metric': performance_metric, 'bops': bops,
                'params': trial.params, 'yaml_path': trial_yaml_path
            }
            if use_hardware_metrics:
                result_data['avg_resource'] = avg_resource
                result_data['clock_cycles'] = clock_cycles
            self.results.append(result_data)
            
            if use_hardware_metrics:
                return performance_metric, bops, avg_resource, clock_cycles
            return performance_metric, bops
        
        return objective

    def calculate_hardware_metrics(self, model, input_shape):
        """
        Calculate hardware metrics using rule4ml.
        Parameters:
            model: TensorFlow model
            input_shape: The input dimension for the model, required for patching.
            
        Returns:
            tuple: (avg_resource, clock_cycles)
        """
        try:
            from rule4ml.models.estimators import MultiModelEstimator
            
            # Patch model layers with shape info, which is required by rule4ml.
            for layer in model.layers:
                if hasattr(layer, "input_spec") and layer.input_spec and layer.input_spec.shape:
                    layer._build_shapes_dict = {"input": layer.input_spec.shape}
                elif hasattr(layer, "input_shape") and layer.input_shape is not None:
                    layer._build_shapes_dict = {"input": layer.input_shape}
                else:
                    # Fallback for the first layer if shape is not yet inferred
                    if isinstance(input_shape, (list, tuple)):
                        layer._build_shapes_dict = {"input": (None, *input_shape)}
                    else:
                        layer._build_shapes_dict = {"input": (None, input_shape)}

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
                clock_cycles = results.get('CYCLES', 1e9) # High default
            else:
                print("Warning: Hardware estimation failed to return results. Returning high-penalty default values.")
                avg_resource = 100.0
                clock_cycles = 1e9

            return avg_resource, clock_cycles
            
        except ImportError:
            # Re-raise to ensure the user knows rule4ml is missing
            raise ImportError("rule4ml package is required for hardware metrics calculation. Please install it.")
        except Exception as e:
            # Catch other potential errors from the estimator
            print(f"An error occurred during hardware estimation: {e}. Returning high-penalty dummy values.")
            return 100.0, 1e9


    def run_search(self, model_type='block', n_trials=10, epochs=10, dataset='mnist',
                   subset_size=10000, resize_val=11, objectives=None, maximize_flags=None,
                   use_hardware_metrics=False, verbose=True):
        """
        Run global search.
        """
        if verbose:
            print(f"\n{'='*50}\nStarting {model_type.upper()} Global Search on {dataset.upper()}\n{'='*50}\n")

        # MODIFICATION: Set objective names based on hardware flag and model type
        if use_hardware_metrics:
            self.objective_names = objectives or ['performance_metric', 'bops', 'avg_resource', 'clock_cycles']
            self.maximize_flags = maximize_flags or [True, False, False, False]
        else:
            self.objective_names = objectives or ['performance_metric', 'bops']
            self.maximize_flags = maximize_flags or [True, False]

        # MODIFICATION: Conditional data loading based on model type
        is_mlp = (model_type == 'mlp')
        x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
            resize_val=resize_val,
            subset_size=subset_size,
            flatten=is_mlp,  # Flatten for MLP, not for block-based
            one_hot=is_mlp   # One-hot for MLP, not for sparse cross-entropy
        )

        # MODIFICATION: Select the correct objective function
        if model_type == 'mlp':
            objective = self.create_mlp_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
            )
        elif model_type == 'block':
            objective = self.create_block_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'mlp' or 'block'.")

        # Set up optimization directions
        directions = ["maximize" if flag else "minimize" for flag in self.maximize_flags]
        study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
        study.optimize(objective, n_trials=n_trials)

        self.save_results(model_type, study)
        if verbose:
            self.print_best_trials(study)
        return study



    def save_results(self, model_type, study):
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, f"{model_type}_search_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nCSV results saved to {csv_file}")

        # --- NEW: Select and save the best model for local search ---
        if df.empty:
            print("No successful trials to select a best model from.")
            return

        # Selection criteria: highest performance_metric (accuracy)
        best_trial_row = df.loc[df['performance_metric'].idxmax()]
        best_trial_yaml_path = best_trial_row['yaml_path']
        
        destination_path = os.path.join(self.results_dir, "best_model_for_local_search.yaml")
        
        # Copy the best trial's YAML to the standardized filename
        import shutil
        shutil.copy(best_trial_yaml_path, destination_path)
        
        print(f"\n🏆 Best model architecture (Trial {best_trial_row['trial']}) saved for local search:")
        print(f"   - Source: {best_trial_yaml_path}")
        print(f"   - Destination: {destination_path}")
        print(f"   - Accuracy: {best_trial_row['performance_metric']:.4f}")


    def print_best_trials(self, study):
        print("\n" + "="*50 + "\nBEST TRIALS FOUND BY OPTUNA\n" + "="*50)
        for i, trial in enumerate(study.best_trials):
            print(f"\nRank {i+1} (Trial {trial.number}):")
            values_dict = {name: val for name, val in zip(self.objective_names, trial.values)}
            print(f"  Values: {values_dict}")
            print(f"  Params: {trial.params}")

def sample_dense_block_tf(trial, prefix, search_space):
    """Samples parameters for a single Dense layer to be used as a block."""
    mlp_width_space = search_space["mlp_width_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]

    units_idx = trial.suggest_categorical(f"{prefix}_units", list(range(len(mlp_width_space))))
    units = mlp_width_space[units_idx]
    
    act_name = trial.suggest_categorical(f"{prefix}_act", act_space)
    norm = trial.suggest_categorical(f"{prefix}_norm", norm_space)

    return units, act_name, norm





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