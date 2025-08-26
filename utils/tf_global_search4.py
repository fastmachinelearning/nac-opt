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
    if act_name is None or act_name == "Identity":
        return tf.keras.layers.Activation('linear')
    elif act_name == "ReLU":
        return tf.keras.layers.ReLU()
    elif act_name == "LeakyReLU":
        return tf.keras.layers.LeakyReLU(negative_slope=0.01)
    elif act_name == "GELU":
        return tf.keras.layers.Activation('gelu')
    else:
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
    acts = [get_activation_tf(trial.suggest_categorical(f"{prefix}_acts_{i}", act_space)) for i in range(num_layers)]
    norms = [trial.suggest_categorical(f"{prefix}_norms_{i}", norm_space) for i in range(num_layers)]
    return channels, kernels, acts, norms


def sample_conv_attention_tf(trial, prefix, search_space):
    hidden_channel_space = search_space["conv_attn"]["hidden_channel_space"]
    act_space = search_space["act_space"]
    hidden_channels_idx = trial.suggest_categorical(f"{prefix}_hiddenchannel", list(range(len(hidden_channel_space))))
    hidden_channels = hidden_channel_space[hidden_channels_idx]
    act = get_activation_tf(trial.suggest_categorical(f"{prefix}_act", act_space))
    return hidden_channels, act


def sample_mlp_tf(trial, in_dim, out_dim, prefix, search_space, num_layers=3):
    mlp_width_space = search_space["mlp_width_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]
    widths = [in_dim]
    for i in range(num_layers - 1):
        width_idx = trial.suggest_categorical(f"{prefix}_width_{i}", list(range(len(mlp_width_space))))
        widths.append(mlp_width_space[width_idx])
    widths.append(out_dim)
    acts = [get_activation_tf(trial.suggest_categorical(f"{prefix}_acts_{i}", act_space)) for i in range(num_layers)]
    norms = [trial.suggest_categorical(f"{prefix}_norms_{i}", norm_space) for i in range(num_layers)]
    return widths, acts, norms


def build_mlp_from_config_tf(widths, acts, norms, name='mlp'):
    layers = []
    for i in range(len(acts)):
        layers.append(tf.keras.layers.Dense(widths[i+1], name=f'{name}_dense_{i}'))
        if norms[i] == 'batch':
            layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        elif norms[i] == 'layer':
            layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
        if acts[i] is not None:
            layers.append(acts[i])
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
                
                feature_extractor_blocks = []
                bops = 0
                
                # --- State Tracking Variables ---
                current_img_size = img_size
                current_channels = x_train.shape[-1]
                is_flattened = False
                last_layer_units = 0

                block_types = [trial.suggest_categorical(f"b{i}", spaces["block_types"]) for i in range(num_blocks)]

                for i, block_type in enumerate(block_types):
                    # If the model is already flat, we cannot add more 2D blocks.
                    if is_flattened and block_type in ["Conv", "ConvAttn"]:
                        continue

                    if not is_flattened and current_img_size <= 0:
                        break

                    if block_type == "Conv":
                        channels, kernels, acts, norms = sample_conv_block_tf(trial, f"b{i}_Conv", current_channels, spaces)
                        size_reduction = sum((k - 1) for k in kernels if k > 1)
                        if current_img_size - size_reduction <= 0:
                            kernels = [1] * len(kernels)
                            size_reduction = 0
                        
                        conv_block = create_conv_block_tf(channels, kernels, acts, norms, name=f'conv_block_{i}')
                        feature_extractor_blocks.append(conv_block)
                        
                        current_img_size -= size_reduction
                        current_channels = channels[-1]
                        bops += estimate_conv_bops(channels, kernels, current_img_size)

                    elif block_type == "MLP":
                        if not is_flattened:
                            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=f'initial_flatten'))
                            last_layer_units = current_channels * (current_img_size ** 2)
                            is_flattened = True
                        
                        units, act, norm = sample_dense_block_tf(trial, f"b{i}_MLP", spaces)
                        
                        dense_layer = tf.keras.layers.Dense(units, name=f'mlp_block_{i}_dense')
                        feature_extractor_blocks.append(dense_layer)
                        
                        if norm == 'batch':
                            feature_extractor_blocks.append(tf.keras.layers.BatchNormalization(name=f'mlp_block_{i}_bn'))
                        if act:
                            feature_extractor_blocks.append(act)
                        
                        bops += estimate_mlp_bops([last_layer_units, units])
                        last_layer_units = units

                # --- Final Classifier Head ---
                if not is_flattened:
                    if current_img_size <= 0:
                        raise optuna.exceptions.TrialPruned("Image size became non-positive.")
                    in_dim = current_channels * (current_img_size ** 2)
                else:
                    in_dim = last_layer_units

                mlp_widths, mlp_acts, mlp_norms = sample_mlp_tf(trial, in_dim, output_dim, "MLP_Head", spaces)
                classifier_head = build_mlp_from_config_tf(mlp_widths, mlp_acts, mlp_norms, name='classifier_head')
                bops += estimate_mlp_bops(mlp_widths)

                # --- Build, Compile, and Train the Model ---
                input_shape = (img_size, img_size, x_train.shape[-1])
                model = BlockArchitectureTF(feature_extractor_blocks, classifier_head, input_shape, needs_flattening=(not is_flattened))
                
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                train_model(model, (x_train, y_train), (x_val, y_val), epochs=epochs, batch_size=128, verbose=0)
                
                val_metrics = evaluate_model(model, (x_val, y_val))
                performance_metric = val_metrics['accuracy']
                
                if verbose:
                    print(f"Trial {trial.number}: Accuracy={performance_metric:.4f}, BOPs={bops}")

                if use_hardware_metrics:
                    try:
                        # The model object is a wrapper; the actual Keras model is .model
                        # The input_shape is already defined as (H, W, C)
                        avg_resource, clock_cycles = self.calculate_hardware_metrics(model.model, input_shape)
                    except Exception as e:
                        if verbose:
                            print(f"Trial {trial.number} warning: Hardware estimation failed: {e}. Using dummy values.")
                        # Use high-penalty dummy values for failed hardware estimation
                        avg_resource = 100.0  # Represents 100% resource usage
                        clock_cycles = 1e9    # A large number of cycles

                    self.results.append({
                        'trial': trial.number, 'performance_metric': performance_metric, 'bops': bops,
                        'avg_resource': avg_resource, 'clock_cycles': clock_cycles, 'params': trial.params
                    })
                    return performance_metric, bops, avg_resource, clock_cycles

                else:
                    self.results.append({
                        'trial': trial.number, 'performance_metric': performance_metric, 
                        'bops': bops, 'params': trial.params
                    })
                    return performance_metric, bops
                
            
            except Exception as e:
                if use_hardware_metrics:
                    return 0.0, 1e12, 100.0, 1e9 # Return 4 values for failure
                else:
                    return 0.0, 1e12 # Return 2 values for failure

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
        if verbose:
            print(f"\n{'='*50}\nStarting {model_type.upper()} Global Search on {dataset.upper()}\n{'='*50}\n")
        
        if use_hardware_metrics:
            self.objective_names = objectives or ['performance_metric', 'bops', 'avg_resource', 'clock_cycles']
            self.maximize_flags = maximize_flags or [True, False, False, False]
        else:
            # If not using hardware, only optimize for performance and BOPs
            self.objective_names = objectives or ['performance_metric', 'bops']
            self.maximize_flags = maximize_flags or [True, False]

        x_train, y_train, x_val, y_val = load_and_preprocess_mnist(
            resize_val=resize_val, 
            subset_size=subset_size, 
            flatten=False, # Always need image format for this search
            one_hot=False
        )

        objective = self.create_block_objective(
            x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
        )

        
        directions = ["maximize" if flag else "minimize" for flag in self.maximize_flags]
        study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
        study.optimize(objective, n_trials=n_trials)

        self.save_results(model_type)
        if verbose:
            self.print_best_trials(study)
        return study

    def save_results(self, model_type):
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(self.results_dir, f"{model_type}_search_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nCSV results saved to {csv_file}")

    def print_best_trials(self, study):
        print("\n" + "="*50 + "\nBEST TRIALS FOUND BY OPTUNA\n" + "="*50)
        for i, trial in enumerate(study.best_trials):
            print(f"\nRank {i+1} (Trial {trial.number}):")
            values_dict = {name: val for name, val in zip(self.objective_names, trial.values)}
            print(f"  Values: {values_dict}")
            print(f"  Params: {trial.params}")


# ----------------------------------

def sample_dense_block_tf(trial, prefix, search_space):
    """Samples parameters for a single Dense layer to be used as a block."""
    mlp_width_space = search_space["mlp_width_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]

    units_idx = trial.suggest_categorical(f"{prefix}_units", list(range(len(mlp_width_space))))
    units = mlp_width_space[units_idx]
    
    act = get_activation_tf(trial.suggest_categorical(f"{prefix}_act", act_space))
    norm = trial.suggest_categorical(f"{prefix}_norm", norm_space)

    return units, act, norm

def load_block_search_space(yaml_path):
    """
    Load block search space configuration from YAML file.
    
    Parameters:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing search space configuration
        
    Example YAML structure:
    ```yaml
    search_spaces:
      channel_space: [4, 8, 16, 32, 64]
      mlp_width_space: [4, 8, 16, 32, 64]
      kernel_space: [1, 3, 5]
      act_space:
        - "ReLU"
        - "LeakyReLU"
        - "GELU"
        - "Identity"
      norm_space: [null, "batch", "layer"]
      block_types: ["Conv", "ConvAttn", "None"]
      conv_attn:
        hidden_channel_space: [1, 2, 4, 8, 16, 32]

    hyperparameters:
      num_blocks: 3
      initial_img_size: 11
      output_dim: 2
    ```
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten the structure for easier access
    search_space = config.get('search_spaces', {})
    hyperparams = config.get('hyperparameters', {})
    
    # Merge hyperparameters into search space
    search_space.update(hyperparams)
    
    return search_space