import os
import yaml
import optuna
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import datetime

from .tf_model_builder import build_mlp_from_config
from .tf_processor import train_model, evaluate_model, get_model_metrics
from .tf_bops import get_MLP_bops_tf, estimate_conv_bops, estimate_attention_bops, estimate_mlp_bops
from .tf_blocks import ConvAttentionBlock
from .tf_data_preprocessing import load_generic_dataset
from .tf_visualization import plot_pareto_fronts, plot_3d_pareto_front_heatmap

# Helper function to save architecture to YAML
def _save_architecture_to_yaml(model_details, file_path):
    """Saves the model architecture to a YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(model_details, f, default_flow_style=False, sort_keys=False)

def _infer_input_shape_yaml(x):
    """Infer YAML-friendly input shape (no batch dim)."""
    shp = tuple(x.shape)
    if len(shp) <= 1:
        raise ValueError(f"Unexpected x shape: {shp}")
    if len(shp) == 2:
        return [int(shp[1])]          # e.g. qubit: [800]
    return [int(d) for d in shp[1:]]  # e.g. images: [H, W, C]

def _stratified_k_fold_indices(y, n_folds, one_hot=False):
    """Return list of (train_indices, val_indices) for stratified k-fold splitting."""
    if one_hot:
        labels = np.argmax(y, axis=1)
    else:
        labels = y.astype(int).ravel()

    classes = np.unique(labels)
    fold_indices = [[] for _ in range(n_folds)]

    rng = np.random.RandomState(42)
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        splits = np.array_split(cls_indices, n_folds)
        for i, split in enumerate(splits):
            fold_indices[i].extend(split.tolist())

    result = []
    for i in range(n_folds):
        val_idx = np.array(fold_indices[i])
        train_idx = np.concatenate([np.array(fold_indices[j]) for j in range(n_folds) if j != i])
        result.append((train_idx, val_idx))
    return result


def _make_blockbased_components_for_input(input_shape_yaml, mlp_params):
    """
    Build components list for the BlockBased YAML schema.
    Only add Flatten when input is image-like (rank > 1).
    """
    components = []
    if len(input_shape_yaml) > 1:
        components.append({"block_type": "Flatten", "name": "initial_flatten", "params": {}})

    components.append(
        {
            "block_type": "MLP",
            "name": "classifier_head",
            "params": mlp_params,
        }
    )
    return components


def _is_rule4ml_unsupported_activation(layer):
    """Check if a layer is an activation type that rule4ml cannot handle.

    rule4ml only supports ReLU, Softmax, and linear Activation layers.
    LeakyReLU, GELU, and other exotic activations cause feature extraction errors.
    """
    if isinstance(layer, tf.keras.layers.LeakyReLU):
        return True
    if isinstance(layer, tf.keras.layers.Activation):
        act = layer.get_config().get("activation", "")
        if act not in ("linear", "relu", "softmax", "sigmoid", "tanh"):
            return True
    return False


def _flatten_keras_model(model):
    """Rebuild a Keras functional model compatible with rule4ml.

    Two transformations are applied:
    1. Sequential sub-models are expanded into individual layers (rule4ml's
       network parser requires a 'dtype' config key that Sequential lacks).
    2. Unsupported activation layers (LeakyReLU, GELU, etc.) are replaced with
       ReLU.  This is safe because rule4ml only uses layer types and shapes for
       resource estimation — actual weights and activation functions don't affect
       the prediction.
    """
    inputs = model.input
    x = inputs

    def _apply(layer, tensor):
        if _is_rule4ml_unsupported_activation(layer):
            return tf.keras.layers.ReLU()(tensor)
        return layer(tensor)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, tf.keras.Sequential):
            for sub_layer in layer.layers:
                x = _apply(sub_layer, x)
        else:
            x = _apply(layer, x)
    return tf.keras.Model(inputs=inputs, outputs=x, name=model.name + "_flat")


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
        elif normalizations[i] == 'layer':
            layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
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
    """
    High-level orchestrator for TensorFlow global architecture and hyperparameter
    searches using Optuna.

    This class:
    - Loads datasets via ``load_generic_dataset`` with dataset-specific kwargs.
    - Builds Optuna objective functions for block-based and MLP models.
    - Optionally computes hardware metrics via rule4ml when ``use_hardware_metrics``
      is enabled.
    - Accumulates per-trial results in ``self.results`` and writes:
        * Incremental CSV files summarizing all trials for a given run.
        * One YAML architecture file per trial in ``trials_dir``.
        * A ``best_model_for_local_search.yaml`` snapshot for the best trial.
        * Pareto-front plots (and optional 3D heatmaps) into ``results_dir``.

    It is the primary programmatic entrypoint for global search when called
    directly from Python or indirectly via higher-level orchestration layers
    (e.g., an MCP server).
    """
    def __init__(self, search_space_path=None, hls_config=None, results_dir="./results_tf"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.trials_dir = os.path.join(results_dir, "trial_yamls")
        os.makedirs(self.trials_dir, exist_ok=True)
        if isinstance(search_space_path, dict):
            self.search_space = search_space_path
        elif search_space_path and os.path.exists(search_space_path):
            with open(search_space_path, 'r') as f:
                self.search_space = yaml.safe_load(f)
        else:
            self.search_space = self.get_default_search_space()
        self.hls_config = hls_config or self.get_default_hls_config()
        self.results = []
        self.objective_names = []
        self.maximize_flags = []
        # Timestamp for the current run; set at the start of run_search
        self.run_timestamp = None
        # Model type for CSV filename; set at the start of run_search
        self.current_model_type = None
        # CSV file path for incremental writing; set at the start of run_search
        self.csv_file_path = None

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
            "output_dim": 2,
            "output_activation": "softmax"
        }

    def get_default_hls_config(self):
        return {
            "model": {"precision": "ap_fixed<8,3>", "reuse_factor": 1, "strategy": "Latency"},
            "board": "zcu102"
        }

    def create_block_objective(self, x_train, y_train, x_val, y_val, epochs=10, use_hardware_metrics=False, verbose=True, one_hot=False, n_folds=1):
        """
        Create the Optuna objective function for block-based architectures.

        Parameters
        ----------
        x_train, y_train, x_val, y_val
            Training and validation splits returned by ``load_generic_dataset``.
        epochs : int, optional
            Number of epochs to train each trial model for.
        use_hardware_metrics : bool, optional
            If True, also estimate hardware usage via rule4ml and treat it as
            additional objectives (average resource utilization and clock cycles).
        verbose : bool, optional
            If True, print a per-trial summary line (accuracy, BOPs, and hardware
            metrics when available).
        one_hot : bool, optional
            Whether labels are provided in one-hot form; controls loss choice and
            stratified k-fold behavior.
        n_folds : int, optional
            If >1, run stratified k-fold cross-validation inside each trial and
            use the mean validation accuracy as the performance metric.

        Returns
        -------
        callable
            A function ``objective(trial)`` suitable for passing to
            ``optuna.Study.optimize``. It returns:

            - ``(performance_metric, bops)`` when ``use_hardware_metrics=False``.
            - ``(performance_metric, bops, avg_resource, clock_cycles)`` when
              ``use_hardware_metrics=True``.

        Side effects
        ------------
        - Appends a per-trial result dict to ``self.results``.
        - Writes a YAML architecture description for each trial into
          ``self.trials_dir``.
        - Incrementally appends CSV rows via ``_append_trial_to_csv`` when
          ``run_search`` has initialized ``self.csv_file_path``.
        """
        def objective(trial):
            try:
                spaces = self.search_space
                num_blocks = spaces.get("num_blocks", 3)
                img_size = spaces.get("initial_img_size", 11)
                output_dim = spaces.get("output_dim", 10)
                
                bops = 0

                if one_hot:
                    loss_function = "categorical_crossentropy"
                else:
                    loss_function = "sparse_categorical_crossentropy"

                data_is_flat = len(x_train.shape) == 2
                if data_is_flat:
                    is_flattened = True
                    last_layer_units = x_train.shape[1]
                    current_img_size = 0
                    current_channels = 0
                else:
                    is_flattened = False
                    last_layer_units = 0
                    current_img_size = img_size
                    current_channels = x_train.shape[-1]
                
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

                # Apply output activation from search space config (e.g. "softmax" for classification, "linear" for regression)
                output_activation = spaces.get("output_activation", "softmax")
                if output_activation:
                    mlp_act_names[-1] = output_activation
                    mlp_norms[-1] = None

                mlp_acts = [get_activation_tf(act) for act in mlp_act_names]
                classifier_head = build_mlp_from_config_classifier(mlp_widths, mlp_acts, mlp_norms, name='classifier_head')
                
                model_components.append({
                    'block_type': 'MLP', 'name': 'classifier_head',
                    'params': {'widths': mlp_widths, 'activations': mlp_act_names, 'normalizations': mlp_norms}
                })
                bops += estimate_mlp_bops(mlp_widths)

                input_shape = tuple(x_train.shape[1:])
                model = BlockArchitectureTF(feature_extractor_blocks, classifier_head, input_shape, needs_flattening=(not is_flattened))

                if n_folds > 1:
                    # Combine train+val into a single pool for k-fold splitting
                    x_all = np.concatenate([x_train, x_val], axis=0)
                    y_all = np.concatenate([y_train, y_val], axis=0)
                    fold_indices = _stratified_k_fold_indices(y_all, n_folds, one_hot=one_hot)

                    fold_accuracies = []
                    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(fold_indices):
                        xf_train, yf_train = x_all[fold_train_idx], y_all[fold_train_idx]
                        xf_val, yf_val = x_all[fold_val_idx], y_all[fold_val_idx]

                        # clone_model creates fresh random weights, same architecture
                        fold_model = tf.keras.models.clone_model(model.model)
                        fold_model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
                        train_model(fold_model, (xf_train, yf_train), (xf_val, yf_val),
                                    epochs=epochs, batch_size=128, verbose=0)
                        fold_metrics = evaluate_model(fold_model, (xf_val, yf_val))
                        fold_accuracies.append(fold_metrics['accuracy'])

                    performance_metric = np.mean(fold_accuracies)
                else:
                    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
                    train_model(model, (x_train, y_train), (x_val, y_val),
                                epochs=epochs, batch_size=128, verbose=0)
                    val_metrics = evaluate_model(model, (x_val, y_val))
                    performance_metric = val_metrics['accuracy']

                if use_hardware_metrics:
                    flat_model = _flatten_keras_model(model.model)
                    avg_resource, clock_cycles = self.calculate_hardware_metrics(flat_model, input_shape)

                model_details = {
                    'metadata': {'trial_id': trial.number, 'global_search_accuracy': float(performance_metric), 'global_search_bops': float(bops)},
                    'architecture': {
                        'model_type': 'BlockBased',
                        'input_shape': list(input_shape),
                        'output_dim': output_dim,
                        'components': model_components
                    }
                }
                trial_yaml_path = os.path.join(self.trials_dir, f"trial_{trial.number}_arch.yaml")
                _save_architecture_to_yaml(model_details, trial_yaml_path)
                
                if verbose:
                    base_msg = f"Trial {trial.number}"
                    if n_folds > 1:
                        fold_str = ", ".join(f"{a:.4f}" for a in fold_accuracies)
                        base_msg += f": Folds=[{fold_str}], MeanAcc={performance_metric:.4f}, BOPs={bops}"
                    else:
                        base_msg += f": Accuracy={performance_metric:.4f}, BOPs={bops}"
                    if use_hardware_metrics:
                        base_msg += f", AvgResource={avg_resource:.2f}%, Cycles={clock_cycles}"
                    print(base_msg)

                result_data = {
                    'trial': trial.number, 'performance_metric': performance_metric, 'bops': bops,
                    'params': trial.params, 'yaml_path': trial_yaml_path
                }
                if use_hardware_metrics:
                    result_data['avg_resource'] = avg_resource
                    result_data['clock_cycles'] = clock_cycles
                self.results.append(result_data)
                # Write trial to CSV immediately
                self._append_trial_to_csv(result_data)

                if use_hardware_metrics:
                    return performance_metric, bops, avg_resource, clock_cycles
                return performance_metric, bops

            except Exception as e:
                print(f"Trial {trial.number} failed with error: {e}")
                if use_hardware_metrics:
                    return 0.0, 1e12, 100.0, 1e9
                return 0.0, 1e12

        return objective

    def calculate_mlp_bops_tf(self, model, input_size, bit_width=32):
        """
        Calculate BOPs for MLP architecture using tf_bops utilities.
        """
        return get_MLP_bops_tf(model, input_shape=(1, input_size), bit_width=bit_width)


    def create_mlp_objective(self, x_train, y_train, x_val, y_val, epochs=10,
                            use_hardware_metrics=False, verbose=True):
        """
        Create the Optuna objective function for simple MLP architectures.

        This objective is tailored for dense-only models and mirrors the
        block-based pipeline by:
        - Sampling a shallow MLP configuration from a fixed search space.
        - Training the model on the provided train/validation splits.
        - Computing accuracy, BOPs, and optional hardware metrics.
        - Emitting a BlockBased-compatible YAML architecture description.

        Parameters
        ----------
        x_train, y_train, x_val, y_val
            Training and validation splits returned by ``load_generic_dataset``.
        epochs : int, optional
            Number of epochs to train each trial model for.
        use_hardware_metrics : bool, optional
            If True, estimate hardware usage via rule4ml and include average
            resource utilization and clock cycles in the objective values and
            per-trial results.
        verbose : bool, optional
            If True, print a per-trial summary line (accuracy and BOPs).

        Returns
        -------
        callable
            A function ``objective(trial)`` suitable for passing to
            ``optuna.Study.optimize``. It returns:

            - ``(performance_metric, bops)`` when ``use_hardware_metrics=False``.
            - ``(performance_metric, bops, avg_resource, clock_cycles)`` when
              ``use_hardware_metrics=True``.

        Side effects
        ------------
        - Appends a per-trial result dict to ``self.results``.
        - Writes a BlockBased-style YAML architecture for each trial to
          ``self.trials_dir``.
        - Incrementally appends CSV rows via ``_append_trial_to_csv`` when
          ``run_search`` has initialized ``self.csv_file_path``.
        """
        def objective(trial):
            # This search space is specific to this simple MLP objective
            mlp_search_space = {
                "num_layers": [2, 3, 4, 5], "hidden_units1": [8, 16, 32, 64, 128],
                "activation1": ["relu", "tanh", "sigmoid"], "batchnorm1": [True, False],
                "hidden_units2": [8, 16, 32, 64], "activation2": ["relu", "tanh", "sigmoid"],
                "batchnorm2": [True, False],
                "hidden_units3": [8, 16, 32], "activation3": ["relu", "tanh", "sigmoid"],
                "batchnorm3": [True, False],
                "hidden_units4": [8, 16], "activation4": ["relu", "tanh", "sigmoid"],
                "batchnorm4": [True, False],
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
            if num_layers >= 4:
                config["hidden_units3"] = trial.suggest_categorical("hidden_units3", mlp_search_space["hidden_units3"])
                config["activation3"] = trial.suggest_categorical("activation3", mlp_search_space["activation3"])
                config["batchnorm3"] = trial.suggest_categorical("batchnorm3", mlp_search_space["batchnorm3"])
            if num_layers >= 5:
                config["hidden_units4"] = trial.suggest_categorical("hidden_units4", mlp_search_space["hidden_units4"])
                config["activation4"] = trial.suggest_categorical("activation4", mlp_search_space["activation4"])
                config["batchnorm4"] = trial.suggest_categorical("batchnorm4", mlp_search_space["batchnorm4"])

            input_size = x_train.shape[1]
            num_classes = y_train.shape[1]
            from .tf_model_builder import build_mlp_from_config
            model = build_mlp_from_config(config, input_size=input_size, num_classes=num_classes)

            train_model(
                model, (x_train, y_train), (x_val, y_val),
                epochs=epochs, batch_size=128, patience=3, verbose=0
            )

            val_metrics = evaluate_model(model, (x_val, y_val))
            performance_metric = val_metrics["accuracy"]
            bops = self.calculate_mlp_bops_tf(model, input_size)

            # --- REPLACE THE MNIST-SPECIFIC YAML LOGIC WITH THIS ---
            input_shape_yaml = _infer_input_shape_yaml(x_train)

            widths = [input_size, config["hidden_units1"]]
            activations = [config["activation1"]]
            normalizations = ["batch" if config["batchnorm1"] else None]

            if num_layers >= 3:
                widths.append(config["hidden_units2"])
                activations.append(config["activation2"])
                normalizations.append("batch" if config["batchnorm2"] else None)
            if num_layers >= 4:
                widths.append(config["hidden_units3"])
                activations.append(config["activation3"])
                normalizations.append("batch" if config["batchnorm3"] else None)
            if num_layers >= 5:
                widths.append(config["hidden_units4"])
                activations.append(config["activation4"])
                normalizations.append("batch" if config["batchnorm4"] else None)

            widths.append(num_classes)
            activations.append("softmax")
            normalizations.append(None)

            model_components = []
            if len(input_shape_yaml) > 1:
                model_components.append({"block_type": "Flatten", "name": "initial_flatten", "params": {}})

            model_components.append(
                {
                    "block_type": "MLP",
                    "name": "classifier_head",
                    "params": {"widths": widths, "activations": activations, "normalizations": normalizations},
                }
            )

            model_details = {
                "metadata": {
                    "trial_id": trial.number,
                    "global_search_accuracy": float(performance_metric),
                    "global_search_bops": float(bops),
                },
                "architecture": {
                    "model_type": "BlockBased",
                    "input_shape": input_shape_yaml,
                    "output_dim": int(num_classes),
                    "components": model_components,
                },
            }

            trial_yaml_path = os.path.join(self.trials_dir, f"trial_{trial.number}_arch.yaml")
            _save_architecture_to_yaml(model_details, trial_yaml_path)
            # --- END REPLACEMENT ---

            if use_hardware_metrics:
                avg_resource, clock_cycles = self.calculate_hardware_metrics(model, input_size)
            else:
                avg_resource, clock_cycles = 0.0, 0.0

            if verbose:
                print(f"Trial {trial.number}: Accuracy={performance_metric:.4f}, BOPs={bops}")

            result_data = {
                "trial": trial.number,
                "performance_metric": performance_metric,
                "bops": bops,
                "params": trial.params,
                "yaml_path": trial_yaml_path,
            }
            if use_hardware_metrics:
                result_data["avg_resource"] = avg_resource
                result_data["clock_cycles"] = clock_cycles
            self.results.append(result_data)
            # Write trial to CSV immediately
            self._append_trial_to_csv(result_data)

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
        # try:
        #     from rule4ml.models.estimators import MultiModelEstimator
            
        #     # Patch model layers with shape info, which is required by rule4ml.
        #     for layer in model.layers:
        #         if hasattr(layer, "input_spec") and layer.input_spec and layer.input_spec.shape:
        #             layer._build_shapes_dict = {"input": layer.input_spec.shape}
        #         elif hasattr(layer, "input_shape") and layer.input_shape is not None:
        #             layer._build_shapes_dict = {"input": layer.input_shape}
        #         else:
        #             # Fallback for the first layer if shape is not yet inferred
        #             if isinstance(input_shape, (list, tuple)):
        #                 layer._build_shapes_dict = {"input": (None, *input_shape)}
        #             else:
        #                 layer._build_shapes_dict = {"input": (None, input_shape)}

        #     estimator = MultiModelEstimator()
        #     estimator.load_default_models()
            
        #     pred_df = estimator.predict([model], [self.hls_config])

        try:
            from rule4ml.models.wrappers import MultiModelWrapper
            
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

            estimator = MultiModelWrapper()
            estimator.load_default_models()

            pred_df = estimator.predict([model], hls_configs=[self.hls_config])
            
            if not pred_df.empty:
                results = pred_df.iloc[0]
                lut = results.get("LUT (%)", 0)
                ff = results.get("FF (%)", 0)
                bram = results.get("BRAM (%)", 0)
                dsp = results.get("DSP (%)", 0)
                avg_resource = np.mean([lut, ff, bram, dsp])
                # Rule4ML may return 'CYCLES' or 'Cycles' depending on version
                clock_cycles = results.get('CYCLES', results.get('Cycles', 1e9))
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


    def run_search( # changed so that dataset_kwargs could be accepted
        self,
        model_type='block',
        n_trials=10,
        epochs=10,
        dataset='mnist',
        subset_size=10000,
        resize_val=11,
        objectives=None,
        maximize_flags=None,
        use_hardware_metrics=False,
        verbose=True,
        one_hot=False,
        n_folds=1,
        callbacks=None,  # <-- NEW: Optuna callbacks for visualization
        storage=None,  # <-- NEW: Optuna storage backend (for multi-node)
        study_name=None,  # <-- NEW: Optuna study name (for multi-node)
        **dataset_kwargs,  # <-- NEW: forward arbitrary dataset loader args
    ):
        """
        Run a full Optuna-based global search over block-based or MLP models.

        This method:
        - Loads the requested dataset via ``load_generic_dataset`` with sensible
          defaults plus any extra ``dataset_kwargs``.
        - Constructs an Optuna study (in-memory or backed by an external storage).
        - Optimizes the appropriate objective function for the chosen
          ``model_type`` ("block" or "mlp").
        - Streams per-trial results into an incremental CSV file in
          ``self.results_dir``.
        - After optimization, rewrites the CSV with the in-memory ``self.results``,
          selects a best trial, saves its architecture to
          ``best_model_for_local_search.yaml``, and plots Pareto fronts.

        Parameters
        ----------
        model_type : {"block", "mlp"}, optional
            Search mode. "block" uses the block-based objective, "mlp" uses the
            simpler dense-only objective.
        n_trials : int, optional
            Number of Optuna trials to run.
        epochs : int, optional
            Number of epochs to train each trial model for.
        dataset : str, optional
            Name of the dataset to load via ``load_generic_dataset``.
        subset_size : int, optional
            Number of training examples to use; forwarded to the loader.
        resize_val : int, optional
            Image resize parameter used only for certain datasets
            (e.g., "mnist", "fashion_mnist").
        objectives : list of str, optional
            Names of objective values in the order returned by the objective
            function. Defaults are chosen based on ``use_hardware_metrics``.
        maximize_flags : list of bool, optional
            For each objective in ``objectives``, whether to maximize (True) or
            minimize (False). Used to derive Optuna ``directions``.
        use_hardware_metrics : bool, optional
            If True, augment objectives with average FPGA resource utilization
            and clock cycles via rule4ml. For block-based searches, the search
            space must not include "ConvAttn" blocks in ``block_types``.
        verbose : bool, optional
            If True, print high-level run information and per-trial summaries.
        one_hot : bool, optional
            Whether labels should be treated as one-hot in the block-based
            objective (MLP objective always uses one-hot labels).
        n_folds : int, optional
            Number of folds for stratified k-fold validation in the block-based
            objective. ``n_folds=1`` disables k-fold and uses a single validation
            split.
        callbacks : list of callable, optional
            Optuna callbacks passed directly to ``study.optimize``.
        storage : str or optuna.storages.BaseStorage, optional
            Optuna storage backend (e.g., RDB URL) for multi-node / resumable
            studies. When set, each worker writes a rank-specific CSV.
        study_name : str, optional
            Name of the Optuna study when ``storage`` is provided.
        **dataset_kwargs
            Additional keyword arguments forwarded into ``load_generic_dataset``
            (e.g., data directory, qubit window specification).

        Returns
        -------
        optuna.Study
            The completed Optuna study object.

        Side effects
        ------------
        - Populates ``self.results`` with a dict per successful trial.
        - Writes or overwrites a CSV summary at ``self.csv_file_path``.
        - Writes per-trial YAML architectures in ``self.trials_dir``.
        - Copies the best-performing trial's YAML to
          ``best_model_for_local_search.yaml`` in ``self.results_dir``.
        - Generates Pareto-front plots (and 3D heatmaps when applicable) into
          ``self.results_dir``.
        """
        # Record a timestamp for this run (used in CSV metadata)
        self.run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Store model type and CSV path for incremental writing
        self.current_model_type = model_type
        # Multi-node: each worker writes its own CSV to avoid append races; merge after job
        if storage is not None:
            rank = os.environ.get("SLURM_PROCID", "0")
            self.csv_file_path = os.path.join(self.results_dir, f"{model_type}_search_results_rank{rank}.csv")
        else:
            self.csv_file_path = os.path.join(self.results_dir, f"{model_type}_search_results.csv")
        # Store whether to include hardware metrics for CSV headers
        self.use_hardware_metrics_for_csv = use_hardware_metrics
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            # Create empty DataFrame with expected columns to write headers only
            columns = ['trial', 'performance_metric', 'bops', 'params', 'yaml_path']
            if use_hardware_metrics:
                columns.extend(['avg_resource', 'clock_cycles'])
            columns.append('run_timestamp')
            df_headers = pd.DataFrame(columns=columns)
            df_headers.to_csv(self.csv_file_path, index=False)
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

        # Build kwargs for loader in a dataset-safe way
        loader_kwargs = dict(
            subset_size=subset_size,
            flatten=is_mlp,
            one_hot=is_mlp or one_hot,
        )
        # Only MNIST-like datasets accept resize_val
        if dataset in ("mnist", "fashion_mnist"):
            loader_kwargs["resize_val"] = resize_val

        # Allow caller to override/extend (e.g. data_dir, window_size for qubit)
        loader_kwargs.update(dataset_kwargs or {})

        x_train, y_train, x_val, y_val = load_generic_dataset(
            dataset_name=dataset,
            **loader_kwargs,
        )

        # Validate search space compatibility with hardware metrics
        if use_hardware_metrics and model_type == 'block':
            block_types = self.search_space.get("block_types", [])
            if "ConvAttn" in block_types:
                raise ValueError(
                    "ConvAttn blocks are not supported by rule4ml and cannot be used with "
                    "use_hardware_metrics=True. Remove 'ConvAttn' from block_types in your "
                    "search space, or set use_hardware_metrics=False."
                )

        # MODIFICATION: Select the correct objective function
        if model_type == 'mlp':
            objective = self.create_mlp_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose
            )
        elif model_type == 'block':
            objective = self.create_block_objective(
                x_train, y_train, x_val, y_val, epochs, use_hardware_metrics, verbose, one_hot, n_folds,
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported. Use 'mlp' or 'block'.")

        # Set up optimization directions
        directions = ["maximize" if flag else "minimize" for flag in self.maximize_flags]
        
        # Multi-node support, use storage backend
        if storage is not None:
            # Storage provided, try to load existing study or create new one
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                if verbose:
                    print(f"Loaded existing Optuna study: {study_name}")
            except KeyError:
                # Study doesn't exist, create it
                study = optuna.create_study(
                    directions=directions,
                    sampler=optuna.samplers.NSGAIISampler(),
                    storage=storage,
                    study_name=study_name
                )
                if verbose:
                    print(f"Created new Optuna study: {study_name}")
        else:
            # No storage - single-node in-memory study
            study = optuna.create_study(directions=directions, sampler=optuna.samplers.NSGAIISampler())
            if verbose:
                print("Using in-memory Optuna study (single-node mode)")
        
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)


        self.save_results(model_type, study)
        if verbose:
            self.print_best_trials(study)
        return study



    def _append_trial_to_csv(self, result_data):
        """Append a single trial result to the CSV file incrementally."""
        if self.csv_file_path is None:
            return  # CSV not initialized yet
        
        trial_data = result_data.copy()
        if self.run_timestamp is not None:
            trial_data['run_timestamp'] = self.run_timestamp
        
        df_trial = pd.DataFrame([trial_data])
        file_exists = os.path.exists(self.csv_file_path)
        write_header = not file_exists or os.path.getsize(self.csv_file_path) == 0
        df_trial.to_csv(self.csv_file_path, mode='a', header=write_header, index=False)
    
    def save_results(self, model_type, study):
        # Results are already written incrementally; overwrite with full in-memory data for this worker
        df = pd.DataFrame(self.results)
        if self.run_timestamp is not None:
            df["run_timestamp"] = self.run_timestamp
        # Use self.csv_file_path (rank-specific in multi-node, single file in single-node)
        csv_file = self.csv_file_path
        if csv_file:
            df.to_csv(csv_file, index=False)
            print(f"\nCSV results saved to {csv_file} ({len(df)} trials)")

        # Best model: only in single-node (multi-node merge step will create it from merged CSV)
        if "_rank" in (csv_file or ""):
            return
        if df.empty:
            print("No successful trials to select a best model from.")
            return

        best_trial_row = df.loc[df['performance_metric'].idxmax()]
        best_trial_yaml_path = best_trial_row['yaml_path']
        destination_path = os.path.join(self.results_dir, "best_model_for_local_search.yaml")
        import shutil
        shutil.copy(best_trial_yaml_path, destination_path)
        print(f"\n🏆 Best model architecture (Trial {best_trial_row['trial']}) saved for local search:")
        print(f"   - Source: {best_trial_yaml_path}")
        print(f"   - Destination: {destination_path}")
        print(f"   - Accuracy: {best_trial_row['performance_metric']:.4f}")

        # Plot all pairwise Pareto fronts; add 3D heatmap when hardware metrics are enabled
        obj_info = list(zip(self.objective_names, self.maximize_flags))
        plot_pareto_fronts(df, obj_info, save_dir=self.results_dir)
        if len(self.objective_names) >= 4:
            plot_3d_pareto_front_heatmap(df, obj_info, save_dir=self.results_dir)


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




def run_mlp_search(
    search_space_path=None,
    results_dir="./results_tf",
    n_trials=20,
    epochs=5,
    dataset="mnist",
    subset_size=5000,
    resize_val=8,
    use_hardware_metrics=False,
    dataset_kwargs=None,
):
    """
    Convenience function to run a global search restricted to MLP architectures.

    This is a thin wrapper around ``GlobalSearchTF.run_search`` with
    ``model_type="mlp"`` and a slightly different default configuration. It is
    useful for quick experiments or when you want to treat MLP search as a
    simple callable without manually instantiating ``GlobalSearchTF``.

    Parameters
    ----------
    search_space_path : str or dict, optional
        Path to a YAML search-space file or an in-memory search-space dict.
        When omitted, ``GlobalSearchTF.get_default_search_space`` is used.
    results_dir : str, optional
        Directory where CSV, YAML, and plot artifacts will be written.
    n_trials : int, optional
        Number of Optuna trials to run.
    epochs : int, optional
        Number of epochs to train each MLP candidate.
    dataset : str, optional
        Name of the dataset to load via ``load_generic_dataset``.
    subset_size : int, optional
        Number of training examples to use; forwarded to the loader.
    resize_val : int, optional
        Image resize parameter used only for certain datasets
        (e.g., "mnist", "fashion_mnist").
    use_hardware_metrics : bool, optional
        If True, compute and include hardware metrics via rule4ml.
    dataset_kwargs : dict, optional
        Extra keyword arguments forwarded into ``load_generic_dataset``
        (e.g., ``start_location``, ``window_size``, ``data_dir``).

    Returns
    -------
    tuple
        ``(study, searcher)`` where ``study`` is the completed Optuna study and
        ``searcher`` is the ``GlobalSearchTF`` instance that produced it. The
        ``searcher`` exposes ``results_dir``, CSV/YAML paths, and helper methods
        for downstream workflows such as local search.
    """
    searcher = GlobalSearchTF(search_space_path=search_space_path, results_dir=results_dir)

    dataset_kwargs = dataset_kwargs or {}

    # Only pass resize_val for image datasets that accept it.
    run_search_kwargs = dict(
        model_type="mlp",
        n_trials=n_trials,
        epochs=epochs,
        dataset=dataset,
        subset_size=subset_size,
        use_hardware_metrics=use_hardware_metrics,
    )
    if dataset in ("mnist", "fashion_mnist"):
        run_search_kwargs["resize_val"] = resize_val

    # Forward qubit/other dataset loader settings via **kwargs by adding support in run_search next if needed.
    # For now: simplest is to store in searcher and have run_search pass through;
    # but a minimal change is to modify run_search signature to accept dataset_kwargs.
    study = searcher.run_search(**run_search_kwargs, **dataset_kwargs)

    return study, searcher