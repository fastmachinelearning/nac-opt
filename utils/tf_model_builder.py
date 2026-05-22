"""
TensorFlow model builder utilities for creating various architectures.
Supports building models from configurations and search spaces.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LayerNormalization, LeakyReLU
import yaml


def _make_mlp_activation_layer(name):
    """
    Convert an activation name (lowercase or Title-cased) to a Keras layer.

    Returns None when the name is None or "Identity" so the caller can skip
    adding an activation layer. The planner emits names matching tf_blocks
    (``ReLU``, ``LeakyReLU``, ``GELU``, ``Identity``); tutorials emit Keras
    lowercase names (``relu``, ``tanh``). Both shapes are normalized here.
    """
    if name is None:
        return None
    key = str(name).strip().lower().replace("_", "")
    if key in ("", "none", "identity", "linear"):
        return None
    if key == "leakyrelu":
        return LeakyReLU(alpha=0.01)
    return Activation(key)


def load_yaml_config(yaml_path):
    """
    Loads configuration from a YAML file.
    
    Parameters:
        yaml_path (str): Path to the YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_activation_layer(activation_name):
    """
    Returns the appropriate activation layer based on name.
    
    Parameters:
        activation_name (str or None): Name of the activation function
        
    Returns:
        Activation layer or None
    """
    if activation_name is None:
        return None
    elif activation_name.lower() == 'identity':
        return Activation('linear')
    else:
        return Activation(activation_name)


def build_mlp_from_config(
    config,
    input_size=None,
    num_classes=None,
    learning_rate=0.001,
    output_activation="softmax",
):
    """
    Build a compiled MLP from a list-based trial config.

    Config schema (as produced by tf_global_search.GlobalSearchTF objective):
        - hidden_units: list[int] of hidden layer widths
        - activations: list[str|None] aligned with hidden_units
        - normalizations: list["batch"|None] aligned with hidden_units

    `num_classes` and `output_activation` control the appended output layer.
    """
    model = Sequential(name="MLP_Model")

    if input_size is not None:
        model.add(Input(shape=(input_size,)))

    hidden_units = list(config.get("hidden_units", []))
    activations = list(config.get("activations", [None] * len(hidden_units)))
    normalizations = list(config.get("normalizations", [None] * len(hidden_units)))

    if len(activations) != len(hidden_units) or len(normalizations) != len(hidden_units):
        raise ValueError(
            "build_mlp_from_config: hidden_units, activations, and normalizations must be the same length. "
            f"Got hidden_units={len(hidden_units)}, activations={len(activations)}, normalizations={len(normalizations)}."
        )

    for units, act, norm in zip(hidden_units, activations, normalizations):
        if norm == "batch":
            model.add(Dense(units, use_bias=False))
            model.add(BatchNormalization())
        else:
            model.add(Dense(units))
        act_layer = _make_mlp_activation_layer(act)
        if act_layer is not None:
            model.add(act_layer)

    if num_classes is not None:
        model.add(Dense(num_classes))
        out_layer = _make_mlp_activation_layer(output_activation)
        if out_layer is not None:
            model.add(out_layer)

    if input_size is not None:
        model.build((None, input_size))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model



def build_deepsets_model(phi_config, rho_config, aggregator_type='mean', 
                        input_shape=(None, 8, 3), num_classes=5):
    """
    Builds a DeepSets model with configurable phi and rho networks.
    
    Parameters:
        phi_config (dict): Configuration for phi network
        rho_config (dict): Configuration for rho network
        aggregator_type (str): Type of aggregation ('mean' or 'max')
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
    
    Returns:
        Keras Model
    """
    from utils.tf_blocks import DeepSetsArchitecture_tf
    
    # Build phi network
    phi_layers = []
    for i in range(phi_config.get('num_layers', 2)):
        units = phi_config.get(f'units_{i}', phi_config.get('units', 32))
        activation = phi_config.get(f'activation_{i}', phi_config.get('activation', 'relu'))
        use_batchnorm = phi_config.get(f'batchnorm_{i}', phi_config.get('batchnorm', False))
        
        phi_layers.append(Dense(units))
        if use_batchnorm:
            phi_layers.append(BatchNormalization())
        if activation:
            phi_layers.append(get_activation_layer(activation))
    
    # Add final layer for bottleneck dimension
    bottleneck_dim = phi_config.get('bottleneck_dim', 16)
    phi_layers.append(Dense(bottleneck_dim))
    if phi_config.get('final_activation', 'relu'):
        phi_layers.append(get_activation_layer(phi_config.get('final_activation', 'relu')))
    
    phi = Sequential(phi_layers, name='phi_network')
    
    # Build rho network
    rho_layers = []
    for i in range(rho_config.get('num_layers', 2)):
        units = rho_config.get(f'units_{i}', rho_config.get('units', 32))
        activation = rho_config.get(f'activation_{i}', rho_config.get('activation', 'relu'))
        use_batchnorm = rho_config.get(f'batchnorm_{i}', rho_config.get('batchnorm', False))
        
        rho_layers.append(Dense(units))
        if use_batchnorm:
            rho_layers.append(BatchNormalization())
        if activation and i < rho_config.get('num_layers', 2) - 1:  # No activation on last layer
            rho_layers.append(get_activation_layer(activation))
    
    # Add final output layer
    rho_layers.append(Dense(num_classes))  # No activation, using from_logits=True
    
    rho = Sequential(rho_layers, name='rho_network')
    
    # Define aggregator
    if aggregator_type == 'mean':
        aggregator = lambda x: tf.reduce_mean(x, axis=1)
    elif aggregator_type == 'max':
        aggregator = lambda x: tf.reduce_max(x, axis=1)
    else:
        raise ValueError(f"Unsupported aggregator type: {aggregator_type}")
    
    # Create model
    model = DeepSetsArchitecture_tf(phi, rho, aggregator)
    model.build(input_shape=input_shape)
    
    return model


def create_model_from_trial(trial, model_type='mlp', **kwargs):
    """
    Creates a model based on Optuna trial suggestions.
    
    Parameters:
        trial: Optuna trial object
        model_type (str): Type of model to create ('mlp' or 'deepsets')
        **kwargs: Additional model-specific parameters
    
    Returns:
        Keras model
    """
    if model_type == 'mlp':
        config = {
            'num_layers': trial.suggest_int('num_layers', 2, 3),
            'hidden_units1': trial.suggest_categorical('hidden_units1', [8, 16, 32, 64]),
            'activation1': trial.suggest_categorical('activation1', ['relu', 'tanh', 'sigmoid']),
            'batchnorm1': trial.suggest_categorical('batchnorm1', [True, False]),
        }
        
        if config['num_layers'] >= 3:
            config['hidden_units2'] = trial.suggest_categorical('hidden_units2', [8, 16, 32, 64])
            config['activation2'] = trial.suggest_categorical('activation2', ['relu', 'tanh', 'sigmoid'])
            config['batchnorm2'] = trial.suggest_categorical('batchnorm2', [True, False])
        
        return build_mlp_from_config(config, **kwargs)
    
    elif model_type == 'deepsets':
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
        
        return build_deepsets_model(phi_config, rho_config, aggregator_type, **kwargs)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")