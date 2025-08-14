"""
TensorFlow model builder utilities for creating various architectures.
Supports building models from configurations and search spaces.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LayerNormalization
import yaml


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


def build_mlp_from_config(config, input_size=None, num_classes=None, learning_rate=0.001):
    """
    Creates a simple MLP model based on configuration.
    
    Parameters:
        config (dict): Model configuration containing:
            - num_layers: Number of layers (2 or 3)
            - hidden_units1, activation1, batchnorm1: First layer params
            - hidden_units2, activation2, batchnorm2: Second layer params (if num_layers=3)
        input_size (int): Input dimension
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential(name="MLP_Model")
    
    if input_size is not None:
        model.add(Input(shape=(input_size,)))
    
    # First hidden layer
    if config.get("batchnorm1", False):
        model.add(Dense(config["hidden_units1"], use_bias=False))
        model.add(BatchNormalization())
        if config.get("activation1") is not None:
            model.add(Activation(config["activation1"]))
    else:
        if config.get("activation1") is not None:
            model.add(Dense(config["hidden_units1"], activation=config["activation1"]))
        else:
            model.add(Dense(config["hidden_units1"]))
    
    # Optional second hidden layer
    if config.get("num_layers", 2) >= 3:
        if config.get("batchnorm2", False):
            model.add(Dense(config["hidden_units2"], use_bias=False))
            model.add(BatchNormalization())
            if config.get("activation2") is not None:
                model.add(Activation(config["activation2"]))
        else:
            if config.get("activation2") is not None:
                model.add(Dense(config["hidden_units2"], activation=config["activation2"]))
            else:
                model.add(Dense(config["hidden_units2"]))
    
    # Output layer
    if num_classes is not None:
        model.add(Dense(num_classes, activation="softmax"))
    
    # Build and compile
    if input_size is not None:
        model.build((None, input_size))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
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
    from models.tf_blocks import DeepSetsArchitecture_tf
    
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