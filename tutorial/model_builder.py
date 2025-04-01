import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
import yaml

FIXED_LR = 0.001

def load_search_space(yaml_path):
    """
    Loads the search space configuration from a YAML file.
    
    Parameters:
        yaml_path (str): Path to the YAML file.
        
    Returns:
        dict: The search space dictionary.
    """
    with open(yaml_path, "r") as f:
        search_space = yaml.safe_load(f)
    return search_space

def create_model_from_config(config, input_size=64, num_classes=10, learning_rate=FIXED_LR):
    """
    Creates a simple MLP Keras model based on the provided configuration.
    
    Parameters:
        config (dict): Dictionary containing hyperparameters.
            Expected keys:
              - num_layers (2 or 3): 2 means one hidden layer; 3 means two hidden layers.
              - hidden_units1, activation1, batchnorm1: Parameters for the first hidden layer.
              - hidden_units2, activation2, batchnorm2: Parameters for the second hidden layer (if num_layers == 3).
        input_size (int): Dimensionality of the flattened input.
        num_classes (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential(name="Simple_MLP")
    model.add(Input(shape=(input_size,)))
    
    # First hidden layer
    if config["batchnorm1"]:
        model.add(Dense(config["hidden_units1"], use_bias=False))
        model.add(BatchNormalization())
        if config["activation1"] is not None:
            model.add(Activation(config["activation1"]))
        # else: no activation layer is added
    else:
        if config["activation1"] is not None:
            model.add(Dense(config["hidden_units1"], activation=config["activation1"]))
        else:
            model.add(Dense(config["hidden_units1"]))
    
    if config["num_layers"] == 3:
        if config["batchnorm2"]:
            model.add(Dense(config["hidden_units2"], use_bias=False))
            model.add(BatchNormalization())
            if config["activation2"] is not None:
                model.add(Activation(config["activation2"]))
        else:
            if config["activation2"] is not None:
                model.add(Dense(config["hidden_units2"], activation=config["activation2"]))
            else:
                model.add(Dense(config["hidden_units2"]))
    
    # Output layer
    model.add(Dense(num_classes, activation="softmax"))
    model.build((None, input_size))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Loading the YAML and creating a model
if __name__ == "__main__":
    # Load the search space from the YAML file.
    search_space = load_search_space("search_space.yaml")
    
    # Default: choose the first value from each list as the default configuration.
    default_config = {key: value[0] for key, value in search_space.items()}
    print("Default configuration:", default_config)
    
    # Build and summarize the model using the default configuration.
    model = create_model_from_config(default_config)
    model.summary()