import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from qkeras import QDense, QActivation, QConv2D, quantizers
from tensorflow.keras.utils import get_custom_objects

# Custom objects for loading QKeras models
custom_objects = {
    "QDense": QDense,
    "QActivation": QActivation,
    "QConv2D": QConv2D,
    "quantized_bits": quantizers.quantized_bits,
    "quantized_relu": quantizers.quantized_relu,
}
get_custom_objects().update(custom_objects)

# Import necessary builder functions from other utils
# These are needed to reconstruct the model from the YAML file
from .tf_global_search5 import create_conv_block_tf, build_mlp_from_config_tf, BlockArchitectureTF, get_activation_tf

def load_model_from_yaml(yaml_path: str) -> tf.keras.Model:
    """
    Reconstructs a Keras model from a YAML architecture file.

    Args:
        yaml_path: Path to the architecture YAML file.

    Returns:
        A compiled TensorFlow Keras model.
    """
    print(f"--- Loading model architecture from: {yaml_path} ---")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    arch_config = config['architecture']
    input_shape = tuple(arch_config['input_shape'])
    
    feature_extractor_blocks = []
    is_flattened = False
    
    # Reconstruct blocks
    for component in arch_config['components']:
        block_type = component['block_type']
        params = component['params']
        name = component['name']

        if block_type == 'Conv':
            # Convert activation names from YAML into layer objects
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            block = create_conv_block_tf(**params, name=name)
            feature_extractor_blocks.append(block)
        elif block_type == 'Flatten':
            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=name))
            is_flattened = True
        elif block_type == 'MLP' and name != 'classifier_head': # Handle intermediate MLP blocks
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            block = build_mlp_from_config_tf(**params, name=name)
            feature_extractor_blocks.append(block)
    
    # Reconstruct classifier head
    classifier_head_config = next(c for c in arch_config['components'] if c['name'] == 'classifier_head')
    mlp_params = classifier_head_config['params']
    mlp_params['activations'] = [get_activation_tf(act) for act in mlp_params['activations']]
    classifier_head = build_mlp_from_config_tf(**mlp_params, name='classifier_head')

    # Build the final model using the BlockArchitecture wrapper
    model_wrapper = BlockArchitectureTF(
        blocks=feature_extractor_blocks,
        mlp=classifier_head,
        input_shape=input_shape,
        needs_flattening=(not is_flattened)
    )
    
    print("--- Model successfully reconstructed ---")
    model_wrapper.model.summary()
    return model_wrapper.model


def convert_to_qat_model(model: tf.keras.Model, total_bits: int, int_bits: int) -> tf.keras.Model:
    """
    Converts a standard Keras model to a QKeras model for QAT.

    Args:
        model: The input Keras model.
        total_bits: Total bits for quantization.
        int_bits: Integer bits for quantization.

    Returns:
        A new model with QKeras layers.
    """
    weight_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    bias_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    activation_quantizer = quantizers.quantized_relu(total_bits, int_bits)

    # Create a new functional model by iterating through the layers of the original model
    input_layer = model.inputs[0]
    x = input_layer
    layer_map = {}
    

    layer_map[model.layers[0].name] = input_layer


    for layer in model.layers[1:]: # Skip original input layer
        if isinstance(layer, tf.keras.layers.Dense):
            new_layer = QDense(
                units=layer.units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name=f"q_{layer.name}"
            )
        elif isinstance(layer, tf.keras.layers.Conv2D):
             new_layer = QConv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name=f"q_{layer.name}"
            )
        elif isinstance(layer, (tf.keras.layers.ReLU, tf.keras.layers.Activation)):
            # Replace ReLU-like activations with QActivation
            new_layer = QActivation(activation=activation_quantizer, name=f"q_{layer.name}")
        else:
            # Clone other layers (like Flatten, BatchNormalization)
            config = layer.get_config()
            config['name'] = f"clone_{layer.name}"
            new_layer = layer.__class__.from_config(config)

        # Connect the new layer
        input_nodes = layer.inbound_nodes[0].parent_nodes
        if len(input_nodes) == 1:
            prev_layer_output = layer_map[input_nodes[0].layer.name]
            x = new_layer(prev_layer_output)
        else: # Handle multiple inputs if necessary (not expected for these models)
            prev_outputs = [layer_map[node.layer.name] for node in input_nodes]
            x = new_layer(prev_outputs)
            
        layer_map[layer.name] = x

    qat_model = tf.keras.Model(inputs=input_layer, outputs=x, name=f"qat_model_{total_bits}b")
    print(f"--- Converted to QAT model with <{total_bits},{int_bits}> precision ---")
    qat_model.summary()
    return qat_model


def run_pruning_loop(model: tf.keras.Model, train_data, val_data, config: dict, precision_str: str, results_dir: str):
    """
    Performs the iterative pruning and training loop.

    Args:
        model: The QAT-ready model to be pruned.
        train_data: Tuple of (x_train, y_train).
        val_data: Tuple of (x_val, y_val).
        config: Dictionary with local search settings.
        precision_str: String representation of the precision (e.g., "<8,3>").
        results_dir: Directory to save logs and models.
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    log_filename = os.path.join(results_dir, "pruning_log.csv")
    if not os.path.exists(log_filename):
        with open(log_filename, "w") as log_file:
            log_file.write("Precision,Iteration,Sparsity,Accuracy\n")

    # Save initial weights for lottery ticket rewinding
    original_weights = model.get_weights()
    
    num_iterations = config['pruning_iterations']
    epochs_per_iter = config['epochs_per_iteration']
    best_val_accuracy = 0.0
    best_weights = None
    
    # Compile the base model once
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for iteration in range(num_iterations):
        target_sparsity = 1 - (config['pruning_rate'] ** (iteration + 1))
        print(f"\n--- Pruning Iteration {iteration+1}/{num_iterations} | Target Sparsity: {target_sparsity:.4f} ---")

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity, begin_step=0, frequency=100
            )
        }
        
        # Clone and wrap the model for pruning
        model_to_prune = tf.keras.models.clone_model(model)
        model_to_prune.set_weights(model.get_weights())
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model_to_prune, **pruning_params)
        
        pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
        pruned_model.fit(x_train, y_train,
                         validation_data=(x_val, y_val),
                         epochs=epochs_per_iter,
                         batch_size=128,
                         callbacks=callbacks,
                         verbose=1)
        
        # Strip pruning wrappers to evaluate and save
        model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
        model_stripped.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        _, val_acc = model_stripped.evaluate(x_val, y_val, verbose=0)
        print(f"Iteration {iteration+1} Accuracy: {val_acc:.4f}")
        
        with open(log_filename, "a") as log_file:
            log_file.write(f"{precision_str},{iteration+1},{target_sparsity:.4f},{val_acc:.4f}\n")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_weights = model_stripped.get_weights()
            print(f"--> New best accuracy for this precision: {best_val_accuracy:.4f}")

        # Rewind weights of the base model for the next iteration
        # This is a key part of the "Lottery Ticket Hypothesis" methodology
        current_mask = [np.where(w != 0, 1.0, 0.0) for w in model_stripped.get_weights()]
        rewound_weights = [orig * mask for orig, mask in zip(original_weights, current_mask)]
        model.set_weights(rewound_weights)

    if best_weights is not None:
        model.set_weights(best_weights)
        # Assuming total_bits is available in scope for filename
        total_bits = int(precision_str.split(',')[0].replace('<',''))
        save_path = os.path.join(results_dir, f"best_model_q{total_bits}b.h5")
        model.save(save_path)
        print(f"Best model for precision {precision_str} saved to {save_path} with accuracy {best_val_accuracy:.4f}")
    else:
        print(f"No improvement found for precision {precision_str}.")

    return best_val_accuracy


def local_search_entrypoint(architecture_yaml_path: str, local_search_config_path: str, dataset, results_dir: str):
    """
    Main entrypoint for the local search stage.

    Args:
        architecture_yaml_path: Path to the model architecture YAML.
        local_search_config_path: Path to the local search settings YAML.
        dataset: Tuple of (x_train, y_train, x_val, y_val).
        results_dir: Directory to save all outputs.
    """
    print("\n" + "="*50)
    print(" STARTING LOCAL SEARCH STAGE ")
    print("="*50)

    os.makedirs(results_dir, exist_ok=True)
    
    # Load local search configuration
    with open(local_search_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Unpack dataset
    x_train, y_train, x_val, y_val = dataset

    # Load the base model from the global search result
    base_model = load_model_from_yaml(architecture_yaml_path)

    results = {}
    # Loop through each precision pair defined in the config
    for precision in config['precision_pairs']:
        total_bits = precision['total_bits']
        int_bits = precision['int_bits']
        precision_str = f"<{total_bits},{int_bits}>"
        print(f"\n{'='*20} Running for Precision: {precision_str} {'='*20}")

        # 1. Convert the base model to a QAT model for the current precision
        qat_model = convert_to_qat_model(base_model, total_bits, int_bits)
        
        # 2. Run the iterative pruning loop
        best_acc = run_pruning_loop(
            model=qat_model,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            config=config,
            precision_str=precision_str,
            results_dir=results_dir
        )
        results[precision_str] = best_acc

    print("\n" + "="*50)
    print(" LOCAL SEARCH STAGE COMPLETE")
    print("="*50)
    print("Summary of best accuracies per precision:")
    for prec, acc in results.items():
        print(f"  - {prec}: {acc:.4f}")
    
    results_df = pd.read_csv(os.path.join(results_dir, "pruning_log.csv"))
    return results_df

