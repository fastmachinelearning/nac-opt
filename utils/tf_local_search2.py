import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from qkeras import QDense, QActivation, QConv2D, quantizers
from tensorflow.keras.utils import get_custom_objects

# --- Self-Contained Helper Functions ---
# To ensure this script is robust and self-contained, the necessary functions
# for model reconstruction are included here.

def get_activation_tf(act_name: str) -> tf.keras.layers.Layer:
    """Returns a Keras activation layer instance from a string name."""
    act_map = {
        "ReLU": tf.keras.layers.ReLU(),
        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
        "GELU": tf.keras.layers.Activation('gelu'),
        "Identity": tf.keras.layers.Activation('linear'),
    }
    # Fallback for other keras supported activations like 'softmax', 'tanh', etc.
    return act_map.get(act_name, tf.keras.layers.Activation(act_name.lower()))

class BlockArchitectureTF(tf.keras.Model):
    """A flexible container for a sequence of TensorFlow layers."""
    def __init__(self, blocks, mlp, input_shape, needs_flattening):
        super().__init__()
        self.input_shape_spec = input_shape
        self.blocks = blocks
        self.mlp = mlp
        self.needs_flattening = needs_flattening
        
        self.final_model = self._build_model()

    def _build_model(self):
        """Builds the Keras model from the provided blocks and MLP."""
        inputs = tf.keras.Input(shape=self.input_shape_spec)
        x = inputs
        for block in self.blocks:
            x = block(x)
        if self.needs_flattening:
            x = tf.keras.layers.Flatten()(x)
        x = self.mlp(x)
        return tf.keras.Model(inputs=inputs, outputs=x, name='BlockArchitecture')

    def call(self, inputs, training=None):
        return self.final_model(inputs, training=training)

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

def build_mlp_from_config_classifier(widths, activations, normalizations, name='mlp'):
    """Builds an MLP classifier head from a configuration."""
    layers = []
    for i in range(len(activations)):
        layers.append(tf.keras.layers.Dense(widths[i+1], name=f'{name}_dense_{i}'))
        if normalizations[i] == 'batch':
            layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        elif normalizations[i] == 'layer':
            layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
        if activations[i] is not None:
            # Check if it's the last layer; if so, it might not need an activation (e.g., softmax is often separate)
            if i < len(activations) - 1 or 'softmax' not in str(activations[i]).lower():
                 layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)


# --- Core Local Search Logic ---

def load_model_from_yaml(yaml_path: str) -> tf.keras.Model:
    """Reconstructs a Keras model from a YAML architecture file."""
    print(f"--- Loading model architecture from: {yaml_path} ---")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    arch_config = config['architecture']
    input_shape = tuple(arch_config['input_shape'])
    
    feature_extractor_blocks, is_flattened = [], False
    
    for component in arch_config['components']:
        block_type, params, name = component['block_type'], component['params'], component['name']
        if block_type == 'Conv':
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            feature_extractor_blocks.append(create_conv_block_tf(**params, name=name))
        elif block_type == 'Flatten':
            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=name))
            is_flattened = True
        elif block_type == 'MLP' and name != 'classifier_head':
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            feature_extractor_blocks.append(build_mlp_from_config_classifier(**params, name=name))
    
    classifier_head_config = next(c for c in arch_config['components'] if c['name'] == 'classifier_head')
    mlp_params = classifier_head_config['params']
    mlp_params['activations'] = [get_activation_tf(act) for act in mlp_params['activations']]
    classifier_head = build_mlp_from_config_classifier(**mlp_params, name='classifier_head')

    model_wrapper = BlockArchitectureTF(
        blocks=feature_extractor_blocks, mlp=classifier_head,
        input_shape=input_shape, needs_flattening=(not is_flattened)
    )
    print("--- Model successfully reconstructed ---")
    model_wrapper.final_model.summary()
    return model_wrapper.final_model

def convert_to_qat_model(model: tf.keras.Model, total_bits: int, int_bits: int) -> tf.keras.Model:
    """Robustly converts a Keras model to a QKeras model for QAT using clone_model."""
    weight_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    bias_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    
    def clone_function(layer):
        config = layer.get_config()
        # Define quantizers for specific layers
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            config['kernel_quantizer'] = weight_quantizer
            config['bias_quantizer'] = bias_quantizer
            # Remove args unsupported by QKeras layers
            for key in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint']:
                config.pop(key, None)
            
            if isinstance(layer, tf.keras.layers.Dense): return QDense.from_config(config)
            if isinstance(layer, tf.keras.layers.Conv2D): return QConv2D.from_config(config)

        # Replace ReLU activations with quantized versions
        if isinstance(layer, tf.keras.layers.Activation) and config['activation'] == 'relu':
            return QActivation(activation=quantizers.quantized_relu(total_bits, int_bits))
        
        return layer

    # Clone the model architecture, applying the function to each layer
    qat_model = tf.keras.models.clone_model(model, clone_function=clone_function)

    # Manually transfer weights from the original model to the new one
    for layer_orig in model.layers:
        if layer_orig.weights:
            try:
                layer_new = qat_model.get_layer(name=layer_orig.name)
                layer_new.set_weights(layer_orig.get_weights())
            except Exception as e:
                print(f"Could not transfer weights for layer: {layer_orig.name}. Error: {e}")
    
    print(f"\n--- Converted to QAT model with <{total_bits},{int_bits}> precision ---")
    qat_model.summary()
    return qat_model

def rewind_weights(model, original_weights):
    """Rewinds the surviving weights (non-zero) of a model to their original values."""
    current_weights = model.get_weights()
    new_weights = [orig * np.where(curr != 0, 1.0, 0.0) for orig, curr in zip(original_weights, current_weights)]
    model.set_weights(new_weights)
    return model

def local_search_entrypoint(architecture_yaml_path: str, local_search_config_path: str, dataset, results_dir: str):
    """Main entrypoint for the local search stage, performing QAT and iterative pruning."""
    print("\n" + "="*50 + "\n STARTING LOCAL SEARCH STAGE \n" + "="*50)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(local_search_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    x_train, y_train, x_val, y_val = dataset
    loss_function = 'categorical_crossentropy' if len(y_train.shape) > 1 and y_train.shape[1] > 1 else 'sparse_categorical_crossentropy'

    log_filename = os.path.join(results_dir, "pruning_log.csv")
    with open(log_filename, "w") as log_file:
        log_file.write("Precision,Iteration,Sparsity,Accuracy\n")

    base_model = load_model_from_yaml(architecture_yaml_path)
    all_results = {}

    for precision in config['precision_pairs']:
        total_bits, int_bits = precision['total_bits'], precision['int_bits']
        precision_str = f"<{total_bits},{int_bits}>"
        print(f"\n{'='*20} Running for Precision: {precision_str} {'='*20}")

        qat_model = convert_to_qat_model(base_model, total_bits, int_bits)
        model = tf.keras.models.clone_model(qat_model)
        model.set_weights(qat_model.get_weights())
        
        original_weights = model.get_weights()
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        num_iterations = config['pruning_iterations']
        epochs_per_iter = config['epochs_per_iteration']
        best_val_accuracy = 0.0
        best_weights = None

        for iteration in range(num_iterations):
            pruning_rate = config.get('pruning_rate', 0.8)
            target_sparsity = 1 - (pruning_rate ** (iteration + 1))
            print(f"\nIteration {iteration+1}/{num_iterations} - Target sparsity: {target_sparsity:.4f}")
            
            pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity, begin_step=0, frequency=100
            )}
            
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
            pruned_model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
            
            pruned_model.fit(x_train, y_train,
                             validation_data=(x_val, y_val),
                             epochs=epochs_per_iter,
                             batch_size=128,
                             callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
                             verbose=1)
            
            model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
            model_stripped.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
            
            _, val_acc = model_stripped.evaluate(x_val, y_val, verbose=0)
            print(f"Iteration {iteration+1} accuracy: {val_acc:.4f}")
            
            with open(log_filename, "a") as log_file:
                log_file.write(f"{precision_str},{iteration+1},{target_sparsity:.4f},{val_acc:.4f}\n")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_weights = model_stripped.get_weights()
                print(f"--> New best model found for this precision with accuracy: {best_val_accuracy:.4f}")
            
            model = rewind_weights(model_stripped, original_weights)

        if best_weights:
            final_model = tf.keras.models.clone_model(qat_model)
            final_model.set_weights(best_weights)
            save_path = os.path.join(results_dir, f"best_model_q_{total_bits}b_{int_bits}i.h5")
            final_model.save(save_path)
            print(f"Best model for precision {precision_str} saved to {save_path}")
        else:
            print(f"No improvement found during pruning for precision {precision_str}.")
        
        all_results[precision_str] = best_val_accuracy

    print("\n" + "="*50 + "\n LOCAL SEARCH STAGE COMPLETE\n" + "="*50)
    print("Summary of best accuracies per precision:")
    for prec, acc in all_results.items():
        print(f"  - {prec}: {acc:.4f}")
    
    return pd.read_csv(log_filename)
