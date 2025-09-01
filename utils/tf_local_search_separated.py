# File: utils/tf_local_search_separated.py

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QDense, QActivation, QConv2D, quantizers

# --- Self-Contained Helper Functions (unchanged, without debug prints) ---
def get_activation_tf(act_name: str) -> tf.keras.layers.Layer:
    """
    More robustly gets a Keras activation layer from a string name.
    """
    act_map = {
        "ReLU": tf.keras.layers.ReLU(), 
        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01), 
        "GELU": tf.keras.layers.Activation('gelu'), 
        "Identity": tf.keras.layers.Activation('linear')
    }
    # First, try a direct match (case-sensitive)
    if act_name in act_map:
        return act_map[act_name]
    
    # Fallback for common variations
    act_lower = act_name.lower()
    if 'leaky' in act_lower and 'relu' in act_lower:
        # Correctly format for Keras
        return tf.keras.layers.Activation('leaky_relu')
    
    # Original fallback for other standard Keras activations
    return tf.keras.layers.Activation(act_lower)

class BlockArchitectureTF(tf.keras.Model):
    def __init__(self, blocks, mlp, input_shape, needs_flattening):
        super().__init__()
        self.input_shape_spec, self.blocks, self.mlp, self.needs_flattening = input_shape, blocks, mlp, needs_flattening
        self.final_model = self._build_model()
    def _build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape_spec)
        x = inputs
        for block in self.blocks: x = block(x)
        if self.needs_flattening: x = tf.keras.layers.Flatten()(x)
        x = self.mlp(x)
        return tf.keras.Model(inputs=inputs, outputs=x, name='BlockArchitecture')
    def call(self, inputs, training=None): return self.final_model(inputs, training=training)

def create_conv_block_tf(channels, kernels, activations, normalizations, name='conv_block'):
    layers = []
    for i in range(len(kernels)):
        layers.append(tf.keras.layers.Conv2D(channels[i+1], kernel_size=kernels[i], strides=1, padding='valid' if kernels[i] > 1 else 'same', name=f'{name}_conv_{i}'))
        if normalizations[i] == 'batch': layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        elif normalizations[i] == 'layer': layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
        if activations[i] is not None: layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)

def build_mlp_from_config_classifier(widths, activations, normalizations, name='mlp'):
    layers = []
    for i in range(len(activations)):
        layers.append(tf.keras.layers.Dense(widths[i+1], name=f'{name}_dense_{i}'))
        if normalizations[i] == 'batch': layers.append(tf.keras.layers.BatchNormalization(name=f'{name}_bn_{i}'))
        elif normalizations[i] == 'layer': layers.append(tf.keras.layers.LayerNormalization(name=f'{name}_ln_{i}'))
        if activations[i] is not None:
            if i < len(activations) - 1 or 'softmax' not in str(activations[i]).lower(): layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)

def load_model_from_yaml(yaml_path: str) -> tf.keras.Model:
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
    arch_config = config['architecture']
    input_shape = tuple(arch_config['input_shape'])
    feature_extractor_blocks, is_flattened = [], False
    for component in arch_config['components']:
        block_type, params, name = component['block_type'], component['params'], component['name']
        if block_type == 'Conv':
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            feature_extractor_blocks.append(create_conv_block_tf(**params, name=name))
        elif block_type == 'Flatten':
            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=name)); is_flattened = True
        elif block_type == 'MLP' and name != 'classifier_head':
            params['activations'] = [get_activation_tf(act) for act in params['activations']]
            feature_extractor_blocks.append(build_mlp_from_config_classifier(**params, name=name))
    classifier_head_config = next(c for c in arch_config['components'] if c['name'] == 'classifier_head')
    mlp_params = classifier_head_config['params']
    mlp_params['activations'] = [get_activation_tf(act) for act in mlp_params['activations']]
    classifier_head = build_mlp_from_config_classifier(**mlp_params, name='classifier_head')
    model_wrapper = BlockArchitectureTF(blocks=feature_extractor_blocks, mlp=classifier_head, input_shape=input_shape, needs_flattening=(not is_flattened))
    return model_wrapper.final_model

def convert_to_qat_model(model: tf.keras.Model, total_bits: int, int_bits: int) -> tf.keras.Model:
    weight_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    bias_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    def get_quantized_layer(layer):
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Dense):
            for key in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint']: config.pop(key, None)
            config['kernel_quantizer'], config['bias_quantizer'] = weight_quantizer, bias_quantizer
            return QDense.from_config(config)
        if isinstance(layer, tf.keras.layers.Activation) and 'relu' in config['activation']:
            return QActivation(activation=quantizers.quantized_relu(total_bits, int_bits))
        return layer.__class__.from_config(config)
    input_tensor = tf.keras.Input(shape=model.input_shape[1:])
    x = input_tensor
    new_layers, original_layers_flat = [], []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer): continue
        if isinstance(layer, tf.keras.Sequential):
            original_layers_flat.extend(layer.layers)
            for sub_layer in layer.layers:
                q_sub_layer = get_quantized_layer(sub_layer)
                new_layers.append(q_sub_layer)
                x = q_sub_layer(x)
        else:
            original_layers_flat.append(layer)
            q_layer = get_quantized_layer(layer)
            new_layers.append(q_layer)
            x = q_layer(x)
    qat_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    for i, new_layer in enumerate(new_layers):
        if original_layers_flat[i].get_weights():
            new_layer.set_weights(original_layers_flat[i].get_weights())
    return qat_model


def run_pruning_only_loop(base_model, dataset, config, results_dir, loss_function):
    """Performs iterative magnitude pruning on a full-precision Keras model."""
    print("\n" + "-"*20 + " Starting Pruning-Only Experiment " + "-"*20)
    x_train, y_train, x_val, y_val = dataset
    pruning_config = config['pruning_settings']
    
    log_filename = os.path.join(results_dir, "pruning_log.csv")
    with open(log_filename, "w") as f: f.write("Iteration,Sparsity,Accuracy\n")
        
    model_to_prune = tf.keras.models.clone_model(base_model)
    model_to_prune.set_weights(base_model.get_weights())
    original_weights = model_to_prune.get_weights()

    for i in range(pruning_config['iterations']):
        target_sparsity = 1 - (pruning_config['pruning_rate'] ** (i + 1))
        print(f"\nPruning Iteration {i+1}/{pruning_config['iterations']} - Target Sparsity: {target_sparsity:.4f}")

        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity, 0, 100)}
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model_to_prune, **pruning_params)
        pruned_model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        
        pruned_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=pruning_config['epochs_per_iteration'], 
                         batch_size=128, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=1)
        
        model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
        def tensor_sparsity(x):
            return (x == 0).sum() / x.size

        actual = []
        for w in model_stripped.get_weights():
            if w.ndim >= 2:  # only kernels / weight matrices
                actual.append(tensor_sparsity(w))
        actual_sparsity = float(np.mean(actual))
        print(f"Actual avg sparsity: {actual_sparsity:.4f}")

        model_stripped.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
        _, val_acc = model_stripped.evaluate(x_val, y_val, verbose=0)
        print(f"  -> Accuracy for sparsity {target_sparsity:.4f}: {val_acc:.4f}")
        with open(log_filename, "a") as f: f.write(f"{i+1},{target_sparsity:.4f},{val_acc:.4f}\n")
            
        rewound_weights = [orig * np.where(curr != 0, 1.0, 0.0) for orig, curr in zip(original_weights, model_stripped.get_weights())]
        model_to_prune.set_weights(rewound_weights)
        
    return pd.read_csv(log_filename)

def run_qat_only_loop(base_model, dataset, config, results_dir, loss_function):
    """Performs QAT fine-tuning for various precisions on a non-pruned QKeras model."""
    print("\n" + "-"*20 + " Starting QAT-Only Experiment " + "-"*20)
    x_train, y_train, x_val, y_val = dataset
    qat_config = config['qat_settings']

    log_filename = os.path.join(results_dir, "qat_log.csv")
    with open(log_filename, "w") as f: f.write("Precision,TotalBits,IntBits,Accuracy\n")

    for precision in qat_config['precision_pairs']:
        total_bits, int_bits = precision['total_bits'], precision['int_bits']
        precision_str = f"<{total_bits},{int_bits}>"
        print(f"\nRunning QAT for Precision: {precision_str}")
        
        qat_model = convert_to_qat_model(base_model, total_bits, int_bits)
        qat_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        qat_model.compile(optimizer=qat_optimizer, loss=loss_function, metrics=['accuracy'])


        print(f"--> Fine-tuning QAT model for {qat_config['epochs']} epochs...")
        qat_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=qat_config['epochs'], 
                      batch_size=128, verbose=1)
        
        _, val_acc = qat_model.evaluate(x_val, y_val, verbose=0)
        print(f"  -> Final accuracy for precision {precision_str}: {val_acc:.4f}")
        with open(log_filename, "a") as f: f.write(f"'{precision_str}',{total_bits},{int_bits},{val_acc:.4f}\n")
        
    return pd.read_csv(log_filename)

def local_search_entrypoint(architecture_yaml_path, local_search_config_path, dataset, results_dir):
    """Main entrypoint that runs pruning and QAT as two separate experiments."""
    print("\n" + "="*50 + "\n STARTING SEPARATED LOCAL SEARCH STAGE \n" + "="*50)
    os.makedirs(results_dir, exist_ok=True)
    with open(local_search_config_path, 'r') as f: config = yaml.safe_load(f)
    
    x_train, y_train, x_val, y_val = dataset
    loss_function = 'categorical_crossentropy' if len(y_train.shape) > 1 and y_train.shape[1] > 1 else 'sparse_categorical_crossentropy'

    base_model = load_model_from_yaml(architecture_yaml_path)
    
    # Run Experiment 1: Pruning
    pruning_df = run_pruning_only_loop(base_model, dataset, config, results_dir, loss_function)
    
    # Run Experiment 2: QAT
    qat_df = run_qat_only_loop(base_model, dataset, config, results_dir, loss_function)
    
    print("\n" + "="*50 + "\n SEPARATED LOCAL SEARCH COMPLETE \n" + "="*50)
    return pruning_df, qat_df