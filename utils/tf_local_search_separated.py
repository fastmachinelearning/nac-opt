# File: utils/tf_local_search_separated.py

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from qkeras import QDense, QActivation, QConv2D, quantizers

# --- Self-Contained Helper Functions (unchanged, without debug prints) ---
def get_activation_tf(act_name):
    """
    Map an activation name from YAML to a Keras layer.

    ``None`` / ``null`` in the architecture YAML means no activation on that
    layer (linear); callers like ``build_mlp_from_config_classifier`` skip Nones.
    """
    if act_name is None:
        return None
    act_map = {
        "ReLU": tf.keras.layers.ReLU(),
        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
        "GELU": tf.keras.layers.Activation("gelu"),
        "Identity": tf.keras.layers.Activation("linear"),
        "linear": tf.keras.layers.Activation("linear"),
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


def loss_and_compile_metrics_from_arch_yaml(architecture_yaml_path):
    """
    If the BlockBased YAML describes a binary 1-output head, return
    ``(BinaryCrossentropy, [BinaryAccuracy])`` aligned with the final activation
    (logit vs sigmoid). Otherwise return ``(None, None)`` for callers to fall
    back to categorical / sparse-categorical training.
    """
    with open(architecture_yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    arch = cfg.get("architecture") or {}
    if int(arch.get("output_dim", 2)) != 1:
        return None, None
    head = next((c for c in arch.get("components", []) if c.get("name") == "classifier_head"), None)
    if not head:
        return None, None
    acts = (head.get("params") or {}).get("activations") or []
    final_act = acts[-1] if acts else None
    if final_act is None:
        la = ""
    else:
        la = str(final_act).strip().lower()
    if la in ("", "none", "null", "linear", "identity"):
        from_logits = True
        threshold = 0.0
    elif la == "sigmoid":
        from_logits = False
        threshold = 0.5
    else:
        from_logits = False
        threshold = 0.5
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
    metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold)]
    return loss_fn, metrics


def convert_to_qat_model(model: tf.keras.Model, total_bits: int, int_bits: int) -> tf.keras.Model:
    """
    Rebuild ``model`` with QKeras layers at ``<total_bits, int_bits>`` and copy its weights.

    Dense -> QDense and Conv2D -> QConv2D with quantized kernels and biases. Activation layers
    keep their nonlinearity: ``relu`` becomes ``QActivation(quantized_relu)``, and other
    nonlinear activations (tanh, sigmoid, gelu, leaky_relu, ...) are kept and followed by a
    ``quantized_bits`` QActivation that quantizes their output. Linear and softmax activations
    are left unchanged. Other layers (BatchNormalization, Flatten, ReLU/LeakyReLU layers, ...)
    are copied as they are.
    """
    weight_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    bias_quantizer = quantizers.quantized_bits(total_bits, int_bits, alpha=1)
    weight_args = ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint', 'bias_constraint']

    def get_quantized_layers(layer):
        """Return the replacement layers; the first one receives ``layer``'s weights."""
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Dense):
            for key in weight_args: config.pop(key, None)
            config['kernel_quantizer'], config['bias_quantizer'] = weight_quantizer, bias_quantizer
            return [QDense.from_config(config)]
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Conv layers need the same treatment, otherwise they stay full-precision floats
            # while the rest of the model is quantized.
            for key in weight_args: config.pop(key, None)
            config['kernel_quantizer'], config['bias_quantizer'] = weight_quantizer, bias_quantizer
            return [QConv2D.from_config(config)]
        if isinstance(layer, tf.keras.layers.Activation):
            activation_name = str(config['activation']).lower()
            # Avoid quantizing linear/identity or final softmax layers
            if 'linear' in activation_name or 'softmax' in activation_name:
                return [layer]
            if activation_name == 'relu':
                return [QActivation(activation=quantizers.quantized_relu(total_bits, int_bits))]
            # A bare quantized_bits QActivation is a (clipped) identity, so replacing e.g. tanh
            # with it would remove the nonlinearity. Keep the activation, then quantize its output.
            return [layer.__class__.from_config(config),
                    QActivation(activation=quantizers.quantized_bits(total_bits, int_bits))]
        return [layer.__class__.from_config(config)]

    input_tensor = tf.keras.Input(shape=model.input_shape[1:])
    x = input_tensor
    flat_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer): continue
        flat_layers.extend(layer.layers if isinstance(layer, tf.keras.Sequential) else [layer])
    new_layers, original_layers_flat = [], []
    for position, sub_layer in enumerate(flat_layers):
        if position == len(flat_layers) - 1 and isinstance(sub_layer, tf.keras.layers.Activation):
            # The output activation (softmax, sigmoid, ...) feeds the loss; keep it exact.
            q_layers = [sub_layer]
        else:
            q_layers = get_quantized_layers(sub_layer)
        for q_layer in q_layers:
            x = q_layer(x)
        original_layers_flat.append(sub_layer)
        new_layers.append(q_layers[0])
    qat_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    for i, new_layer in enumerate(new_layers):
        if original_layers_flat[i].get_weights():
            new_layer.set_weights(original_layers_flat[i].get_weights())
    return qat_model

# --- NEW: Independent Local Search Functions ---

def run_pruning_only_loop(base_model, dataset, config, results_dir, loss_function, compile_metrics=None):
    """Performs iterative magnitude pruning on a full-precision Keras model."""
    print("\n" + "-"*20 + " Starting Pruning-Only Experiment " + "-"*20)
    _metrics = compile_metrics if compile_metrics is not None else ["accuracy"]
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
        pruned_model.compile(optimizer='adam', loss=loss_function, metrics=_metrics)
        
        pruned_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=pruning_config['epochs_per_iteration'], 
                         batch_size=128, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=1)
        
        model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
        model_stripped.compile(optimizer='adam', loss=loss_function, metrics=_metrics)
        _, val_acc = model_stripped.evaluate(x_val, y_val, verbose=0)
        print(f"  -> Accuracy for sparsity {target_sparsity:.4f}: {val_acc:.4f}")
        with open(log_filename, "a") as f: f.write(f"{i+1},{target_sparsity:.4f},{val_acc:.4f}\n")
            
        rewound_weights = [orig * np.where(curr != 0, 1.0, 0.0) for orig, curr in zip(original_weights, model_stripped.get_weights())]
        model_to_prune.set_weights(rewound_weights)
        
    return pd.read_csv(log_filename)

def run_qat_only_loop(base_model, dataset, config, results_dir, loss_function, compile_metrics=None):
    """Performs QAT for various precisions on a non-pruned QKeras model.

    ``base_model`` comes from ``load_model_from_yaml``, which rebuilds the architecture
    with fresh (untrained) weights. So each precision trains its QAT model from that
    initialization, the same way the combined local search does its QAT warmup, at
    ``qat_settings.learning_rate`` (default 1e-3, matching the combined search).
    """
    print("\n" + "-"*20 + " Starting QAT-Only Experiment " + "-"*20)
    _metrics = compile_metrics if compile_metrics is not None else ["accuracy"]
    x_train, y_train, x_val, y_val = dataset
    qat_config = config['qat_settings']
    qat_learning_rate = float(qat_config.get("learning_rate", 1e-3))

    log_filename = os.path.join(results_dir, "qat_log.csv")
    with open(log_filename, "w") as f: f.write("Precision,TotalBits,IntBits,Accuracy\n")

    for precision in qat_config['precision_pairs']:
        total_bits, int_bits = precision['total_bits'], precision['int_bits']
        precision_str = f"<{total_bits},{int_bits}>"
        print(f"\nRunning QAT for Precision: {precision_str}")

        qat_model = convert_to_qat_model(base_model, total_bits, int_bits)
        # The weights are untrained, so this is QAT training, not fine-tuning. A fine-tuning
        # rate (previously 5e-5) leaves the model near chance after a few epochs.
        qat_optimizer = tf.keras.optimizers.Adam(learning_rate=qat_learning_rate)
        qat_model.compile(optimizer=qat_optimizer, loss=loss_function, metrics=_metrics)

        print(f"--> Training QAT model for {qat_config['epochs']} epochs (lr={qat_learning_rate:g})...")
        qat_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=qat_config['epochs'], 
                      batch_size=128, verbose=1)
        
        _, val_acc = qat_model.evaluate(x_val, y_val, verbose=0)
        print(f"  -> Final accuracy for precision {precision_str}: {val_acc:.4f}")
        with open(log_filename, "a") as f: f.write(f"'{precision_str}',{total_bits},{int_bits},{val_acc:.4f}\n")
        
    return pd.read_csv(log_filename)

def local_search_entrypoint(architecture_yaml_path, local_search_config_path, dataset, results_dir, run_pruning=True):
    """
    Main entrypoint that runs local search experiments. Pruning is optional.
    """
    print("\n" + "="*50 + "\n STARTING SEPARATED LOCAL SEARCH STAGE \n" + "="*50)
    
    os.makedirs(results_dir, exist_ok=True)
    with open(local_search_config_path, 'r') as f: config = yaml.safe_load(f)
    
    x_train, y_train, x_val, y_val = dataset
    loss_fn, comp_metrics = loss_and_compile_metrics_from_arch_yaml(architecture_yaml_path)
    if loss_fn is not None:
        loss_function = loss_fn
        compile_metrics = comp_metrics
    else:
        loss_function = (
            "categorical_crossentropy"
            if len(y_train.shape) > 1 and y_train.shape[1] > 1
            else "sparse_categorical_crossentropy"
        )
        compile_metrics = None

    base_model = load_model_from_yaml(architecture_yaml_path)
    
    pruning_df = pd.DataFrame() # Initialize empty dataframe
    if run_pruning:
        if 'pruning_settings' in config:
            # Run Experiment 1: Pruning
            pruning_df = run_pruning_only_loop(
                base_model, dataset, config, results_dir, loss_function, compile_metrics
            )
        else:
            print("Warning: 'run_pruning' is True, but 'pruning_settings' not found in config. Skipping pruning.")
    
    # Run Experiment 2: QAT
    qat_df = run_qat_only_loop(
        base_model, dataset, config, results_dir, loss_function, compile_metrics
    )
    
    print("\n" + "="*50 + "\n SEPARATED LOCAL SEARCH COMPLETE \n" + "="*50)
    return pruning_df, qat_df