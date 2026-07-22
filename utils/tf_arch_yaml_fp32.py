import yaml
import tensorflow as tf


def _get_activation_tf(act_name):
    if act_name is None:
        return None
    act_map = {
        "ReLU": tf.keras.layers.ReLU(),
        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
        "GELU": tf.keras.layers.Activation("gelu"),
        "Identity": tf.keras.layers.Activation("linear"),
        "linear": tf.keras.layers.Activation("linear"),
    }
    if act_name in act_map:
        return act_map[act_name]
    return tf.keras.layers.Activation(str(act_name).strip().lower())


def _create_conv_block_tf(channels, kernels, activations, normalizations, name="conv_block"):
    layers = []
    for i in range(len(kernels)):
        layers.append(
            tf.keras.layers.Conv2D(
                channels[i + 1],
                kernel_size=kernels[i],
                strides=1,
                padding="valid" if kernels[i] > 1 else "same",
                name="{}_conv_{}".format(name, i),
            )
        )
        if normalizations[i] == "batch":
            layers.append(tf.keras.layers.BatchNormalization(name="{}_bn_{}".format(name, i)))
        elif normalizations[i] == "layer":
            layers.append(tf.keras.layers.LayerNormalization(name="{}_ln_{}".format(name, i)))
        if activations[i] is not None:
            layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)


def _build_mlp_from_config_classifier(widths, activations, normalizations, name="mlp"):
    layers = []
    for i in range(len(activations)):
        layers.append(tf.keras.layers.Dense(widths[i + 1], name="{}_dense_{}".format(name, i)))
        if normalizations[i] == "batch":
            layers.append(tf.keras.layers.BatchNormalization(name="{}_bn_{}".format(name, i)))
        elif normalizations[i] == "layer":
            layers.append(tf.keras.layers.LayerNormalization(name="{}_ln_{}".format(name, i)))
        if activations[i] is not None:
            layers.append(activations[i])
    return tf.keras.Sequential(layers, name=name)


def load_model_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f) or {}
    arch_config = config["architecture"]
    input_shape = tuple(arch_config["input_shape"])

    feature_extractor_blocks = []
    is_flattened = False
    for component in arch_config["components"]:
        block_type = component["block_type"]
        params = component["params"]
        name = component["name"]
        if block_type == "Conv":
            params = dict(params)
            params["activations"] = [_get_activation_tf(act) for act in params["activations"]]
            feature_extractor_blocks.append(_create_conv_block_tf(**params, name=name))
        elif block_type == "Flatten":
            feature_extractor_blocks.append(tf.keras.layers.Flatten(name=name))
            is_flattened = True
        elif block_type == "MLP" and name != "classifier_head":
            params = dict(params)
            params["activations"] = [_get_activation_tf(act) for act in params["activations"]]
            feature_extractor_blocks.append(_build_mlp_from_config_classifier(**params, name=name))

    classifier_head_config = next(c for c in arch_config["components"] if c["name"] == "classifier_head")
    mlp_params = dict(classifier_head_config["params"])
    mlp_params["activations"] = [_get_activation_tf(act) for act in mlp_params["activations"]]
    classifier_head = _build_mlp_from_config_classifier(**mlp_params, name="classifier_head")

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for block in feature_extractor_blocks:
        x = block(x)
    if not is_flattened:
        x = tf.keras.layers.Flatten()(x)
    outputs = classifier_head(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="BlockArchitecture")


def loss_and_compile_metrics_from_arch_yaml(architecture_yaml_path):
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
    la = "" if final_act is None else str(final_act).strip().lower()
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

