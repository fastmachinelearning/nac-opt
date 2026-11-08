"""Conv layers in BOPs counting (global + local search), QAT conversion and sparsity.

Before these fixes, for models with Conv blocks:
- global search computed conv BOPs from an image size that had already been shrunk once,
  so it under-counted them (down to ~5% of the real count on Tutorial 2 architectures);
- the combined local search skipped Conv2D layers when computing baseline BOPs;
- convert_to_qat_model left Conv2D layers as full-precision floats;
- the sparsity measurement ignored Conv2D layers.
"""

import math

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml
from qkeras import QConv2D, QDense

from utils.tf_bops import get_linear_bops_tf
from utils.tf_global_search import GlobalSearchTF
from utils.tf_local_search_combined import _compute_baseline_bops, _compute_model_sparsity
from utils.tf_local_search_separated import convert_to_qat_model, load_model_from_yaml

BITS = 32


def _true_bops(model):
    """32x32-bit multiply count from the built layers: Conv2D H*W*C_out*k*k*C_in, Dense in*out."""
    macs = 0
    for layer in model.layers:
        subs = layer.layers if isinstance(layer, tf.keras.Sequential) else [layer]
        for s in subs:
            if isinstance(s, tf.keras.layers.Conv2D):
                _, h, w, c_out = s.output_shape
                macs += h * w * c_out * int(np.prod(s.kernel_size)) * s.input_shape[-1]
            elif isinstance(s, tf.keras.layers.Dense):
                macs += s.input_shape[-1] * s.units
    return macs * BITS * BITS


def _conv_arch_yaml(path):
    arch = {
        "architecture": {
            "model_type": "BlockBased",
            "input_shape": [12, 12, 1],
            "output_dim": 3,
            "components": [
                {
                    "block_type": "Conv",
                    "name": "conv_block_0",
                    "params": {
                        "channels": [1, 4, 8],
                        "kernels": [3, 3],
                        "activations": ["ReLU", "ReLU"],
                        "normalizations": [None, None],
                    },
                },
                {"block_type": "Flatten", "name": "initial_flatten", "params": {}},
                {
                    "block_type": "MLP",
                    "name": "classifier_head",
                    "params": {"widths": [512, 16, 3], "activations": ["relu", "softmax"], "normalizations": [None, None]},
                },
            ],
        }
    }
    path.write_text(yaml.safe_dump(arch))
    return str(path)


def test_global_search_conv_bops_match_the_built_layers(tmp_path):
    space = {
        "channel_space": [2, 4],
        "mlp_width_space": [8, 16],
        "kernel_space": [1, 3, 5],
        "act_space": ["ReLU"],
        "norm_space": [None],
        "block_types": ["Conv"],
        "num_blocks": 2,
        "initial_img_size": 12,
        "output_dim": 3,
        "output_activation": "softmax",
        "conv_attn": {"hidden_channel_space": [4]},
    }
    searcher = GlobalSearchTF(search_space_path=space, results_dir=str(tmp_path))
    searcher.run_search(
        model_type="block",
        n_trials=6,
        epochs=1,
        dataset="images",
        loader_path="tests.synthetic_data:load_images",
        subset_size=None,
        one_hot=True,
        verbose=False,
        seed=3,
    )
    df = pd.read_csv(tmp_path / "block_search_results.csv")
    assert len(df) == 6
    for _, row in df.iterrows():
        model = load_model_from_yaml(row["yaml_path"])
        assert row["bops"] == _true_bops(model), row["yaml_path"]


def test_local_baseline_bops_count_conv_layers(tmp_path):
    model = load_model_from_yaml(_conv_arch_yaml(tmp_path / "arch.yaml"))

    # conv_0: 12x12 -> 10x10, 1 -> 4 channels; conv_1: 10x10 -> 8x8, 4 -> 8; head 512 -> 16 -> 3
    def per_position(fan_in, fan_out):
        return fan_in * fan_out * (BITS**2 + 2 * BITS) + fan_out * math.log2(fan_in) * BITS

    expected = 10 * 10 * per_position(9 * 1, 4) + 8 * 8 * per_position(9 * 4, 8)
    expected += per_position(512, 16) + per_position(16, 3)
    assert _compute_baseline_bops(model, (1, 12, 12, 1)) == pytest.approx(expected)


def test_local_baseline_bops_unchanged_for_mlp_only_models(tmp_path):
    arch = {
        "architecture": {
            "model_type": "BlockBased",
            "input_shape": [20],
            "output_dim": 4,
            "components": [
                {
                    "block_type": "MLP",
                    "name": "classifier_head",
                    "params": {
                        "widths": [20, 32, 8, 4],
                        "activations": ["relu", "relu", "softmax"],
                        "normalizations": ["batch", None, None],
                    },
                }
            ],
        }
    }
    (tmp_path / "mlp.yaml").write_text(yaml.safe_dump(arch))
    model = load_model_from_yaml(str(tmp_path / "mlp.yaml"))
    dense = [layer for layer in model.layers[-1].layers if isinstance(layer, tf.keras.layers.Dense)]
    expected = sum(get_linear_bops_tf(d, BITS, input_shape=(1, d.kernel.shape[0])) for d in dense)
    assert _compute_baseline_bops(model, (1, 20)) == pytest.approx(expected)


def test_qat_conversion_quantizes_conv_layers(tmp_path):
    model = load_model_from_yaml(_conv_arch_yaml(tmp_path / "arch.yaml"))
    qat = convert_to_qat_model(model, 16, 6)
    kinds = [type(layer) for layer in qat.layers]
    assert kinds.count(QConv2D) == 2 and kinds.count(QDense) == 2
    assert not any(k is tf.keras.layers.Conv2D for k in kinds)
    # At 16 bits the quantized model should compute (almost) the same function.
    x = np.random.default_rng(0).normal(size=(8, 12, 12, 1)).astype("float32")
    assert np.max(np.abs(qat.predict(x, verbose=0) - model.predict(x, verbose=0))) < 1e-2


def test_sparsity_counts_conv_weights(tmp_path):
    model = load_model_from_yaml(_conv_arch_yaml(tmp_path / "arch.yaml"))
    qat = convert_to_qat_model(model, 8, 3)
    conv = [layer for layer in qat.layers if isinstance(layer, QConv2D)][0]
    kernel, bias = conv.get_weights()
    conv.set_weights([np.zeros_like(kernel), bias])
    weighted = [layer for layer in qat.layers if isinstance(layer, (QDense, QConv2D))]
    total = sum(w.size for layer in weighted for w in layer.get_weights())
    zeros = sum(int(np.count_nonzero(w == 0)) for layer in weighted for w in layer.get_weights())
    assert _compute_model_sparsity(qat) == pytest.approx(zeros / total)
    assert _compute_model_sparsity(qat) >= kernel.size / total


def test_qat_conversion_keeps_nonlinear_activations(tmp_path):
    arch = {
        "architecture": {
            "model_type": "BlockBased",
            "input_shape": [10],
            "output_dim": 3,
            "components": [
                {
                    "block_type": "MLP",
                    "name": "classifier_head",
                    "params": {
                        "widths": [10, 16, 16, 3],
                        "activations": ["tanh", "relu", "softmax"],
                        "normalizations": [None, None, None],
                    },
                }
            ],
        }
    }
    (tmp_path / "mlp.yaml").write_text(yaml.safe_dump(arch))
    model = load_model_from_yaml(str(tmp_path / "mlp.yaml"))
    qat = convert_to_qat_model(model, 16, 6)

    names = [type(layer).__name__ for layer in qat.layers]
    assert names == [
        "InputLayer",
        "Flatten",
        "QDense",
        "Activation",
        "QActivation",
        "QDense",
        "QActivation",
        "QDense",
        "Activation",
    ]
    assert qat.layers[3].get_config()["activation"] == "tanh"  # nonlinearity kept, output quantized next
    assert "quantized_relu" in str(qat.layers[6].get_config()["activation"])
    x = np.random.default_rng(1).normal(size=(16, 10)).astype("float32")
    assert np.max(np.abs(qat.predict(x, verbose=0) - model.predict(x, verbose=0))) < 1e-2
