"""Regression test for the QAT-only experiment in the separated local search.

``load_model_from_yaml`` rebuilds the architecture with untrained weights, and the
QAT loop used a fine-tuning learning rate (5e-5). After the few epochs the tutorial
runs, the "quantized" model was still close to its random initialization, so
Tutorial 1 reported ~20% accuracy at every precision, including 16-bit.
"""

import numpy as np
import tensorflow as tf
import yaml

from utils.tf_local_search_separated import local_search_entrypoint

N_FEATURES, N_CLASSES = 16, 4


def _write_arch_yaml(path):
    arch = {
        "architecture": {
            "model_type": "BlockBased",
            "input_shape": [N_FEATURES],
            "output_dim": N_CLASSES,
            "components": [
                {
                    "block_type": "MLP",
                    "name": "classifier_head",
                    "params": {
                        "widths": [N_FEATURES, 32, N_CLASSES],
                        "activations": ["relu", "softmax"],
                        "normalizations": [None, None],
                    },
                }
            ],
        }
    }
    path.write_text(yaml.safe_dump(arch))


def _gaussian_blobs(n, seed):
    """Four well-separated classes: any trained linear-ish model gets ~100%."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(N_CLASSES, N_FEATURES))
    labels = rng.integers(0, N_CLASSES, size=n)
    x = (centers[labels] + rng.normal(size=(n, N_FEATURES))).astype("float32")
    return x, tf.keras.utils.to_categorical(labels, N_CLASSES)


def test_qat_only_loop_trains_from_untrained_yaml_model(tmp_path):
    tf.keras.utils.set_random_seed(0)
    arch_path = tmp_path / "arch.yaml"
    _write_arch_yaml(arch_path)
    config_path = tmp_path / "local_search_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "qat_settings": {
                    "epochs": 2,
                    "precision_pairs": [{"total_bits": 16, "int_bits": 6}, {"total_bits": 8, "int_bits": 3}],
                }
            }
        )
    )
    x_train, y_train = _gaussian_blobs(2000, seed=1)
    x_val, y_val = x_train[:500], y_train[:500]

    _, qat_df = local_search_entrypoint(
        architecture_yaml_path=str(arch_path),
        local_search_config_path=str(config_path),
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=str(tmp_path / "out"),
        run_pruning=False,
    )

    # Chance is 25%. With the old 5e-5 learning rate it stayed near or below chance (~0.06 here);
    # a trained ReLU model reaches 0.72-1.0 after 2 epochs depending on the seed.
    assert len(qat_df) == 2
    assert (qat_df["Accuracy"] > 0.5).all(), qat_df
