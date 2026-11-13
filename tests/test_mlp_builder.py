"""Tests for build_mlp_from_config and the MLP global-search objective.

Regression for: Tutorial 1 crashed on its first trial with
``TypeError: build_mlp_from_config() got an unexpected keyword argument
'output_activation'`` because the MLP objective sends a list-format config
that the builder did not understand.
"""

import numpy as np
import optuna
import pytest
import tensorflow as tf

from utils.tf_global_search import GlobalSearchTF
from utils.tf_model_builder import build_mlp_from_config


def _layer_types(model):
    return [type(layer).__name__ for layer in model.layers]


def test_list_format_builds_layers_in_order():
    config = {
        "hidden_units": [32, 16],
        "activations": ["tanh", "relu"],
        "normalizations": ["batch", None],
    }
    model = build_mlp_from_config(config, input_size=8, num_classes=4, output_activation="softmax")

    assert _layer_types(model) == ["Dense", "BatchNormalization", "Activation", "Dense", "Dense"]
    assert model.layers[0].units == 32 and not model.layers[0].use_bias
    assert model.layers[2].activation.__name__ == "tanh"
    assert model.layers[3].units == 16 and model.layers[3].activation.__name__ == "relu"
    assert model.layers[-1].units == 4 and model.layers[-1].activation.__name__ == "softmax"
    assert isinstance(model.loss, tf.keras.losses.CategoricalCrossentropy)
    assert model.output_shape == (None, 4)


def test_legacy_numbered_format_still_works():
    config = {
        "num_layers": 3,
        "hidden_units1": 16,
        "activation1": "relu",
        "batchnorm1": True,
        "hidden_units2": 8,
        "activation2": "sigmoid",
        "batchnorm2": False,
    }
    model = build_mlp_from_config(config, input_size=8, num_classes=3)

    assert _layer_types(model) == ["Dense", "BatchNormalization", "Activation", "Dense", "Dense"]
    assert model.layers[3].units == 8 and model.layers[3].activation.__name__ == "sigmoid"
    assert model.layers[-1].activation.__name__ == "softmax"


def test_linear_single_unit_head_uses_logits_and_zero_threshold():
    config = {"hidden_units": [4], "activations": ["relu"], "normalizations": [None]}
    model = build_mlp_from_config(config, input_size=8, num_classes=1, output_activation=None)

    assert model.layers[-1].activation.__name__ == "linear"
    assert isinstance(model.loss, tf.keras.losses.BinaryCrossentropy)
    assert model.loss.get_config()["from_logits"] is True

    # Force every logit to +0.2 and label everything 1. With a logit threshold
    # of 0 that is 100% accuracy; with a probability threshold of 0.5 it would be 0%.
    model.set_weights([np.zeros_like(w) for w in model.get_weights()])
    out = model.layers[-1]
    out.set_weights([out.get_weights()[0], np.full_like(out.get_weights()[1], 0.2)])
    x = np.ones((10, 8), dtype="float32")
    y = np.ones((10, 1), dtype="float32")
    _, accuracy = model.evaluate(x, y, verbose=0)
    assert accuracy == pytest.approx(1.0)


def test_mismatched_list_lengths_raise():
    config = {"hidden_units": [8, 8], "activations": ["relu"], "normalizations": [None, None]}
    with pytest.raises(ValueError, match="equal length"):
        build_mlp_from_config(config, input_size=4, num_classes=2)


def test_mlp_objective_completes_a_trial(tmp_path):
    """The MLP search objective runs end to end (this is the call that crashed)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 8)).astype("float32")
    y = tf.keras.utils.to_categorical(rng.integers(0, 3, size=256), 3)

    searcher = GlobalSearchTF(
        search_space_path={
            "mlp_width_space": [4, 8],
            "act_space": ["relu", "tanh"],
            "norm_space": [None, "batch"],
            "output_activation": "softmax",
        },
        results_dir=str(tmp_path),
    )
    searcher.objective_names = ["performance_metric", "bops"]
    searcher.maximize_flags = [True, False]
    objective = searcher.create_mlp_objective(x, y, x, y, epochs=1, use_hardware_metrics=False, verbose=False)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len(searcher.results) == 2
    assert all((tmp_path / "trial_yamls" / f"trial_{i}_arch.yaml").exists() for i in range(2))
