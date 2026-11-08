"""Seeded global search: same seed -> same run, and sampler vs training randomness can be separated."""

import ast

import optuna
import pandas as pd
import pytest
import yaml

from utils.tf_global_search import GlobalSearchTF

SEARCH_SPACE = {
    "mlp_num_layers_space": [2, 3, 4],
    "mlp_width_space": [4, 8, 16, 32],
    "act_space": ["relu", "tanh"],
    "norm_space": [None, "batch"],
    "output_activation": "softmax",
}


def _run(results_dir, n_trials=4, **seed_kwargs):
    searcher = GlobalSearchTF(search_space_path=dict(SEARCH_SPACE), results_dir=str(results_dir))
    study = searcher.run_search(
        model_type="mlp",
        n_trials=n_trials,
        epochs=2,
        dataset="blobs",
        loader_path="tests.synthetic_data:load_blobs",
        subset_size=None,
        verbose=False,
        deterministic_ops=True,
        **seed_kwargs,
    )
    df = pd.read_csv(results_dir / "mlp_search_results.csv").sort_values("trial")
    params = [ast.literal_eval(p) for p in df["params"]]
    return study, params, df["performance_metric"].tolist(), df["bops"].tolist()


def test_same_seed_reproduces_the_search(tmp_path):
    _, params_a, acc_a, bops_a = _run(tmp_path / "a", seed=7)
    _, params_b, acc_b, bops_b = _run(tmp_path / "b", seed=7)

    assert params_a == params_b
    assert bops_a == bops_b
    assert acc_a == acc_b  # bit-for-bit with deterministic_ops


def test_different_seeds_sample_different_architectures(tmp_path):
    _, params_a, _, _ = _run(tmp_path / "a", seed=7)
    _, params_b, _, _ = _run(tmp_path / "b", seed=8)
    assert params_a != params_b


def test_train_seed_changes_training_but_not_the_sampled_architectures(tmp_path):
    # n_trials <= population_size, so every trial is in NSGA-II's first (random) generation
    # and the proposals depend only on the sampler seed.
    _, params_a, acc_a, _ = _run(tmp_path / "a", sampler_seed=3, train_seed=1, population_size=10)
    _, params_b, acc_b, _ = _run(tmp_path / "b", sampler_seed=3, train_seed=2, population_size=10)

    assert params_a == params_b
    assert acc_a != acc_b


def test_search_settings_are_saved(tmp_path):
    _run(tmp_path, n_trials=1, seed=5, population_size=12)
    settings = yaml.safe_load((tmp_path / "search_settings.yaml").read_text())
    assert settings["sampler_seed"] == 5 and settings["train_seed"] == 5
    assert settings["population_size"] == 12


def test_workers_sharing_a_storage_all_use_nsga2(tmp_path, monkeypatch):
    """A worker that loads an existing study must still use NSGA-II (load_study alone gives TPE)."""
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    studies = []
    for rank in ("0", "1"):
        monkeypatch.setenv("SLURM_PROCID", rank)
        searcher = GlobalSearchTF(search_space_path=dict(SEARCH_SPACE), results_dir=str(tmp_path))
        studies.append(
            searcher.run_search(
                model_type="mlp",
                n_trials=1,
                epochs=1,
                dataset="blobs",
                loader_path="tests.synthetic_data:load_blobs",
                subset_size=None,
                verbose=False,
                storage=storage,
                study_name="shared",
                seed=11,
            )
        )
    assert all(isinstance(s.sampler, optuna.samplers.NSGAIISampler) for s in studies)
    assert len(studies[1].trials) == 2  # second worker joined the first worker's study


def test_deterministic_ops_without_a_seed_is_rejected(tmp_path):
    searcher = GlobalSearchTF(search_space_path=dict(SEARCH_SPACE), results_dir=str(tmp_path))
    with pytest.raises(ValueError, match="needs seed or train_seed"):
        searcher.run_search(
            model_type="mlp",
            n_trials=1,
            epochs=1,
            dataset="blobs",
            loader_path="tests.synthetic_data:load_blobs",
            deterministic_ops=True,
        )


def test_unseeded_runs_differ_and_determinism_is_switched_back_off(tmp_path):
    from utils.tf_global_search import _op_determinism_enabled

    _run(tmp_path / "seeded", n_trials=1, seed=1)  # turns determinism on, then must turn it off
    assert not _op_determinism_enabled()

    def unseeded(d):
        searcher = GlobalSearchTF(search_space_path=dict(SEARCH_SPACE), results_dir=str(d))
        searcher.run_search(
            model_type="mlp",
            n_trials=4,
            epochs=2,
            dataset="blobs",
            loader_path="tests.synthetic_data:load_blobs",
            verbose=False,
        )
        df = pd.read_csv(d / "mlp_search_results.csv").sort_values("trial")
        return [ast.literal_eval(p) for p in df["params"]]

    assert unseeded(tmp_path / "u1") != unseeded(tmp_path / "u2")
