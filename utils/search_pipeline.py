"""
High-level orchestration utilities for running SNAC-Pack global + local search.

This module reuses the existing building blocks:

- `GlobalSearchTF` from `tf_global_search.py` for the global (Optuna) search
- `local_search_entrypoint` from `tf_local_search_separated.py` for MLP-style local search
- `combined_local_search_entrypoint` from `tf_local_search_combined.py` for block-based local search

The main entrypoint is `run_pipeline_from_config`, which is designed to work
with the same YAML config structure used by the tutorial scripts
(`t1_config.yaml`, `t2_config.yaml`, `t3_config.yaml`).
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from utils.search_planner import (
    build_search_config,
    write_search_config,
)
from utils.dataset_catalog import describe_dataset
from utils.dataset_inspector import inspect_dataset_path
from utils.request_inference import infer_constraints_from_request
from utils.tf_global_search import GlobalSearchTF
from utils.tf_local_search_separated import local_search_entrypoint
from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_generic_dataset


def _build_local_search_config_yaml(ls_cfg: Dict[str, Any], results_dir: str) -> str:
    """
    Materialize a minimal local-search config YAML (pruning + QAT) into `results_dir`
    and return its path.
    """
    local_search_settings = {
        "pruning_settings": {
            "iterations": ls_cfg["pruning_iterations"],
            "epochs_per_iteration": ls_cfg["pruning_epochs"],
            "pruning_rate": ls_cfg["pruning_rate"],
        },
        "qat_settings": {
            "epochs": ls_cfg["qat_epochs"],
            "precision_pairs": ls_cfg["precision_pairs"],
        },
    }
    os.makedirs(results_dir, exist_ok=True)
    local_config_path = os.path.join(results_dir, "local_search_config.yaml")
    with open(local_config_path, "w") as f:
        yaml.dump(local_search_settings, f)
    return local_config_path


def _resolve_dataset_loader_kwargs(
    dataset_cfg: Dict[str, Any],
    config_path: str,
    *,
    flatten_override: bool | None = None,
    one_hot_override: bool | None = None,
) -> Dict[str, Any]:
    """
    Build kwargs for ``load_generic_dataset`` from a dataset config section.
    """
    loader_kwargs = dict(dataset_cfg.get("loader_kwargs", {}))

    info_only_keys = {
        "name",
        "display_name",
        "description",
        "loader_path",
        "modality",
        "input_shape",
        "sample_count",
        "task_type",
        "notes",
        "num_classes",
        "dataset_path",
        "resolved_path",
        "constraints_hint",
    }
    for key, value in dataset_cfg.items():
        if key in info_only_keys or key == "loader_kwargs":
            continue
        loader_kwargs.setdefault(key, value)

    if flatten_override is not None:
        loader_kwargs["flatten"] = flatten_override
    if one_hot_override is not None:
        loader_kwargs["one_hot"] = one_hot_override

    base_dir = Path(config_path).resolve().parent
    for key, value in list(loader_kwargs.items()):
        if key.endswith("_dir") and isinstance(value, str) and value and not os.path.isabs(value):
            loader_kwargs[key] = str((base_dir / value).resolve())

    return loader_kwargs


def _load_dataset_for_local_search(
    dataset_cfg: Dict[str, Any],
    dataset_name: str,
    config_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recreate the dataset splits for local search, mirroring the tutorial scripts.
    """
    if dataset_name == "qubit":
        loader_kwargs = _resolve_dataset_loader_kwargs(
            dataset_cfg,
            config_path,
            flatten_override=dataset_cfg.get("flatten", True),
            one_hot_override=True,
        )
        x_train, y_train, _, _ = load_generic_dataset(
            dataset_name=dataset_name,
            **loader_kwargs,
        )
        x_val_empty = np.empty((0, *x_train.shape[1:]), dtype=x_train.dtype)
        y_val_empty = np.empty((0, *y_train.shape[1:]), dtype=y_train.dtype)
        return x_train, y_train, x_val_empty, y_val_empty

    loader_kwargs = _resolve_dataset_loader_kwargs(
        dataset_cfg,
        config_path,
        flatten_override=dataset_cfg.get("flatten", dataset_name == "mnist"),
        one_hot_override=True,
    )
    return load_generic_dataset(
        dataset_name=dataset_name,
        **loader_kwargs,
    )


def _run_global_search_from_config(
    cfg: Dict[str, Any],
    config_path: str,
) -> Tuple[GlobalSearchTF, Any]:
    """
    Run the global Optuna search given a full tutorial-style config dict.

    Returns the `GlobalSearchTF` instance and the completed Optuna Study.
    """
    ds_cfg = cfg["dataset"]
    s_cfg = cfg["search"]
    ss_cfg = cfg["search_space"]
    out_cfg = cfg["output"]

    results_dir = out_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    searcher = GlobalSearchTF(
        search_space_path=ss_cfg,
        results_dir=results_dir,
    )
    searcher.selection_constraints = dict(s_cfg.get("selection", {}))

    obj_names = s_cfg["objective_names"]
    max_flags = s_cfg["maximize_flags"]
    dataset_name = ds_cfg["name"]

    run_search_kwargs: Dict[str, Any] = dict(
        model_type=s_cfg["model_type"],
        n_trials=s_cfg["n_trials"],
        epochs=s_cfg["epochs"],
        dataset=dataset_name,
        subset_size=ds_cfg.get("subset_size"),
        objectives=obj_names,
        maximize_flags=max_flags,
        use_hardware_metrics=s_cfg["use_hardware_metrics"],
        one_hot=ds_cfg.get("one_hot", False),
        n_folds=s_cfg.get("n_folds", 1),
    )

    loader_kwargs = _resolve_dataset_loader_kwargs(
        ds_cfg,
        config_path,
        flatten_override=(s_cfg["model_type"] == "mlp"),
        one_hot_override=(s_cfg["model_type"] == "mlp") or ds_cfg.get("one_hot", False),
    )
    run_search_kwargs.update(loader_kwargs)
    if ds_cfg.get("loader_path"):
        run_search_kwargs["loader_path"] = ds_cfg["loader_path"]

    study = searcher.run_search(**run_search_kwargs)
    return searcher, study


def run_pipeline_from_config(
    config_path: str,
    run_local_search: bool = True,
) -> Dict[str, Any]:
    """
    High-level pipeline:

    1. Run global Optuna search via `GlobalSearchTF.run_search`.
    2. (Optional) Run local search (QAT + pruning) using the best architecture.

    Parameters
    ----------
    config_path
        Path to a tutorial-style YAML config (e.g. `tutorial_1_mlp/t1_config.yaml`).
    run_local_search
        If True, run the appropriate local search stage after global search.

    Returns
    -------
    dict
        A small dictionary summarizing key artifacts, e.g.:

        {
            "results_dir": ...,
            "architecture_yaml": ...,
            "local_results_dir": ... or None,
        }
    """
    cfg = yaml.safe_load(open(config_path, "r"))
    ds_cfg = cfg["dataset"]
    s_cfg = cfg["search"]
    ls_cfg = cfg["local_search"]
    out_cfg = cfg["output"]

    results_dir = out_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    searcher, study = _run_global_search_from_config(cfg, config_path=config_path)

    arch_yaml_path = os.path.join(results_dir, "best_model_for_local_search.yaml")
    if not os.path.exists(arch_yaml_path):
        raise FileNotFoundError(
            f"Expected best architecture YAML not found at {arch_yaml_path}. "
            "Ensure global search completed successfully."
        )

    summary: Dict[str, Any] = {
        "results_dir": results_dir,
        "architecture_yaml": arch_yaml_path,
        "optuna_study": study,
        "local_results_dir": None,
        "local_results": None,
    }

    if not run_local_search:
        return summary

    local_config_path = _build_local_search_config_yaml(ls_cfg, results_dir=results_dir)

    dataset_name = ds_cfg["name"]
    x_train, y_train, x_val, y_val = _load_dataset_for_local_search(
        dataset_cfg=ds_cfg,
        dataset_name=dataset_name,
        config_path=config_path,
    )

    if s_cfg["model_type"] == "mlp":
        local_results_dir = os.path.join(results_dir, "local_search_separated")
        pruning_df, qat_df = local_search_entrypoint(
            architecture_yaml_path=arch_yaml_path,
            local_search_config_path=local_config_path,
            dataset=(x_train, y_train, x_val, y_val),
            results_dir=local_results_dir,
        )
        summary["local_results_dir"] = local_results_dir
        summary["local_results"] = {
            "pruning": pruning_df,
            "qat": qat_df,
        }
        return summary

    local_results_dir = os.path.join(results_dir, "local_search_combined")
    n_folds = s_cfg.get("n_folds", 1)
    combined_df = combined_local_search_entrypoint(
        architecture_yaml_path=arch_yaml_path,
        local_search_config_path=local_config_path,
        dataset=(x_train, y_train, x_val, y_val),
        results_dir=local_results_dir,
        n_folds=n_folds,
    )
    summary["local_results_dir"] = local_results_dir
    summary["local_results"] = combined_df
    return summary


def materialize_config(
    config: Dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Write a config dict to disk and return the absolute path.
    """
    return write_search_config(config, output_path=output_path)


def run_pipeline_from_spec(
    dataset_spec: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
    *,
    run_local_search: bool = True,
    config_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Build a config from dataset metadata + constraints, write it, and execute the
    standard search pipeline.
    """
    config = build_search_config(dataset_spec, constraints)
    config_path = materialize_config(config, output_path=config_output_path)
    summary = run_pipeline_from_config(str(config_path), run_local_search=run_local_search)
    summary["generated_config"] = str(config_path)
    summary["planned_dataset"] = dataset_spec
    summary["planned_constraints"] = constraints or {}
    return summary


def run_agentic_search(
    request_text: str,
    *,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_spec: Dict[str, Any] | None = None,
    constraints: Dict[str, Any] | None = None,
    run_local_search: bool | None = None,
    config_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    High-level plain-English entrypoint for agent-driven search.

    The caller may provide either a dataset path to inspect or an already-built
    dataset spec. Built-in dataset names can be resolved through the dataset
    catalog. Plain-English request text is converted into planner
    constraints and merged with any explicit overrides.
    """
    if not dataset_spec and not dataset_path and not dataset_name:
        raise ValueError("Provide dataset_path, dataset_name, or dataset_spec.")

    inferred_spec: Dict[str, Any] = {}
    if dataset_path:
        inferred_spec = inspect_dataset_path(dataset_path)
    elif dataset_name:
        inferred_spec = describe_dataset(dataset_name)["profile"]

    merged_dataset_spec = dict(inferred_spec)
    merged_dataset_spec.update(dataset_spec or {})

    inferred_constraints = infer_constraints_from_request(request_text)
    merged_constraints = dict(inferred_constraints)
    merged_constraints.update(constraints or {})

    if run_local_search is None:
        run_local_search = not bool(merged_constraints.pop("disable_local_search", False))

    summary = run_pipeline_from_spec(
        dataset_spec=merged_dataset_spec,
        constraints=merged_constraints,
        run_local_search=run_local_search,
        config_output_path=config_output_path,
    )
    summary["request_text"] = request_text
    summary["inspected_dataset"] = merged_dataset_spec
    summary["inferred_constraints"] = inferred_constraints
    return summary


def _detect_local_search_mode(arch_config: Dict[str, Any]) -> str:
    """
    Infer the right local-search dispatch from a parsed best-architecture YAML.

    MLP-only architectures (only MLP and optional Flatten blocks) go through
    the separated pruning+QAT loop. Anything with Conv or ConvAttn blocks
    needs the combined entrypoint.
    """
    components = arch_config.get("architecture", {}).get("components", [])
    block_types = {str(c.get("block_type", "")) for c in components}
    if block_types - {"MLP", "Flatten", ""}:
        return "combined"
    return "separated"


def _flat_to_nested_local_config(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert planner-style flat local_search keys to the nested form the entrypoints consume."""
    return {
        "pruning_settings": {
            "iterations": int(flat["pruning_iterations"]),
            "epochs_per_iteration": int(flat["pruning_epochs"]),
            "pruning_rate": float(flat["pruning_rate"]),
        },
        "qat_settings": {
            "epochs": int(flat["qat_epochs"]),
            "precision_pairs": list(flat["precision_pairs"]),
        },
    }


def _resolve_dataset_for_local_search(
    *,
    dataset_spec: Dict[str, Any] | None,
    dataset_name: str | None,
    dataset_path: str | None,
    flatten: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load the dataset given any of the three spec styles. Returns the split
    plus the resolved spec dict (handy for the caller's summary).
    """
    from utils.dataset_inspector import inspect_dataset_path

    if dataset_spec is None and dataset_path:
        dataset_spec = inspect_dataset_path(dataset_path)
    elif dataset_spec is None and dataset_name:
        dataset_spec = {"name": dataset_name}

    if dataset_spec is None:
        raise ValueError("Provide dataset_spec, dataset_name, or dataset_path.")

    spec = deepcopy(dataset_spec)
    loader_kwargs = dict(spec.get("loader_kwargs", {}))

    info_keys = {
        "name", "display_name", "description", "loader_path", "modality",
        "input_shape", "sample_count", "task_type", "notes", "num_classes",
        "dataset_path", "resolved_path", "constraints_hint",
    }
    for key, value in spec.items():
        if key in info_keys or key == "loader_kwargs":
            continue
        loader_kwargs.setdefault(key, value)

    loader_kwargs["flatten"] = bool(flatten)
    loader_kwargs.setdefault("one_hot", True)
    loader_kwargs.setdefault("normalize", True)

    name = str(spec.get("name") or dataset_name or "custom_dataset")
    x_train, y_train, x_val, y_val = load_generic_dataset(dataset_name=name, **loader_kwargs)
    return x_train, y_train, x_val, y_val, spec


def run_local_search(
    architecture_yaml_path: str | Path,
    *,
    dataset_spec: Dict[str, Any] | None = None,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    local_search_config: Dict[str, Any] | None = None,
    budget: str | None = None,
    mode: str = "auto",
    results_dir: str | Path | None = None,
    n_folds: int = 1,
) -> Dict[str, Any]:
    """
    Run local search (QAT + pruning) on an existing best-architecture YAML.

    The caller can describe the dataset three ways:
        - ``dataset_spec``: a dict matching the inspector output (highest fidelity)
        - ``dataset_path``: a file path, inspected on the fly
        - ``dataset_name``: a built-in dataset name (mnist / fashion_mnist / qubit)

    The local-search knobs come from ``local_search_config`` (flat planner format)
    OR ``budget`` (``light`` / ``balanced`` / ``heavy``). ``mode`` is
    ``auto`` (detect from architecture YAML), ``separated``, or ``combined``.

    Returns a summary dict with paths and dataframes the agent can read back.
    """
    import yaml as _yaml
    from utils.search_planner import _build_local_search_config

    arch_path = Path(architecture_yaml_path).expanduser().resolve()
    if not arch_path.is_file():
        raise FileNotFoundError(f"Architecture YAML not found: {arch_path}")

    with open(arch_path, "r", encoding="utf-8") as handle:
        arch_config = _yaml.safe_load(handle)

    resolved_mode = mode if mode != "auto" else _detect_local_search_mode(arch_config)
    if resolved_mode not in ("separated", "combined"):
        raise ValueError(f"Unknown local-search mode: {mode!r}")

    if local_search_config is None:
        budget = (budget or "balanced").lower()
        constraints = {"local_search": {"budget": budget}}
        flat_cfg, _ = _build_local_search_config(constraints, use_hardware_metrics=False)
    else:
        flat_cfg = dict(local_search_config)

    input_shape = arch_config.get("architecture", {}).get("input_shape", [])
    flatten = len(input_shape) == 1

    if results_dir is None:
        subdir = "local_search_combined" if resolved_mode == "combined" else "local_search_separated"
        results_dir = arch_path.parent / subdir
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    local_config_path = results_dir / "local_search_config.yaml"
    with open(local_config_path, "w", encoding="utf-8") as handle:
        _yaml.safe_dump(_flat_to_nested_local_config(flat_cfg), handle)

    x_train, y_train, x_val, y_val, resolved_spec = _resolve_dataset_for_local_search(
        dataset_spec=dataset_spec,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        flatten=flatten,
    )

    summary: Dict[str, Any] = {
        "architecture_yaml": str(arch_path),
        "local_search_config_path": str(local_config_path),
        "results_dir": str(results_dir),
        "mode": resolved_mode,
        "budget": budget,
        "flat_local_search_config": flat_cfg,
        "dataset_spec": resolved_spec,
    }

    if resolved_mode == "separated":
        pruning_df, qat_df = local_search_entrypoint(
            architecture_yaml_path=str(arch_path),
            local_search_config_path=str(local_config_path),
            dataset=(x_train, y_train, x_val, y_val),
            results_dir=str(results_dir),
        )
        summary["pruning_rows"] = pruning_df.to_dict(orient="records") if pruning_df is not None else None
        summary["qat_rows"] = qat_df.to_dict(orient="records") if qat_df is not None else None
    else:
        combined_df = combined_local_search_entrypoint(
            architecture_yaml_path=str(arch_path),
            local_search_config_path=str(local_config_path),
            dataset=(x_train, y_train, x_val, y_val),
            results_dir=str(results_dir),
            n_folds=int(n_folds),
        )
        summary["combined_rows"] = combined_df.to_dict(orient="records") if combined_df is not None else None

    return summary


__all__ = [
    "materialize_config",
    "run_agentic_search",
    "run_local_search",
    "run_pipeline_from_config",
    "run_pipeline_from_spec",
]


if __name__ == "__main__":
    """
    Simple CLI entrypoint so this module can be run directly, e.g.:

        python -m utils.search_pipeline \
            --config tutorials/tutorial_3_qubit/t3_config.yaml \
            --no-local-search

    or, from the repo root:

        python utils/search_pipeline.py \
            --config tutorials/tutorial_3_qubit/t3_config.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run SNAC-Pack global (+ optional local) search pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a tutorial-style YAML config (relative to current working directory).",
    )
    parser.add_argument(
        "--no-local-search",
        action="store_true",
        help="If set, skip the local QAT/pruning stage.",
    )

    args = parser.parse_args()

    summary = run_pipeline_from_config(
        config_path=args.config,
        run_local_search=not args.no_local_search,
    )
    # Print a concise summary to stdout
    print(summary)

    # run with:

    # conda activate rule4ml_update
    # export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    # python -m utils.search_pipeline --config tutorials/tutorial_3_qubit/t3_config.yaml
