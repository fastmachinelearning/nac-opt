#!/usr/bin/env python3
"""
Baseline readout head: train (default **5 epochs**), optional **stratified k-fold** on the
combined train+test pool (same convention as ``GlobalSearchTF.create_block_objective`` for
qubit), then report **performance_metric** and **BOPs** + **rule4ml** estimates.

Architecture (FP32 Keras): ``Dense(4) -> BatchNorm -> Dense(1, sigmoid)``.

Outputs (JSON / stdout):

  - ``performance_metric`` — mean validation accuracy (k-fold) or single-split val accuracy
  - ``fold_accuracies`` — per-fold val accuracies when ``n_folds > 1``
  - ``bops``, ``lut_pct``, ``ff_pct``, ``bram_pct``, ``dsp_pct``, ``avg_resource``, ``clock_cycles``

Dataset handling matches ``run_local_search_slurm.sh``: load ``t3_config.yaml`` (``--config``
or ``tutorials/tutorial_3_qubit/t3_config.yaml`` next to this script) for ``dataset`` fields;
if ``--arch_yaml`` is set, IQ window ``start_location`` / ``window_size`` follow
``metadata.dataset_window`` with the same fallbacks as local search.

Usage (from ``tutorials/tutorial_3_qubit`` with conda env that has TensorFlow + rule4ml):

  python estimate_baseline_objectives.py --data_dir /path/to/qubit_npy_dir
  python estimate_baseline_objectives.py --data_dir ... --arch_yaml ./results/.../best_model_for_local_search.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.tf_bops import get_linear_bops_tf
from utils.tf_data_preprocessing import load_and_preprocess_qubit
from utils.tf_global_search import (
    GlobalSearchTF,
    _binary_loss_and_metrics_from_final_activation,
    _flatten_keras_model,
    _stratified_k_fold_indices,
)
from utils.tf_processor import evaluate_model, train_model


def build_keras_model(input_shape: int) -> tf.keras.Sequential:
    """FP32 baseline matching the global-search MLP head shape (no QKeras)."""
    m = tf.keras.Sequential(name="baseline_qubit_readout")
    m.add(layers.Dense(4, activation=None, name="fc1", input_shape=(input_shape,)))
    m.add(layers.BatchNormalization(name="batchnorm1"))
    m.add(layers.Dense(1, activation="sigmoid", name="fc2"))
    return m


def mlp_like_bops(model, input_features: int, bit_width: int = 32) -> float:
    """Sum ``get_linear_bops_tf`` over ``Dense``-like layers (``units``), skipping norm layers."""
    batch_dim = 1
    shape = (batch_dim, input_features)
    total = 0.0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            continue
        if hasattr(layer, "units"):
            total += float(get_linear_bops_tf(layer, bit_width=bit_width, input_shape=shape))
            shape = (batch_dim, int(layer.units))
    return total


def _bool01(v) -> int:
    return 1 if bool(v) else 0


def _defaults_from_config(path: str | None) -> dict:
    """Defaults from ``t3_config``-style YAML (CLI still overrides via argparse)."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ds = cfg.get("dataset") or {}
    sc = cfg.get("search") or {}
    out: dict = {}
    if "data_dir" in ds and ds["data_dir"] is not None:
        out["data_dir"] = str(ds["data_dir"])
    if "start_location" in ds:
        out["start_location"] = int(ds["start_location"])
    if "window_size" in ds:
        out["window_size"] = int(ds["window_size"])
    if "subset_size" in ds and ds["subset_size"] is not None:
        out["subset_size"] = int(ds["subset_size"])
    if "normalize" in ds:
        out["normalize"] = _bool01(ds["normalize"])
    if "flatten" in ds:
        out["flatten"] = _bool01(ds["flatten"])
    if "one_hot" in ds:
        out["one_hot"] = _bool01(ds["one_hot"])
    if "num_classes" in ds:
        out["num_classes"] = int(ds["num_classes"])
    if "epochs" in sc:
        out["epochs"] = int(sc["epochs"])
    if "n_folds" in sc:
        out["n_folds"] = int(sc["n_folds"])
    return out


def _explicit_arg_dests_from_argv(argv: list[str]) -> set[str]:
    """Which argparse dest names had a flag on the command line (so YAML must not override)."""
    bool_flags = {"quiet"}
    dests: set[str] = set()
    i = 0
    argv = list(argv)
    while i < len(argv):
        a = argv[i]
        if a == "--":
            break
        if not a.startswith("--"):
            i += 1
            continue
        if "=" in a:
            opt = a[2:].split("=", 1)[0]
            dests.add(opt.replace("-", "_"))
            i += 1
            continue
        opt = a[2:]
        dests.add(opt.replace("-", "_"))
        name = opt.replace("-", "_")
        if name not in bool_flags and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            i += 2
        else:
            i += 1
    return dests


def merge_yaml_config(args: argparse.Namespace, argv: list[str] | None = None) -> None:
    """Apply ``t3_config``-style YAML for keys not explicitly set on the CLI."""
    if not getattr(args, "config", None) or not os.path.isfile(args.config):
        return
    argv = argv if argv is not None else sys.argv[1:]
    explicit = _explicit_arg_dests_from_argv(argv)
    yd = _defaults_from_config(args.config)
    for k, v in yd.items():
        if k not in explicit:
            setattr(args, k, v)


def _resolve_t3_config_path(args: argparse.Namespace) -> str | None:
    """Prefer ``--config`` when it exists; else ``t3_config.yaml`` beside this script."""
    if getattr(args, "config", None) and os.path.isfile(args.config):
        return args.config
    sibling = Path(__file__).resolve().parent / "t3_config.yaml"
    if sibling.is_file():
        return str(sibling)
    return None


def apply_local_search_style_qubit_settings(
    args: argparse.Namespace, argv: list[str] | None = None
) -> dict:
    """
    Match ``run_local_search_slurm.sh`` inline Python: ``dataset`` from ``t3_config.yaml``,
    then ``start_location`` / ``window_size`` from ``metadata.dataset_window`` in
    ``--arch_yaml`` when provided (same fallbacks as local search).
    """
    argv = argv if argv is not None else sys.argv[1:]
    explicit = _explicit_arg_dests_from_argv(argv)
    provenance: dict = {"t3_config_resolved": None, "arch_yaml_resolved": None}

    ds_cfg: dict = {}
    t3_path = _resolve_t3_config_path(args)
    if t3_path:
        provenance["t3_config_resolved"] = os.path.abspath(t3_path)
        with open(t3_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ds_cfg = cfg.get("dataset") or {}

        if "subset_size" not in explicit and "subset_size" in ds_cfg:
            ss = ds_cfg.get("subset_size")
            args.subset_size = int(ss) if ss is not None else None
        if "normalize" not in explicit:
            args.normalize = _bool01(ds_cfg.get("normalize", True))
        if "flatten" not in explicit:
            args.flatten = _bool01(ds_cfg.get("flatten", True))
        if "one_hot" not in explicit:
            args.one_hot = _bool01(ds_cfg.get("one_hot", True))
        if "num_classes" not in explicit and "num_classes" in ds_cfg:
            args.num_classes = int(ds_cfg["num_classes"])

        if "start_location" not in explicit and "start_location" in ds_cfg:
            args.start_location = int(ds_cfg["start_location"])
        if "window_size" not in explicit and "window_size" in ds_cfg:
            args.window_size = int(ds_cfg["window_size"])

    arch_path = (getattr(args, "arch_yaml", None) or "").strip()
    if arch_path and os.path.isfile(arch_path):
        provenance["arch_yaml_resolved"] = os.path.abspath(arch_path)
        with open(arch_path, encoding="utf-8") as f:
            arch = yaml.safe_load(f) or {}
        window = (arch.get("metadata", {}) or {}).get("dataset_window", {}) or {}
        if "start_location" not in explicit:
            args.start_location = int(
                window.get("start_location", ds_cfg.get("start_location", args.start_location))
            )
        if "window_size" not in explicit:
            args.window_size = int(window.get("window_size", ds_cfg.get("window_size", args.window_size)))

    return provenance


def train_baseline(
    *,
    input_dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    n_folds: int,
    one_hot: bool,
    train_verbose: int,
) -> tuple[float, list[float] | None]:
    """
    Train baseline; return (performance_metric, fold_accuracies or None).

    Matches ``create_block_objective`` pooling: when ``n_folds > 1``, concatenate
    train and val (qubit loader's train + test files) then stratified k-fold.
    """
    loss_fn, metrics = _binary_loss_and_metrics_from_final_activation("sigmoid")
    patience = max(50, int(epochs) + 10)

    if n_folds <= 1:
        model = build_keras_model(input_dim)
        model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)
        train_model(
            model,
            (x_train, y_train),
            (x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=train_verbose,
            patience=patience,
        )
        val_metrics = evaluate_model(model, (x_val, y_val))
        return float(val_metrics["accuracy"]), None

    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    strat_one_hot = bool(one_hot) and (y_all.ndim > 1 and y_all.shape[1] > 1)
    fold_indices = _stratified_k_fold_indices(y_all, n_folds, one_hot=strat_one_hot)

    template = build_keras_model(input_dim)
    fold_accuracies: list[float] = []
    for fold_train_idx, fold_val_idx in fold_indices:
        xf_train, yf_train = x_all[fold_train_idx], y_all[fold_train_idx]
        xf_val, yf_val = x_all[fold_val_idx], y_all[fold_val_idx]

        fold_model = tf.keras.models.clone_model(template)
        fold_model.compile(optimizer="adam", loss=loss_fn, metrics=metrics)
        train_model(
            fold_model,
            (xf_train, yf_train),
            (xf_val, yf_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=train_verbose,
            patience=patience,
        )
        fold_metrics = evaluate_model(fold_model, (xf_val, yf_val))
        fold_accuracies.append(float(fold_metrics["accuracy"]))

    return float(np.mean(fold_accuracies)), fold_accuracies


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML (e.g. t3_config.yaml): fills dataset/search fields unless overridden on the CLI",
    )
    p.add_argument("--data_dir", type=str, default=None, help="Directory with qubit .npy files (required unless set via --config)")
    p.add_argument("--start_location", type=int, default=100)
    p.add_argument("--window_size", type=int, default=400)
    p.add_argument("--subset_size", type=int, default=None, help="Random subset of training set (None = all)")
    p.add_argument("--normalize", type=int, default=1, help="1/true: normalize using train statistics")
    p.add_argument("--flatten", type=int, default=1, help="1/true: flatten IQ window for MLP")
    p.add_argument("--one_hot", type=int, default=0, help="0/false: scalar 0/1 labels for sigmoid head (matches block output_dim=1)")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n_folds", type=int, default=3, help="Stratified k-fold on train+test pool; 1 = train on train, val on test")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--bop_bit_width",
        type=int,
        default=32,
        help="Bit width for analytic BOP formula (same style as global search)",
    )
    p.add_argument(
        "--hls_json",
        type=str,
        default=None,
        help="Optional JSON overriding rule4ml HLS/board config (else GlobalSearchTF defaults)",
    )
    p.add_argument("--json_out", type=str, default=None, help="Write report dict as JSON to this path")
    p.add_argument("--quiet", action="store_true", help="Skip model.summary(); training logs use verbose=0")
    p.add_argument(
        "--input_shape",
        type=int,
        default=None,
        help="If set, must match flattened feature dimension from the loader (sanity check)",
    )
    p.add_argument(
        "--arch_yaml",
        type=str,
        default=None,
        help="Global-search / local-search architecture YAML; IQ window from metadata.dataset_window (like run_local_search_slurm.sh)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    merge_yaml_config(args)
    data_provenance = apply_local_search_style_qubit_settings(args)

    if not args.data_dir:
        print("error: pass --data_dir or set data_dir in --config YAML", file=sys.stderr)
        sys.exit(2)

    normalize = bool(int(args.normalize))
    flatten = bool(int(args.flatten))
    one_hot = bool(int(args.one_hot))

    x_train, y_train, x_val, y_val = load_and_preprocess_qubit(
        data_dir=args.data_dir,
        start_location=args.start_location,
        window_size=args.window_size,
        subset_size=args.subset_size,
        normalize=normalize,
        flatten=flatten,
        one_hot=one_hot,
        num_classes=args.num_classes,
    )

    if len(x_train.shape) != 2:
        print(f"error: expected flattened x_train (N, F), got shape {x_train.shape}", file=sys.stderr)
        sys.exit(2)
    input_dim = int(x_train.shape[1])
    if args.input_shape is not None and int(args.input_shape) != input_dim:
        print(f"error: --input_shape {args.input_shape} != data feature dim {input_dim}", file=sys.stderr)
        sys.exit(2)

    train_verbose = 0 if args.quiet else 1

    performance_metric, fold_accuracies = train_baseline(
        input_dim=input_dim,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_folds=args.n_folds,
        one_hot=one_hot,
        train_verbose=train_verbose,
    )

    hw_cfg = None
    if args.hls_json and os.path.isfile(args.hls_json):
        with open(args.hls_json, encoding="utf-8") as f:
            hw_cfg = json.load(f)

    searcher = GlobalSearchTF(search_space_path=None, hls_config=hw_cfg, results_dir=str(Path.cwd() / "baseline_obj_tmp"))

    arch = build_keras_model(input_dim)
    if not args.quiet:
        arch.summary()

    bops_val = mlp_like_bops(arch, input_dim, bit_width=args.bop_bit_width)
    flat = _flatten_keras_model(arch)
    hw = searcher.calculate_hardware_metrics(flat, (input_dim,))

    report = {
        "model": "baseline Dense(4)+BN+Dense(1,sigmoid) FP32",
        "data_dir": os.path.abspath(args.data_dir),
        "t3_config_resolved": data_provenance.get("t3_config_resolved"),
        "arch_yaml_resolved": data_provenance.get("arch_yaml_resolved"),
        "start_location": args.start_location,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "n_folds": args.n_folds,
        "batch_size": args.batch_size,
        "input_shape": input_dim,
        "performance_metric": performance_metric,
        "fold_accuracies": fold_accuracies,
        "bops_analytic_bit_width": args.bop_bit_width,
        "bops": bops_val,
        "lut_pct": hw["lut_pct"],
        "ff_pct": hw["ff_pct"],
        "bram_pct": hw["bram_pct"],
        "dsp_pct": hw["dsp_pct"],
        "avg_resource": hw["avg_resource"],
        "clock_cycles": hw["clock_cycles"],
    }

    print("\n=== Baseline train + BOPs + rule4ml ===")
    print(json.dumps(report, indent=2))
    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
