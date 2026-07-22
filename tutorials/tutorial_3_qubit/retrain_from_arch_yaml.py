#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import yaml
import tensorflow as tf


def _write_model_summary(model, out_path):
    lines = []
    model.summary(print_fn=lambda s: lines.append(s), expand_nested=True, show_trainable=True)
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Train an FP32 model from a BlockBased architecture YAML.")
    ap.add_argument("--arch-yaml", required=True, type=Path, help="Path to BlockBased architecture YAML.")
    ap.add_argument("--data-dir", required=True, type=str, help="Directory containing qubit .npy files.")
    ap.add_argument("--epochs", type=int, default=30, help="Max number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    ap.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--results-dir", required=True, type=Path, help="Output directory for artifacts.")
    ap.add_argument("--subset-size", type=int, default=None, help="Optional subset size for training data.")
    ap.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience (epochs) on val_accuracy.")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Repo imports (script lives under tutorials/tutorial_3_qubit)
    repo_root = Path(__file__).resolve().parent.parent.parent
    import sys

    sys.path.insert(0, str(repo_root))
    from utils.tf_data_preprocessing import load_and_preprocess_qubit
    from utils.tf_arch_yaml_fp32 import load_model_from_yaml, loss_and_compile_metrics_from_arch_yaml

    arch = yaml.safe_load(args.arch_yaml.read_text()) or {}
    arch_cfg = arch.get("architecture") or {}
    window = ((arch.get("metadata") or {}).get("dataset_window") or {})
    start_location = int(window.get("start_location", 100))
    window_size = int(window.get("window_size", 400))

    output_dim = int(arch_cfg.get("output_dim", 2))
    # BlockBased qubit architectures use output_dim=1 (binary). In that case, do NOT one-hot labels.
    one_hot = False if output_dim == 1 else True

    x_train, y_train, x_val, y_val = load_and_preprocess_qubit(
        data_dir=args.data_dir,
        start_location=start_location,
        window_size=window_size,
        subset_size=args.subset_size,
        normalize=True,
        flatten=True,
        one_hot=one_hot,
        num_classes=2,
    )

    model = load_model_from_yaml(str(args.arch_yaml))
    loss_fn, metrics = loss_and_compile_metrics_from_arch_yaml(str(args.arch_yaml))
    if loss_fn is None:
        # Fallback for non-binary heads
        loss_fn = "categorical_crossentropy" if one_hot else "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    ckpt_best_path = args.results_dir / "best_model.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=int(args.early_stopping_patience),
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # Save final model (after training / restore_best_weights)
    final_model_path = args.results_dir / "final_model.h5"
    model.save(str(final_model_path), include_optimizer=False)

    # Save history
    hist_path = args.results_dir / "history.csv"
    with hist_path.open("w", newline="") as f:
        w = csv.writer(f)
        keys = list(history.history.keys())
        w.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            w.writerow([i + 1] + [history.history[k][i] for k in keys])

    # Save summary + resolved metadata
    _write_model_summary(model, args.results_dir / "model_summary.txt")
    meta_out = {
        "arch_yaml": str(args.arch_yaml.resolve()),
        "data_dir": os.path.abspath(args.data_dir),
        "start_location": start_location,
        "window_size": window_size,
        "max_epochs": args.epochs,
        "early_stopping_patience": int(args.early_stopping_patience),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "final_metrics": {k: float(v[-1]) for k, v in history.history.items()},
        "best_model_path": str(ckpt_best_path.resolve()),
        "final_model_path": str(final_model_path.resolve()),
    }
    (args.results_dir / "train_metadata.yaml").write_text(yaml.safe_dump(meta_out, sort_keys=False))

    # Copy arch yaml for provenance
    (args.results_dir / "arch.yaml").write_text(args.arch_yaml.read_text())

    print("Saved:", final_model_path)
    print("Best checkpoint:", ckpt_best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

