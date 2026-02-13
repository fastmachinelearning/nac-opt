# File: utils/tf_local_search_combined.py

import io
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import yaml
from qkeras import QActivation, QConv2D, QDense, quantizers
from qkeras.utils import _add_supported_quantized_objects

from utils.tf_bops import get_linear_bops_tf
from utils.tf_global_search import _stratified_k_fold_indices
from utils.tf_local_search_separated import convert_to_qat_model, load_model_from_yaml


# --- Helper Functions ---


def _compute_baseline_bops(model, input_shape, bit_width=32):
    """
    Compute BOPs for a model, recursing into Sequential sub-models.

    load_model_from_yaml produces models with Sequential blocks as layers
    (e.g. mlp_block_0, classifier_head). The standard get_model_bops_tf
    only sees top-level layers, so this helper flattens the layer tree.
    """
    bops = 0
    current_input_shape = input_shape
    for layer in model.layers:
        if isinstance(layer, tf.keras.Sequential):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Dense):
                    bops += get_linear_bops_tf(sub_layer, bit_width, input_shape=current_input_shape)
                    current_input_shape = (current_input_shape[0], sub_layer.units)
        elif isinstance(layer, tf.keras.layers.Dense):
            bops += get_linear_bops_tf(layer, bit_width, input_shape=current_input_shape)
            current_input_shape = (current_input_shape[0], layer.units)
        elif isinstance(layer, tf.keras.layers.Flatten):
            flat_dim = 1
            for d in current_input_shape[1:]:
                flat_dim *= d
            current_input_shape = (current_input_shape[0], flat_dim)
    return bops


def _get_qkeras_custom_objects():
    """Returns a dict of QKeras custom objects for clone_model / strip_pruning."""
    co = {}
    _add_supported_quantized_objects(co)
    co.update({"QDense": QDense, "QActivation": QActivation, "QConv2D": QConv2D})
    return co


def _clone_qat_model(qat_model):
    """Clone a QKeras model with proper custom object handling."""
    co = _get_qkeras_custom_objects()
    with tf.keras.utils.custom_object_scope(co):
        cloned = tf.keras.models.clone_model(qat_model)
    cloned.set_weights(qat_model.get_weights())
    return cloned


def _compute_effective_bops(baseline_bops, total_bits, sparsity):
    """
    Effective BOPs = baseline_bops * (total_bits/32)^2 * (1 - sparsity)

    Accounts for both quantization (fewer bits per operation) and pruning
    (fewer non-zero operations).
    """
    quantization_factor = (total_bits / 32.0) ** 2
    pruning_factor = 1.0 - sparsity
    return baseline_bops * quantization_factor * pruning_factor


def _compute_model_sparsity(model):
    """Compute fraction of zero-valued weights across Dense/QDense layers."""
    total_params = 0
    zero_params = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, QDense)):
            for w in layer.get_weights():
                total_params += w.size
                zero_params += np.count_nonzero(w == 0)
    if total_params == 0:
        return 0.0
    return zero_params / total_params


class _QuietTraining:
    """Context manager that redirects all training output (stdout, stderr, TF/Python warnings) to a log file."""

    def __init__(self, log_file):
        self._log_file = log_file
        self._original_stdout = None
        self._original_stderr = None
        self._tf_log_level = None
        self._warning_filters = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._log_file
        sys.stderr = self._log_file
        # Suppress TF logging and Python warnings to console
        self._tf_log_level = logging.getLogger("tensorflow").level
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self._warning_filters = warnings.filters[:]
        warnings.filterwarnings("ignore")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        logging.getLogger("tensorflow").setLevel(self._tf_log_level)
        warnings.filters = self._warning_filters
        return False


# --- Core Combined Search ---


def run_combined_search(base_model, dataset, config, results_dir, loss_function, n_folds=1):
    """
    Combined QAT + iterative magnitude pruning loop.

    For each quantization precision:
      1. Convert FP32 model to QKeras model
      2. QAT warmup (fine-tune with quantization constraints)
      3. Save trained weights as LTH checkpoint
      4. Iteratively prune with LTH weight rewinding

    When ``n_folds > 1``, train and val data are concatenated and split into
    stratified folds. Each fold gets a fresh QAT model and independent pruning
    trajectory; reported accuracy at each iteration is the mean across folds.

    Full training logs are saved to ``training_log.txt`` in results_dir.
    Console output shows only a compact summary per step.

    Parameters:
        base_model: FP32 tf.keras.Model from load_model_from_yaml
        dataset: Tuple of (x_train, y_train, x_val, y_val)
        config: Dict with 'pruning_settings' and 'qat_settings'
        results_dir: Directory to save logs and model weights
        loss_function: Loss string (e.g. 'categorical_crossentropy')
        n_folds: Number of cross-validation folds (1 = no CV, default)

    Returns:
        pd.DataFrame with columns: Precision, TotalBits, IntBits, Iteration,
                                    Sparsity, Accuracy, EffectiveBOPs
    """
    x_train, y_train, x_val, y_val = dataset
    pruning_config = config["pruning_settings"]
    qat_config = config["qat_settings"]

    # Build fold splits
    # Check if validation data is empty (for CV scenarios where val is intentionally empty)
    # Handle various empty cases: None, empty array, or array with 0 samples
    try:
        has_validation = x_val is not None and (hasattr(x_val, '__len__') and len(x_val) > 0)
    except (TypeError, AttributeError):
        has_validation = False
    
    if n_folds > 1:
        if has_validation:
            x_all = np.concatenate([x_train, x_val], axis=0)
            y_all = np.concatenate([y_train, y_val], axis=0)
        else:
            # If validation is empty, use only training data for CV
            x_all = x_train
            y_all = y_train
        one_hot = len(y_all.shape) > 1 and y_all.shape[1] > 1
        fold_splits = _stratified_k_fold_indices(y_all, n_folds, one_hot=one_hot)
    else:
        # Single fold: use provided train/val directly
        fold_splits = None

    # Compute baseline FP32 BOPs once (recurse into Sequential sub-models)
    input_shape = (1,) + tuple(x_train.shape[1:])
    baseline_bops = _compute_baseline_bops(base_model, input_shape, bit_width=32)
    print(f"Baseline FP32 BOPs: {baseline_bops:.2e}")

    # CSV log
    log_path = os.path.join(results_dir, "combined_qat_pruning_log.csv")
    with open(log_path, "w") as f:
        f.write("Precision,TotalBits,IntBits,Iteration,Sparsity,Accuracy,EffectiveBOPs\n")

    # Full training log file (verbose Keras output goes here)
    training_log_path = os.path.join(results_dir, "training_log.txt")

    all_rows = []
    co = _get_qkeras_custom_objects()

    # Suppress noisy TF/Keras warnings for the duration of the search
    _prev_tf_level = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    _prev_warning_filters = warnings.filters[:]
    warnings.filterwarnings("ignore")

    n_iterations = pruning_config["iterations"]

    try:
      with open(training_log_path, "w") as training_log:
        for precision in qat_config["precision_pairs"]:
            total_bits, int_bits = precision["total_bits"], precision["int_bits"]
            precision_str = f"<{total_bits},{int_bits}>"
            print(f"\n{'='*20} Precision: {precision_str} {'='*20}")
            training_log.write(f"\n{'='*60}\nPrecision: {precision_str}\n{'='*60}\n")

            if fold_splits is not None:
                # --- K-fold path ---
                # Collect per-iteration accuracies across folds
                # Index 0 = baseline (iteration 0), indices 1..n_iterations = pruning iterations
                fold_iteration_accs = [[] for _ in range(n_iterations + 1)]
                fold_iteration_sparsities = [[] for _ in range(n_iterations + 1)]
                last_fold_best_weights = None
                last_fold_best_acc = -1.0

                for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
                    xf_train, yf_train = x_all[train_idx], y_all[train_idx]
                    xf_val, yf_val = x_all[val_idx], y_all[val_idx]

                    batch_size = 128
                    steps_per_epoch = max(1, len(xf_train) // batch_size)
                    pruning_frequency = max(1, steps_per_epoch // 2)

                    training_log.write(f"\n--- Fold {fold_idx + 1}/{n_folds} ---\n")

                    # Fresh QAT model per fold
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        qat_model = convert_to_qat_model(base_model, total_bits, int_bits)

                    qat_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
                    qat_model.compile(optimizer=qat_optimizer, loss=loss_function, metrics=["accuracy"])

                    print(f"  Fold {fold_idx + 1}/{n_folds}: QAT warmup...", end="", flush=True)
                    training_log.write(f"\n--- QAT Warmup ({qat_config['epochs']} epochs) ---\n")
                    training_log.flush()
                    with _QuietTraining(training_log):
                        qat_model.fit(
                            xf_train,
                            yf_train,
                            validation_data=(xf_val, yf_val),
                            epochs=qat_config["epochs"],
                            batch_size=batch_size,
                            verbose=1,
                        )

                    # LTH checkpoint for this fold
                    original_weights = qat_model.get_weights()

                    # Baseline (iteration 0)
                    _, val_acc_baseline = qat_model.evaluate(xf_val, yf_val, verbose=0)
                    fold_iteration_accs[0].append(val_acc_baseline)
                    fold_iteration_sparsities[0].append(0.0)
                    print(f" acc={val_acc_baseline:.4f}")
                    training_log.write(f"Fold {fold_idx + 1} baseline: accuracy={val_acc_baseline:.4f}\n")

                    # Iterative pruning for this fold
                    model_to_prune = _clone_qat_model(qat_model)
                    fold_best_acc = val_acc_baseline
                    fold_best_weights = qat_model.get_weights()

                    for i in range(n_iterations):
                        target_sparsity = 1 - (pruning_config["pruning_rate"] ** (i + 1))

                        pruning_params = {
                            "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                                target_sparsity, begin_step=0, frequency=pruning_frequency
                            )
                        }
                        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                            model_to_prune, **pruning_params
                        )
                        pruned_model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

                        training_log.write(
                            f"\n--- Fold {fold_idx + 1} Pruning iter {i + 1}/{n_iterations} "
                            f"(target sparsity={target_sparsity:.4f}) ---\n"
                        )
                        training_log.flush()
                        with _QuietTraining(training_log):
                            pruned_model.fit(
                                xf_train,
                                yf_train,
                                validation_data=(xf_val, yf_val),
                                epochs=pruning_config["epochs_per_iteration"],
                                batch_size=batch_size,
                                callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
                                verbose=1,
                            )

                        with tf.keras.utils.custom_object_scope(co):
                            model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
                        model_stripped.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

                        _, val_acc = model_stripped.evaluate(xf_val, yf_val, verbose=0)
                        actual_sparsity = _compute_model_sparsity(model_stripped)

                        fold_iteration_accs[i + 1].append(val_acc)
                        fold_iteration_sparsities[i + 1].append(actual_sparsity)

                        training_log.write(
                            f"Fold {fold_idx + 1} iter {i + 1}: "
                            f"accuracy={val_acc:.4f}, sparsity={actual_sparsity:.4f}\n"
                        )

                        if val_acc > fold_best_acc:
                            fold_best_acc = val_acc
                            fold_best_weights = model_stripped.get_weights()

                        # LTH weight rewinding
                        rewound_weights = [
                            orig * np.where(curr != 0, 1.0, 0.0)
                            for orig, curr in zip(original_weights, model_stripped.get_weights())
                        ]
                        model_to_prune.set_weights(rewound_weights)

                    # Track best weights from last fold for saving
                    if fold_best_acc > last_fold_best_acc:
                        last_fold_best_acc = fold_best_acc
                        last_fold_best_weights = fold_best_weights

                # Aggregate across folds and log results
                effective_bops_baseline = _compute_effective_bops(baseline_bops, total_bits, sparsity=0.0)
                avg_acc_baseline = float(np.mean(fold_iteration_accs[0]))
                print(
                    f"  Iter  0 (baseline): acc={avg_acc_baseline:.4f} ({n_folds} folds)"
                    f"  eff_bops={effective_bops_baseline:.2e}"
                )

                row = {
                    "Precision": precision_str,
                    "TotalBits": total_bits,
                    "IntBits": int_bits,
                    "Iteration": 0,
                    "Sparsity": 0.0,
                    "Accuracy": avg_acc_baseline,
                    "EffectiveBOPs": effective_bops_baseline,
                }
                all_rows.append(row)
                with open(log_path, "a") as f:
                    f.write(
                        f"'{precision_str}',{total_bits},{int_bits},0,"
                        f"0.0000,{avg_acc_baseline:.4f},{effective_bops_baseline:.2e}\n"
                    )

                best_avg_acc = avg_acc_baseline

                for i in range(n_iterations):
                    avg_acc = float(np.mean(fold_iteration_accs[i + 1]))
                    avg_sparsity = float(np.mean(fold_iteration_sparsities[i + 1]))
                    effective_bops = _compute_effective_bops(baseline_bops, total_bits, avg_sparsity)

                    best_marker = " *" if avg_acc > best_avg_acc else ""
                    if avg_acc > best_avg_acc:
                        best_avg_acc = avg_acc
                    print(
                        f"  Iter {i + 1:>{len(str(n_iterations))}}"
                        f"/{n_iterations}:"
                        f" acc={avg_acc:.4f} ({n_folds} folds)"
                        f"  sparsity={avg_sparsity:.4f}"
                        f"  eff_bops={effective_bops:.2e}{best_marker}"
                    )
                    training_log.write(
                        f"Avg iter {i + 1}: accuracy={avg_acc:.4f}, sparsity={avg_sparsity:.4f}, "
                        f"eff_bops={effective_bops:.2e}\n"
                    )

                    row = {
                        "Precision": precision_str,
                        "TotalBits": total_bits,
                        "IntBits": int_bits,
                        "Iteration": i + 1,
                        "Sparsity": avg_sparsity,
                        "Accuracy": avg_acc,
                        "EffectiveBOPs": effective_bops,
                    }
                    all_rows.append(row)
                    with open(log_path, "a") as f:
                        f.write(
                            f"'{precision_str}',{total_bits},{int_bits},{i + 1},"
                            f"{avg_sparsity:.4f},{avg_acc:.4f},{effective_bops:.2e}\n"
                        )

                # Save best model weights for this precision (from best fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    save_model = convert_to_qat_model(base_model, total_bits, int_bits)
                save_model.set_weights(last_fold_best_weights)
                save_path = os.path.join(results_dir, f"best_model_{total_bits}b{int_bits}i.weights.h5")
                save_model.save_weights(save_path)
                print(f"  Best {precision_str}: avg_acc={best_avg_acc:.4f} -> {save_path}")

            else:
                # --- Single-fold path (original behavior) ---
                batch_size = 128
                steps_per_epoch = max(1, len(x_train) // batch_size)
                pruning_frequency = max(1, steps_per_epoch // 2)

                # Check if validation data is empty (for CV scenarios where val is intentionally empty)
                # Use the same robust check as defined at top level
                eval_data = (x_val, y_val) if has_validation else (x_train, y_train)
                eval_label = "val" if has_validation else "train"

                # Step 1: Convert FP32 model to QKeras
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qat_model = convert_to_qat_model(base_model, total_bits, int_bits)

                # Step 2: QAT warmup
                qat_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
                qat_model.compile(optimizer=qat_optimizer, loss=loss_function, metrics=["accuracy"])

                print(f"  QAT warmup ({qat_config['epochs']} epochs)...", end="", flush=True)
                training_log.write(f"\n--- QAT Warmup ({qat_config['epochs']} epochs) ---\n")
                training_log.flush()
                fit_kwargs = {
                    "x": x_train,
                    "y": y_train,
                    "epochs": qat_config["epochs"],
                    "batch_size": batch_size,
                    "verbose": 1,
                }
                if has_validation:
                    fit_kwargs["validation_data"] = (x_val, y_val)
                with _QuietTraining(training_log):
                    qat_model.fit(**fit_kwargs)

                # Step 3: Save QAT-trained weights as LTH checkpoint
                original_weights = qat_model.get_weights()

                # Step 4: Evaluate baseline (iteration 0 = QAT only, no pruning)
                _, val_acc_baseline = qat_model.evaluate(eval_data[0], eval_data[1], verbose=0)
                effective_bops_baseline = _compute_effective_bops(baseline_bops, total_bits, sparsity=0.0)
                print(f" accuracy={val_acc_baseline:.4f} ({eval_label}), eff_bops={effective_bops_baseline:.2e}")
                training_log.write(
                    f"Baseline ({eval_label}): accuracy={val_acc_baseline:.4f}, eff_bops={effective_bops_baseline:.2e}\n"
                )

                row = {
                    "Precision": precision_str,
                    "TotalBits": total_bits,
                    "IntBits": int_bits,
                    "Iteration": 0,
                    "Sparsity": 0.0,
                    "Accuracy": val_acc_baseline,
                    "EffectiveBOPs": effective_bops_baseline,
                }
                all_rows.append(row)
                with open(log_path, "a") as f:
                    f.write(
                        f"'{precision_str}',{total_bits},{int_bits},0,"
                        f"0.0000,{val_acc_baseline:.4f},{effective_bops_baseline:.2e}\n"
                    )

                best_acc = val_acc_baseline
                best_weights = qat_model.get_weights()

                # Step 5: Iterative magnitude pruning on the QAT model
                model_to_prune = _clone_qat_model(qat_model)

                for i in range(n_iterations):
                    target_sparsity = 1 - (pruning_config["pruning_rate"] ** (i + 1))

                    pruning_params = {
                        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                            target_sparsity, begin_step=0, frequency=pruning_frequency
                        )
                    }
                    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                        model_to_prune, **pruning_params
                    )
                    pruned_model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

                    training_log.write(
                        f"\n--- Pruning iter {i + 1}/{n_iterations} "
                        f"(target sparsity={target_sparsity:.4f}) ---\n"
                    )
                    training_log.flush()
                    fit_kwargs = {
                        "x": x_train,
                        "y": y_train,
                        "epochs": pruning_config["epochs_per_iteration"],
                        "batch_size": batch_size,
                        "callbacks": [tfmot.sparsity.keras.UpdatePruningStep()],
                        "verbose": 1,
                    }
                    if has_validation:
                        fit_kwargs["validation_data"] = (x_val, y_val)
                    with _QuietTraining(training_log):
                        pruned_model.fit(**fit_kwargs)

                    with tf.keras.utils.custom_object_scope(co):
                        model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
                    model_stripped.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

                    _, val_acc = model_stripped.evaluate(eval_data[0], eval_data[1], verbose=0)
                    actual_sparsity = _compute_model_sparsity(model_stripped)
                    effective_bops = _compute_effective_bops(baseline_bops, total_bits, actual_sparsity)

                    best_marker = " *" if val_acc > best_acc else ""
                    print(
                        f"  Iter {i + 1:>{len(str(n_iterations))}}"
                        f"/{n_iterations}:"
                        f" acc={val_acc:.4f} ({eval_label})  sparsity={actual_sparsity:.4f}"
                        f"  eff_bops={effective_bops:.2e}{best_marker}"
                    )
                    training_log.write(
                        f"Result ({eval_label}): accuracy={val_acc:.4f}, sparsity={actual_sparsity:.4f}, "
                        f"eff_bops={effective_bops:.2e}\n"
                    )

                    row = {
                        "Precision": precision_str,
                        "TotalBits": total_bits,
                        "IntBits": int_bits,
                        "Iteration": i + 1,
                        "Sparsity": actual_sparsity,
                        "Accuracy": val_acc,
                        "EffectiveBOPs": effective_bops,
                    }
                    all_rows.append(row)
                    with open(log_path, "a") as f:
                        f.write(
                            f"'{precision_str}',{total_bits},{int_bits},{i + 1},"
                            f"{actual_sparsity:.4f},{val_acc:.4f},{effective_bops:.2e}\n"
                        )

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_weights = model_stripped.get_weights()

                    # LTH weight rewinding
                    rewound_weights = [
                        orig * np.where(curr != 0, 1.0, 0.0)
                        for orig, curr in zip(original_weights, model_stripped.get_weights())
                    ]
                    model_to_prune.set_weights(rewound_weights)

                # Save best model weights for this precision
                save_model = _clone_qat_model(qat_model)
                save_model.set_weights(best_weights)
                save_path = os.path.join(results_dir, f"best_model_{total_bits}b{int_bits}i.weights.h5")
                save_model.save_weights(save_path)
                print(f"  Best {precision_str}: acc={best_acc:.4f} -> {save_path}")

    finally:
        # Restore warning/logging state
        logging.getLogger("tensorflow").setLevel(_prev_tf_level)
        warnings.filters = _prev_warning_filters

    print(f"\nFull training logs: {training_log_path}")
    return pd.DataFrame(all_rows)


# --- Entrypoint ---


def combined_local_search_entrypoint(architecture_yaml_path, local_search_config_path, dataset, results_dir, n_folds=1):
    """
    Main entrypoint for combined QAT + pruning local search.

    Mirrors the API of local_search_entrypoint from tf_local_search_separated.py.

    Parameters:
        architecture_yaml_path: Path to best_model_for_local_search.yaml from global search
        local_search_config_path: Path to YAML with pruning_settings and qat_settings
        dataset: Tuple of (x_train, y_train, x_val, y_val) numpy arrays
        results_dir: Directory to save results and model weights
        n_folds: Number of cross-validation folds (1 = no CV, default)

    Returns:
        pd.DataFrame with columns: Precision, TotalBits, IntBits, Iteration,
                                    Sparsity, Accuracy, EffectiveBOPs
    """
    folds_str = f" (k-fold CV, {n_folds} folds)" if n_folds > 1 else ""
    print("\n" + "=" * 50 + f"\n STARTING COMBINED QAT+PRUNING LOCAL SEARCH{folds_str} \n" + "=" * 50)

    os.makedirs(results_dir, exist_ok=True)
    with open(local_search_config_path, "r") as f:
        config = yaml.safe_load(f)

    x_train, y_train, x_val, y_val = dataset
    loss_function = (
        "categorical_crossentropy" if len(y_train.shape) > 1 and y_train.shape[1] > 1 else "sparse_categorical_crossentropy"
    )

    base_model = load_model_from_yaml(architecture_yaml_path)

    results_df = run_combined_search(
        base_model=base_model,
        dataset=dataset,
        config=config,
        results_dir=results_dir,
        loss_function=loss_function,
        n_folds=n_folds,
    )

    print("\n" + "=" * 50 + "\n COMBINED LOCAL SEARCH COMPLETE \n" + "=" * 50)
    return results_df
