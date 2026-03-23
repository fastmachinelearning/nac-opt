"""
Load the best local-search model at a given precision and build an hls4ml HLS project.
Used by Tutorial 3 (qubit) Synthesis section.
"""
import os
from pathlib import Path

import yaml

from utils.tf_local_search_separated import load_model_from_yaml, convert_to_qat_model


def run_synthesis_from_config(
    config_path=None,
    config=None,
    results_dir=None,
    base_dir=None,
):
    """
    Run synthesis from tutorial config: load best QAT model at configured precision
    and build the hls4ml HLS project.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config (e.g. t3_config.yaml). Ignored if config is provided.
    config : dict, optional
        Config dict with keys: synthesis (total_bits, int_bits, hls_output_dir),
        optional hls_config, optional output.results_dir.
    results_dir : str or Path, optional
        Override results directory (contains best_model_for_local_search.yaml and
        local_search_combined/). If None, uses config["output"]["results_dir"].
    base_dir : str or Path, optional
        If set, resolve relative results_dir and synthesis.hls_output_dir against this.

    Returns
    -------
    model : tf.keras.Model
        Loaded QKeras model at the synthesis precision.
    input_shape : tuple
        Input shape (without batch) for the model.
    hls_model : hls4ml.model.HLSModel
        hls4ml model. Call hls_model.compile() to generate HLS, or hls_model.build() for full synthesis.
    """
    if config is None:
        if config_path is None:
            raise ValueError("Provide either config_path or config")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    out_cfg = config.get("output", {})
    res_dir = results_dir if results_dir is not None else out_cfg.get("results_dir", ".")
    res_dir = Path(res_dir)
    if base_dir is not None:
        base_dir = Path(base_dir)
        res_dir = (base_dir / res_dir).resolve()

    syn_cfg = config["synthesis"]
    total_bits = syn_cfg["total_bits"]
    int_bits = syn_cfg["int_bits"]
    hls_output_dir = Path(syn_cfg["hls_output_dir"])
    if base_dir is not None:
        hls_output_dir = (base_dir / hls_output_dir).resolve()
    else:
        hls_output_dir = hls_output_dir.resolve()

    architecture_yaml_path = res_dir / "best_model_for_local_search.yaml"
    local_results_dir = res_dir / "local_search_combined"

    if not architecture_yaml_path.exists():
        raise FileNotFoundError(
            f"Best architecture YAML not found: {architecture_yaml_path}. Run global search first."
        )

    model, input_shape = load_best_model_at_precision(
        str(architecture_yaml_path),
        str(local_results_dir),
        total_bits=total_bits,
        int_bits=int_bits,
    )
    hls_config = config.get("hls_config")
    hls_model = build_hls_model(model, str(hls_output_dir), hls_config=hls_config)
    return model, input_shape, hls_model


def load_best_model_at_precision(
    architecture_yaml_path,
    local_results_dir,
    total_bits=8,
    int_bits=3,
):
    """
    Load the best local-search QAT model at the specified precision.

    Local search saves weights as best_model_{total_bits}b{int_bits}i.weights.h5.
    We rebuild the QAT architecture from the global-search YAML and load those weights.

    Parameters
    ----------
    architecture_yaml_path : str or Path
        Path to best_model_for_local_search.yaml from global search.
    local_results_dir : str or Path
        Directory containing best_model_*b*i.weights.h5 from combined local search.
    total_bits : int
        Total bit width (e.g. 8).
    int_bits : int
        Integer bit width (e.g. 3).

    Returns
    -------
    model : tf.keras.Model
        Loaded QKeras model (QAT) with best weights for this precision.
    input_shape : tuple
        Input shape (without batch) for the model, e.g. (800,) for flattened qubit.
    """
    with open(architecture_yaml_path, "r") as f:
        arch_config = yaml.safe_load(f)
    input_shape = tuple(arch_config["architecture"]["input_shape"])

    base_model = load_model_from_yaml(str(architecture_yaml_path))
    model = convert_to_qat_model(base_model, total_bits, int_bits)

    weights_path = Path(local_results_dir) / f"best_model_{total_bits}b{int_bits}i.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Best-model weights not found: {weights_path}. "
            "Run the Local Search (combined QAT + pruning) section first."
        )
    model.load_weights(str(weights_path))
    return model, input_shape


def build_hls_model(keras_model, output_dir, hls_config=None):
    """
    Convert the Keras/QKeras model to an hls4ml HLS project and write files.
    Does not run synthesis; use the returned hls_model in a separate cell to call
    hls_model.compile() or hls_model.build().

    Parameters
    ----------
    keras_model : tf.keras.Model
        Trained model (e.g. QKeras from load_best_model_at_precision).
    output_dir : str or Path
        Directory where the HLS project will be written.
    hls_config : dict or None
        Optional overrides. May include:
        - board / Part
        - model.precision (e.g. "ap_fixed<8,3>")
        - model.reuse_factor
        - model.strategy ("Latency" or "Resource")

    Returns
    -------
    hls_model : hls4ml.model.HLSModel
        The hls4ml model. Call hls_model.compile() to generate HLS, or
        hls_model.build() to run full synthesis (Vivado/Vitis).
    """
    try:
        import hls4ml
    except ImportError as e:
        raise ImportError(
            "hls4ml is required for synthesis. Install with: pip install hls4ml"
        ) from e

    # Default config from model; granularity 'name' gives per-layer control
    config = hls4ml.utils.config_from_keras_model(
        keras_model,
        granularity="name",
    )

    # Apply tutorial config overrides
    if hls_config:
        if "board" in hls_config:
            config["Part"] = hls_config["board"]
        if "model" in hls_config:
            m = hls_config["model"]
            if "precision" in m:
                config["Model"]["Precision"] = m["precision"]
            if "reuse_factor" in m:
                config["Model"]["ReuseFactor"] = m["reuse_factor"]
            if "strategy" in m:
                config["Model"]["Strategy"] = m["strategy"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=str(output_dir),
    )

    # Write HLS project to disk (no compile/synthesis yet)
    hls_model.write()
    return hls_model
