#!/usr/bin/env python3
"""
Create and compile an hls4ml project for **trial 448** / ``best_low_lut_min_acc.yaml``:
flat input **500** -> ``Dense(2)`` -> ``BatchNormalization`` -> ``Dense(1)`` (linear, no softmax).

Uses **plain Keras** layers only (no QKeras). Fixed-point **``ap_fixed<8,3>``** in the hls4ml config.
Synthesis is not run; only ``hls_model.compile()`` is called.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import hls4ml
import yaml
from tensorflow import keras
from tensorflow.keras import layers


def build_keras_model(input_shape: int) -> keras.Sequential:
    """
    Manual FP32 model matching ``best_low_lut_min_acc.yaml`` classifier_head
    (same layer names as the saved search yaml for trace-friendly hls4ml configs).
    """
    m = keras.Sequential(name="classifier_head")
    m.add(
        layers.Dense(
            2,
            activation=None,
            name="classifier_head_dense_0",
            input_shape=(input_shape,),
        )
    )
    m.add(layers.BatchNormalization(name="classifier_head_bn_0"))
    m.add(layers.Dense(1, activation=None, name="classifier_head_dense_1"))
    return m


def build_hls_config(keras_model: keras.Model) -> dict:
    """hls4ml fixed-point: ``ap_fixed<8,3>`` (8 total bits, 3 integer bits) for activations/weights style defaults."""
    ap = "ap_fixed<8,3>"
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")
    hls_config["Model"]["Precision"] = ap
    hls_config["Model"]["ReuseFactor"] = 1
    hls_config["Model"]["Strategy"] = "Resource"

    for layer_name, layer_config in hls_config["LayerName"].items():
        layer_config.setdefault("Precision", {})
        layer_config["Trace"] = not layer_name.endswith("_input")

    input_candidates = [k for k in hls_config["LayerName"] if k.endswith("_input")]
    for input_layer_name in input_candidates:
        hls_config["LayerName"].setdefault(input_layer_name, {})
        hls_config["LayerName"][input_layer_name]["Trace"] = False
        hls_config["LayerName"][input_layer_name]["Precision"] = ap

    for layer in keras_model.layers:
        if isinstance(layer, layers.Dense):
            ln = layer.name
            hls_config["LayerName"].setdefault(ln, {"Precision": {}})
            hls_config["LayerName"][ln].setdefault("Precision", {})
            hls_config["LayerName"][ln]["Precision"]["result"] = ap
            hls_config["LayerName"][ln]["accum_t"] = ap
        elif isinstance(layer, layers.BatchNormalization):
            ln = layer.name
            if ln in hls_config["LayerName"]:
                hls_config["LayerName"][ln].setdefault("Precision", {})
                # hls4ml may expect a dict for BN; string often propagates as default type
                p = hls_config["LayerName"][ln]["Precision"]
                if isinstance(p, dict):
                    p.setdefault("result", ap)
                else:
                    hls_config["LayerName"][ln]["Precision"] = ap

    return hls_config


def parse_args() -> argparse.Namespace:
    _here = Path(__file__).resolve().parent
    _default_yaml = _here / "results/qubit_optuna_job_52893384/best_low_lut_min_acc.yaml"

    parser = argparse.ArgumentParser(
        description="hls4ml compile for trial-448 classifier (500->2+BN->1), FP32 Keras."
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        default=500,
        help="Flattened feature size (500 for this trial yaml)",
    )
    parser.add_argument(
        "--arch-yaml",
        type=Path,
        default=_default_yaml,
        help="Optional: path to best_low_lut_min_acc.yaml for metadata / --print-metadata",
    )
    parser.add_argument(
        "--output-dir",
        default="hls4ml_snac_prj_low_lut/",
        help="hls4ml project output directory",
    )
    parser.add_argument("--project-name", default="NN")
    parser.add_argument("--part", default="xczu9eg-ffvb1156-2-e")
    parser.add_argument("--clock-period", type=float, default=3.225)
    parser.add_argument("--io-type", default="io_parallel")
    parser.add_argument("--backend", default="VivadoAccelerator")
    parser.add_argument("--board", default="zcu216")
    parser.add_argument("--interface", default="axi_stream")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write an hls4ml model diagram to <output-dir>/hls_model.png.",
    )
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print metadata JSON from --arch-yaml and exit (no compile).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    arch_path = args.arch_yaml.resolve()
    arch_data: dict = {}
    if arch_path.is_file():
        with open(arch_path, encoding="utf-8") as f:
            arch_data = yaml.safe_load(f) or {}
    elif args.print_metadata:
        raise SystemExit(f"Missing --arch-yaml file for --print-metadata: {arch_path}")

    if args.print_metadata:
        print(json.dumps(arch_data.get("metadata") or {}, indent=2))
        return

    if "XILINX_VIVADO" in os.environ:
        vivado_bin = os.path.join(os.environ["XILINX_VIVADO"], "bin")
        os.environ["PATH"] = vivado_bin + os.pathsep + os.environ["PATH"]

    keras.utils.set_random_seed(32)
    keras_model = build_keras_model(args.input_shape)
    keras_model.summary()

    if arch_data.get("metadata"):
        print("Loaded trial metadata from:", arch_path)
        print(json.dumps(arch_data["metadata"], indent=2))

    hls_config = build_hls_config(keras_model)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model=keras_model,
        hls_config=hls_config,
        output_dir=args.output_dir,
        part=args.part,
        io_type=args.io_type,
        clock_period=args.clock_period,
        backend=args.backend,
        board=args.board,
        interface=args.interface,
        project_name=args.project_name,
    )

    if args.plot:
        plot_path = os.path.join(args.output_dir, "hls_model.png")
        hls4ml.utils.plot_model(
            hls_model,
            show_shapes=True,
            show_precision=True,
            to_file=plot_path,
        )

    print(f"Compiling hls4ml project in {args.output_dir}")
    hls_model.compile()
    print("Compile complete. Synthesis was not run.")


if __name__ == "__main__":
    main()
