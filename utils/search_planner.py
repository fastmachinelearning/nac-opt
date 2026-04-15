"""
Planning utilities for turning dataset metadata and user constraints into
SNAC-Pack search configurations.

The planner keeps the LLM at the "reason about the design space" layer while
leaving execution to deterministic Python code.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List
import re

import yaml

BUILTIN_DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "name": "mnist",
        "display_name": "MNIST",
        "task_type": "classification",
        "modality": "image",
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "normalize": True,
        "flatten": False,
        "one_hot": True,
        "subset_size": 10000,
        "resize_val": 8,
    },
    "fashion_mnist": {
        "name": "fashion_mnist",
        "display_name": "Fashion-MNIST",
        "task_type": "classification",
        "modality": "image",
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "normalize": True,
        "flatten": False,
        "one_hot": True,
        "subset_size": 20000,
        "resize_val": 16,
    },
    "qubit": {
        "name": "qubit",
        "display_name": "Qubit Readout",
        "task_type": "classification",
        "modality": "signal",
        "input_shape": [800],
        "num_classes": 2,
        "normalize": True,
        "flatten": True,
        "one_hot": True,
        "subset_size": 1000,
        "data_dir": "../../data/qubit/data",
        "start_location": 100,
        "window_size": 400,
    },
}

DATASET_INFO_KEYS = {
    "display_name",
    "description",
    "sample_count",
    "notes",
    "constraints_hint",
}

OBJECTIVE_ALIASES = {
    "accuracy": "performance_metric",
    "performance": "performance_metric",
    "performance_metric": "performance_metric",
    "bops": "bops",
    "resource": "avg_resource",
    "resources": "avg_resource",
    "avg_resource": "avg_resource",
    "clock_cycles": "clock_cycles",
    "cycles": "clock_cycles",
    "latency": "clock_cycles",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_shape(value: Any) -> List[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"Unsupported input_shape value: {value!r}")


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _powers_and_multiples(max_value: int) -> List[int]:
    candidates = [8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 384, 512]
    return [candidate for candidate in candidates if candidate <= max_value]


def _normalize_modality(profile: Dict[str, Any]) -> str:
    modality = profile.get("modality")
    if modality:
        return str(modality).lower()

    shape = profile.get("input_shape") or []
    if len(shape) >= 2:
        return "image"
    if len(shape) == 1:
        return "vector"
    return "unknown"


def normalize_dataset_spec(dataset_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a user-provided dataset spec into a stable planning profile.
    """
    dataset_spec = deepcopy(dataset_spec or {})
    dataset_name = dataset_spec.get("name")

    profile: Dict[str, Any] = {}
    if dataset_name in BUILTIN_DATASET_PROFILES:
        profile = deepcopy(BUILTIN_DATASET_PROFILES[dataset_name])

    profile = _deep_merge(profile, dataset_spec)

    if "input_shape" in profile:
        profile["input_shape"] = _normalize_shape(profile["input_shape"])
    if profile.get("resize_val") and profile.get("modality") in {None, "image", "spatial"}:
        resize = int(profile["resize_val"])
        channels = 1
        if profile.get("input_shape") and len(profile["input_shape"]) >= 3:
            channels = int(profile["input_shape"][-1])
        profile["input_shape"] = [resize, resize, channels]
    elif profile.get("window_size") and profile.get("name") == "qubit":
        width = int(profile["window_size"]) * 2
        profile["input_shape"] = [width] if profile.get("flatten", True) else [width, 1]

    profile.setdefault("task_type", "classification")
    profile["task_type"] = str(profile["task_type"]).lower()
    profile["modality"] = _normalize_modality(profile)
    profile.setdefault("name", dataset_name or profile.get("display_name") or "custom_dataset")
    profile.setdefault("display_name", str(profile["name"]).replace("_", " ").title())
    profile.setdefault("num_classes", 2)
    profile.setdefault("subset_size", profile.get("sample_count"))
    profile.setdefault("normalize", True)
    profile.setdefault("flatten", profile["modality"] not in {"image", "spatial"})
    profile.setdefault("one_hot", profile["task_type"] == "classification")
    return profile


def _normalize_constraints(constraints: Dict[str, Any] | None) -> Dict[str, Any]:
    constraints = deepcopy(constraints or {})
    constraints.setdefault("search_style", "balanced")
    constraints.setdefault("open_ended", True)
    constraints.setdefault("local_search", {})
    constraints.setdefault("hardware", {})
    constraints.setdefault("search_space_overrides", {})
    constraints.setdefault("requested_model_families", [])
    if constraints.get("latency_budget") is not None:
        constraints.setdefault("prefer_low_latency", True)
        constraints.setdefault("use_hardware_metrics", True)
    return constraints


def _use_hardware_metrics(constraints: Dict[str, Any]) -> bool:
    if "use_hardware_metrics" in constraints:
        return bool(constraints["use_hardware_metrics"])

    requested = {
        OBJECTIVE_ALIASES.get(str(item).lower(), str(item).lower())
        for item in _as_list(constraints.get("optimize_for"))
    }
    if requested & {"avg_resource", "clock_cycles"}:
        return True
    hardware = constraints.get("hardware", {})
    return bool(hardware or constraints.get("latency_budget") or constraints.get("resource_budget"))


def _resolve_objectives(constraints: Dict[str, Any]) -> tuple[list[str], list[bool], list[str]]:
    use_hardware_metrics = _use_hardware_metrics(constraints)
    rationale: List[str] = []

    if use_hardware_metrics:
        rationale.append(
            "Using hardware-aware objectives so global search optimizes accuracy alongside FPGA-oriented proxy metrics."
        )
        return (
            ["performance_metric", "bops", "avg_resource", "clock_cycles"],
            [True, False, False, False],
            rationale,
        )

    rationale.append("Using the lighter accuracy-plus-BOPs objective set for faster exploratory search.")
    return (["performance_metric", "bops"], [True, False], rationale)


def _select_model_family(profile: Dict[str, Any], constraints: Dict[str, Any]) -> tuple[str, list[str]]:
    rationale: List[str] = []
    requested = constraints.get("model_family") or constraints.get("model_type")
    if requested in {"mlp", "block"}:
        rationale.append(f"Respecting the explicit model family request: {requested}.")
        return requested, rationale

    modality = profile["modality"]
    prefer_low_latency = bool(constraints.get("prefer_low_latency"))
    prefer_expressive_models = bool(constraints.get("prefer_expressive_models") or constraints.get("prefer_attention"))
    open_ended = bool(constraints.get("open_ended", True))

    if modality in {"image", "spatial"}:
        if prefer_low_latency and not prefer_expressive_models:
            rationale.append(
                "Choosing block search for image-like data, but the planner will bias toward cheaper Conv/MLP blocks because latency or cost matters."
            )
        else:
            rationale.append(
                "Choosing block search because image-like inputs benefit from compositional feature extractors and can support attention-like blocks."
            )
        return "block", rationale

    if modality in {"vector", "tabular", "signal"}:
        rationale.append(
            "Choosing plain MLP search by default for non-spatial data because the current execution stack does not provide true transformer-style blocks for these inputs."
        )
        return "mlp", rationale

    if open_ended:
        rationale.append(
            "Choosing plain MLP search because the dataset shape is not clearly spatial and dense architectures are the safest default."
        )
        return "mlp", rationale

    rationale.append("Choosing plain MLP search for a narrower, lower-cost search over dense architectures.")
    return "mlp", rationale


def _resolve_architecture_policy(
    profile: Dict[str, Any],
    constraints: Dict[str, Any],
    model_type: str,
    use_hardware_metrics: bool,
) -> tuple[Dict[str, Any], List[str], List[str]]:
    rationale: List[str] = []
    warnings: List[str] = []

    modality = profile["modality"]
    prefer_attention = bool(constraints.get("prefer_attention"))
    avoid_attention = bool(constraints.get("avoid_attention"))
    prefer_low_latency = bool(constraints.get("prefer_low_latency"))
    prefer_expressive_models = bool(constraints.get("prefer_expressive_models") or prefer_attention)
    requested_families = set(constraints.get("requested_model_families") or [])
    style = str(constraints.get("search_style", "balanced")).lower()

    policy = {
        "allow_conv": False,
        "allow_attention": False,
        "block_types": ["MLP"] if model_type == "mlp" else ["MLP", "None"],
        "attention_supported": False,
    }

    if model_type != "block":
        unsupported = requested_families - {"mlp"}
        if unsupported:
            warnings.append(
                f"Requested model families {sorted(unsupported)} are not executable in the current dense-only path. Falling back to MLP search."
            )
        rationale.append("Using MLP search, so the search space will stay dense-only.")
        return policy, rationale, warnings

    if modality in {"image", "spatial"}:
        policy["allow_conv"] = True
        policy["attention_supported"] = True
        unsupported = requested_families - {"conv", "attention", "transformer", "mlp", "block"}
        if unsupported:
            warnings.append(
                f"Requested model families {sorted(unsupported)} are not supported by the current image-style search stack. Falling back to the nearest supported block families."
            )

        if use_hardware_metrics:
            policy["block_types"] = ["Conv", "MLP", "None"]
            rationale.append(
                "Excluding ConvAttn blocks because the current rule4ml-backed hardware estimator cannot score them safely."
            )
            if prefer_attention:
                warnings.append(
                    "Transformer-like ConvAttn blocks were requested, but hardware-aware mode disables them. Falling back to Conv/MLP blocks."
                )
        elif avoid_attention or prefer_low_latency or style == "fast":
            policy["block_types"] = ["Conv", "MLP", "None"]
            rationale.append(
                "Biasing the search toward cheaper Conv/MLP blocks because the request emphasizes speed, cost, or low latency."
            )
        elif prefer_attention or prefer_expressive_models or style in {"exploratory", "aggressive"}:
            policy["allow_attention"] = True
            policy["block_types"] = ["Conv", "ConvAttn", "MLP", "None"]
            rationale.append(
                "Allowing transformer-like ConvAttn blocks because the request prioritizes expressiveness or accuracy over raw speed."
            )
        else:
            policy["allow_attention"] = True
            policy["block_types"] = ["Conv", "ConvAttn", "MLP", "None"]
            rationale.append("Allowing Conv, ConvAttn, and MLP blocks for a balanced image-style search.")

        return policy, rationale, warnings

    if prefer_attention:
        warnings.append(
            "Transformer-like blocks were requested, but the current execution stack only supports ConvAttn for image-like tensors. Falling back to dense MLP search."
        )
    unsupported = requested_families - {"mlp"}
    if unsupported:
        warnings.append(
            f"Requested model families {sorted(unsupported)} are not supported for this dataset shape in the current runtime. Falling back to MLP search."
        )
    rationale.append("Restricting the search space to dense-style architectures because the dataset profile is non-spatial.")
    return policy, rationale, warnings


def _estimate_search_breadth(
    profile: Dict[str, Any],
    constraints: Dict[str, Any],
    use_hardware_metrics: bool,
) -> Dict[str, Any]:
    shape = profile.get("input_shape") or [128]
    feature_count = _product(shape)
    sample_count = profile.get("sample_count") or profile.get("subset_size") or 10000
    style = str(constraints.get("search_style", "balanced")).lower()

    if style == "exploratory":
        default_trials = 48
        default_epochs = 8
    elif style == "aggressive":
        default_trials = 96
        default_epochs = 12
    elif style == "fast":
        default_trials = 8
        default_epochs = 3
    else:
        default_trials = 24
        default_epochs = 6

    if sample_count < 3000:
        default_trials = min(default_trials, 20)
        default_epochs = min(default_epochs, 6)
    elif sample_count > 50000:
        default_epochs = max(3, default_epochs - 2)

    if use_hardware_metrics:
        default_trials = max(default_trials, 20)

    max_width = constraints.get("max_width")
    if max_width is None:
        if feature_count <= 64:
            max_width = 128
        elif feature_count <= 1024:
            max_width = 256
        else:
            max_width = 512

    if use_hardware_metrics:
        max_width = min(max_width, 256)

    max_blocks = constraints.get("max_blocks")
    if max_blocks is None:
        max_blocks = 4 if profile["modality"] in {"image", "spatial"} else 5
        if style in {"exploratory", "aggressive"}:
            max_blocks += 1

    return {
        "n_trials": int(constraints.get("max_trials", default_trials)),
        "epochs": int(constraints.get("epochs", default_epochs)),
        "max_width": int(max_width),
        "max_blocks": int(max_blocks),
        "feature_count": int(feature_count),
        "sample_count": int(sample_count),
    }


def _build_search_space(
    profile: Dict[str, Any],
    constraints: Dict[str, Any],
    model_type: str,
    use_hardware_metrics: bool,
    breadth: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str], list[str]]:
    rationale: List[str] = []
    warnings: List[str] = []

    modality = profile["modality"]
    output_dim = int(profile.get("num_classes", 2))
    shape = profile.get("input_shape") or [128]
    architecture_policy, policy_rationale, policy_warnings = _resolve_architecture_policy(
        profile=profile,
        constraints=constraints,
        model_type=model_type,
        use_hardware_metrics=use_hardware_metrics,
    )
    rationale.extend(policy_rationale)
    warnings.extend(policy_warnings)
    width_space = _powers_and_multiples(breadth["max_width"])
    if not width_space:
        width_space = [8, 16, 32]

    search_space: Dict[str, Any] = {
        "mlp_width_space": width_space,
        "act_space": ["ReLU", "Tanh", "Sigmoid", "Identity"],
        "norm_space": [None, "batch", "layer"],
        "output_dim": output_dim,
        "output_activation": "softmax" if profile["task_type"] == "classification" else "linear",
        "mlp_num_layers_space": list(range(2, min(8, breadth["max_blocks"] + 2))),
    }

    if modality in {"image", "spatial"}:
        search_space["initial_img_size"] = int(shape[0])
        search_space["channel_space"] = _powers_and_multiples(min(max(breadth["max_width"] // 2, 16), 128))
        search_space["kernel_space"] = [1, 3, 5]
        search_space["num_blocks"] = breadth["max_blocks"]
        search_space["block_types"] = architecture_policy["block_types"]
        if architecture_policy["allow_attention"]:
            search_space["conv_attn"] = {"hidden_channel_space": [4, 8, 16, 32]}
            search_space["act_space"] = ["ReLU", "LeakyReLU", "GELU", "Tanh", "Identity"]
        else:
            search_space["conv_attn"] = {"hidden_channel_space": [4, 8, 16]}
    else:
        search_space["initial_img_size"] = int(shape[0]) if shape else 1
        search_space["channel_space"] = [8, 16]
        search_space["kernel_space"] = [1]
        search_space["conv_attn"] = {"hidden_channel_space": [4, 8]}
        search_space["num_blocks"] = breadth["max_blocks"]
        search_space["block_types"] = architecture_policy["block_types"]

    if profile["task_type"] != "classification":
        warnings.append(
            "The current execution stack is classification-oriented. Regression-like tasks can be planned, but may need custom training/evaluation code before execution."
        )

    search_space = _deep_merge(search_space, constraints.get("search_space_overrides", {}))
    return search_space, rationale, warnings


def _build_hls_config(constraints: Dict[str, Any]) -> Dict[str, Any]:
    hardware = constraints.get("hardware", {})
    return {
        "board": hardware.get("board", constraints.get("board", "zcu102")),
        "model": {
            "precision": hardware.get("precision", constraints.get("precision", "ap_fixed<8,3>")),
            "reuse_factor": int(hardware.get("reuse_factor", constraints.get("reuse_factor", 1))),
            "strategy": hardware.get("strategy", constraints.get("strategy", "Latency")),
        },
    }


def _build_local_search_config(
    constraints: Dict[str, Any],
    use_hardware_metrics: bool,
) -> tuple[Dict[str, Any], list[str]]:
    local_constraints = constraints.get("local_search", {})
    budget = str(local_constraints.get("budget", constraints.get("local_search_budget", "balanced"))).lower()
    rationale: List[str] = []

    if budget == "light":
        cfg = {
            "qat_epochs": 3,
            "pruning_iterations": 4,
            "pruning_epochs": 3,
            "pruning_rate": 0.6,
            "precision_pairs": [
                {"total_bits": 8, "int_bits": 3},
                {"total_bits": 6, "int_bits": 2},
                {"total_bits": 4, "int_bits": 1},
            ],
        }
    elif budget == "heavy":
        cfg = {
            "qat_epochs": 12,
            "pruning_iterations": 10,
            "pruning_epochs": 10,
            "pruning_rate": 0.7,
            "precision_pairs": [
                {"total_bits": 16, "int_bits": 6},
                {"total_bits": 12, "int_bits": 4},
                {"total_bits": 8, "int_bits": 3},
                {"total_bits": 6, "int_bits": 2},
                {"total_bits": 4, "int_bits": 1},
            ],
        }
    else:
        cfg = {
            "qat_epochs": 6,
            "pruning_iterations": 6,
            "pruning_epochs": 5,
            "pruning_rate": 0.65,
            "precision_pairs": [
                {"total_bits": 16, "int_bits": 6},
                {"total_bits": 8, "int_bits": 3},
                {"total_bits": 6, "int_bits": 2},
                {"total_bits": 4, "int_bits": 1},
            ],
        }

    if use_hardware_metrics:
        rationale.append("Keeping local search enabled so quantization and pruning can refine hardware-aware global-search winners.")
    cfg = _deep_merge(cfg, local_constraints)
    return cfg, rationale


def _default_results_dir(profile: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    explicit = constraints.get("results_dir")
    if explicit:
        return explicit
    name = _slugify(str(profile.get("name") or profile.get("display_name") or "dataset"))
    return f"./results/planned_{name}"


def recommend_search_plan(
    dataset_spec: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Produce a plan and config for a dataset/constraint pair.
    """
    profile = normalize_dataset_spec(dataset_spec)
    constraints = _normalize_constraints(constraints)

    objectives, maximize_flags, objective_rationale = _resolve_objectives(constraints)
    use_hardware_metrics = len(objectives) > 2
    model_type, model_rationale = _select_model_family(profile, constraints)
    breadth = _estimate_search_breadth(profile, constraints, use_hardware_metrics)
    search_space, search_rationale, warnings = _build_search_space(
        profile=profile,
        constraints=constraints,
        model_type=model_type,
        use_hardware_metrics=use_hardware_metrics,
        breadth=breadth,
    )
    hls_config = _build_hls_config(constraints)
    local_search, local_rationale = _build_local_search_config(constraints, use_hardware_metrics)

    dataset_cfg = deepcopy(profile)
    results_dir = _default_results_dir(profile, constraints)

    config = {
        "dataset": dataset_cfg,
        "search": {
            "model_type": model_type,
            "n_trials": breadth["n_trials"],
            "epochs": breadth["epochs"],
            "n_folds": int(constraints.get("n_folds", 1)),
            "use_hardware_metrics": use_hardware_metrics,
            "objective_names": objectives,
            "maximize_flags": maximize_flags,
            "one_hot": bool(profile.get("one_hot", True)),
            "selection": {
                "latency_budget": constraints.get("latency_budget"),
                "resource_budget": constraints.get("resource_budget"),
            },
        },
        "search_space": search_space,
        "hls_config": hls_config,
        "local_search": local_search,
        "output": {
            "results_dir": results_dir,
        },
        "planner_metadata": {
            "source": "search_planner",
            "dataset_profile": {
                "name": profile["name"],
                "display_name": profile["display_name"],
                "modality": profile["modality"],
                "task_type": profile["task_type"],
                "input_shape": profile.get("input_shape"),
                "num_classes": profile.get("num_classes"),
                "sample_count": profile.get("sample_count") or profile.get("subset_size"),
            },
            "constraints": deepcopy(constraints),
        },
    }

    if profile["task_type"] == "classification" and int(profile.get("num_classes", 2)) <= 1:
        warnings.append("Classification runs expect num_classes >= 2. Falling back to output_dim=2 may not match the dataset labels.")
    if not profile.get("input_shape"):
        warnings.append(
            "No input_shape was provided. The planner can still generate a dense-biased config, but image-like search-space choices will be conservative."
        )

    rationale = []
    rationale.extend(model_rationale)
    rationale.extend(objective_rationale)
    rationale.extend(search_rationale)
    rationale.extend(local_rationale)

    return {
        "dataset_profile": profile,
        "constraints": constraints,
        "config": config,
        "rationale": rationale,
        "warnings": warnings,
    }


def build_search_config(
    dataset_spec: Dict[str, Any],
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Return only the config payload from a full recommendation plan.
    """
    return recommend_search_plan(dataset_spec, constraints)["config"]


def default_generated_config_path(config: Dict[str, Any]) -> Path:
    results_dir = Path(config["output"]["results_dir"])
    return results_dir / "generated_search_config.yaml"


def write_search_config(
    config: Dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Persist a generated config to YAML without importing the heavier execution stack.
    """
    config_copy = deepcopy(config)
    target_path = Path(output_path) if output_path else default_generated_config_path(config_copy)
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_copy, handle, sort_keys=False)
    return target_path.resolve()


__all__ = [
    "BUILTIN_DATASET_PROFILES",
    "build_search_config",
    "default_generated_config_path",
    "normalize_dataset_spec",
    "recommend_search_plan",
    "write_search_config",
]
