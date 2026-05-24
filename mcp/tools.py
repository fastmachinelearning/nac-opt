import contextlib
import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable, List

import yaml
from typing_extensions import TypedDict


def _stdout_safe(fn: Callable) -> Callable:
    """
    Redirect ``sys.stdout`` to ``sys.stderr`` for the duration of the wrapped
    call. Required for every MCP-exposed tool: under stdio transport, fd 1 is
    the JSON-RPC channel, so any ``print`` from search/TF code would corrupt
    the protocol and the client would hang waiting for a parseable response.

    The transport captured its own reference to stdout at server startup, so
    swapping ``sys.stdout`` here only affects user code.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    return wrapper


class PrecisionPair(TypedDict):
    total_bits: int
    int_bits: int


class LocalSearchConfig(TypedDict):
    qat_epochs: int
    pruning_iterations: int
    pruning_epochs: int
    pruning_rate: float
    precision_pairs: List[PrecisionPair]


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)


def _resolve_repo_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes the repository root: {relative_path}") from exc
    return path


def echo(text: str) -> str:
    """Echo text back to the caller."""
    return text


def read_repo_file(relative_path: str) -> str:
    """Read a UTF-8 text file from this repo by relative path."""
    path = _resolve_repo_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {relative_path}")
    return path.read_text(encoding="utf-8")


def run_search_pipeline(config_relative_path: str, run_local_search: bool = True) -> str:
    """Run the repo's existing global search plus optional local search pipeline."""
    from utils.search_pipeline import run_pipeline_from_config

    config_path = _resolve_repo_path(config_relative_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"No such config file: {config_relative_path}")

    summary = run_pipeline_from_config(str(config_path), run_local_search=run_local_search)
    safe_summary = {
        "results_dir": summary.get("results_dir"),
        "architecture_yaml": summary.get("architecture_yaml"),
        "local_results_dir": summary.get("local_results_dir"),
    }
    return yaml_dump(safe_summary)


def recommend_search_plan(dataset_spec: dict, constraints: dict | None = None) -> str:
    """
    Recommend a generated SNAC-Pack config from dataset metadata and constraints.
    """
    from utils.search_planner import recommend_search_plan as _recommend_search_plan

    plan = _recommend_search_plan(dataset_spec=dataset_spec, constraints=constraints or {})
    return yaml_dump(plan)


def create_search_config(
    dataset_spec: dict,
    constraints: dict | None = None,
    output_relative_path: str | None = None,
) -> str:
    """
    Generate a config from dataset metadata and constraints, and optionally save it.
    """
    from utils.search_planner import build_search_config, write_search_config

    config = build_search_config(dataset_spec=dataset_spec, constraints=constraints or {})

    response = {"config": config}
    if output_relative_path:
        output_path = _resolve_repo_path(output_relative_path)
        written_path = write_search_config(config, output_path=output_path)
        response["config_relative_path"] = str(written_path.relative_to(REPO_ROOT))
    return yaml_dump(response)


def run_search_pipeline_from_spec(
    dataset_spec: dict,
    constraints: dict | None = None,
    run_local_search: bool = True,
    config_relative_path: str | None = None,
) -> str:
    """
    Plan, materialize, and run a search directly from dataset metadata + constraints.
    """
    from utils.search_pipeline import run_pipeline_from_spec

    output_path = _resolve_repo_path(config_relative_path) if config_relative_path else None
    summary = run_pipeline_from_spec(
        dataset_spec=dataset_spec,
        constraints=constraints or {},
        run_local_search=run_local_search,
        config_output_path=output_path,
    )
    safe_summary = {
        "generated_config": summary.get("generated_config"),
        "results_dir": summary.get("results_dir"),
        "architecture_yaml": summary.get("architecture_yaml"),
        "local_results_dir": summary.get("local_results_dir"),
    }
    return yaml_dump(safe_summary)


def inspect_dataset(dataset_path: str) -> str:
    """
    Inspect a local dataset path and infer a dataset spec for planning.
    """
    from utils.dataset_inspector import inspect_dataset_path

    path = _resolve_repo_path(dataset_path)
    result = inspect_dataset_path(path)
    if "dataset_path" in result:
        result["dataset_path"] = str(Path(result["dataset_path"]).resolve().relative_to(REPO_ROOT))
    if "resolved_path" in result:
        result["resolved_path"] = str(Path(result["resolved_path"]).resolve().relative_to(REPO_ROOT))
    loader_kwargs = result.get("loader_kwargs")
    if isinstance(loader_kwargs, dict):
        normalized = {}
        for key, value in loader_kwargs.items():
            if isinstance(value, str):
                maybe_path = Path(value)
                try:
                    normalized[key] = str(maybe_path.resolve().relative_to(REPO_ROOT))
                except Exception:
                    normalized[key] = value
            else:
                normalized[key] = value
        result["loader_kwargs"] = normalized
    return yaml_dump(result)


def list_available_datasets() -> str:
    """
    List built-in datasets the agent can reason about via tool use.
    """
    from utils.dataset_catalog import list_available_datasets as _list_available_datasets

    return yaml_dump(_list_available_datasets())


def list_available_boards() -> str:
    """
    List the FPGA boards supported by rule4ml for hardware-aware search.

    Returns each board's canonical name (the value to pass in hls_config.board or
    as a hardware constraint), its Xilinx part string, and its maximum resource counts.
    Only these boards produce valid hardware estimates; any other board name will raise
    an error in the estimator.
    """
    import json
    from pathlib import Path as _Path

    try:
        import rule4ml.parsers as _parsers

        boards_path = _Path(_parsers.__file__).parent / "supported_boards.json"
        boards_data = json.loads(boards_path.read_text())
    except Exception:
        # Fallback if rule4ml is not installed in this environment.
        boards_data = {
            "pynq-z2": {
                "part": "xc7z020clg400-1",
                "max_bram": 280,
                "max_dsp": 220,
                "max_ff": 106400,
                "max_lut": 53200,
                "max_uram": 0,
            },
            "zcu102": {
                "part": "xczu9eg-ffvb1156-2-e",
                "max_bram": 1824,
                "max_dsp": 2520,
                "max_ff": 548160,
                "max_lut": 274080,
                "max_uram": 0,
            },
            "alveo-u200": {
                "part": "xcu200-fsgd2104-2-e",
                "max_bram": 4320,
                "max_dsp": 6840,
                "max_ff": 2364480,
                "max_lut": 1182240,
                "max_uram": 960,
            },
            "alveo-u250": {
                "part": "xcu250-figd2104-2L-e",
                "max_bram": 5376,
                "max_dsp": 12288,
                "max_ff": 3456000,
                "max_lut": 1728000,
                "max_uram": 1280,
            },
        }

    from utils.request_inference import BOARD_ALIASES

    # Build a reverse map: canonical name → list of recognized aliases
    alias_map: dict = {}
    for alias, canonical in BOARD_ALIASES.items():
        alias_map.setdefault(canonical, []).append(alias)

    result = {
        "default_board": "zcu102",
        "supported_boards": {
            name: {
                **specs,
                "recognized_aliases": sorted(alias_map.get(name, [])),
            }
            for name, specs in boards_data.items()
        },
    }
    return yaml_dump(result)


def describe_dataset(dataset_name: str) -> str:
    """
    Describe one built-in dataset in a structured way.
    """
    from utils.dataset_catalog import describe_dataset as _describe_dataset

    return yaml_dump(_describe_dataset(dataset_name))


def run_local_search(
    architecture_relative_path: str,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    dataset_spec: dict | None = None,
    local_search_config: LocalSearchConfig | None = None,
    budget: str | None = None,
    mode: str = "auto",
    results_relative_path: str | None = None,
    n_folds: int = 1,
) -> str:
    """
    Run local search (QAT + iterative pruning) on an existing best-architecture YAML.

    Provide exactly one of ``budget`` or ``local_search_config``:
      - ``budget``: one of "light", "balanced", "heavy" (uses planner defaults).
      - ``local_search_config``: a flat dict with ALL of these required keys
        (no aliases are accepted):

            {
              "qat_epochs": int,                # QAT warmup epochs
              "pruning_iterations": int,        # number of iterative pruning steps
              "pruning_epochs": int,            # fine-tune epochs per pruning step
              "pruning_rate": float,            # in (0, 1]; sparsity schedule base
              "precision_pairs": [              # list of QKeras quant pairs
                {"total_bits": int, "int_bits": int},
                ...
              ]
            }

        Do NOT pass alternate names like "epochs", "precision_options",
        or "pruning_targets" - they will cause a KeyError. If you want
        looser control, omit local_search_config and use budget instead.

    Dataset is supplied by exactly one of dataset_spec, dataset_path, or
    dataset_name. ``mode`` is "auto" (detect from architecture YAML),
    "separated" (MLP: pruning then QAT), or "combined" (Conv/ConvAttn:
    simultaneous QAT+pruning with optional k-fold CV).
    """
    from utils.search_pipeline import run_local_search as _run_local_search

    arch_path = _resolve_repo_path(architecture_relative_path)
    if not arch_path.is_file():
        raise FileNotFoundError(f"Architecture YAML not found: {architecture_relative_path}")

    resolved_dataset_path = str(_resolve_repo_path(dataset_path)) if dataset_path else None
    results_dir = _resolve_repo_path(results_relative_path) if results_relative_path else None

    summary = _run_local_search(
        architecture_yaml_path=arch_path,
        dataset_spec=dataset_spec,
        dataset_name=dataset_name,
        dataset_path=resolved_dataset_path,
        local_search_config=local_search_config,
        budget=budget,
        mode=mode,
        results_dir=results_dir,
        n_folds=n_folds,
    )

    def _to_relative(value):
        if isinstance(value, str):
            try:
                return str(Path(value).resolve().relative_to(REPO_ROOT))
            except (ValueError, OSError):
                return value
        if isinstance(value, dict):
            return {k: _to_relative(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_relative(v) for v in value]
        return value

    safe_summary = {
        "architecture_yaml": _to_relative(summary.get("architecture_yaml")),
        "results_dir": _to_relative(summary.get("results_dir")),
        "local_search_config_path": _to_relative(summary.get("local_search_config_path")),
        "mode": summary.get("mode"),
        "budget": summary.get("budget"),
        "flat_local_search_config": summary.get("flat_local_search_config"),
        "pruning_rows": summary.get("pruning_rows"),
        "qat_rows": summary.get("qat_rows"),
        "combined_rows": summary.get("combined_rows"),
    }
    return yaml_dump(safe_summary)


def read_search_results(results_relative_path: str, top_n: int = 5) -> str:
    """
    Parse a SNAC-Pack results directory and return a structured summary
    covering global search trials, the selected best architecture, and any
    local-search QAT/pruning logs.
    """
    from utils.results_reader import read_search_results as _read_search_results

    results_path = _resolve_repo_path(results_relative_path)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_relative_path}")

    summary = _read_search_results(results_path, top_n=top_n)

    def _normalize(value):
        if isinstance(value, str):
            try:
                relative = Path(value).resolve().relative_to(REPO_ROOT)
            except (ValueError, OSError):
                return value
            return str(relative)
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize(v) for v in value]
        return value

    summary = _normalize(summary)
    return yaml_dump(summary)


def run_agentic_search(
    request_text: str,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    run_local_search: bool | None = None,
    config_relative_path: str | None = None,
    constraints: dict | None = None,
) -> str:
    """
    High-level tool that lets an LLM work from plain English plus either a dataset path or a built-in dataset name.
    """
    from utils.search_pipeline import run_agentic_search as _run_agentic_search

    path = _resolve_repo_path(dataset_path) if dataset_path else None
    output_path = _resolve_repo_path(config_relative_path) if config_relative_path else None
    summary = _run_agentic_search(
        request_text=request_text,
        dataset_path=str(path) if path else None,
        dataset_name=dataset_name,
        constraints=constraints or {},
        run_local_search=run_local_search,
        config_output_path=output_path,
    )
    safe_summary = {
        "generated_config": summary.get("generated_config"),
        "results_dir": summary.get("results_dir"),
        "architecture_yaml": summary.get("architecture_yaml"),
        "local_results_dir": summary.get("local_results_dir"),
        "inspected_dataset": summary.get("inspected_dataset"),
        "inferred_constraints": summary.get("inferred_constraints"),
    }
    return yaml_dump(safe_summary)


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text back to the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_repo_file",
            "description": "Read a UTF-8 text file from this repository using a path relative to the repo root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {"type": "string"},
                },
                "required": ["relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_search_pipeline",
            "description": (
                "Run the repository's global search pipeline from a YAML config, "
                "optionally followed by local search, and return a YAML summary of result paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "config_relative_path": {"type": "string"},
                    "run_local_search": {"type": "boolean", "default": True},
                },
                "required": ["config_relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_search_plan",
            "description": (
                "Recommend a SNAC-Pack search plan from dataset metadata and user constraints. "
                "Returns a generated config plus planner rationale and warnings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_spec": {
                        "type": "object",
                        "description": "Dataset metadata such as name, modality, input_shape, num_classes, loader_path, or loader kwargs.",
                        "additionalProperties": True,
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Hardware, latency, search-budget, or architecture-space constraints.",
                        "additionalProperties": True,
                    },
                },
                "required": ["dataset_spec"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_search_config",
            "description": (
                "Generate a config from dataset metadata and constraints, and optionally save it inside the repository."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_spec": {
                        "type": "object",
                        "description": "Dataset metadata such as name, modality, input_shape, num_classes, loader_path, or loader kwargs.",
                        "additionalProperties": True,
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Hardware, latency, search-budget, or architecture-space constraints.",
                        "additionalProperties": True,
                    },
                    "output_relative_path": {
                        "type": "string",
                        "description": "Optional repo-relative path where the generated YAML config should be written.",
                    },
                },
                "required": ["dataset_spec"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_search_pipeline_from_spec",
            "description": (
                "Plan, materialize, and run a search directly from dataset metadata and constraints without starting from a tutorial config."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_spec": {
                        "type": "object",
                        "description": "Dataset metadata such as name, modality, input_shape, num_classes, loader_path, or loader kwargs.",
                        "additionalProperties": True,
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Hardware, latency, search-budget, or architecture-space constraints.",
                        "additionalProperties": True,
                    },
                    "run_local_search": {"type": "boolean", "default": True},
                    "config_relative_path": {
                        "type": "string",
                        "description": "Optional repo-relative path where the generated config should be written before execution.",
                    },
                },
                "required": ["dataset_spec"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_datasets",
            "description": "List the built-in datasets this repo knows about, including whether each one is ready to use locally.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_boards",
            "description": (
                "List the FPGA boards supported by rule4ml for hardware-aware NAS. "
                "Returns each board's canonical name (use this in constraints or hls_config), "
                "its Xilinx part string, max resource counts (LUT/FF/DSP/BRAM/URAM), "
                "and all recognized human-language aliases (e.g. 'pynq', 'z2', 'alveo'). "
                "Call this before setting a board constraint to get the exact canonical name."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": "Return structured details for one built-in dataset, including shape, classes, loader, and local availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Built-in dataset name such as mnist, fashion_mnist, or qubit.",
                    },
                },
                "required": ["dataset_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": (
                "Inspect a local dataset file or directory and infer a dataset spec the planner can use."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Repo-relative path to a dataset file or directory.",
                    },
                },
                "required": ["dataset_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_local_search",
            "description": (
                "Run local search (QAT + iterative pruning) on an existing best-architecture YAML, "
                "without re-running global search. Useful for trying heavier local-search budgets, "
                "different precision pairs, or forcing the combined path. The mode is auto-detected "
                "from the architecture (MLP-only -> separated; Conv/ConvAttn -> combined) unless overridden."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "architecture_relative_path": {
                        "type": "string",
                        "description": "Repo-relative path to a best_model_for_local_search.yaml (or trial_*_arch.yaml).",
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Built-in dataset name such as mnist, fashion_mnist, or qubit.",
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "Repo-relative path to a dataset file or directory (inspected via dataset_inspector).",
                    },
                    "dataset_spec": {
                        "type": "object",
                        "description": "Pre-built dataset spec (inspector output) with loader_kwargs.",
                        "additionalProperties": True,
                    },
                    "local_search_config": {
                        "type": "object",
                        "description": (
                            "Flat planner-style local_search config. All five keys are REQUIRED "
                            "with exactly these names - no aliases (e.g. do NOT use 'epochs', "
                            "'precision_options', or 'pruning_targets'). Mutually exclusive with budget; "
                            "if you want looser control, omit this and pass budget instead."
                        ),
                        "properties": {
                            "qat_epochs": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "QAT warmup epochs.",
                            },
                            "pruning_iterations": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Number of iterative magnitude-pruning steps.",
                            },
                            "pruning_epochs": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Fine-tuning epochs per pruning step.",
                            },
                            "pruning_rate": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 1,
                                "description": "Base of the sparsity schedule, in (0, 1].",
                            },
                            "precision_pairs": {
                                "type": "array",
                                "minItems": 1,
                                "description": "List of QKeras quantization pairs to sweep.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "total_bits": {"type": "integer", "minimum": 1},
                                        "int_bits": {"type": "integer", "minimum": 0},
                                    },
                                    "required": ["total_bits", "int_bits"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "qat_epochs",
                            "pruning_iterations",
                            "pruning_epochs",
                            "pruning_rate",
                            "precision_pairs",
                        ],
                        "additionalProperties": False,
                    },
                    "budget": {
                        "type": "string",
                        "description": "Shortcut for local-search intensity: 'light', 'balanced', or 'heavy'. Ignored when local_search_config is provided.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "'auto' (default), 'separated' (MLP-style: pruning then QAT), or 'combined' (block-style: simultaneous QAT+pruning).",
                        "default": "auto",
                    },
                    "results_relative_path": {
                        "type": "string",
                        "description": "Optional repo-relative path for the local-search outputs. Defaults to a sibling directory next to the architecture YAML.",
                    },
                    "n_folds": {
                        "type": "integer",
                        "description": "K-folds for combined mode (ignored in separated mode).",
                        "default": 1,
                    },
                },
                "required": ["architecture_relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_search_results",
            "description": (
                "Read a SNAC-Pack results directory and return a structured summary: "
                "global-search trial table, top-N trials by performance, best architecture YAML, "
                "and local-search QAT/pruning logs (separated or combined). Use this instead of "
                "writing ad-hoc CSV-reading scripts after a search run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "results_relative_path": {
                        "type": "string",
                        "description": "Repo-relative path to the results directory (e.g. results/planned_iris).",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top-performing trials to surface in top_by_performance.",
                        "default": 5,
                    },
                },
                "required": ["results_relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agentic_search",
            "description": (
                "Run the end-to-end LLM-facing workflow from plain-English request text plus either a local dataset path or a built-in dataset name: inspect or describe the dataset, infer constraints, generate config, and execute the search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request_text": {
                        "type": "string",
                        "description": "Plain-English description of the modeling and hardware goals.",
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "Repo-relative path to a dataset file or directory.",
                    },
                    "dataset_name": {
                        "type": "string",
                        "description": "Built-in dataset name such as mnist, fashion_mnist, or qubit.",
                    },
                    "run_local_search": {
                        "type": "boolean",
                        "description": "Optional override for whether local search should run.",
                    },
                    "config_relative_path": {
                        "type": "string",
                        "description": "Optional repo-relative path where the generated YAML config should be written.",
                    },
                    "constraints": {
                        "type": "object",
                        "description": "Optional explicit planner constraint overrides merged on top of inferred constraints.",
                        "additionalProperties": True,
                    },
                },
                "required": ["request_text"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_REGISTRY = {
    "echo": echo,
    "read_repo_file": read_repo_file,
    "list_available_datasets": list_available_datasets,
    "list_available_boards": list_available_boards,
    "describe_dataset": describe_dataset,
    "inspect_dataset": inspect_dataset,
    "recommend_search_plan": recommend_search_plan,
    "create_search_config": create_search_config,
    "read_search_results": read_search_results,
    "run_local_search": run_local_search,
    "run_agentic_search": run_agentic_search,
    "run_search_pipeline": run_search_pipeline,
    "run_search_pipeline_from_spec": run_search_pipeline_from_spec,
}


def call_openai_tool(name: str, arguments_json: str) -> str:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    arguments = json.loads(arguments_json or "{}")
    result = TOOL_REGISTRY[name](**arguments)
    if isinstance(result, str):
        return result
    return yaml_dump(result)
