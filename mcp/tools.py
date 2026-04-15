import json
import sys
from pathlib import Path
from typing import Any

import yaml


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


def describe_dataset(dataset_name: str) -> str:
    """
    Describe one built-in dataset in a structured way.
    """
    from utils.dataset_catalog import describe_dataset as _describe_dataset

    return yaml_dump(_describe_dataset(dataset_name))


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
    "describe_dataset": describe_dataset,
    "inspect_dataset": inspect_dataset,
    "recommend_search_plan": recommend_search_plan,
    "create_search_config": create_search_config,
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
