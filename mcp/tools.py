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
]

TOOL_REGISTRY = {
    "echo": echo,
    "read_repo_file": read_repo_file,
    "run_search_pipeline": run_search_pipeline,
}


def call_openai_tool(name: str, arguments_json: str) -> str:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")

    arguments = json.loads(arguments_json or "{}")
    result = TOOL_REGISTRY[name](**arguments)
    if isinstance(result, str):
        return result
    return yaml_dump(result)
