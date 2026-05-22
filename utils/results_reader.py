"""
Read structured results from a SNAC-Pack run directory.

This is the data side of the `read_search_results` MCP tool. It parses the
global-search CSV, the best-architecture YAML, and either the separated
(MLP) or combined (block) local-search logs, returning a dict an agent can
reason about without writing ad-hoc Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


_GLOBAL_SEARCH_CSV_PATTERNS = (
    "mlp_search_results.csv",
    "block_search_results.csv",
)
_GLOBAL_SEARCH_RANK_CSV_GLOB = "*_search_results_rank*.csv"
_DEFAULT_TOP_N = 5
_SEPARATED_DIR_NAME = "local_search_separated"
_COMBINED_DIR_NAME = "local_search_combined"


def read_search_results(
    results_dir: str | Path,
    *,
    top_n: int = _DEFAULT_TOP_N,
) -> Dict[str, Any]:
    """
    Parse global-search + local-search artifacts in ``results_dir`` and return
    a structured summary.

    The returned dict has three top-level sections (any may be ``None`` if the
    corresponding artifact is missing):
        - ``global_search``: trial-level CSV summary
        - ``best_architecture``: parsed best-architecture YAML
        - ``local_search``: separated or combined local-search logs
    """
    root = Path(results_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    return {
        "results_dir": str(root),
        "global_search": _read_global_search(root, top_n=top_n),
        "best_architecture": _read_best_architecture(root),
        "local_search": _read_local_search(root),
    }


# ---------------------------------------------------------------------------
# Global search
# ---------------------------------------------------------------------------


def _read_global_search(root: Path, *, top_n: int) -> Optional[Dict[str, Any]]:
    csv_path = _locate_global_csv(root)
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    df = df.replace({float("nan"): None})
    objective_cols = [c for c in df.columns if c in {
        "performance_metric", "bops", "avg_resource", "clock_cycles"
    }]
    rows = df.to_dict(orient="records")
    for row in rows:
        if "params" in row and isinstance(row["params"], str):
            row["params"] = _try_parse_dict_like(row["params"])

    top_rows: List[Dict[str, Any]] = []
    if "performance_metric" in df.columns:
        ranked = df.sort_values("performance_metric", ascending=False).head(top_n)
        for row in ranked.to_dict(orient="records"):
            if "params" in row and isinstance(row["params"], str):
                row["params"] = _try_parse_dict_like(row["params"])
            top_rows.append(row)

    return {
        "csv_path": str(csv_path),
        "trial_count": int(len(df)),
        "objective_columns": objective_cols,
        "top_by_performance": top_rows,
        "all_trials": rows,
    }


def _locate_global_csv(root: Path) -> Optional[Path]:
    for name in _GLOBAL_SEARCH_CSV_PATTERNS:
        candidate = root / name
        if candidate.is_file():
            return candidate
    rank_csvs = sorted(root.glob(_GLOBAL_SEARCH_RANK_CSV_GLOB))
    if rank_csvs:
        return rank_csvs[0]
    return None


def _try_parse_dict_like(value: str) -> Any:
    """
    Trial params land in the CSV as a Python repr (single quotes). Try to
    coerce to a real dict for nicer downstream consumption, falling back to
    the raw string if it isn't parseable.
    """
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text.replace("'", '"').replace("None", "null"))
    except json.JSONDecodeError:
        return value


# ---------------------------------------------------------------------------
# Best architecture YAML
# ---------------------------------------------------------------------------


def _read_best_architecture(root: Path) -> Optional[Dict[str, Any]]:
    yaml_path = root / "best_model_for_local_search.yaml"
    if not yaml_path.is_file():
        return None
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {
        "yaml_path": str(yaml_path),
        "config": config,
    }


# ---------------------------------------------------------------------------
# Local search (separated MLP path or combined block path)
# ---------------------------------------------------------------------------


def _read_local_search(root: Path) -> Optional[Dict[str, Any]]:
    separated = root / _SEPARATED_DIR_NAME
    if separated.is_dir():
        return _read_separated_local_search(separated)
    combined = root / _COMBINED_DIR_NAME
    if combined.is_dir():
        return _read_combined_local_search(combined)
    return None


def _read_separated_local_search(dir_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "kind": "separated",
        "results_dir": str(dir_path),
        "qat": None,
        "pruning": None,
        "best_qat": None,
        "best_pruning": None,
    }

    qat_csv = dir_path / "qat_log.csv"
    if qat_csv.is_file():
        qat_df = _read_quoted_csv(qat_csv)
        qat_rows = qat_df.to_dict(orient="records")
        result["qat"] = qat_rows
        if qat_rows and "Accuracy" in qat_df.columns:
            best = qat_df.sort_values("Accuracy", ascending=False).iloc[0].to_dict()
            result["best_qat"] = best

    pruning_csv = dir_path / "pruning_log.csv"
    if pruning_csv.is_file():
        pruning_df = pd.read_csv(pruning_csv)
        pruning_rows = pruning_df.to_dict(orient="records")
        result["pruning"] = pruning_rows
        if pruning_rows and "Accuracy" in pruning_df.columns:
            best = pruning_df.sort_values("Accuracy", ascending=False).iloc[0].to_dict()
            result["best_pruning"] = best

    return result


def _read_combined_local_search(dir_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "kind": "combined",
        "results_dir": str(dir_path),
        "rows": None,
        "best_row": None,
    }
    csv_path = dir_path / "combined_qat_pruning_log.csv"
    if csv_path.is_file():
        df = _read_quoted_csv(csv_path)
        rows = df.to_dict(orient="records")
        result["rows"] = rows
        if rows and "Accuracy" in df.columns:
            best = df.sort_values("Accuracy", ascending=False).iloc[0].to_dict()
            result["best_row"] = best
    return result


def _read_quoted_csv(path: Path) -> pd.DataFrame:
    """
    Read a CSV that may be either properly double-quoted (post-fix) or
    legacy single-quoted (pre-fix). Picks the quotechar from the first
    data row.
    """
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline()
        first_row = handle.readline()
    quotechar = "'" if first_row.lstrip().startswith("'") else '"'
    df = pd.read_csv(path, quotechar=quotechar)
    return df


__all__ = ["read_search_results"]
