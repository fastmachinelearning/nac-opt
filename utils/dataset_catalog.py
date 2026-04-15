"""
Structured dataset discovery helpers for built-in SNAC-Pack datasets.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from utils.search_planner import BUILTIN_DATASET_PROFILES, normalize_dataset_spec
from utils.tf_data_preprocessing import _load_dataset_loader


REPO_ROOT = Path(__file__).resolve().parents[1]

QUBIT_REQUIRED_FILES = [
    "0528_X_train_0_770.npy",
    "0528_y_train_0_770.npy",
    "0528_X_test_0_770.npy",
    "0528_y_test_0_770.npy",
]


BUILTIN_DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "description": "Handwritten digit classification from TensorFlow/Keras.",
        "availability_mode": "download_on_first_use",
        "notes": [
            "Downloaded automatically by Keras the first time it is loaded.",
            "Works well for quick smoke tests and small MLP or block-search runs.",
        ],
        "default_prompt_hint": "Use the built-in mnist dataset and find me a quick low-cost classifier.",
    },
    "fashion_mnist": {
        "description": "Fashion article classification from TensorFlow/Keras.",
        "availability_mode": "download_on_first_use",
        "notes": [
            "Downloaded automatically by Keras the first time it is loaded.",
            "A slightly harder image benchmark than MNIST, useful for block-style searches.",
        ],
        "default_prompt_hint": "Use the built-in fashion_mnist dataset and find me a quick low-cost classifier.",
    },
    "qubit": {
        "description": "Superconducting qubit IQ readout classification dataset stored as split .npy arrays.",
        "availability_mode": "manual_download_required",
        "expected_data_dir": "data/qubit/data",
        "required_files": deepcopy(QUBIT_REQUIRED_FILES),
        "source_readme": "data/qubit/README.md",
        "source_links": [
            "https://www.dropbox.com/scl/fo/i30pf90fpingvc2o87yrf/h?rlkey=8wfkli0nin11bnnc5ynf457g1&dl=0",
            "https://purdue0-my.sharepoint.com/personal/oyesilyu_purdue_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Foyesilyu%5Fpurdue%5Fedu%2FDocuments%2FQubit%20Readout%20%2D%20Purdue%20%2D%20New%20Data&ga=1",
        ],
        "notes": [
            "The repo loader expects four split .npy files in data/qubit/data.",
            "The default planning profile uses a 400-sample window, flattened to width 800.",
        ],
        "default_prompt_hint": "Use the built-in qubit dataset and find me a small classifier.",
    },
}


def _normalize_repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _dataset_availability(dataset_name: str) -> Dict[str, Any]:
    if dataset_name != "qubit":
        return {
            "is_ready": True,
            "status": "available_via_loader",
        }

    data_dir = REPO_ROOT / "data" / "qubit" / "data"
    present_files = sorted(path.name for path in data_dir.glob("*.npy"))
    missing_files = [name for name in QUBIT_REQUIRED_FILES if not (data_dir / name).exists()]
    return {
        "is_ready": not missing_files,
        "status": "ready" if not missing_files else "missing_files",
        "data_dir": _normalize_repo_relative(data_dir),
        "present_files": present_files,
        "missing_files": missing_files,
    }


def list_available_datasets() -> Dict[str, Any]:
    """
    Return a compact catalog of built-in datasets the agent can reason about.
    """
    datasets: List[Dict[str, Any]] = []
    for name in sorted(BUILTIN_DATASET_PROFILES):
        profile = normalize_dataset_spec({"name": name})
        metadata = BUILTIN_DATASET_METADATA.get(name, {})
        availability = _dataset_availability(name)
        loader = _load_dataset_loader(dataset_name=name)
        datasets.append(
            {
                "name": name,
                "display_name": profile.get("display_name"),
                "modality": profile.get("modality"),
                "task_type": profile.get("task_type"),
                "input_shape": profile.get("input_shape"),
                "num_classes": profile.get("num_classes"),
                "loader": f"{loader.__module__}:{loader.__name__}",
                "availability": availability,
                "description": metadata.get("description"),
            }
        )
    return {
        "datasets": datasets,
    }


def describe_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Return structured details for one built-in dataset.
    """
    if dataset_name not in BUILTIN_DATASET_PROFILES:
        available = ", ".join(sorted(BUILTIN_DATASET_PROFILES))
        raise ValueError(f"Unknown built-in dataset {dataset_name!r}. Available: {available}")

    profile = normalize_dataset_spec({"name": dataset_name})
    metadata = deepcopy(BUILTIN_DATASET_METADATA.get(dataset_name, {}))
    availability = _dataset_availability(dataset_name)
    loader = _load_dataset_loader(dataset_name=dataset_name)

    result = {
        "name": dataset_name,
        "profile": profile,
        "loader": {
            "path": f"{loader.__module__}:{loader.__name__}",
            "dataset_name": dataset_name,
        },
        "availability": availability,
    }
    result.update(metadata)
    return result


__all__ = [
    "describe_dataset",
    "list_available_datasets",
]
