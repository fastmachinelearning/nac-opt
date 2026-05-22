"""
Infer a SNAC-Pack dataset_spec from a local file or directory.

The returned dict feeds into `utils.search_planner.normalize_dataset_spec` and
`utils.tf_data_preprocessing.load_generic_dataset` (via the `loader_kwargs`
sub-dict, dispatched through the fall-through generic loader).

Supported formats:
- CSV / TSV / TXT (delimited tabular)
- ARFF (Weka/OpenML attribute-relation file format)
- Parquet
- Single .npy (treated as features+optional-label-column)
- .npz with x/y arrays
- Directory of sibling .npy files (x*.npy + y*.npy)
- Image directory (ImageNet-style: one subdirectory per class)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TABULAR_EXTS = {".csv", ".tsv", ".txt"}
ARFF_EXTS = {".arff"}
PARQUET_EXTS = {".parquet", ".pq"}
NUMPY_SINGLE_EXTS = {".npy"}
NUMPY_BUNDLE_EXTS = {".npz"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif"}

_LABEL_CANDIDATES = ("label", "target", "class", "y", "labels", "targets", "classes")
_DEFAULT_VAL_SPLIT = 0.2
_DEFAULT_RANDOM_STATE = 42
_TABULAR_SAMPLE_ROWS = 2048
_FULL_READ_ROW_THRESHOLD = 1_000_000
_CLASSIFICATION_MAX_UNIQUE = 200
_IMAGE_DEFAULT_RESIZE = 32
_IMAGE_RESIZE_CAP = 64
_SIGNAL_MIN_LENGTH = 200


def inspect_dataset_path(path: str | Path) -> Dict[str, Any]:
    """
    Inspect a dataset file or directory and return a dataset_spec dict.

    The result is compatible with `normalize_dataset_spec` from
    `utils.search_planner` and carries a `loader_kwargs` payload that the
    generic loader in `utils.tf_data_preprocessing` consumes.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {resolved}")

    fmt = _detect_format(resolved)

    if fmt == "csv" or fmt == "tsv":
        spec = _inspect_tabular(resolved, fmt)
    elif fmt == "arff":
        spec = _inspect_arff(resolved)
    elif fmt == "parquet":
        spec = _inspect_parquet(resolved)
    elif fmt == "npy":
        spec = _inspect_npy(resolved)
    elif fmt == "npz":
        spec = _inspect_npz(resolved)
    elif fmt == "npy_pair":
        spec = _inspect_npy_pair(resolved)
    elif fmt == "image_dir":
        spec = _inspect_image_dir(resolved)
    else:
        raise ValueError(
            f"Unsupported dataset path: {resolved}. "
            "Supported: .csv/.tsv/.txt, .arff, .parquet, .npy, .npz, "
            "directory of paired x*/y*.npy, or image directory."
        )

    spec.setdefault("name", _slugify(resolved.stem if resolved.is_file() else resolved.name))
    spec.setdefault("display_name", spec["name"].replace("_", " ").title())
    spec.setdefault("task_type", "classification")
    spec["dataset_path"] = str(resolved)
    spec["resolved_path"] = str(resolved)
    spec.setdefault("notes", [])
    return spec


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    if path.is_file():
        ext = path.suffix.lower()
        if ext in TABULAR_EXTS:
            return "tsv" if ext == ".tsv" or _looks_tab_delimited(path) else "csv"
        if ext in ARFF_EXTS:
            return "arff"
        if ext in PARQUET_EXTS:
            return "parquet"
        if ext in NUMPY_SINGLE_EXTS:
            return "npy"
        if ext in NUMPY_BUNDLE_EXTS:
            return "npz"
        return "unknown"

    if path.is_dir():
        x_match, y_match = _find_npy_pair(path)
        if x_match and y_match:
            return "npy_pair"
        if _has_image_files(path):
            return "image_dir"
        return "unknown"

    return "unknown"


def _looks_tab_delimited(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        if not sample:
            return False
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=",\t;|")
        return dialect.delimiter == "\t"
    except (csv.Error, UnicodeDecodeError):
        return False


def _find_npy_pair(directory: Path) -> Tuple[Optional[Path], Optional[Path]]:
    npys = sorted(directory.glob("*.npy"))
    x_match = next((p for p in npys if re.search(r"(^|[_-])x([_-].*)?$", p.stem, re.IGNORECASE)), None)
    y_match = next((p for p in npys if re.search(r"(^|[_-])y([_-].*)?$", p.stem, re.IGNORECASE)), None)
    return x_match, y_match


def _has_image_files(directory: Path) -> bool:
    for child in directory.iterdir():
        if child.is_dir():
            for img in child.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    return True
        elif child.suffix.lower() in IMAGE_EXTS:
            return True
    return False


# ---------------------------------------------------------------------------
# Tabular (CSV / TSV)
# ---------------------------------------------------------------------------


def _inspect_tabular(path: Path, fmt: str) -> Dict[str, Any]:
    sep = "\t" if fmt == "tsv" else ","
    sample_df = pd.read_csv(path, sep=sep, nrows=_TABULAR_SAMPLE_ROWS)
    notes: List[str] = []

    sample_count, full_df = _resolve_sample_count(path, sep, sample_df, notes)

    label_col = _pick_label_column(sample_df.columns)
    if label_col is None:
        notes.append("No conventional label column found; using the last column as the label.")
        label_col = sample_df.columns[-1]

    label_series = full_df[label_col] if full_df is not None else sample_df[label_col]
    task_type, num_classes = _classify_label_series(label_series)

    feature_count = len(sample_df.columns) - 1
    input_shape = [feature_count]

    spec: Dict[str, Any] = {
        "modality": "vector",
        "input_shape": input_shape,
        "task_type": task_type,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": fmt,
            "data_path": str(path),
            "label_column": str(label_col),
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }
    if task_type == "classification" and num_classes is not None:
        spec["num_classes"] = int(num_classes)
    return spec


def _resolve_sample_count(
    path: Path,
    sep: str,
    sample_df: pd.DataFrame,
    notes: List[str],
) -> Tuple[Optional[int], Optional[pd.DataFrame]]:
    """
    Decide whether to do a full read for accurate counts/labels.
    Returns (sample_count, full_df_or_None).
    """
    if len(sample_df) < _TABULAR_SAMPLE_ROWS:
        return len(sample_df), sample_df

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes > 200 * 1024 * 1024:
        notes.append("Sample-only inspection: file is large, sample_count is approximate.")
        avg_row_bytes = sample_df.memory_usage(index=False, deep=True).sum() / max(len(sample_df), 1)
        approx = int(size_bytes / max(avg_row_bytes, 1)) if avg_row_bytes else None
        return approx, None

    full_df = pd.read_csv(path, sep=sep)
    if len(full_df) > _FULL_READ_ROW_THRESHOLD:
        notes.append("Sample-only label inspection above 1M rows.")
        return len(full_df), None
    return len(full_df), full_df


def _pick_label_column(columns) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in _LABEL_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _classify_label_series(series: pd.Series) -> Tuple[str, Optional[int]]:
    cleaned = series.dropna()
    if cleaned.empty:
        return "classification", None

    unique_count = cleaned.nunique()

    if pd.api.types.is_bool_dtype(cleaned) or pd.api.types.is_object_dtype(cleaned):
        return "classification", unique_count

    if pd.api.types.is_integer_dtype(cleaned):
        if unique_count <= _CLASSIFICATION_MAX_UNIQUE:
            return "classification", unique_count
        return "regression", None

    if pd.api.types.is_float_dtype(cleaned):
        threshold = max(20, int(0.05 * len(cleaned)))
        looks_integral = bool(np.all(np.isclose(cleaned.to_numpy() % 1, 0)))
        if looks_integral and unique_count <= _CLASSIFICATION_MAX_UNIQUE:
            return "classification", unique_count
        if unique_count <= threshold and unique_count <= _CLASSIFICATION_MAX_UNIQUE:
            return "classification", unique_count
        return "regression", None

    return "classification", unique_count


# ---------------------------------------------------------------------------
# ARFF (Weka / OpenML)
# ---------------------------------------------------------------------------


_ARFF_ATTRIBUTE_RE = re.compile(
    r"""^@attribute\s+
        (?P<name>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)
        \s+
        (?P<type>.+)$""",
    re.IGNORECASE | re.VERBOSE,
)


def parse_arff_header(path: Path) -> Tuple[List[Tuple[str, Any]], int]:
    """
    Parse an ARFF header and return (attributes, data_start_line).

    Each attribute is a (name, type_spec) tuple. ``type_spec`` is either a list
    of allowed values (nominal attribute) or a primitive type token string
    (``numeric``/``real``/``integer``/``string``/``date``).

    ``data_start_line`` is the 0-indexed line number of the first data row,
    suitable for ``pandas.read_csv(..., skiprows=data_start_line)``.
    """
    attributes: List[Tuple[str, Any]] = []
    data_start_line = 0
    found_data = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f):
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@relation"):
                continue
            if lower.startswith("@attribute"):
                attributes.append(_parse_arff_attribute(line))
                continue
            if lower.startswith("@data"):
                data_start_line = lineno + 1
                found_data = True
                break
    if not found_data:
        raise ValueError(f"ARFF file {path} has no @DATA marker.")
    return attributes, data_start_line


def _parse_arff_attribute(line: str) -> Tuple[str, Any]:
    match = _ARFF_ATTRIBUTE_RE.match(line)
    if not match:
        raise ValueError(f"Could not parse ARFF attribute line: {line!r}")
    name = match.group("name").strip()
    if (name.startswith("'") and name.endswith("'")) or (name.startswith('"') and name.endswith('"')):
        name = name[1:-1]
    type_text = match.group("type").strip()
    if type_text.startswith("{") and type_text.endswith("}"):
        inner = type_text[1:-1]
        values = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()]
        return name, values
    return name, type_text.split()[0].lower()


def _inspect_arff(path: Path) -> Dict[str, Any]:
    attributes, data_start_line = parse_arff_header(path)
    if not attributes:
        raise ValueError(f"No @ATTRIBUTE entries found in {path}.")

    notes: List[str] = []
    column_names = [name for name, _ in attributes]
    label_col = _pick_label_column(column_names)
    if label_col is None:
        nominal_idx = next(
            (i for i in range(len(attributes) - 1, -1, -1) if isinstance(attributes[i][1], list)),
            None,
        )
        if nominal_idx is not None:
            label_col = column_names[nominal_idx]
            notes.append(
                f"No conventional label column found; using nominal attribute '{label_col}' as the label."
            )
        else:
            label_col = column_names[-1]
            notes.append("No conventional label column found; using the last column as the label.")

    sample_df = _read_arff_data(path, column_names, data_start_line, nrows=_TABULAR_SAMPLE_ROWS)
    sample_count, full_df = _resolve_arff_sample_count(
        path, column_names, data_start_line, sample_df, notes
    )

    label_series = full_df[label_col] if full_df is not None else sample_df[label_col]
    task_type, num_classes = _classify_label_series(label_series)
    label_attr = next((spec for name, spec in attributes if name == label_col), None)
    if isinstance(label_attr, list):
        task_type = "classification"
        num_classes = len(label_attr)

    feature_count = len(column_names) - 1
    spec: Dict[str, Any] = {
        "modality": "vector",
        "input_shape": [feature_count],
        "task_type": task_type,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "arff",
            "data_path": str(path),
            "label_column": str(label_col),
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }
    if task_type == "classification" and num_classes is not None:
        spec["num_classes"] = int(num_classes)
    return spec


def _read_arff_data(
    path: Path,
    column_names: List[str],
    data_start_line: int,
    *,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=",",
        header=None,
        names=column_names,
        skiprows=data_start_line,
        na_values=["?"],
        skipinitialspace=True,
        nrows=nrows,
        comment="%",
        engine="python",
    )


def _resolve_arff_sample_count(
    path: Path,
    column_names: List[str],
    data_start_line: int,
    sample_df: pd.DataFrame,
    notes: List[str],
) -> Tuple[Optional[int], Optional[pd.DataFrame]]:
    if len(sample_df) < _TABULAR_SAMPLE_ROWS:
        return len(sample_df), sample_df
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    if size_bytes > 200 * 1024 * 1024:
        notes.append("Sample-only inspection: ARFF file is large, sample_count is approximate.")
        avg_row_bytes = sample_df.memory_usage(index=False, deep=True).sum() / max(len(sample_df), 1)
        approx = int(size_bytes / max(avg_row_bytes, 1)) if avg_row_bytes else None
        return approx, None
    full_df = _read_arff_data(path, column_names, data_start_line)
    if len(full_df) > _FULL_READ_ROW_THRESHOLD:
        notes.append("Sample-only label inspection above 1M rows.")
        return len(full_df), None
    return len(full_df), full_df


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def _inspect_parquet(path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    notes: List[str] = []
    if len(df) > _FULL_READ_ROW_THRESHOLD:
        notes.append("Parquet contains >1M rows; full read still performed (Parquet is columnar).")

    label_col = _pick_label_column(df.columns)
    if label_col is None:
        notes.append("No conventional label column found; using the last column as the label.")
        label_col = df.columns[-1]

    task_type, num_classes = _classify_label_series(df[label_col])
    feature_count = len(df.columns) - 1

    spec: Dict[str, Any] = {
        "modality": "vector",
        "input_shape": [feature_count],
        "task_type": task_type,
        "sample_count": int(len(df)),
        "loader_kwargs": {
            "format": "parquet",
            "data_path": str(path),
            "label_column": str(label_col),
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }
    if task_type == "classification" and num_classes is not None:
        spec["num_classes"] = int(num_classes)
    return spec


# ---------------------------------------------------------------------------
# .npy single-file
# ---------------------------------------------------------------------------


def _inspect_npy(path: Path) -> Dict[str, Any]:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(d) for d in arr.shape)
    notes: List[str] = []

    if arr.ndim < 1:
        raise ValueError(f"NPY at {path} has scalar shape; cannot infer dataset.")

    sample_count = shape[0]

    if arr.ndim == 1:
        notes.append("1-D npy treated as a single feature vector per index; synthesizing zero labels.")
        return _unsupervised_vector_spec(path, sample_count, notes, input_shape=[1])

    if arr.ndim == 2:
        last_col_looks_label = _last_column_looks_like_label(arr)
        if last_col_looks_label:
            label_arr = np.asarray(arr[:, -1])
            num_classes = int(np.unique(label_arr).size)
            feature_dim = shape[1] - 1
            modality = "signal" if feature_dim >= _SIGNAL_MIN_LENGTH and np.issubdtype(arr.dtype, np.floating) else "vector"
            return {
                "modality": modality,
                "input_shape": [feature_dim],
                "task_type": "classification",
                "num_classes": num_classes,
                "sample_count": sample_count,
                "loader_kwargs": {
                    "format": "npy",
                    "data_path": str(path),
                    "label_in_last_column": True,
                    "val_split": _DEFAULT_VAL_SPLIT,
                    "random_state": _DEFAULT_RANDOM_STATE,
                },
                "notes": notes,
            }
        feature_dim = shape[1]
        modality = "signal" if feature_dim >= _SIGNAL_MIN_LENGTH and np.issubdtype(arr.dtype, np.floating) else "vector"
        notes.append("2-D npy without obvious label column; synthesizing zero labels.")
        return _unsupervised_vector_spec(path, sample_count, notes, input_shape=[feature_dim], modality=modality)

    if arr.ndim == 3:
        notes.append("3-D npy treated as grayscale image stack (N, H, W); synthesizing zero labels.")
        height, width = shape[1], shape[2]
        return _unsupervised_image_spec(path, sample_count, notes, input_shape=[height, width, 1])

    if arr.ndim == 4:
        notes.append("4-D npy treated as image stack (N, H, W, C); synthesizing zero labels.")
        h, w, c = shape[1], shape[2], shape[3]
        return _unsupervised_image_spec(path, sample_count, notes, input_shape=[h, w, c])

    raise ValueError(f"NPY at {path} has unsupported rank {arr.ndim}.")


def _last_column_looks_like_label(arr: np.ndarray) -> bool:
    last = np.asarray(arr[:, -1])
    if last.size == 0:
        return False
    unique = np.unique(last)
    if unique.size > _CLASSIFICATION_MAX_UNIQUE:
        return False
    if np.issubdtype(last.dtype, np.integer):
        return True
    if np.issubdtype(last.dtype, np.floating):
        return bool(np.all(np.isclose(last % 1, 0)))
    return False


def _unsupervised_vector_spec(
    path: Path,
    sample_count: int,
    notes: List[str],
    *,
    input_shape: List[int],
    modality: str = "vector",
) -> Dict[str, Any]:
    notes.append("Unsupervised: caller must supply labels separately or downstream training will see constant targets.")
    return {
        "modality": modality,
        "input_shape": input_shape,
        "task_type": "classification",
        "num_classes": 1,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "npy",
            "data_path": str(path),
            "label_in_last_column": False,
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }


def _unsupervised_image_spec(
    path: Path,
    sample_count: int,
    notes: List[str],
    *,
    input_shape: List[int],
) -> Dict[str, Any]:
    return {
        "modality": "image",
        "input_shape": input_shape,
        "task_type": "classification",
        "num_classes": 1,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "npy",
            "data_path": str(path),
            "label_in_last_column": False,
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# .npz bundle
# ---------------------------------------------------------------------------


def _inspect_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as npz:
        keys = sorted(npz.files)
        x_key, y_key = _pick_npz_keys(keys)
        x_arr = np.asarray(npz[x_key])
        y_arr = np.asarray(npz[y_key])

    notes = [f"Using arrays x_key='{x_key}', y_key='{y_key}' from npz."]
    sample_count = int(x_arr.shape[0])
    input_shape, modality = _shape_to_input_shape_and_modality(x_arr.shape)
    num_classes = int(np.unique(y_arr).size) if y_arr.size else None

    return {
        "modality": modality,
        "input_shape": input_shape,
        "task_type": "classification",
        "num_classes": num_classes,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "npz",
            "data_path": str(path),
            "x_key": x_key,
            "y_key": y_key,
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }


def _pick_npz_keys(keys: List[str]) -> Tuple[str, str]:
    preferred_x = ("x", "X", "x_train", "X_train", "features", "data")
    preferred_y = ("y", "Y", "y_train", "Y_train", "labels", "targets")
    x_key = next((k for k in preferred_x if k in keys), None)
    y_key = next((k for k in preferred_y if k in keys), None)
    if x_key and y_key:
        return x_key, y_key
    if len(keys) < 2:
        raise ValueError(f"NPZ needs at least two arrays; found {keys}.")
    return keys[0], keys[1]


# ---------------------------------------------------------------------------
# Sibling .npy pair (x.npy + y.npy in a directory)
# ---------------------------------------------------------------------------


def _inspect_npy_pair(directory: Path) -> Dict[str, Any]:
    x_path, y_path = _find_npy_pair(directory)
    if not x_path or not y_path:
        raise ValueError(f"Could not find paired x/y .npy files in {directory}.")

    x_arr = np.load(x_path, mmap_mode="r", allow_pickle=False)
    y_arr = np.load(y_path, allow_pickle=False)
    notes = [f"Paired npy load: x='{x_path.name}', y='{y_path.name}'."]
    input_shape, modality = _shape_to_input_shape_and_modality(x_arr.shape)
    num_classes = int(np.unique(y_arr).size) if y_arr.size else None
    sample_count = int(x_arr.shape[0])

    return {
        "modality": modality,
        "input_shape": input_shape,
        "task_type": "classification",
        "num_classes": num_classes,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "npy_pair",
            "x_path": str(x_path),
            "y_path": str(y_path),
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }


def _shape_to_input_shape_and_modality(shape: Tuple[int, ...]) -> Tuple[List[int], str]:
    if len(shape) == 1:
        return [1], "vector"
    if len(shape) == 2:
        return [int(shape[1])], "vector"
    if len(shape) == 3:
        return [int(shape[1]), int(shape[2]), 1], "image"
    if len(shape) == 4:
        return [int(shape[1]), int(shape[2]), int(shape[3])], "image"
    raise ValueError(f"Unsupported feature-array rank: {len(shape)}")


# ---------------------------------------------------------------------------
# Image directory (ImageNet-style)
# ---------------------------------------------------------------------------


def _inspect_image_dir(directory: Path) -> Dict[str, Any]:
    class_dirs = sorted([p for p in directory.iterdir() if p.is_dir()])
    notes: List[str] = []

    if class_dirs:
        first_image = _first_image_under(class_dirs[0])
        num_classes = len(class_dirs)
        sample_count = sum(_count_images(c) for c in class_dirs)
    else:
        notes.append("No class subdirectories found; using a single synthetic class.")
        first_image = _first_image_under(directory)
        num_classes = 1
        sample_count = _count_images(directory)

    if first_image is None:
        raise ValueError(f"No image files found under {directory}.")

    height, width, channels = _peek_image_shape(first_image)
    resize_target = min(min(height, width), _IMAGE_RESIZE_CAP)
    resize_target = max(resize_target, 8)
    input_shape = [resize_target, resize_target, channels]

    return {
        "modality": "image",
        "input_shape": input_shape,
        "task_type": "classification",
        "num_classes": num_classes,
        "sample_count": sample_count,
        "loader_kwargs": {
            "format": "image_dir",
            "data_path": str(directory),
            "image_size": resize_target,
            "channels": channels,
            "val_split": _DEFAULT_VAL_SPLIT,
            "random_state": _DEFAULT_RANDOM_STATE,
        },
        "notes": notes,
    }


def _first_image_under(directory: Path) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        match = next(directory.rglob(f"*{ext}"), None)
        if match:
            return match
    return None


def _count_images(directory: Path) -> int:
    total = 0
    for ext in IMAGE_EXTS:
        total += sum(1 for _ in directory.rglob(f"*{ext}"))
    return total


def _peek_image_shape(image_path: Path) -> Tuple[int, int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image-directory inspection requires Pillow (PIL).") from exc
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())
    return int(height), int(width), int(channels)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return slug or "custom_dataset"
