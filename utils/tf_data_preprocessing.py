"""
TensorFlow data preprocessing utilities for various datasets.
Includes MNIST and other dataset preprocessing functions.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from data.qubit_dataset import load_and_preprocess_qubit
from data.hls4ml_jets_dataset import load_and_preprocess_hls4ml_jets


def load_and_preprocess_mnist(resize_val=8, subset_size=None, normalize=True, flatten=True, one_hot=True):
    """
    Loads and preprocesses MNIST dataset with configurable options.
    
    Parameters:
        resize_val (int): Target height and width for resizing images
        subset_size (int or None): Number of samples to use (None for full dataset)
        normalize (bool): Whether to normalize pixel values to [0, 1]
        flatten (bool): Whether to flatten images
        one_hot (bool): Whether to one-hot encode labels
    
    Returns:
        x_train, y_train, x_val, y_val: Preprocessed training and validation data
    """
    # Load MNIST dataset
    (x_train_full, y_train_full), (x_val_full, y_val_full) = mnist.load_data()
    
    # Expand dims: (num_samples, 28, 28) -> (num_samples, 28, 28, 1)
    x_train_full = x_train_full[..., None]
    x_val_full = x_val_full[..., None]
    
    # Resize images if needed
    if resize_val != 28:
        x_train_full = tf.image.resize(x_train_full, [resize_val, resize_val]).numpy()
        x_val_full = tf.image.resize(x_val_full, [resize_val, resize_val]).numpy()
    
    # Normalize pixel values
    if normalize:
        x_train_full = x_train_full.astype("float32") / 255.0
        x_val_full = x_val_full.astype("float32") / 255.0
    
    # Flatten images if requested
    if flatten:
        flat_size = resize_val ** 2
        x_train_full = x_train_full.reshape(-1, flat_size)
        x_val_full = x_val_full.reshape(-1, flat_size)
    
    # One-hot encode labels if requested
    if one_hot:
        num_classes = 10
        y_train_full = to_categorical(y_train_full, num_classes)
        y_val_full = to_categorical(y_val_full, num_classes)
    
    # Subset data if specified
    if subset_size is not None:
        x_train = x_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
        x_val = x_val_full[:subset_size]
        y_val = y_val_full[:subset_size]
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val, y_val = x_val_full, y_val_full
    
    print(f"Data loaded and preprocessed:")
    print(f"  Resize: {resize_val}x{resize_val}")
    print(f"  x_train shape: {x_train.shape}, x_val shape: {x_val.shape}")
    print(f"  y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
    
    return x_train, y_train, x_val, y_val

def load_and_preprocess_fashion_mnist(resize_val=8, subset_size=None, normalize=True, flatten=True, one_hot=True):
    """
    Loads and preprocesses Fashion MNIST dataset.
    """
    from tensorflow.keras.datasets import fashion_mnist
    (x_train_full, y_train_full), (x_val_full, y_val_full) = fashion_mnist.load_data()

    # Expand dims: (num_samples, 28, 28) -> (num_samples, 28, 28, 1)
    x_train_full = x_train_full[..., None]
    x_val_full = x_val_full[..., None]
    
    # Resize images if needed
    if resize_val != 28:
        x_train_full = tf.image.resize(x_train_full, [resize_val, resize_val]).numpy()
        x_val_full = tf.image.resize(x_val_full, [resize_val, resize_val]).numpy()
    
    # Normalize pixel values
    if normalize:
        x_train_full = x_train_full.astype("float32") / 255.0
        x_val_full = x_val_full.astype("float32") / 255.0
    
    # Flatten images if requested
    if flatten:
        flat_size = resize_val ** 2
        x_train_full = x_train_full.reshape(-1, flat_size)
        x_val_full = x_val_full.reshape(-1, flat_size)
    
    # One-hot encode labels if requested
    if one_hot:
        num_classes = 10
        y_train_full = to_categorical(y_train_full, num_classes)
        y_val_full = to_categorical(y_val_full, num_classes)
    
    # Subset data if specified
    if subset_size is not None:
        x_train = x_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
        x_val = x_val_full[:subset_size]
        y_val = y_val_full[:subset_size]
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val, y_val = x_val_full, y_val_full
    
    print(f"Data loaded and preprocessed:")
    print(f"  Resize: {resize_val}x{resize_val}")
    print(f"  x_train shape: {x_train.shape}, x_val shape: {x_val.shape}")
    print(f"  y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

    return x_train, y_train, x_val, y_val
    


def create_tf_dataset(x_data, y_data, batch_size=32, shuffle=True, buffer_size=10000):
    """
    Creates a tf.data.Dataset from numpy arrays.
    
    Parameters:
        x_data: Input features
        y_data: Labels
        batch_size: Batch size for the dataset
        shuffle: Whether to shuffle the dataset
        buffer_size: Buffer size for shuffling
    
    Returns:
        tf.data.Dataset object
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


_BUILTIN_LOADERS = {
    'mnist': load_and_preprocess_mnist,
    'fashion_mnist': load_and_preprocess_fashion_mnist,
    'qubit': load_and_preprocess_qubit,
    'hls4ml_jets': load_and_preprocess_hls4ml_jets,
}


def _load_dataset_loader(dataset_name):
    """Return the loader callable for a builtin dataset, or the generic loader otherwise."""
    return _BUILTIN_LOADERS.get(dataset_name, load_and_preprocess_from_path)


def load_generic_dataset(dataset_name, **kwargs):
    """
    Generic dataset loader that dispatches to specific dataset preprocessing functions.

    For unknown dataset names, falls through to ``load_and_preprocess_from_path``
    whenever ``loader_kwargs`` carries a ``format`` or ``data_path``. This is the
    path used by datasets inspected via ``utils.dataset_inspector``.

    Parameters:
        dataset_name (str): Name of the dataset to load
        **kwargs: Additional arguments specific to each dataset (passed through)

    Returns:
        tuple: (x_train, y_train, x_val, y_val) - Preprocessed numpy arrays
    """
    if dataset_name in _BUILTIN_LOADERS:
        return _BUILTIN_LOADERS[dataset_name](**kwargs)

    if kwargs.get("format") or kwargs.get("data_path") or kwargs.get("x_path"):
        return load_and_preprocess_from_path(dataset_name=dataset_name, **kwargs)

    raise ValueError(
        f"Dataset '{dataset_name}' not supported. "
        f"Available builtins: {list(_BUILTIN_LOADERS)}. "
        "Pass loader_kwargs from utils.dataset_inspector for arbitrary files."
    )


def load_and_preprocess_from_path(
    *,
    format,
    data_path=None,
    x_path=None,
    y_path=None,
    x_key=None,
    y_key=None,
    label_column=None,
    label_in_last_column=False,
    image_size=None,
    channels=None,
    val_split=0.2,
    random_state=42,
    subset_size=None,
    normalize=True,
    flatten=False,
    one_hot=True,
    dataset_name=None,
    **_ignored,
):
    """
    Generic file-based dataset loader for CSV/TSV/Parquet/NPY/NPZ/image dirs.

    Consumes the ``loader_kwargs`` payload produced by
    ``utils.dataset_inspector.inspect_dataset_path`` plus the planner-level
    kwargs (``subset_size``, ``normalize``, ``flatten``, ``one_hot``).

    Returns ``(x_train, y_train, x_val, y_val)`` as numpy arrays.
    """
    fmt = str(format).lower()

    if fmt in ("csv", "tsv"):
        sep = "\t" if fmt == "tsv" else ","
        x, y, label_lookup = _load_tabular_file(data_path, sep=sep, label_column=label_column)
    elif fmt == "arff":
        x, y, label_lookup = _load_arff_file(data_path, label_column=label_column)
    elif fmt == "parquet":
        x, y, label_lookup = _load_parquet_file(data_path, label_column=label_column)
    elif fmt == "npy":
        x, y, label_lookup = _load_single_npy(data_path, label_in_last_column=label_in_last_column)
    elif fmt == "npz":
        x, y, label_lookup = _load_npz_bundle(data_path, x_key=x_key, y_key=y_key)
    elif fmt == "npy_pair":
        x, y, label_lookup = _load_npy_pair(x_path, y_path)
    elif fmt == "image_dir":
        x, y, label_lookup = _load_image_directory(
            data_path,
            image_size=image_size,
            channels=channels,
        )
    else:
        raise ValueError(f"Unsupported generic loader format: {format!r}")

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Feature/label length mismatch: x={x.shape}, y={y.shape}")

    if normalize and np.issubdtype(x.dtype, np.floating) is False and x.dtype != np.bool_:
        x = x.astype("float32")
    elif normalize:
        x = x.astype("float32")

    if flatten and x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    if normalize:
        x = _zscore_normalize(x)

    y_int, num_classes = _encode_labels(y, label_lookup)

    x_train, y_train, x_val, y_val = _train_val_split(
        x, y_int, val_split=val_split, random_state=random_state, num_classes=num_classes
    )

    if subset_size is not None:
        x_train = x_train[:subset_size]
        y_train = y_train[:subset_size]
        x_val = x_val[: max(1, subset_size // max(1, int(1 / max(val_split, 1e-6))))]
        y_val = y_val[: max(1, subset_size // max(1, int(1 / max(val_split, 1e-6))))]

    if one_hot and num_classes > 1:
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)

    print(
        f"Generic-loader data loaded (format={fmt}, name={dataset_name}):\n"
        f"  x_train shape: {x_train.shape}, x_val shape: {x_val.shape}\n"
        f"  y_train shape: {y_train.shape}, y_val shape: {y_val.shape}"
    )
    return x_train, y_train, x_val, y_val


def _load_tabular_file(path, *, sep, label_column):
    if not path:
        raise ValueError("Tabular load requires data_path.")
    df = pd.read_csv(path, sep=sep)
    if label_column not in df.columns:
        raise ValueError(f"label_column={label_column!r} not in columns {list(df.columns)}")
    y = df[label_column].to_numpy()
    features = df.drop(columns=[label_column])
    x = features.to_numpy(dtype=np.float32, na_value=0.0)
    return x, y, None


def _load_arff_file(path, *, label_column):
    if not path:
        raise ValueError("ARFF load requires data_path.")
    from pathlib import Path

    from utils.dataset_inspector import parse_arff_header

    arff_path = Path(path)
    attributes, data_start_line = parse_arff_header(arff_path)
    if not attributes:
        raise ValueError(f"No @ATTRIBUTE entries found in {arff_path}.")
    column_names = [name for name, _ in attributes]
    if label_column is None:
        label_column = column_names[-1]
    if label_column not in column_names:
        raise ValueError(
            f"label_column={label_column!r} not in ARFF attributes {column_names}"
        )
    df = pd.read_csv(
        arff_path,
        sep=",",
        header=None,
        names=column_names,
        skiprows=data_start_line,
        na_values=["?"],
        skipinitialspace=True,
        comment="%",
        engine="python",
    )
    label_attr = next((spec for name, spec in attributes if name == label_column), None)
    label_lookup = None
    if isinstance(label_attr, list):
        label_lookup = {value: idx for idx, value in enumerate(label_attr)}
        y_raw = df[label_column].astype(str).str.strip().str.strip("'\"")
        y = y_raw.map(label_lookup).to_numpy()
        if pd.isna(y).any():
            unknown = sorted({v for v, mapped in zip(y_raw, y) if pd.isna(mapped)})
            raise ValueError(
                f"ARFF label column {label_column!r} contains values not declared in "
                f"the @ATTRIBUTE nominal set: {unknown[:5]}"
            )
        y = y.astype(np.int64)
    else:
        y = df[label_column].to_numpy()
    features = df.drop(columns=[label_column])
    x = features.to_numpy(dtype=np.float32, na_value=0.0)
    return x, y, label_lookup


def _load_parquet_file(path, *, label_column):
    if not path:
        raise ValueError("Parquet load requires data_path.")
    df = pd.read_parquet(path)
    if label_column not in df.columns:
        raise ValueError(f"label_column={label_column!r} not in columns {list(df.columns)}")
    y = df[label_column].to_numpy()
    features = df.drop(columns=[label_column])
    x = features.to_numpy(dtype=np.float32, na_value=0.0)
    return x, y, None


def _load_single_npy(path, *, label_in_last_column):
    if not path:
        raise ValueError("NPY load requires data_path.")
    arr = np.load(path, allow_pickle=False)
    if label_in_last_column and arr.ndim == 2:
        x = arr[:, :-1].astype("float32")
        y = arr[:, -1]
    else:
        x = arr.astype("float32") if not np.issubdtype(arr.dtype, np.floating) else arr
        y = np.zeros(arr.shape[0], dtype=np.int64)
    return x, y, None


def _load_npz_bundle(path, *, x_key, y_key):
    if not path:
        raise ValueError("NPZ load requires data_path.")
    with np.load(path, allow_pickle=False) as npz:
        if not x_key or x_key not in npz.files:
            raise ValueError(f"x_key={x_key!r} not in npz keys {list(npz.files)}")
        if not y_key or y_key not in npz.files:
            raise ValueError(f"y_key={y_key!r} not in npz keys {list(npz.files)}")
        x = np.asarray(npz[x_key])
        y = np.asarray(npz[y_key])
    return x, y, None


def _load_npy_pair(x_path, y_path):
    if not x_path or not y_path:
        raise ValueError("npy_pair load requires both x_path and y_path.")
    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    return x, y, None


def _load_image_directory(path, *, image_size, channels):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("image_dir loader requires Pillow (PIL).") from exc
    if not path:
        raise ValueError("image_dir load requires data_path.")
    root = os.path.abspath(path)
    target_size = int(image_size) if image_size else 32
    target_channels = int(channels) if channels else 3
    pil_mode = "RGB" if target_channels == 3 else ("L" if target_channels == 1 else "RGBA")

    class_dirs = sorted(
        [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    )
    label_lookup = {name: idx for idx, name in enumerate(class_dirs)}

    images = []
    labels = []
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif")

    if class_dirs:
        for class_name in class_dirs:
            class_dir = os.path.join(root, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(extensions):
                    continue
                fpath = os.path.join(class_dir, fname)
                with Image.open(fpath) as img:
                    img = img.convert(pil_mode).resize((target_size, target_size))
                    images.append(np.asarray(img, dtype=np.float32))
                labels.append(label_lookup[class_name])
    else:
        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith(extensions):
                continue
            fpath = os.path.join(root, fname)
            with Image.open(fpath) as img:
                img = img.convert(pil_mode).resize((target_size, target_size))
                images.append(np.asarray(img, dtype=np.float32))
            labels.append(0)
        label_lookup = {"_unlabeled": 0}

    if not images:
        raise ValueError(f"No images found under {root}")

    x = np.stack(images, axis=0)
    if x.ndim == 3:
        x = x[..., None]
    x = x / 255.0
    y = np.array(labels, dtype=np.int64)
    return x, y, label_lookup


def _encode_labels(y, label_lookup):
    y = np.asarray(y)
    if label_lookup is not None:
        num_classes = len(label_lookup)
        return y.astype(np.int64), num_classes
    if np.issubdtype(y.dtype, np.integer) or y.dtype == np.bool_:
        unique = np.unique(y)
        mapping = {v: i for i, v in enumerate(unique)}
        remapped = np.array([mapping[v] for v in y], dtype=np.int64)
        return remapped, len(unique)
    if np.issubdtype(y.dtype, np.floating):
        is_integral = np.all(np.isclose(y % 1, 0))
        if is_integral:
            ints = y.astype(np.int64)
            unique = np.unique(ints)
            mapping = {v: i for i, v in enumerate(unique)}
            remapped = np.array([mapping[v] for v in ints], dtype=np.int64)
            return remapped, len(unique)
        return y.astype("float32"), 1
    unique = np.unique(y)
    mapping = {v: i for i, v in enumerate(unique)}
    remapped = np.array([mapping[v] for v in y], dtype=np.int64)
    return remapped, len(unique)


def _zscore_normalize(x):
    if x.ndim < 2:
        return x
    if x.ndim == 2:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (x - mean) / std
    if x.dtype == np.float32 or x.dtype == np.float64:
        peak = float(np.max(np.abs(x))) if x.size else 1.0
        if peak > 1.5:
            return x / 255.0
    return x


def _train_val_split(x, y, *, val_split, random_state, num_classes):
    n = x.shape[0]
    if n <= 1 or val_split <= 0:
        return x, y, x[:0], y[:0]

    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_count = max(1, int(round(n * float(val_split))))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]