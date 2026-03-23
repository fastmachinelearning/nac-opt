"""
Qubit readout dataset loader.
Loads and preprocesses the qubit IQ data from .npy files.
"""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_and_preprocess_qubit(
    data_dir="./data/qubit",
    x_train_file="0528_X_train_0_770.npy",
    y_train_file="0528_y_train_0_770.npy",
    x_test_file="0528_X_test_0_770.npy",
    y_test_file="0528_y_test_0_770.npy",
    start_location=100,
    window_size=400,
    subset_size=None,
    normalize=False,
    flatten=True,
    one_hot=True,
    num_classes=2,
):
    """
    Load qubit dataset from .npy files (already split into train/test).

    Returns:
        x_train, y_train, x_test, y_test

    Windowing matches train.ipynb:
        X[:, start_location*2 : (start_location+window_size)*2]
    """
    x_train_path = os.path.join(data_dir, x_train_file)
    y_train_path = os.path.join(data_dir, y_train_file)
    x_test_path = os.path.join(data_dir, x_test_file)
    y_test_path = os.path.join(data_dir, y_test_file)

    for p in (x_train_path, y_train_path, x_test_path, y_test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset file: {p}")

    x_train_full = np.load(x_train_path)
    y_train_full = np.load(y_train_path)
    x_test_full = np.load(x_test_path)
    y_test_full = np.load(y_test_path)

    end_window = start_location + window_size
    x_train_full = x_train_full[:, start_location * 2 : end_window * 2]
    x_test_full = x_test_full[:, start_location * 2 : end_window * 2]

    x_train_full = x_train_full.astype(np.float32)
    x_test_full = x_test_full.astype(np.float32)

    if normalize:
        mu = np.mean(x_train_full, axis=0, keepdims=True)
        sigma = np.std(x_train_full, axis=0, keepdims=True) + 1e-8
        x_train_full = (x_train_full - mu) / sigma
        x_test_full = (x_test_full - mu) / sigma

    if not flatten:
        x_train_full = np.expand_dims(x_train_full, axis=-1)
        x_test_full = np.expand_dims(x_test_full, axis=-1)

    if one_hot:
        y_train_full = to_categorical(y_train_full, num_classes=num_classes)
        y_test_full = to_categorical(y_test_full, num_classes=num_classes)
    else:
        y_train_full = y_train_full.astype(np.int64)
        y_test_full = y_test_full.astype(np.int64)

    if subset_size is not None:
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(x_train_full))
        x_train = x_train_full[idx[:subset_size]]
        y_train = y_train_full[idx[:subset_size]]
    else:
        x_train, y_train = x_train_full, y_train_full

    x_test, y_test = x_test_full, y_test_full

    print("Qubit data loaded and preprocessed:")
    print(f"  data_dir: {os.path.abspath(data_dir)}")
    print(f"  x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"  y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test
