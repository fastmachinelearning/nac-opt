"""Small deterministic dataset for tests, loadable via loader_path="tests.synthetic_data:load_blobs"."""

import numpy as np
import tensorflow as tf


def load_blobs(subset_size=None, flatten=True, one_hot=True, n_features=16, n_classes=4, n_samples=1200, **kwargs):
    rng = np.random.default_rng(1234)  # fixed: the data never depends on the global seed
    centers = rng.normal(scale=1.5, size=(n_classes, n_features))
    labels = rng.integers(0, n_classes, size=n_samples)
    x = (centers[labels] + rng.normal(size=(n_samples, n_features))).astype("float32")
    y = tf.keras.utils.to_categorical(labels, n_classes) if one_hot else labels
    split = n_samples * 3 // 4
    return x[:split], y[:split], x[split:], y[split:]


def load_images(subset_size=None, flatten=False, one_hot=True, size=12, n_classes=3, n_samples=240, **kwargs):
    """Tiny single-channel images (N, size, size, 1) for block/conv search tests."""
    rng = np.random.default_rng(99)
    labels = rng.integers(0, n_classes, size=n_samples)
    x = rng.normal(size=(n_samples, size, size, 1)).astype("float32")
    x[:, : size // 2, : size // 2, 0] += labels[:, None, None]  # class shows up in one quadrant
    y = tf.keras.utils.to_categorical(labels, n_classes) if one_hot else labels
    split = n_samples * 3 // 4
    return x[:split], y[:split], x[split:], y[split:]
