"""
TensorFlow data preprocessing utilities for various datasets.
Includes MNIST and other dataset preprocessing functions.
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


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

# new function to load the qubit dataset

def load_and_preprocess_qubit(
    data_dir="./data",
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
        x_train = x_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
    else:
        x_train, y_train = x_train_full, y_train_full

    x_test, y_test = x_test_full, y_test_full

    print("Qubit data loaded and preprocessed:")
    print(f"  data_dir: {os.path.abspath(data_dir)}")
    print(f"  x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
    print(f"  y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def load_generic_dataset(dataset_name, **kwargs):
    """
    Generic dataset loader that dispatches to specific dataset preprocessing functions.

    Parameters:
        dataset_name (str): Name of the dataset to load
        **kwargs: Additional arguments specific to each dataset (passed through)

    Returns:
        tuple: (x_train, y_train, x_val, y_val) - Preprocessed numpy arrays

    To add a new dataset:
        1. Create a load_and_preprocess_<name>() function above
        2. Register it in the loaders dict below
        3. See CUSTOM_DATASETS.md for detailed instructions
    """
    loaders = {
        'mnist': load_and_preprocess_mnist,
        'fashion_mnist': load_and_preprocess_fashion_mnist,
        'qubit': load_and_preprocess_qubit,
        # Add your custom dataset loader here:
        # 'my_dataset': load_and_preprocess_my_dataset,
    }

    if dataset_name not in loaders:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available: {list(loaders.keys())}\n"
            f"See CUSTOM_DATASETS.md for how to add custom datasets."
        )

    return loaders[dataset_name](**kwargs)