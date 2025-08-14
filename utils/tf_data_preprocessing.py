"""
TensorFlow data preprocessing utilities for various datasets.
Includes MNIST and other dataset preprocessing functions.
"""

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


def load_generic_dataset(dataset_name, **kwargs):
    """
    Generic dataset loader that can be extended for different datasets.
    
    Parameters:
        dataset_name (str): Name of the dataset to load
        **kwargs: Additional arguments specific to each dataset
    
    Returns:
        Preprocessed dataset
    """
    loaders = {
        'mnist': load_and_preprocess_mnist,
        # Add more datasets here as needed
        # 'cifar10': load_and_preprocess_cifar10,
        # 'fashion_mnist': load_and_preprocess_fashion_mnist,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name](**kwargs)