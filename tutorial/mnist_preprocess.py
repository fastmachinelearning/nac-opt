import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(resize_val=8, subset_size=None):
    """
    Loads the MNIST dataset, preprocesses it by resizing, normalizing, and flattening the images,
    and converts the labels to one-hot encoding. Optionally, a subset of the data can be used.
    
    Parameters:
        resize_val (int): The target height and width for resizing the images.
        subset_size (int or None): If specified, the number of samples to use from the training and validation sets.
    
    Returns:
        x_train, y_train, x_val, y_val: Preprocessed training and validation data.
    """
    # Load MNIST dataset
    (x_train_full, y_train_full), (x_val_full, y_val_full) = mnist.load_data()

    # Expand dims: from (num_samples, 28, 28) to (num_samples, 28, 28, 1)
    x_train_full = x_train_full[..., None]
    x_val_full = x_val_full[..., None]

    # Resize images
    x_train_full = tf.image.resize(x_train_full, [resize_val, resize_val]).numpy()
    x_val_full = tf.image.resize(x_val_full, [resize_val, resize_val]).numpy()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_val_full = x_val_full.astype("float32") / 255.0

    # Flatten images: reshape (num_samples, resize_val, resize_val, 1) to (num_samples, resize_val*resize_val)
    squared_resize = resize_val ** 2
    x_train_full = x_train_full.reshape(-1, squared_resize)
    x_val_full = x_val_full.reshape(-1, squared_resize)

    # Convert labels to one-hot encoding (10 classes)
    num_classes = 10
    y_train_full = to_categorical(y_train_full, num_classes)
    y_val_full = to_categorical(y_val_full, num_classes)

    # Subset of data is specified
    if subset_size is not None:
        x_train = x_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
        x_val = x_val_full[:subset_size]
        y_val = y_val_full[:subset_size]
    else:
        x_train, y_train, x_val, y_val = x_train_full, y_train_full, x_val_full, y_val_full

    print(f"Data loaded and resized to {resize_val}x{resize_val}")
    print(f"x_train shape: {x_train.shape}, x_val shape: {x_val.shape}")
    return x_train, y_train, x_val, y_val

def visualize_sample(x_train, index=0, resize_val=8, save_path="sample_image.png"):
    """
    Visualizes a sample image from the training set and optionally saves it.
    
    Parameters:
        x_train (np.array): The training data containing flattened images.
        index (int): The index of the image to visualize.
        resize_val (int): The height and width used in preprocessing.
        save_path (str or None): If provided, the image will be saved to this path.
    """
    # Reshape the flattened image back to its 2D form
    image = x_train[index].reshape(resize_val, resize_val)
    plt.imshow(image, cmap="gray")
    plt.title(f"Sample Image (Index {index}, {resize_val}x{resize_val})")
    plt.axis("off")
    
    # Save the image when path provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Image saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess MNIST data with configurable resize value.")
    parser.add_argument("--resize", type=int, default=8, help="Resize value for MNIST images (default: 8)")
    parser.add_argument("--subset_size", type=int, default=60000, help="Number of training samples to use (default: 60000)")
    parser.add_argument("--index", type=int, default=0, help="Index of sample image to visualize (default: 0)")
    args = parser.parse_args()

    # Load data
    x_train, y_train, x_val, y_val = load_and_preprocess_data(resize_val=args.resize, subset_size=args.subset_size)

    # Visualize sample image
    visualize_sample(x_train, index=args.index, resize_val=args.resize)