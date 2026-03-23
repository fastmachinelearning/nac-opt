"""
TensorFlow model training and evaluation utilities.
"""

import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_model(model, train_data, val_data, epochs=50, batch_size=32, 
                patience=5, verbose=1, callbacks=None):
    """
    Trains a TensorFlow model with early stopping.
    
    Parameters:
        model: Keras model to train
        train_data: Training data (x_train, y_train) or tf.data.Dataset
        val_data: Validation data (x_val, y_val) or tf.data.Dataset
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        verbose: Verbosity level
        callbacks: Additional callbacks to use
    
    Returns:
        dict: Training history
    """
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    callbacks.append(early_stop)
    
    # Add learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience//2,
        min_lr=1e-6,
        verbose=verbose
    )
    callbacks.append(reduce_lr)
    
    # Train the model
    if isinstance(train_data, tf.data.Dataset):
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        x_train, y_train = train_data
        x_val, y_val = val_data
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    
    return history


def evaluate_model(model, test_data, return_predictions=False):
    """
    Evaluates a model on test data.
    
    Parameters:
        model: Trained Keras model
        test_data: Test data (x_test, y_test) or tf.data.Dataset
        return_predictions: Whether to return predictions
    
    Returns:
        dict: Evaluation metrics (and predictions if requested)
    """
    if isinstance(test_data, tf.data.Dataset):
        loss, accuracy = model.evaluate(test_data, verbose=0)
        if return_predictions:
            predictions = model.predict(test_data)
    else:
        x_test, y_test = test_data
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        if return_predictions:
            predictions = model.predict(x_test)
    
    results = {
        'loss': loss,
        'accuracy': accuracy
    }
    
    if return_predictions:
        results['predictions'] = predictions
    
    return results


def get_inference_time(model, input_shape, num_runs=100):
    """
    Measures average inference time for a model.
    
    Parameters:
        model: Keras model
        input_shape: Shape of input data (without batch dimension)
        num_runs: Number of inference runs to average
    
    Returns:
        float: Average inference time in seconds
    """
    # Create dummy input
    if len(input_shape) == 1:
        x = tf.random.normal((1, *input_shape))
    else:
        x = tf.random.normal((1, *input_shape))
    
    # Warm up
    _ = model(x, training=False)
    
    # Measure inference time
    start = time.time()
    for _ in range(num_runs):
        _ = model(x, training=False)
    end = time.time()
    
    return (end - start) / num_runs


def get_model_metrics(model, input_shape=None):
    """
    Gets various metrics for a model.
    
    Parameters:
        model: Keras model
        input_shape: Input shape for the model
    
    Returns:
        dict: Dictionary of model metrics
    """
    metrics = {
        'param_count': model.count_params(),
        'num_layers': len(model.layers)
    }
    
    if input_shape is not None:
        metrics['inference_time'] = get_inference_time(model, input_shape)
    
    return metrics


def evaluate_deepsets_tf(model, train_loader, val_loader, test_loader, 
                        num_epochs=100, lr=0.001, patience=7):
    """
    Evaluates DeepSets models by training and computing performance metrics.
    
    Parameters:
        model: DeepSets model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
    
    Returns:
        dict: Dictionary of metrics
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Train model
    history = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=num_epochs,
        patience=patience,
        verbose=0
    )
    
    # Evaluate on validation and test sets
    val_metrics = evaluate_model(model, val_loader)
    test_metrics = evaluate_model(model, test_loader)
    
    # Get model metrics
    model_metrics = get_model_metrics(model, input_shape=(8, 3))
    
    # Combine all metrics
    metrics = {
        'val_accuracy': val_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'param_count': model_metrics['param_count'],
        'inference_time': model_metrics.get('inference_time', 0)
    }
    
    return metrics