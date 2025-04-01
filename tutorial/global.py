#!/usr/bin/env python3
import os
import argparse
import optuna
import yaml
import pandas as pd
import tensorflow as tf
import time
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
from rule4ml.models.estimators import MultiModelEstimator

# from data_preprocessing import load_and_preprocess_data

FIXED_LR = 0.001 # add as a config

def load_search_space(yaml_path):
    """Load the search space from a YAML file."""
    with open(yaml_path, "r") as f:
        search_space = yaml.safe_load(f)
    return search_space

def load_mnist_data(subset_size=100000, resize_val=8):
    """Loads and preprocesses MNIST (resize, normalize, flatten, one-hot)."""
    (x_train_full, y_train_full), (x_val_full, y_val_full) = mnist.load_data()
    # Expand dims: (num_samples, 28, 28) -> (num_samples, 28, 28, 1)
    x_train_full = x_train_full[..., None]
    x_val_full = x_val_full[..., None]
    # Resize images
    x_train_full = tf.image.resize(x_train_full, [resize_val, resize_val]).numpy()
    x_val_full = tf.image.resize(x_val_full, [resize_val, resize_val]).numpy()
    # Normalize pixel values
    x_train_full = x_train_full.astype("float32") / 255.0
    x_val_full = x_val_full.astype("float32") / 255.0
    # Flatten images
    flat_size = resize_val ** 2
    x_train_full = x_train_full.reshape(-1, flat_size)
    x_val_full = x_val_full.reshape(-1, flat_size)
    # One-hot encode labels
    num_classes = 10
    y_train_full = to_categorical(y_train_full, num_classes)
    y_val_full = to_categorical(y_val_full, num_classes)
    # Subset the data if needed
    x_train = x_train_full[:subset_size]
    y_train = y_train_full[:subset_size]
    x_val = x_val_full[:subset_size]
    y_val = y_val_full[:subset_size]
    return x_train, y_train, x_val, y_val

def create_objective(x_train, y_train, x_val, y_val, search_space, epochs, global_estimator, hls_config, objectives):
    """
    Returns an objective function that:
      - Samples hyperparameters from the YAML-defined search space.
      - Builds, trains, and evaluates a Keras model.
      - Estimates hardware metrics (resource usage, clock cycles, Bit Operations).
    """
    def objective(trial):
        # Sample hyperparameters using the lists defined in the YAML.
        num_layers   = trial.suggest_categorical("num_layers", search_space["num_layers"])
        hidden_units1 = trial.suggest_categorical("hidden_units1", search_space["hidden_units1"])
        activation1  = trial.suggest_categorical("activation1", search_space["activation1"])
        batchnorm1   = trial.suggest_categorical("batchnorm1", search_space["batchnorm1"])
        
        if num_layers == 3:
            hidden_units2 = trial.suggest_categorical("hidden_units2", search_space["hidden_units2"])
            activation2   = trial.suggest_categorical("activation2", search_space["activation2"])
            batchnorm2    = trial.suggest_categorical("batchnorm2", search_space["batchnorm2"])
        
        input_size = x_train.shape[1]
        num_classes = 10
        
        # Build the model.
        model = Sequential(name="Simple_MLP")
        model.add(Input(shape=(input_size,)))
        
        # First hidden layer.
        if batchnorm1:
            model.add(Dense(hidden_units1, use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation(activation1))
        else:
            model.add(Dense(hidden_units1, activation=activation1))
        
        # Optional second hidden layer.
        if num_layers == 3:
            if batchnorm2:
                model.add(Dense(hidden_units2, use_bias=False))
                model.add(BatchNormalization())
                model.add(Activation(activation2))
            else:
                model.add(Dense(hidden_units2, activation=activation2))
        
        # Output layer.
        model.add(Dense(num_classes, activation="softmax"))
        model.build((None, input_size))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FIXED_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=128,
            verbose=0  # Set to 1 for logs
        )
        val_accuracy = history.history["val_accuracy"][-1]
        
        # Compute Bit Operations (BOPs) as a simple proxy.
        if num_layers == 2:
            bops_first = input_size * hidden_units1
            bops_output = hidden_units1 * num_classes
            total_bops = bops_first + bops_output
        else:
            bops_first  = input_size * hidden_units1
            bops_second = hidden_units1 * hidden_units2
            bops_output = hidden_units2 * num_classes
            total_bops = bops_first + bops_second + bops_output

        # Patch model layers for rule4ml to extract shape information.
        start_patch = time.time()
        for layer in model.layers:
            if hasattr(layer, "input_spec") and layer.input_spec and layer.input_spec.shape:
                layer._build_shapes_dict = {"input": layer.input_spec.shape}
            elif hasattr(layer, "input_shape") and layer.input_shape is not None:
                layer._build_shapes_dict = {"input": layer.input_shape}
            else:
                layer._build_shapes_dict = {"input": (None, input_size)}
        end_patch = time.time()
        print(f"Time for patching model layers: {end_patch - start_patch:.2f} seconds")
        
        # Estimate hardware metrics using rule4ml.
        start_estimation = time.time()
        pred_df = global_estimator.predict([model], [hls_config])
        end_estimation = time.time()
        print(f"Time for hardware estimation: {end_estimation - start_estimation:.2f} seconds")
        
        # if not pred_df.empty:
        #     row = pred_df.iloc[0]
        #     avg_resource = (row.get("LUT (%)", 0) + row.get("BRAM (%)", 0) +
        #                     row.get("DSP (%)", 0) + row.get("FF (%)", 0)) / 4.0
        #     latency = row.get("CYCLES", 0)
        # else:
        #     avg_resource = 0.0
        #     latency = 0.0

        # Get individual resource values.
        if not pred_df.empty:
            row = pred_df.iloc[0]
            lut   = row.get("LUT (%)", 0)
            bram  = row.get("BRAM (%)", 0)
            dsp   = row.get("DSP (%)", 0)
            ff    = row.get("FF (%)", 0)
            avg_resource = (lut + bram + dsp + ff) / 4.0
            clock_cycles = row.get("CYCLES", 0)
        else:
            lut = bram = dsp = ff = avg_resource = clock_cycles = 0.0

        # Save individual resource values as trial attr
        trial.set_user_attr("LUT (%)", lut)
        trial.set_user_attr("BRAM (%)", bram)
        trial.set_user_attr("DSP (%)", dsp)
        trial.set_user_attr("FF (%)", ff)

        print(f"Trial {trial.number}: num_layers={num_layers}, hidden_units1={hidden_units1}, activation1={activation1}, batchnorm1={batchnorm1}", end="")
        if num_layers == 3:
            print(f", hidden_units2={hidden_units2}, activation2={activation2}, batchnorm2={batchnorm2}")
        else:
            print()
        print(f"Val Accuracy: {val_accuracy:.4f}, BOPs: {total_bops}, Avg Resource: {avg_resource}, Clock Cycles: {clock_cycles}")
        
        # Dictionary of all metrics
        metrics = {
            "accuracy": val_accuracy,
            "BOPs": total_bops,
            "avg_hw": avg_resource,
            "clock_cycles": clock_cycles,
            "LUT (%)": lut,
            "BRAM (%)": bram,
            "DSP (%)": dsp,
            "FF (%)": ff
        }
        
        # Build the tuple of objectives in the order specified.
        objective_values = tuple(metrics[obj] for obj in objectives)
        
        print(f"Trial {trial.number}: Objectives {objective_values}")
        return objective_values
    return objective

    #     # Return a tuple with objectives:
    #     #   - Accuracy (to maximize)
    #     #   - BOPs, resource, and latency (to minimize)
    #     return val_accuracy, total_bops, avg_resource, clock_cycles
    
    # return objective

# def run_global_search(n_trials, epochs, search_space_path, hls_config):
#     # Load the search space from YAML.
#     search_space = load_search_space(search_space_path)
#     # Load the data.
#     x_train, y_train, x_val, y_val = load_mnist_data(subset_size=60000, resize_val=8)
#     # Instantiate the global hardware estimator.
#     global_estimator = MultiModelEstimator()
#     global_estimator.load_default_models()
#     # Create the objective function with our current configuration.
#     objective = create_objective(x_train, y_train, x_val, y_val, search_space, epochs, global_estimator, hls_config)
#     # Create an Optuna study for multi-objective optimization.
#     study = optuna.create_study(
#         directions=["maximize", "minimize", "minimize", "minimize"]
#     )
#     study.optimize(objective, n_trials=n_trials)
    
#     results_str = f"Number of trials: {len(study.trials)}\n"
#     for trial in study.trials:
#         # Get individual resource metrics from user attributes.
#         lut   = trial.user_attrs.get("LUT (%)", "N/A")
#         bram  = trial.user_attrs.get("BRAM (%)", "N/A")
#         dsp   = trial.user_attrs.get("DSP (%)", "N/A")
#         ff    = trial.user_attrs.get("FF (%)", "N/A")
#         results_str += (f"Trial {trial.number} | Objectives: {trial.values} | "
#                         f"Params: {trial.params}\n"
#                         f"Resources: LUT = {lut}, BRAM = {bram}, DSP = {dsp}, FF = {ff}\n")
    
#     print(results_str)

#     # Save results to a text file.
#     with open("global_search_results.txt", "w") as f:
#         f.write(results_str)

def run_global_search(n_trials, epochs, search_space_path, hls_config, objectives):
    search_space = load_search_space(search_space_path)
    x_train, y_train, x_val, y_val = load_mnist_data(subset_size=60000, resize_val=8)
    global_estimator = MultiModelEstimator()
    global_estimator.load_default_models()
    objective = create_objective(x_train, y_train, x_val, y_val, search_space, epochs, global_estimator, hls_config, objectives)
    
    # Determine optimization directions: accuracy is maximized, all others minimized.
    directions = []
    for obj in objectives:
        if obj.lower() == "accuracy":
            directions.append("maximize")
        else:
            directions.append("minimize")
    
    study = optuna.create_study(directions=directions)
    study.optimize(objective, n_trials=n_trials)
    
    results_str = f"Number of trials: {len(study.trials)}\n"
    for trial in study.trials:
        # results_str += (f"Trial {trial.number} | Objectives: {trial.values} | Params: {trial.params}\n"
        #                 f"Resources: {resources}\n")
        objective_dict = dict(zip(objectives, trial.values))

        resources = (f"LUT = {trial.user_attrs.get('LUT (%)', 'N/A')}, "
                     f"BRAM = {trial.user_attrs.get('BRAM (%)', 'N/A')}, "
                     f"DSP = {trial.user_attrs.get('DSP (%)', 'N/A')}, "
                     f"FF = {trial.user_attrs.get('FF (%)', 'N/A')}")
        results_str += f"Trial {trial.number} | Objectives: {objective_dict} | Params: {trial.params}\n"
    
    print(results_str)
    with open("global_search_results.txt", "w") as f:
        f.write(results_str)

def main():
    parser = argparse.ArgumentParser(description="Global Search Script for NAC")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training each candidate")
    parser.add_argument("--search_space", type=str, default="search_space.yaml", help="Path to the search space YAML file")
    parser.add_argument("--board", type=str, default="zcu102", help="Target board for hardware estimation")
    
    # arguments for hardware configuration
    parser.add_argument("--precision", type=str, default="ap_fixed<8, 3>", help="Model precision for hardware estimation")
    parser.add_argument("--reuse_factor", type=int, default=1, help="Reuse factor for hardware estimation")
    parser.add_argument("--strategy", type=str, default="Latency", help="Strategy for hardware estimation")

    parser.add_argument(
    "--objectives",
    type=str,
    default="accuracy,BOPs,avg_hw,clock_cycles",
    help=("Comma-separated list of objectives to optimize. "
          "Default is 'accuracy,BOPs,avg_hw,clock_cycles'. "
          "You can also include hardware specifics like 'LUT (%)', 'BRAM (%)', 'DSP (%)', 'FF (%)'.")
    )
    
    args = parser.parse_args()
    objectives = [obj.strip() for obj in args.objectives.split(",")]
    
    # Use the command-line arguments in the hardware config
    hls_config = {
        "model": {
            "precision": args.precision,
            "reuse_factor": args.reuse_factor,
            "strategy": args.strategy
        },
        "board": args.board
    }
    
    # run_global_search(n_trials=args.n_trials,
    #                   epochs=args.epochs,
    #                   search_space_path=args.search_space,
    #                   hls_config=hls_config)

    run_global_search(
    n_trials=args.n_trials,
    epochs=args.epochs,
    search_space_path=args.search_space,
    hls_config=hls_config,
    objectives=objectives)

if __name__ == "__main__":
    main()