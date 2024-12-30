import sys
import os
import torch
import optuna
import yaml

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from data.DeepsetsDataset import setup_data_loaders

def run_deepsets_search():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set hyperparameters
    batch_size = 4096
    num_workers = 8
    
    # Load configurations
    config_dir = os.path.join(root_dir, "examples")
    with open(os.path.join(config_dir, "DeepSets/deepsets_model_example_configs.yaml"), "r") as f:
        deepsets_configs = yaml.safe_load(f)
    
    # Setup data loaders
    base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"
    train_loader, val_loader, test_loader = setup_data_loaders(
        base_file_name,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )
    print("Loaded Dataset...")
    
    # Create and configure the study
    study = optuna.create_study(
        sampler=optuna.samplers.NSGAIISampler(population_size=20),
        directions=['maximize', 'minimize']  # Note: DeepSets maximizes accuracy while minimizing BOPs
    )
    
    # Queue example architectures from config
    study.enqueue_trial(deepsets_configs['base'])
    study.enqueue_trial(deepsets_configs['large'])
    study.enqueue_trial(deepsets_configs['medium'])
    study.enqueue_trial(deepsets_configs['small'])
    study.enqueue_trial(deepsets_configs['tiny'])
    
    # Import the objective function from the main script
    from global_search import Deepsets_objective
    
    # Run optimization
    study.optimize(Deepsets_objective, n_trials=1000)
    
    return study

if __name__ == "__main__":
    study = run_deepsets_search()
    
    # Print out the best trials
    print("\nBest trials:")
    trials = study.best_trials
    
    for trial in trials:
        print(f"\nTrial {trial.number}")
        print(f"Accuracy: {trial.values[0]}")  # First objective is accuracy for DeepSets
        print(f"BOPs: {trial.values[1]}")
        print("Params:", trial.params)