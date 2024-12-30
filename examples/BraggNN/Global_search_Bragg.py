import sys
import os
import torch
import optuna
import yaml

# Add the root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from data.BraggnnDataset import setup_data_loaders

def run_braggnn_search():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set hyperparameters
    batch_size = 4096
    num_workers = 8
    
    # Load configurations
    config_dir = os.path.join(root_dir, "examples")
    with open(os.path.join(config_dir, "BraggNN/braggnn_model_example_configs.yaml"), "r") as f:
        braggnn_configs = yaml.safe_load(f)
    
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
        directions=['minimize', 'minimize']
    )
    
    # Queue example architectures from config
    study.enqueue_trial(braggnn_configs['openhls'])
    study.enqueue_trial(braggnn_configs['braggnn'])
    study.enqueue_trial(braggnn_configs['example1'])
    study.enqueue_trial(braggnn_configs['example2'])
    study.enqueue_trial(braggnn_configs['example3'])
    
    # Import the objective function from the main script
    from global_search import BraggNN_objective
    
    # Run optimization
    study.optimize(BraggNN_objective, n_trials=1000)
    
    return study

if __name__ == "__main__":
    study = run_braggnn_search()
    
    # Print out the best trials
    print("\nBest trials:")
    trials = study.best_trials
    
    for trial in trials:
        print(f"\nTrial {trial.number}")
        print(f"Mean Distance: {trial.values[0]}")
        print(f"BOPs: {trial.values[1]}")
        print("Params:", trial.params)