import optuna
import torch
import torch.nn as nn

from data import DeepsetsDataset
from models.blocks import *
from utils.processor import get_acc, train

large_phi = Phi(widths=[3, 32, 32], acts=[nn.ReLU(), nn.ReLU()], norms=["batch", "batch"])

large_rho = Rho(
    widths=[32, 32, 64, 5], acts=[nn.ReLU(), nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], norms=["batch", None, "batch"]
)


def create_scheduler(optimizer, trial):
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "CosineAnnealingLR", "ExponentialLR", "ReduceLROnPlateau"]
    )
    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 1, 100)
        gamma = trial.suggest_float("StepLR_gamma", 0.05, 0.4, log=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("ExponentialLR_gamma", 0.05, 0.4, log=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        gamma = trial.suggest_float("ReduceLROnPlateau_gamma", 0.05, 0.4, log=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=3)
    return scheduler


def objective(trial):
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    num_epochs = 200
    model = DeepSetsArchitecture(large_phi, large_rho, lambda x: torch.mean(x, dim=1)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = create_scheduler(optimizer, trial)
    criterion = torch.nn.MSELoss()

    validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs, patience=7)
    val_accuracy = get_acc(model, val_loader, device)
    test_accuracy = get_acc(model, test_loader, device)

    with open("./NAC_Deepsets_HPO_trials.txt", "a") as file:
        file.write(
            f"Large Model, Trial {trial.number}, Val Accuracy: {test_accuracy}, Val Accuracy: {test_accuracy}, Validation Loss: {validation_loss}, Hyperparams: {trial.params}\n"
        )

    return val_accuracy


if __name__ == "__main__":
    batch_size = 256
    device = torch.device("cuda:0")  # TODO: Change to fit anyones device
    batch_size = 4096
    num_workers = 8

    train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders(
        "jet_images_c8_minpt2_ptetaphi_robust_fast", batch_size, num_workers, prefetch_factor=True, pin_memory=True
    )
    print("Loaded Dataset...")

    n_trials = 100

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"lr": 0.0032, "scheduler": "ReduceLROnPlateau", "ReduceLROnPlateau_gamma": 0.1})
    study.optimize(objective, n_trials=n_trials)
