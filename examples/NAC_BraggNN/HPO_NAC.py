import os
from datetime import datetime
from functools import partial

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
from models.blocks import *
from models.train_utils import *


def create_scheduler(optimizer, trial):
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR", "ExponentialLR"])
    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 1, 100)
        gamma = trial.suggest_float("gamma", 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_300)
    elif scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return scheduler


def objective(trial):
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-8, log=True)
    momentum = trial.suggest_float("momentum", 0.7, 0.999)

    # Initialize the model
    # Trial 164 Model
    Blocks = nn.Sequential(
        ConvBlock([32, 4, 32], [1, 1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, "batch"], img_size=9),
        ConvBlock([32, 4, 32], [1, 3], [nn.GELU(), nn.GELU()], ["batch", "layer"], img_size=9),
        ConvBlock([32, 8, 64], [3, 3], [nn.GELU(), None], ["layer", None], img_size=7),
    )
    mlp = MLP(widths=[576, 8, 4, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=["layer", None, "layer", None])
    model = CandidateArchitecture(Blocks, mlp, 32).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, trial)
    criterion = torch.nn.MSELoss()

    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
    mean_dist = get_performance(model, val_loader, device, psz=11)

    with open("./NAC_HPO_trials.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Mean Distance: {mean_dist}, Validation Loss: {validation_loss}, Hyperparams: {trial.params}\n"
        )

    return mean_dist


if __name__ == "__main__":
    batch_size = 256
    IMG_SIZE = 11
    aug = 1
    num_epochs = 300
    device = torch.device("cuda:2")
    print(device)
    train_loader, val_loader, test_loader = setup_data_loaders(
        batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2
    )
    print("Loaded Dataset...")
    n_trials = 100
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)
