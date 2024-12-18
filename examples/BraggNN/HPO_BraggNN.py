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
        T_max = trial.suggest_int("T_max", 10, 300)
        eta_min = trial.suggest_float("eta_min", 1e-5, 1e-2, log=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.1, 1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return scheduler


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.7, 0.999)

    # Initialize the model
    # BraggNN model
    Blocks = nn.Sequential(
        ConvAttn(64, 32, norm=None, act=nn.LeakyReLU(negative_slope=0.01)),
        ConvBlock(
            [64, 32, 8],
            [3, 3],
            [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)],
            [None, None],
            img_size=9,
        ),
    )
    mlp = MLP(
        widths=[200, 64, 32, 16, 8, 2],
        acts=[
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            None,
        ],
        norms=[None, None, None, None, None],
    )
    model = CandidateArchitecture(Blocks, mlp, 64).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, trial)
    criterion = torch.nn.MSELoss()

    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)
    mean_dist = get_performance(model, val_loader, device, psz=11)

    with open("./BraggNN_HPO_trials.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Mean Distance: {mean_dist}, Validation Loss: {validation_loss}, Hyperparams: {trial.params}\n"
        )
    return mean_dist


if __name__ == "__main__":
    batch_size = 256
    IMG_SIZE = 11
    aug = 1
    num_epochs = 300
    device = torch.device("cuda:3")
    print(device)
    train_loader, val_loader, test_loader = setup_data_loaders(
        batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2
    )
    print("Loaded Dataset...")
    n_trials = 100
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials)
