import os
import time
from datetime import datetime
from functools import partial

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


# Metric Functions
def get_mean_dist(model, dataloader, device, psz=11):
    distances = []
    with torch.no_grad():
        for features, true_locs in dataloader:
            features = features.to(device)
            preds = model(features)  # assuming model outputs normalized [px, py]
            preds = preds.cpu().numpy()

            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((preds - true_locs.numpy()) ** 2, axis=1)) * 11  # psz=11
            distances.extend(distance)  # Changed from append to extend

    mean_distance = np.mean(distances)
    return mean_distance


def get_param_count_BraggNN(model):
    count = 0
    count += sum(p.numel() for p in model.Blocks.parameters())
    count += sum(p.numel() for p in model.MLP.parameters())
    count += sum(p.numel() for p in model.conv.parameters())
    return count


def get_param_count_Deepsets(model):
    count = 0
    count += sum(p.numel() for p in model.phi.parameters())
    count += sum(p.numel() for p in model.rho.parameters())
    return count


def get_inference_time(model, device, img_size=(256, 1, 11, 11)):
    x = torch.randn(img_size).to(device)
    start = time.time()
    for _ in range(100):
        y = model(x)
    end = time.time()
    return end - start


def get_acc(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in dataloader:
            data = data.to(device).float()
            targets = targets.to(device).float()

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(targets, 1)  # Get the true class labels

            total += true_labels.size(0)
            correct += (predicted == true_labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy
