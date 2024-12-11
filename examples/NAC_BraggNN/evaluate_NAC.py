import time

import numpy
import optuna
import torch
import torch.nn as nn

from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
from models.blocks import *
from models.train_utils import *
from utils.bops import *


def compute_distance_histogram(
    model,
    dataloader,
    psz=11,
    num_bins=100,
    range=(0, 2),
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
):
    model.to(device)
    model.eval()

    histogram = np.zeros(num_bins)

    with torch.no_grad():
        for features, true_locs in dataloader:
            features = features.to(device)
            preds = model(features).cpu().numpy()
            true_locs = true_locs.numpy()

            distances = np.sqrt(np.sum((preds - true_locs) ** 2, axis=1)) * psz
            hist, _ = np.histogram(distances, bins=num_bins, range=range)
            histogram += hist

    return histogram


def evaluate(model):
    num_epochs = 300
    device = torch.device('cuda:5')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0015, weight_decay=2.2e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)

    # Evaluate Performance
    val_mean_distance = get_performance(model, val_loader, device)
    test_mean_distance = get_performance(model, test_loader, device)
    # Evaluate Efficiency
    param_count = get_param_count(model)
    inference_time = get_inference_time(model, device)

    test_euclidean_distance = compute_distance_histogram(model, test_loader)

    print(
        'Test Mean Distance: ',
        test_mean_distance,
        'Val Mean Distance: ',
        val_mean_distance,
        ', Inference time: ',
        inference_time,
        ', Validation Loss: ',
        validation_loss,
        ', Param Count: ',
        param_count,
    )
    return test_euclidean_distance


if __name__ == "__main__":
    batch_size = 256
    train_loader, val_loader, test_loader = setup_data_loaders(
        batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2
    )
    print('Loaded Dataset...')

    # NAC Model
    Blocks = nn.Sequential(
        ConvBlock([32, 4, 32], [1, 1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, 'batch'], img_size=9),
        ConvBlock([32, 4, 32], [1, 3], [nn.GELU(), nn.GELU()], ['batch', 'layer'], img_size=9),
        ConvBlock([32, 8, 64], [3, 3], [nn.GELU(), None], ['layer', None], img_size=7),
    )
    mlp = MLP(widths=[576, 8, 4, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', None, 'layer', None])
    model = CandidateArchitecture(Blocks, mlp, 32)

    conv_bops = calculate_convblock_bops(Blocks[0], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
    conv_bops += calculate_convblock_bops(Blocks[1], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
    conv_bops += calculate_convblock_bops(Blocks[2], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
    mlp_bops = calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
    total_bops = conv_bops + mlp_bops
    print('NAC Model w/ BOPs = ', total_bops, ', Conv Bops = ', conv_bops, ', MLP Bops =', mlp_bops)
    print(model)
    print()
    print('Evaluating Model...')
    test_euclidean_distance = evaluate(model)

    with open("./model_evaluations.txt", "a") as file:
        file.write(f"NAC Euclidean Distances: {test_euclidean_distance}")
