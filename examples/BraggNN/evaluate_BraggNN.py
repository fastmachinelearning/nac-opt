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
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
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
    device = torch.device('cuda:4')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0015, weight_decay=2.2e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    validation_loss = train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, device, num_epochs)

    # Evaluate Performance
    val_mean_distance = get_performance(model, val_loader, device)
    test_mean_distance = get_performance(model, test_loader, device)
    # Evaluate Efficiency
    param_count = get_param_count(model)
    inference_time = get_inference_time(model, device)

    # TODO: Get euclidean distance plots
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
    # train_loader, val_loader, test_loader = setup_data_loaders(batch_size, IMG_SIZE = 11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2)
    print('Loaded Dataset...')

    # Create BraggNN model
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
    model = CandidateArchitecture(Blocks, mlp, 64)
    print(model)

    # Calculate BOPS
    patch_embed_bops = get_conv2d_bops(model.conv, input_shape=[batch_size, 1, 11, 11], bit_width=32)
    attn_bops = get_convattn_bops(Blocks[0], input_shape=[batch_size, 16, 9, 9], bit_width=32)
    conv_bops = get_conv_bops(Blocks[1], input_shape=[batch_size, 16, 9, 9], bit_width=32)
    mlp_bops = get_mlp_bops(mlp, bit_width=32)

    total_bops = attn_bops + conv_bops + mlp_bops
    print(
        'BraggNN Model w/ BOPs = ',
        total_bops,
        ', Attn Bops = ',
        attn_bops,
        ', Conv Bops = ',
        conv_bops,
        ', MLP Bops =',
        mlp_bops,
    )
    print(model)
    print()
    print('Evaluating Model...')
    test_euclidean_distance = evaluate(model)

    with open("./model_evaluations.txt", "a") as file:
        file.write(f"BraggNN Euclidean Distances: {test_euclidean_distance}")
