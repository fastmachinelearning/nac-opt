# from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna

from models.blocks import *
from utils.bops import *
from utils.processor import evaluate_BraggNN, evaluate_Deepsets
import yaml
import os

from data.BraggnnDataset import *
from data.DeepsetsDataset import *


"""
Optuna Objective to evaluate a trial
1) Samples architecture from hierarchical search space
2) Trains Model
3) Evaluates Mean Distance, bops, param count, inference time, and val loss
Saves all information in global_search.txt
"""


def load_configs(config_dir="examples/"):
    """Load YAML configuration files."""

    
    with open(os.path.join(config_dir, "BraggNN/braggnn_model_example_configs.yaml"), "r") as f:
        braggnn_configs = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, "DeepSets/deepsets_model_example_configs.yaml"), "r") as f:
        deepsets_configs = yaml.safe_load(f)
    
    return braggnn_configs, deepsets_configs

def BraggNN_objective(trial):
    # Build Model
    num_blocks = 3
    channel_space = (8, 16, 32, 64)
    block_channels = [
        channel_space[trial.suggest_int("Proj_outchannel", 0, len(channel_space) - 1)]
    ]  # sample the first channel dimension, save future dimensions here

    # Sample Block Types
    b = [trial.suggest_categorical("b" + str(i), ["Conv", "ConvAttn", "None"]) for i in range(num_blocks)]

    Blocks = []  # Save list of blocks
    img_size = 9  # Size after first conv patch embedding
    bops = 0  # Record Estimated BOPs

    # Build Blocks
    for i, block_type in enumerate(b):
        if block_type == "Conv":
            # Create block and add to Blocks
            channels, kernels, acts, norms = sample_ConvBlock(trial, "b" + str(i) + "_Conv", block_channels[-1])
            reduce_img_size = 2 * sum(
                [1 if k == 3 else 0 for k in kernels]
            )  # amount the image size will be reduced by kernel size, assuming no padding
            while img_size - reduce_img_size <= 0:
                kernels[kernels.index(3)] = 1
                reduce_img_size = 2 * sum([1 if k == 3 else 0 for k in kernels])
            Blocks.append(ConvBlock(channels, kernels, acts, norms, img_size))

            # Calculate bops for this block
            bops += get_Conv_bops(Blocks[-1], input_shape=[batch_size, channels[0], img_size, img_size], bit_width=32)
            img_size -= reduce_img_size
            block_channels.append(channels[-1])  # save the final out dimension so next block knows what to expect

        elif block_type == "ConvAttn":
            # Create block and add to Blocks
            hidden_channels, act = sample_ConvAttn(trial, "b" + str(i) + "_ConvAttn")
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels, act))

            # Calculate bops for this block
            bops += get_ConvAttn_bops(
                Blocks[-1], input_shape=[batch_size, block_channels[-1], img_size, img_size], bit_width=32
            )
            # Note: ConvAttn does not change the input shape because we use a skip connection

    # Build MLP
    in_dim = block_channels[-1] * img_size**2  # this assumes spatial dim stays same with padding trick
    widths, acts, norms = sample_MLP(trial, in_dim)
    mlp = MLP(widths, acts, norms)

    # Calculate bops for the mlp
    bops += get_MLP_bops(mlp, bit_width=32)

    # Initialize Model
    Blocks = nn.Sequential(*Blocks)
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])
    bops += get_conv2d_bops(
        model.conv, input_shape=[batch_size, 1, 11, 11], bit_width=32
    )  # Calculate bops for the patch embedding

    # Evaluate Model
    print(model)
    print("BOPs:", bops)
    print("Trial ", trial.number, " begins evaluation...")
    mean_distance, inference_time, validation_loss, param_count = evaluate_BraggNN(model, train_loader, val_loader, device)
    with open("./global_search.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Mean Distance: {mean_distance}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n"
        )
    return mean_distance, bops


def Deepsets_objective(trial):
    bops = 0
    in_dim, out_dim = 3, 5

    bottleneck_dim = 2 ** trial.suggest_int("bottleneck_dim", 0, 6)

    aggregator_space = [lambda x: torch.mean(x, dim=1), lambda x: torch.max(x, dim=1).values]
    aggregator_type = trial.suggest_int("aggregator_type", 0, 1)
    if aggregator_type == 0:
        bops += get_AvgPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)
    else:
        bops += get_MaxPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)
    aggregator = aggregator_space[aggregator_type]

    # Initialize Phi (first MLP)
    phi_len = trial.suggest_int("phi_len", 1, 4)
    widths, acts, norms = sample_MLP(trial, in_dim, bottleneck_dim, "phi_MLP", num_layers=phi_len)
    phi = Phi(widths, acts, norms)  # QAT_Phi(widths, acts, norms)
    bops += get_MLP_bops(phi, bit_width=8)

    # Initialize Rho (second MLP)
    rho_len = trial.suggest_int("rho_len", 1, 4)
    widths, acts, norms = sample_MLP(trial, bottleneck_dim, out_dim, "rho_MLP", num_layers=rho_len)
    rho = Rho(widths, acts, norms)  # QAT_Rho(widths, acts, norms)
    bops += get_MLP_bops(rho, bit_width=8)

    model = DeepSetsArchitecture(phi, rho, aggregator)

    print(model)
    print("BOPs:", bops)
    print("Trial ", trial.number, " begins evaluation...")
    accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(model, train_loader, val_loader, device)
    with open("./global_search.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Accuracy: {accuracy}, BOPs: {bops}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}, Hyperparams: {trial.params}\n"
        )
    return accuracy, bops


if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 4096  # 1024
    num_workers = 8

    # Load configurations
    braggnn_configs, deepsets_configs = load_configs()

    # train_loader, val_loader, test_loader = BraggNNDataset.setup_data_loaders(
    #     batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2, data_folder= "data/")
    base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"

    train_loader, val_loader, test_loader = setup_data_loaders(
        base_file_name,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )
    print("Loaded Dataset...")

    # For BraggNN optimization
    """
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
    
    study.optimize(BraggNN_objective, n_trials=1000)
    """

    # For DeepSets optimization
    study = optuna.create_study(
        sampler=optuna.samplers.NSGAIISampler(population_size=20),
        directions=["maximize", "minimize"]
    )

    # Queue example architectures from config
    study.enqueue_trial(deepsets_configs['base'])
    study.enqueue_trial(deepsets_configs['large'])
    study.enqueue_trial(deepsets_configs['medium'])
    study.enqueue_trial(deepsets_configs['small'])
    study.enqueue_trial(deepsets_configs['tiny'])

    study.optimize(Deepsets_objective, n_trials=1000)