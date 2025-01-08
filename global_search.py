# from data.BraggnnDataset import setup_data_loaders
import torch
import torch.nn as nn
import optuna

from models.blocks import *
from utils.bops import *
from utils.processor import evaluate_BraggNN, evaluate_deepsets
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


def load_configs(task="deepsets", config_dir="examples/"):
    """Load YAML configuration files based on specified task.
    
    Args:
        task (str): Task to load configs for. Either "deepsets" or "braggnn".
        config_dir (str): Directory containing config files.
        
    Returns:
        tuple: (task_configs, search_space) containing model configs and search space for specified task
        
    Raises:
        ValueError: If task is not "deepsets" or "braggnn"
    """
    if task not in ["deepsets", "braggnn"]:
        raise ValueError('Task must be either "deepsets" or "braggnn"')
        
    if task == "deepsets":
        with open(os.path.join(config_dir, "DeepSets/deepsets_search_space.yaml"), "r") as f:
            search_space = yaml.safe_load(f)
        
        with open(os.path.join(config_dir, "DeepSets/deepsets_model_example_configs.yaml"), "r") as f:
            task_configs = yaml.safe_load(f)
            
    else:  # task == "braggnn"
        with open(os.path.join(config_dir, "BraggNN/braggnn_search_space.yaml"), "r") as f:
            search_space = yaml.safe_load(f)
        
        with open(os.path.join(config_dir, "BraggNN/bragg_model_example_configs.yaml"), "r") as f:
            task_configs = yaml.safe_load(f)
    
    return task_configs, search_space


def BraggNN_objective(trial):
    """BraggNN objective using search space config"""
    task_configs, search_space = load_configs(task="braggnn")
    spaces = search_space["search_spaces"]
    hyper_params = search_space["hyperparameters"]
    
    num_blocks = hyper_params["num_blocks"]
    img_size = hyper_params["initial_img_size"]
    output_dim = hyper_params["output_dim"]
    
    # Sample first channel dimension
    block_channels = [spaces["channel_space"][
        trial.suggest_int("Proj_outchannel", 0, len(spaces["channel_space"]) - 1)
    ]]

    # Sample Block Types
    b = [trial.suggest_categorical(f"b{i}", spaces["block_types"]) 
         for i in range(num_blocks)]

    Blocks = []
    bops = 0

    # Build Blocks
    for i, block_type in enumerate(b):
        if block_type == "Conv":
            channels, kernels, acts, norms = sample_ConvBlock(
                trial, 
                f"b{i}_Conv", 
                block_channels[-1],
                search_space=spaces,  # Pass search space
                num_layers=2
            )
            
            reduce_img_size = 2 * sum([1 if k == 3 else 0 for k in kernels])
            while img_size - reduce_img_size <= 0:
                kernels[kernels.index(3)] = 1
                reduce_img_size = 2 * sum([1 if k == 3 else 0 for k in kernels])
            
            Blocks.append(ConvBlock(channels, kernels, acts, norms, img_size))

            bops += get_Conv_bops(Blocks[-1], input_shape=[batch_size, channels[0], img_size, img_size], bit_width=32)
            img_size -= reduce_img_size
            block_channels.append(channels[-1])

        elif block_type == "ConvAttn":
            hidden_channels, act = sample_ConvAttn(
                trial, 
                f"b{i}_ConvAttn",
                search_space=spaces  # Pass search space
            )
            Blocks.append(ConvAttn(block_channels[-1], hidden_channels, act))

            bops += get_ConvAttn_bops(
                Blocks[-1], 
                input_shape=[batch_size, block_channels[-1], img_size, img_size], 
                bit_width=32
            )

    # Build MLP
    in_dim = block_channels[-1] * img_size**2
    widths, acts, norms = sample_MLP(
        trial, 
        in_dim, 
        output_dim, 
        "MLP",
        search_space=spaces,  # Pass search space
        num_layers=3
    )
    mlp = MLP(widths, acts, norms)

    bops += get_MLP_bops(mlp, bit_width=32)

    # Initialize Model
    Blocks = nn.Sequential(*Blocks)
    model = CandidateArchitecture(Blocks, mlp, block_channels[0])
    bops += get_conv2d_bops(
        model.conv, 
        input_shape=[batch_size, 1, 11, 11], 
        bit_width=32
    )

    print(model)
    print("BOPs:", bops)
    print("Trial ", trial.number, " begins evaluation...")
    mean_distance, inference_time, validation_loss, param_count = evaluate_BraggNN(model, train_loader, val_loader, device)
    
    with open("./global_search.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Mean Distance: {mean_distance}, BOPs: {bops}, "
            f"Inference time: {inference_time}, Validation Loss: {validation_loss}, "
            f"Param Count: {param_count}, Hyperparams: {trial.params}\n"
        )
    return mean_distance, bops


def Deepsets_objective(trial):
    """DeepSets objective using search space config"""
    task_configs, search_space = load_configs(task="deepsets")
    spaces = search_space["search_spaces"]
    hyper_params = search_space["hyperparameters"]
    
    bops = 0
    in_dim, out_dim = 3, 5 #3 kinematic features input, 5 possible particle decay classes

    # Sample architecture parameters
    bottleneck_dim = 2 ** trial.suggest_int("bottleneck_dim", 
                                           *spaces["bottleneck_range"])

    aggregator_type = trial.suggest_categorical("aggregator_type", 
                                              spaces["aggregator_space"])
    aggregator = (lambda x: torch.mean(x, dim=2) if aggregator_type == "mean" 
                 else lambda x: torch.max(x, dim=2).values)
    
    if aggregator_type == "mean":
        bops += get_AvgPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)
    else:
        bops += get_MaxPool_bops(input_shape=(8, bottleneck_dim), bit_width=8)

    # Initialize networks
    phi_len = trial.suggest_int("phi_len", *hyper_params["phi_len_range"])
    phi_widths, phi_acts, phi_norms = sample_MLP(
        trial, 
        in_dim, 
        bottleneck_dim, 
        "phi_MLP", 
        search_space=spaces,
        num_layers=phi_len
    )
    phi = ConvPhi(phi_widths, phi_acts, phi_norms)
    bops += get_MLP_bops(phi, bit_width=8)

    rho_len = trial.suggest_int("rho_len", *hyper_params["rho_len_range"])
    rho_widths, rho_acts, rho_norms = sample_MLP(
        trial, 
        bottleneck_dim, 
        out_dim, 
        "rho_MLP", 
        search_space=spaces,
        num_layers=rho_len
    )
    rho = Rho(rho_widths, rho_acts, rho_norms)
    bops += get_MLP_bops(rho, bit_width=8)

    model = DeepSetsArchitecture(phi, rho, aggregator)

    print(model)
    print("BOPs:", bops)
    print("Trial ", trial.number, " begins evaluation...")
    
    metrics = evaluate_deepsets(model, train_loader, val_loader, test_loader, device)
    
    accuracy = metrics['val_accuracy']
    inference_time = metrics['inference_time']
    validation_loss = metrics['val_loss']
    param_count = metrics['param_count']

    with open("./global_search.txt", "a") as file:
        file.write(
            f"Trial {trial.number}, Accuracy: {accuracy}, BOPs: {bops}, "
            f"Inference time: {inference_time}, Validation Loss: {validation_loss}, "
            f"Param Count: {param_count}, Hyperparams: {trial.params}\n"
        )
    return accuracy, bops

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 4096
    num_workers = 4

    # BraggNN optimization
    if False:  # Change this flag to switch between tasks
        # BraggNN data setup
        braggnn_configs, braggnn_search_space = load_configs(task="braggnn")
        
        train_loader, val_loader, test_loader = setup_data_loaders_braggnn(
            batch_size, IMG_SIZE=11, aug=1, num_workers=4, 
            pin_memory=False, prefetch_factor=2, 
            data_folder="/home/users/ddemler/dima_stuff/Morph/data/"
        )
        
        study = optuna.create_study(
            sampler=optuna.samplers.NSGAIISampler(population_size=20),
            directions=['minimize', 'minimize']
        )

        # Queue example architectures from config
        study.enqueue_trial(braggnn_configs['openhls'])
        study.enqueue_trial(braggnn_configs['braggnn'])
        
        study.optimize(BraggNN_objective, n_trials=5)
        
    else:
        # Deepsets optimization
        deepsets_configs, deepsets_search_space = load_configs(task="deepsets")

        base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"
        
        train_loader, val_loader, test_loader = setup_data_loaders_deepsets(
            base_file_name,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            pin_memory=True
        )
        
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