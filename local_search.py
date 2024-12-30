import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Union, Callable
from dataclasses import dataclass
from torch.utils.data import DataLoader
import brevitas.nn as qnn


from models.blocks import *


@dataclass
class SearchConfig:
    """Configuration for local search parameters"""
    num_prune_iterations: int = 20
    prune_amount: float = 0.2
    include_bias: bool = False
    log_file: str = "local_search_results.txt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LocalSearch:
    def __init__(self, config: SearchConfig):
        self.config = config
        
    @staticmethod
    def get_parameters_to_prune(model: nn.Module, bias: bool = False) -> tuple:
        """Get all parameters that can be pruned from the model"""
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, qnn.QuantLinear, qnn.QuantConv1d, qnn.QuantConv2d)):
                parameters_to_prune.append((module, "weight"))
                if bias and module.bias is not None:
                    parameters_to_prune.append((module, "bias"))
        return tuple(parameters_to_prune)

    @staticmethod
    def get_sparsities(model: nn.Module) -> tuple:
        """Calculate sparsity for each layer in the model"""
        sparsities = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, qnn.QuantLinear, qnn.QuantConv1d, qnn.QuantConv2d)):
                layer_sparsity = torch.sum(module.weight_mask == 0).float() / module.weight_mask.numel()
                sparsities.append(layer_sparsity)
        return tuple(sparsities)

    def search_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        evaluate_fn: Callable,
        model_name: str = "Model",
        extra_info: str = ""
    ) -> None:
        """
        Perform local search on a single model
        
        Args:
            model: The model to search
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            evaluate_fn: Function to train and evaluate the model
            model_name: Name identifier for the model
            extra_info: Additional information to log
        """
        model = model.to(self.config.device)
        # Initialize pruning
        prune.global_unstructured(
            self.get_parameters_to_prune(model, self.config.include_bias),
            pruning_method=prune.L1Unstructured,
            amount=0
        )

        for prune_iter in range(self.config.num_prune_iterations):
            
            # Train and evaluate model
            metrics = evaluate_fn(model, train_loader, val_loader, test_loader, self.config.device)
            sparsities = self.get_sparsities(model)

            print(f"Pruning Iter {prune_iter + 1}/{self.config.num_prune_iterations}")
            
            # Log results
            with open(self.config.log_file, "a") as file:
                log_str = f"{model_name} {extra_info} Prune Iter: {prune_iter}, "
                log_str += f"Metrics: {metrics}, Sparsities: {sparsities}\n"
                file.write(log_str)

            # Apply pruning
            if prune_iter < self.config.num_prune_iterations - 1:  # Don't prune on last iteration
                prune.global_unstructured(
                    self.get_parameters_to_prune(model, self.config.include_bias),
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.prune_amount
                )

    def search_multiple_models(
        self,
        models: List[Tuple[nn.Module, str]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        evaluate_fn: Callable,
        extra_info: str = ""
    ) -> None:
        """
        Perform local search on multiple models
        
        Args:
            models: List of (model, model_name) tuples
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            evaluate_fn: Function to evaluate model performance
            extra_info: Additional information to log
        """
        for model, model_name in models:
            print(f"Searching {model_name}...")
            self.search_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                evaluate_fn=evaluate_fn,
                model_name=model_name,
                extra_info=extra_info
            )




# Example usage for DeepSets
from utils.processor import evaluate_deepsets
from data.DeepsetsDataset import *
if __name__ == "__main__":
    # DeepSets Dataset Configuration
    batch_size = 4096
    num_workers = 8
    base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"

    # Load datasets
    train_loader, val_loader, test_loader = setup_data_loaders(
        base_file_name,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )
    print("Loaded Dataset...")

    # Search Configuration
    config = SearchConfig(
        num_prune_iterations=20,
        prune_amount=0.2,
        include_bias=False,
        log_file="Results/deepsets_search_results.txt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize search
    local_search = LocalSearch(config)

    # Define models (using QAT models as example)
    bit_width = 32
    aggregator = lambda x: torch.mean(x, dim=2)

    # Large model
    large_phi = QAT_ConvPhi(
        widths=[3, 32, 32], 
        acts=[nn.ReLU(), nn.ReLU()], 
        norms=["batch", "batch"], 
        bit_width=bit_width
    )
    large_rho = QAT_Rho(
        widths=[32, 32, 64, 5],
        acts=[nn.ReLU(), nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)],
        norms=["batch", None, "batch"],
        bit_width=bit_width
    )
    large_model = DeepSetsArchitecture(large_phi, large_rho, aggregator)

    # Small model
    small_phi = QAT_ConvPhi(
        widths=[3, 8, 8], 
        acts=[nn.LeakyReLU(negative_slope=0.01), nn.ReLU()], 
        norms=["batch", None], 
        bit_width=bit_width
    )
    small_rho = QAT_Rho(
        widths=[8, 16, 16, 5],
        acts=[nn.LeakyReLU(negative_slope=0.01), nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)],
        norms=["batch", "batch", None],
        bit_width=bit_width
    )
    small_model = DeepSetsArchitecture(small_phi, small_rho, aggregator)

    # Define models to search
    deepsets_models = [
        (large_model, "Large"),
        (small_model, "Small")
    ]

    # Run search on multiple models
    local_search.search_multiple_models(
        models=deepsets_models,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        evaluate_fn=evaluate_deepsets,
        extra_info=f"{bit_width}-Bit QAT"
    )


#Example Usage for BraggNN
"""
from data.BraggnnDataset import *
from utils.processor import evaluate_braggnn
from data.BraggnnDataset import setup_data_loaders

if __name__ == "__main__":
    # BraggNN Dataset Configuration
    batch_size = 4096
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    

    # Load datasets
    train_loader, val_loader, test_loader = setup_data_loaders(
        batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2, data_folder= "/home/users/ddemler/dima_stuff/Morph/data/"
    )
    print("Loaded Dataset...")

    config = SearchConfig(
        num_prune_iterations=20,
        prune_amount=0.2,
        include_bias=False,
        log_file="Results/bragg_search_results.txt",
        device=device, 
    )

    # Initialize search
    local_search = LocalSearch(config)

    # NAC Model
    b = 8  # Bit width
    Blocks = nn.Sequential(
        QAT_ConvBlock(
            [32, 4, 32], [1, 1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, "batch"], img_size=9, bit_width=b
        ),
        QAT_ConvBlock([32, 4, 32], [1, 3], [nn.GELU(), nn.GELU()], ["batch", "layer"], img_size=9, bit_width=b),
        QAT_ConvBlock([32, 8, 64], [3, 3], [nn.GELU(), None], ["layer", None], img_size=7, bit_width=b),
    )

    mlp = QAT_MLP(
        widths=[576, 8, 4, 4, 2],
        acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None],
        norms=["layer", None, "layer", None],
        bit_width=b,
    )

    braggnn_model = QAT_CandidateArchitecture(Blocks, mlp, 32).to(device)

    #initialize pruning
    local_search.search_single_model(
        model=braggnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        evaluate_fn=evaluate_braggnn,
        model_name="BraggNN",
        extra_info=f"{b}-Bit QAT"
    )
"""






