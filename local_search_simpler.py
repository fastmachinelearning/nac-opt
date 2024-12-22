import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Union, Callable
from dataclasses import dataclass
from torch.utils.data import DataLoader

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
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, "weight"))
                if bias and module.bias is not None:
                    parameters_to_prune.append((module, "bias"))
        return tuple(parameters_to_prune)

    @staticmethod
    def get_sparsities(model: nn.Module) -> tuple:
        """Calculate sparsity for each layer in the model"""
        sparsities = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
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
            evaluate_fn: Function to evaluate model performance
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
            # Evaluate model
            metrics = evaluate_fn(model, train_loader, val_loader, test_loader, self.config.device)
            sparsities = self.get_sparsities(model)
            
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
            self.search_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                evaluate_fn=evaluate_fn,
                model_name=model_name,
                extra_info=extra_info
            )

# Example evaluation functions for different model types
# def evaluate_braggnn(model, train_loader, val_loader, test_loader, device):
#     """Evaluation function for BraggNN models"""
#     val_mean_dist = get_mean_dist(model, val_loader, device, psz=11)
#     test_mean_dist = get_mean_dist(model, test_loader, device, psz=11)
#     validation_loss = train(model, optimizer, scheduler, criterion, 
#                           train_loader, val_loader, device, num_epochs)
#     return {
#         "val_mean_dist": val_mean_dist,
#         "test_mean_dist": test_mean_dist,
#         "val_loss": validation_loss
#     }

def evaluate_deepsets(model, train_loader, val_loader, test_loader, device):
    """Evaluatio function for DeepSets models"""
    val_accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(
        model, train_loader, val_loader, device, num_epochs=100
    )
    test_accuracy = get_acc(model, test_loader, device)
    return {
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "val_loss": validation_loss,
        "inference_time": inference_time,
        "param_count": param_count
    }



# from data import DeepsetsDataset
from models.blocks import *
from utils.processor import evaluate_Deepsets, get_acc
from data.DeepsetsDataset import *


# Example usage for DeepSets
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
        log_file="deepsets_search_results.txt",
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


# # Example usage
# if __name__ == "__main__":
#     # Configuration
#     config = SearchConfig(
#         num_prune_iterations=20,
#         prune_amount=0.2,
#         include_bias=False,
#         log_file="search_results.txt"
#     )
    
#     # Initialize search
#     local_search = LocalSearch(config)
    
#     # # BraggNN example (single model)
#     # bragg_model = QAT_CandidateArchitecture(Blocks, mlp, 32)
#     # local_search.search_single_model(
#     #     model=bragg_model,
#     #     train_loader=bragg_train_loader,
#     #     val_loader=bragg_val_loader,
#     #     test_loader=bragg_test_loader,
#     #     evaluate_fn=evaluate_braggnn,
#     #     model_name="BraggNN",
#     #     extra_info="8-Bit QAT"
#     # )
    
#     # DeepSets example (multiple models)
#     deepsets_models = [
#         (large_model, "Large"),
#         (medium_model, "Medium"),
#         (small_model, "Small"),
#         (tiny_model, "Tiny")
#     ]
#     local_search.search_multiple_models(
#         models=deepsets_models,
#         train_loader=deepsets_train_loader,
#         val_loader=deepsets_val_loader,
#         test_loader=deepsets_test_loader,
#         evaluate_fn=evaluate_deepsets,
#         extra_info="32-Bit QAT"
#     )