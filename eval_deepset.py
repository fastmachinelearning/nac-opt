import torch
import torch.nn as nn

from data import DeepsetsDataset
from models.blocks import *
from utils.processor import evaluate_Deepsets, get_acc

aggregator = lambda x: torch.mean(x, dim=2)

phi = ConvPhi(widths=[3, 32, 32, 32], acts=[nn.ReLU(), nn.ReLU(), nn.ReLU()], norms=[None, None, None])

rho = Rho(widths=[32, 16, 5], acts=[nn.ReLU(), None], norms=[None, None])

deepsets_model = DeepSetsArchitecture(phi, rho, aggregator)


large_phi = ConvPhi(widths=[3, 32, 32], acts=[nn.ReLU(), nn.ReLU()], norms=["batch", "batch"])

large_rho = Rho(
    widths=[32, 32, 64, 5], acts=[nn.ReLU(), nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], norms=["batch", None, "batch"]
)

large_model = DeepSetsArchitecture(large_phi, large_rho, aggregator)

medium_phi = ConvPhi(widths=[3, 32, 16], acts=[nn.ReLU(), nn.ReLU()], norms=["batch", "batch"])

medium_rho = Rho(
    widths=[16, 64, 8, 32, 5],
    acts=[nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), nn.ReLU(), nn.ReLU()],
    norms=["batch", "batch", "batch", "batch"],
)

medium_model = DeepSetsArchitecture(medium_phi, medium_rho, aggregator)

small_phi = ConvPhi(widths=[3, 8, 8], acts=[nn.LeakyReLU(negative_slope=0.01), nn.ReLU()], norms=["batch", None])

small_rho = Rho(
    widths=[8, 16, 16, 5],
    acts=[nn.LeakyReLU(negative_slope=0.01), nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)],
    norms=["batch", "batch", None],
)

small_model = DeepSetsArchitecture(small_phi, small_rho, aggregator)

tiny_phi = ConvPhi(widths=[3, 16], acts=[nn.ReLU()], norms=["batch"])

tiny_rho = Rho(widths=[16, 8, 8, 4, 5], acts=[nn.ReLU(), None, nn.ReLU(), nn.ReLU()], norms=["batch", None, None, "batch"])

tiny_model = DeepSetsArchitecture(tiny_phi, tiny_rho, aggregator)


adjusted_tiny_rho = Rho(widths=[16, 8, 4, 5], acts=[nn.ReLU(), nn.ReLU(), nn.ReLU()], norms=["batch", None, "batch"])

adjusted_tiny_model = DeepSetsArchitecture(tiny_phi, adjusted_tiny_rho, aggregator)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    batch_size = 4096
    num_workers = 8

    train_loader, val_loader, test_loader = DeepsetsDataset.setup_data_loaders(
        "jet_images_c8_minpt2_ptetaphi_robust_fast", batch_size, num_workers, prefetch_factor=True, pin_memory=True
    )
    print("Loaded Dataset...")

    for model in [large_model, medium_model, small_model, tiny_model, adjusted_tiny_model]:
        print()
        print(model)
        val_accuracy, inference_time, validation_loss, param_count = evaluate_Deepsets(
            model, train_loader, val_loader, device, num_epochs=100
        )
        test_accuracy = get_acc(model, test_loader, device)
        print(
            f"Test Accuracy: {test_accuracy}, Val Accuracy: {val_accuracy}, Inference time: {inference_time}, Validation Loss: {validation_loss}, Param Count: {param_count}"
        )
