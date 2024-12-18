import brevitas.nn as qnn
import numpy as np
import torch
import torch.nn as nn
from brevitas.quant import IntBias
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.DeepsetsDataset import setup_data_loaders

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepSetsInv(nn.Module):
    def __init__(self, input_size, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "relu", weight_bit_width=8):
        super().__init__()
        self.nclasses = 5
        self.phi = nn.Sequential(
            qnn.QuantLinear(input_size, nnodes_phi, bias=True, weight_bit_width=weight_bit_width),
            self.get_activation(activ),
            qnn.QuantLinear(nnodes_phi, nnodes_phi, bias=True, weight_bit_width=weight_bit_width),
            self.get_activation(activ),
            qnn.QuantLinear(nnodes_phi, nnodes_phi, bias=True, weight_bit_width=weight_bit_width),
            self.get_activation(activ),
        )
        self.rho = nn.Sequential(
            qnn.QuantLinear(nnodes_phi, nnodes_rho, bias=True, weight_bit_width=weight_bit_width),
            self.get_activation(activ),
            qnn.QuantLinear(nnodes_rho, self.nclasses, bias=True, weight_bit_width=weight_bit_width),
        )

    def get_activation(self, activ):
        if activ == "relu":
            return qnn.QuantReLU(bit_width=8)
        elif activ == "sigmoid":
            return qnn.QuantSigmoid(bit_width=8)
        elif activ == "tanh":
            return qnn.QuantTanh(bit_width=8)
        else:
            raise ValueError(f"Unsupported activation: {activ}")

    def forward(self, inputs):
        phi_output = self.phi(inputs)
        # print("phi_output dtype:", phi_output.dtype)
        sum_output = torch.mean(phi_output, dim=1)
        # print( "sum_output dtype:", sum_output.dtype)
        rho_output = self.rho(sum_output)
        # print( "rho_output dtype:", rho_output.dtype)
        return rho_output


def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for test_data, test_targets in test_loader:
            test_data = test_data.to(device).float()
            test_targets = test_targets.to(device).float()

            outputs = model(test_data)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(test_targets, 1)  # Get the true class labels

            test_total += true_labels.size(0)
            test_correct += (predicted == true_labels).sum().item()

        test_accuracy = test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":

    # Create the model
    input_size = 3  # Assuming each input feature vector has a size of 3
    model = DeepSetsInv(input_size=input_size, nnodes_phi=32, nnodes_rho=16, activ="relu", weight_bit_width=32)
    print(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0032)  # lr=0.0032)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

    # Set up data loaders
    base_file_name = "jet_images_c8_minpt2_ptetaphi_robust_fast"
    batch_size = 1024
    num_workers = 8
    train_loader, val_loader, test_loader = setup_data_loaders(
        base_file_name, batch_size, num_workers, prefetch_factor=True, pin_memory=True
    )

    num_epochs = 100
    model.to(device)

    # Training loop
    best_val_accuracy = 0.0
    patience = 7
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        lossval = 0.0
        model.train()
        print(len(train_loader))
        for batch_data, batch_targets in train_loader:

            batch_data = batch_data.to(device).float()
            batch_targets = batch_targets.to(device).float()  # Convert targets to long tensor

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            lossval += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for val_data, val_targets in val_loader:
                val_data = val_data.to(device).float()
                # val_targets = val_targets.to(device).long()  # Convert targets to long tensor
                val_targets = val_targets.to(device).float()  # Convert targets to long tensor

                outputs = model(val_data)
                loss = criterion(outputs, val_targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                true_labels = torch.argmax(val_targets, 1)  # Get the true class labels

                val_total += val_targets.size(0)
                val_correct += (predicted == true_labels).sum().item()

            val_accuracy = val_correct / val_total
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print(f"Early stopping: Validation accuracy has not increased in {patience} epochs")
                    break

            # Update the learning rate based on validation accuracy
            lr_scheduler.step(val_accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {lossval/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    test_model(model, test_loader)
