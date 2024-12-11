import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from data.BraggnnDataset import setup_data_loaders
from utils.utils import *


class NAC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.act1 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 32, kernel_size=1, stride=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.act2 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(4)
        self.act3 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
        self.norm3 = nn.BatchNorm2d(32)  # nn.LayerNorm((32, 7, 7))
        self.act4 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(32, 8, kernel_size=3, stride=1)
        self.norm4 = nn.BatchNorm2d(8)  # nn.LayerNorm((8, 5, 5))
        self.act5 = nn.LeakyReLU()
        self.conv7 = nn.Conv2d(8, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(576, 8)
        self.norm5 = nn.BatchNorm1d(8)  # nn.LayerNorm((8))
        self.act6 = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)
        self.act7 = nn.LeakyReLU()
        self.fc3 = nn.Linear(4, 4)
        self.norm6 = nn.BatchNorm1d(4)  # nn.LayerNorm((4))
        self.act8 = nn.LeakyReLU()
        self.fc4 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act1(x)
        x = self.conv3(x)
        x = self.norm1(x)
        x = self.act2(x)
        x = self.conv4(x)
        x = self.norm2(x)
        x = self.act3(x)
        x = self.conv5(x)
        x = self.norm3(x)
        x = self.act4(x)
        x = self.conv6(x)
        x = self.norm4(x)
        x = self.act5(x)
        x = self.conv7(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.norm5(x)
        x = self.act6(x)
        x = self.fc2(x)
        x = self.act7(x)
        x = self.fc3(x)
        x = self.norm6(x)
        x = self.act8(x)
        x = self.fc4(x)
        return x


# Helper function for pruning
def get_parameters_to_prune(model, bias=False):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            if bias and module.bias != None:
                parameters_to_prune.append((module, 'bias'))

    return tuple(parameters_to_prune)


def get_sparsities(model):
    sparsities = []
    zeros, total = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            layer_sparsity = torch.sum(module.weight_mask == 0).float() / module.weight_mask.numel()
            zeros += torch.sum(module.weight_mask == 0).float()
            total += module.weight_mask.numel()
            sparsities.append(layer_sparsity)

    print('Overall sparsity: ', zeros / total)
    return tuple(sparsities)


if __name__ == "__main__":
    device = torch.device('cuda:4')
    batch_size = 1024
    train_loader, val_loader, test_loader = setup_data_loaders(
        batch_size, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2
    )
    print('Loaded Dataset...')
    model = NAC().to(device)
    prune.global_unstructured(get_parameters_to_prune(model, bias=False), pruning_method=prune.L1Unstructured, amount=0)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00015, weight_decay=2.2e-9)  # chagned lr from .0015
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    print('Starting run...')
    for prune_iter in range(0, 7):
        print('Starting prune iter: ', prune_iter)
        validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, 300)
        # val_mean_dist = get_mean_dist(model, val_loader, device, psz=11)
        test_mean_dist = get_mean_dist(model, test_loader, device, psz=11)
        print('Test Mean Distance: ', test_mean_dist)

        sparsities = get_sparsities(model)
        print('Sparsity: ', sparsities)
        torch.save(model.state_dict(), 'models/pruned_unquantized_LeakyReLU_NAC_iter' + str(prune_iter) + '.pth')
        test_model = NAC().to(device)
        test_model.load_state_dict(torch.load('models/pruned_unquantized_LeakyReLU_NAC_iter' + str(prune_iter) + '.pth'))
        test_mean_dist = get_mean_dist(model, test_loader, device, psz=11)
        print('Test Mean Distance: ', test_mean_dist)
        prune.global_unstructured(
            get_parameters_to_prune(model, bias=False), pruning_method=prune.L1Unstructured, amount=0.2
        )

    for module, name in get_parameters_to_prune(model):
        prune.remove(module, name)

    torch.save(model.state_dict(), 'models/pruned_unquantized_LeakyReLU_NAC.pth')
