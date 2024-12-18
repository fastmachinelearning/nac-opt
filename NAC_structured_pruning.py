import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from NAC import NAC, get_parameters_to_prune

from data.BraggnnDataset import setup_data_loaders
from utils.utils import *

device = torch.device("cuda:4")


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


train_loader, val_loader, test_loader = setup_data_loaders(
    256, IMG_SIZE=11, aug=1, num_workers=4, pin_memory=False, prefetch_factor=2
)
model = NAC()
prune.global_unstructured(get_parameters_to_prune(model, bias=False), pruning_method=prune.L1Unstructured, amount=0)
model.load_state_dict(torch.load("/home/ubuntu/luke/Morph/models/d0.pth"))
prune.global_unstructured(get_parameters_to_prune(model, bias=False), pruning_method=prune.L1Unstructured, amount=0)
for module, name in get_parameters_to_prune(model, bias=False):
    prune.remove(module, name)
model.to(device)

val_mean_dist = get_mean_dist(model, val_loader, device, psz=11)
print(val_mean_dist)
test_mean_dist = get_mean_dist(model, test_loader, device, psz=11)
print(test_mean_dist)
torch.save(model.state_dict(), "/home/ubuntu/luke/Morph/models/removed_iter0.pth")

"""
model = model.to('cpu')
structured_pairs = [
    (model.conv1, model.conv2),
    (model.conv2, model.conv3),
    (model.conv3, model.conv4),
    (model.conv4, model.conv5),
    (model.conv5, model.conv6),
    (model.conv6, model.conv7),
    (model.conv7, model.fc1),
    (model.fc1, model.fc2),
    (model.fc2, model.fc3),
    (model.fc3, model.fc4)
]


for a, b in structured_pairs[:6]:
    assert a.out_channels == b.in_channels
    #get indices of zeroed-outgoing dim for a, dim=0
    outgoing_zeros = torch.where(a.weight.data.sum(dim=(1,2,3)) == 0)[0]
    incoming_zeros = torch.where(b.weight.data.sum(dim=(0,2,3)) == 0)[0]
    pruned_indices = torch.cat((outgoing_zeros, incoming_zeros)).unique()
    print(outgoing_zeros, incoming_zeros, pruned_indices)
    new_a = nn.Conv2d(a.in_channels, a.out_channels - len(pruned_indices), kernel_size=a.kernel_size, stride=a.stride)
    new_b = nn.Conv2d(b.in_channels - len(pruned_indices), b.out_channels, kernel_size=b.kernel_size, stride=b.stride)
    #copy weights from a to a'
    new_a.weight.data = a.weight.data[~pruned_indices]
    #copy weights from b to b'
    new_b.weight.data = b.weight.data[:, ~pruned_indices]
    a = new_a
    b = new_b

pruned_indices = torch.tensor([ 2, 10, 15, 18, 21, 22, 26, 28])
nonpruned_indices = torch.tensor([0,1,3,4,5,6,7,8,9,11,12,13,14,16,17,19,20,23,24,25,27,29,30,31])
print('conv1 bias', model.conv1.bias.data)
new_conv1 = nn.Conv2d(1,32-len(pruned_indices), kernel_size=3, stride=1)
new_conv2 = nn.Conv2d(32-len(pruned_indices),4, kernel_size=1, stride=1)
new_conv1.weight.data = model.conv1.weight.data[nonpruned_indices]
new_conv1.bias.data = model.conv1.bias.data[nonpruned_indices]
new_conv2.weight.data = model.conv2.weight.data[:, nonpruned_indices]
model.conv1 = new_conv1
model.conv2 = new_conv2


pruned_indices = torch.tensor([ 3,  4,  5, 16])
nonpruned_indices = torch.tensor([ 0,  1,  2,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
print('conv3 bias', model.conv3.bias.data)
new_conv3 = nn.Conv2d(4, 32-len(pruned_indices), kernel_size=1, stride=1)
new_conv4 = nn.Conv2d(32-len(pruned_indices), 4, kernel_size=1, stride=1)
new_conv3.weight.data = model.conv3.weight.data[nonpruned_indices]
new_conv3.bias.data = model.conv3.bias.data[nonpruned_indices]
new_conv4.weight.data = model.conv4.weight.data[:, nonpruned_indices]
model.conv3 = new_conv3
model.conv4 = new_conv4

model.norm1 = nn.BatchNorm2d(32-len(pruned_indices))

model = model.to(device)


print('After: ', get_mean_dist(model, test_loader, device, psz=11))
for i in range(10):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=.0015, weight_decay=2.2e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    validation_loss = train(model, optimizer, scheduler, criterion, train_loader, val_loader, device, 1)
    print('After finetuning ', i, 'epochs : ', get_mean_dist(model, test_loader, device, psz=11))
print(model)
"""
