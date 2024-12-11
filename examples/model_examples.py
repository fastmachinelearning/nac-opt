import time

import optuna
import torch
import torch.nn as nn

from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
from models.blocks import *
from models.train_utils import *
from utils.bops import *

x = torch.randn((256, 1, 11, 11))
batch_size = 256

# High performing model found by search, Mean Distance: 0.203, BOPs: 17.3 Million, Param Count: 13922
Blocks = nn.Sequential(ConvBlock([16, 4, 2], [1, 3], [nn.GELU(), nn.GELU()], [None, 'batch'], img_size=9))
mlp = MLP(widths=[98, 64, 8, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', 'batch', 'layer', None])
model = CandidateArchitecture(Blocks, mlp, 16)
y = model(x)
conv_bops = calculate_convblock_bops(Blocks[0], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
mlp_bops = calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
total_bops = conv_bops + mlp_bops
print('Example Model w/ BOPs = ', total_bops, ', Conv Bops = ', conv_bops, ', MLP Bops =', mlp_bops)
print(model)

# OpenHLS Model
Blocks = nn.Sequential(
    ConvAttn(16, 8, norm=None, act=nn.ReLU()),
    ConvBlock([16, 8, 2], [3, 3], [nn.ReLU(), nn.ReLU()], [None, None], img_size=9),
)
mlp = MLP(widths=[50, 16, 8, 4, 2], acts=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()], norms=[None, None, None, None])
model = CandidateArchitecture(Blocks, mlp, 16)
y = model(x)
attn_bops = calculate_convattn_bops(Blocks[0], input_shape=[batch_size, 16, 9, 9], bit_width=32)
conv_bops = calculate_convblock_bops(Blocks[1], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
mlp_bops = calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
total_bops = attn_bops + conv_bops + mlp_bops
print(
    'OpenHLS Model w/ BOPs = ',
    total_bops,
    ', Attn Bops = ',
    attn_bops,
    ', Conv Bops = ',
    conv_bops,
    ', MLP Bops =',
    mlp_bops,
)
print(model)


# BraggNN model
Blocks = nn.Sequential(
    ConvAttn(64, 32, norm=None, act=nn.LeakyReLU(negative_slope=0.01)),
    ConvBlock(
        [64, 32, 8], [3, 3], [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)], [None, None], img_size=9
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
y = model(x)
attn_bops = calculate_convattn_bops(Blocks[0], input_shape=[batch_size, 16, 9, 9], bit_width=32)
conv_bops = calculate_convblock_bops(Blocks[1], sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
mlp_bops = calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=32, activation_bit_width=32)
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


# QUANTIZE THE NAC MODEL
# First initialize a normal model to calculate BOPs, then input the bits you will quantize to
bit_width = 8

Blocks = nn.Sequential(ConvBlock([16, 4, 2], [1, 3], [nn.GELU(), nn.GELU()], [None, 'batch'], img_size=9))
mlp = MLP(widths=[98, 64, 8, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', 'batch', 'layer', None])
model = CandidateArchitecture(Blocks, mlp, 16)

conv_bops = calculate_convblock_bops(
    Blocks[0], sparsity_dict=None, weight_bit_width=bit_width, activation_bit_width=bit_width
)
mlp_bops = calculate_mlpblock_bops(mlp, sparsity_dict=None, weight_bit_width=bit_width, activation_bit_width=bit_width)
total_bops = conv_bops + mlp_bops
print('Quantized Model w/ BOPs = ', total_bops, ', Conv Bops = ', conv_bops, ', MLP Bops =', mlp_bops)

# Now, initialize the quantized model
Blocks = nn.Sequential(QAT_ConvBlock([16, 4, 2], [1, 3], [nn.GELU(), nn.GELU()], [None, 'batch'], img_size=9))
mlp = QAT_MLP(
    widths=[98, 64, 8, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', 'batch', 'layer', None]
)
model = QAT_CandidateArchitecture(Blocks, mlp, 16, bit_width=bit_width)
y = model(x)
