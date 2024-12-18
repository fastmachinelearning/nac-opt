import time

import optuna
import torch
import torch.nn as nn

from data import BraggnnDataset
from data.BraggnnDataset import setup_data_loaders
from models.blocks import *
from models.train_utils import *
from utils.bops import *

# Replace 'your_file.txt' with the path to your text file
txt_file = "NAC_Compress.txt"

Blocks = nn.Sequential(
    ConvBlock([32, 4, 32], [1, 1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, "batch"], img_size=9),
    ConvBlock([32, 4, 32], [1, 3], [nn.GELU(), nn.GELU()], ["batch", "layer"], img_size=9),
    ConvBlock([32, 8, 64], [3, 3], [nn.GELU(), None], ["layer", None], img_size=7),
)
mlp = MLP(widths=[576, 8, 4, 4, 2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=["layer", None, "layer", None])
model = CandidateArchitecture(Blocks, mlp, 32)

bops_list = []
dist_list = []
bits = []

with open(txt_file) as f:
    lines = f.readlines()
    for line in lines:
        arr_line = line.split(" ")
        bit_width = float(arr_line[2][0])
        prune_iter = float(arr_line[7][:-1])
        dist = float(arr_line[11].replace(",", ""))

        conv_sparsity = float(arr_line[20][8:-1])
        block1_sparsity = {"layers[0]": float(arr_line[22][7:-1]), "layers[2]": float(arr_line[24][7:-1])}
        block2_sparsity = {"layers[0]": float(arr_line[26][7:-1]), "layers[2]": float(arr_line[28][7:-1])}
        block3_sparsity = {"layers[0]": float(arr_line[30][7:-1]), "layers[2]": float(arr_line[32][7:-1])}
        mlp_sparsity = {
            "layers[0]": float(arr_line[34][7:-1]),
            "layers[3]": float(arr_line[36][7:-1]),
            "layers[5]": float(arr_line[38][7:-1]),
            "layers[8]": float(arr_line[40][7:-1]),
        }

        conv_bops = calculate_convblock_bops(
            Blocks[0], sparsity_dict=block1_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width
        )
        conv_bops += calculate_convblock_bops(
            Blocks[1], sparsity_dict=block2_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width
        )
        conv_bops += calculate_convblock_bops(
            Blocks[2], sparsity_dict=block3_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width
        )
        mlp_bops = calculate_mlpblock_bops(
            mlp, sparsity_dict=mlp_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width
        )
        bops = conv_bops + mlp_bops

        bops_list.append(bops)
        dist_list.append(dist)
        bits.append(bit_width)


print(bops_list)
print(dist_list)
print(bits)

"""
Blocks = nn.Sequential(
    ConvBlock([32,4,32], [1,1], [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)], [None, 'batch'],img_size=9),
    ConvBlock([32,4,32], [1,3], [nn.GELU(), nn.GELU()], ['batch', 'layer'],img_size=9),
    ConvBlock([32,8,64], [3,3], [nn.GELU(), None], ['layer', None],img_size=7),
)
mlp = MLP(widths=[576,8,4,4,2], acts=[nn.ReLU(), nn.GELU(), nn.GELU(), None], norms=['layer', None, 'layer', None])
model = CandidateArchitecture(Blocks,mlp,32)

for bit_width in [32,16,8,7,6,5,4]:
    conv_sparsity = 0.7917
    block1_sparsity = {'layers[0]':0.8750, 'layers[2]':0.5703}
    block2_sparsity = {'layers[0]':0.3594, 'layers[2]':0.2865}
    block3_sparsity = {'layers[0]':0.4249, 'layers[2]':0.9544}
    mlp_sparsity = {'layers[0]':0.9562,'layers[3]':0.2812,'layers[5]':0.1875,'layers[8]':0.7500}
    conv_bops=calculate_convblock_bops(Blocks[0], sparsity_dict=block1_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width)
    conv_bops+=calculate_convblock_bops(Blocks[1], sparsity_dict=block2_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width)
    conv_bops+=calculate_convblock_bops(Blocks[2], sparsity_dict=block3_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width)
    mlp_bops=calculate_mlpblock_bops(mlp, sparsity_dict=mlp_sparsity, weight_bit_width=bit_width, activation_bit_width=bit_width)
    total_bops=conv_bops+mlp_bops
    print('Quantized Model ', bit_width,'-bit BOPs = ',total_bops,', Conv Bops = ', conv_bops,', MLP Bops =', mlp_bops)

"""
