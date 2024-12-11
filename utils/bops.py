import math

import torch
import torch.nn as nn


# Returns sparsity calculated as percentage of zeros in a tensor
def get_sparsity(tensor):
    num_zeros = torch.sum(tensor == 0)
    total_params = tensor.numel()
    return num_zeros / total_params


"""
Calculate the bops in the matrix multiplication in attention
Let a,b be the shape of matrices A & B
For the QK matmul, we perform a dot product across the embedding dimension, for each row and column,
so there are seq_len^2 many dot products. Each dot product uses embed_dim many multiplications and
embed_dim - 1 additions.

DISCLAIMER: This only calculates bops for dense matmuls
"""


def get_matmul_bops(a, b, bit_width=32):
    if a[0] != b[0] or a[1] != b[2]:
        raise ValueError("Inner dimensions of arrays do not match for matrix multiplication.")
    batch_size = a[0]
    embed_dim = a[1]
    seq_len = a[2]

    bops_per_mult = bit_width**2
    bops_per_add = bit_width

    mult_bops = seq_len * seq_len * embed_dim * bops_per_mult
    add_bops = seq_len * seq_len * (embed_dim - 1) * bops_per_add
    bops = mult_bops + add_bops
    return bops


def get_linear_bops(layer, bit_width=32):
    sparsity = get_sparsity(layer.weight.data)
    return (
        layer.out_features
        * layer.in_features
        * ((1 - sparsity) * bit_width**2 + 2 * bit_width + math.log2(layer.in_features))
    )


def get_conv2d_bops(layer, input_shape, bit_width=32):
    output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
    output_shape = (input_shape[0], layer.out_channels, output_spatial_dim, output_spatial_dim)

    input_numel = torch.prod(torch.tensor(input_shape[1:]))
    output_numel = torch.prod(torch.tensor(output_shape[1:]))

    sparsity = get_sparsity(layer.weight.data)
    return (
        output_numel
        * input_numel
        * layer.kernel_size[0] ** 2
        * ((1 - sparsity) * bit_width**2 + 2 * bit_width + math.log2(input_numel * layer.kernel_size[0] ** 2))
    )


def get_Conv_bops(block, input_shape, bit_width=32):
    bops = 0
    for i, layer in enumerate(block.layers):
        if isinstance(layer, nn.Conv2d):
            sparsity = get_sparsity(layer.weight.data)
            bops += get_conv2d_bops(layer, input_shape, bit_width)

            # Update input_shape for future Conv2D layers
            output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
            input_shape = (input_shape[0], layer.out_channels, output_spatial_dim, output_spatial_dim)

    return bops


def get_ConvAttn_bops(block, input_shape=(64, 1, 9, 9), bit_width=32):
    bops = 0

    # Add bops for each Wk, Wq, Wv, Proj
    qkv_layers = [block.Wk, block.Wq, block.Wv]
    for layer in qkv_layers:
        bops += get_conv2d_bops(layer, input_shape, bit_width)

    hidden_shape = (input_shape[0], block.hidden_channels, input_shape[2], input_shape[3])
    bops += get_conv2d_bops(block.proj, input_shape, bit_width)

    # Get Input Shape and Reshaped dims
    batch_size, seq_len, h, w = input_shape
    embed_dim = h * w

    # Add softmax bops
    bops += (
        batch_size * embed_dim**2 * 1.5 * (bit_width - 1)
        + batch_size * embed_dim * (embed_dim - 1)
        + batch_size * (embed_dim) ** 2
    )

    # Add QK MatMul bops
    Q_shape = (batch_size, embed_dim, seq_len)
    K_shape = (batch_size, seq_len, embed_dim)
    bops += get_matmul_bops(Q_shape, K_shape, bit_width=32)

    # Add SV Matmul bops
    S_shape = (batch_size, seq_len, seq_len)  # S is the output scores from softmax
    V_shape = (batch_size, embed_dim, seq_len)
    bops += get_matmul_bops(S_shape, V_shape, bit_width=32)

    return bops


def get_MLP_bops(block, bit_width=32):
    bops = 0
    for i, layer in enumerate(block.layers):
        if isinstance(layer, nn.Linear):
            bops += get_linear_bops(layer, bit_width)
    return bops


def get_AvgPool_bops(input_shape, dim=1, bit_width=32):
    # number of elements in the dimension to be reduced
    num_elements_in_dim = input_shape[dim]

    # Calculate the number of elements in the output tensor
    output_elements = 1
    for i, d in enumerate(input_shape):
        if i != dim:
            output_elements *= d

    # bit operations for summing up the elements
    sum_bit_operations = (num_elements_in_dim - 1) * output_elements * bit_width  # similar to how we calculated sum

    div_bit_operations = (
        output_elements * math.log2(output_elements) * bit_width
    )  # Similar to how we previosuly calculated division

    # memory access operations for reading the input tensor
    input_elements = math.prod(input_shape)
    read_ops = input_elements * math.log2(input_elements)

    # memry access operations for writing the output tensor
    # write_ops = output_elements * math.log2(output_elements)

    total_bit_operations = sum_bit_operations + div_bit_operations + read_ops

    return total_bit_operations


def get_MaxPool_bops(input_shape, dim=1, bit_width=32):

    # number of elements in the dimension to be reduced
    num_elements_in_dim = input_shape[dim]

    # Calculate the number of elements in the output tensor
    output_elements = 1
    for i, d in enumerate(input_shape):
        if i != dim:
            output_elements *= d

    # max of an n-long tensor compares t[0] > t[1], max(t[0],t[1]) > t[2]... so n-1 comparisons. But we have num_elements_in_dim many tensors.
    num_comparisons = (output_elements - 1) * num_elements_in_dim

    # worst case time complexity is O(n) becuase you are iterating through all the bits to see which is larger.
    bops_per_comparison = bit_width

    input_elements = math.prod(input_shape)
    read_bops = input_elements * math.log2(input_elements)

    bops = num_comparisons * bops_per_comparison + read_bops
    return bops


# NOTE: BatchNorm, LayerNorm, etc. are extremely small relative to the Conv2D, linear, and matmul operations. We skip these as they are negligible.
