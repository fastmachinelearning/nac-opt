import math
import tensorflow as tf

def get_sparsity_tf(tensor):
    num_zeros = tf.math.count_nonzero(tensor == 0)
    total_params = tf.size(tensor)
    return num_zeros / total_params

def get_matmul_bops_tf(a, b, bit_width=32):
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

def get_linear_bops_tf(layer, bit_width=32, input_shape=None):
    """
    Calculate BOPs for a Dense (Linear) layer.
    
    Parameters:
        layer: tf.keras.layers.Dense layer
        bit_width: Bit width for calculations
        input_shape: Input shape tuple (batch_size, input_features)
        
    Returns:
        BOPs for this layer
    """
    if input_shape is None:
        input_shape = layer.input_shape if hasattr(layer, 'input_shape') else layer.input_spec.shape
    
    input_features = input_shape[-1]
    output_features = layer.units
    
    # BOPs = input_features * output_features * (bit_width^2 + 2*bit_width + log2(input_features))
    mult_bops = input_features * output_features * bit_width**2
    add_bops = input_features * output_features * 2 * bit_width
    log_bops = output_features * math.log2(max(input_features, 1)) * bit_width
    
    return mult_bops + add_bops + log_bops

def get_conv2d_bops_tf(layer, input_shape, bit_width=32):
    output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
    output_shape = (input_shape[0], layer.filters, output_spatial_dim, output_spatial_dim)

    input_numel = tf.reduce_prod(input_shape[1:])
    output_numel = tf.reduce_prod(output_shape[1:])

    return (
        output_numel
        * input_numel
        * layer.kernel_size[0] ** 2
        * (bit_width**2 + 2 * bit_width + math.log2(input_numel * layer.kernel_size[0] ** 2))
    )

def get_Conv_bops_tf(block, input_shape, bit_width=32):
    bops = 0
    for i, layer in enumerate(block.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            bops += get_conv2d_bops_tf(layer, input_shape, bit_width)

            # Update input_shape for future Conv2D layers
            output_spatial_dim = input_shape[-1] if layer.kernel_size == 1 else input_shape[-1] - 2
            input_shape = (input_shape[0], layer.filters, output_spatial_dim, output_spatial_dim)

    return bops

def get_ConvAttn_bops_tf(block, input_shape=(64, 1, 9, 9), bit_width=32):
    bops = 0

    # Add bops for each Wk, Wq, Wv, Proj
    qkv_layers = [block.Wk, block.Wq, block.Wv]
    for layer in qkv_layers:
        bops += get_conv2d_bops_tf(layer, input_shape, bit_width)

    hidden_shape = (input_shape[0], block.hidden_channels, input_shape[2], input_shape[3])
    bops += get_conv2d_bops_tf(block.proj, input_shape, bit_width)

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
    bops += get_matmul_bops_tf(Q_shape, K_shape, bit_width=32)

    # Add SV Matmul bops
    S_shape = (batch_size, seq_len, seq_len)  # S is the output scores from softmax
    V_shape = (batch_size, embed_dim, seq_len)
    bops += get_matmul_bops_tf(S_shape, V_shape, bit_width=32)

    return bops

def get_MLP_bops_tf(model, input_shape, bit_width=32):
    """
    Calculate BOPs for a TensorFlow MLP model.
    
    Parameters:
        model: TensorFlow/Keras model
        input_shape: Input shape tuple (batch_size, features)
        bit_width: Bit width for calculations
        
    Returns:
        Total BOPs for the model
    """
    bops = 0
    current_input_shape = input_shape
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            bops += get_linear_bops_tf(layer, bit_width, input_shape=current_input_shape)
            current_input_shape = (current_input_shape[0], layer.units)
        elif isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            # Normalization layers don't add significant BOPs
            pass
        elif hasattr(layer, 'output_shape') and layer.output_shape is not None:
            current_input_shape = layer.output_shape
    
    return bops

def get_model_bops_tf(model, input_shape, bit_width=32):
    """
    Generic function to calculate BOPs for any TensorFlow model.
    
    Parameters:
        model: TensorFlow/Keras model
        input_shape: Input shape tuple
        bit_width: Bit width for calculations
        
    Returns:
        Total BOPs for the model
    """
    bops = 0
    current_input_shape = input_shape
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            bops += get_linear_bops_tf(layer, bit_width, input_shape=current_input_shape)
            current_input_shape = (current_input_shape[0], layer.units)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            bops += get_conv2d_bops_tf(layer, current_input_shape, bit_width)
            # Update shape after conv2d
            if hasattr(layer, 'compute_output_shape'):
                current_input_shape = layer.compute_output_shape(current_input_shape)
        elif isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization)):
            # Normalization layers have minimal BOPs
            pass
        # Add more layer types as needed
        
    return bops

def get_AvgPool_bops_tf(input_shape, dim=1, bit_width=32):
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

    # memory access operations for writing the output tensor
    # write_ops = output_elements * math.log2(output_elements)

    total_bit_operations = sum_bit_operations + div_bit_operations + read_ops

    return total_bit_operations


def get_MaxPool_bops_tf(input_shape, dim=1, bit_width=32):

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