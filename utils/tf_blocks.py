import tensorflow as tf

def get_activation_tf(act_name: str) -> tf.keras.layers.Layer:
    """Convert activation function name to TensorFlow layer"""
    act_map = {
        "ReLU": tf.keras.layers.ReLU(),
        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
        "GELU": tf.keras.layers.Activation('gelu'),
        "Identity": tf.keras.layers.Activation('linear'),
    }
    return act_map[act_name]

def sample_MLP_tf(trial, in_dim, out_dim, prefix, search_space, num_layers=3):
    """Generic MLP sampling function using provided search space for TensorFlow"""
    mlp_width_space = search_space["mlp_width_space"]
    act_space = search_space["act_space"]
    norm_space = search_space["norm_space"]

    # Create widths list
    widths = [in_dim]
    for i in range(num_layers - 1):
        widths.append(mlp_width_space[trial.suggest_int(f"{prefix}_width_{i}", 0, len(mlp_width_space) - 1)])
    widths.append(out_dim)

    # Sample activations
    acts = []
    for i in range(num_layers):
        act_name = trial.suggest_categorical(f"{prefix}_acts_{i}", act_space)
        acts.append(get_activation_tf(act_name))

    # Sample normalizations
    norms = [trial.suggest_categorical(f"{prefix}_norms_{i}", norm_space)
             for i in range(num_layers)]

    # Create layers
    layers = []
    for i in range(len(acts)):
        layers.append(tf.keras.layers.Dense(widths[i+1]))
        if norms[i] == 'batch':
            layers.append(tf.keras.layers.BatchNormalization())
        elif norms[i] == 'layer':
            layers.append(tf.keras.layers.LayerNormalization())
        if acts[i] is not None and i < len(acts) - 1:  # Don't add activation after last layer
            layers.append(acts[i])

    return layers, acts, norms


class DeepSetsArchitecture_tf(tf.keras.Model):
    def __init__(self, phi, rho, aggregator):
        super().__init__()
        self.phi = phi  # Use the actual phi network
        self.rho = rho  # Use the actual rho network
        self.aggregator = aggregator

    def call(self, x, training=None):
        # Input x shape: (batch_size, num_particles, features) e.g., (None, 8, 3)
        batch_size = tf.shape(x)[0]
        num_particles = tf.shape(x)[1]
        features = tf.shape(x)[2]

        # Apply phi to each particle
        # Reshape x from (batch_size, num_particles, features) to (batch_size * num_particles, features)
        x_reshaped = tf.reshape(x, (-1, features))
        x_phi = self.phi(x_reshaped, training=training)  # Pass training flag for BatchNorm
        
        # Get the output dimension of phi
        phi_output_dim = tf.shape(x_phi)[-1]
        
        # Reshape x_phi back to (batch_size, num_particles, phi_output_dim)
        x_phi_reshaped = tf.reshape(x_phi, (batch_size, num_particles, phi_output_dim))

        # Apply aggregator (mean/max over num_particles dimension)
        x_agg = self.aggregator(x_phi_reshaped)  # Output: (batch_size, phi_output_dim)

        # Apply rho
        x_rho = self.rho(x_agg, training=training)  # Pass training flag for BatchNorm
        return x_rho



class ConvAttentionBlock(tf.keras.layers.Layer):
    """Custom Keras layer for convolutional self-attention."""
    def __init__(self, in_channels, hidden_channels, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.activation = activation

        # Define sub-layers in __init__
        self.q_conv = tf.keras.layers.Conv2D(hidden_channels, 1, name='query')
        self.k_conv = tf.keras.layers.Conv2D(hidden_channels, 1, name='key')
        self.v_conv = tf.keras.layers.Conv2D(hidden_channels, 1, name='value')
        self.proj_conv = tf.keras.layers.Conv2D(in_channels, 1, name='projection')

        if self.activation is not None:
            if isinstance(self.activation, str):
                self.act_layer = tf.keras.layers.Activation(self.activation)
            else:
                self.act_layer = self.activation
        else:
            self.act_layer = None

    def call(self, x):
        """Forward pass logic using TensorFlow and Keras operations."""
        # Get dynamic shape components
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        
        q = tf.reshape(q, [batch_size, height * width, self.hidden_channels])
        k = tf.reshape(k, [batch_size, height * width, self.hidden_channels])
        v = tf.reshape(v, [batch_size, height * width, self.hidden_channels])
        
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attended = tf.matmul(attention_weights, v)
        
        attended = tf.reshape(attended, [batch_size, height, width, self.hidden_channels])
        
        projected = self.proj_conv(attended)
        
        output = x + projected # Residual connection
        
        if self.act_layer is not None:
            output = self.act_layer(output)
            
        return output