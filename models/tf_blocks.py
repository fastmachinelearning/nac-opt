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