import os
import tensorflow as tf
import pandas as pd

# Import the unified global search utilities
from utils.tf_global_search4 import GlobalSearchTF, load_block_search_space

# --- Configuration ---
RESULTS_DIR = "./results/hybrid_search_experiments"
N_TRIALS = 10
EPOCHS = 20
SUBSET_SIZE = 50000

def run_hybrid_experiment():
    """Runs the hybrid architecture search experiment."""
    print("\n" + "="*60 + "\nRUNNING HYBRID (CONV + MLP) ARCHITECTURE SEARCH\n" + "="*60)
    try:
        searcher = GlobalSearchTF(results_dir=RESULTS_DIR)
        
        # --- Updated search space including 'MLP' as a block type ---
        searcher.search_space = {
            "channel_space": [4, 8, 16, 32],
            "mlp_width_space": [32, 64, 128],
            "kernel_space": [1, 3],
            "act_space": ["ReLU", "LeakyReLU", "GELU"],
            "norm_space": [None, "batch"],
            "block_types": ["Conv", "ConvAttn", "MLP", "None"],
            "conv_attn": {"hidden_channel_space": [4, 8, 16]},
            "num_blocks": 4,  # Total number of blocks in the feature extractor
            "initial_img_size": 28,
            "output_dim": 10
        }
        
        study = searcher.run_search(
            n_trials=N_TRIALS,
            epochs=EPOCHS,
            subset_size=SUBSET_SIZE,
            resize_val=28,
            verbose=True
        )
        
    except Exception as e:
        print(f"An error occurred during the experiment: {e}")

def main():
    """Main function to run the experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tf.get_logger().setLevel('ERROR')
    
    run_hybrid_experiment()
    
    print("\n" + "="*60)
    print("SCRIPT COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
