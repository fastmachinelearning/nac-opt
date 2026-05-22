"""Run combined QAT+pruning local search on an existing best architecture."""
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from utils.tf_local_search_combined import combined_local_search_entrypoint
from utils.tf_data_preprocessing import load_generic_dataset

ARCH_YAML = "results/planned_digits/best_model_for_local_search.yaml"
RESULTS_DIR = "results/planned_digits/local_search_combined"
DATASET_CSV = "data/digits_demo/digits.csv"

LOCAL_SEARCH_CONFIG = {
    "pruning_settings": {
        "iterations": 5,
        "epochs_per_iteration": 15,
        "pruning_rate": 0.6,
    },
    "qat_settings": {
        "epochs": 20,
        "precision_pairs": [
            {"total_bits": 8, "int_bits": 3},
            {"total_bits": 6, "int_bits": 2},
            {"total_bits": 4, "int_bits": 1},
        ],
    },
}

os.makedirs(RESULTS_DIR, exist_ok=True)
local_config_path = os.path.join(RESULTS_DIR, "local_search_config.yaml")
with open(local_config_path, "w") as f:
    yaml.dump(LOCAL_SEARCH_CONFIG, f)

x_train, y_train, x_val, y_val = load_generic_dataset(
    format="csv",
    data_path=DATASET_CSV,
    label_column="target",
    val_split=0.2,
    random_state=42,
    flatten=True,
    one_hot=True,
)

print(f"Dataset: train={x_train.shape}, val={x_val.shape}")
print(f"Architecture: {ARCH_YAML}")
print(f"Results -> {RESULTS_DIR}")
print()

combined_df = combined_local_search_entrypoint(
    architecture_yaml_path=ARCH_YAML,
    local_search_config_path=local_config_path,
    dataset=(x_train, y_train, x_val, y_val),
    results_dir=RESULTS_DIR,
    n_folds=1,
)

print("\n=== Combined local search complete ===")
print(combined_df.to_string(index=False))
