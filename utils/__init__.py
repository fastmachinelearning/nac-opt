"""
Utils package for SNAC-Pack (TensorFlow implementation).
"""

# TensorFlow utilities
try:
    from .tf_bops import *
    from .tf_data_preprocessing import *
    from .tf_model_builder import *
    from .tf_processor import *
    from .tf_visualization import *
    from .tf_global_search import GlobalSearchTF, run_mlp_search
    from .tf_local_search_combined import combined_local_search_entrypoint
except ImportError:
    pass  # TensorFlow not available

__all__ = [
    # TensorFlow utilities
    'load_and_preprocess_mnist', 'create_tf_dataset', 'load_generic_dataset',
    'build_mlp_from_config', 'build_deepsets_model', 'load_yaml_config',
    'train_model', 'evaluate_model', 'get_model_metrics',
    'get_linear_bops_tf', 'get_MLP_bops_tf', 'get_model_bops_tf',
    'plot_pareto_fronts', 'plot_3d_pareto_front_heatmap',
    'GlobalSearchTF', 'run_mlp_search',
    'combined_local_search_entrypoint',
]
