"""
Utils package for Neural Architecture Codesign (NAC)
Provides utilities for both PyTorch and TensorFlow implementations.
"""

# PyTorch utilities
try:
    from .bops import *
    from .metrics import *
    from .processor import *
except ImportError:
    pass  # PyTorch not available

# TensorFlow utilities  
try:
    from .tf_bops import *
    from .tf_data_preprocessing import *
    from .tf_model_builder import *
    from .tf_processor import *
    from .tf_visualization import *
    from .tf_global_search import GlobalSearchTF, run_mlp_search, run_deepsets_search
except ImportError:
    pass  # TensorFlow not available

__all__ = [
    # PyTorch utilities
    'get_sparsity', 'get_linear_bops', 'get_conv2d_bops', 'get_MLP_bops',
    'get_mean_dist', 'get_param_count_BraggNN', 'get_param_count_Deepsets',
    'get_acc', 'get_inference_time', 'evaluate_BraggNN', 'evaluate_deepsets', 'train',
    
    # TensorFlow utilities
    'load_and_preprocess_mnist', 'create_tf_dataset', 'load_generic_dataset',
    'build_mlp_from_config', 'build_deepsets_model', 'load_yaml_config',
    'train_model', 'evaluate_model', 'get_model_metrics',
    'get_linear_bops_tf', 'get_MLP_bops_tf', 'get_model_bops_tf',
    'plot_pareto_fronts', 'plot_3d_pareto_front_heatmap',
    'GlobalSearchTF', 'run_mlp_search', 'run_deepsets_search',
]