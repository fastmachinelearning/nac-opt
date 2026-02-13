"""
Example code to add animated visualization to global search in the notebook.

Copy this code into cells in your notebook:
1. Add plotly imports to your imports cell
2. Create the callback before running search
3. Pass callback to run_search
4. Display animated figure after search completes
"""

# ============================================================================
# STEP 1: Add to your imports cell (add plotly imports)
# ============================================================================
"""
import plotly.graph_objects as go
import plotly.express as px
from utils.animated_search_viz import AnimatedSearchCallback
"""

# ============================================================================
# STEP 2: Create callback and run search in the SAME cell
# The figure will display automatically and update in real-time!
# ============================================================================
"""
print("\\n" + "=" * 50)
print("Running Qubit Block-Based Hardware-Aware Global Search...")
print("This may take a few minutes...")
print("=" * 50)

# Setup animated visualization callback - figure displays automatically
# For interactive 3D Plotly plot that updates live:
search_callback = AnimatedSearchCallback(
    OBJECTIVE_NAMES, 
    MAXIMIZE_FLAGS,
    use_3d=True,  # Use interactive Plotly 3D plot
    use_matplotlib_3d=False,  # Use Plotly, not matplotlib
    x_axis_3d='avg_resource',  # X-axis: avg_resource
    y_axis_3d='clock_cycles',  # Y-axis: clock_cycles
    z_axis_3d='performance_metric',  # Z-axis: performance_metric
    color_by_3d='bops',  # Color: bops
    use_size=False  # Fixed size markers
)

# Alternative: 2D plot with 3 metrics (x, y, color)
# search_callback = AnimatedSearchCallback(
#     OBJECTIVE_NAMES, 
#     MAXIMIZE_FLAGS,
#     use_3d=False,  # Use 2D plot
#     x_axis='avg_resource',  # X-axis: avg_resource
#     y_axis='clock_cycles',  # Y-axis: clock_cycles
#     color_by='performance_metric',  # Color: performance_metric (3rd dimension)
# )

# Alternative: 2D with different axes
# search_callback = AnimatedSearchCallback(
#     OBJECTIVE_NAMES, 
#     MAXIMIZE_FLAGS,
#     x_axis='bops',  # X-axis: BOPs
#     y_axis='avg_resource',  # Y-axis: avg_resource
#     color_by='performance_metric',  # Color: performance_metric
# )

# For 2D plot (alternative):
# search_callback = AnimatedSearchCallback(
#     OBJECTIVE_NAMES, 
#     MAXIMIZE_FLAGS,
#     x_axis='bops',
#     y_axis='performance_metric',
#     size_by='clock_cycles',
#     color_by='avg_resource'
# )

searcher = GlobalSearchTF(search_space_path=SEARCH_SPACE_PATH, results_dir=RESULTS_DIR)

study = searcher.run_search(
    model_type='block',
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    dataset='qubit',
    subset_size=SUBSET_SIZE,
    objectives=OBJECTIVE_NAMES,
    maximize_flags=MAXIMIZE_FLAGS,
    use_hardware_metrics=True,
    one_hot=True,
    n_folds=N_FOLDS,
    data_dir=QUBIT_DATA_DIR,
    start_location=START_LOCATION,
    window_size=WINDOW_SIZE,
    num_classes=NUM_CLASSES,
    normalize=False,
    flatten=True,
    callbacks=[search_callback],  # <-- ADD THIS LINE
)

print("\\nGlobal Search Complete!")

# Display the final figure (updates automatically, but show explicitly to ensure it's visible)
search_callback.show_figure()
# The figure above will have updated automatically as trials completed!
"""

# ============================================================================
# OPTIONAL STEP 3: Create playable animation after search (for presentations)
# ============================================================================
"""
# Optional: Create animated figure with frames for playback
fig = search_callback.create_animated_figure(
    x_axis='bops',
    y_axis='performance_metric',
    size_by='clock_cycles',
    color_by='avg_resource'
)
if fig:
    fig.show()
"""
