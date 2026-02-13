"""
Create a cumulative 3D animation showing search progress.
Each frame shows all trials up to that point.
"""
import pandas as pd
import plotly.graph_objects as go

def create_cumulative_3d_animation(results_df, x_col='avg_resource', y_col='clock_cycles', 
                                   z_col='performance_metric', color_col='bops'):
    """
    Create a 3D animated scatter plot where each frame cumulatively adds trials.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with trial results
    x_col, y_col, z_col : str
        Column names for x, y, z axes
    color_col : str
        Column name for marker color
    """
    # Sort by trial number to ensure correct order
    results_df = results_df.sort_values('trial').reset_index(drop=True)
    
    # Calculate fixed color range across ALL trials (so colors stay consistent)
    if len(results_df) > 0:
        color_min = results_df[color_col].min()
        color_max = results_df[color_col].max()
    else:
        color_min = color_max = 0
    
    # Create frames - each frame shows all trials up to that point
    frames = []
    for i in range(1, len(results_df) + 1):
        frame_data = results_df.iloc[:i]
        
        # Convert to lists to ensure proper data handling
        x_vals = frame_data[x_col].tolist()
        y_vals = frame_data[y_col].tolist()
        z_vals = frame_data[z_col].tolist()
        color_vals = frame_data[color_col].tolist()
        trial_nums = frame_data['trial'].tolist()
        
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color_vals,
                            colorscale='Viridis',
                            cmin=color_min,  # Fixed min for consistent colors
                            cmax=color_max,  # Fixed max for consistent colors
                            showscale=True,
                            colorbar=dict(title=color_col.replace('_', ' ').title()),
                            line=dict(width=1, color='black'),
                            opacity=0.8
                        ),
                        text=[f"Trial {t}" for t in trial_nums],
                        hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                                    f'{x_col.replace("_", " ").title()}: %{{x:.2f}}<br>' +
                                    f'{y_col.replace("_", " ").title()}: %{{y:,.0f}}<br>' +
                                    f'{z_col.replace("_", " ").title()}: %{{z:.4f}}<br>' +
                                    f'{color_col.replace("_", " ").title()}: %{{marker.color:,.0f}}<br>' +
                                    f'<extra></extra>',
                        name='Trials'
                    )
                ],
                name=str(i-1),
                traces=[0]  # Specify which trace to update (the first/only trace)
            )
        )
    
    # Initial data - show ALL points by default (use slider/play to see progression)
    if len(results_df) > 0:
        # Show all points initially, frames show progression
        x_init = results_df[x_col].tolist()
        y_init = results_df[y_col].tolist()
        z_init = results_df[z_col].tolist()
        color_init = results_df[color_col].tolist()
        trial_init = results_df['trial'].tolist()
    else:
        x_init = y_init = z_init = color_init = trial_init = []
    
    # Determine axis ranges
    if len(results_df) > 0:
        x_min, x_max = results_df[x_col].min(), results_df[x_col].max()
        y_min, y_max = results_df[y_col].min(), results_df[y_col].max()
        z_min, z_max = results_df[z_col].min(), results_df[z_col].max()
    else:
        x_min = x_max = y_min = y_max = z_min = z_max = 0
    
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_init,
                y=y_init,
                z=z_init,
                mode='markers',
                marker=dict(
                    size=10,
                    color=color_init,
                    colorscale='Viridis',
                    cmin=color_min if len(results_df) > 0 else 0,  # Fixed min for consistent colors
                    cmax=color_max if len(results_df) > 0 else 0,  # Fixed max for consistent colors
                    showscale=True,
                    colorbar=dict(title=color_col.replace('_', ' ').title()),
                    line=dict(width=1, color='black'),
                    opacity=0.8
                ),
                text=[f"Trial {t}" for t in trial_init],
                hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                            f'{x_col.replace("_", " ").title()}: %{{x:.2f}}<br>' +
                            f'{y_col.replace("_", " ").title()}: %{{y:,.0f}}<br>' +
                            f'{z_col.replace("_", " ").title()}: %{{z:.4f}}<br>' +
                            f'{color_col.replace("_", " ").title()}: %{{marker.color:,.0f}}<br>' +
                            f'<extra></extra>',
                name='Trials'
            )
        ],
        frames=frames,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title=x_col.replace('_', ' ').title(), range=[x_min * 0.9, x_max * 1.1]),
                yaxis=dict(title=y_col.replace('_', ' ').title(), range=[y_min * 0.9, y_max * 1.1]),
                zaxis=dict(title=z_col.replace('_', ' ').title(), range=[z_min * 0.95, z_max * 1.05]),
            ),
            title=dict(
                text=f'Global Search Progress (3D): {z_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}<br><sub>Use slider or Play button to see progression</sub>',
                x=0.5
            ),
            height=700,
            margin=dict(l=0, r=0, b=0, t=50),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 500, "redraw": True},  # redraw=True for 3D plots
                                "fromcurrent": True,
                                "transition": {"duration": 300, "easing": "quadratic-in-out"}
                            }]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.1,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )
            ],
            sliders=[dict(
                active=len(results_df) - 1 if len(results_df) > 0 else 0,  # Start at last frame (all points visible)
                yanchor="top",
                xanchor="left",
                currentvalue={
                    "font": {"size": 20},
                    "prefix": "Trial:",
                    "visible": True,
                    "xanchor": "right"
                },
                transition={"duration": 300, "easing": "cubic-in-out"},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[str(i)], {
                            "frame": {"duration": 300, "redraw": True},  # redraw=True for 3D plots
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        label=f"Trial {i+1}",
                        method="animate"
                    )
                    for i in range(len(results_df))
                ]
            )]
        )
    )
    
    return fig

# Usage example:
# results_df = pd.DataFrame(searcher.results)
# fig = create_cumulative_3d_animation(
#     results_df,
#     x_col='avg_resource',
#     y_col='clock_cycles',
#     z_col='performance_metric',
#     color_col='bops'
# )
# fig.show()
