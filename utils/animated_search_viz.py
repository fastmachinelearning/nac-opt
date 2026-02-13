"""
Animated visualization for global search progress using Plotly and matplotlib.
"""
import pandas as pd
import plotly.graph_objects as go
import optuna
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Disable FigureWidget for VS Code compatibility - use regular Figure instead
# FigureWidget requires anywidget which doesn't work well in VS Code Jupyter
_FigureWidget = None
_USE_WIDGET = False  # Set to False to avoid anywidget issues in VS Code


class AnimatedSearchCallback:
    """Callback to collect trial data and create animated visualization with real-time updates."""
    def __init__(self, objective_names, maximize_flags, x_axis='avg_resource', y_axis='clock_cycles',
                 size_by='clock_cycles', color_by='performance_metric', display_figure=True, use_3d=False,
                 x_axis_3d='avg_resource', y_axis_3d='clock_cycles', z_axis_3d='performance_metric',
                 color_by_3d='bops', use_size=False, use_matplotlib_3d=False):
        self.objective_names = objective_names
        self.maximize_flags = maximize_flags
        self.trial_data = []
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.size_by = size_by
        self.color_by = color_by
        self.display_figure = display_figure
        self.use_3d = use_3d
        self.x_axis_3d = x_axis_3d
        self.y_axis_3d = y_axis_3d
        self.z_axis_3d = z_axis_3d
        self.color_by_3d = color_by_3d  # Which metric to use for color in 3D
        self.use_size = use_size  # Whether to vary marker size (False = same size, color only)
        self.use_matplotlib_3d = use_matplotlib_3d  # Use matplotlib 3D (non-interactive) instead of Plotly
        # Disable widget mode - use regular Figure for better compatibility
        self.use_widget = False  # Always use regular Figure
        
        # Initialize empty figure widget for real-time updates
        self.fig = None
        self._display_handle = None
        self._display_id = f"search_viz_{id(self)}"  # Unique ID for this callback
        self._initialize_figure()
        
    def _initialize_figure(self):
        """Initialize a regular Plotly Figure or matplotlib 3D (FigureWidget disabled for VS Code compatibility)."""
        if self.use_matplotlib_3d:
            self._initialize_matplotlib_3d()
        else:
            # Use Plotly Figure (can be 2D or 3D)
            self._initialize_figure_regular()
    
    def _initialize_figure_regular(self):
        """Initialize a regular Figure (fallback when FigureWidget not available)."""
        if self.fig is not None:
            return  # Already initialized
        
        if self.use_3d:
            # Create 3D scatter plot
            self.fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=[],
                        y=[],
                        z=[],
                        mode='markers',
                        marker=dict(
                            size=10 if not self.use_size else [],  # Fixed size or variable
                            color=[],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=self.color_by_3d.replace('_', ' ').title()),
                            line=dict(width=1, color='black'),
                            opacity=0.8
                        ),
                        text=[],
                        hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                                    f'{self.x_axis_3d.replace("_", " ").title()}: %{{x:.2f}}<br>' +
                                    f'{self.y_axis_3d.replace("_", " ").title()}: %{{y:,.0f}}<br>' +
                                    f'{self.z_axis_3d.replace("_", " ").title()}: %{{z:.4f}}<br>' +
                                    f'{self.color_by_3d.replace("_", " ").title()}: %{{marker.color:.2f}}<br>' +
                                    f'<extra></extra>',
                        name='Trials'
                    )
                ],
                layout=go.Layout(
                    scene=dict(
                        xaxis=dict(title=self.x_axis_3d.replace('_', ' ').title()),
                        yaxis=dict(title=self.y_axis_3d.replace('_', ' ').title()),
                        zaxis=dict(title=self.z_axis_3d.replace('_', ' ').title()),
                    ),
                    title=dict(
                        text=f'Global Search Progress (3D): {self.z_axis_3d.replace("_", " ").title()} vs {self.x_axis_3d.replace("_", " ").title()} vs {self.y_axis_3d.replace("_", " ").title()}',
                        x=0.5
                    ),
                    height=700,
                    margin=dict(l=0, r=0, b=0, t=50)
                )
            )
        else:
            # Create 2D scatter plot
            self.fig = go.Figure(
                data=[
                    go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        marker=dict(
                            size=[],
                            color=[],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=self.color_by.replace('_', ' ').title()),
                            line=dict(width=1, color='black')
                        ),
                        text=[],
                        hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                                    f'{self.x_axis.replace("_", " ").title()}: %{{x:,.0f}}<br>' +
                                    f'{self.y_axis.replace("_", " ").title()}: %{{y:.4f}}<br>' +
                                    f'{self.color_by.replace("_", " ").title()}: %{{marker.color:.2f}}<br>' +
                                    f'<extra></extra>',
                        name='Trials'
                    )
                ],
                layout=go.Layout(
                    xaxis=dict(title=self.x_axis.replace('_', ' ').title()),
                    yaxis=dict(title=self.y_axis.replace('_', ' ').title()),
                    title=dict(
                        text=f'Global Search Progress (Live): {self.y_axis.replace("_", " ").title()} vs {self.x_axis.replace("_", " ").title()} [Color: {self.color_by.replace("_", " ").title()}]',
                        x=0.5
                    ),
                    height=600
                )
            )
        
        if self.display_figure:
            if self.use_matplotlib_3d:
                # For matplotlib, we'll display it differently
                self._display_matplotlib_figure()
            else:
                # Display Plotly figure with a persistent display_id for in-place updates
                try:
                    from IPython.display import display
                    # Display with a unique ID - this allows us to update the same output
                    # Don't print here to avoid creating separate output areas
                    self._display_handle = display(self.fig, display_id=self._display_id)
                except ImportError:
                    # Fallback if not in Jupyter
                    try:
                        self.fig.show()
                    except Exception:
                        pass
                except Exception:
                    # If display with ID fails, try regular display
                    try:
                        from IPython.display import display
                        self._display_handle = display(self.fig, display_id=self._display_id)
                    except Exception:
                        pass
    
    def _initialize_matplotlib_3d(self):
        """Initialize a matplotlib 3D figure (non-interactive, updates better in VS Code)."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None  # Will be created on first update
        
    def _display_matplotlib_figure(self):
        """Display matplotlib figure."""
        try:
            from IPython.display import display
            display(self.fig, display_id=self._display_id)
        except Exception:
            plt.show()
    
    def _update_matplotlib_3d(self, df):
        """Update matplotlib 3D plot."""
        if self.fig is None or self.ax is None:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        # Get data
        x_data = df[self.x_axis_3d].tolist()
        y_data = df[self.y_axis_3d].tolist()
        z_data = df[self.z_axis_3d].tolist()
        
        # Get color data
        if self.color_by_3d in df.columns:
            color_data = df[self.color_by_3d].tolist()
        else:
            color_data = [0] * len(df)
        
        # Create scatter plot
        self.scatter = self.ax.scatter(
            x_data, y_data, z_data,
            c=color_data,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set labels
        self.ax.set_xlabel(self.x_axis_3d.replace('_', ' ').title())
        self.ax.set_ylabel(self.y_axis_3d.replace('_', ' ').title())
        self.ax.set_zlabel(self.z_axis_3d.replace('_', ' ').title())
        
        # Set title
        self.ax.set_title(
            f'Global Search Progress (3D): {self.z_axis_3d.replace("_", " ").title()} vs '
            f'{self.x_axis_3d.replace("_", " ").title()} vs {self.y_axis_3d.replace("_", " ").title()}'
        )
        
        # Add colorbar
        if len(color_data) > 0:
            cbar = plt.colorbar(self.scatter, ax=self.ax, pad=0.1)
            cbar.set_label(self.color_by_3d.replace('_', ' ').title())
        
        # Adjust layout
        plt.tight_layout()
        
        # Draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def show_figure(self):
        """Explicitly display/update the figure - call this after search completes."""
        if self.fig is None:
            print("No figure available.")
            return
        
        # Update with latest data
        if self.trial_data:
            self._update_figure()
        
        # Display the figure
        try:
            self.fig.show()
        except Exception as e:
            print(f"Error displaying figure: {e}")
            # Fallback: try IPython display
            try:
                from IPython.display import display
                display(self.fig)
            except:
                print("Could not display figure. Try: search_callback.fig.show()")
        
    def __call__(self, study, trial):
        """Called after each trial completes - updates the figure in real-time."""
        if trial.state.name == "COMPLETE":
            values = trial.values
            data_point = {
                'trial': trial.number,
                'performance_metric': values[0],
                'bops': values[1] if len(values) > 1 else None,
                'avg_resource': values[2] if len(values) > 2 else None,
                'clock_cycles': values[3] if len(values) > 3 else None,
            }
            self.trial_data.append(data_point)
            
            # Update the figure widget in real-time
            self._update_figure()
            
            # Update and refresh display on every trial (without printing to avoid new outputs)
            self._refresh_display()
            
            # Suppress print statements during updates to avoid creating new output areas
            # Progress is shown in the figure itself via hover tooltips
    
    def _update_figure(self):
        """Update the figure widget with new trial data."""
        if not self.trial_data or self.fig is None:
            return
        
        df = pd.DataFrame(self.trial_data)
        
        # Handle matplotlib 3D updates
        if self.use_matplotlib_3d:
            self._update_matplotlib_3d(df)
            self._refresh_display()
            return
        
        # Calculate normalized sizes and colors
        if self.size_by in df.columns and len(df) > 0:
            size_max = df[self.size_by].max()
            if size_max > 0:
                sizes = (df[self.size_by] / size_max * 50 + 10).tolist()
            else:
                sizes = [15] * len(df)
        else:
            sizes = [15] * len(df)
            
        if self.color_by in df.columns:
            colors = df[self.color_by].tolist()
        else:
            colors = [0] * len(df)
        
        # Update the scatter plot data
        if self.use_widget and hasattr(self.fig, 'batch_update'):
            # Use batch_update for FigureWidget (more efficient)
            try:
                with self.fig.batch_update():
                    self._update_figure_data(df, sizes, colors)
            except Exception:
                # Fallback if batch_update fails
                self._update_figure_data(df, sizes, colors)
        else:
            # For regular Figure, update directly and refresh display
            self._update_figure_data(df, sizes, colors)
            # Update the display in real-time using update_display
            self._refresh_display()
    
    def _update_figure_data(self, df, sizes, colors):
        """Update figure data directly (works for both 2D and 3D)."""
        if self.use_3d:
            # Update 3D scatter plot
            self.fig.data[0].x = df[self.x_axis_3d].tolist()
            self.fig.data[0].y = df[self.y_axis_3d].tolist()
            self.fig.data[0].z = df[self.z_axis_3d].tolist()
            
            # Set marker size (fixed or variable)
            if self.use_size:
                self.fig.data[0].marker.size = sizes
            else:
                self.fig.data[0].marker.size = 10  # Fixed size
            
            # Set marker color based on color_by_3d
            if self.color_by_3d in df.columns:
                self.fig.data[0].marker.color = df[self.color_by_3d].tolist()
            else:
                self.fig.data[0].marker.color = [0] * len(df)
            
            self.fig.data[0].text = [f"Trial {t}" for t in df['trial']]
            
            # Update 3D axis ranges
            if len(df) > 0:
                x_min, x_max = df[self.x_axis_3d].min(), df[self.x_axis_3d].max()
                y_min, y_max = df[self.y_axis_3d].min(), df[self.y_axis_3d].max()
                z_min, z_max = df[self.z_axis_3d].min(), df[self.z_axis_3d].max()
                
                if x_max > x_min:
                    self.fig.layout.scene.xaxis.range = [x_min * 0.9, x_max * 1.1]
                if y_max > y_min:
                    self.fig.layout.scene.yaxis.range = [y_min * 0.9, y_max * 1.1]
                if z_max > z_min:
                    self.fig.layout.scene.zaxis.range = [z_min * 0.95, z_max * 1.05]
        else:
            # Update 2D scatter plot
            self.fig.data[0].x = df[self.x_axis].tolist()
            self.fig.data[0].y = df[self.y_axis].tolist()
            self.fig.data[0].marker.size = sizes
            self.fig.data[0].marker.color = colors
            self.fig.data[0].text = [f"Trial {t}" for t in df['trial']]
            
            # Update axis ranges
            if len(df) > 0:
                x_min, x_max = df[self.x_axis].min(), df[self.x_axis].max()
                y_min, y_max = df[self.y_axis].min(), df[self.y_axis].max()
                
                if x_max > x_min:
                    self.fig.layout.xaxis.range = [x_min * 0.9, x_max * 1.1]
                if y_max > y_min:
                    self.fig.layout.yaxis.range = [y_min * 0.95, y_max * 1.05]
    
    def _refresh_display(self):
        """Refresh the displayed figure in real-time - updates the same output area."""
        if not self.display_figure or self.fig is None:
            return
        
        if self.use_matplotlib_3d:
            # For matplotlib, use clear_output + display
            try:
                from IPython.display import display, clear_output
                clear_output(wait=True)
                display(self.fig, display_id=self._display_id)
            except Exception:
                try:
                    from IPython.display import display
                    display(self.fig, display_id=self._display_id)
                except Exception:
                    pass
            return
        
        # For Plotly 3D interactive plot, use update_display to update in place
        # This should update the same output without creating new plots
        try:
            from IPython.display import update_display
            # update_display updates the existing output with the same display_id
            update_display(self.fig, display_id=self._display_id)
            return
        except (ImportError, AttributeError, TypeError, NameError) as e:
            # If update_display not available, try alternative
            pass
        
        # Alternative: Use display with update=True (if supported)
        try:
            from IPython.display import display
            # Some versions support update=True parameter
            try:
                display(self.fig, display_id=self._display_id, update=True)
                return
            except TypeError:
                # update parameter not supported, try regular display with ID
                display(self.fig, display_id=self._display_id)
                return
        except Exception:
            pass
        
        # Last resort: clear and re-display (creates flicker but updates)
        try:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig, display_id=self._display_id)
        except Exception:
            pass
    
    def get_figure(self):
        """Get the figure widget for display."""
        return self.fig
    
    def show_figure(self):
        """Explicitly display/update the figure - call this after search completes."""
        if self.fig is None:
            print("No figure available.")
            return
        
        # Update with latest data
        if self.trial_data:
            self._update_figure()
        
        # Display the figure
        try:
            self.fig.show()
        except Exception as e:
            # Fallback: try IPython display
            try:
                from IPython.display import display
                display(self.fig)
            except Exception:
                print(f"Could not display figure. Try accessing: search_callback.fig")
                print(f"Or use: search_callback.fig.show()")
    
    def create_animated_figure(self, x_axis='bops', y_axis='performance_metric', 
                               size_by='clock_cycles', color_by='avg_resource'):
        """Create an animated Plotly figure showing search progress (for post-search playback)."""
        if not self.trial_data:
            print("No trial data collected yet.")
            return None
        
        df = pd.DataFrame(self.trial_data)
        
        # Determine axis ranges
        x_min, x_max = df[x_axis].min(), df[x_axis].max()
        y_min, y_max = df[y_axis].min(), df[y_axis].max()
        x_range = [x_min * 0.9, x_max * 1.1]
        y_range = [y_min * 0.95, y_max * 1.05]
        
        # Normalize size and color for markers
        if size_by in df.columns:
            size_normalized = (df[size_by] / df[size_by].max() * 50 + 10).tolist()
        else:
            size_normalized = [15] * len(df)
            
        if color_by in df.columns:
            color_values = df[color_by].tolist()
        else:
            color_values = [0] * len(df)
        
        # Create frames - each frame adds one more point
        frames = []
        for i in range(1, len(df) + 1):
            frame_data = df.iloc[:i]
            frame_sizes = size_normalized[:i]
            frame_colors = color_values[:i]
            
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=frame_data[x_axis],
                            y=frame_data[y_axis],
                            mode='markers',
                            marker=dict(
                                size=frame_sizes,
                                color=frame_colors,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title=color_by.replace('_', ' ').title()),
                                line=dict(width=1, color='black')
                            ),
                            text=[f"Trial {t}" for t in frame_data['trial']],
                            hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                                        f'{x_axis.replace("_", " ").title()}: %{{x:,.0f}}<br>' +
                                        f'{y_axis.replace("_", " ").title()}: %{{y:.4f}}<br>' +
                                        (f'{color_by.replace("_", " ").title()}: %{{marker.color:.2f}}<br>' if color_by in df.columns else '') +
                                        f'<extra></extra>',
                            name='Trials'
                        )
                    ],
                    name=str(i-1)
                )
            )
        
        # Initial data (first point)
        initial_data = df.iloc[:1] if len(df) > 0 else df
        
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=initial_data[x_axis] if len(initial_data) > 0 else [],
                    y=initial_data[y_axis] if len(initial_data) > 0 else [],
                    mode='markers',
                    marker=dict(
                        size=[size_normalized[0]] if len(size_normalized) > 0 else [15],
                        color=[color_values[0]] if len(color_values) > 0 else [0],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=color_by.replace('_', ' ').title()),
                        line=dict(width=1, color='black')
                    ),
                    text=[f"Trial {t}" for t in initial_data['trial']] if len(initial_data) > 0 else [],
                    hovertemplate=f'<b>Trial %{{text}}</b><br>' +
                                f'{x_axis.replace("_", " ").title()}: %{{x:,.0f}}<br>' +
                                f'{y_axis.replace("_", " ").title()}: %{{y:.4f}}<br>' +
                                (f'{color_by.replace("_", " ").title()}: %{{marker.color:.2f}}<br>' if color_by in df.columns else '') +
                                f'<extra></extra>',
                    name='Trials'
                )
            ],
            frames=frames,
            layout=go.Layout(
                xaxis=dict(
                    title=x_axis.replace('_', ' ').title(),
                    range=x_range
                ),
                yaxis=dict(
                    title=y_axis.replace('_', ' ').title(),
                    range=y_range
                ),
                title=dict(
                    text=f'Global Search Progress: {y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}',
                    x=0.5
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[None, {
                                    "frame": {"duration": 500, "redraw": False},
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
                    active=0,
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
                                "frame": {"duration": 300, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }],
                            label=f"Trial {i}",
                            method="animate"
                        )
                        for i in range(len(df))
                    ]
                )]
            )
        )
        
        return fig
