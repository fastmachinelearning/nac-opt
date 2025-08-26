import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
import itertools
import math
import ast
import os
import numpy as np




def parse_results_from_df(df, objective_names):
    """
    Extracts objective values from a DataFrame.
    This is simpler as we now have a direct DataFrame.
    """
    # Ensure all objective names are present as columns
    for obj in objective_names:
        # The column names in the dataframe are already sanitized
        col_name = obj.lower().replace(" ", "_")
        if col_name not in df.columns:
            raise ValueError(f"Objective '{obj}' not found in the results DataFrame columns: {df.columns}")
    return df

def parse_results_from_file(filepath, objective_names):
    """
    Parse a results text file and return a DataFrame.
    Assumes each trial line starts with "Trial" and contains a dictionary
    of objectives. The keys in the dictionary should match the objective_names.
    """
    results = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip().startswith("Objectives:"):
                # Look for a dictionary enclosed in curly braces.
                m = re.search(r"Objectives:\s+({[^}]+})", line)
                if m:
                    objectives_dict_str = m.group(1)
                    try:
                        # Safely evaluate the dictionary string.
                        objectives_dict = ast.literal_eval(objectives_dict_str)
                        results.append(objectives_dict)
                    except Exception as e:
                        print("Error parsing objectives line:", line, "Error:", e)
                        continue
    return pd.DataFrame(results)


def get_pareto_front_indices(df, objectives_info):
    """
    Returns indices of non-dominated (Pareto optimal) points for a set of objectives.
    
    Parameters:
        df (pd.DataFrame): DataFrame with objective columns.
        objectives_info (list of tuples): List of (name, maximize_bool) tuples.
        
    Returns:
        list: Indices of non-dominated points.
    """
    
    num_points = len(df)
    is_pareto = [True] * num_points
    
    # Convert df columns to numpy for speed
    objective_values = []
    for name, maximize in objectives_info:
        col_name = name.lower().replace(" ", "_")
        values = df[col_name].values
        if maximize:
            values = -values
        objective_values.append(values)
    
    objective_values = np.array(objective_values).T # Shape: (num_points, num_objectives)

    for i in range(num_points):
        if not is_pareto[i]:
            continue
        
        # Check if point i is dominated by any other point j
        # A point i is dominated by j if j is better or equal in all objectives, and strictly better in at least one
        is_dominated_by_any = np.any(np.all(objective_values <= objective_values[i], axis=1) & np.any(objective_values < objective_values[i], axis=1))
        
        if is_dominated_by_any:
            is_pareto[i] = False

    return df.index[is_pareto].tolist()


def plot_pareto_fronts(df, objective_info, save_dir="."):
    """
    Plots all pairwise combinations of objectives.
    objective_info is a list of tuples (name, maximize), where maximize is a boolean.
    """
    objective_names = [obj[0] for obj in objective_info]
    pairs = list(itertools.combinations(objective_names, 2))
    num_plots = len(pairs)
    if num_plots == 0:
        return
        
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)
    
    plt.figure(figsize=(6 * ncols, 5 * nrows))
    
    for i, (obj1, obj2) in enumerate(pairs, 1):
        maximize1 = dict(objective_info)[obj1]
        maximize2 = dict(objective_info)[obj2]
        
        plt.subplot(nrows, ncols, i)
        
        col1 = obj1.lower().replace(" ", "_")
        col2 = obj2.lower().replace(" ", "_")

        plt.scatter(df[col1], df[col2], label="All Trials", alpha=0.6, s=30)
        
        # Compute Pareto front indices for this pair
        pareto_indices = get_pareto_front_indices(df, [(obj1, maximize1), (obj2, maximize2)])
        pareto_points = df.loc[pareto_indices]
        
        plt.scatter(pareto_points[col1], pareto_points[col2],
                    color="red", marker="D", s=60, label="Pareto Front", zorder=5)
        
        plt.xlabel(obj1)
        plt.ylabel(obj2)
        plt.title(f"{obj1} vs {obj2}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "pareto_fronts_2d.png")
    plt.savefig(save_path)
    print(f"2D Pareto fronts plot saved to {save_path}")
    plt.show()


def plot_3d_pareto_front_heatmap(df, objectives_info, save_dir="."):
    """
    Plots a 3D scatter plot using the first three objectives as axes
    and the fourth objective as a heat map (color).

    Parameters:
        df (pd.DataFrame): DataFrame containing the objective data.
        objectives_info (list of tuples): A list of four tuples (name, maximize),
            where the first three correspond to the x, y, and z dimensions,
            and the fourth is used for the color mapping.
    """
    if len(objectives_info) < 4:
        raise ValueError("At least 4 objectives must be provided for the 3D heatmap plot.")
    
    # Unpack objective names and maximize flags.
    obj1_info, obj2_info, obj3_info, obj4_info = objectives_info[:4]
    obj1, max1 = obj1_info
    obj2, max2 = obj2_info
    obj3, max3 = obj3_info
    obj4, _ = obj4_info # Maximize flag for color axis is not used for plotting

    col1 = obj1.lower().replace(" ", "_")
    col2 = obj2.lower().replace(" ", "_")
    col3 = obj3.lower().replace(" ", "_")
    col4 = obj4.lower().replace(" ", "_")

    # Use the generalized Pareto front function for the first three objectives.
    pareto_indices = get_pareto_front_indices(df, [obj1_info, obj2_info, obj3_info])
    pareto_points = df.loc[pareto_indices]
    
    fig = go.Figure()
    
    # Trace for all trials, colored by the 4th objective.
    fig.add_trace(go.Scatter3d(
        x=df[col1],
        y=df[col2],
        z=df[col3],
        mode="markers",
        marker=dict(
            size=5,
            color=df[col4],
            colorscale="Viridis",
            opacity=0.6,
            colorbar=dict(title=obj4)
        ),
        name="All Trials",
        text=[f"Trial {i}" for i in df.index],
        hoverinfo='text+x+y+z'
    ))
    
    # Trace for Pareto front points.
    fig.add_trace(go.Scatter3d(
        x=pareto_points[col1],
        y=pareto_points[col2],
        z=pareto_points[col3],
        mode="markers",
        marker=dict(
            size=8,
            color=pareto_points[col4],
            colorscale="Viridis",
            symbol="diamond",
            line=dict(color='black', width=1)
        ),
        name="Pareto Front (3D)",
        text=[f"Trial {i}" for i in pareto_points.index],
        hoverinfo='text+x+y+z'
    ))
    
    # Update layout with axis titles and overall title.
    fig.update_layout(
        title=f"3D Pareto Front: {obj1} vs {obj2} vs {obj3} with {obj4} as color",
        scene=dict(
            xaxis_title=obj1,
            yaxis_title=obj2,
            zaxis_title=obj3
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    save_path = os.path.join(save_dir, "pareto_front_3d.html")
    fig.write_html(save_path)
    print(f"3D Pareto front plot saved to {save_path}")
    fig.show()