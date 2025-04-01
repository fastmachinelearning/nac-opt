import argparse
import pandas as pd
import plotly.graph_objects as go

# Function to parse the results file
def parse_results(filepath, objective_names):
    """
    Parse a results text file and return a DataFrame.
    Assumes each trial line starts with "Trial" and contains a dictionary
    of objectives. The keys in the dictionary should match the objective_names.
    """
    results = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Trial"):
                # Look for a dictionary enclosed in curly braces.
                m = re.search(r"Trial\s+(\d+)\s+\|\s+Objectives:\s+({[^}]+})", line)
                if m:
                    trial_number = int(m.group(1))
                    objectives_dict_str = m.group(2)
                    try:
                        # Safely evaluate the dictionary string.
                        objectives_dict = ast.literal_eval(objectives_dict_str)
                    except Exception as e:
                        print("Error parsing objectives for trial", trial_number, ":", e)
                        continue
                    # Combine trial number and objectives into one dictionary.
                    row = {"trial": trial_number}
                    row.update(objectives_dict)
                    results.append(row)
    return pd.DataFrame(results)

def pareto_front_indices_general(df, objectives, maximize_flags):
    """
    Returns indices of non-dominated (Pareto optimal) points for a set of objectives.
    
    Parameters:
        df (pd.DataFrame): DataFrame with objective columns.
        objectives (list of str): List of objective names to compare.
        maximize_flags (list of bool): List of booleans indicating if each objective should be maximized.
        
    Returns:
        list: Indices of non-dominated points.
    """
    # Create a list of transformed objective arrays. If an objective is to be maximized,
    # we flip its sign so that all objectives can be treated as minimization.
    f_list = []
    for obj, maximize in zip(objectives, maximize_flags):
        f_list.append(-df[obj] if maximize else df[obj])
    
    n = len(df)
    indices = []
    
    # Loop through each point in the DataFrame.
    for i in range(n):
        current = [f.iloc[i] for f in f_list]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            other = [f.iloc[j] for f in f_list]
            # Check if other dominates current:
            if all(o <= c for o, c in zip(other, current)) and any(o < c for o, c in zip(other, current)):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    
    return indices

def plot_3d_with_heatmap(df, objectives_info):
    """
    Plots a 3D scatter plot using the first three objectives as axes
    and the fourth objective as a heat map (color).

    Parameters:
        df (pd.DataFrame): DataFrame containing the objective data.
        objectives_info (list of tuples): A list of four tuples (name, maximize),
            where the first three correspond to the x, y, and z dimensions,
            and the fourth is used for the color mapping.
    """
    if len(objectives_info) != 4:
        raise ValueError("Exactly 4 objectives must be provided.")
    
    # Unpack objective names and maximize flags.
    obj1, obj2, obj3, obj4 = [info[0] for info in objectives_info]
    max1, max2, max3, _ = [info[1] for info in objectives_info]
    
    # Use the generalized Pareto front function for the first three objectives.
    pareto_indices = pareto_front_indices_general(df, [obj1, obj2, obj3], [max1, max2, max3])
    pareto_points = df.iloc[pareto_indices]
    
    fig = go.Figure()
    
    # Trace for all trials, colored by the 4th objective.
    fig.add_trace(go.Scatter3d(
        x=df[obj1],
        y=df[obj2],
        z=df[obj3],
        mode="markers",
        marker=dict(
            size=5,
            color=df[obj4],
            colorscale="Viridis",
            opacity=0.6,
            colorbar=dict(title=obj4)
        ),
        name="All Trials"
    ))
    
    # Trace for Pareto front points.
    fig.add_trace(go.Scatter3d(
        x=pareto_points[obj1],
        y=pareto_points[obj2],
        z=pareto_points[obj3],
        mode="markers",
        marker=dict(
            size=8,
            color=pareto_points[obj4],
            colorscale="Viridis",
            symbol="diamond",
            colorbar=dict(title=obj4)
        ),
        name="Pareto Front (3D)"
    ))
    
    # Update layout with axis titles and overall title.
    fig.update_layout(
        title=f"3D Pareto Front: {obj1} vs {obj2} vs {obj3} with {obj4} as color",
        scene=dict(
            xaxis_title=obj1,
            yaxis_title=obj2,
            zaxis_title=obj3
        )
    )
    
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Plot Pareto front and visualize 3D scatter plot.")
    parser.add_argument(
        "--results_file",
        type=str,
        default="global_search_results.txt",
        help="Path to the results text file."
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="accuracy,avg_hw,clock_cycles,BOPs",
        help=("Comma-separated list of objective names, in the order they appear "
              "in the results file.")
    )
    parser.add_argument(
        "--maximize",
        type=str,
        default="True,False,False,False",
        help=("Comma-separated list of booleans (True/False) indicating whether "
              "each corresponding objective should be maximized.")
    )
    
    args = parser.parse_args()
    
    # Parse the objectives and maximize flags.
    objective_names = [s.strip() for s in args.objectives.split(",")]
    maximize_list = [s.strip().lower() == "true" for s in args.maximize.split(",")]
    objective_info = list(zip(objective_names, maximize_list))
    
    df = parse_results(args.results_file, objective_names)
    print("Parsed results:")
    print(df)
    
    plot_3d_with_heatmap(df, objective_info)

if __name__ == "__main__":
    main()