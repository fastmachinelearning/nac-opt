#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import itertools
import math
import ast

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
                # Look for dict in curly braces
                m = re.search(r"Trial\s+(\d+)\s+\|\s+Objectives:\s+({[^}]+})", line)
                if m:
                    trial_number = int(m.group(1))
                    objectives_dict_str = m.group(2)
                    try:
                        # Eval the dictionary string.
                        objectives_dict = ast.literal_eval(objectives_dict_str)
                    except Exception as e:
                        print("Error parsing objectives for trial", trial_number, ":", e)
                        continue
                    # Combine trial number and objectives into one dictionary.
                    row = {"trial": trial_number}
                    row.update(objectives_dict)
                    results.append(row)
    return pd.DataFrame(results)

def pareto_front_indices(df, obj1, obj2, maximize1, maximize2):
    """
    Returns indices of non-dominated (Pareto optimal) points for two objectives.
    If maximize is True for an objective, we multiply by -1 so that
    lower values become better.
    """
    f1 = -df[obj1] if maximize1 else df[obj1]
    f2 = -df[obj2] if maximize2 else df[obj2]
    
    indices = []
    for i, (a, b) in enumerate(zip(f1, f2)):
        dominated = False
        for j, (a2, b2) in enumerate(zip(f1, f2)):
            if j == i:
                continue
            if a2 <= a and b2 <= b and (a2 < a or b2 < b):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return indices

def plot_pareto_fronts(df, objective_info):
    """
    Plots all pairwise combinations of objectives.
    objective_info is a list of tuples (name, maximize), where maximize is a boolean.
    """
    objective_names = [obj[0] for obj in objective_info]
    # Generate all pairwise combinations
    pairs = list(itertools.combinations(objective_names, 2))
    num_plots = len(pairs)
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)
    
    plt.figure(figsize=(5 * ncols, 5 * nrows))
    
    for i, (obj1, obj2) in enumerate(pairs, 1):
        maximize1 = dict(objective_info)[obj1]
        maximize2 = dict(objective_info)[obj2]
        plt.subplot(nrows, ncols, i)
        plt.scatter(df[obj1], df[obj2], label="All Trials", alpha=0.6)
        # Compute Pareto front indices
        pareto_indices = pareto_front_indices(df, obj1, obj2, maximize1, maximize2)
        pareto_points = df.iloc[pareto_indices]
        plt.scatter(pareto_points[obj1], pareto_points[obj2],
                    color="red", marker="D", s=80, label="Pareto Front")
        plt.xlabel(obj1)
        plt.ylabel(obj2)
        plt.title(f"{obj1} vs {obj2}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("pareto_fronts.png")  # Save to file
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto fronts for multi-objective global search results."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="global_search_results.txt",
        help="Path to the results text file."
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="accuracy,bops,resource,clock_cycles",
        help=("Comma-separated list of objective names, in the order they appear "
              "in the results file. Default: 'accuracy,bops,resource,clock_cycles'")
    )
    parser.add_argument(
        "--maximize",
        type=str,
        default="True,False,False,False",
        help=("Comma-separated list of booleans (True/False) indicating whether "
              "each corresponding objective should be maximized. Default: "
              "'True,False,False,False'")
    )
    
    args = parser.parse_args()
    
    # Parse the objectives and maximize flags.
    objective_names = [s.strip() for s in args.objectives.split(",")]
    maximize_list = [s.strip().lower() == "true" for s in args.maximize.split(",")]
    objective_info = list(zip(objective_names, maximize_list))
    
    df = parse_results(args.results_file, objective_names)
    print("Parsed results:")
    print(df)
    
    plot_pareto_fronts(df, objective_info)

if __name__ == "__main__":
    main()