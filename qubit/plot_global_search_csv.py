#!/usr/bin/env python3
"""
Produce the same 3D cumulative animation as in the notebook using the CSV
saved by run_global_search.py.

Usage:
    python plot_global_search_csv.py [--csv PATH] [--model_type block]
"""
import argparse
from pathlib import Path

import pandas as pd

# Default paths (run from nac-opt/qubit/)
DEFAULT_RESULTS_DIR = Path("./results/global_search")
DEFAULT_CSV = None  # Will use results_dir / f"{model_type}_search_results.csv"


def main():
    p = argparse.ArgumentParser(description="Plot global search results from CSV")
    p.add_argument("--csv", type=str, default=None,
                   help="Path to CSV file (e.g. results/global_search/block_search_results.csv)")
    p.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                   help="Results directory (used if --csv not set)")
    p.add_argument("--model_type", type=str, default="block",
                   help="Model type for CSV name when using --results_dir")
    p.add_argument("--save", type=str, default=None,
                   help="Save figure to HTML file instead of showing")
    args = p.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = Path(args.results_dir) / f"{args.model_type}_search_results.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run run_global_search.py first (with --use_hardware_metrics for avg_resource/clock_cycles).")
        return 1

    results_df = pd.read_csv(csv_path)

    # Same columns as the notebook (requires hardware metrics)
    required = ["trial", "performance_metric", "bops", "avg_resource", "clock_cycles"]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        print(f"CSV is missing columns: {missing}")
        print("Run with hardware metrics: python run_global_search.py ... (default is on)")
        return 1

    from create_3d_animation import create_cumulative_3d_animation

    fig = create_cumulative_3d_animation(
        results_df,
        x_col="avg_resource",
        y_col="clock_cycles",
        z_col="performance_metric",
        color_col="bops",
    )
    if args.save:
        fig.write_html(args.save)
        print(f"Saved to {args.save}")
    else:
        fig.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
