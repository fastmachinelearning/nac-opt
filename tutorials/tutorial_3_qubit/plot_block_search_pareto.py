#!/usr/bin/env python3
"""
Pareto-style plots from ``block_search_results.csv`` (matplotlib, Optuna-style).

**Default (``--mode cycle`` / ``star``):** writes **four separate PNG files** and the **same four as PDF** (no figure title; only axis labels):

  1. **Average Resources** (x) vs **Clock Cycles** (y)
  2. **Average Resources** (x) vs **Accuracy** (y)
  3. **Clock Cycles** (x) vs **Accuracy** (y)
  4. **BOPs** (x) vs **Accuracy** (y)

Each figure:
  - scatter colored by **trial** (viridis + slim colorbar ``Trial #``)
  - **Pareto front** for that pair (minimize costs; maximize accuracy when it appears): black line + hollow black ring markers
    (same dominance rule as ``utils.tf_visualization``)
  - optional **Baseline** red star from ``--baseline-json``, or by default
    ``results/baseline_objectives_52923746.json`` next to this script (from ``estimate_baseline_objectives.py``).
    Use ``--no-baseline`` to omit the marker.

``--mode all``: single combined figure via ``plot_pareto_fronts`` (six pairwise subplots, classic style).

``--combined``: one 2×2 PNG using ``plot_pareto_pairs_subplots`` (legacy layout).

Example:
  python3 plot_block_search_pareto.py \\
    --csv results/qubit_optuna_job_52893384/block_search_results.csv

  # explicit baseline path (overrides default ``baseline_objectives_52923746.json``):
  python3 plot_block_search_pareto.py --csv .../block_search_results.csv \\
    --baseline-json results/other_baseline.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.tf_visualization import get_pareto_front_indices, plot_pareto_fronts, plot_pareto_pairs_subplots

DEFAULT_OBJECTIVE_INFO = [
    ("performance_metric", True),
    ("avg_resource", False),
    ("clock_cycles", False),
    ("bops", False),
]

# Default four panels: (x_name, y_name) using CSV column semantics.
DEFAULT_FOUR_PAIRS = [
    ("avg_resource", "clock_cycles"),
    ("avg_resource", "performance_metric"),
    ("clock_cycles", "performance_metric"),
    ("bops", "performance_metric"),
]

# Legacy 2×2 grid (``--combined``): original axis pairs.
LEGACY_GRID_PAIRS = [
    ("performance_metric", "avg_resource"),
    ("avg_resource", "clock_cycles"),
    ("clock_cycles", "bops"),
    ("bops", "performance_metric"),
]

_OBJ_MAP = dict(DEFAULT_OBJECTIVE_INFO)

AXIS_LABEL = {
    "performance_metric": "Accuracy",
    "avg_resource": "Average Resources",
    "clock_cycles": "Clock Cycles",
    "bops": "BOPs",
    "lut_pct": "LUT (%)",
}


def _col(name):
    return name.lower().replace(" ", "_")


def _apply_plot_style():
    for style in (
        "seaborn-v0_8-whitegrid",
        "seaborn-whitegrid",
        "seaborn-v0_8",
        "seaborn",
        "ggplot",
    ):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    else:
        plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.35})
    mpl.rcParams["axes.formatter.useoffset"] = False
    mpl.rcParams["axes.formatter.use_locale"] = False
    mpl.rcParams["axes.formatter.use_mathtext"] = False


def _safe_filename_part(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


def _trial_series(df):
    if "trial" in df.columns:
        s = pd.to_numeric(df["trial"], errors="coerce")
        return s.fillna(pd.Series(range(len(df)), index=df.index))
    return pd.Series(range(len(df)), index=df.index)


def _axis_limits_with_padding(
    values,
    maximize: bool,
    exclude_worst: int = 0,
    include: float | None = None,
):
    """Axis limits from ``values`` with 5% padding; optional ``include`` keeps a baseline in range."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None, None
    if 0 < exclude_worst < v.size:
        order = np.argsort(v)
        if maximize:
            kept = v[order[exclude_worst:]]
        else:
            kept = v[order[: v.size - exclude_worst]]
    else:
        kept = v
    lo, hi = float(np.min(kept)), float(np.max(kept))
    if include is not None and not (isinstance(include, float) and np.isnan(include)):
        lo = min(lo, float(include))
        hi = max(hi, float(include))
    span = hi - lo
    if span <= 0:
        lo -= 1e-6
        hi += 1e-6
    else:
        pad = 0.05 * span
        lo -= pad
        hi += pad
    return lo, hi


DEFAULT_BASELINE_JSON = Path(__file__).resolve().parent / "results" / "baseline_objectives_52923746.json"


def _default_baseline_json_path() -> Path | None:
    """``results/baseline_objectives_52923746.json`` if present."""
    return DEFAULT_BASELINE_JSON if DEFAULT_BASELINE_JSON.is_file() else None


def plot_one_pair_separate(
    df,
    x_name,
    y_name,
    out_path,
    baseline,
    dpi=150,
    show=False,
    exclude_worst: int = 0,
):
    x_col, y_col = _col(x_name), _col(y_name)
    obj_x = (x_name, _OBJ_MAP[x_name])
    obj_y = (y_name, _OBJ_MAP[y_name])

    trials = _trial_series(df)
    vmin = float(trials.min())
    vmax = float(trials.max())
    if vmin == vmax:
        vmax = vmin + 1.0

    bx = by = None
    if baseline and x_col in baseline and y_col in baseline:
        bx, by = float(baseline[x_col]), float(baseline[y_col])

    xlim = _axis_limits_with_padding(
        df[x_col].to_numpy(),
        maximize=obj_x[1],
        exclude_worst=exclude_worst,
        include=bx,
    )
    ylim = _axis_limits_with_padding(
        df[y_col].to_numpy(),
        maximize=obj_y[1],
        exclude_worst=exclude_worst,
        include=by,
    )

    _apply_plot_style()
    fig = plt.figure(figsize=(6.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    sc = ax.scatter(
        df[x_col],
        df[y_col],
        c=trials,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=48,
        alpha=0.8,
        edgecolors="none",
        zorder=1,
    )
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Trial #", fontsize=9)
    cax.tick_params(labelsize=8)

    pareto_ix = get_pareto_front_indices(df, [obj_x, obj_y])
    p_df = df.loc[pareto_ix].sort_values(x_col)
    if len(p_df) > 0:
        ax.plot(
            p_df[x_col].values,
            p_df[y_col].values,
            color="black",
            linewidth=2.0,
            zorder=6,
            label=None,
        )
        ax.scatter(
            p_df[x_col],
            p_df[y_col],
            facecolors="none",
            edgecolors="black",
            s=50,
            linewidths=1.25,
            zorder=6,
        )

    legend_handles = []
    if bx is not None and by is not None:
        ax.scatter(
            [bx],
            [by],
            marker="*",
            s=220,
            c="red",
            edgecolors="none",
            zorder=10,
        )
        legend_handles.append(
            mlines.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="red",
                markersize=12,
                linestyle="None",
                label="Baseline",
            )
        )
    if len(p_df) > 0:
        legend_handles.append(
            mlines.Line2D([0], [0], color="black", linewidth=2, label="Pareto front"),
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)

    ax.set_xlabel(AXIS_LABEL.get(x_name, x_name.replace("_", " ")))
    ax.set_ylabel(AXIS_LABEL.get(y_name, y_name.replace("_", " ")))
    if xlim[0] is not None:
        ax.set_xlim(xlim)
    if ylim[0] is not None:
        ax.set_ylim(ylim)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print("Saved {} and {}".format(out_path, pdf_path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG/PDF outputs (default: same folder as CSV)",
    )
    p.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Filename prefix for separate PNGs (default: CSV parent folder name)",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="JSON with performance_metric, avg_resource, clock_cycles, bops (default: results/baseline_objectives_52923746.json)",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not load or plot a baseline marker (ignore default file and --baseline-json).",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--exclude-worst",
        type=int,
        default=0,
        metavar="N",
        help="When computing axis limits only, ignore the worst N points per axis (0 = use full data + baseline).",
    )
    p.add_argument(
        "--mode",
        choices=("cycle", "star", "all"),
        default="cycle",
        help="cycle|star = Accuracy on y (same plots); all = classic six-panel figure",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Legacy: single 2×2 PNG via plot_pareto_pairs_subplots (use --out)",
    )
    p.add_argument("--out", type=Path, default=None, help="With --combined or --mode all + rename")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    names = [n for n, _ in DEFAULT_OBJECTIVE_INFO]
    missing = [n for n in names if _col(n) not in df.columns]
    if missing:
        sys.stderr.write("CSV missing columns for objectives: {}\n".format(missing))
        sys.stderr.write("Have: {}\n".format(list(df.columns)))
        return 2

    baseline = None
    if not args.no_baseline:
        baseline_path = args.baseline_json
        if baseline_path is None:
            baseline_path = _default_baseline_json_path()
        elif not baseline_path.is_file():
            sys.stderr.write("Warning: --baseline-json not a file: {}\n".format(baseline_path))
            baseline_path = None
        if baseline_path is not None and baseline_path.is_file():
            with open(baseline_path, encoding="utf-8") as f:
                baseline = json.load(f)
            print("Using baseline JSON: {}".format(baseline_path.resolve()))

    out_dir = Path(args.output_dir) if args.output_dir is not None else args.csv.parent
    prefix = args.name_prefix if args.name_prefix else args.csv.parent.name

    if args.combined:
        pairs = LEGACY_GRID_PAIRS
        out_path = args.out or (out_dir / "{}_pareto_grid.png".format(_safe_filename_part(prefix)))
        plot_pareto_pairs_subplots(
            df,
            DEFAULT_OBJECTIVE_INFO,
            pairs,
            str(out_path),
            figsize_per=(6, 5),
            show=args.show,
        )
        return 0

    if args.mode == "all":
        save_dir = str(out_dir)
        if args.out:
            plot_pareto_fronts(df, DEFAULT_OBJECTIVE_INFO, save_dir=save_dir, show=args.show)
            default_png = os.path.join(save_dir, "pareto_fronts_2d.png")
            if os.path.abspath(str(args.out)) != os.path.abspath(default_png):
                os.replace(default_png, args.out)
                print("Renamed output to {}".format(args.out))
        else:
            plot_pareto_fronts(df, DEFAULT_OBJECTIVE_INFO, save_dir=save_dir, show=args.show)
        return 0

    pairs = []
    for xn, yn in DEFAULT_FOUR_PAIRS:
        if _col(xn) not in df.columns or _col(yn) not in df.columns:
            continue
        pairs.append((xn, yn))
    if not pairs:
        sys.stderr.write("No valid plot pairs (check CSV columns).\n")
        return 2

    for i, (xn, yn) in enumerate(pairs, start=1):
        stem = "{:02d}_{}_vs_{}".format(
            i,
            _safe_filename_part(xn),
            _safe_filename_part(yn),
        )
        png = out_dir / "{}_{}.png".format(_safe_filename_part(prefix), stem)
        plot_one_pair_separate(
            df,
            xn,
            yn,
            png,
            baseline=baseline,
            dpi=args.dpi,
            show=args.show,
            exclude_worst=args.exclude_worst,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
