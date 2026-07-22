#!/usr/bin/env python3
"""
Pick a global-search trial from block_search_results.csv and copy its architecture
YAML next to best_model_for_local_search.yaml for use with local search (ARCH_YAML).

Default selection: row with lowest lut_pct (ties: smallest trial id, then lowest avg_resource).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def _resolve_trial_yaml(results_dir: Path, trial: int) -> Path:
    """Architecture file written during global search for this Optuna trial."""
    return results_dir / "trial_yamls" / f"trial_{trial}_arch.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results_dir",
        type=Path,
        help="Global-search results folder containing block_search_results.csv and trial_yamls/",
    )
    p.add_argument(
        "--csv-name",
        default="block_search_results.csv",
        help="CSV filename under results_dir (default: block_search_results.csv)",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1 = lowest lut_pct, 2 = second-lowest, etc. (default: 1)",
    )
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="If set, require performance_metric >= this value before ranking by lut_pct.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Destination YAML path. Default: "
        "<results_dir>/best_model_for_local_search_trial_<trial>.yaml",
    )
    args = p.parse_args()

    results_dir = args.results_dir.resolve()
    csv_path = results_dir / args.csv_name
    if not csv_path.is_file():
        print(f"[error] Missing CSV: {csv_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(csv_path)
    required = {"trial", "lut_pct"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    work = df.dropna(subset=["lut_pct"]).copy()
    if work.empty:
        print("[error] No rows with non-null lut_pct", file=sys.stderr)
        return 2

    if args.min_accuracy is not None:
        if "performance_metric" not in work.columns:
            print(
                "[error] --min-accuracy requires 'performance_metric' column in the CSV",
                file=sys.stderr,
            )
            return 2
        work = work[pd.to_numeric(work["performance_metric"], errors="coerce") >= float(args.min_accuracy)]
        if work.empty:
            print(
                f"[error] No rows meet performance_metric >= {float(args.min_accuracy):g} with non-null lut_pct",
                file=sys.stderr,
            )
            return 2

    work = work.sort_values(
        ["lut_pct", "trial"] + (["avg_resource"] if "avg_resource" in work.columns else []),
        ascending=[True, True] + ([True] if "avg_resource" in work.columns else []),
    ).reset_index(drop=True)

    rank = args.rank
    if rank < 1 or rank > len(work):
        print(
            f"[error] --rank {rank} out of range (1..{len(work)} feasible rows)",
            file=sys.stderr,
        )
        return 2

    row = work.iloc[rank - 1]
    trial = int(row["trial"])
    src = _resolve_trial_yaml(results_dir, trial)
    if not src.is_file():
        print(f"[error] Trial YAML not found: {src}", file=sys.stderr)
        return 2

    if args.output is not None:
        dst = args.output.expanduser().resolve()
    else:
        dst = results_dir / f"best_model_for_local_search_trial_{trial}.yaml"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    lut = float(row["lut_pct"])
    perf = row.get("performance_metric")
    perf_s = f"{float(perf):.6f}" if pd.notna(perf) else "n/a"
    print(f"Selected trial {trial} (lut_pct rank {rank} among non-null rows): lut_pct={lut:g}, performance_metric={perf_s}")
    print(f"  Source: {src}")
    print(f"  Wrote:  {dst}")
    print()
    print("Local search (example):")
    print(
        f'  sbatch --export=ALL,GLOBAL_RESULTS_DIR={results_dir},'
        f"ARCH_YAML={dst},LOCAL_SEARCH_USE_SLURM_JOB_ID=1 "
        f"run_local_search_slurm.sh"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
