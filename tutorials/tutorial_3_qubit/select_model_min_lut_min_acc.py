#!/usr/bin/env python3
"""
Pick a trial from block_search_results.csv: filter by minimum accuracy, then minimize LUT %.

Example:
  cd tutorials/tutorial_3_qubit
  python3 select_model_min_lut_min_acc.py \\
    --csv results/qubit_optuna_job_52893343/block_search_results.csv \\
    --min-acc 0.95

Optional tie-break (ascending): --tie clock_cycles
Copy chosen yaml: --copy-to ./my_best.yaml
"""
import sys

if sys.version_info[0] < 3:
    sys.stderr.write("Use Python 3, e.g. python3 select_model_min_lut_min_acc.py ...\n")
    sys.exit(2)

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True, help="Merged block_search_results.csv")
    p.add_argument("--min-acc", type=float, default=0.95, dest="min_acc", help="Min performance_metric (accuracy)")
    p.add_argument(
        "--lut-col",
        type=str,
        default="lut_pct",
        help="Column to minimize among feasible trials (default lut_pct)",
    )
    p.add_argument(
        "--tie",
        type=str,
        default="clock_cycles",
        help="Secondary sort ascending if lut ties (default clock_cycles)",
    )
    p.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="If set, copy the chosen trial yaml_path to this file",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "performance_metric" not in df.columns or args.lut_col not in df.columns:
        print("CSV must contain performance_metric and the LUT column.", file=sys.stderr)
        return 2

    feas = df[df["performance_metric"] >= args.min_acc].copy()
    if feas.empty:
        print(f"No trials with performance_metric >= {args.min_acc}")
        return 1

    tie = args.tie
    if tie not in feas.columns:
        feas = feas.sort_values(args.lut_col, ascending=True)
    else:
        feas = feas.sort_values([args.lut_col, tie], ascending=[True, True])

    row = feas.iloc[0]
    trial = int(row["trial"]) if "trial" in row and pd.notna(row["trial"]) else None
    print(f"Selected trial: {trial}")
    print(f"  performance_metric={row['performance_metric']:.6f}  {args.lut_col}={row[args.lut_col]}")
    if tie in row.index:
        print(f"  {tie}={row[tie]}")
    yp = row.get("yaml_path")
    print(f"  yaml_path: {yp}")

    def _resolve_yaml(src_field):
        if not src_field or not isinstance(src_field, str):
            return None
        raw = src_field.strip()
        p = Path(raw)
        tutorial_root = args.csv.parent.parent  # .../tutorial_3_qubit
        candidates = [
            p,
            Path.cwd() / p,
            tutorial_root / raw.lstrip("./"),
        ]
        for c in candidates:
            try:
                r = c.resolve()
            except OSError:
                continue
            if r.is_file():
                return r
        return None

    if args.copy_to:
        src = _resolve_yaml(str(yp))
        if src is None:
            print(f"[WARN] yaml not found for yaml_path={yp!r}", file=sys.stderr)
        else:
            args.copy_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, args.copy_to)
            print(f"Copied -> {args.copy_to.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
