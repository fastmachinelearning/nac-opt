#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import shutil
from typing import Dict, Iterable, List


class TrialRow(object):
    __slots__ = (
        "trial",
        "performance_metric",
        "bops",
        "yaml_path",
        "lut_pct",
        "ff_pct",
        "bram_pct",
        "dsp_pct",
        "avg_resource",
        "clock_cycles",
    )

    def __init__(
        self,
        trial,
        performance_metric,
        bops,
        yaml_path,
        lut_pct,
        ff_pct,
        bram_pct,
        dsp_pct,
        avg_resource,
        clock_cycles,
    ):
        self.trial = int(trial)
        self.performance_metric = float(performance_metric)
        self.bops = float(bops)
        self.yaml_path = str(yaml_path)
        self.lut_pct = float(lut_pct)
        self.ff_pct = float(ff_pct)
        self.bram_pct = float(bram_pct)
        self.dsp_pct = float(dsp_pct)
        self.avg_resource = float(avg_resource)
        self.clock_cycles = float(clock_cycles)


METRICS = {
    "lut_pct": "LUT (%)",
    "ff_pct": "FF (%)",
    "bram_pct": "BRAM (%)",
    "dsp_pct": "DSP (%)",
    "avg_resource": "Average resources (%)",
    "clock_cycles": "Clock cycles",
}

MAXIMIZE = {"performance_metric"}


def _to_float(row, key):
    try:
        return float(row[key])
    except KeyError as e:
        raise KeyError(f"Missing required column '{key}' in CSV.") from e
    except ValueError as e:
        raise ValueError(f"Could not parse '{key}' as float: {row.get(key)!r}") from e


def _to_int(row, key):
    try:
        return int(row[key])
    except KeyError as e:
        raise KeyError(f"Missing required column '{key}' in CSV.") from e
    except ValueError as e:
        raise ValueError(f"Could not parse '{key}' as int: {row.get(key)!r}") from e


def read_trials(csv_path):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"trial", "performance_metric", "bops", "yaml_path", *METRICS.keys()}
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise KeyError(f"CSV missing required columns: {sorted(missing)}")

        rows = []
        for r in reader:
            rows.append(
                TrialRow(
                    trial=_to_int(r, "trial"),
                    performance_metric=_to_float(r, "performance_metric"),
                    bops=_to_float(r, "bops"),
                    yaml_path=r["yaml_path"],
                    lut_pct=_to_float(r, "lut_pct"),
                    ff_pct=_to_float(r, "ff_pct"),
                    bram_pct=_to_float(r, "bram_pct"),
                    dsp_pct=_to_float(r, "dsp_pct"),
                    avg_resource=_to_float(r, "avg_resource"),
                    clock_cycles=_to_float(r, "clock_cycles"),
                )
            )
    return rows


def top_k_lowest(
    rows,
    metric,
    k,
):
    return sorted(rows, key=lambda r: getattr(r, metric))[:k]

def top_k_highest(rows, metric, k):
    return sorted(rows, key=lambda r: getattr(r, metric), reverse=True)[:k]

def best_one(rows, metric, maximize=False):
    if not rows:
        return None
    if maximize:
        return max(rows, key=lambda r: getattr(r, metric))
    return min(rows, key=lambda r: getattr(r, metric))

def resolve_yaml_path(csv_path, yaml_path_str):
    p = Path(yaml_path_str)
    if p.is_absolute():
        return p
    # Many CSVs store paths like "./results/..."; those are relative to the tutorial directory,
    # not the CSV directory (which is already under results/...).
    parts = p.parts
    if len(parts) >= 1 and (parts[0] == "results" or (parts[0] == "." and len(parts) >= 2 and parts[1] == "results")):
        tutorial_dir = csv_path.parent.parent.parent  # .../tutorial_3_qubit
        candidate = (tutorial_dir / p).resolve()
        if candidate.exists():
            return candidate

    # Otherwise, try relative to the CSV directory.
    candidate = (csv_path.parent / p).resolve()
    if candidate.exists():
        return candidate

    # Last resort: resolve from current working directory.
    return p.resolve()

def export_yaml(src_path, dest_dir, dest_name):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_name
    shutil.copyfile(str(src_path), str(dest_path))
    return dest_path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Print best trial numbers among those with performance_metric >= threshold, "
            "selecting the k lowest trials per metric."
        )
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "results/qubit_optuna_job_52893384/block_search_results.csv"
        ),
        help="Path to block_search_results.csv (default: paper run).",
    )
    ap.add_argument(
        "--min-accuracy",
        type=float,
        default=0.95,
        help="Minimum performance_metric (accuracy/fidelity) threshold.",
    )
    ap.add_argument(
        "--no-accuracy-floor",
        action="store_true",
        help="Ignore --min-accuracy and consider all trials.",
    )
    ap.add_argument(
        "-k",
        type=int,
        default=2,
        help="How many trials to print per metric.",
    )
    ap.add_argument(
        "--print-one-each",
        action="store_true",
        help=(
            "Print exactly one selection for: highest accuracy overall; "
            "lowest per metric with accuracy floor; lowest per metric with no floor."
        ),
    )
    ap.add_argument(
        "--export-yamls",
        action="store_true",
        help="When used with --print-one-each, copy each selected trial YAML to a named file.",
    )
    ap.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write exported YAMLs. "
            "Default: <job_dir>/models_to_train (job_dir is the CSV's directory)."
        ),
    )
    args = ap.parse_args()

    rows = read_trials(args.csv)
    if args.no_accuracy_floor:
        eligible = list(rows)
    else:
        eligible = [r for r in rows if r.performance_metric >= args.min_accuracy]

    print(f"CSV: {args.csv}")
    if args.no_accuracy_floor:
        print(f"Eligible trials: {len(eligible)} / {len(rows)} (no accuracy floor)")
    else:
        print(f"Eligible trials: {len(eligible)} / {len(rows)} (performance_metric >= {args.min_accuracy})")
    if not eligible:
        return 0

    if args.print_one_each:
        export_dir = args.export_dir if args.export_dir is not None else (args.csv.parent / "models_to_train")

        # 1) highest accuracy overall (ignores floor)
        best_acc_all = best_one(rows, metric="performance_metric", maximize=True)
        print("\n1) Highest accuracy (overall):")
        print(
            "  trial={trial}  performance_metric={pm:.6f}".format(
                trial=best_acc_all.trial, pm=best_acc_all.performance_metric
            )
        )
        if args.export_yamls:
            src = resolve_yaml_path(args.csv, best_acc_all.yaml_path)
            export_yaml(src, export_dir, "best_highest_accuracy.yaml")

        # 2-7) with accuracy floor: lowest per metric
        floor_rows = [r for r in rows if r.performance_metric >= args.min_accuracy]
        print("\nAccuracy floor: performance_metric >= {v}".format(v=args.min_accuracy))
        print("Eligible trials under floor: {n} / {tot}".format(n=len(floor_rows), tot=len(rows)))
        if floor_rows:
            order = ["lut_pct", "ff_pct", "bram_pct", "dsp_pct", "avg_resource", "clock_cycles"]
            names = {
                "lut_pct": "best_low_lut_min_acc.yaml",
                "ff_pct": "best_low_ff_min_acc.yaml",
                "bram_pct": "best_low_bram_min_acc.yaml",
                "dsp_pct": "best_low_dsp_min_acc.yaml",
                "avg_resource": "best_low_avg_resource_min_acc.yaml",
                "clock_cycles": "best_low_clock_cycles_min_acc.yaml",
            }
            start_idx = 2
            for i, metric in enumerate(order):
                r = best_one(floor_rows, metric=metric, maximize=False)
                print(
                    "{idx}) Lowest {metric}: trial={trial}  {metric}={val:g}  performance_metric={pm:.6f}".format(
                        idx=start_idx + i,
                        metric=metric,
                        trial=r.trial,
                        val=getattr(r, metric),
                        pm=r.performance_metric,
                    )
                )
                if args.export_yamls:
                    src = resolve_yaml_path(args.csv, r.yaml_path)
                    export_yaml(src, export_dir, names[metric])
        else:
            print("  (No trials meet the accuracy floor.)")

        # 8-13) no accuracy floor: lowest per metric
        print("\nNo accuracy floor:")
        order = ["lut_pct", "ff_pct", "bram_pct", "dsp_pct", "avg_resource", "clock_cycles"]
        names = {
            "lut_pct": "best_low_lut_no_floor.yaml",
            "ff_pct": "best_low_ff_no_floor.yaml",
            "bram_pct": "best_low_bram_no_floor.yaml",
            "dsp_pct": "best_low_dsp_no_floor.yaml",
            "avg_resource": "best_low_avg_resource_no_floor.yaml",
            "clock_cycles": "best_low_clock_cycles_no_floor.yaml",
        }
        start_idx = 8
        for i, metric in enumerate(order):
            r = best_one(rows, metric=metric, maximize=False)
            print(
                "{idx}) Lowest {metric}: trial={trial}  {metric}={val:g}  performance_metric={pm:.6f}".format(
                    idx=start_idx + i,
                    metric=metric,
                    trial=r.trial,
                    val=getattr(r, metric),
                    pm=r.performance_metric,
                )
            )
            if args.export_yamls:
                src = resolve_yaml_path(args.csv, r.yaml_path)
                export_yaml(src, export_dir, names[metric])
        return 0

    best_acc = top_k_highest(eligible, metric="performance_metric", k=args.k)
    print(f"\nAccuracy (highest {args.k} among eligible):")
    for rank, r in enumerate(best_acc, start=1):
        print(f"  {rank}. trial={r.trial}  performance_metric={r.performance_metric:.6f}")

    for metric, label in METRICS.items():
        best = top_k_lowest(eligible, metric=metric, k=args.k)
        print(f"\n{label} (lowest {args.k} among eligible):")
        for rank, r in enumerate(best, start=1):
            val = getattr(r, metric)
            print(
                f"  {rank}. trial={r.trial}  {metric}={val:g}  performance_metric={r.performance_metric:.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

