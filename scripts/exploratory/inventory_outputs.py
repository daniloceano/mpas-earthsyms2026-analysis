#!/usr/bin/env python
"""Inventory the retained history output of both simulations.

Reports, per simulation: number of history files, first/last valid time, whether
hourly coverage is complete (no gaps/duplicates), and which required variables
are present. Also serves as the lightweight path-validation command — it fails
loudly if a configured run directory is missing.

Only tiny metadata (``xtime``, variable names) is read; no data fields are
loaded. Run:

    python scripts/exploratory/inventory_outputs.py
    python scripts/exploratory/inventory_outputs.py --data-root /other/path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402


def check_hourly_coverage(timestamps: pd.DatetimeIndex) -> dict:
    """Summarize whether *timestamps* form a complete, gap-free hourly series."""
    ts = timestamps.sort_values()
    diffs = ts.to_series().diff().dropna()
    hour = pd.Timedelta(hours=1)
    return {
        "duplicates": int(ts.duplicated().sum()),
        "gaps": int((diffs > hour).sum()),
        "irregular": int((diffs != hour).sum()),
        "complete": bool(ts.is_unique and (diffs == hour).all()),
    }


def inventory_simulation(sim: cfg.Simulation, required: list[str]) -> bool:
    """Print an inventory block for one simulation. Return False on any problem."""
    print(f"\n=== {sim.key}  ({sim.label}) ===")
    print(f"  history_dir: {sim.history_dir}")
    if not sim.history_dir.is_dir():
        print("  MISSING directory — cannot inventory.")
        return False

    files = io.find_history_files(sim.history_dir)
    print(f"  history files: {len(files)}  (expected {sim.expected_history_records})")
    if not files:
        print("  no history.*.nc found.")
        return False

    timestamps = io.read_timestamps(files)
    print(f"  first: {timestamps.min()}   last: {timestamps.max()}")

    cov = check_hourly_coverage(timestamps)
    verdict = "complete hourly series" if cov["complete"] else "INCOMPLETE"
    print(
        f"  coverage: {verdict} "
        f"(gaps={cov['gaps']}, duplicates={cov['duplicates']})"
    )
    count_ok = len(files) == sim.expected_history_records

    present = set(io.list_variables(files[0]))
    missing = [v for v in required if v not in present]
    print("  required variables:")
    for v in required:
        print(f"    [{'ok' if v in present else 'MISSING'}] {v}")

    return cov["complete"] and count_ok and not missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument(
        "--paths", type=Path, help="alternative paths.local.yaml (ignored if --data-root)"
    )
    ap.add_argument(
        "sims", nargs="*", help="simulation keys to inventory (default: all)"
    )
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)

    problems = config.validate()
    if problems:
        print("Path validation problems:")
        for p in problems:
            print(f"  - {p}")

    keys = args.sims or list(config.simulations)
    all_ok = True
    for key in keys:
        ok = inventory_simulation(config.simulation(key), config.required_variables)
        all_ok = all_ok and ok

    print(f"\nOverall: {'OK' if all_ok and not problems else 'problems found'}")
    return 0 if all_ok and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
