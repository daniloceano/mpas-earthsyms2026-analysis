#!/usr/bin/env python
"""Minimal quick-look: LPI fixed-LiDAR time series (in-situ observations).

Loads ``data/LPI_processed.csv.gz`` (10-min profiles, 6 heights 10-200 m
AGL; see data/README.md for how it was derived from the raw spreadsheet),
and writes one exploratory PNG with:

  1. wind speed at every height, with sim_2022's analysis window shaded
     (this is the only simulation the deployment overlaps — see data/README.md);
  2. wind direction at 100 m AGL;
  3. sensor-level air temperature and pressure.

Unlike P0's LiDAR (MATLAB struct + a per-height 'avail' completeness field),
this file is already the QC-"screened" sheet with plain NaN gaps, so no
extra masking threshold is needed here.

This is a sanity check ahead of a proper model-vs-obs comparison, not a
scientific analysis.

    python scripts/exploratory/quicklook_lidar_lpi.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"

PLOT_HEIGHTS_M = [10, 26, 50, 100, 150, 200]
WDIR_HEIGHT_M = 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("LPI")

    csv_path = cfg.REPO_ROOT / site["data_file"]
    if not csv_path.exists():
        print(f"missing observation file: {csv_path}", file=sys.stderr)
        return 1

    print(f"[LPI] reading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["time"])
    times = df["time"]

    n_total = sum(int(df[f"spd_{h}"].notna().sum()) for h in PLOT_HEIGHTS_M) \
        + sum(int(df[f"spd_{h}"].isna().sum()) for h in PLOT_HEIGHTS_M)
    n_masked = sum(int(df[f"spd_{h}"].isna().sum()) for h in PLOT_HEIGHTS_M)
    print(f"[LPI] {len(times)} records, {times.min()} to {times.max()}, "
          f"heights {min(PLOT_HEIGHTS_M)}-{max(PLOT_HEIGHTS_M)} m")
    print(f"[LPI] {n_masked}/{n_total} height-time bins are gaps (NaN)")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    for h in PLOT_HEIGHTS_M:
        ax.plot(times, df[f"spd_{h}"], lw=0.8, label=f"{h} m")
    sim = config.simulation(site["overlaps_simulation"])
    ax.axvspan(pd.Timestamp(sim.analysis_start), pd.Timestamp(sim.analysis_end),
               color="grey", alpha=0.15, label=f"{sim.key} analysis window")
    ax.set_ylabel("wind speed (m s$^{-1}$)")
    ax.legend(ncol=7, fontsize=8, loc="upper center")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(times, df[f"dir_{WDIR_HEIGHT_M}"], s=2, alpha=0.4)
    ax.set_ylabel(f"wind dir @ {WDIR_HEIGHT_M} m (deg)")
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(times, df["temp_10"], lw=0.8, color="tab:red", label="air temp (degC)")
    ax.set_ylabel("air temperature (degC)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(times, df["pres_10"], lw=0.8, color="tab:blue", label="pressure (mbar)")
    ax2.set_ylabel("pressure (mbar)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f"{site['label']} ({site['lat']:.4f}, {site['lon']:.4f}) — "
        f"{times.min():%Y-%m-%d} to {times.max():%Y-%m-%d}, "
        f"{site['temporal_resolution']} records"
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / "quicklook_lidar_lpi.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
