#!/usr/bin/env python
"""Minimal quick-look: P0 floating-LiDAR time series (in-situ observations).

Loads the MATLAB struct in ``data/P0_LIDAR_matrix.mat`` (10-min profiles,
20 heights from 40-260 m AGL), decodes MATLAB datenum to timestamps, masks
bins whose per-height ``avail`` (data completeness, %) falls below
``--min-avail``, and writes one exploratory PNG with:

  1. wind speed at a few representative heights, with sim_2021's analysis
     window shaded (this obs period sits fully inside it — see data/README.md);
  2. wind direction at ~100 m AGL;
  3. sensor-level air temperature and pressure.

This is a sanity check ahead of a proper model-vs-obs comparison, not a
scientific analysis.

    python scripts/exploratory/quicklook_lidar_p0.py
    python scripts/exploratory/quicklook_lidar_p0.py --min-avail 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import loadmat  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"

# Days between the MATLAB datenum epoch (0000-01-01) and the Unix epoch.
MATLAB_EPOCH_OFFSET_DAYS = 719529

# Representative heights to plot (all present in L.heights); 50 m matches
# this project's turbine-relevant level (config/simulations.yaml: vertical).
PLOT_HEIGHTS_M = [40, 50, 100, 200, 260]
WDIR_HEIGHT_M = 100


def matlab_datenum_to_timestamps(mtime: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.Timestamp("1970-01-01") + pd.to_timedelta(mtime - MATLAB_EPOCH_OFFSET_DAYS, unit="D")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument(
        "--min-avail", type=float, default=50.0,
        help="mask (NaN) height/time bins with 'avail' below this percent "
        "(default: 50.0)",
    )
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("P0")

    mat_path = cfg.REPO_ROOT / site["data_file"]
    if not mat_path.exists():
        print(f"missing observation file: {mat_path}", file=sys.stderr)
        return 1

    print(f"[P0] reading {mat_path}")
    L = loadmat(mat_path, simplify_cells=True)["L"]

    times = matlab_datenum_to_timestamps(L["mtime"])
    heights = np.asarray(L["heights"][0], dtype=int)
    avail = L["avail"]
    wspeed = np.where(avail < args.min_avail, np.nan, L["wspeed"])
    wdir = np.where(avail < args.min_avail, np.nan, L["wdir"])

    n_masked = int(np.isnan(wspeed).sum())
    n_total = wspeed.size
    print(f"[P0] {len(times)} records, {times.min()} to {times.max()}, "
          f"heights {heights.min()}-{heights.max()} m")
    print(f"[P0] masked {n_masked}/{n_total} height-time bins "
          f"(avail < {args.min_avail:.0f}%)")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    for h in PLOT_HEIGHTS_M:
        k = int(np.argmin(np.abs(heights - h)))
        ax.plot(times, wspeed[:, k], lw=0.8, label=f"{heights[k]} m")
    sim = config.simulation(site["overlaps_simulation"])
    ax.axvspan(pd.Timestamp(sim.analysis_start), pd.Timestamp(sim.analysis_end),
               color="grey", alpha=0.15, label=f"{sim.key} analysis window")
    ax.set_ylabel("wind speed (m s$^{-1}$)")
    ax.legend(ncol=6, fontsize=8, loc="upper center")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    k_wdir = int(np.argmin(np.abs(heights - WDIR_HEIGHT_M)))
    ax.scatter(times, wdir[:, k_wdir], s=2, alpha=0.4)
    ax.set_ylabel(f"wind dir @ {heights[k_wdir]} m (deg)")
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)

    # The met sensor (temp/press/humid) drops to an exact-0 fill value after a
    # dropout partway through the record, unlike wspeed/wdir which come from
    # the LiDAR itself and stay valid throughout — 0 hPa is unphysical, so mask
    # on pressure rather than plot it as if it were a real reading.
    invalid_met = L["press"] <= 0
    temp = np.where(invalid_met, np.nan, L["temp"])
    press = np.where(invalid_met, np.nan, L["press"])
    n_valid_met = int((~invalid_met).sum())
    print(f"[P0] sensor-level temp/press valid for {n_valid_met}/{len(times)} "
          "records (rest flagged invalid: met sensor dropout, not LiDAR)")

    ax = axes[2]
    ax.plot(times, temp, lw=0.8, color="tab:red", label="air temp (degC)")
    ax.set_ylabel("air temperature (degC)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    ax2.plot(times, press, lw=0.8, color="tab:blue", label="pressure (hPa)")
    ax2.set_ylabel("pressure (hPa)", color="tab:blue")
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
    out = args.output or FIG_DIR / "quicklook_lidar_p0.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
