#!/usr/bin/env python
"""Minimal wind quick-look: 10 m vs turbine-height (~50 m) wind speed.

Demonstrates the core conventions for these runs on a small time subset:
  * 10 m wind speed from ``u10``/``v10``;
  * deriving layer-center heights from ``zgrid`` interfaces (MPAS staggering);
  * locating the layer whose center is ~50 m above ground and reading the
    reconstructed wind there (NOT indexing zgrid/interface heights directly).

Writes one exploratory PNG (domain-mean wind-speed time series) to
figures/exploratory/. This is a sanity check, not a scientific analysis.

    python scripts/exploratory/quicklook_wind.py --sim sim_2021 --hours 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis import vertical  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"


def identify_50m_level(zgrid_values: np.ndarray, target_m: float) -> tuple[int, float]:
    """Return (level index, its AGL height) nearest *target_m* over an ocean cell.

    Uses the flattest (minimum-terrain) cell so the AGL profile matches the
    designed near-surface spacing cleanly.
    """
    zg = np.squeeze(zgrid_values)  # (nCells, nVertLevelsP1)
    agl = vertical.heights_above_ground(zg)  # (nCells, nVertLevels)
    ocean_cell = int(np.argmin(zg[:, 0]))
    k = vertical.nearest_level_index(agl[ocean_cell], target_m)
    return k, float(agl[ocean_cell, k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", default="sim_2021", help="simulation key")
    ap.add_argument("--hours", type=int, default=24, help="number of history files")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    sim = config.simulation(args.sim)
    target = float(config.vertical.get("turbine_level_height_m", 50.0))

    files = io.find_history_files(sim.history_dir)
    if not files:
        print(f"no history files in {sim.history_dir}", file=sys.stderr)
        return 1
    subset = files[: args.hours]
    times = io.read_timestamps(subset)

    # Static grid from the first file; find the ~50 m layer center.
    with io.open_history(subset[0], variables=["zgrid"]) as ds0:
        level, level_height = identify_50m_level(ds0["zgrid"].values, target)
    expected = config.vertical.get("turbine_level_index_python")
    print(
        f"[{sim.key}] layer center nearest {target:.0f} m AGL: python index "
        f"{level} at {level_height:.1f} m "
        f"(config expects 0-based index {expected})"
    )

    # Lazy time series over the subset.
    wind_vars = ["u10", "v10", "uReconstructZonal", "uReconstructMeridional"]
    with io.open_history(subset, variables=wind_vars, chunks={"Time": 1}) as ds:
        spd10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).mean("nCells")
        uz = ds["uReconstructZonal"].isel(nVertLevels=level)
        um = ds["uReconstructMeridional"].isel(nVertLevels=level)
        spd_hub = np.sqrt(uz**2 + um**2).mean("nCells")
        spd10 = spd10.compute().values  # domain-mean (unweighted; quick-look only)
        spd_hub = spd_hub.compute().values

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / f"quicklook_wind_{sim.key}.png"

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(times, spd10, label="10 m wind speed", lw=1.5)
    ax.plot(times, spd_hub, label=f"~{target:.0f} m wind speed (level {level})", lw=1.5)
    ax.set_ylabel("domain-mean wind speed (m s$^{-1}$)")
    ax.set_title(
        f"meqbr_05km {sim.label} — first {len(subset)} h "
        f"({times.min():%Y-%m-%d %H:%M} to {times.max():%Y-%m-%d %H:%M})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
