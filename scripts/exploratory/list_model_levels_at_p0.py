#!/usr/bin/env python
"""List MPAS layer-center heights (AGL) at the cell nearest an observation site.

MPAS is terrain-following, so level heights above ground are per-cell, not a
fixed global table — this reads ``zgrid`` from the nearest ocean cell to the
site and derives AGL layer-center heights (see docs/analysis_conventions.md:
"Vertical grid"), so they can be lined up against the instrument's fixed
measurement heights (config/simulations.yaml: observations.<site>.heights_m).

    python scripts/exploratory/list_model_levels_at_p0.py
    python scripts/exploratory/list_model_levels_at_p0.py --site LPI --sim sim_2022 --max-height 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import vertical  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="P0", help="observation site key in "
                     "config/simulations.yaml (default: P0)")
    ap.add_argument("--sim", default=None, help="simulation key (default: the "
                     "site's overlaps_simulation)")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--max-height", type=float, default=300.0,
                     help="print levels up to this AGL height in m (default: 300)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation(args.site)
    sim = config.simulation(args.sim or site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    init_path = sim.history_dir / f"{mesh_name}.init.nc"
    if not init_path.exists():
        print(f"missing init file: {init_path}", file=sys.stderr)
        return 1

    ds = xr.open_dataset(init_path)
    lat = np.rad2deg(ds["latCell"].values)
    lon = np.rad2deg(ds["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)

    dist2 = (lat - site["lat"]) ** 2 + (lon - site["lon"]) ** 2
    icell = int(np.argmin(dist2))
    dist_km = float(np.sqrt(dist2[icell])) * 111.0
    is_land = bool(ds["landmask"].values[icell])

    print(f"[{sim.key}] nearest cell to {site['label']} "
          f"({site['lat']:.4f}, {site['lon']:.4f}): "
          f"idx={icell}, ({lat[icell]:.4f}, {lon[icell]:.4f}), "
          f"~{dist_km:.2f} km away, landmask={'land' if is_land else 'ocean'}")
    if is_land:
        print("WARNING: nearest cell is land, not ocean — unexpected for a "
              "floating LiDAR site; check the coordinates.", file=sys.stderr)

    zgrid = ds["zgrid"].values[icell]
    surface_m = float(zgrid[0])
    centers_agl = vertical.layer_center_heights(zgrid) - surface_m
    print(f"terrain (zgrid[0]) at this cell: {surface_m:.2f} m MSL "
          f"({ds.sizes['nVertLevels']} layers total)\n")

    obs_heights = set(site["heights_m"])
    print(f"{'idx(py)':>8} {'idx(fortran)':>13} {'height AGL (m)':>15}  matches LiDAR height?")
    for k, h in enumerate(centers_agl):
        if h > args.max_height:
            break
        flag = "  <-- exact match" if round(float(h), 1) in obs_heights else ""
        print(f"{k:8d} {k + 1:13d} {h:15.2f}{flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
