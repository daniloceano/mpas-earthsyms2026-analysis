#!/usr/bin/env python
"""Poster Figure 1: native-mesh domain — topography over land, initial SST over ocean.

Draws every cell of the ``meqbr_05km`` Voronoi mesh once, no regridding: land
cells (``landmask == 1``, static file) are filled with terrain height (``ter``);
ocean cells are filled with the model's initial sea-surface temperature (first
time step of the run's ``sfc_update.nc``, i.e. the SST the run was initialized
with). Two independent colormaps/colorbars keep the two physical quantities
(m vs degC) visually distinct.

A small number of ocean cells near river mouths/estuaries (e.g. the Amazon/Pará
outflow) carry the sfc_update land-fill value (273.15 K) because they fall
outside the source SST product's water mask despite being flagged ocean in the
model's landmask. These are detected via ``--sst-min-valid`` (default 10 degC,
implausible for this equatorial domain) and drawn in flat gray rather than
polluting the color scale.

    python scripts/poster/fig01_domain_topo_sst.py --sim sim_2021
    python scripts/poster/fig01_domain_topo_sst.py --sim sim_2022
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "poster"

RAD2DEG = 180.0 / np.pi


def _decode_xtime(raw) -> str:
    if isinstance(raw, bytes):
        text = raw.decode()
    elif isinstance(raw, np.ndarray) and raw.dtype.kind == "S" and raw.ndim == 0:
        text = raw.item().decode()
    elif isinstance(raw, np.ndarray):
        text = b"".join(np.atleast_1d(raw).astype("S1").ravel()).decode()
    else:
        text = str(raw)
    return text.strip().strip("\x00").strip()


def build_cell_polygons(
    vertices_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    vert_lat: np.ndarray,
    vert_lon: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (polygon vertex arrays, cell indices) for every interior cell.

    MPAS connectivity is 1-based; a 0 entry marks an unused/border vertex slot
    (fewer than ``maxEdges`` sides, or a boundary cell without a full ring) —
    such cells are skipped, matching ``mpas_viz.py``'s convention.
    """
    n_cells = vertices_on_cell.shape[0]
    polygons: list[np.ndarray] = []
    valid_idx = np.empty(n_cells, dtype=np.int64)
    n_valid = 0

    for i in range(n_cells):
        n = int(n_edges_on_cell[i])
        vidx = vertices_on_cell[i, :n] - 1
        if (vidx < 0).any():
            continue
        lons = vert_lon[vidx]
        lats = vert_lat[vidx]
        if lons.max() > 170.0 and lons.min() < -170.0:
            lons = np.where(lons >= 170.0, lons - 360.0, lons)
        polygons.append(np.column_stack([lons, lats]))
        valid_idx[n_valid] = i
        n_valid += 1

    return polygons, valid_idx[:n_valid]


def land_colormap() -> mcolors.Colormap:
    """Land-only portion of 'terrain' (skips its low-end blue bathymetry band)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "land_terrain", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
    )


SST_PALETTE = [
    "#FFD133", "#FFAD00", "#FF8000", "#FA4C00",
    "#E61A00", "#BF001A", "#8C0033",
]


def sst_colormap() -> mcolors.Colormap:
    """Custom warm SST palette (yellow -> orange -> maroon)."""
    return mcolors.LinearSegmentedColormap.from_list("sst_custom", SST_PALETTE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", default="sim_2021", choices=["sim_2021", "sim_2022"])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/poster/)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--sst-min-valid",
        type=float,
        default=10.0,
        help="ocean SST below this (degC) is treated as a fill artifact "
        "(e.g. river-mouth cells outside the SST product's water mask) "
        "and drawn in flat gray, excluded from the color scale (default: 10.0)",
    )
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    sim = config.simulation(args.sim)
    mesh_name = config.mesh["name"]

    static_path = config.data_root / f"{mesh_name}.static.nc"
    sfc_path = sim.history_dir / f"{mesh_name}.sfc_update.nc"
    for p in (static_path, sfc_path):
        if not p.exists():
            print(f"missing required file: {p}", file=sys.stderr)
            return 1

    print(f"[{sim.key}] reading mesh + terrain: {static_path}")
    static = xr.open_dataset(static_path)
    landmask = static["landmask"].values
    ter = static["ter"].values
    vert_lat = np.rad2deg(static["latVertex"].values)
    vert_lon = np.rad2deg(static["lonVertex"].values)
    vert_lon = np.where(vert_lon > 180.0, vert_lon - 360.0, vert_lon)
    vertices_on_cell = static["verticesOnCell"].values
    n_edges_on_cell = static["nEdgesOnCell"].values

    print(f"[{sim.key}] reading initial SST: {sfc_path}")
    sfc = xr.open_dataset(sfc_path)
    sst0_k = sfc["sst"].isel(Time=0).values
    # Indexing the raw .values array (not xarray .isel(...).values) returns a
    # numpy.bytes_ scalar rather than a 0-d ndarray, which _decode_xtime needs.
    init_xtime = _decode_xtime(sfc["xtime"].values[0])
    sst0_c = sst0_k - 273.15
    static.close()
    sfc.close()

    print(f"[{sim.key}] building {landmask.size} cell polygons...")
    t0 = time.time()
    polygons, cell_idx = build_cell_polygons(
        vertices_on_cell, n_edges_on_cell, vert_lat, vert_lon
    )
    print(f"  {len(polygons)} interior cells in {time.time() - t0:.1f}s "
          f"({landmask.size - len(polygons)} border cells skipped)")

    is_land = landmask[cell_idx] == 1
    is_ocean = ~is_land
    sst_c = sst0_c[cell_idx]
    valid_sst = is_ocean & (sst_c >= args.sst_min_valid)
    fill_artifact = is_ocean & ~valid_sst
    print(f"  land cells: {is_land.sum()}  ocean cells: {is_ocean.sum()} "
          f"(of which {fill_artifact.sum()} below {args.sst_min_valid:.0f} degC "
          "-> treated as SST fill artifacts, drawn gray)")

    land_polys = [polygons[i] for i in np.where(is_land)[0]]
    land_values = ter[cell_idx][is_land]

    ocean_polys = [polygons[i] for i in np.where(valid_sst)[0]]
    ocean_values = sst_c[valid_sst]

    gray_polys = [polygons[i] for i in np.where(fill_artifact)[0]]

    cmap_land = land_colormap()
    norm_land = mcolors.Normalize(vmin=0.0, vmax=float(land_values.max()))

    cmap_ocean = sst_colormap()
    norm_ocean = mcolors.Normalize(vmin=25.0, vmax=float(ocean_values.max()))

    domain = config.mesh["domain"]
    margin = 0.15
    extent = [
        domain["lon_min"] - margin,
        domain["lon_max"] + margin,
        domain["lat_min"] - margin,
        domain["lat_max"] + margin,
    ]

    print(f"[{sim.key}] rendering...")
    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if gray_polys:
        gray_coll = PolyCollection(
            gray_polys, facecolors="lightgray", edgecolors="none",
            transform=ccrs.PlateCarree(), zorder=1,
        )
        ax.add_collection(gray_coll)

    land_coll = PolyCollection(
        land_polys, array=land_values, cmap=cmap_land, norm=norm_land,
        edgecolors="none", transform=ccrs.PlateCarree(), zorder=2,
    )
    ax.add_collection(land_coll)

    ocean_coll = PolyCollection(
        ocean_polys, array=ocean_values, cmap=cmap_ocean, norm=norm_ocean,
        edgecolors="none", transform=ccrs.PlateCarree(), zorder=2,
    )
    ax.add_collection(ocean_coll)

    ax.coastlines(resolution="10m", linewidth=0.4, color="black", zorder=3)
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle="--", linewidth=0.4)
    gl.top_labels = False
    gl.right_labels = False

    # In-situ observation sites (config/simulations.yaml: observations), for
    # later model-vs-obs comparison.
    for site_key, obs_site in config.observations.items():
        ax.plot(
            obs_site["lon"], obs_site["lat"], marker="*", markersize=18,
            markerfacecolor="black", markeredgecolor="white", markeredgewidth=1.2,
            transform=ccrs.PlateCarree(), zorder=6,
        )
        ax.annotate(
            site_key, xy=(obs_site["lon"], obs_site["lat"]),
            xytext=(6, 5), textcoords="offset points", fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black",
                      alpha=0.85, lw=0.5),
            zorder=7,
        )

    cbar_land = fig.colorbar(land_coll, ax=ax, shrink=0.5, pad=-0.05,
                             aspect=30, extend="max")
    cbar_land.set_label("Terrain height (m)")
    cbar_ocean = fig.colorbar(ocean_coll, ax=ax, shrink=0.5, pad=0.05,
                              aspect=30, extend="min")
    cbar_ocean.set_label("Initial SST (°C)")

    ax.set_title(
        f"MPAS {mesh_name} native mesh (~{config.mesh['mean_cell_spacing_km']:.1f} km) "
        f"— topography & initial SST\n{sim.label}  —  SST init: {init_xtime}",
        fontsize=12,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / f"fig01_domain_topo_sst_{sim.key}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
