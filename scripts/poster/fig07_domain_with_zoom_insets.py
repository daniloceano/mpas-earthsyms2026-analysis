#!/usr/bin/env python
"""Poster Figure 7: native-mesh domain (terrain + SST) with P0 and LPI zoom
panels showing the individual Voronoi cells at each observation site.

Left panel: full domain — land cells coloured by terrain height, ocean cells
by initial SST (same sources as fig01). Right column: two stacked zoom panels,
one per LiDAR site (fig03 / fig05 geometry), showing the cell-level Voronoi
mesh and highlighting the nearest-neighbour cell used for model comparison.
A coloured rectangle on the main map indicates each site's zoom extent.

Because the two simulations share the same mesh, terrain geometry is identical
across both sims; SST is taken from the sim specified by ``--sim`` (default:
sim_2021, which covers P0). The LPI zoom panel also uses static terrain only.

    python scripts/poster/fig07_domain_with_zoom_insets.py
    python scripts/poster/fig07_domain_with_zoom_insets.py --sim sim_2022
    python scripts/poster/fig07_domain_with_zoom_insets.py --margin-km 20
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
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "poster"
RAD2DEG = 180.0 / np.pi
KM_PER_DEG_LAT = 111.0

# Poster font sizes
TITLE_FS = 12
LABEL_FS = 10
TICK_FS = 9

# Per-site zoom box colours (must be visually distinct on the main map)
SITE_COLORS = {"P0": "#e6194b", "LPI": "#3cb44b"}


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def build_all_polygons(
    vertices_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    vert_lat: np.ndarray,
    vert_lon: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Voronoi polygons for the full mesh; border cells (any 0-vertex) skipped.

    Returns (polygons, valid_cell_indices).
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
        # Collapse antimeridian-spanning cells (shouldn't occur for this domain)
        if lons.max() > 170.0 and lons.min() < -170.0:
            lons = np.where(lons >= 170.0, lons - 360.0, lons)
        polygons.append(np.column_stack([lons, lats]))
        valid_idx[n_valid] = i
        n_valid += 1
    return polygons, valid_idx[:n_valid]


def build_box_polygons(
    box_idx: np.ndarray,
    polygons_all: list[np.ndarray],
    valid_idx: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Subset of polygons for cells whose global index is in box_idx.

    Returns (box_polygons, local_valid_mask) where local_valid_mask selects
    which elements of box_idx have a polygon (i.e. are not border cells).
    """
    global_to_local = {int(v): i for i, v in enumerate(valid_idx)}
    box_polys: list[np.ndarray] = []
    local_mask = np.zeros(len(box_idx), dtype=bool)
    for k, gi in enumerate(box_idx):
        li = global_to_local.get(int(gi))
        if li is not None:
            box_polys.append(polygons_all[li])
            local_mask[k] = True
    return box_polys, local_mask


def land_colormap() -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "land_terrain", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
    )


def sst_colormap() -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "sst_custom",
        ["#FFD133", "#FFAD00", "#FF8000", "#FA4C00",
         "#E61A00", "#BF001A", "#8C0033"],
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_domain(
    ax: "matplotlib.axes.Axes",
    polygons: list[np.ndarray],
    valid_idx: np.ndarray,
    landmask: np.ndarray,
    ter: np.ndarray,
    sst_c: np.ndarray,
    sst_min_valid: float,
) -> tuple:
    """Draw full-domain terrain + SST on a cartopy axes. Returns colormappables."""
    is_land = landmask[valid_idx] == 1
    is_ocean = ~is_land
    valid_sst = is_ocean & (sst_c[valid_idx] >= sst_min_valid)
    fill_artifact = is_ocean & ~valid_sst

    land_polys = [p for p, m in zip(polygons, is_land) if m]
    land_values = ter[valid_idx][is_land]
    ocean_polys = [p for p, m in zip(polygons, valid_sst) if m]
    ocean_values = sst_c[valid_idx][valid_sst]
    gray_polys = [p for p, m in zip(polygons, fill_artifact) if m]

    cmap_land = land_colormap()
    norm_land = mcolors.Normalize(vmin=0.0, vmax=float(land_values.max()))
    cmap_ocean = sst_colormap()
    norm_ocean = mcolors.Normalize(vmin=25.0, vmax=float(ocean_values.max()))

    if gray_polys:
        ax.add_collection(PolyCollection(
            gray_polys, facecolors="lightgray", edgecolors="none",
            transform=ccrs.PlateCarree(), zorder=1,
        ))
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
    return land_coll, ocean_coll


def draw_zoom_panel(
    ax: "matplotlib.axes.Axes",
    polygons: list[np.ndarray],
    valid_idx: np.ndarray,
    landmask: np.ndarray,
    ter: np.ndarray,
    site: dict,
    site_key: str,
    icell: int,
    cell_lat: float,
    cell_lon: float,
    color: str,
) -> None:
    """Render a site zoom on a plain (non-cartopy) axes.

    Polygons are in lon/lat; the axes xlim/ylim must already be set to the
    zoom extent so the PolyCollection coordinates map directly.
    """
    is_land = landmask[valid_idx] == 1
    land_polys = [p for p, m in zip(polygons, is_land) if m]
    land_values = ter[valid_idx][is_land]
    ocean_polys = [p for p, m in zip(polygons, ~is_land) if m]

    cmap_land = land_colormap()
    norm_land = mcolors.Normalize(vmin=0.0, vmax=max(float(land_values.max()), 1.0))

    if ocean_polys:
        ax.add_collection(PolyCollection(
            ocean_polys, facecolors="#cfe8f3", edgecolors="grey",
            linewidths=0.3, zorder=1,
        ))
    if land_polys:
        land_coll = PolyCollection(
            land_polys, array=land_values, cmap=cmap_land, norm=norm_land,
            edgecolors="grey", linewidths=0.3, zorder=1,
        )
        ax.add_collection(land_coll)

    # Highlight the model comparison cell (outline only, no fill)
    # Find polygon for icell in the valid_idx of this zoom subset
    local_positions = np.where(valid_idx == icell)[0]
    if len(local_positions) > 0:
        highlight_poly = polygons[local_positions[0]]
        ax.add_collection(PolyCollection(
            [highlight_poly], facecolor="none", edgecolor=color,
            linewidth=2.0, zorder=3,
        ))
        ax.annotate(
            f"MPAS cell\n(~{np.sqrt((cell_lat - site['lat'])**2 + (cell_lon - site['lon'])**2) * KM_PER_DEG_LAT:.1f} km)",
            xy=(cell_lon, cell_lat), xytext=(8, -18),
            textcoords="offset points", fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.9, lw=0.5),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
            zorder=5,
        )

    ax.plot(site["lon"], site["lat"], marker="*", markersize=14,
            markerfacecolor="black", markeredgecolor="white",
            markeredgewidth=1.0, zorder=6)
    ax.annotate(
        site_key, xy=(site["lon"], site["lat"]), xytext=(5, 5),
        textcoords="offset points", fontsize=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black",
                  alpha=0.85, lw=0.5),
        zorder=7,
    )

    ax.set_xlabel("Longitude (°)", fontsize=TICK_FS)
    ax.set_ylabel("Latitude (°)", fontsize=TICK_FS)
    ax.tick_params(labelsize=TICK_FS - 1)
    ax.set_title(f"{site['label']}", fontsize=LABEL_FS, color=color, pad=4)
    ax.set_aspect("equal")


def draw_zoom_box(
    ax: "matplotlib.axes.Axes",
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    color: str, label: str,
) -> None:
    """Draw a labelled rectangle on a cartopy axes marking a zoom extent."""
    rect = mpatches.Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=2.0, edgecolor=color, facecolor="none",
        transform=ccrs.PlateCarree(), zorder=7,
    )
    ax.add_patch(rect)
    ax.text(
        lon_min, lat_max, label,
        transform=ccrs.PlateCarree(), fontsize=8, fontweight="bold",
        color=color, va="bottom", ha="left", zorder=8,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color,
                  alpha=0.85, lw=0.5),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sim", default="sim_2021", choices=["sim_2021", "sim_2022"],
                    help="simulation to source initial SST from (default: sim_2021)")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--margin-km", type=float, default=15.0,
                    help="zoom half-width around each site, km (default: 15)")
    ap.add_argument("--sst-min-valid", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output", type=Path)
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

    # --- Load mesh once ---
    print(f"reading mesh: {static_path}")
    static = xr.open_dataset(static_path)
    landmask = static["landmask"].values
    ter = static["ter"].values
    lat_cell = np.rad2deg(static["latCell"].values)
    lon_cell = np.rad2deg(static["lonCell"].values)
    lon_cell = np.where(lon_cell > 180.0, lon_cell - 360.0, lon_cell)
    vert_lat = np.rad2deg(static["latVertex"].values)
    vert_lon = np.rad2deg(static["lonVertex"].values)
    vert_lon = np.where(vert_lon > 180.0, vert_lon - 360.0, vert_lon)
    vertices_on_cell = static["verticesOnCell"].values
    n_edges_on_cell = static["nEdgesOnCell"].values
    static.close()

    print(f"reading SST: {sfc_path}")
    with xr.open_dataset(sfc_path) as sfc:
        sst_c = sfc["sst"].isel(Time=0).values - 273.15

    print("building cell polygons...")
    t0 = time.time()
    all_polygons, valid_idx = build_all_polygons(
        vertices_on_cell, n_edges_on_cell, vert_lat, vert_lon
    )
    print(f"  {len(all_polygons)} polygons in {time.time() - t0:.1f}s")

    # --- Figure layout: main map left, two zoom panels stacked on right ---
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3.2, 1],
        height_ratios=[1, 1],
        hspace=0.08,
        wspace=0.04,
        left=0.04, right=0.97,
        top=0.93, bottom=0.06,
    )
    ax_main = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    ax_p0 = fig.add_subplot(gs[0, 1])
    ax_lpi = fig.add_subplot(gs[1, 1])
    zoom_axes = {"P0": ax_p0, "LPI": ax_lpi}

    # --- Main domain ---
    domain = config.mesh["domain"]
    margin = 0.15
    extent = [
        domain["lon_min"] - margin, domain["lon_max"] + margin,
        domain["lat_min"] - margin, domain["lat_max"] + margin,
    ]
    ax_main.set_extent(extent, crs=ccrs.PlateCarree())

    print("rendering domain...")
    land_coll, ocean_coll = draw_domain(
        ax_main, all_polygons, valid_idx, landmask, ter, sst_c, args.sst_min_valid
    )

    ax_main.coastlines(resolution="10m", linewidth=0.4, color="black", zorder=3)
    gl = ax_main.gridlines(draw_labels=True, alpha=0.35, linestyle="--",
                            linewidth=0.35)
    gl.top_labels = False
    gl.right_labels = False

    # Site markers on main map
    for site_key, obs in config.observations.items():
        color = SITE_COLORS[site_key]
        ax_main.plot(
            obs["lon"], obs["lat"], marker="*", markersize=16,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.0,
            transform=ccrs.PlateCarree(), zorder=8,
        )
        ax_main.annotate(
            site_key,
            xy=(obs["lon"], obs["lat"]),
            xytext=(7, 5), textcoords="offset points",
            fontsize=9, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.9, lw=0.6),
            zorder=9,
        )

    # Colorbars for main map
    cbar_land = fig.colorbar(land_coll, ax=ax_main, shrink=0.48, pad=0.01,
                             aspect=28, extend="max",
                             location="left")
    cbar_land.set_label("Terrain height (m)", fontsize=LABEL_FS)
    cbar_land.ax.tick_params(labelsize=TICK_FS)
    cbar_ocean = fig.colorbar(ocean_coll, ax=ax_main, shrink=0.48, pad=0.05,
                              aspect=28, extend="min",
                              location="left")
    cbar_ocean.set_label(f"Initial SST (°C) — {sim.label}", fontsize=LABEL_FS)
    cbar_ocean.ax.tick_params(labelsize=TICK_FS)

    ax_main.set_title(
        f"MPAS {mesh_name} (~{config.mesh['mean_cell_spacing_km']:.1f} km) "
        "— terrain & initial SST",
        fontsize=TITLE_FS,
    )

    # --- Zoom panels ---
    for site_key, obs in config.observations.items():
        color = SITE_COLORS[site_key]
        dlat = args.margin_km / KM_PER_DEG_LAT
        dlon = args.margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(obs["lat"])))

        lon_min = obs["lon"] - dlon
        lon_max = obs["lon"] + dlon
        lat_min = obs["lat"] - dlat
        lat_max = obs["lat"] + dlat

        # Nearest cell for this site
        dist2 = (lat_cell - obs["lat"]) ** 2 + (lon_cell - obs["lon"]) ** 2
        icell = int(np.argmin(dist2))

        # Cells within zoom box
        in_box = (
            (lat_cell >= lat_min) & (lat_cell <= lat_max)
            & (lon_cell >= lon_min) & (lon_cell <= lon_max)
        )
        box_global_idx = np.where(in_box)[0]

        box_polys, local_mask = build_box_polygons(box_global_idx, all_polygons, valid_idx)
        box_valid_global = box_global_idx[local_mask]

        ax_zoom = zoom_axes[site_key]
        ax_zoom.set_xlim(lon_min, lon_max)
        ax_zoom.set_ylim(lat_min, lat_max)

        draw_zoom_panel(
            ax_zoom, box_polys, box_valid_global,
            landmask, ter, obs, site_key,
            icell=icell,
            cell_lat=float(lat_cell[icell]),
            cell_lon=float(lon_cell[icell]),
            color=color,
        )

        # Mark zoom extent on the main map
        draw_zoom_box(ax_main, lon_min, lon_max, lat_min, lat_max,
                      color=color, label=site_key)

        print(f"  {site_key}: nearest cell {icell} "
              f"({lat_cell[icell]:.4f}, {lon_cell[icell]:.4f}), "
              f"{len(box_polys)} cells in zoom")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / f"fig07_domain_with_zoom_insets_{args.sim}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
