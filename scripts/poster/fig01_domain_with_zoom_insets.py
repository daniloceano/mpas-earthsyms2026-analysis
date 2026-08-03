#!/usr/bin/env python
"""Poster Figure 1: native-mesh domain (terrain + SST) with P0 and LPI zoom
panels showing the individual Voronoi cells at each observation site.

Single poster panel combining the former domain overview and both site zooms:
the domain map fills the whole figure, and the two site zooms are drawn as
true insets *inside* the map (bottom-left = P0, top-right = LPI), each linked
to its zoom extent on the main map by a black box and a dashed connector
line (black throughout — the site colours used in earlier drafts didn't
contrast against the red/green map fill).

Because the two simulations share the same mesh, terrain geometry is identical
across both sims; SST is taken from the sim specified by ``--sim`` (default:
sim_2021, which covers P0). The LPI zoom panel also uses static terrain only.

    python scripts/poster/fig01_domain_with_zoom_insets.py
    python scripts/poster/fig01_domain_with_zoom_insets.py --sim sim_2022
    python scripts/poster/fig01_domain_with_zoom_insets.py --margin-km 20
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
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "poster"
RAD2DEG = 180.0 / np.pi
KM_PER_DEG_LAT = 111.0

# Poster font sizes (bumped for legibility at poster viewing distance)
TITLE_FS = 20
LABEL_FS = 16
TICK_FS = 14
ANNOT_FS = 11

# Site marker/box/connector colour. Both sites used to have distinct colours
# (red/green) but those washed out against the SST/terrain fill, so both use
# plain black now for contrast; text labels still disambiguate the sites.
SITE_COLORS = {"P0": "black", "LPI": "black"}

# Inset placement, in ax_main axes-fraction coordinates: (x0, y0, width, height).
# P0's y0 is nudged up from 0.03 to clear the main map's bottom border (was
# overlapping it; ~0.03 axes-fraction ~= half a degree of latitude here).
INSET_BOUNDS = {
    "P0": (0.015, 0.055, 0.33, 0.40),
    "LPI": (0.66, 0.57, 0.33, 0.40),
}


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
            linewidths=0.6, zorder=1,
        ))
    if land_polys:
        land_coll = PolyCollection(
            land_polys, array=land_values, cmap=cmap_land, norm=norm_land,
            edgecolors="grey", linewidths=0.6, zorder=1,
        )
        ax.add_collection(land_coll)

    # Highlight the model comparison cell (outline only, no fill)
    # Find polygon for icell in the valid_idx of this zoom subset
    local_positions = np.where(valid_idx == icell)[0]
    if len(local_positions) > 0:
        highlight_poly = polygons[local_positions[0]]
        ax.add_collection(PolyCollection(
            [highlight_poly], facecolor="none", edgecolor=color,
            linewidth=2.6, zorder=3,
        ))
        ax.annotate(
            f"MPAS cell\n(~{np.sqrt((cell_lat - site['lat'])**2 + (cell_lon - site['lon'])**2) * KM_PER_DEG_LAT:.1f} km)",
            xy=(cell_lon, cell_lat), xytext=(14, -25),
            textcoords="offset points", fontsize=ANNOT_FS, color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                      alpha=0.9, lw=0.9),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.1),
            zorder=5,
        )

    ax.plot(site["lon"], site["lat"], marker="*", markersize=17,
            markerfacecolor="black", markeredgecolor="white",
            markeredgewidth=1.3, zorder=6)
    ax.annotate(
        site_key, xy=(site["lon"], site["lat"]), xytext=(9, 9),
        textcoords="offset points", fontsize=TICK_FS - 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black",
                  alpha=0.85, lw=0.9),
        zorder=7,
    )

    # Compact inset: no title, no axis labels, few ticks — the box+connector
    # on the main map plus the site's own star+label already convey what this
    # is, so the inset stays map-only.
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.tick_params(labelsize=TICK_FS - 3, width=1.2)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def draw_zoom_box(
    ax_main: "matplotlib.axes.Axes",
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    color: str,
    inset_bounds: tuple[float, float, float, float],
) -> None:
    """Draw a rectangle marking a zoom extent on the main map, plus a dashed
    connector line to the matching inset (anchored in ax_main's axes-fraction
    coordinates, and may sit above or below the box).

    No separate text label here — the site marker's own annotation on the
    main map already names the site; a third label at the (tiny) box would
    just overlap it.
    """
    plate = ccrs.PlateCarree()
    rect = mpatches.Rectangle(
        (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
        linewidth=2.6, edgecolor=color, facecolor="none",
        transform=plate, zorder=7,
    )
    ax_main.add_patch(rect)

    x0, y0, w, h = inset_bounds
    inset_cx = x0 + w / 2
    if y0 + h / 2 > 0.5:
        # Inset sits in the top half of the map: connect its bottom edge to
        # the top of the box.
        inset_anchor = (inset_cx, y0)
        box_anchor = ((lon_min + lon_max) / 2, lat_max)
    else:
        # Inset sits in the bottom half: connect its top edge to the bottom
        # of the box.
        inset_anchor = (inset_cx, y0 + h)
        box_anchor = ((lon_min + lon_max) / 2, lat_min)

    plate_transform = plate._as_mpl_transform(ax_main)
    con = ConnectionPatch(
        xyA=box_anchor, coordsA=plate_transform, axesA=ax_main,
        xyB=inset_anchor, coordsB=ax_main.transAxes, axesB=ax_main,
        color=color, linewidth=1.6, linestyle="--", alpha=0.85, zorder=6,
    )
    ax_main.add_artist(con)


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

    # --- Figure layout: precompute ax_main's box to exactly match the domain's
    # data aspect ratio, then hand-place the two colorbars just to its left.
    # (Letting cartopy's aspect-lock resize ax_main *after* fig.colorbar()
    # calls — the previous approach — produced large, unpredictable gaps
    # because each colorbar call and the aspect correction each resize the
    # axes independently; precomputing removes that interaction entirely.)
    # FIG_W_BASE is the width the map/colorbar fractions below were tuned
    # for; RIGHT_MARGIN_IN adds pure canvas width reserved for the east-side
    # gridline labels (moved there from the west so they'd clear the
    # colorbars) without changing the map's or colorbars' absolute size or
    # position — see CBAR_*_X below, which rescale by FIG_W_BASE/FIG_W to
    # keep those axes at the same physical (inch) spot on the wider canvas.
    FIG_W_BASE = 13.0
    RIGHT_MARGIN_IN = 0.9
    FIG_W, FIG_H = FIG_W_BASE + RIGHT_MARGIN_IN, 7.5
    MAP_TOP, MAP_BOTTOM = 0.95, 0.05
    MAP_RIGHT = 1.0 - RIGHT_MARGIN_IN / FIG_W
    domain = config.mesh["domain"]
    margin = 0.15
    lon_span = (domain["lon_max"] + margin) - (domain["lon_min"] - margin)
    lat_span = (domain["lat_max"] + margin) - (domain["lat_min"] - margin)
    data_aspect = lon_span / lat_span
    map_height_frac = MAP_TOP - MAP_BOTTOM
    map_width_frac = map_height_frac * (FIG_H / FIG_W) * data_aspect
    map_left = MAP_RIGHT - map_width_frac

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = fig.add_axes(
        [map_left, MAP_BOTTOM, map_width_frac, map_height_frac],
        projection=ccrs.PlateCarree(),
    )

    # --- Main domain ---
    extent = [
        domain["lon_min"] - margin, domain["lon_max"] + margin,
        domain["lat_min"] - margin, domain["lat_max"] + margin,
    ]
    ax_main.set_extent(extent, crs=ccrs.PlateCarree())

    print("rendering domain...")
    land_coll, ocean_coll = draw_domain(
        ax_main, all_polygons, valid_idx, landmask, ter, sst_c, args.sst_min_valid
    )

    ax_main.coastlines(resolution="10m", linewidth=0.9, color="black", zorder=3)
    gl = ax_main.gridlines(draw_labels=True, alpha=0.35, linestyle="--",
                            linewidth=0.6)
    # Latitude labels on the east (right) side, not west: the colorbars sit
    # just left of the map, and west-side labels crowded/overlapped them.
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = True
    gl.xlabel_style = {"size": TICK_FS}
    gl.ylabel_style = {"size": TICK_FS}
    for spine in ax_main.spines.values():
        spine.set_linewidth(1.4)

    # Site markers on main map
    for site_key, obs in config.observations.items():
        color = SITE_COLORS[site_key]
        ax_main.plot(
            obs["lon"], obs["lat"], marker="*", markersize=19,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.3,
            transform=ccrs.PlateCarree(), zorder=8,
        )
        ax_main.annotate(
            site_key,
            xy=(obs["lon"], obs["lat"]),
            xytext=(7, 5), textcoords="offset points",
            fontsize=TICK_FS, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                      alpha=0.9, lw=1.0),
            zorder=9,
        )

    # Colorbars: hand-placed narrow axes just left of the map, ticks/labels
    # facing further left (away from the map) so the two bars sit close
    # together with no dead space between them. X positions/width are
    # given as fractions of FIG_W_BASE, then rescaled to the actual (wider)
    # FIG_W so they land at the same physical spot regardless of how much
    # right-margin was reserved for the gridline labels.
    _resc = FIG_W_BASE / FIG_W
    cbar_land_ax = fig.add_axes([0.08 * _resc, 0.20, 0.014 * _resc, 0.55])
    cbar_land = fig.colorbar(land_coll, cax=cbar_land_ax, extend="max")
    cbar_land_ax.yaxis.set_ticks_position("left")
    cbar_land_ax.yaxis.set_label_position("left")
    cbar_land.set_label("Terrain height (m)", fontsize=LABEL_FS)
    cbar_land_ax.tick_params(labelsize=TICK_FS)

    cbar_ocean_ax = fig.add_axes([0.15 * _resc, 0.20, 0.014 * _resc, 0.55])
    cbar_ocean = fig.colorbar(ocean_coll, cax=cbar_ocean_ax, extend="min")
    cbar_ocean_ax.yaxis.set_ticks_position("left")
    cbar_ocean_ax.yaxis.set_label_position("left")
    cbar_ocean.set_label(f"Initial SST (°C) — {sim.label}", fontsize=LABEL_FS)
    cbar_ocean_ax.tick_params(labelsize=TICK_FS)

    # y=1.0 is required here: with a poster-size TITLE_FS, matplotlib's
    # automatic title-offset (anti-overlap-with-tick-labels) logic collides
    # with cartopy's Gridliner and resolves to an infinite y-position,
    # silently dropping the title entirely. Passing y explicitly bypasses
    # that automatic offset computation.
    ax_main.set_title(
        f"MPAS domain (~{config.mesh['mean_cell_spacing_km']:.1f} km) "
        "— terrain & initial SST",
        fontsize=TITLE_FS, y=1.0,
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

        bounds = INSET_BOUNDS[site_key]
        ax_zoom = ax_main.inset_axes(bounds, zorder=10)
        ax_zoom.set_facecolor("white")
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
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

        # Mark zoom extent on the main map and connect it to the inset
        draw_zoom_box(ax_main, lon_min, lon_max, lat_min, lat_max,
                      color=color, inset_bounds=bounds)

        print(f"  {site_key}: nearest cell {icell} "
              f"({lat_cell[icell]:.4f}, {lon_cell[icell]:.4f}), "
              f"{len(box_polys)} cells in zoom")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / f"fig01_domain_with_zoom_insets_{args.sim}.png"
    # NOTE: bbox_inches="tight" is deliberately NOT used here — cartopy's
    # GeoAxes reports an incorrect tight bbox for hand-added PolyCollections
    # (it crops to just the colorbars, losing the whole map). Margins are
    # already hand-tuned via MAP_TOP/MAP_BOTTOM/MAP_RIGHT above instead.
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
