#!/usr/bin/env python
"""Poster Figure 5: native-mesh terrain zoomed on LPI, highlighting the MPAS
cell used for the model-vs-observation comparison.

Same terrain source as fig01 (``static.nc``: ``ter``/``landmask``) but zoomed
to a small box around the LPI (Porto-Ilha) LiDAR site, with individual
Voronoi cells outlined (few enough at this zoom to be legible) so the
specific cell picked by ``scatter_wspeed_model_vs_lpi.py``/
``list_model_levels_at_p0.py --site LPI`` (nearest-neighbour in lat/lon) is
visibly justified, not just a reported index. The highlighted cell is
outlined only (no fill), so its land/ocean color stays visible underneath —
confirming at a glance that the comparison cell is ocean, not land.

    python scripts/poster/fig05_map_lpi_nearest_cell.py
    python scripts/poster/fig05_map_lpi_nearest_cell.py --margin-km 25
"""

from __future__ import annotations

import argparse
import sys
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

KM_PER_DEG_LAT = 111.0


def build_cell_polygons(
    vertices_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    vert_lat: np.ndarray,
    vert_lon: np.ndarray,
) -> list[np.ndarray | None]:
    """One polygon (or None for a border cell) per row, same order as input."""
    polygons: list[np.ndarray | None] = []
    for i in range(vertices_on_cell.shape[0]):
        n = int(n_edges_on_cell[i])
        vidx = vertices_on_cell[i, :n] - 1
        if (vidx < 0).any():
            polygons.append(None)
            continue
        lons = vert_lon[vidx]
        lats = vert_lat[vidx]
        polygons.append(np.column_stack([lons, lats]))
    return polygons


def land_colormap() -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "land_terrain", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--margin-km", type=float, default=15.0,
                     help="half-width of the zoom box around LPI, in km (default: 15)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/poster/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("LPI")
    mesh_name = config.mesh["name"]

    static_path = config.data_root / f"{mesh_name}.static.nc"
    if not static_path.exists():
        print(f"missing static file: {static_path}", file=sys.stderr)
        return 1

    print(f"reading mesh + terrain: {static_path}")
    static = xr.open_dataset(static_path)
    landmask = static["landmask"].values
    ter = static["ter"].values
    lat = np.rad2deg(static["latCell"].values)
    lon = np.rad2deg(static["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    vert_lat = np.rad2deg(static["latVertex"].values)
    vert_lon = np.rad2deg(static["lonVertex"].values)
    vert_lon = np.where(vert_lon > 180.0, vert_lon - 360.0, vert_lon)
    vertices_on_cell = static["verticesOnCell"].values
    n_edges_on_cell = static["nEdgesOnCell"].values
    static.close()

    dist2 = (lat - site["lat"]) ** 2 + (lon - site["lon"]) ** 2
    icell = int(np.argmin(dist2))
    dist_km = float(np.sqrt(dist2[icell])) * KM_PER_DEG_LAT
    print(f"nearest cell to {site['label']}: idx={icell}, "
          f"({lat[icell]:.4f}, {lon[icell]:.4f}), ~{dist_km:.2f} km away, "
          f"{'land' if landmask[icell] else 'ocean'}")

    dlat = args.margin_km / KM_PER_DEG_LAT
    dlon = args.margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(site["lat"])))
    in_box = (
        (lat >= site["lat"] - dlat) & (lat <= site["lat"] + dlat)
        & (lon >= site["lon"] - dlon) & (lon <= site["lon"] + dlon)
    )
    box_idx = np.where(in_box)[0]
    print(f"{len(box_idx)} cells within {args.margin_km:.0f} km of LPI")

    polygons = build_cell_polygons(
        vertices_on_cell[box_idx], n_edges_on_cell[box_idx], vert_lat, vert_lon
    )
    keep = [i for i, p in enumerate(polygons) if p is not None]
    box_idx = box_idx[keep]
    polygons = [polygons[i] for i in keep]

    is_land = landmask[box_idx] == 1
    land_polys = [p for p, m in zip(polygons, is_land) if m]
    land_values = ter[box_idx][is_land]
    ocean_polys = [p for p, m in zip(polygons, is_land) if not m]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [site["lon"] - dlon, site["lon"] + dlon,
         site["lat"] - dlat, site["lat"] + dlat],
        crs=ccrs.PlateCarree(),
    )

    if ocean_polys:
        ax.add_collection(PolyCollection(
            ocean_polys, facecolors="#cfe8f3", edgecolors="grey", linewidths=0.4,
            transform=ccrs.PlateCarree(), zorder=1,
        ))
    if land_polys:
        norm_land = mcolors.Normalize(vmin=0.0, vmax=max(float(land_values.max()), 1.0))
        land_coll = PolyCollection(
            land_polys, array=land_values, cmap=land_colormap(), norm=norm_land,
            edgecolors="grey", linewidths=0.4, transform=ccrs.PlateCarree(), zorder=1,
        )
        ax.add_collection(land_coll)
        fig.colorbar(land_coll, ax=ax, shrink=0.6, pad=0.03, aspect=30,
                     label="Terrain height (m)")

    # Highlight the cell used for the model comparison. Outline only (no fill)
    # so the land/ocean color underneath stays visible through the highlight.
    cell_poly = polygons[int(np.where(box_idx == icell)[0][0])]
    ax.add_collection(PolyCollection(
        [cell_poly], facecolor="none", edgecolor="red", linewidth=2.5,
        transform=ccrs.PlateCarree(), zorder=3,
    ))
    cell_lon, cell_lat = float(lon[icell]), float(lat[icell])
    ax.annotate(
        f"MPAS cell {icell}\n(~{dist_km:.2f} km from LPI)",
        xy=(cell_lon, cell_lat), xytext=(12, -22), textcoords="offset points",
        fontsize=9, fontweight="bold", color="darkred",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkred", alpha=0.9, lw=0.6),
        arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8),
        zorder=5,
    )

    ax.plot(
        site["lon"], site["lat"], marker="*", markersize=20,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=1.2,
        transform=ccrs.PlateCarree(), zorder=6,
    )
    ax.annotate(
        "LPI", xy=(site["lon"], site["lat"]), xytext=(8, 8),
        textcoords="offset points", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85, lw=0.5),
        zorder=7,
    )

    ax.coastlines(resolution="10m", linewidth=0.8, color="black", zorder=4)
    gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle="--", linewidth=0.4)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(
        f"{mesh_name} mesh near {site['label']} — cell used for model comparison\n"
        f"zoom: ±{args.margin_km:.0f} km"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / "fig05_map_lpi_nearest_cell.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
