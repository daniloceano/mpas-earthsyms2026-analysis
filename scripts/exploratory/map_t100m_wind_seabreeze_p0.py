#!/usr/bin/env python
"""Exploratory map: 100 m temperature + wind streamlines at the two extreme
moments of the windiest day near an observation site, to visualize the
sea/land-breeze cycle.

Picks the day with the strongest 100 m wind at the MPAS cell nearest the site
(``mpas_analysis.seabreeze.find_moments``), then within that day the hour of
maximum and the hour of minimum 100 m temperature at the same cell — the
diurnal extremes expected to bracket the onshore (warm/afternoon) and
offshore (cool/pre-dawn) circulation. Draws both moments as a 2-panel
native-mesh map (temperature fill + wind streamlines, both at 100 m), zoomed
to a ~100 km box around the site, on a shared color scale so the panels are
directly comparable. The left panel also outlines (thin red dashed line) the
meridional transect used by ``xsection_temp_seabreeze_p0.py``, so the two
figures can be read side by side.

    python scripts/exploratory/map_t100m_wind_seabreeze_p0.py
    python scripts/exploratory/map_t100m_wind_seabreeze_p0.py --site LPI --margin-km 150
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
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis.seabreeze import find_moments  # noqa: E402
from mpas_analysis.thermo import temperature_from_theta_pressure  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"
KM_PER_DEG_LAT = 111.0


def build_cell_polygons(
    vertices_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    vert_lat: np.ndarray,
    vert_lon: np.ndarray,
) -> list[np.ndarray | None]:
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


def interp_wind_to_grid(
    cell_lon: np.ndarray, cell_lat: np.ndarray, u: np.ndarray, v: np.ndarray,
    extent: list[float], grid_n: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scattered cell-center (u, v) -> a regular lat/lon grid for streamplot
    (which, unlike quiver, needs a structured grid). Linear interpolation
    with a nearest-neighbour fill for edge points outside the convex hull of
    cell centers.
    """
    grid_lon = np.linspace(extent[0], extent[1], grid_n)
    grid_lat = np.linspace(extent[2], extent[3], grid_n)
    glon, glat = np.meshgrid(grid_lon, grid_lat)
    points = np.column_stack([cell_lon, cell_lat])
    target = (glon, glat)
    u_grid = griddata(points, u, target, method="linear")
    v_grid = griddata(points, v, target, method="linear")
    u_near = griddata(points, u, target, method="nearest")
    v_near = griddata(points, v, target, method="nearest")
    u_grid = np.where(np.isnan(u_grid), u_near, u_grid)
    v_grid = np.where(np.isnan(v_grid), v_near, v_grid)
    return grid_lon, grid_lat, u_grid, v_grid


def load_panel_fields(history_dir: Path, mesh_name: str, ts: pd.Timestamp,
                      level_index: int, box_idx: np.ndarray) -> dict:
    """Temperature (degC) and wind components (m/s) at one level/timestamp,
    subset to box_idx cells, from the single matching history file."""
    files = io.find_history_files(history_dir)
    times = pd.DatetimeIndex(io.read_timestamps(files))
    loc = times.get_loc(ts)
    file = files[loc]
    with io.open_history(
        file, variables=["theta", "pressure", "uReconstructZonal", "uReconstructMeridional"]
    ) as ds:
        ds = ds.isel(Time=0, nVertLevels=level_index).isel(nCells=box_idx)
        temp_c = temperature_from_theta_pressure(
            ds["theta"].values, ds["pressure"].values) - 273.15
        u = ds["uReconstructZonal"].values
        v = ds["uReconstructMeridional"].values
    return {"file": file, "temp_c": temp_c, "u": u, "v": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="P0", help="observation site key in "
                     "config/simulations.yaml (default: P0)")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--margin-km", type=float, default=100.0,
                     help="half-width of the zoom box around the site, in km (default: 100)")
    ap.add_argument("--stream-density", type=float, default=0.8,
                     help="streamplot line density (default: 0.8)")
    ap.add_argument("--transect-half-width-deg", type=float, default=2.0,
                     help="half-width (deg latitude) of the transect line drawn on "
                     "the left panel, matching xsection_temp_seabreeze_p0.py's "
                     "--half-width-deg (default: 2.0)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation(args.site)
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    print(f"[{sim.key}] scanning for the windiest day near {args.site} (this reads "
          "the whole run once — ~45s)...")
    moments = find_moments(sim.history_dir, mesh_name, site["lat"], site["lon"],
                           target_height_m=100.0)
    print(f"  cell {moments.icell} (~{moments.dist_km:.2f} km from {args.site}), "
          f"level {moments.level_index} ({moments.level_height_m:.1f} m AGL)")
    print(f"  windiest hour: {moments.windiest_time} "
          f"({moments.windiest_speed:.2f} m/s) -> day {moments.day.date()}")
    print(f"  Tmax: {moments.tmax_time} ({moments.tmax_value - 273.15:.2f} degC)")
    print(f"  Tmin: {moments.tmin_time} ({moments.tmin_value - 273.15:.2f} degC)")

    static_path = config.data_root / f"{mesh_name}.static.nc"
    static = xr.open_dataset(static_path)
    lat = np.rad2deg(static["latCell"].values)
    lon = np.rad2deg(static["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    vert_lat = np.rad2deg(static["latVertex"].values)
    vert_lon = np.rad2deg(static["lonVertex"].values)
    vert_lon = np.where(vert_lon > 180.0, vert_lon - 360.0, vert_lon)
    vertices_on_cell = static["verticesOnCell"].values
    n_edges_on_cell = static["nEdgesOnCell"].values
    static.close()

    dlat = args.margin_km / KM_PER_DEG_LAT
    dlon = args.margin_km / (KM_PER_DEG_LAT * np.cos(np.radians(site["lat"])))
    in_box = (
        (lat >= site["lat"] - dlat) & (lat <= site["lat"] + dlat)
        & (lon >= site["lon"] - dlon) & (lon <= site["lon"] + dlon)
    )
    box_idx = np.where(in_box)[0]
    print(f"{len(box_idx)} cells within {args.margin_km:.0f} km of {args.site}")

    polygons = build_cell_polygons(
        vertices_on_cell[box_idx], n_edges_on_cell[box_idx], vert_lat, vert_lon
    )
    keep = [i for i, p in enumerate(polygons) if p is not None]
    box_idx = box_idx[keep]
    polygons = [polygons[i] for i in keep]
    cell_lat, cell_lon = lat[box_idx], lon[box_idx]

    panels = [
        ("Tmin", moments.tmin_time),
        ("Tmax", moments.tmax_time),
    ]
    print("reading full-domain fields for both moments...")
    data = [(label, ts, load_panel_fields(sim.history_dir, mesh_name, ts,
                                          moments.level_index, box_idx))
            for label, ts in panels]

    vmin = min(float(np.min(d["temp_c"])) for _, _, d in data)
    vmax = max(float(np.max(d["temp_c"])) for _, _, d in data)

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    extent = [site["lon"] - dlon, site["lon"] + dlon,
             site["lat"] - dlat, site["lat"] + dlat]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = None
    for i, (ax, (label, ts, d)) in enumerate(zip(axes, data)):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        coll = PolyCollection(
            polygons, array=d["temp_c"], cmap="RdYlBu_r", norm=norm,
            edgecolors="none", antialiaseds=False,
            transform=ccrs.PlateCarree(), zorder=1,
        )
        ax.add_collection(coll)
        mappable = coll

        grid_lon, grid_lat, u_grid, v_grid = interp_wind_to_grid(
            cell_lon, cell_lat, d["u"], d["v"], extent)
        ax.streamplot(
            grid_lon, grid_lat, u_grid, v_grid,
            transform=ccrs.PlateCarree(), density=args.stream_density,
            color="black", linewidth=0.9, arrowsize=0.9, zorder=3,
        )

        if i == 0:
            ax.plot(
                [site["lon"], site["lon"]],
                [site["lat"] - args.transect_half_width_deg,
                 site["lat"] + args.transect_half_width_deg],
                transform=ccrs.PlateCarree(), color="red", linestyle="--",
                linewidth=1.0, zorder=5,
            )
            # Anchor the label inside the visible map box, not at the
            # transect's true endpoint (which typically lies outside this
            # ~100 km zoom, since the transect spans +/-2 deg latitude).
            label_lat = min(site["lat"] + args.transect_half_width_deg,
                            site["lat"] + dlat * 0.92)
            ax.annotate("cross-section transect", xy=(site["lon"], label_lat),
                       xytext=(6, -2), textcoords="offset points", fontsize=8,
                       fontweight="bold", color="red",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red",
                                 alpha=0.85, lw=0.5), zorder=6)

        ax.plot(site["lon"], site["lat"], marker="*", markersize=16,
               markerfacecolor="black", markeredgecolor="white",
               markeredgewidth=1.0, transform=ccrs.PlateCarree(), zorder=5)
        ax.annotate(args.site, xy=(site["lon"], site["lat"]), xytext=(6, 6),
                   textcoords="offset points", fontsize=9, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black",
                             alpha=0.85, lw=0.5), zorder=6)

        ax.coastlines(resolution="10m", linewidth=0.6, color="black", zorder=4)
        gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle="--", linewidth=0.4)
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f"{label}\n{ts} UTC", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.02, aspect=30,
                label=f"Temperature @ {moments.level_height_m:.0f} m (°C)")

    fig.suptitle(
        f"MPAS {sim.label} — windiest day near {args.site} ({moments.day.date()}, "
        f"peak {moments.windiest_speed:.1f} m/s @ {moments.windiest_time:%H:%M} UTC)\n"
        f"temperature & wind @ {moments.level_height_m:.0f} m — cell {moments.icell}, "
        f"~{moments.dist_km:.1f} km from {args.site}",
        fontsize=12, y=0.985,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    default_name = f"map_t100m_seabreeze_{args.site.lower()}.png"
    out = args.output or FIG_DIR / default_name
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
