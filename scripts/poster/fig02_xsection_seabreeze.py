#!/usr/bin/env python
"""Poster Figure 2: meridional cross-section of temperature + wind at the
sea-breeze moment (Tmax on the windiest day) through a LiDAR site.

Promoted from ``scripts/exploratory/xsection_temp_seabreeze_p0.py``;
single-panel poster version showing the sea-breeze (Tmax) moment only. A
small coastline inset in the top-right corner shows where the transect sits
within the domain.

    python scripts/poster/fig02_xsection_seabreeze.py
    python scripts/poster/fig02_xsection_seabreeze.py --site LPI
    python scripts/poster/fig02_xsection_seabreeze.py --half-width-deg 3 --zmax 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis.seabreeze import find_moments  # noqa: E402
from mpas_analysis.thermo import temperature_from_theta_pressure  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "poster"
KM_PER_DEG_LAT = 111.0

# Poster-level base font sizes
TITLE_FS = 14
LABEL_FS = 13
TICK_FS = 11
ANNOT_FS = 10


def sample_meridional_transect(
    lat_center: float, lon_fixed: float, half_width_deg: float,
    lat_cell: np.ndarray, lon_cell: np.ndarray, npoints: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cell sampling along a fixed-longitude line, duplicates collapsed."""
    plat = np.linspace(lat_center - half_width_deg, lat_center + half_width_deg, npoints)
    plon = np.full_like(plat, lon_fixed)
    tree = cKDTree(np.column_stack([lat_cell, lon_cell]))
    _, idx = tree.query(np.column_stack([plat, plon]))
    cells = [int(idx[0])]
    for c in idx[1:]:
        if int(c) != cells[-1]:
            cells.append(int(c))
    return np.asarray(cells, dtype=int), lat_cell[np.asarray(cells, dtype=int)]


def x_edges(xc: np.ndarray) -> np.ndarray:
    """Midpoint edges for column centers (length N → N+1)."""
    if len(xc) == 1:
        return np.array([xc[0] - 0.5, xc[0] + 0.5])
    mids = 0.5 * (xc[:-1] + xc[1:])
    first = xc[0] - 0.5 * (xc[1] - xc[0])
    last = xc[-1] + 0.5 * (xc[-1] - xc[-2])
    return np.concatenate([[first], mids, [last]])


def interface_corners(z_iface: np.ndarray) -> np.ndarray:
    """(nInterfaces, ncols) → (nInterfaces, ncols+1) averaged onto x-edges."""
    return np.concatenate(
        [z_iface[:, :1],
         0.5 * (z_iface[:, :-1] + z_iface[:, 1:]),
         z_iface[:, -1:]], axis=1
    )


def load_column_fields(
    history_dir: Path, ts: pd.Timestamp, cells: np.ndarray, n_levels: int
) -> dict:
    files = io.find_history_files(history_dir)
    times = pd.DatetimeIndex(io.read_timestamps(files))
    file = files[times.get_loc(ts)]
    with io.open_history(
        file,
        variables=["theta", "pressure", "uReconstructZonal",
                   "uReconstructMeridional", "w"],
    ) as ds:
        ds = ds.isel(Time=0, nVertLevels=slice(0, n_levels)).isel(nCells=cells)
        theta = ds["theta"].values.T
        pressure = ds["pressure"].values.T
        u = ds["uReconstructZonal"].values.T
        v = ds["uReconstructMeridional"].values.T
        w_iface = ds["w"].isel(nVertLevelsP1=slice(0, n_levels + 1)).values.T
        w = 0.5 * (w_iface[:-1] + w_iface[1:])
    return {
        "file": file,
        "temp_c": temperature_from_theta_pressure(theta, pressure) - 273.15,
        "u": u, "v": v, "w": w,
    }


def draw_location_inset(
    ax: "matplotlib.axes.Axes",
    domain: dict,
    site: dict,
    half_width_deg: float,
) -> None:
    """Small coastline inset (top-right) showing where the meridional
    transect (a north-south line through the site) sits within the domain.
    Kept map-only: no ticks, labels, or title — just land/ocean fill, the
    transect line, and the site star, framed in a box.
    """
    ax_loc = ax.inset_axes([0.70, 0.66, 0.28, 0.32], projection=ccrs.PlateCarree(),
                           zorder=20)
    margin = 0.15
    ax_loc.set_extent([
        domain["lon_min"] - margin, domain["lon_max"] + margin,
        domain["lat_min"] - margin, domain["lat_max"] + margin,
    ], crs=ccrs.PlateCarree())
    # Gray continent / blue sea — same ocean tint used for fig01's zoom insets.
    ax_loc.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "10m"),
                       facecolor="#cfe8f3", edgecolor="none", zorder=0)
    ax_loc.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"),
                       facecolor="0.75", edgecolor="none", zorder=1)
    ax_loc.coastlines(resolution="10m", linewidth=0.5, color="black", zorder=2)
    ax_loc.plot(
        [site["lon"], site["lon"]],
        [site["lat"] - half_width_deg, site["lat"] + half_width_deg],
        color="black", linewidth=1.6, zorder=4,
        transform=ccrs.PlateCarree(),
    )
    ax_loc.plot(
        site["lon"], site["lat"], marker="*", markersize=9,
        markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.7,
        zorder=5, transform=ccrs.PlateCarree(),
    )
    ax_loc.set_xticks([])
    ax_loc.set_yticks([])
    for spine in ax_loc.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="P0",
                    help="observation site key in config/simulations.yaml (default: P0)")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--half-width-deg", type=float, default=2.0,
                    help="transect half-width in latitude degrees (default: 2.0)")
    ap.add_argument("--zmax", type=float, default=1000.0,
                    help="cap vertical axis at this height in m MSL (default: 1000)")
    ap.add_argument("--npoints", type=int, default=400)
    ap.add_argument("--wind-stride", type=int, default=5)
    ap.add_argument("--wind-lstride", type=int, default=3)
    ap.add_argument("--w-exag", type=float, default=10.0)
    ap.add_argument("--wind-scale", type=float, default=130.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation(args.site)
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    print(f"[{sim.key}] finding windiest-day Tmax moment near {args.site}...")
    moments = find_moments(
        sim.history_dir, mesh_name, site["lat"], site["lon"], target_height_m=100.0
    )
    print(f"  day {moments.day.date()}, sea-breeze moment (Tmax): {moments.tmax_time}")

    static_path = config.data_root / f"{mesh_name}.static.nc"
    static = xr.open_dataset(static_path)
    lat = np.rad2deg(static["latCell"].values)
    lon = np.rad2deg(static["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    static.close()

    cells, lat_cols = sample_meridional_transect(
        site["lat"], site["lon"], args.half_width_deg, lat, lon, args.npoints
    )
    dist_km = (lat_cols - site["lat"]) * KM_PER_DEG_LAT

    init_path = sim.history_dir / f"{mesh_name}.init.nc"
    with xr.open_dataset(init_path) as ds_init:
        zgrid_full = ds_init["zgrid"].values[cells].T

    n_levels = int(np.searchsorted(
        zgrid_full[:, np.argmin(zgrid_full[0])], args.zmax * 1.3
    )) + 1
    n_levels = min(n_levels, zgrid_full.shape[0] - 1)
    z_iface = zgrid_full[: n_levels + 1]
    ter_cols = zgrid_full[0]

    print(f"  loading fields for Tmax ({moments.tmax_time})...")
    d = load_column_fields(sim.history_dir, moments.tmax_time, cells, n_levels)

    xe = x_edges(dist_km)
    x2d = np.tile(xe, (n_levels + 1, 1))
    y2d = interface_corners(z_iface)
    z_center = 0.5 * (z_iface[:-1] + z_iface[1:])

    norm = mcolors.Normalize(vmin=float(np.nanmin(d["temp_c"])),
                             vmax=float(np.nanmax(d["temp_c"])))

    fig, ax = plt.subplots(figsize=(12, 6))

    mesh = ax.pcolormesh(x2d, y2d, d["temp_c"], cmap="RdYlBu_r", norm=norm,
                         shading="flat")
    ax.fill_between(dist_km, 0.0, ter_cols, color="0.4", zorder=5, linewidth=0)
    ax.plot(dist_km, ter_cols, color="k", linewidth=0.6, zorder=6)

    # Wind: along-transect = meridional (v) + w; cross-transect = zonal (u)
    ci = np.arange(0, len(cells), args.wind_stride)
    li = np.arange(0, n_levels, args.wind_lstride)
    C, L = np.meshgrid(ci, li)
    X = np.tile(dist_km, (n_levels, 1))[L, C]
    Z = z_center[L, C]
    ax.quiver(X, Z, d["v"][L, C], d["w"][L, C] * args.w_exag, angles="uv",
              pivot="mid", scale_units="width", scale=args.wind_scale,
              width=0.0022, color="k", alpha=0.85, zorder=8)

    amax = float(np.nanmax(np.abs(d["u"]))) or 1.0
    Un = d["u"][L, C]
    sig = np.abs(Un) >= 0.15 * amax
    size = 10.0 + 60.0 * (np.abs(Un) / amax)
    toward = sig & (Un > 0)
    away = sig & (Un <= 0)
    ax.scatter(X[sig], Z[sig], s=size[sig], facecolors="none",
               edgecolors="k", linewidths=0.5, alpha=0.7, zorder=9)
    ax.scatter(X[toward], Z[toward], s=size[toward] * 0.22,
               c="k", marker="o", alpha=0.7, zorder=10)
    ax.scatter(X[away], Z[away], s=size[away] * 0.5,
               c="k", marker="x", linewidths=0.7, alpha=0.7, zorder=10)

    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.2, zorder=4)

    ax.set_ylim(0.0, args.zmax)
    ax.set_xlabel(f"Distance from {args.site} (km, north +)", fontsize=LABEL_FS)
    ax.set_ylabel("Height (m MSL)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.25)

    lat_ax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: site["lat"] + x / KM_PER_DEG_LAT,
                   lambda lt: (lt - site["lat"]) * KM_PER_DEG_LAT),
    )
    lat_ax.set_xlabel("Latitude (°)", fontsize=TICK_FS, labelpad=3)
    lat_ax.tick_params(labelsize=TICK_FS)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02, aspect=28)
    cbar.set_label("Temperature (°C)", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    draw_location_inset(ax, config.mesh["domain"], site, args.half_width_deg)

    fig.text(
        0.5, -0.02,
        f"wind: in-plane = v + w (×{args.w_exag:g})  |  "
        "cross-plane: ● eastward  ✕ westward",
        ha="center", fontsize=ANNOT_FS, color="dimgray",
    )

    ax.set_title(
        f"Sea-breeze cross-section — {args.site}  |  {moments.tmax_time} UTC  "
        f"(windiest day: {moments.day.date()})",
        fontsize=TITLE_FS, pad=10,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / f"fig02_xsection_seabreeze_{args.site.lower()}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
