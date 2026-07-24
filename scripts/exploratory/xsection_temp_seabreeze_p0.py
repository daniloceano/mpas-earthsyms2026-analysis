#!/usr/bin/env python
"""Exploratory cross-section: temperature + wind along a 4-deg meridional
transect through an observation site, for the same two sea-breeze moments as
``map_t100m_wind_seabreeze_p0.py`` (Tmin/Tmax on the windiest day near the site).

"Longitudinal" transect per the request: longitude fixed at the site's,
latitude spanning site_lat +/- 2 deg (4 deg total), capped at 1000 m MSL.
Adapts the terrain-following pcolormesh technique and transect-relative wind
decomposition from ``mpas_cross_section.py`` (same source repo as
``mpas_viz.py``): temperature colors the field, terrain is filled from
``zgrid``, and wind is decomposed into along-transect (meridional wind +
vertical velocity, in-plane arrows) and cross-transect (zonal wind, normal to
the page - dot=towards viewer/eastward, cross=away/westward) components.

    python scripts/exploratory/xsection_temp_seabreeze_p0.py
    python scripts/exploratory/xsection_temp_seabreeze_p0.py --site LPI --half-width-deg 2 --zmax 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"
KM_PER_DEG_LAT = 111.0


def sample_meridional_transect(
    lat_center: float, lon_fixed: float, half_width_deg: float,
    lat_cell: np.ndarray, lon_cell: np.ndarray, npoints: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cell sampling along a fixed-longitude line.

    Returns (cells, lat_of_cells) - 0-based cell indices in transect order,
    consecutive duplicates collapsed (adapted from mpas_cross_section.py's
    sample_transect, specialized to a straight meridian).
    """
    plat = np.linspace(lat_center - half_width_deg, lat_center + half_width_deg, npoints)
    plon = np.full_like(plat, lon_fixed)
    tree = cKDTree(np.column_stack([lat_cell, lon_cell]))
    _, idx = tree.query(np.column_stack([plat, plon]))

    cells = [int(idx[0])]
    for c in idx[1:]:
        if int(c) != cells[-1]:
            cells.append(int(c))
    cells = np.asarray(cells, dtype=int)
    return cells, lat_cell[cells]


def x_edges(xc: np.ndarray) -> np.ndarray:
    """Midpoint edges for column centers (length N -> N+1)."""
    if len(xc) == 1:
        return np.array([xc[0] - 0.5, xc[0] + 0.5])
    mids = 0.5 * (xc[:-1] + xc[1:])
    first = xc[0] - 0.5 * (xc[1] - xc[0])
    last = xc[-1] + 0.5 * (xc[-1] - xc[-2])
    return np.concatenate([[first], mids, [last]])


def interface_corners(z_iface: np.ndarray) -> np.ndarray:
    """(nInterfaces, ncols) -> (nInterfaces, ncols+1), columns averaged onto x-edges."""
    left = z_iface[:, :1]
    mid = 0.5 * (z_iface[:, :-1] + z_iface[:, 1:])
    right = z_iface[:, -1:]
    return np.concatenate([left, mid, right], axis=1)


def load_column_fields(history_dir: Path, mesh_name: str, ts: pd.Timestamp,
                       cells: np.ndarray, n_levels: int) -> dict:
    files = io.find_history_files(history_dir)
    times = pd.DatetimeIndex(io.read_timestamps(files))
    file = files[times.get_loc(ts)]
    with io.open_history(
        file, variables=["theta", "pressure", "uReconstructZonal",
                         "uReconstructMeridional", "w"]
    ) as ds:
        ds = ds.isel(Time=0, nVertLevels=slice(0, n_levels)).isel(nCells=cells)
        theta = ds["theta"].values.T  # (n_levels, ncols)
        pressure = ds["pressure"].values.T
        u = ds["uReconstructZonal"].values.T
        v = ds["uReconstructMeridional"].values.T
        w_iface = ds["w"].isel(nVertLevelsP1=slice(0, n_levels + 1)).values.T
        w = 0.5 * (w_iface[:-1, :] + w_iface[1:, :])  # interfaces -> centers
    temp_c = temperature_from_theta_pressure(theta, pressure) - 273.15
    return {"file": file, "temp_c": temp_c, "u": u, "v": v, "w": w}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="P0", help="observation site key in "
                     "config/simulations.yaml (default: P0)")
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--half-width-deg", type=float, default=2.0,
                     help="transect half-width in latitude degrees (default: 2.0, "
                     "i.e. a 4-deg transect)")
    ap.add_argument("--zmax", type=float, default=1000.0,
                     help="cap the vertical axis at this height, m MSL (default: 1000)")
    ap.add_argument("--npoints", type=int, default=400)
    ap.add_argument("--wind-stride", type=int, default=5,
                     help="along-transect stride for wind symbols (default: 5)")
    ap.add_argument("--wind-lstride", type=int, default=3,
                     help="vertical stride for wind symbols (default: 3)")
    ap.add_argument("--w-exag", type=float, default=10.0,
                     help="vertical-velocity exaggeration for in-plane arrow tilt "
                     "(default: 10)")
    ap.add_argument("--wind-scale", type=float, default=130.0,
                     help="in-plane quiver scale (data units per axes-width unit; "
                     "larger = shorter arrows). Explicit, not auto, since "
                     "auto-scaling was making arrows overlap (default: 25)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation(args.site)
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    print(f"[{sim.key}] finding the windiest day + Tmin/Tmax moments near "
          f"{args.site} (reads the whole run once - ~45s)...")
    moments = find_moments(sim.history_dir, mesh_name, site["lat"], site["lon"],
                           target_height_m=100.0)
    print(f"  day {moments.day.date()}, Tmin {moments.tmin_time}, "
          f"Tmax {moments.tmax_time}")

    static_path = config.data_root / f"{mesh_name}.static.nc"
    static = xr.open_dataset(static_path)
    lat = np.rad2deg(static["latCell"].values)
    lon = np.rad2deg(static["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    static.close()

    cells, lat_cols = sample_meridional_transect(
        site["lat"], site["lon"], args.half_width_deg, lat, lon, args.npoints)
    print(f"transect: {len(cells)} distinct columns, "
          f"lat {lat_cols.min():.3f} to {lat_cols.max():.3f} @ lon {site['lon']:.3f}")
    # Pure meridional line: distance from P0 is exact from the latitude delta
    # alone (no cosine term needed). Plotting/quiver use this km distance, not
    # raw latitude degrees, so the wind decomposition's "uv" angles stay
    # physically meaningful (x and y axes both in length units).
    dist_km = (lat_cols - site["lat"]) * KM_PER_DEG_LAT

    init_path = sim.history_dir / f"{mesh_name}.init.nc"
    with xr.open_dataset(init_path) as ds_init:
        zgrid_full = ds_init["zgrid"].values[cells].T  # (nVertLevelsP1, ncols)

    # Enough interfaces to comfortably cover zmax at the flattest column.
    n_levels = int(np.searchsorted(zgrid_full[:, np.argmin(zgrid_full[0])],
                                   args.zmax * 1.3)) + 1
    n_levels = min(n_levels, zgrid_full.shape[0] - 1)
    z_iface = zgrid_full[: n_levels + 1]
    ter_cols = zgrid_full[0]
    print(f"using {n_levels} vertical levels (covers up to "
          f"~{z_iface[-1].min():.0f}-{z_iface[-1].max():.0f} m MSL across columns)")

    panels = [("Tmin - land breeze?", moments.tmin_time),
             ("Tmax - sea breeze?", moments.tmax_time)]
    data = [(label, ts, load_column_fields(sim.history_dir, mesh_name, ts, cells, n_levels))
            for label, ts in panels]

    vmin = min(float(np.min(d["temp_c"])) for _, _, d in data)
    vmax = max(float(np.max(d["temp_c"])) for _, _, d in data)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    xe = x_edges(dist_km)
    x2d = np.tile(xe, (n_levels + 1, 1))
    y2d = interface_corners(z_iface)
    z_center = 0.5 * (z_iface[:-1] + z_iface[1:])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    mappable = None
    for i, (ax, (label, ts, d)) in enumerate(zip(axes, data)):
        mesh = ax.pcolormesh(x2d, y2d, d["temp_c"], cmap="RdYlBu_r", norm=norm,
                             shading="flat")
        mappable = mesh
        ax.fill_between(dist_km, 0.0, ter_cols, color="0.4", zorder=5, linewidth=0)
        ax.plot(dist_km, ter_cols, color="k", linewidth=0.6, zorder=6)

        # Wind decomposed relative to the transect: along-transect = meridional
        # wind (v), normal-to-page = zonal wind (u); positive u (eastward) reads
        # as towards-viewer (dot), matching mpas_cross_section.py's convention.
        ci = np.arange(0, len(cells), args.wind_stride)
        li = np.arange(0, n_levels, args.wind_lstride)
        C, L = np.meshgrid(ci, li)
        X = np.tile(dist_km, (n_levels, 1))[L, C]
        Z = z_center[L, C]
        Ut, W, Un = d["v"][L, C], d["w"][L, C], d["u"][L, C]

        ax.quiver(X, Z, Ut, W * args.w_exag, angles="uv", pivot="mid",
                  scale_units="width", scale=args.wind_scale, width=0.0022,
                  color="k", alpha=0.85, zorder=8)

        amax = float(np.nanmax(np.abs(Un))) or 1.0
        sig = np.abs(Un) >= 0.15 * amax
        size = 10.0 + 60.0 * (np.abs(Un) / amax)
        toward, away = sig & (Un > 0), sig & (Un <= 0)
        ax.scatter(X[sig], Z[sig], s=size[sig], facecolors="none", edgecolors="k",
                  linewidths=0.5, alpha=0.7, zorder=9)
        ax.scatter(X[toward], Z[toward], s=size[toward] * 0.22, c="k", marker="o",
                  alpha=0.7, zorder=10)
        ax.scatter(X[away], Z[away], s=size[away] * 0.5, c="k", marker="x",
                  linewidths=0.7, alpha=0.7, zorder=10)

        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, zorder=4)
        ax.set_ylim(0.0, args.zmax)
        ax.set_xlabel(f"Distance from {args.site} (km, north positive)")
        ax.set_title(f"{label}\n{ts} UTC", fontsize=11, pad=26)
        ax.grid(True, alpha=0.25)

        lat_ax = ax.secondary_xaxis(
            "top", functions=(lambda x: site["lat"] + x / KM_PER_DEG_LAT,
                              lambda lt: (lt - site["lat"]) * KM_PER_DEG_LAT))
        if i == 0:
            lat_ax.set_xlabel("Latitude (deg)", fontsize=9, labelpad=2)
        lat_ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Height (m MSL)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.colorbar(mappable, ax=axes, shrink=0.8, pad=0.06, aspect=25,
                label="Temperature (°C)")
    fig.text(0.5, 0.03,
             "wind: in-plane = meridional + w (x" + f"{args.w_exag:g})"
             "  |  normal: dot=eastward (towards viewer)  x=westward (away)",
             ha="center", fontsize=8.5, color="dimgray")

    fig.suptitle(
        f"MPAS {sim.label} - meridional cross-section thru {args.site} "
        f"(lon {site['lon']:.3f}, lat {site['lat'] - args.half_width_deg:.2f} to "
        f"{site['lat'] + args.half_width_deg:.2f}) - windiest day {moments.day.date()}",
        fontsize=12, y=0.995,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    default_name = f"xsection_temp_seabreeze_{args.site.lower()}.png"
    out = args.output or FIG_DIR / default_name
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
