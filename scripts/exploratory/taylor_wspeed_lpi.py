#!/usr/bin/env python
"""Taylor diagram: MPAS vs LPI fixed-LiDAR wind speed, per vertical level + pooled.

Same model/obs matching as ``scatter_wspeed_model_vs_lpi.py`` (nearest ocean
cell to LPI, hourly model paired with the nearest LPI record <= 5 min away,
50/100/150/200 m all exact matches — LPI has no 240/260 m pair for a ~250 m
level the way P0 does), summarized on a single normalized Taylor diagram:
each level is one marker at (normalized std dev = std(model)/std(obs),
correlation), plus one marker for all levels pooled together, for reference.

    python scripts/exploratory/taylor_wspeed_lpi.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpl_toolkits.axisartist.floating_axes as FA  # noqa: E402
import mpl_toolkits.axisartist.grid_finder as GF  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.projections import PolarAxes  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis import vertical  # noqa: E402
from mpas_analysis.verification import taylor_stats  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"

LEVEL_MATCHES = [50, 100, 150, 200]
HEIGHT_TOL_M = 1.0
MARKERS = ["o", "s", "^", "D", "v"]


def nearest_cell(ds: xr.Dataset, lat0: float, lon0: float) -> tuple[int, float, bool]:
    lat = np.rad2deg(ds["latCell"].values)
    lon = np.rad2deg(ds["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    dist2 = (lat - lat0) ** 2 + (lon - lon0) ** 2
    icell = int(np.argmin(dist2))
    dist_km = float(np.sqrt(dist2[icell])) * 111.0
    is_land = bool(ds["landmask"].values[icell])
    return icell, dist_km, is_land


def model_level_index(centers_agl: np.ndarray, target_m: float, tol_m: float = HEIGHT_TOL_M) -> int:
    k = int(np.argmin(np.abs(centers_agl - target_m)))
    if abs(centers_agl[k] - target_m) > tol_m:
        raise SystemExit(
            f"\nERROR: no model level within {tol_m} m of {target_m} m AGL "
            f"(closest is {centers_agl[k]:.2f} m). Mesh/vertical config may "
            "have changed — check with list_model_levels_at_p0.py --site LPI.")
    return k


def setup_taylor_axes(fig, rect=111, srange=(0.0, 1.6)):
    """Build a normalized Taylor-diagram quarter-polar axes (correlation as
    angle 0-90 deg, normalized std dev as radius), using the standard
    floating-axes construction for curvilinear polar grids in matplotlib.
    Returns (grid_axes, data_axes) — plot data on the returned data_axes.
    """
    tr = PolarAxes.PolarTransform(apply_theta_transforms=False)

    rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    tlocs = np.arccos(rlocs)
    gl1 = GF.FixedLocator(tlocs.tolist())
    tf1 = GF.DictFormatter({t: f"{r:.2f}".rstrip("0").rstrip(".") or "0"
                            for t, r in zip(tlocs, rlocs)})

    ghelper = FA.GridHelperCurveLinear(
        tr, extremes=(0.0, np.pi / 2, srange[0], srange[1]),
        grid_locator1=gl1, tick_formatter1=tf1,
    )
    ax = fig.add_subplot(rect, axes_class=FA.FloatingAxes, grid_helper=ghelper)

    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Normalized standard deviation")

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")

    ax.axis["bottom"].set_visible(False)

    ax_polar = ax.get_aux_axes(tr)
    ax_polar.patch = ax.patch
    ax.patch.zorder = 0.9
    return ax, ax_polar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--tolerance-min", type=float, default=5.0,
                     help="max minutes between a model hour and its matched "
                     "LPI record (default: 5.0)")
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/exploratory/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("LPI")
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    csv_path = cfg.REPO_ROOT / site["data_file"]
    if not csv_path.exists():
        print(f"missing observation file: {csv_path}", file=sys.stderr)
        return 1
    obs_df_full = pd.read_csv(csv_path, parse_dates=["time"])
    obs_df_full["time"] = obs_df_full["time"].dt.as_unit("ns")

    init_path = sim.history_dir / f"{mesh_name}.init.nc"
    if not init_path.exists():
        print(f"missing init file: {init_path}", file=sys.stderr)
        return 1
    ds_init = xr.open_dataset(init_path)
    icell, dist_km, is_land = nearest_cell(ds_init, site["lat"], site["lon"])
    if is_land:
        print("WARNING: nearest cell is land, not ocean.", file=sys.stderr)
    zgrid = ds_init["zgrid"].values[icell]
    centers_agl = vertical.layer_center_heights(zgrid) - zgrid[0]
    ds_init.close()
    print(f"[LPI] nearest cell idx={icell}, ~{dist_km:.2f} km away")

    files = io.find_history_files(sim.history_dir)
    model_times = io.read_timestamps(files)
    print(f"[{sim.key}] reading uReconstructZonal/Meridional at cell {icell} "
          f"from {len(files)} history files...")
    with io.open_history(files, variables=["uReconstructZonal", "uReconstructMeridional"],
                          chunks={"Time": 1}) as ds:
        ds_cell = ds.isel(nCells=icell)
        speed = np.sqrt(ds_cell["uReconstructZonal"] ** 2
                        + ds_cell["uReconstructMeridional"] ** 2).values

    model_df_base = pd.DataFrame({"time": pd.DatetimeIndex(model_times).as_unit("ns")})

    pooled_obs: list[np.ndarray] = []
    pooled_model: list[np.ndarray] = []
    points: list[tuple[str, float, float, int]] = []  # (label, stdratio, corr, n)

    for h in LEVEL_MATCHES:
        k = model_level_index(centers_agl, h)
        obs_df = obs_df_full[["time", f"spd_{h}"]].rename(
            columns={f"spd_{h}": "obs"}).sort_values("time")
        model_df = model_df_base.assign(model=speed[:, k]).sort_values("time")
        merged = pd.merge_asof(
            model_df, obs_df, on="time", direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_min),
        ).dropna(subset=["model", "obs"])

        n = len(merged)
        print(f"  level ~{h} m: {n} matched records")
        if n == 0:
            continue
        x = merged["obs"].to_numpy()
        y = merged["model"].to_numpy()
        stdratio, corr = taylor_stats(x, y)
        points.append((f"{h} m", stdratio, corr, n))
        pooled_obs.append(x)
        pooled_model.append(y)

    x_all = np.concatenate(pooled_obs)
    y_all = np.concatenate(pooled_model)
    stdratio_all, corr_all = taylor_stats(x_all, y_all)
    print(f"pooled: N={len(x_all)} stdratio={stdratio_all:.2f} corr={corr_all:.2f}")

    fig = plt.figure(figsize=(8, 7))
    srange = (0.0, 1.3)
    _, ax_polar = setup_taylor_axes(fig, 111, srange=srange)

    ax_polar.plot(0, 1, marker="o", color="black", markersize=8, zorder=5)
    ax_polar.annotate("REF (obs)", xy=(0, 1), xytext=(-28, 10),
                      textcoords="offset points", fontsize=8, annotation_clip=False)

    theta_arc = np.linspace(0, np.pi / 2, 100)
    ax_polar.plot(theta_arc, np.ones_like(theta_arc), "k--", lw=0.8, alpha=0.6)

    rs, ts = np.meshgrid(np.linspace(srange[0], srange[1], 150),
                         np.linspace(0, np.pi / 2, 150))
    rms = np.sqrt(1 + rs**2 - 2 * rs * np.cos(ts))
    contours = ax_polar.contour(ts, rs, rms, levels=np.arange(0.25, srange[1], 0.25),
                                colors="grey", linestyles=":", linewidths=0.6)
    ax_polar.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    cmap = plt.cm.tab10
    for i, (label, stdratio, corr, n) in enumerate(points):
        theta = np.arccos(np.clip(corr, -1, 1))
        ax_polar.plot(theta, stdratio, marker=MARKERS[i % len(MARKERS)],
                     color=cmap(i), markersize=11, markeredgecolor="black",
                     markeredgewidth=0.5, linestyle="none",
                     label=f"{label} (N={n})", zorder=6)

    theta_all = np.arccos(np.clip(corr_all, -1, 1))
    ax_polar.plot(theta_all, stdratio_all, marker="*", color="black",
                 markersize=18, markeredgecolor="white", markeredgewidth=0.6,
                 linestyle="none", label=f"all levels pooled (N={len(x_all)})",
                 zorder=7)

    ax_polar.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=9,
                    frameon=True)
    fig.suptitle(f"Taylor diagram — MPAS {sim.label} vs LPI wind speed", fontsize=13)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / "taylor_wspeed_lpi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
