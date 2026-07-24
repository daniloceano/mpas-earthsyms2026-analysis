#!/usr/bin/env python
"""Density scatter: MPAS vs P0 LiDAR wind speed, matched per vertical level.

For each model layer that lines up with a LiDAR measurement height (see
``scripts/exploratory/list_model_levels_at_p0.py``: 50/100/150/200 m match
exactly; the model's ~250 m layer is compared against the mean of the LiDAR's
240 m and 260 m channels), pairs hourly model output at the nearest ocean cell
to P0 with the nearest LiDAR record (<= 5 min away) and writes one
exploratory PNG: density-colored scatter, 1:1 line, linear regression, and
fit/error stats (N, R, RMSE, bias).

Comparison window is the intersection of model output (sim_2021 analysis
window) and LiDAR coverage: 2021-11-09 to 2021-12-01.

This is a sanity check ahead of a proper validation, not a final analysis —
in particular hourly-vs-instantaneous matching (no averaging window) and a
single nearest cell (no spatial interpolation) are both simplifications.

    python scripts/exploratory/scatter_wspeed_model_vs_p0.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.stats import gaussian_kde, linregress  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis import vertical  # noqa: E402
from mpas_analysis.verification import mse_decomposition  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "exploratory"
MATLAB_EPOCH_OFFSET_DAYS = 719529

# (label, model target height AGL in m, obs heights averaged to represent it)
LEVEL_MATCHES = [
    (50, 50, [50]),
    (100, 100, [100]),
    (150, 150, [150]),
    (200, 200, [200]),
    (250, 250, [240, 260]),
]
HEIGHT_TOL_M = 1.0  # required closeness between requested and actual model level height


def matlab_datenum_to_timestamps(mtime: np.ndarray) -> pd.DatetimeIndex:
    ts = pd.Timestamp("1970-01-01") + pd.to_timedelta(mtime - MATLAB_EPOCH_OFFSET_DAYS, unit="D")
    # Match io.read_timestamps' datetime64[ns] precision so merge_asof's join
    # keys have identical dtypes (pandas errors on e.g. ns vs us otherwise).
    return pd.DatetimeIndex(ts).as_unit("ns")


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
            "have changed — check with list_model_levels_at_p0.py.")
    return k


def density_scatter(ax, x: np.ndarray, y: np.ndarray) -> None:
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    order = np.argsort(density)
    sc = ax.scatter(x[order], y[order], c=density[order], cmap="viridis",
                     s=16, edgecolor="none")
    plt.colorbar(sc, ax=ax, label="point density", shrink=0.85)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    margin = 0.05 * (hi - lo)
    lims = (lo - margin, hi + margin)
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1")

    fit = linregress(x, y)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, fit.slope * xs + fit.intercept, color="crimson", lw=1.5,
            label=f"fit: y = {fit.slope:.2f}x + {fit.intercept:.2f}")

    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    # Takacs (1985) MSE decomposition: MSE_total = bias**2 + MSE_diss + MSE_disp.
    mse_diss, mse_disp = mse_decomposition(x, y)
    ax.text(
        0.03, 0.97,
        f"N = {len(x)}\nR = {fit.rvalue:.2f}\nRMSE = {rmse:.2f} m s$^{{-1}}$\n"
        f"Bias = {bias:+.2f} m s$^{{-1}}$\n"
        f"MSE$_{{disp}}$ = {mse_disp:.2f} m$^2$ s$^{{-2}}$\n"
        f"MSE$_{{diss}}$ = {mse_diss:.2f} m$^2$ s$^{{-2}}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--min-avail", type=float, default=50.0,
                     help="mask LiDAR bins with 'avail' below this percent (default: 50.0)")
    ap.add_argument("--tolerance-min", type=float, default=5.0,
                     help="max minutes between a model hour and its matched "
                     "LiDAR record (default: 5.0)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("P0")
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    # --- Observations ---
    mat_path = cfg.REPO_ROOT / site["data_file"]
    if not mat_path.exists():
        print(f"missing observation file: {mat_path}", file=sys.stderr)
        return 1
    L = loadmat(mat_path, simplify_cells=True)["L"]
    obs_times = matlab_datenum_to_timestamps(L["mtime"])
    obs_heights = np.asarray(L["heights"][0], dtype=int)
    obs_wspeed = np.where(L["avail"] < args.min_avail, np.nan, L["wspeed"])

    # --- Model: nearest cell + level heights (from init.nc, terrain-following) ---
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
    print(f"[P0] nearest cell idx={icell}, ~{dist_km:.2f} km away")

    # --- Model: hourly wind speed profile at that cell, whole sim_2021 run ---
    files = io.find_history_files(sim.history_dir)
    model_times = io.read_timestamps(files)
    print(f"[{sim.key}] reading uReconstructZonal/Meridional at cell {icell} "
          f"from {len(files)} history files...")
    with io.open_history(files, variables=["uReconstructZonal", "uReconstructMeridional"],
                          chunks={"Time": 1}) as ds:
        ds_cell = ds.isel(nCells=icell)
        speed = np.sqrt(ds_cell["uReconstructZonal"] ** 2
                        + ds_cell["uReconstructMeridional"] ** 2).values  # (Time, nVertLevels)

    # io.read_timestamps and matlab_datenum_to_timestamps can land on different
    # datetime64 resolutions (e.g. 'us' vs 'ns'); merge_asof requires identical
    # dtypes on both join keys.
    model_df_base = pd.DataFrame(
        {"time": pd.DatetimeIndex(model_times).as_unit("ns")}
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for label_m, model_target_m, obs_h_list in LEVEL_MATCHES:
        k = model_level_index(centers_agl, model_target_m)
        model_height = float(centers_agl[k])

        obs_cols = [int(np.argmin(np.abs(obs_heights - h))) for h in obs_h_list]
        with warnings.catch_warnings():
            # All-NaN rows (both heights masked by --min-avail) are expected
            # and become NaN here too, dropped later by the merge/dropna.
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            obs_series = np.nanmean(obs_wspeed[:, obs_cols], axis=1)
        obs_label = (f"{obs_h_list[0]} m" if len(obs_h_list) == 1
                     else f"mean of {obs_h_list[0]}-{obs_h_list[-1]} m")

        model_df = model_df_base.assign(model=speed[:, k]).sort_values("time")
        obs_df = pd.DataFrame({"time": obs_times, "obs": obs_series}).sort_values("time")

        merged = pd.merge_asof(
            model_df, obs_df, on="time", direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_min),
        ).dropna(subset=["model", "obs"])

        n = len(merged)
        print(f"  level ~{label_m} m (model {model_height:.1f} m, obs {obs_label}): "
              f"{n} matched records")
        if n < 10:
            print(f"    skipping plot (too few matched points: {n})")
            continue

        x = merged["obs"].to_numpy()
        y = merged["model"].to_numpy()

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        density_scatter(ax, x, y)
        ax.set_xlabel(f"P0 LiDAR wind speed @ {obs_label} (m s$^{{-1}}$)")
        ax.set_ylabel(f"MPAS wind speed @ {model_height:.0f} m (m s$^{{-1}}$)")
        ax.set_title(f"MPAS {sim.label} vs P0 — ~{label_m} m AGL\n"
                     f"{merged['time'].min():%Y-%m-%d} to {merged['time'].max():%Y-%m-%d}, "
                     "hourly")
        fig.tight_layout()

        out = FIG_DIR / f"scatter_wspeed_p0_{label_m:03d}m.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"    wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
