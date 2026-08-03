#!/usr/bin/env python
"""Poster Figure 4: MPAS vs LPI fixed-LiDAR wind speed, all matched levels pooled.

Pairs hourly model wind speed (nearest ocean cell to LPI) with the nearest
LPI LiDAR record (<= 5 min away) at every model layer that lines up with a
measurement height — 50/100/150/200 m all match exactly (see
``scripts/exploratory/list_model_levels_at_p0.py --site LPI``; unlike P0,
LPI has no 240/260 m pair to average for a ~250 m level) — then pools all
levels into one density scatter with a 1:1 line, linear regression, and
fit/error stats (N, R, RMSE, bias, and the Takacs 1985
MSE_dissipative/MSE_dispersive decomposition — see
``mpas_analysis.verification.mse_decomposition``).

Comparison window is the intersection of model output (sim_2022 analysis
window) and LPI coverage — the only simulation LPI overlaps at all.

    python scripts/poster/fig04_scatter_wspeed_lpi_all_levels.py
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
from scipy.stats import gaussian_kde, linregress  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402
from mpas_analysis import io  # noqa: E402
from mpas_analysis import vertical  # noqa: E402
from mpas_analysis.verification import mse_decomposition  # noqa: E402

FIG_DIR = cfg.REPO_ROOT / "figures" / "poster"

LEVEL_MATCHES = [50, 100, 150, 200]
HEIGHT_TOL_M = 1.0

# Poster-level font sizes (bumped for legibility at poster viewing distance)
TITLE_FS = 16
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 13
STAT_FS = 14
CAPTION_FS = 12
PANEL_LABEL = "(b)"


def density_colormap() -> "mcolors.Colormap":
    """Truncated 'inferno': drops the near-black floor and the pale-yellow
    ceiling of the stock colormap so low-density points stay visible against
    a white background and the high-density core reads as a strong, saturated
    orange-red instead of washing out to near-white."""
    base = plt.get_cmap("inferno")
    return mcolors.LinearSegmentedColormap.from_list(
        "inferno_poster", base(np.linspace(0.12, 0.92, 256))
    )


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", help="override data_root from paths.local.yaml")
    ap.add_argument("--paths", type=Path, help="alternative paths.local.yaml")
    ap.add_argument("--tolerance-min", type=float, default=5.0,
                     help="max minutes between a model hour and its matched "
                     "LPI record (default: 5.0)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--output", type=Path, help="output PNG (default: figures/poster/)")
    args = ap.parse_args()

    config = cfg.load_config(paths_file=args.paths, data_root=args.data_root)
    site = config.observation("LPI")
    sim = config.simulation(site["overlaps_simulation"])
    mesh_name = config.mesh["name"]

    # --- Observations ---
    csv_path = cfg.REPO_ROOT / site["data_file"]
    if not csv_path.exists():
        print(f"missing observation file: {csv_path}", file=sys.stderr)
        return 1
    obs_df_full = pd.read_csv(csv_path, parse_dates=["time"])
    obs_df_full["time"] = obs_df_full["time"].dt.as_unit("ns")

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
    print(f"[LPI] nearest cell idx={icell}, ~{dist_km:.2f} km away")

    # --- Model: hourly wind speed profile at that cell, whole sim run ---
    files = io.find_history_files(sim.history_dir)
    model_times = io.read_timestamps(files)
    print(f"[{sim.key}] reading uReconstructZonal/Meridional at cell {icell} "
          f"from {len(files)} history files...")
    with io.open_history(files, variables=["uReconstructZonal", "uReconstructMeridional"],
                          chunks={"Time": 1}) as ds:
        ds_cell = ds.isel(nCells=icell)
        speed = np.sqrt(ds_cell["uReconstructZonal"] ** 2
                        + ds_cell["uReconstructMeridional"] ** 2).values  # (Time, nVertLevels)

    model_df_base = pd.DataFrame({"time": pd.DatetimeIndex(model_times).as_unit("ns")})

    # --- Match + pool every level ---
    pooled_obs: list[np.ndarray] = []
    pooled_model: list[np.ndarray] = []
    level_counts: list[tuple[int, int]] = []

    for h in LEVEL_MATCHES:
        k = model_level_index(centers_agl, h)
        model_height = float(centers_agl[k])

        obs_df = obs_df_full[["time", f"spd_{h}"]].rename(
            columns={f"spd_{h}": "obs"}).sort_values("time")
        model_df = model_df_base.assign(model=speed[:, k]).sort_values("time")

        merged = pd.merge_asof(
            model_df, obs_df, on="time", direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_min),
        ).dropna(subset=["model", "obs"])

        n = len(merged)
        print(f"  level ~{h} m (model {model_height:.1f} m): {n} matched records")
        level_counts.append((h, n))
        if n == 0:
            continue
        pooled_obs.append(merged["obs"].to_numpy())
        pooled_model.append(merged["model"].to_numpy())

    x = np.concatenate(pooled_obs)
    y = np.concatenate(pooled_model)
    n_total = len(x)
    print(f"pooled {n_total} records across {len(LEVEL_MATCHES)} levels")

    # --- Density scatter (all levels pooled) ---
    fig, ax = plt.subplots(figsize=(8.3, 7.5))
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    order = np.argsort(density)
    sc = ax.scatter(x[order], y[order], c=density[order], cmap=density_colormap(),
                    s=20, edgecolor="none")
    cbar = fig.colorbar(sc, ax=ax, label="point density", shrink=0.8, pad=0.02, aspect=30)
    cbar.set_label("point density", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS, width=1.2)
    cbar.outline.set_linewidth(1.3)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    margin = 0.05 * (hi - lo)
    lims = (lo - margin, hi + margin)
    ax.plot(lims, lims, "k--", lw=1.8, label="1:1")

    fit = linregress(x, y)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, fit.slope * xs + fit.intercept, color="royalblue", lw=2.4,
            label=f"fit: y = {fit.slope:.2f}x + {fit.intercept:.2f}")

    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    mse_diss, mse_disp = mse_decomposition(x, y)
    level_range_str = f"{level_counts[0][0]}–{level_counts[-1][0]} m"
    per_level_str = "; ".join(f"{h}m N={n}" for h, n in level_counts)
    ax.text(
        0.03, 0.97,
        f"N = {n_total}\nR = {fit.rvalue:.2f}\nRMSE = {rmse:.2f} m s$^{{-1}}$\n"
        f"Bias = {bias:+.2f} m s$^{{-1}}$\n"
        f"MSE$_{{disp}}$ = {mse_disp:.2f} m$^2$ s$^{{-2}}$\n"
        f"MSE$_{{diss}}$ = {mse_diss:.2f} m$^2$ s$^{{-2}}$",
        transform=ax.transAxes, va="top", ha="left", fontsize=STAT_FS,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.92, lw=1.3),
    )

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=LEGEND_FS)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.tick_params(labelsize=TICK_FS, width=1.2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.set_xlabel("LPI LiDAR wind speed (m s$^{-1}$)", fontsize=LABEL_FS)
    ax.set_ylabel("MPAS wind speed (m s$^{-1}$)", fontsize=LABEL_FS)
    ax.set_title(f"{PANEL_LABEL} MPAS {sim.label} vs LPI — wind speed",
                fontsize=TITLE_FS)
    fig.tight_layout()
    # Pooled level range moved here (from the title) to keep the title short
    # enough to clear the colorbar at poster-scale font sizes.
    fig.text(0.5, 0.01, f"levels pooled ({level_range_str}): {per_level_str}",
              ha="center", fontsize=CAPTION_FS, color="dimgray")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or FIG_DIR / "fig04_scatter_wspeed_lpi_all_levels.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
