"""Find the day of strongest wind at the P0 grid cell, and that day's warmest
and coolest moments at the same level — the reference points shared by the
sea-breeze map and cross-section figures (they must agree on the exact same
timestamps, so this lives in one place rather than being reimplemented twice).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import io, vertical
from .thermo import temperature_from_theta_pressure

KM_PER_DEG_LAT = 111.0


@dataclass(frozen=True)
class SeaBreezeMoments:
    icell: int
    dist_km: float
    level_index: int
    level_height_m: float
    windiest_time: pd.Timestamp
    windiest_speed: float
    day: pd.Timestamp
    tmax_time: pd.Timestamp
    tmax_value: float
    tmin_time: pd.Timestamp
    tmin_value: float


def nearest_cell(ds: xr.Dataset, lat0: float, lon0: float) -> tuple[int, float]:
    lat = np.rad2deg(ds["latCell"].values)
    lon = np.rad2deg(ds["lonCell"].values)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    dist2 = (lat - lat0) ** 2 + (lon - lon0) ** 2
    icell = int(np.argmin(dist2))
    dist_km = float(np.sqrt(dist2[icell])) * KM_PER_DEG_LAT
    return icell, dist_km


def find_moments(
    history_dir: Path,
    mesh_name: str,
    site_lat: float,
    site_lon: float,
    target_height_m: float = 100.0,
    height_tol_m: float = 1.0,
) -> SeaBreezeMoments:
    """Locate the cell nearest (site_lat, site_lon), the hour of strongest
    wind speed at ``target_height_m`` AGL across the whole run in
    ``history_dir``, and — within that calendar day — the hours of maximum
    and minimum temperature at the same level and cell.
    """
    init_path = history_dir / f"{mesh_name}.init.nc"
    ds_init = xr.open_dataset(init_path)
    icell, dist_km = nearest_cell(ds_init, site_lat, site_lon)
    zgrid = ds_init["zgrid"].values[icell]
    centers_agl = vertical.layer_center_heights(zgrid) - zgrid[0]
    ds_init.close()

    k = int(np.argmin(np.abs(centers_agl - target_height_m)))
    if abs(centers_agl[k] - target_height_m) > height_tol_m:
        raise SystemExit(
            f"\nERROR: no model level within {height_tol_m} m of "
            f"{target_height_m} m AGL at cell {icell} "
            f"(closest is {centers_agl[k]:.2f} m).")
    level_height = float(centers_agl[k])

    files = io.find_history_files(history_dir)
    times = pd.DatetimeIndex(io.read_timestamps(files))

    with io.open_history(files, variables=["uReconstructZonal", "uReconstructMeridional"],
                          chunks={"Time": 1}) as ds:
        ds_cell = ds.isel(nCells=icell, nVertLevels=k)
        speed = np.sqrt(ds_cell["uReconstructZonal"] ** 2
                        + ds_cell["uReconstructMeridional"] ** 2).values

    i_max = int(np.argmax(speed))
    windiest_time = times[i_max]
    windiest_speed = float(speed[i_max])
    day = windiest_time.normalize()

    day_mask = (times >= day) & (times < day + pd.Timedelta(days=1))
    day_files = [f for f, m in zip(files, day_mask) if m]
    day_times = times[day_mask]

    with io.open_history(day_files, variables=["theta", "pressure"],
                          chunks={"Time": 1}) as ds:
        ds_cell = ds.isel(nCells=icell, nVertLevels=k)
        temp = temperature_from_theta_pressure(
            ds_cell["theta"].values, ds_cell["pressure"].values)

    i_tmax = int(np.argmax(temp))
    i_tmin = int(np.argmin(temp))

    return SeaBreezeMoments(
        icell=icell, dist_km=dist_km, level_index=k, level_height_m=level_height,
        windiest_time=windiest_time, windiest_speed=windiest_speed, day=day,
        tmax_time=day_times[i_tmax], tmax_value=float(temp[i_tmax]),
        tmin_time=day_times[i_tmin], tmin_value=float(temp[i_tmin]),
    )
