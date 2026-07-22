"""Lazy history I/O for the meqbr_05km runs.

MPAS writes one timestep per ``history.*.nc`` file. These helpers list and open
those files without ever pulling whole 3D fields into memory: timestamps come
from the tiny ``xtime`` variable, and full opens are chunked/lazy via xarray.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAD2DEG = 180.0 / np.pi


def find_history_files(run_dir: str | Path) -> list[Path]:
    """Return ``history.*.nc`` files in *run_dir*, sorted by filename.

    MPAS encodes the valid time in the filename, so lexical sort is chronological.
    """
    return sorted(Path(run_dir).glob("history.*.nc"))


def _decode_xtime(raw) -> pd.Timestamp:
    """Decode one MPAS ``xtime`` entry ('YYYY-MM-DD_hh:mm:ss') to a Timestamp."""
    if isinstance(raw, bytes):
        text = raw.decode()
    elif isinstance(raw, np.ndarray):  # array of single-char bytes/str
        text = b"".join(np.atleast_1d(raw).astype("S1").ravel()).decode()
    else:
        text = str(raw)
    return pd.Timestamp(text.strip().replace("_", " "))


def read_timestamp(path: str | Path) -> pd.Timestamp:
    """Read the valid time of a single history file (reads only ``xtime``)."""
    with xr.open_dataset(path, decode_times=False) as ds:
        return _decode_xtime(ds["xtime"].values[0])


def read_timestamps(paths: list[Path]) -> pd.DatetimeIndex:
    """Read valid times for many files without loading their data variables."""
    return pd.DatetimeIndex([read_timestamp(p) for p in paths])


def list_variables(path: str | Path) -> list[str]:
    """Variable names present in a history file (no data read)."""
    with xr.open_dataset(path, decode_times=False) as ds:
        return list(ds.variables)


def open_history(
    paths: list[Path] | str | Path,
    *,
    variables: list[str] | None = None,
    chunks: dict | None = None,
) -> xr.Dataset:
    """Open one or many history files as a single lazy dataset concatenated on Time.

    Pass *variables* to drop everything else at open time (cheap way to avoid
    touching large 3D fields). *chunks* enables dask-backed lazy access.
    """
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    open_kwargs = dict(decode_times=False)
    if chunks is not None:
        open_kwargs["chunks"] = chunks

    if len(paths) == 1:
        ds = xr.open_dataset(paths[0], **open_kwargs)
    else:
        ds = xr.open_mfdataset(
            paths, combine="nested", concat_dim="Time", **open_kwargs
        )
    if variables is not None:
        keep = [v for v in variables if v in ds.variables]
        ds = ds[keep]
    return ds


def cell_lonlat_degrees(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (lon, lat) of cell centers in degrees, converted from radians.

    ``lonCell`` is wrapped to (-180, 180] for convenient plotting.
    """
    lat = ds["latCell"] * RAD2DEG
    lon = ((ds["lonCell"] * RAD2DEG + 180.0) % 360.0) - 180.0
    return lon, lat
