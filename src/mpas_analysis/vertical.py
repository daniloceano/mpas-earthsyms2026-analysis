"""MPAS vertical-grid helpers.

MPAS is vertically staggered: ``zgrid`` and ``w`` live on layer *interfaces*
(``nVertLevelsP1``), while winds, theta, rho, pressure live at layer *centers*
(``nVertLevels``). Height-specific extraction (e.g. "wind at 50 m") must use
layer-center heights derived from adjacent interfaces — never the interface
heights themselves. These helpers encapsulate that.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def layer_center_heights(zgrid: xr.DataArray | np.ndarray):
    """Layer-center heights (MSL) as the midpoint of adjacent ``zgrid`` interfaces.

    Input has an interface dimension of length ``nVertLevelsP1``; output has the
    center dimension ``nVertLevels`` (one shorter). Works on numpy arrays or
    xarray DataArrays (interfaces assumed to be the last dimension).
    """
    if isinstance(zgrid, xr.DataArray):
        dim = zgrid.dims[-1]
        lower = zgrid.isel({dim: slice(None, -1)})
        upper = zgrid.isel({dim: slice(1, None)})
        centers = 0.5 * (lower.data + upper.data)
        out_dim = "nVertLevels"
        coords = {d: zgrid.coords[d] for d in zgrid.dims[:-1] if d in zgrid.coords}
        return xr.DataArray(centers, dims=(*zgrid.dims[:-1], out_dim), coords=coords)
    z = np.asarray(zgrid)
    return 0.5 * (z[..., :-1] + z[..., 1:])


def heights_above_ground(zgrid: xr.DataArray | np.ndarray):
    """Layer-center heights relative to local terrain (AGL).

    ``zgrid[..., 0]`` is the surface interface (terrain height), so AGL center
    heights are the MSL centers minus that surface value. Over ocean (terrain 0)
    AGL equals MSL.
    """
    centers = layer_center_heights(zgrid)
    if isinstance(zgrid, xr.DataArray):
        surface = zgrid.isel({zgrid.dims[-1]: 0}).data
        return centers - surface[..., np.newaxis] if surface.ndim else centers - surface
    z = np.asarray(zgrid)
    return centers - z[..., 0:1]


def nearest_level_index(heights_agl, target_m: float) -> int:
    """Index of the layer center closest to *target_m* metres above ground.

    *heights_agl* is a 1-D profile of center AGL heights for a single cell.
    """
    h = np.asarray(heights_agl)
    if h.ndim != 1:
        raise ValueError("expected a 1-D height profile for a single cell")
    return int(np.argmin(np.abs(h - target_m)))
